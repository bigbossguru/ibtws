# Non-official package. Not affiliated with ib_async upstream.

"""OrderManager — the orchestrator class.

Responsibilities:

* Place / cancel / bracket orders via ``IBKRClient``.
* Bind to IB global events (``orderStatusEvent``, ``execDetailsEvent``,
  ``positionEvent``) and funnel them into typed :class:`OrderEvent`s.
* Persist every state transition to the :class:`OrderStore`.
* Publish events to the :class:`OrderMonitor` for downstream consumers.
* Reconcile against IB on ``start()`` (IB is the source of truth).
* Enforce the paper-vs-live safety interlock.

The manager is the only stateful class in the package; everything else is a
pure dataclass, pure function, or thin I/O wrapper.
"""

from __future__ import annotations

import asyncio
import logging

from ib_async import Contract
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional, Sequence

from ibtws.unofficial._pacing import ThrottledExecutor

from .factory import (
    bracket_to_orders,
    build_bracket,
    build_limit,
    build_market,
    build_stop,
    request_to_order,
)
from .models import (
    BracketRequest,
    Cancelled,
    Filled,
    OrderEvent,
    OrderRequest,
    OrderSide,
    OrderState,
    PositionChanged,
    PositionPnL,
    Rejected,
    RequestSubmitted,
    StatusChanged,
    TimeInForce,
    TrackedOrder,
    serialise_contract,
)
from .monitor import OrderMonitor
from .reconciler import ReconciliationReport, reconcile
from .store import OrderStore
from .utils import is_paper_account, make_order_ref, validate_request

logger = logging.getLogger(__name__)


_IB_STATUS_TO_STATE = {
    "PendingSubmit": OrderState.PENDING_SUBMIT,
    # IB has acknowledged the cancel request but the order is still live
    # until a terminal "Cancelled" / "ApiCancelled" / "Filled" arrives.
    # Map to SUBMITTED so cancel_all and downstream consumers treat it as
    # an open order (and don't, e.g., re-fire cancelOrder on it).
    "PendingCancel": OrderState.SUBMITTED,
    "PreSubmitted": OrderState.SUBMITTED,
    "Submitted": OrderState.SUBMITTED,
    "Filled": OrderState.FILLED,
    "ApiCancelled": OrderState.CANCELLED,
    "Cancelled": OrderState.CANCELLED,
    "Inactive": OrderState.INACTIVE,
}


class OrderManager:
    """Place orders, track their lifecycle, persist state, expose events."""

    def __init__(
        self,
        client: Any,
        store: OrderStore,
        *,
        allow_live: bool = False,
        max_concurrency: int = 10,
        pace_per_sec: float = 10.0,
        executor: Optional[ThrottledExecutor] = None,
    ) -> None:
        self._client = client
        self._store = store
        self._allow_live = allow_live
        self._monitor = OrderMonitor()

        # Populated by start() once IB has reported managedAccounts. Used to
        # police per-request account routing so a paper-primary session
        # cannot accidentally route an individual order to a live sub-account.
        self._managed_accounts: tuple[str, ...] = ()

        self._tracked: dict[str, TrackedOrder] = {}
        self._uuid_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._positions: dict[tuple, PositionChanged] = {}
        # Live ib_async.Contract refs keyed by conId, captured from position
        # events. Used by close_position to avoid re-qualifying from a flat
        # dict (which can fail for synthetic contracts like ContFuture).
        self._position_contracts: dict[int, Any] = {}

        # IB can replay execDetails on reconnect; dedup so downstream consumers
        # don't see the same fill twice (would re-fire exit logic).
        self._seen_exec_ids: set[str] = set()

        self._executor = executor or ThrottledExecutor(max_concurrency=max_concurrency, pace_per_sec=pace_per_sec)

        self._persist_queue: asyncio.Queue[OrderEvent] = asyncio.Queue()
        self._persist_task: Optional[asyncio.Task] = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> ReconciliationReport:
        """Bind IB events, run reconciliation, return the divergence report.

        Raises ``RuntimeError`` if the connected account is live and
        ``allow_live`` was not explicitly set.
        """
        if self._started:
            raise RuntimeError("OrderManager already started.")

        accounts = self._client.ib.managedAccounts()
        if not accounts:
            # Without a confirmed account list we cannot enforce the
            # paper-vs-live interlock — fail closed.
            raise RuntimeError(
                "OrderManager.start(): IB has not yet reported managedAccounts. "
                "Wait for the connection to settle before starting the manager."
            )
        primary = accounts[0]
        if not self._allow_live and not is_paper_account(primary):
            raise RuntimeError(
                f"Refusing to attach to live account {primary!r}. "
                f"Pass allow_live=True to override (you'd better mean it)."
            )
        # Remember the set of accounts considered safe for this session so
        # per-request account routing can be policed in place() / place_bracket().
        self._managed_accounts = tuple(accounts)

        self._client.ib.orderStatusEvent += self._on_order_status
        self._client.ib.execDetailsEvent += self._on_exec_details
        self._client.ib.positionEvent += self._on_position

        report = await reconcile(self._client, self._store)

        # Rehydrate tracked orders for matched UUIDs so cancel/inspect works
        # post-restart without an extra explicit hydration step.
        open_trades = {t.order.orderRef: t for t in self._client.ib.openTrades() if getattr(t.order, "orderRef", None)}
        for uuid in report.matched:
            trade = open_trades.get(uuid)
            if trade is None:
                continue
            self._tracked[uuid] = TrackedOrder(
                uuid=uuid,
                request=None,  # original request not recoverable; the audit log carries it
                trade=trade,
                state=_IB_STATUS_TO_STATE.get(trade.orderStatus.status, OrderState.SUBMITTED),
                filled=float(trade.orderStatus.filled or 0.0),
                remaining=float(trade.orderStatus.remaining or 0.0),
                avg_fill_price=float(trade.orderStatus.avgFillPrice or 0.0),
                perm_id=int(trade.order.permId or 0),
            )

        # Seed position cache.
        for snap in report.positions:
            key = (snap.account, snap.contract.get("conId", 0))
            self._positions[key] = PositionChanged(
                account=snap.account,
                contract=snap.contract,
                quantity=snap.quantity,
                avg_cost=snap.avg_cost,
                timestamp=snap.timestamp,
            )

        # Cache live Contract refs for close_position. reconcile() returns
        # serialised snapshots only — we need the raw ib_async.Contract
        # objects to flatten positions without re-qualifying from a dict
        # (which fails for synthetic contracts like ContFuture).
        for p in await self._client.ib.reqPositionsAsync():
            conid = getattr(p.contract, "conId", 0)
            if conid:
                self._position_contracts[conid] = p.contract

        self._started = True
        self._persist_task = asyncio.ensure_future(self._persist_worker())
        logger.info(f"OrderManager: started (account={primary}, tracked={len(self._tracked)})")
        return report

    async def stop(self) -> None:
        if not self._started:
            return
        self._client.ib.orderStatusEvent -= self._on_order_status
        self._client.ib.execDetailsEvent -= self._on_exec_details
        self._client.ib.positionEvent -= self._on_position
        self._started = False
        # Drain pending persist writes before shutting down.
        if self._persist_task is not None:
            await self._persist_queue.join()
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
            self._persist_task = None
        logger.info("OrderManager: stopped.")

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    async def place(self, request: OrderRequest) -> TrackedOrder:
        """Submit a single-leg order. Persist the request *before* the IB call.

        Persist-first ensures that a crash between persist and submit is
        detectable by the reconciler (local_only entry that IB never saw → we
        know we never sent it).
        """
        self._require_started()
        validate_request(request)
        self._check_account_safety(getattr(request, "account", None))
        uuid = make_order_ref()
        order = request_to_order(request, uuid)

        async with self._uuid_lock(uuid):
            await self._store.append(_build_submitted_event(uuid, request))
            async with self._slot():
                trade = self._client.ib.placeOrder(request.contract, order)
            tracked = TrackedOrder(
                uuid=uuid,
                request=request,
                trade=trade,
                state=OrderState.PENDING_SUBMIT,
                remaining=request.quantity,
                perm_id=int(getattr(trade.order, "permId", 0) or 0),
            )
            self._tracked[uuid] = tracked

        self._monitor.publish(_build_submitted_event(uuid, request))
        return tracked

    async def place_bracket(self, request: BracketRequest) -> list[TrackedOrder]:
        """Submit a bracket (entry + TP, plus optional SL) as one atomic group.

        Returns the :class:`TrackedOrder`s sharing a ``bracket_group``: three
        when a stop-loss is present (parent + TP + SL), two for a TP-only
        bracket (parent + TP).
        """
        self._require_started()
        validate_request(request)
        self._check_account_safety(getattr(request, "account", None))
        group = make_order_ref()
        parent_uuid = f"{group}_parent"
        tp_uuid = f"{group}_tp"
        sl_uuid = f"{group}_sl"

        orders = bracket_to_orders(
            request,
            parent_uuid,
            tp_uuid,
            sl_uuid,
            parent_order_id=self._client.ib.client.getReqId(),
            oca_group=group,
        )

        # Persist the group submission first.
        await self._store.append(_build_submitted_event(parent_uuid, request, bracket_group=group))

        # zip truncates to len(orders): TP-only brackets yield [parent, tp] and
        # the trailing sl_uuid is simply unused.
        uuids = (parent_uuid, tp_uuid, sl_uuid)
        tracked_list: list[TrackedOrder] = []
        async with self._slot():
            for o, uuid in zip(orders, uuids):
                trade = self._client.ib.placeOrder(request.contract, o)
                tracked = TrackedOrder(
                    uuid=uuid,
                    request=request,
                    trade=trade,
                    state=OrderState.PENDING_SUBMIT,
                    remaining=request.quantity,
                    perm_id=int(getattr(trade.order, "permId", 0) or 0),
                    bracket_group=group,
                )
                self._tracked[uuid] = tracked
                tracked_list.append(tracked)

        self._monitor.publish(_build_submitted_event(parent_uuid, request, bracket_group=group))
        return tracked_list

    # ------------------------------------------------------------------
    # Convenience builders: one-call build + place
    # ------------------------------------------------------------------

    async def market(
        self,
        contract: Any,
        side: OrderSide,
        quantity: float,
        *,
        tif: TimeInForce = TimeInForce.DAY,
        account: Optional[str] = None,
        outside_rth: bool = False,
    ) -> TrackedOrder:
        """Build a market request and submit it in one call."""
        return await self.place(
            build_market(contract, side, quantity, tif=tif, account=account, outside_rth=outside_rth)
        )

    async def limit(
        self,
        contract: Any,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        *,
        tif: TimeInForce = TimeInForce.DAY,
        account: Optional[str] = None,
        outside_rth: bool = False,
    ) -> TrackedOrder:
        """Build a limit request and submit it in one call."""
        return await self.place(
            build_limit(contract, side, quantity, limit_price, tif=tif, account=account, outside_rth=outside_rth)
        )

    async def stop_order(
        self,
        contract: Any,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        *,
        tif: TimeInForce = TimeInForce.DAY,
        account: Optional[str] = None,
        outside_rth: bool = False,
    ) -> TrackedOrder:
        """Build a stop request and submit it in one call.

        Named ``stop_order`` rather than ``stop`` to avoid shadowing the
        :meth:`stop` lifecycle method that tears the manager down.
        """
        return await self.place(
            build_stop(contract, side, quantity, stop_price, tif=tif, account=account, outside_rth=outside_rth)
        )

    async def bracket(
        self,
        contract: Any,
        side: OrderSide,
        quantity: float,
        *,
        take_profit_price: float,
        stop_loss_price: Optional[float] = None,
        entry_limit_price: Optional[float] = None,
        tif: TimeInForce = TimeInForce.DAY,
        account: Optional[str] = None,
        outside_rth: bool = False,
    ) -> list[TrackedOrder]:
        """Build a bracket request and submit it in one call.

        ``stop_loss_price=None`` submits a TP-only bracket (entry + take-profit).
        """
        return await self.place_bracket(
            build_bracket(
                contract,
                side,
                quantity,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                entry_limit_price=entry_limit_price,
                tif=tif,
                account=account,
                outside_rth=outside_rth,
            )
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def close_position(
        self,
        con_id: int,
        *,
        kind: str = "market",
        limit_price: Optional[float] = None,
        cancel_working: bool = True,
    ) -> Optional[TrackedOrder]:
        """Flatten a single position by ``conId``.

        Cancels any working orders on the same contract first (unless
        ``cancel_working=False``), re-qualifies the contract via IB by
        ``conId``, then submits an opposite-side market or limit order for
        ``abs(quantity)``.

        Returns the new :class:`TrackedOrder`, or ``None`` if no non-zero
        position exists for that ``conId``.
        """
        self._require_started()
        pos = next(
            (p for p in self._positions.values() if p.contract.get("conId") == con_id),
            None,
        )
        if pos is None:
            known = [(p.contract.get("symbol"), p.contract.get("conId"), p.quantity) for p in self._positions.values()]
            logger.warning(
                f"close_position: no position for conId={con_id}. Known positions (symbol, conId, qty): {known}"
            )
            return None
        if pos.quantity == 0:
            logger.warning(f"close_position: conId={con_id} has zero quantity, nothing to close.")
            return None

        if cancel_working:
            for t in list(self._tracked.values()):
                if t.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                    continue
                req_contract = getattr(t.request, "contract", None) if t.request else None
                if req_contract is not None and getattr(req_contract, "conId", 0) == con_id:
                    await self.cancel(t.uuid)

        skeleton = self._position_contracts.get(con_id)
        if skeleton is None:
            skeleton = Contract(
                conId=con_id,
                symbol=pos.contract.get("symbol") or "",
                secType=pos.contract.get("secType") or "",
                exchange=pos.contract.get("exchange") or "",
                currency=pos.contract.get("currency") or "",
                lastTradeDateOrContractMonth=pos.contract.get("lastTradeDateOrContractMonth") or "",
                strike=pos.contract.get("strike") or 0.0,
                right=pos.contract.get("right") or "",
                multiplier=pos.contract.get("multiplier") or "",
                tradingClass=pos.contract.get("tradingClass") or "",
            )
        # Positions from reqPositionsAsync come back with exchange="" — IB
        # refuses orders without a routing destination. qualifyContractsAsync
        # backfills exchange/primaryExchange from conId.
        contract = skeleton
        if not getattr(contract, "exchange", ""):
            qualified = await self._client.ib.qualifyContractsAsync(contract)
            if not qualified:
                raise RuntimeError(f"Failed to qualify contract conId={con_id}")
            contract = qualified[0]
        self._position_contracts[con_id] = contract

        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        qty = abs(pos.quantity)
        account = pos.account or None
        if kind == "market":
            return await self.market(contract, side, qty, account=account)
        if kind == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required when kind='limit'")
            return await self.limit(contract, side, qty, limit_price, account=account)
        raise ValueError(f"Unsupported close kind: {kind!r} (use 'market' or 'limit')")

    async def refresh_positions(self) -> list[PositionChanged]:
        """Pull fresh positions from IB, update the local cache, return them.

        Useful right before flattening — ``positionEvent`` from IB can lag
        several seconds after a fill, so the cached ``self.positions`` may be
        stale or empty.
        """
        self._require_started()
        fresh = await self._client.ib.reqPositionsAsync()
        snapshots: list[PositionChanged] = []
        for p in fresh:
            conid = getattr(p.contract, "conId", 0)
            if not conid:
                continue
            snap = PositionChanged(
                account=p.account,
                contract=serialise_contract(p.contract),
                quantity=float(p.position),
                avg_cost=float(p.avgCost),
            )
            self._positions[(p.account, conid)] = snap
            self._position_contracts[conid] = p.contract
            snapshots.append(snap)
        return snapshots

    async def close_all_positions(self, *, kind: str = "market", cancel_working: bool = True) -> list[TrackedOrder]:
        """Flatten every non-zero position. Returns the list of closing orders.

        Always pulls fresh positions from IB before acting — don't rely on
        ``positionEvent`` having reached us yet.
        """
        await self.refresh_positions()
        closed: list[TrackedOrder] = []
        non_zero = [p for p in self._positions.values() if p.quantity != 0]
        if not non_zero:
            logger.info("close_all_positions: no non-zero positions found.")
            return closed
        for pos in non_zero:
            con_id = pos.contract.get("conId")
            if not con_id:
                continue
            tracked = await self.close_position(con_id, kind=kind, cancel_working=cancel_working)
            if tracked is not None:
                closed.append(tracked)
        return closed

    async def cancel_all(self) -> list[str]:
        """Cancel every non-terminal tracked order. Returns the cancelled uuids."""
        self._require_started()
        cancelled: list[str] = []
        for t in list(self._tracked.values()):
            if t.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                continue
            await self.cancel(t.uuid)
            cancelled.append(t.uuid)
        return cancelled

    async def cancel(self, uuid: str) -> None:
        """Request cancellation. Idempotent — already-terminal orders are a no-op."""
        self._require_started()
        tracked = self._tracked.get(uuid)
        if tracked is None:
            raise KeyError(f"Unknown order uuid: {uuid}")
        if tracked.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
            logger.info(f"OrderManager: cancel ignored, {uuid} already {tracked.state.value}.")
            return
        async with self._uuid_lock(uuid):
            async with self._slot():
                self._client.ib.cancelOrder(tracked.trade.order)
        logger.info(f"OrderManager: cancel requested for {uuid}.")

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def open_orders(self) -> list[TrackedOrder]:
        return [
            t
            for t in self._tracked.values()
            if t.state not in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED)
        ]

    @property
    def positions(self) -> list[PositionChanged]:
        return list(self._positions.values())

    async def current_pnl(
        self,
        con_ids: Optional[Sequence[int]] = None,
        *,
        snapshot_timeout: float = 5.0,
    ) -> list[PositionPnL]:
        """Quote each open position once and return its unrealized PnL.

        On-demand only — no streaming subscription, no persistence. Skips
        zero-quantity positions. ``con_ids`` filters to a subset; ``None``
        (default) marks every open position.

        Each :class:`PositionPnL` carries ``market_price`` / ``market_value``
        / ``unrealized_pnl`` as ``Optional[float]`` — ``None`` means the quote
        was unavailable (after-hours, missing market-data subscription,
        snapshot timeout). Treat ``None`` as "unknown", not as ``0.0``.

        Pricing rule per ticker (first non-stale value wins):
            1. mid of bid/ask if both > 0
            2. last
            3. close
        Otherwise ``market_price = None``.
        """
        self._require_started()

        wanted = set(con_ids) if con_ids is not None else None
        targets: list[tuple[PositionChanged, Any]] = []
        for pos in self._positions.values():
            if not pos.quantity:
                continue
            conid = pos.contract.get("conId")
            if not conid:
                continue
            if wanted is not None and conid not in wanted:
                continue
            contract = self._position_contracts.get(conid)
            if contract is None:
                # Build a skeleton from the serialised position dict so we can
                # qualify it below. reqPositions returns contracts with empty
                # exchange — reqTickers / placeOrder both reject those.
                contract = Contract(
                    conId=int(conid),
                    symbol=pos.contract.get("symbol") or "",
                    secType=pos.contract.get("secType") or "",
                    exchange=pos.contract.get("exchange") or "",
                    currency=pos.contract.get("currency") or "",
                    lastTradeDateOrContractMonth=pos.contract.get("lastTradeDateOrContractMonth") or "",
                    strike=pos.contract.get("strike") or 0.0,
                    right=pos.contract.get("right") or "",
                    multiplier=pos.contract.get("multiplier") or "",
                    tradingClass=pos.contract.get("tradingClass") or "",
                )
            # Backfill the routing exchange via qualifyContractsAsync when it's
            # missing (positionEvent contracts often arrive without one).
            if not getattr(contract, "exchange", ""):
                try:
                    qualified = await self._client.ib.qualifyContractsAsync(contract)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"OrderManager.current_pnl: qualify failed for conId={conid}: {exc}")
                    qualified = []
                if qualified:
                    contract = qualified[0]
                else:
                    targets.append((pos, None))
                    continue
            self._position_contracts[int(conid)] = contract
            targets.append((pos, contract))

        # Snapshot all qualified contracts in one IB call.
        quotable = [c for _, c in targets if c is not None]
        ticker_by_conid: dict[int, Any] = {}
        if quotable:
            try:
                async with self._slot():
                    tickers = await asyncio.wait_for(
                        self._client.ib.reqTickersAsync(*quotable, regulatorySnapshot=False),
                        timeout=snapshot_timeout,
                    )
                for t in tickers or []:
                    cid = int(getattr(t.contract, "conId", 0) or 0)
                    if cid:
                        ticker_by_conid[cid] = t
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"OrderManager.current_pnl: snapshot failed: {exc}")

        results: list[PositionPnL] = []
        for pos, contract in targets:
            conid = int(pos.contract.get("conId") or 0)
            multiplier = _contract_multiplier(pos.contract, contract)
            cost_basis = pos.quantity * pos.avg_cost  # IB's avgCost already includes multiplier
            price = _price_from_ticker(ticker_by_conid.get(conid))
            if price is None:
                results.append(
                    PositionPnL(
                        account=pos.account,
                        contract=pos.contract,
                        quantity=pos.quantity,
                        avg_cost=pos.avg_cost,
                        multiplier=multiplier,
                        cost_basis=cost_basis,
                        market_price=None,
                        market_value=None,
                        unrealized_pnl=None,
                    )
                )
                continue
            market_value = pos.quantity * price * multiplier
            results.append(
                PositionPnL(
                    account=pos.account,
                    contract=pos.contract,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    multiplier=multiplier,
                    cost_basis=cost_basis,
                    market_price=price,
                    market_value=market_value,
                    unrealized_pnl=market_value - cost_basis,
                )
            )
        return results

    def events(self) -> AsyncIterator[OrderEvent]:
        return self._monitor.stream()

    def on_event(self, fn: Callable[[OrderEvent], None]) -> None:
        self._monitor.register(fn)

    # ------------------------------------------------------------------
    # IB event handlers (run on the event-loop thread)
    # ------------------------------------------------------------------

    def _on_order_status(self, trade: Any) -> None:
        uuid = getattr(trade.order, "orderRef", None)
        if not uuid:
            return  # not ours
        tracked = self._tracked.get(uuid)
        status = trade.orderStatus
        new_state = _IB_STATUS_TO_STATE.get(status.status, OrderState.SUBMITTED)
        event = StatusChanged(
            uuid=uuid,
            perm_id=int(trade.order.permId or 0),
            state=new_state.value,
            filled=float(status.filled or 0.0),
            remaining=float(status.remaining or 0.0),
            avg_fill_price=float(status.avgFillPrice or 0.0),
        )
        if tracked is not None:
            tracked.state = new_state
            tracked.filled = event.filled
            tracked.remaining = event.remaining
            tracked.avg_fill_price = event.avg_fill_price
            tracked.perm_id = event.perm_id
            tracked.last_update = event.timestamp
        self._dispatch(event)

        if new_state == OrderState.CANCELLED:
            self._dispatch(Cancelled(uuid=uuid, perm_id=event.perm_id))
        elif new_state == OrderState.INACTIVE:
            # IB uses Inactive for rejected orders too; the actual reason rides
            # the error channel.
            self._dispatch(Rejected(uuid=uuid, perm_id=event.perm_id, reason=status.status))
            if tracked is not None:
                tracked.state = OrderState.REJECTED

    def _on_exec_details(self, trade: Any, fill: Any) -> None:
        uuid = getattr(trade.order, "orderRef", None)
        if not uuid:
            return
        exec_id = str(fill.execution.execId)
        if exec_id in self._seen_exec_ids:
            # IB replays execDetails on reconnect — drop the duplicate so
            # downstream consumers (TP/SL triggers, fill counters) don't double-count.
            logger.debug(f"OrderManager: duplicate exec {exec_id} for {uuid} ignored")
            return
        self._seen_exec_ids.add(exec_id)
        event = Filled(
            uuid=uuid,
            perm_id=int(trade.order.permId or 0),
            exec_id=exec_id,
            price=float(fill.execution.price),
            quantity=float(fill.execution.shares),
        )
        self._dispatch(event)

    def _on_position(self, position: Any) -> None:
        contract = serialise_contract(position.contract)
        key = (position.account, contract["conId"])
        event = PositionChanged(
            account=position.account,
            contract=contract,
            quantity=float(position.position),
            avg_cost=float(position.avgCost),
        )
        self._positions[key] = event
        if contract["conId"]:
            self._position_contracts[contract["conId"]] = position.contract
        self._dispatch(event)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(self, event: OrderEvent) -> None:
        """Enqueue for persistence + publish immediately to subscribers."""
        self._persist_queue.put_nowait(event)
        self._monitor.publish(event)

    async def _persist_worker(self) -> None:
        """Background task that drains the persist queue sequentially."""
        while True:
            event = await self._persist_queue.get()
            try:
                await self._store.append(event)
            except Exception:
                logger.exception(f"OrderManager: failed to persist {type(event).__name__}")
            finally:
                self._persist_queue.task_done()

    def _uuid_lock(self, uuid: str) -> asyncio.Lock:
        return self._uuid_locks[uuid]

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("OrderManager.start() must be called first.")

    def _check_account_safety(self, account: Optional[str]) -> None:
        """Police per-request account routing against the paper interlock.

        Even on a paper-primary session, an explicit ``request.account`` can
        target any account the session is permitted to trade (e.g. FA setups
        with mixed paper/live sub-accounts). Refuse to route to a non-paper
        account unless ``allow_live=True`` was set at construction time.
        """
        if not account:
            return
        if account not in self._managed_accounts:
            raise RuntimeError(
                f"Refusing to place order on account {account!r}: not in the "
                f"session's managedAccounts {self._managed_accounts!r}."
            )
        if not self._allow_live and not is_paper_account(account):
            raise RuntimeError(
                f"Refusing to route order to live account {account!r}. "
                f"Pass allow_live=True to OrderManager to override."
            )

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        """Acquire concurrency + pacing slot via the shared executor."""
        async with self._executor.slot():
            yield

    # Backwards-compat shims for callers / tests reaching into the old API.
    @property
    def _min_interval(self) -> float:
        return self._executor.min_interval

    async def _await_next_slot(self) -> None:
        await self._executor._await_next_slot()


# ---------------------------------------------------------------------------
# Event builders (shared between place() and place_bracket())
# ---------------------------------------------------------------------------


def _contract_multiplier(serialised: dict, contract: Any) -> float:
    """Best-effort numeric multiplier for a position.

    IB serialises ``multiplier`` as a string ("100" for US options, "50" for
    ES, "" for stocks). Default to 1.0 when missing or non-numeric — wrong
    for option / future positions, but visibly so (PnL will be off by 100x),
    and we'd rather not silently fabricate a value.
    """
    raw = serialised.get("multiplier") if serialised else None
    if not raw and contract is not None:
        raw = getattr(contract, "multiplier", None)
    try:
        return float(raw) if raw else 1.0
    except (TypeError, ValueError):
        return 1.0


def _price_from_ticker(ticker: Any) -> Optional[float]:
    """Pick a usable mark price from an ib_async Ticker, or ``None``.

    Order of preference:
      1. mid of bid/ask when both are positive (tightest live mark)
      2. last trade
      3. previous close (stale but acceptable for cold sessions)
    NaN / 0 / negative values are treated as missing.
    """
    if ticker is None:
        return None
    bid = _safe_pos(getattr(ticker, "bid", None))
    ask = _safe_pos(getattr(ticker, "ask", None))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0
    last = _safe_pos(getattr(ticker, "last", None))
    if last is not None:
        return last
    close = _safe_pos(getattr(ticker, "close", None))
    if close is not None:
        return close
    return None


def _safe_pos(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v <= 0:  # NaN check: NaN != NaN
        return None
    return v


def _build_submitted_event(
    uuid: str,
    request: OrderRequest | BracketRequest,
    *,
    bracket_group: Optional[str] = None,
) -> RequestSubmitted:
    extra: dict[str, Any] = {}
    kind: str
    if isinstance(request, BracketRequest):
        kind = "bracket"
        extra = {
            "take_profit_price": request.take_profit_price,
            "stop_loss_price": request.stop_loss_price,
            "entry_limit_price": request.entry_limit_price,
        }
    elif hasattr(request, "limit_price"):
        kind = "limit"
        extra = {"limit_price": request.limit_price}
    elif hasattr(request, "stop_price"):
        kind = "stop"
        extra = {"stop_price": request.stop_price}
    else:
        kind = "market"
    extra["outside_rth"] = bool(getattr(request, "outside_rth", False))
    side: OrderSide = request.side
    return RequestSubmitted(
        uuid=uuid,
        request_kind=kind,
        contract=serialise_contract(request.contract),
        side=side.value,
        quantity=float(request.quantity),
        tif=request.tif.value,
        account=request.account,
        extra=extra,
        bracket_group=bracket_group,
    )
