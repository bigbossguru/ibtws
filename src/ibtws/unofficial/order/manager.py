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
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from .factory import bracket_to_orders, request_to_order
from .models import (
    BracketRequest,
    Cancelled,
    Filled,
    OrderEvent,
    OrderRequest,
    OrderSide,
    OrderState,
    PositionChanged,
    Rejected,
    RequestSubmitted,
    StatusChanged,
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
    "PendingCancel": OrderState.PENDING_SUBMIT,
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
    ) -> None:
        self._client = client
        self._store = store
        self._allow_live = allow_live
        self._monitor = OrderMonitor()

        self._tracked: dict[str, TrackedOrder] = {}
        self._uuid_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._positions: dict[tuple, PositionChanged] = {}

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._min_interval = 1.0 / pace_per_sec if pace_per_sec > 0 else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()

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
        primary = accounts[0] if accounts else ""
        if not self._allow_live and not is_paper_account(primary):
            raise RuntimeError(
                f"Refusing to attach to live account {primary!r}. "
                f"Pass allow_live=True to override (you'd better mean it)."
            )

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

        self._started = True
        logger.info(f"OrderManager: started (account={primary}, tracked={len(self._tracked)})")
        return report

    async def stop(self) -> None:
        if not self._started:
            return
        self._client.ib.orderStatusEvent -= self._on_order_status
        self._client.ib.execDetailsEvent -= self._on_exec_details
        self._client.ib.positionEvent -= self._on_position
        self._started = False
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
        """Submit a bracket (entry + TP + SL) as one atomic group.

        Returns three :class:`TrackedOrder`s with a shared ``bracket_group``.
        """
        self._require_started()
        validate_request(request)
        group = make_order_ref()
        parent_uuid = f"{group}_parent"
        tp_uuid = f"{group}_tp"
        sl_uuid = f"{group}_sl"

        orders = bracket_to_orders(request, parent_uuid, tp_uuid, sl_uuid)

        # Persist the group submission first.
        await self._store.append(_build_submitted_event(parent_uuid, request, bracket_group=group))

        tracked_list: list[TrackedOrder] = []
        async with self._slot():
            for o, uuid in zip(orders, (parent_uuid, tp_uuid, sl_uuid)):
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
        event = Filled(
            uuid=uuid,
            perm_id=int(trade.order.permId or 0),
            exec_id=str(fill.execution.execId),
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
        self._dispatch(event)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(self, event: OrderEvent) -> None:
        """Persist + publish. Persist failures are logged, not raised."""
        asyncio.ensure_future(self._persist_safely(event))
        self._monitor.publish(event)

    async def _persist_safely(self, event: OrderEvent) -> None:
        try:
            await self._store.append(event)
        except Exception:
            logger.exception(f"OrderManager: failed to persist {type(event).__name__}")

    def _uuid_lock(self, uuid: str) -> asyncio.Lock:
        return self._uuid_locks[uuid]

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("OrderManager.start() must be called first.")

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        """Acquire concurrency + pacing slot — same pattern as OptionChainFetcher."""
        async with self._semaphore:
            await self._await_next_slot()
            yield

    async def _await_next_slot(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._min_interval


# ---------------------------------------------------------------------------
# Event builders (shared between place() and place_bracket())
# ---------------------------------------------------------------------------


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
