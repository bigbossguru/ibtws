# Non-official package. Not affiliated with ib_async upstream.

"""Vertical credit-spread strategy (bull-put / bear-call) for IBKR.

Design goals
------------
* **Comprehensive** – tunables for short-leg delta, wing width, DTE window,
  credit floor, risk-reward floor, slippage, expiry filtering, TP/SL, account
  override, exchange and trading-class.
* **Robust**        – every selection step is fault-tolerant: missing greeks,
  empty chain slices, untradeable strikes and stale quotes are reported as
  ``CreditSpreadError`` instead of silent failures.
* **Resilient**     – placement uses an atomic two-leg ``BAG`` combo order so
  IBKR fills both legs at the target net credit or neither, eliminating the
  classic "naked-short after a single-leg fill" risk. Every exit is confirmed:
  an unfilled closing order is cancelled, re-priced and re-sent, and stop-loss
  exits escalate to a marketable price instead of leaving the position open.

Architecture
------------
``CreditSpreadStrategy`` orchestrates four stages:

1. **Discover** – :class:`OptionChainFetcher` resolves the chain universe and
   pulls greek-bearing snapshots for the relevant rights / expirations.
2. **Select**   – pure functions in this module pick the short leg by delta
   proximity and the long leg by wing-width offset, then sanity-check the
   spread economics against the user's risk knobs.
3. **Place**    – a ``Bag`` combo contract is sent through
   :class:`OrderManager` as a single net-credit ``LimitRequest``. The manager
   persists the request, publishes lifecycle events, and reconciles against
   IB on restart — the strategy inherits all of that for free.
4. **Manage**   – :meth:`monitor_and_exit` waits for the entry (cancelling an
   unfilled remainder on timeout), streams both legs' quotes and closes the
   filled quantity at the configured take-profit / stop-loss debit through
   :meth:`close_and_confirm`.

The order layer's :func:`validate_request` is BAG-aware: a combo contract is
accepted as long as every ``comboLeg`` carries a non-zero ``conId``. Strategy
code only ever talks to the manager — never ``ib.placeOrder`` — so persistence,
reconciliation, the paper-account interlock and the event stream all apply
uniformly to combo and single-leg orders.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable, Optional

from ib_async import Bag, ComboLeg

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher, OptionQuote
from ibtws.unofficial.helpers import safe_pick_value, snapshot_each
from ibtws.unofficial.order.manager import OrderManager
from ibtws.unofficial.order.models import OrderSide, OrderState, TimeInForce, TrackedOrder

from .models import CreditSpreadParams, CreditSpreadPlan, ExitResult, SpreadLeg, SpreadType
from .utils import (
    CreditSpreadError,
    _quote_mid,
    _round_to_tick,
    select_expiry,
    select_long_leg,
    select_short_leg,
)

logger = logging.getLogger(__name__)

_TERMINAL = frozenset({OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED})
_LIVE_DATA = 1  # IB market data type for real-time quotes


class CreditSpreadStrategy:
    """Build, place and manage two-leg vertical credit spreads.

    Each :meth:`build_plan` / :meth:`place` invocation can use different
    parameters. The class keeps the :class:`OptionChainFetcher` it was given
    (or instantiated) so consecutive plans on the same underlying re-use the
    cached chain definition, and it owns the streaming quote subscriptions
    used while monitoring.

    Exit tuning
    -----------
    exit_fill_timeout:
        Seconds a closing order may work before it is cancelled and re-priced.
    exit_max_attempts:
        Closing attempts per exit. For a stop-loss the last attempt is priced
        at the spread width — the most a vertical can be worth — so it is
        marketable. Take-profit exits never escalate; what stays unfilled is
        left to the monitor loop.
    cancel_timeout:
        Seconds to wait for IB to confirm a cancel before giving up.
    quote_max_age:
        Streaming quotes older than this many seconds count as unavailable.

    Example
    -------
    >>> async with IBKRClient(cfg) as client:        # doctest: +SKIP
    ...     store = JsonStore("orders.jsonl")
    ...     om = OrderManager(client, store)
    ...     await om.start()
    ...     strat = CreditSpreadStrategy(client, order_manager=om)
    ...     params = CreditSpreadParams(
    ...         underlying=Stock("AAPL", "SMART", "USD"),
    ...         spread_type=SpreadType.BULL_PUT,
    ...     )
    ...     plan = await strat.build_plan(params)
    ...     tracked = await strat.place(plan)
    ...     await strat.monitor_and_exit(plan, tracked)
    """

    def __init__(
        self,
        client: IBKRClient,
        order_manager: OrderManager,
        *,
        fetcher: Optional[OptionChainFetcher] = None,
        tick_size: float = 0.05,
        exit_fill_timeout: float = 15.0,
        exit_max_attempts: int = 4,
        cancel_timeout: float = 10.0,
        quote_max_age: float = 60.0,
    ) -> None:
        if order_manager is None:
            raise ValueError("order_manager is required — combo placement goes through OrderManager.")
        self._client = client
        self._om = order_manager
        self._fetcher = fetcher or OptionChainFetcher(client)
        self._tick = tick_size
        self._exit_fill_timeout = exit_fill_timeout
        self._exit_max_attempts = max(1, exit_max_attempts)
        self._cancel_timeout = cancel_timeout
        self._quote_max_age = quote_max_age
        # conId -> [ticker, refcount, contract] for legs being monitored.
        self._streams: dict[int, list[Any]] = {}
        add_listener = getattr(client, "add_reconnect_listener", None)
        if callable(add_listener):
            add_listener(self._resubscribe)

    # ------------------------------------------------------------------
    # Stage 1 + 2: discover + select
    # ------------------------------------------------------------------

    async def build_plan(self, params: CreditSpreadParams) -> CreditSpreadPlan:
        """Resolve a spread satisfying *params*; raise :class:`CreditSpreadError` if impossible.

        The underlying contract is qualified if needed; the chain definition
        is fetched; the expiry is chosen; the relevant side of the chain is
        snapshotted; the short and long legs are selected and economic
        constraints are checked.
        """
        underlying = params.underlying
        if not getattr(underlying, "conId", 0):
            (underlying,) = await self._client.ib.qualifyContractsAsync(underlying)

        chain = await self._fetcher.fetch_chain_definition(
            underlying, exchange=params.exchange, trading_class=params.trading_class
        )

        if params.expirations is not None:
            available_exp: Iterable[str] = [e for e in chain.expirations if e in set(params.expirations)]
        else:
            available_exp = [
                e
                for e in chain.expirations
                if (params.expiry_from is None or e >= params.expiry_from)
                and (params.expiry_to is None or e <= params.expiry_to)
            ]
        expiry = select_expiry(
            available_exp,
            target_dte=params.target_dte,
            dte_tolerance=params.dte_tolerance,
        )

        quotes = await self._fetcher.fetch_snapshot(
            underlying,
            exchange=params.exchange,
            currency=params.currency,
            trading_class=params.trading_class,
            rights=(params.spread_type.right,),
            expirations=[expiry],
            strike_window_pct=params.strike_window_pct,
        )
        if not quotes:
            raise CreditSpreadError(
                f"Chain snapshot returned no quotes for {underlying.symbol} {expiry} right={params.spread_type.right}"
            )
        if params.require_live_quotes:
            live = [q for q in quotes if q.market_data_type == _LIVE_DATA]
            if not live:
                types = sorted({q.market_data_type for q in quotes}, key=str)
                raise CreditSpreadError(
                    f"require_live_quotes=True but no live quotes for {underlying.symbol} {expiry} "
                    f"(market data types seen: {types}). Check the market-data subscription."
                )
            quotes = live

        short_quote = select_short_leg(
            quotes,
            target_short_delta=params.target_short_delta,
            max_short_delta=params.max_short_delta,
            min_open_interest=params.min_open_interest,
            min_volume=params.min_volume,
        )
        long_quote = select_long_leg(
            quotes,
            short=short_quote,
            wing_width=params.wing_width,
            spread_type=params.spread_type,
            min_open_interest=params.min_open_interest,
            min_volume=params.min_volume,
        )

        spot = next((q.underlying_price for q in quotes if q.underlying_price), None)
        plan = self._materialise_plan(params, expiry, short_quote, long_quote, chain.multiplier, spot)
        self._enforce_economics(plan)
        logger.info(f"CreditSpread: built plan — {plan.describe()}")
        return plan

    # ------------------------------------------------------------------
    # Stage 3: place
    # ------------------------------------------------------------------

    async def place(
        self,
        plan: CreditSpreadPlan,
        *,
        limit_credit: Optional[float] = None,
    ) -> TrackedOrder:
        """Submit the spread as one atomic BAG net-credit limit order.

        Routed through :class:`OrderManager`, so the submission is persisted,
        published on the event bus, and rehydrated by the reconciler after a
        restart — exactly like a single-leg order.

        ``limit_credit`` is the *positive* per-share net premium you want to
        collect (e.g. ``0.45`` to collect $0.45 per share). ``None`` derives
        it from ``plan.net_credit`` minus ``params.limit_slippage``. The
        value is rounded *down* to the tick (never ask for more credit than
        intended) and then **negated** before being sent to IB.

        IB combo convention used here
        -----------------------------
        The BAG is submitted with ``action="BUY"`` and a *signed net cost*
        as the limit price: negative = credit collected, positive = debit
        paid. So a $0.45 credit goes out as ``BUY @ -0.45``. This matches
        TWS's combo-limit display and avoids the SELL/+price sign-flip
        ambiguity that bites SMART-routed combos in some configurations.
        The leg directions live inside ``plan.bag.comboLegs`` (SELL short,
        BUY long) — the bag-level action only governs how the signed limit
        is interpreted.
        """
        if limit_credit is None:
            credit_per_share = plan.net_credit / plan.multiplier
            limit_credit = credit_per_share * (1.0 - plan.params.limit_slippage)
        limit_credit = _round_to_tick(limit_credit, self._tick, mode="down")
        if limit_credit <= 0:
            raise CreditSpreadError(f"Computed entry limit credit {limit_credit:.2f} <= 0 — refusing to submit")

        signed_limit = -limit_credit  # IB combo: BUY @ -credit = collect credit
        logger.info(
            f"CreditSpread: placing combo BUY x{plan.params.quantity} @ net {signed_limit:.2f} "
            f"(credit {limit_credit:.2f}, {plan.describe()})"
        )
        return await self._om.limit(
            plan.bag,
            OrderSide.BUY,
            plan.params.quantity,
            signed_limit,
            tif=plan.params.tif,
            account=plan.params.account,
            outside_rth=plan.params.outside_rth,
        )

    # ------------------------------------------------------------------
    # Stage 4: close
    # ------------------------------------------------------------------

    async def close(
        self,
        plan: CreditSpreadPlan,
        *,
        limit_debit: Optional[float] = None,
        tif: Optional[TimeInForce] = None,
        quantity: Optional[float] = None,
    ) -> TrackedOrder:
        """Send one closing order: sell the BAG (reversing leg actions) at a debit.

        Fire-and-forget — use :meth:`close_and_confirm` when the position must
        actually end up closed. ``quantity`` defaults to ``params.quantity``;
        pass the filled entry size when the entry was only partially filled.

        IB combo convention: SELL the same BAG reverses the leg directions
        (BUY back the short, SELL the long). The limit price on a SELL order
        is the minimum net credit you'll accept — negative means you're
        willing to pay a debit. So closing at $0.20 debit → SELL @ -0.20.
        The debit is rounded *up* to the tick and capped at the spread width.
        """
        if limit_debit is None:
            limit_debit = plan.take_profit_debit
        if limit_debit is None:
            mid_debit = await self.current_mid_debit(plan)
            if mid_debit is None:
                raise CreditSpreadError("No mid-debit available and no limit_debit provided")
            limit_debit = mid_debit * (1.0 + plan.params.limit_slippage)

        limit_debit = min(_round_to_tick(limit_debit, self._tick, mode="up"), plan.width)
        if limit_debit <= 0:
            raise CreditSpreadError(f"Close limit debit {limit_debit:.2f} <= 0")

        qty = quantity if quantity is not None else plan.params.quantity
        # SELL BAG @ negative = pay debit to close (IB reverses leg actions).
        signed_limit = -limit_debit
        logger.info(
            f"CreditSpread: closing combo SELL x{qty} @ net {signed_limit:.2f} "
            f"(debit {limit_debit:.2f}, {plan.describe()})"
        )
        return await self._om.limit(
            plan.bag,
            OrderSide.SELL,
            qty,
            signed_limit,
            tif=tif or plan.params.tif,
            account=plan.params.account,
            outside_rth=plan.params.outside_rth,
        )

    async def close_and_confirm(
        self,
        plan: CreditSpreadPlan,
        quantity: float,
        *,
        urgent: bool,
        mid_debit: Optional[float] = None,
    ) -> ExitResult:
        """Close ``quantity`` spreads and wait until IB confirms the fills.

        Each attempt sends a closing limit, waits ``exit_fill_timeout`` for it
        to fill, then cancels the remainder (waiting for the cancel to be
        confirmed, so the next attempt can never over-close) and re-prices
        from a fresh mid with more slippage. ``urgent=True`` (stop-loss,
        timeout, lost quotes) makes the final attempt marketable by pricing
        it at the spread width. Returns how much was actually closed.
        """
        remaining = float(quantity)
        last: Optional[TrackedOrder] = None
        mid = mid_debit
        attempts = self._exit_max_attempts
        for attempt in range(1, attempts + 1):
            if mid is None:
                mid = await self.current_mid_debit(plan)
            debit = self._exit_debit(plan, mid, attempt=attempt, attempts=attempts, urgent=urgent)
            order = await self.close(plan, limit_debit=debit, quantity=remaining)
            last = order

            await self._om.wait_for(lambda: order.state in _TERMINAL, timeout=self._exit_fill_timeout)
            if order.state not in _TERMINAL:
                await self._om.cancel(order.uuid)
                await self._om.wait_for(lambda: order.state in _TERMINAL, timeout=self._cancel_timeout)

            filled = (order.filled or remaining) if order.state == OrderState.FILLED else order.filled
            remaining -= min(filled, remaining)
            if remaining <= 1e-9:
                break
            if order.state not in _TERMINAL:
                logger.error(
                    f"CreditSpread: cancel of closing order {order.uuid} not confirmed — "
                    f"stopping the exit to avoid over-closing ({remaining:g} still open)"
                )
                break
            logger.warning(
                f"CreditSpread: exit attempt {attempt}/{attempts} left {remaining:g} open "
                f"(order {order.uuid} {order.state.value} @ debit {debit:.2f})"
            )
            mid = None  # re-quote before the next attempt

        result = ExitResult(order=last, requested_quantity=float(quantity), closed_quantity=quantity - remaining)
        if not result.complete:
            log = logger.error if urgent else logger.warning
            log(f"CreditSpread: exit incomplete — {result.remaining_quantity:g} of {quantity:g} still open")
        return result

    def _exit_debit(
        self,
        plan: CreditSpreadPlan,
        mid: Optional[float],
        *,
        attempt: int,
        attempts: int,
        urgent: bool,
    ) -> float:
        if urgent and attempt == attempts:
            return plan.width  # the most a vertical can be worth: marketable
        if mid is not None:
            base = mid
        elif urgent:
            base = plan.stop_loss_debit if plan.stop_loss_debit is not None else plan.width
        elif plan.take_profit_debit is not None:
            base = plan.take_profit_debit
        else:
            base = plan.net_credit / plan.multiplier
        debit = base * (1.0 + plan.params.limit_slippage * attempt)
        return min(max(debit, self._tick), plan.width)

    # ------------------------------------------------------------------
    # Stage 4b: monitor + exit
    # ------------------------------------------------------------------

    async def await_entry(
        self,
        entry: TrackedOrder,
        *,
        max_wait: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> float:
        """Wait for the entry order and return the quantity actually filled.

        If the order is not done by ``max_wait`` seconds — or IB parks it as
        Inactive — the unfilled remainder is cancelled and the cancel is
        waited for, so the entry can no longer fill behind the caller's back.
        A partial fill is returned as-is: the caller must manage that size.
        ``quantity`` is the fallback size when IB reports Filled without a
        filled count.
        """
        await self._om.wait_for(
            lambda: entry.state in _TERMINAL or entry.state == OrderState.INACTIVE,
            timeout=max_wait,
        )
        if entry.state == OrderState.FILLED:
            return float(entry.filled or quantity or 0.0)
        if entry.state not in _TERMINAL:
            reason = "is Inactive" if entry.state == OrderState.INACTIVE else f"not done after {max_wait}s"
            logger.warning(f"CreditSpread: entry {entry.uuid} {reason} — cancelling the unfilled remainder")
            await self._om.cancel(entry.uuid)
            confirmed = await self._om.wait_for(lambda: entry.state in _TERMINAL, timeout=self._cancel_timeout)
            if not confirmed:
                logger.error(f"CreditSpread: cancel of entry {entry.uuid} not confirmed; it may still fill — check TWS")
            if entry.state == OrderState.FILLED:
                return float(entry.filled or quantity or 0.0)
        filled = float(entry.filled or 0.0)
        if filled > 0:
            logger.warning(f"CreditSpread: entry {entry.uuid} ended {entry.state.value} with {filled:g} filled")
        else:
            logger.warning(f"CreditSpread: entry {entry.uuid} ended in {entry.state.value} without a fill")
        return filled

    async def monitor_and_exit(
        self,
        plan: CreditSpreadPlan,
        entry: TrackedOrder,
        *,
        poll_interval: float = 2.0,
        max_wait: Optional[float] = None,
        close_on_timeout: bool = True,
        max_quote_failures: Optional[int] = 30,
    ) -> Optional[TrackedOrder]:
        """Manage the filled entry until it is closed.

        1. :meth:`await_entry` — cancels any unfilled remainder at the
           deadline; a partial fill is managed at its filled size.
        2. Streams both legs and checks the mid debit every ``poll_interval``
           seconds. Take-profit closes via :meth:`close_and_confirm` without
           escalation (an unfilled part keeps being monitored); stop-loss
           closes urgently, escalating to a marketable price.
        3. ``max_wait`` is one deadline for the whole call. When it passes
           with the position open, ``close_on_timeout=True`` (default)
           closes it urgently; ``False`` returns ``None`` and leaves it open.
        4. After ``max_quote_failures`` consecutive polls without a usable
           quote the position is closed urgently while the connection is up;
           while it is down the loop waits for the reconnect (the client
           resubscribes the streams). ``None`` disables this guard.

        Returns the last closing :class:`TrackedOrder`, or ``None`` if nothing
        was filled or no close was sent.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait if max_wait is not None else None

        filled = await self.await_entry(entry, max_wait=max_wait, quantity=plan.params.quantity)
        if filled <= 0:
            return None

        remaining = filled
        last_order: Optional[TrackedOrder] = None
        failures = 0
        self.watch(plan)
        try:
            while remaining > 1e-9:
                now = loop.time()
                if deadline is not None and now >= deadline:
                    if not close_on_timeout:
                        logger.warning(
                            f"CreditSpread: monitor timed out after {max_wait}s — leaving {remaining:g} open"
                        )
                        return None
                    logger.warning(f"CreditSpread: monitor timed out after {max_wait}s — closing {remaining:g}")
                    return (await self.close_and_confirm(plan, remaining, urgent=True)).order

                sleep_for = poll_interval if deadline is None else min(poll_interval, deadline - now)
                await asyncio.sleep(max(sleep_for, 0.0))
                mid_debit = await self.current_mid_debit(plan)
                if mid_debit is None:
                    failures += 1
                    if max_quote_failures is not None and failures >= max_quote_failures:
                        if self._is_connected():
                            logger.error(
                                f"CreditSpread: no usable quote for {failures} polls — closing {remaining:g} blind"
                            )
                            return (await self.close_and_confirm(plan, remaining, urgent=True)).order
                        if failures == max_quote_failures:
                            logger.error("CreditSpread: no quotes and TWS disconnected — waiting for reconnect")
                    elif failures % 5 == 0:
                        logger.warning(f"CreditSpread: monitor — no usable quote for {failures} polls")
                    continue
                failures = 0

                if plan.stop_loss_debit is not None and mid_debit >= plan.stop_loss_debit:
                    logger.warning(
                        f"CreditSpread: stop-loss hit — mid debit {mid_debit:.2f} >= SL {plan.stop_loss_debit:.2f}"
                    )
                    return (await self.close_and_confirm(plan, remaining, urgent=True, mid_debit=mid_debit)).order

                if plan.take_profit_debit is not None and mid_debit <= plan.take_profit_debit:
                    logger.info(
                        f"CreditSpread: take-profit hit — mid debit {mid_debit:.2f} <= TP {plan.take_profit_debit:.2f}"
                    )
                    result = await self.close_and_confirm(plan, remaining, urgent=False, mid_debit=mid_debit)
                    last_order = result.order or last_order
                    remaining = result.remaining_quantity
            return last_order
        finally:
            self.unwatch(plan)

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------

    def watch(self, plan: CreditSpreadPlan) -> None:
        """Start streaming both legs' quotes (reference-counted across plans)."""
        for leg in (plan.short_leg, plan.long_leg):
            contract = leg.quote.contract
            entry = self._streams.get(contract.conId)
            if entry is not None:
                entry[1] += 1
                continue
            ticker = self._client.ib.reqMktData(contract)
            self._streams[contract.conId] = [ticker, 1, contract]

    def unwatch(self, plan: CreditSpreadPlan) -> None:
        """Release the subscriptions taken by :meth:`watch`."""
        for leg in (plan.short_leg, plan.long_leg):
            entry = self._streams.get(leg.quote.contract.conId)
            if entry is None:
                continue
            entry[1] -= 1
            if entry[1] <= 0:
                del self._streams[leg.quote.contract.conId]
                try:
                    self._client.ib.cancelMktData(entry[2])
                except Exception:  # noqa: BLE001
                    logger.exception(f"CreditSpread: cancelMktData failed for conId={leg.quote.contract.conId}")

    def _resubscribe(self) -> None:
        """Reconnect hook: IB dropped every subscription, request them again."""
        for entry in self._streams.values():
            entry[0] = self._client.ib.reqMktData(entry[2])
        if self._streams:
            logger.info(f"CreditSpread: resubscribed {len(self._streams)} leg quote stream(s)")

    async def current_mid_debit(self, plan: CreditSpreadPlan) -> Optional[float]:
        """Current mid debit per share to close the spread, or ``None``.

        Uses the streaming tickers while the plan is watched (no IB round
        trip; quotes older than ``quote_max_age`` count as missing), and a
        one-off snapshot otherwise. ``None`` when either leg lacks a sane
        two-sided quote – the caller is expected to retry rather than treat
        it as zero.
        """
        short_c = plan.short_leg.quote.contract
        long_c = plan.long_leg.quote.contract
        short_entry = self._streams.get(short_c.conId)
        long_entry = self._streams.get(long_c.conId)
        if short_entry is not None and long_entry is not None:
            short_t, long_t = short_entry[0], long_entry[0]
            if self._is_stale(short_t) or self._is_stale(long_t):
                return None
        else:
            short_t, long_t = await snapshot_each(
                self._client.ib, [short_c, long_c], timeout=15.0, regulatorySnapshot=False
            )
            if short_t is None or long_t is None:
                return None
        if plan.params.require_live_quotes and not (_is_live(short_t) and _is_live(long_t)):
            return None
        return _spread_mid_debit(short_t, long_t)

    # Backwards-compatible name used by earlier callers.
    _current_mid_debit = current_mid_debit

    def _is_stale(self, ticker: Any) -> bool:
        t = getattr(ticker, "time", None)
        if t is None or self._quote_max_age is None:
            return False
        try:
            return time.time() - t.timestamp() > self._quote_max_age
        except (AttributeError, TypeError, ValueError, OSError):
            return False

    def _is_connected(self) -> bool:
        check = getattr(self._client, "is_connected", None)
        try:
            return bool(check()) if callable(check) else bool(self._client.ib.isConnected())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _materialise_plan(
        self,
        params: CreditSpreadParams,
        expiry: str,
        short_quote: OptionQuote,
        long_quote: OptionQuote,
        multiplier: str,
        spot: Optional[float],
    ) -> CreditSpreadPlan:
        short_mid = _quote_mid(short_quote)
        long_mid = _quote_mid(long_quote)
        if short_mid is None or long_mid is None:
            raise CreditSpreadError(
                f"Missing bid/ask on a leg (short_mid={short_mid}, long_mid={long_mid}) — "
                f"market may be closed or feed paused"
            )

        mult = float(multiplier) if multiplier else 100.0
        width = abs(short_quote.contract.strike - long_quote.contract.strike)
        net_credit = (short_mid - long_mid) * mult
        if net_credit <= 0:
            raise CreditSpreadError(
                f"Computed net credit {net_credit:.2f} <= 0 — short leg is not richer than long leg "
                f"(short_mid={short_mid:.2f}, long_mid={long_mid:.2f})"
            )

        max_loss = width * mult - net_credit
        max_profit = net_credit
        if params.spread_type is SpreadType.BULL_PUT:
            breakeven = short_quote.contract.strike - net_credit / mult
        else:
            breakeven = short_quote.contract.strike + net_credit / mult

        bag = self._build_bag(params, short_quote, long_quote)

        tp_debit: Optional[float] = None
        sl_debit: Optional[float] = None
        if params.take_profit_pct is not None:
            # We exit when remaining debit equals (1 - tp_pct) * original credit per share.
            tp_debit = (1.0 - params.take_profit_pct) * (net_credit / mult)
        if params.stop_loss_multiplier is not None:
            # Stop when round-trip loss = multiplier * credit, i.e. close debit =
            # original credit + multiplier * original credit (per share).
            sl_debit = (1.0 + params.stop_loss_multiplier) * (net_credit / mult)
            # Cap at the spread width — losing more than width is impossible.
            sl_debit = min(sl_debit, width)

        return CreditSpreadPlan(
            spread_type=params.spread_type,
            underlying_symbol=short_quote.contract.symbol,
            expiry=expiry,
            short_leg=SpreadLeg(quote=short_quote, action=OrderSide.SELL),
            long_leg=SpreadLeg(quote=long_quote, action=OrderSide.BUY),
            width=width,
            multiplier=mult,
            net_credit=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            short_delta=float(short_quote.delta or 0.0),
            bag=bag,
            take_profit_debit=tp_debit,
            stop_loss_debit=sl_debit,
            params=params,
            spot_price=spot,
        )

    def _build_bag(
        self,
        params: CreditSpreadParams,
        short: OptionQuote,
        long: OptionQuote,
    ) -> Bag:
        """Assemble the BAG combo contract for the two legs.

        Each ``ComboLeg`` ratio is 1; ``action`` is "SELL" on the short leg,
        "BUY" on the long. ``exchange`` defaults to the params.exchange –
        IB requires explicit per-leg routing for combo orders.
        """
        bag = Bag(
            symbol=short.contract.symbol,
            currency=params.currency,
            exchange=params.exchange,
        )
        bag.comboLegs = [
            ComboLeg(
                conId=short.contract.conId,
                ratio=1,
                action="SELL",
                exchange=params.exchange,
            ),
            ComboLeg(
                conId=long.contract.conId,
                ratio=1,
                action="BUY",
                exchange=params.exchange,
            ),
        ]
        return bag

    def _enforce_economics(self, plan: CreditSpreadPlan) -> None:
        """Apply the user's credit-floor and risk-reward constraints."""
        p = plan.params
        if p.min_credit is not None and plan.net_credit < p.min_credit:
            raise CreditSpreadError(f"Net credit {plan.net_credit:.2f} below min_credit {p.min_credit:.2f}")
        if p.min_credit_width_ratio is not None:
            width_dollars = plan.width * plan.multiplier
            ratio = plan.net_credit / width_dollars if width_dollars > 0 else 0.0
            if ratio < p.min_credit_width_ratio:
                raise CreditSpreadError(
                    f"Credit/width ratio {ratio:.3f} below floor {p.min_credit_width_ratio:.3f} "
                    f"(credit={plan.net_credit:.2f}, width$={width_dollars:.2f})"
                )


def _is_live(ticker: Any) -> bool:
    return getattr(ticker, "marketDataType", _LIVE_DATA) == _LIVE_DATA


def _spread_mid_debit(short_t: Any, long_t: Any) -> Optional[float]:
    """Mid debit per share (short mid − long mid) from two tickers, or ``None``."""
    short_ask = safe_pick_value(short_t, "ask")
    short_bid = safe_pick_value(short_t, "bid")
    long_ask = safe_pick_value(long_t, "ask")
    long_bid = safe_pick_value(long_t, "bid")
    if short_ask is None or short_bid is None or long_ask is None or long_bid is None:
        return None
    # Reject non-positive or crossed quotes — IB sometimes streams a 0
    # bid before the book is loaded, which would silently bias the mid.
    if short_bid <= 0 or short_ask <= 0 or long_bid <= 0 or long_ask <= 0:
        return None
    if short_ask < short_bid or long_ask < long_bid:
        return None
    # Closing debit = buy back short at ask − sell long at bid (worst case),
    # but we use mids for "fair" decisions: short_mid − long_mid.
    return (short_ask + short_bid) / 2.0 - (long_ask + long_bid) / 2.0
