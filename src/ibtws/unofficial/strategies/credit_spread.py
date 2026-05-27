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
  classic "naked-short after a single-leg fill" risk. Exit polling tolerates
  transient quote drop-outs and re-uses the cached IB connection.

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
4. **Manage**   – :meth:`monitor_and_exit` watches the :class:`TrackedOrder`
   for fill and then polls the spread mid-price, closing the position at the
   configured take-profit / stop-loss debit via the same manager.

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
from typing import Iterable, Optional

from ib_async import Bag, ComboLeg

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher, OptionQuote
from ibtws.unofficial.helpers import safe_pick_value
from ibtws.unofficial.order.manager import OrderManager
from ibtws.unofficial.order.models import OrderSide, OrderState, TimeInForce, TrackedOrder

from .models import CreditSpreadParams, CreditSpreadPlan, SpreadLeg, SpreadType
from .utils import (
    CreditSpreadError,
    _quote_mid,
    _round_to_tick,
    select_expiry,
    select_long_leg,
    select_short_leg,
)

logger = logging.getLogger(__name__)


class CreditSpreadStrategy:
    """Build, place and manage two-leg vertical credit spreads.

    Stateless across calls — each :meth:`build_plan` /
    :meth:`place` invocation can use different parameters. The class only
    caches the :class:`OptionChainFetcher` it was given (or instantiated) so
    consecutive plans on the same underlying re-use the chain definition.

    Example
    -------
    >>> async with IBKRClient(cfg) as client:        # doctest: +SKIP
    ...     await client.connect()
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
    ) -> None:
        if order_manager is None:
            raise ValueError("order_manager is required — combo placement goes through OrderManager.")
        self._client = client
        self._om = order_manager
        self._fetcher = fetcher or OptionChainFetcher(client)
        self._tick = tick_size

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
        value is rounded to the nearest tick and then **negated** before
        being sent to IB.

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
        limit_credit = _round_to_tick(limit_credit, self._tick)
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
    ) -> TrackedOrder:
        """Close the spread by selling the BAG (reversing leg actions) at a debit.

        IB combo convention: SELL the same BAG reverses the leg directions
        (BUY back the short, SELL the long). The limit price on a SELL order
        is the minimum net credit you'll accept — negative means you're
        willing to pay a debit. So closing at $0.20 debit → SELL @ -0.20.
        """
        if limit_debit is None:
            limit_debit = plan.take_profit_debit
        if limit_debit is None:
            mid_debit = await self._current_mid_debit(plan)
            if mid_debit is None:
                raise CreditSpreadError("No mid-debit available and no limit_debit provided")
            limit_debit = mid_debit * (1.0 + plan.params.limit_slippage)

        limit_debit = _round_to_tick(limit_debit, self._tick)
        if limit_debit <= 0:
            raise CreditSpreadError(f"Close limit debit {limit_debit:.2f} <= 0")

        # SELL BAG @ negative = pay debit to close (IB reverses leg actions).
        signed_limit = -limit_debit
        logger.info(
            f"CreditSpread: closing combo SELL x{plan.params.quantity} @ net {signed_limit:.2f} "
            f"(debit {limit_debit:.2f}, {plan.describe()})"
        )
        return await self._om.limit(
            plan.bag,
            OrderSide.SELL,
            plan.params.quantity,
            signed_limit,
            tif=tif or plan.params.tif,
            account=plan.params.account,
            outside_rth=plan.params.outside_rth,
        )

    # ------------------------------------------------------------------
    # Stage 4b: monitor + exit
    # ------------------------------------------------------------------

    async def monitor_and_exit(
        self,
        plan: CreditSpreadPlan,
        entry: TrackedOrder,
        *,
        poll_interval: float = 15.0,
        max_wait: Optional[float] = None,
    ) -> Optional[TrackedOrder]:
        """Poll the spread mid-price and exit at TP / SL.

        Blocks until one of the following happens (whichever comes first):

        * the spread mid debit drops to ``plan.take_profit_debit``;
        * the spread mid debit rises to ``plan.stop_loss_debit``;
        * the entry order is cancelled or rejected;
        * ``max_wait`` seconds elapse (returns ``None``).

        Quote drop-outs (mid unavailable, IB pacing) are logged and skipped –
        the loop keeps polling rather than aborting.

        Returns the closing :class:`TrackedOrder`, or ``None`` if no close was
        sent (timeout or entry cancelled before fill).
        """
        if not (await self._await_entry_fill(entry, max_wait=max_wait)):
            return None

        started = time.monotonic()
        while True:
            if max_wait is not None and time.monotonic() - started > max_wait:
                logger.info(f"CreditSpread: monitor_and_exit timed out after {max_wait}s")
                return None

            await asyncio.sleep(poll_interval)
            mid_debit = await self._current_mid_debit(plan)
            if mid_debit is None:
                logger.debug("CreditSpread: monitor – no mid available, retrying")
                continue

            if plan.take_profit_debit is not None and mid_debit <= plan.take_profit_debit:
                logger.info(
                    f"CreditSpread: take-profit hit — mid debit {mid_debit:.2f} <= TP {plan.take_profit_debit:.2f}"
                )
                return await self.close(plan, limit_debit=mid_debit * (1.0 + plan.params.limit_slippage))

            if plan.stop_loss_debit is not None and mid_debit >= plan.stop_loss_debit:
                logger.warning(
                    f"CreditSpread: stop-loss hit — mid debit {mid_debit:.2f} >= SL {plan.stop_loss_debit:.2f}"
                )
                return await self.close(plan, limit_debit=mid_debit * (1.0 + plan.params.limit_slippage))

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

    async def _current_mid_debit(self, plan: CreditSpreadPlan) -> Optional[float]:
        """Re-quote both legs and return current mid-debit per share.

        Returns ``None`` when either leg has no two-sided quote – the caller
        is expected to retry rather than treat it as zero.
        """
        try:
            tickers = await asyncio.wait_for(
                self._client.ib.reqTickersAsync(
                    plan.short_leg.quote.contract,
                    plan.long_leg.quote.contract,
                    regulatorySnapshot=False,
                ),
                timeout=15.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"CreditSpread: failed to re-quote spread for monitor: {exc}")
            return None
        if len(tickers) != 2:
            return None
        short_t, long_t = tickers
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
        short_mid = (short_ask + short_bid) / 2.0
        long_mid = (long_ask + long_bid) / 2.0
        return short_mid - long_mid

    async def _await_entry_fill(
        self,
        entry: TrackedOrder,
        *,
        max_wait: Optional[float],
        poll: float = 1.0,
    ) -> bool:
        """Wait until the entry :class:`TrackedOrder` is filled.

        Returns ``False`` if the order ends up cancelled, rejected, or the
        ``max_wait`` deadline elapses. ``TrackedOrder.state`` is kept in sync
        by the manager's IB event handlers, so this is just a poll loop on
        a locally-maintained value (no IB round-trips).
        """
        deadline = time.monotonic() + max_wait if max_wait else None
        terminal_failure = {OrderState.CANCELLED, OrderState.REJECTED, OrderState.INACTIVE}
        while True:
            if entry.state == OrderState.FILLED and entry.remaining == 0:
                return True
            if entry.state in terminal_failure:
                logger.warning(
                    f"CreditSpread: entry order {entry.uuid} ended in {entry.state.value} — skipping monitor"
                )
                return False
            if deadline is not None and time.monotonic() > deadline:
                logger.info("CreditSpread: timed out waiting for entry fill")
                return False
            await asyncio.sleep(poll)
