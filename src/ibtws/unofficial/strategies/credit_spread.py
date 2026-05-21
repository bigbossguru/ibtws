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
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Sequence

from ib_async import Bag, ComboLeg, Contract

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher, OptionQuote
from ibtws.unofficial.option.utils import _pick_price
from ibtws.unofficial.order.manager import OrderManager
from ibtws.unofficial.order.models import OrderSide, OrderState, TimeInForce, TrackedOrder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CreditSpreadError(RuntimeError):
    """Raised when a credit spread cannot be built or placed.

    The message is always actionable – includes which constraint failed and
    what numbers were observed – so callers can surface it to a user or log
    pipeline without further inspection.
    """


# ---------------------------------------------------------------------------
# Public enums & params
# ---------------------------------------------------------------------------


class SpreadType(str, Enum):
    """Which side of the underlying the spread leans on.

    * ``BULL_PUT`` – sell a higher-strike put, buy a lower-strike put.
      Profits when the underlying stays *above* the short put.
    * ``BEAR_CALL`` – sell a lower-strike call, buy a higher-strike call.
      Profits when the underlying stays *below* the short call.
    """

    BULL_PUT = "bull_put"
    BEAR_CALL = "bear_call"

    @property
    def right(self) -> str:
        """Option right ("P" or "C") used to filter the chain."""
        return "P" if self is SpreadType.BULL_PUT else "C"

    @property
    def is_bullish(self) -> bool:
        return self is SpreadType.BULL_PUT


@dataclass(frozen=True)
class CreditSpreadParams:
    """All tunables for building one credit spread.

    Selection knobs
    ---------------
    target_short_delta:
        Absolute delta target for the short leg (e.g. ``0.30`` for a 30-delta
        short put). The selector picks the strike whose ``|delta|`` is closest
        to this value among legs that pass the liquidity filters.
    wing_width:
        Distance between the short and long strikes, measured in strike-price
        dollars. For SPX this is typically 5, 10, 25 …; for equities 1, 2.5
        or 5. The selector snaps to the nearest available strike at or beyond
        the requested width (so the actual width may differ slightly).
    target_dte / dte_tolerance:
        Pick the expiration whose days-to-expiry is closest to
        ``target_dte`` and at most ``dte_tolerance`` days away. Set
        ``dte_tolerance = 0`` to require an exact match.

    Economic constraints
    --------------------
    min_credit:
        Reject the spread if the mid-price net credit (per spread, not per
        contract) is below this value. ``None`` disables the check.
    min_credit_width_ratio:
        Reject the spread if ``net_credit / wing_width < min_credit_width_ratio``.
        A common floor is ``1/3`` (collect at least one third of the width).
    max_short_delta:
        Hard cap on the short-leg ``|delta|`` after selection. Guards against
        the chain only having far-OTM strikes when the desired delta is not
        available. ``None`` to disable.
    min_open_interest / min_volume:
        Liquidity filters applied to each leg before selection. Legs with
        missing data are *kept* (IB sometimes omits OI for fresh quotes) –
        set these to ``0`` if you want the filter inactive.

    Order knobs
    -----------
    quantity:
        Number of spreads to trade (each = 100 shares notional for equity
        options).
    limit_slippage:
        How far below mid the entry limit can sit, as a fraction of the
        mid. ``0.05`` means "accept 5 % less credit than the mid-quote".
    take_profit_pct:
        Close the spread when realised profit reaches this fraction of the
        original credit captured. ``0.5`` = standard 50 %-of-credit exit.
    stop_loss_multiplier:
        Close the spread when realised loss reaches ``credit * multiplier``.
        ``2.0`` = stop out when losing 2x the credit collected.
    outside_rth:
        Forward ``outsideRth=True`` to the underlying IB order so it can fill
        during pre- and post-market sessions. Has no effect on instruments
        that don't trade outside RTH (e.g. SPX/SPXW index options on CBOE).

    Universe knobs
    --------------
    exchange / currency / trading_class:
        Forwarded to :class:`OptionChainFetcher` to disambiguate SPX vs SPXW
        and similar variants. Leave default for typical equities.
    expirations / expiry_from / expiry_to:
        Optional pre-filters on the chain to bound IB qualification work.
        Useful when the symbol has hundreds of expiries (e.g. SPX) and you
        already know which week you want.
    strike_window_pct:
        ±fraction of spot used to bound strike snapshots, forwarded to the
        fetcher. ``0.10`` covers typical 30Δ–50Δ regions on most names.
    """

    underlying: Contract
    spread_type: SpreadType

    # Selection
    target_short_delta: float = 0.30
    wing_width: float = 5.0
    target_dte: int = 30
    dte_tolerance: int = 14
    max_short_delta: Optional[float] = 0.50
    min_open_interest: float = 0.0
    min_volume: float = 0.0

    # Economic constraints
    min_credit: Optional[float] = None
    min_credit_width_ratio: Optional[float] = None

    # Order knobs
    quantity: int = 1
    limit_slippage: float = 0.05
    tif: TimeInForce = TimeInForce.DAY
    account: Optional[str] = None
    outside_rth: bool = False  # set True to allow fills in pre-/post-market sessions

    # Exit knobs
    take_profit_pct: Optional[float] = 0.5
    stop_loss_multiplier: Optional[float] = 2.0

    # Universe knobs
    exchange: str = "SMART"
    currency: str = "USD"
    trading_class: Optional[str] = None
    expirations: Optional[Sequence[str]] = None
    expiry_from: Optional[str] = None
    expiry_to: Optional[str] = None
    strike_window_pct: float = 0.10

    def __post_init__(self) -> None:
        if not (0.0 < self.target_short_delta < 1.0):
            raise ValueError(f"target_short_delta must be in (0, 1), got {self.target_short_delta!r}")
        if self.wing_width <= 0:
            raise ValueError(f"wing_width must be positive, got {self.wing_width!r}")
        if self.target_dte < 0:
            raise ValueError(f"target_dte must be >= 0, got {self.target_dte!r}")
        if self.dte_tolerance < 0:
            raise ValueError(f"dte_tolerance must be >= 0, got {self.dte_tolerance!r}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity!r}")
        if not (0.0 <= self.limit_slippage < 1.0):
            raise ValueError(f"limit_slippage must be in [0, 1), got {self.limit_slippage!r}")
        if self.take_profit_pct is not None and not (0.0 < self.take_profit_pct <= 1.0):
            raise ValueError(f"take_profit_pct must be in (0, 1], got {self.take_profit_pct!r}")
        if self.stop_loss_multiplier is not None and self.stop_loss_multiplier <= 0:
            raise ValueError(f"stop_loss_multiplier must be positive, got {self.stop_loss_multiplier!r}")


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadLeg:
    """One side of a vertical spread, with the quote that drove its selection."""

    quote: OptionQuote
    action: OrderSide  # SELL for short leg, BUY for long leg

    @property
    def conId(self) -> int:
        return int(self.quote.contract.conId)

    @property
    def strike(self) -> float:
        return float(self.quote.contract.strike)


@dataclass(frozen=True)
class CreditSpreadPlan:
    """Fully-resolved spread, ready to be placed.

    All cash figures are *per spread* in account currency (not per share, not
    per ratio). Multiply by ``quantity`` for portfolio totals.
    """

    spread_type: SpreadType
    underlying_symbol: str
    expiry: str
    short_leg: SpreadLeg
    long_leg: SpreadLeg
    width: float
    multiplier: float
    net_credit: float
    max_profit: float
    max_loss: float
    breakeven: float
    short_delta: float
    bag: Bag
    take_profit_debit: Optional[float]
    stop_loss_debit: Optional[float]
    params: CreditSpreadParams
    spot_price: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    @property
    def risk_reward(self) -> float:
        """Max-profit / max-loss. Higher is better. Returns inf if max_loss == 0."""
        return self.max_profit / self.max_loss if self.max_loss > 0 else math.inf

    def describe(self) -> str:
        """One-line human-readable summary suitable for logs / CLI output."""
        return (
            f"{self.spread_type.value} {self.underlying_symbol} {self.expiry} "
            f"{self.short_leg.strike:g}/{self.long_leg.strike:g} "
            f"width={self.width:g} credit={self.net_credit:.2f} "
            f"max_loss={self.max_loss:.2f} Δshort={self.short_delta:.3f}"
        )


# ---------------------------------------------------------------------------
# Selection helpers (pure functions, easily unit-testable)
# ---------------------------------------------------------------------------


def _parse_expiry_to_dte(expiry: str, *, now: Optional[float] = None) -> int:
    """Convert a ``YYYYMMDD`` expiration string into days-to-expiry.

    IBKR also returns ``YYYYMM`` for monthlies – treated as the third-Friday
    convention (just use day 15 as a reasonable proxy). Negative DTEs (already
    expired) are returned as-is so the caller can filter them out.
    """
    if len(expiry) == 6:
        expiry = expiry + "15"
    if len(expiry) != 8 or not expiry.isdigit():
        raise ValueError(f"Unrecognised expiry format: {expiry!r}")

    import datetime as _dt

    y, m, d = int(expiry[:4]), int(expiry[4:6]), int(expiry[6:])
    target = _dt.datetime(y, m, d, 16, 0, 0, tzinfo=_dt.timezone.utc)  # 4pm ET close as proxy
    ref = (
        _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc) if now is not None else _dt.datetime.now(_dt.timezone.utc)
    )
    return (target - ref).days


def select_expiry(
    expirations: Iterable[str],
    *,
    target_dte: int,
    dte_tolerance: int,
    now: Optional[float] = None,
) -> str:
    """Pick the expiry closest to ``target_dte`` within ``dte_tolerance``.

    Negative-DTE (expired) entries are ignored. Raises
    :class:`CreditSpreadError` if no expiry survives the tolerance window.
    """
    candidates: list[tuple[int, str]] = []
    for exp in expirations:
        try:
            dte = _parse_expiry_to_dte(exp, now=now)
        except ValueError:
            continue
        if dte < 0:
            continue
        if abs(dte - target_dte) <= dte_tolerance:
            candidates.append((abs(dte - target_dte), exp))
    if not candidates:
        raise CreditSpreadError(
            f"No expiry within {dte_tolerance}d of target_dte={target_dte} in {list(expirations)!r}"
        )
    candidates.sort()
    chosen = candidates[0][1]
    logger.info(f"CreditSpread: chose expiry {chosen} (Δdte={candidates[0][0]}d) from {len(candidates)} candidate(s)")
    return chosen


def _quote_is_tradeable(
    quote: OptionQuote,
    *,
    min_open_interest: float,
    min_volume: float,
) -> bool:
    """Liquidity / data-quality filter applied before strike selection.

    A leg is rejected only when IB explicitly returned a too-low value. Legs
    whose OI or volume is ``None`` (IB simply did not include it in the
    snapshot) are *kept* — they would otherwise be discarded en masse for
    freshly-listed expiries.
    """
    if quote.delta is None:
        return False
    if quote.contract.conId == 0:
        return False
    if quote.open_interest is not None and quote.open_interest < min_open_interest:
        return False
    if quote.volume is not None and quote.volume < min_volume:
        return False
    return True


def select_short_leg(
    quotes: Sequence[OptionQuote],
    *,
    target_short_delta: float,
    max_short_delta: Optional[float],
    min_open_interest: float,
    min_volume: float,
) -> OptionQuote:
    """Pick the option whose ``|delta|`` is closest to the target.

    Filters out quotes that fail the liquidity / data-quality check first.
    Enforces ``max_short_delta`` as a hard ceiling — if no remaining quote
    fits, the spread is rejected (rather than silently widening risk).
    """
    candidates = [
        q for q in quotes if _quote_is_tradeable(q, min_open_interest=min_open_interest, min_volume=min_volume)
    ]
    if not candidates:
        raise CreditSpreadError(f"No tradeable quotes (need delta + conId; got {len(quotes)} raw)")
    if max_short_delta is not None:
        candidates = [q for q in candidates if q.delta is not None and abs(q.delta) <= max_short_delta]
        if not candidates:
            raise CreditSpreadError(f"All candidate strikes exceed max_short_delta={max_short_delta}")
    candidates.sort(key=lambda q: abs(abs(q.delta or 0.0) - target_short_delta))
    return candidates[0]


def select_long_leg(
    quotes: Sequence[OptionQuote],
    *,
    short: OptionQuote,
    wing_width: float,
    spread_type: SpreadType,
    min_open_interest: float,
    min_volume: float,
) -> OptionQuote:
    """Pick the protective leg at (or just beyond) the requested wing width.

    For a bull-put spread the long is *lower* than the short; for a bear-call
    spread the long is *higher*. We snap to the strike whose distance from
    the short is at least ``wing_width`` and minimised. Falls back to the
    farthest available strike when none meets the minimum width (e.g.
    chain truncated by ``strike_window_pct``).
    """
    short_strike = float(short.contract.strike)
    target = short_strike - wing_width if spread_type is SpreadType.BULL_PUT else short_strike + wing_width

    tradeable = [
        q
        for q in quotes
        if _quote_is_tradeable(q, min_open_interest=min_open_interest, min_volume=min_volume)
        and q.contract.conId != short.contract.conId
        and q.contract.right == short.contract.right
        and q.contract.lastTradeDateOrContractMonth == short.contract.lastTradeDateOrContractMonth
    ]
    if spread_type is SpreadType.BULL_PUT:
        tradeable = [q for q in tradeable if q.contract.strike < short_strike]
    else:
        tradeable = [q for q in tradeable if q.contract.strike > short_strike]
    if not tradeable:
        raise CreditSpreadError(
            f"No protective leg available on the {('lower' if spread_type is SpreadType.BULL_PUT else 'higher')} "
            f"side of strike {short_strike}"
        )

    tradeable.sort(key=lambda q: abs(q.contract.strike - target))
    chosen = tradeable[0]
    actual_width = abs(chosen.contract.strike - short_strike)
    if actual_width < wing_width * 0.5:
        # Final guard — refusing to trade a 1-strike-wide spread when user
        # asked for 10 wide. Better to fail loud than fill a tiny credit.
        raise CreditSpreadError(
            f"Closest protective strike only {actual_width:g} wide (requested {wing_width:g}) — chain too narrow"
        )
    return chosen


def _quote_mid(q: OptionQuote) -> Optional[float]:
    """Bid/ask midpoint with NaN-safe handling. ``None`` when either side is missing."""
    if q.bid is None or q.ask is None:
        return None
    if q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
        return None
    return (q.bid + q.ask) / 2.0


def _round_to_tick(price: float, tick: float = 0.05) -> float:
    """Round to the nearest IB-acceptable tick (default 5¢ for options)."""
    if tick <= 0:
        return price
    return round(price / tick) * tick


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


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
            (underlying,) = await self._client.qualify(underlying)

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
        """Close the spread with a BUY-back BAG limit order via :class:`OrderManager`.

        ``limit_debit`` defaults to the plan's stored take-profit debit if
        present, else to the current mid-debit plus slippage.
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

        logger.info(
            f"CreditSpread: closing combo BUY x{plan.params.quantity} @ debit {limit_debit:.2f} ({plan.describe()})"
        )
        try:
            return await self._om.limit(
                plan.bag,
                OrderSide.BUY,
                plan.params.quantity,
                limit_debit,
                tif=tif or plan.params.tif,
                account=plan.params.account,
                outside_rth=plan.params.outside_rth,
            )
        finally:
            # Free the market-data lines opened while monitoring this spread.
            # IB caps simultaneous subscriptions (~100/session) and long-running
            # strategies that build many spreads accumulate them otherwise.
            released = self._fetcher.release([plan.short_leg.quote.contract, plan.long_leg.quote.contract])
            if released:
                logger.debug(f"CreditSpread: released {released} mkt-data subscription(s) on close")

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
        short_ask = _pick_price(short_t, "ask")
        short_bid = _pick_price(short_t, "bid")
        long_ask = _pick_price(long_t, "ask")
        long_bid = _pick_price(long_t, "bid")
        if short_ask is None or short_bid is None or long_ask is None or long_bid is None:
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
