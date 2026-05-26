# Non-official package. Not affiliated with ib_async upstream.

"""Dataclasses and enums for the credit-spread strategy layer.

No I/O, no IB calls — pure data definitions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from ib_async import Bag, Contract

from ibtws.unofficial.option import OptionQuote
from ibtws.unofficial.order.models import OrderSide, TimeInForce


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
