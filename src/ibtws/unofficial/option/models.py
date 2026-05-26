"""Plain dataclasses returned by :mod:`ibtws.unofficial.option`."""

from __future__ import annotations

import time
import datetime as _dt
from dataclasses import dataclass, field

from ib_async import Option


@dataclass(frozen=True)
class ChainDefinition:
    """Universe of option contracts available for an underlying on one exchange."""

    underlying_conId: int
    underlying_symbol: str
    trading_class: str
    multiplier: str
    exchange: str
    expirations: tuple[str, ...]  # YYYYMMDD strings
    strikes: tuple[float, ...]


@dataclass
class OptionQuote:
    """Snapshot of one option contract's market metrics at a point in time."""

    contract: Option
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    open_interest: float | None = None

    # From modelGreeks (preferred — uses IB's pricing model with current vol surface).
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    underlying_price: float | None = None

    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class IVRankResult:
    """Result of an IV Rank / IV Percentile computation for one underlying."""

    underlying_symbol: str
    as_of: _dt.date | None
    current_iv: float | None
    min_iv: float | None
    max_iv: float | None
    iv_rank: float | None  # 0..100, or None when the band is degenerate
    iv_percentile: float | None  # 0..100
    sample_size: int
    lookback_days: int
