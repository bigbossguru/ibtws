"""Plain dataclasses returned by :mod:`ibtws.unofficial.option`."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

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
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None

    # From modelGreeks (preferred — uses IB's pricing model with current vol surface).
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    underlying_price: Optional[float] = None

    timestamp: float = field(default_factory=time.time)
