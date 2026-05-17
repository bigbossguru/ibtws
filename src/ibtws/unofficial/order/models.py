# Non-official package. Not affiliated with ib_async upstream.

"""Dataclasses, enums, and event types used across the order package.

No I/O, no IB calls — pure data. ``OrderEvent`` is the discriminated union
emitted by :class:`OrderManager` and persisted by :class:`OrderStore`.
Serialisation is symmetric: ``event.to_dict()`` → ``event_from_dict(data)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderState(str, Enum):
    """Coarse local state — mirrors ib_async ``OrderStatus.status`` collapsed to outcomes."""

    PENDING_SUBMIT = "PendingSubmit"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"
    INACTIVE = "Inactive"


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------


def serialise_contract(contract: Any) -> dict:
    """Flatten an ``ib_async.Contract`` into a JSON-safe dict.

    Stores only the fields needed to round-trip through the audit log; the
    Contract class itself isn't JSON-friendly because of its ``ContractDetails``
    references.
    """
    return {
        "conId": getattr(contract, "conId", 0) or 0,
        "symbol": getattr(contract, "symbol", "") or "",
        "secType": getattr(contract, "secType", "") or "",
        "exchange": getattr(contract, "exchange", "") or "",
        "currency": getattr(contract, "currency", "") or "",
        "strike": getattr(contract, "strike", 0.0) or 0.0,
        "right": getattr(contract, "right", "") or "",
        "lastTradeDateOrContractMonth": getattr(contract, "lastTradeDateOrContractMonth", "") or "",
        "tradingClass": getattr(contract, "tradingClass", "") or "",
        "multiplier": getattr(contract, "multiplier", "") or "",
    }


# ---------------------------------------------------------------------------
# Order requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketRequest:
    contract: Any  # ib_async.Contract — kept typeless to avoid hard dep at module import
    side: OrderSide
    quantity: float
    tif: TimeInForce = TimeInForce.DAY
    account: Optional[str] = None


@dataclass(frozen=True)
class LimitRequest:
    contract: Any
    side: OrderSide
    quantity: float
    limit_price: float
    tif: TimeInForce = TimeInForce.DAY
    account: Optional[str] = None


@dataclass(frozen=True)
class StopRequest:
    contract: Any
    side: OrderSide
    quantity: float
    stop_price: float
    tif: TimeInForce = TimeInForce.DAY
    account: Optional[str] = None


@dataclass(frozen=True)
class BracketRequest:
    """One atomic submit = entry leg + TP limit + SL stop, OCA-grouped."""

    contract: Any
    side: OrderSide
    quantity: float
    take_profit_price: float
    stop_loss_price: float
    entry_limit_price: Optional[float] = None  # None → market entry
    tif: TimeInForce = TimeInForce.GTC
    account: Optional[str] = None


OrderRequest = Union[MarketRequest, LimitRequest, StopRequest]


# ---------------------------------------------------------------------------
# Tracked state
# ---------------------------------------------------------------------------


@dataclass
class TrackedOrder:
    """Live, mutable view of one submitted order — kept in sync by the manager."""

    uuid: str
    request: Optional[Union[OrderRequest, BracketRequest]]
    trade: Any  # ib_async.Trade
    state: OrderState = OrderState.PENDING_SUBMIT
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float = 0.0
    perm_id: int = 0
    bracket_group: Optional[str] = None  # UUID shared by parent/tp/sl
    last_update: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PositionSnapshot:
    account: str
    contract: dict  # serialised
    quantity: float
    avg_cost: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Events (audit log + monitor stream)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestSubmitted:
    uuid: str
    request_kind: str  # "market" | "limit" | "stop" | "bracket"
    contract: dict
    side: str
    quantity: float
    tif: str
    account: Optional[str]
    extra: dict  # kind-specific: limit_price / stop_price / tp / sl
    bracket_group: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": "request_submitted", **self.__dict__}


@dataclass(frozen=True)
class StatusChanged:
    uuid: str
    perm_id: int
    state: str
    filled: float
    remaining: float
    avg_fill_price: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": "status_changed", **self.__dict__}


@dataclass(frozen=True)
class Filled:
    uuid: str
    perm_id: int
    exec_id: str
    price: float
    quantity: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": "fill", **self.__dict__}


@dataclass(frozen=True)
class Cancelled:
    uuid: str
    perm_id: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": "cancelled", **self.__dict__}


@dataclass(frozen=True)
class Rejected:
    uuid: str
    perm_id: int
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": "rejected", **self.__dict__}


@dataclass(frozen=True)
class PositionChanged:
    account: str
    contract: dict
    quantity: float
    avg_cost: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"kind": "position_changed", **self.__dict__}


OrderEvent = Union[RequestSubmitted, StatusChanged, Filled, Cancelled, Rejected, PositionChanged]


_EVENT_TYPES = {
    "request_submitted": RequestSubmitted,
    "status_changed": StatusChanged,
    "fill": Filled,
    "cancelled": Cancelled,
    "rejected": Rejected,
    "position_changed": PositionChanged,
}


def event_from_dict(data: dict) -> OrderEvent:
    """Reverse of ``OrderEvent.to_dict()`` — dispatch on the ``kind`` discriminator."""
    kind = data.get("kind")
    cls = _EVENT_TYPES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ValueError(f"Unknown event kind: {kind!r}")
    payload = {k: v for k, v in data.items() if k != "kind"}
    return cls(**payload)
