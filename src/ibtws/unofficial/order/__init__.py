# Non-official package. Not affiliated with ib_async upstream.

"""Order placement & tracking for IBKR via :class:`IBKRClient`.

Public surface:

* :class:`OrderManager`            — orchestrator: place / cancel / monitor / persist.
* :class:`OrderMonitor`            — fan-out event bus.
* :class:`OrderStore` / :class:`JsonStore` — pluggable persistence (append-only JSONL).
* :class:`ReconciliationReport`    — startup IB-vs-local diff.
* :func:`build_market` / :func:`build_limit` / :func:`build_stop` / :func:`build_bracket`
* Dataclasses: :class:`MarketRequest`, :class:`LimitRequest`, :class:`StopRequest`,
  :class:`BracketRequest`, :class:`TrackedOrder`, :class:`PositionSnapshot`.
* Event types: :class:`RequestSubmitted`, :class:`StatusChanged`, :class:`Filled`,
  :class:`Cancelled`, :class:`Rejected`, :class:`PositionChanged`.
* Enums: :class:`OrderSide`, :class:`TimeInForce`, :class:`OrderState`.
"""

from .factory import build_bracket, build_limit, build_market, build_stop
from .manager import OrderManager
from .models import (
    BracketRequest,
    Cancelled,
    Filled,
    LimitRequest,
    MarketRequest,
    OrderEvent,
    OrderRequest,
    OrderSide,
    OrderState,
    PositionChanged,
    PositionSnapshot,
    Rejected,
    RequestSubmitted,
    StatusChanged,
    StopRequest,
    TimeInForce,
    TrackedOrder,
)
from .monitor import OrderMonitor
from .reconciler import ReconciliationReport, reconcile
from .store import JsonStore, OrderStore

__all__ = [
    "BracketRequest",
    "Cancelled",
    "Filled",
    "JsonStore",
    "LimitRequest",
    "MarketRequest",
    "OrderEvent",
    "OrderManager",
    "OrderMonitor",
    "OrderRequest",
    "OrderSide",
    "OrderState",
    "OrderStore",
    "PositionChanged",
    "PositionSnapshot",
    "ReconciliationReport",
    "Rejected",
    "RequestSubmitted",
    "StatusChanged",
    "StopRequest",
    "TimeInForce",
    "TrackedOrder",
    "build_bracket",
    "build_limit",
    "build_market",
    "build_stop",
    "reconcile",
]
