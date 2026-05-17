# Non-official package. Not affiliated with ib_async upstream.

"""Stateless helpers used across the order package."""

from __future__ import annotations

import uuid as _uuid
from typing import Any

from .models import (
    BracketRequest,
    LimitRequest,
    MarketRequest,
    OrderSide,
    StopRequest,
)


def make_order_ref() -> str:
    """Generate a short, IB-safe orderRef UUID (32 hex chars, well under TWS' limit)."""
    return _uuid.uuid4().hex


def is_paper_account(account_id: str) -> bool:
    """Paper accounts start with ``DU`` (Demo User); live accounts with ``U`` / ``DI`` etc."""
    return bool(account_id) and account_id.upper().startswith("DU")


def validate_request(request: Any) -> None:
    """Reject obviously-malformed requests before they hit IB.

    Raises ``ValueError`` with a precise message. Only catches issues the
    static type system cannot — IB will still bounce anything we miss.
    """
    if not isinstance(request, (MarketRequest, LimitRequest, StopRequest, BracketRequest)):
        raise ValueError(f"Unsupported request type: {type(request).__name__}")

    contract = getattr(request, "contract", None)
    if contract is None or not getattr(contract, "conId", 0):
        raise ValueError("Contract must be qualified (non-zero conId).")

    if not isinstance(request.side, OrderSide):
        raise ValueError(f"side must be OrderSide, got {type(request.side).__name__}")

    if request.quantity is None or request.quantity <= 0:
        raise ValueError(f"quantity must be positive, got {request.quantity!r}")

    if isinstance(request, LimitRequest) and request.limit_price <= 0:
        raise ValueError(f"limit_price must be positive, got {request.limit_price!r}")
    if isinstance(request, StopRequest) and request.stop_price <= 0:
        raise ValueError(f"stop_price must be positive, got {request.stop_price!r}")
    if isinstance(request, BracketRequest):
        if request.take_profit_price <= 0:
            raise ValueError(f"take_profit_price must be positive, got {request.take_profit_price!r}")
        if request.stop_loss_price <= 0:
            raise ValueError(f"stop_loss_price must be positive, got {request.stop_loss_price!r}")
        if request.entry_limit_price is not None and request.entry_limit_price <= 0:
            raise ValueError(f"entry_limit_price must be positive when set, got {request.entry_limit_price!r}")
        # Sanity-check TP/SL geometry vs side.
        if request.side == OrderSide.BUY and request.take_profit_price <= request.stop_loss_price:
            raise ValueError("For BUY bracket: take_profit_price must be above stop_loss_price.")
        if request.side == OrderSide.SELL and request.take_profit_price >= request.stop_loss_price:
            raise ValueError("For SELL bracket: take_profit_price must be below stop_loss_price.")
