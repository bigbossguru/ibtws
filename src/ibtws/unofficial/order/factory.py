# Non-official package. Not affiliated with ib_async upstream.

"""Order-request builders + ``*Request → ib_async.Order`` translation.

Two layers:

* ``build_*`` — public, type-safe constructors that produce frozen dataclasses.
  Use these from strategy code.
* ``request_to_order`` / ``bracket_to_orders`` — internal, called by
  :class:`OrderManager` to materialise an ib_async ``Order`` just before
  ``placeOrder``. Kept here so all order-shape knowledge lives in one file.
"""

from __future__ import annotations

from typing import Any, Optional

from ib_async import LimitOrder, MarketOrder, Order, StopOrder

from .models import (
    BracketRequest,
    LimitRequest,
    MarketRequest,
    OrderRequest,
    OrderSide,
    StopRequest,
    TimeInForce,
)
from .utils import validate_request


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_market(
    contract: Any,
    side: OrderSide,
    quantity: float,
    *,
    tif: TimeInForce = TimeInForce.DAY,
    account: Optional[str] = None,
) -> MarketRequest:
    req = MarketRequest(contract=contract, side=side, quantity=quantity, tif=tif, account=account)
    validate_request(req)
    return req


def build_limit(
    contract: Any,
    side: OrderSide,
    quantity: float,
    limit_price: float,
    *,
    tif: TimeInForce = TimeInForce.DAY,
    account: Optional[str] = None,
) -> LimitRequest:
    req = LimitRequest(
        contract=contract, side=side, quantity=quantity, limit_price=limit_price, tif=tif, account=account
    )
    validate_request(req)
    return req


def build_stop(
    contract: Any,
    side: OrderSide,
    quantity: float,
    stop_price: float,
    *,
    tif: TimeInForce = TimeInForce.DAY,
    account: Optional[str] = None,
) -> StopRequest:
    req = StopRequest(contract=contract, side=side, quantity=quantity, stop_price=stop_price, tif=tif, account=account)
    validate_request(req)
    return req


def build_bracket(
    contract: Any,
    side: OrderSide,
    quantity: float,
    *,
    take_profit_price: float,
    stop_loss_price: float,
    entry_limit_price: Optional[float] = None,
    tif: TimeInForce = TimeInForce.DAY,
    account: Optional[str] = None,
) -> BracketRequest:
    req = BracketRequest(
        contract=contract,
        side=side,
        quantity=quantity,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        entry_limit_price=entry_limit_price,
        tif=tif,
        account=account,
    )
    validate_request(req)
    return req


# ---------------------------------------------------------------------------
# Request → ib_async.Order
# ---------------------------------------------------------------------------


def request_to_order(request: OrderRequest, order_ref: str) -> Order:
    """Convert a single-leg request to an ``ib_async.Order`` ready for ``placeOrder``."""
    order: Order
    if isinstance(request, MarketRequest):
        order = MarketOrder(request.side.value, request.quantity)
    elif isinstance(request, LimitRequest):
        order = LimitOrder(request.side.value, request.quantity, request.limit_price)
    elif isinstance(request, StopRequest):
        order = StopOrder(request.side.value, request.quantity, request.stop_price)
    else:
        raise TypeError(f"request_to_order does not handle {type(request).__name__}")
    order.tif = request.tif.value
    order.orderRef = order_ref
    if request.account:
        order.account = request.account
    return order


def bracket_to_orders(
    request: BracketRequest,
    parent_ref: str,
    tp_ref: str,
    sl_ref: str,
    *,
    parent_order_id: int,
    oca_group: str,
) -> list[Order]:
    """Build a parent + take-profit + stop-loss triplet with real IB bracket wiring.

    Two IB mechanisms are applied:

    * ``child.parentId = parent.orderId`` — IB cancels the children automatically
      if the parent is cancelled pre-fill, and activates them once parent fills.
    * Shared ``ocaGroup`` + ``ocaType=1`` on TP and SL — when one fills or is
      cancelled, IB cancels the other (cancel-with-block).

    The parent's ``orderId`` must be pre-allocated by the caller (typically via
    ``ib.client.getReqId()``) so the children's ``parentId`` can reference it
    before any ``placeOrder`` call.
    """
    exit_side = OrderSide.SELL if request.side == OrderSide.BUY else OrderSide.BUY

    parent: Order
    if request.entry_limit_price is None:
        parent = MarketOrder(request.side.value, request.quantity)
    else:
        parent = LimitOrder(request.side.value, request.quantity, request.entry_limit_price)
    parent.orderId = parent_order_id
    parent.orderRef = parent_ref
    parent.tif = request.tif.value
    parent.transmit = False  # hold until children are queued

    take_profit = LimitOrder(exit_side.value, request.quantity, request.take_profit_price)
    take_profit.orderRef = tp_ref
    take_profit.tif = request.tif.value
    take_profit.parentId = parent_order_id
    take_profit.ocaGroup = oca_group
    take_profit.ocaType = 1  # cancel-with-block
    take_profit.transmit = False

    stop_loss = StopOrder(exit_side.value, request.quantity, request.stop_loss_price)
    stop_loss.orderRef = sl_ref
    stop_loss.tif = request.tif.value
    stop_loss.parentId = parent_order_id
    stop_loss.ocaGroup = oca_group
    stop_loss.ocaType = 1
    stop_loss.transmit = True  # transmits the whole group atomically

    if request.account:
        for o in (parent, take_profit, stop_loss):
            o.account = request.account

    return [parent, take_profit, stop_loss]
