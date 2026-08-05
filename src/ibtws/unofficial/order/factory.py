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
    outside_rth: bool = False,
) -> MarketRequest:
    req = MarketRequest(
        contract=contract,
        side=side,
        quantity=quantity,
        tif=tif,
        account=account,
        outside_rth=outside_rth,
    )
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
    outside_rth: bool = False,
) -> LimitRequest:
    req = LimitRequest(
        contract=contract,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        tif=tif,
        account=account,
        outside_rth=outside_rth,
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
    outside_rth: bool = False,
) -> StopRequest:
    req = StopRequest(
        contract=contract,
        side=side,
        quantity=quantity,
        stop_price=stop_price,
        tif=tif,
        account=account,
        outside_rth=outside_rth,
    )
    validate_request(req)
    return req


def build_bracket(
    contract: Any,
    side: OrderSide,
    quantity: float,
    *,
    take_profit_price: float,
    stop_loss_price: Optional[float] = None,
    entry_limit_price: Optional[float] = None,
    tif: TimeInForce = TimeInForce.DAY,
    account: Optional[str] = None,
    outside_rth: bool = False,
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
        outside_rth=outside_rth,
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
    order.outsideRth = bool(request.outside_rth)
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
    """Build a parent + take-profit (+ optional stop-loss) with IB bracket wiring.

    Two IB mechanisms are applied:

    * ``child.parentId = parent.orderId`` — IB cancels the children automatically
      if the parent is cancelled pre-fill, and activates them once parent fills.
    * When a stop-loss is present, a shared ``ocaGroup`` + ``ocaType=1`` on TP
      and SL means IB cancels the other when one fills or is cancelled
      (cancel-with-block).

    If ``request.stop_loss_price`` is ``None`` only ``[parent, take_profit]`` is
    returned (TP-only bracket); the take-profit then transmits the group.

    The parent's ``orderId`` must be pre-allocated by the caller (typically via
    ``ib.client.getReqId()``) so the children's ``parentId`` can reference it
    before any ``placeOrder`` call.
    """
    exit_side = OrderSide.SELL if request.side == OrderSide.BUY else OrderSide.BUY
    has_sl = request.stop_loss_price is not None

    parent: Order
    if request.entry_limit_price is None:
        parent = MarketOrder(request.side.value, request.quantity)
    else:
        parent = LimitOrder(request.side.value, request.quantity, request.entry_limit_price)
    parent.orderId = parent_order_id
    parent.orderRef = parent_ref
    parent.tif = request.tif.value
    parent.outsideRth = bool(request.outside_rth)
    parent.transmit = False  # hold until children are queued

    take_profit = LimitOrder(exit_side.value, request.quantity, request.take_profit_price)
    take_profit.orderRef = tp_ref
    take_profit.tif = request.tif.value
    take_profit.outsideRth = bool(request.outside_rth)
    take_profit.parentId = parent_order_id
    if has_sl:
        # OCA pair with the stop-loss; SL transmits the whole group.
        take_profit.ocaGroup = oca_group
        take_profit.ocaType = 1  # cancel-with-block
        take_profit.transmit = False
    else:
        # TP-only: this order transmits the group atomically.
        take_profit.transmit = True

    if request.account:
        parent.account = request.account
        take_profit.account = request.account

    if not has_sl:
        return [parent, take_profit]

    stop_loss = StopOrder(exit_side.value, request.quantity, request.stop_loss_price)
    stop_loss.orderRef = sl_ref
    stop_loss.tif = request.tif.value
    stop_loss.outsideRth = bool(request.outside_rth)
    stop_loss.parentId = parent_order_id
    stop_loss.ocaGroup = oca_group
    stop_loss.ocaType = 1
    stop_loss.transmit = True  # transmits the whole group atomically
    if request.account:
        stop_loss.account = request.account

    return [parent, take_profit, stop_loss]
