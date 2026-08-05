"""Tests for ``ibtws.unofficial.order.factory``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from ib_async import LimitOrder, MarketOrder, StopOrder

from ibtws.unofficial.order import (
    BracketRequest,
    LimitRequest,
    MarketRequest,
    OrderSide,
    StopRequest,
    TimeInForce,
    build_bracket,
    build_limit,
    build_market,
    build_stop,
)
from ibtws.unofficial.order.factory import bracket_to_orders, request_to_order

from .conftest import make_contract


def test_build_market_returns_request():
    c = make_contract()
    req = build_market(c, OrderSide.BUY, 5)
    assert isinstance(req, MarketRequest)
    assert req.side == OrderSide.BUY
    assert req.quantity == 5
    assert req.tif == TimeInForce.DAY


def test_build_limit_validates_price():
    c = make_contract()
    with pytest.raises(ValueError, match="limit_price"):
        build_limit(c, OrderSide.BUY, 1, 0.0)


def test_build_stop_returns_request():
    c = make_contract()
    req = build_stop(c, OrderSide.SELL, 2, 95.0)
    assert isinstance(req, StopRequest)
    assert req.stop_price == 95.0


def test_build_bracket_geometry_buy():
    c = make_contract()
    with pytest.raises(ValueError, match="BUY bracket"):
        build_bracket(c, OrderSide.BUY, 1, take_profit_price=100, stop_loss_price=110)


def test_build_bracket_geometry_sell():
    c = make_contract()
    with pytest.raises(ValueError, match="SELL bracket"):
        build_bracket(c, OrderSide.SELL, 1, take_profit_price=110, stop_loss_price=100)


def test_build_bracket_happy():
    c = make_contract()
    req = build_bracket(c, OrderSide.BUY, 1, take_profit_price=110, stop_loss_price=90)
    assert isinstance(req, BracketRequest)
    assert req.tif == TimeInForce.DAY


def test_build_quantity_must_be_positive():
    c = make_contract()
    with pytest.raises(ValueError, match="quantity"):
        build_market(c, OrderSide.BUY, 0)


def test_build_unqualified_contract_rejected():
    c = make_contract(con_id=0)
    with pytest.raises(ValueError, match="qualified"):
        build_market(c, OrderSide.BUY, 1)


def test_request_to_order_market():
    req = MarketRequest(contract=make_contract(), side=OrderSide.BUY, quantity=3)
    order = request_to_order(req, "uuid-123")
    assert isinstance(order, MarketOrder)
    assert order.action == "BUY"
    assert order.totalQuantity == 3
    assert order.orderRef == "uuid-123"
    assert order.tif == "DAY"


def test_request_to_order_limit():
    req = LimitRequest(contract=make_contract(), side=OrderSide.SELL, quantity=1, limit_price=99.5)
    order = request_to_order(req, "u")
    assert isinstance(order, LimitOrder)
    assert order.lmtPrice == 99.5
    assert order.action == "SELL"


def test_request_to_order_stop():
    req = StopRequest(contract=make_contract(), side=OrderSide.SELL, quantity=1, stop_price=95)
    order = request_to_order(req, "u")
    assert isinstance(order, StopOrder)
    assert order.auxPrice == 95


def test_request_to_order_account_propagates():
    req = MarketRequest(contract=make_contract(), side=OrderSide.BUY, quantity=1, account="DU777")
    order = request_to_order(req, "u")
    assert order.account == "DU777"


def test_bracket_to_orders_market_entry():
    req = BracketRequest(
        contract=make_contract(),
        side=OrderSide.BUY,
        quantity=2,
        take_profit_price=110,
        stop_loss_price=90,
    )
    parent, tp, sl = bracket_to_orders(req, "p", "tp", "sl", parent_order_id=42, oca_group="grp-1")
    assert isinstance(parent, MarketOrder)
    assert parent.action == "BUY"
    assert parent.orderId == 42
    assert isinstance(tp, LimitOrder) and tp.action == "SELL" and tp.lmtPrice == 110
    assert isinstance(sl, StopOrder) and sl.action == "SELL" and sl.auxPrice == 90
    assert parent.transmit is False
    assert tp.transmit is False
    assert sl.transmit is True
    assert parent.orderRef == "p"
    assert tp.orderRef == "tp"
    assert sl.orderRef == "sl"
    # IB bracket wiring
    assert tp.parentId == 42 and sl.parentId == 42
    assert tp.ocaGroup == "grp-1" and sl.ocaGroup == "grp-1"
    assert tp.ocaType == 1 and sl.ocaType == 1


def test_bracket_to_orders_limit_entry_sell():
    req = BracketRequest(
        contract=make_contract(),
        side=OrderSide.SELL,
        quantity=1,
        take_profit_price=90,
        stop_loss_price=110,
        entry_limit_price=100,
    )
    parent, tp, sl = bracket_to_orders(req, "p", "tp", "sl", parent_order_id=7, oca_group="g")
    assert isinstance(parent, LimitOrder)
    assert parent.action == "SELL"
    assert parent.lmtPrice == 100
    assert tp.action == "BUY"
    assert sl.action == "BUY"
    assert tp.parentId == 7 and sl.parentId == 7
    assert tp.ocaGroup == "g" and sl.ocaGroup == "g"


# ---------------------------------------------------------------------------
# TP-only bracket (stop_loss_price=None)
# ---------------------------------------------------------------------------


def _bag_contract(symbol: str = "SPX"):
    """A minimal BAG (combo) contract with two qualified legs."""
    return SimpleNamespace(
        conId=0,
        symbol=symbol,
        secType="BAG",
        exchange="SMART",
        currency="USD",
        strike=0.0,
        right="",
        lastTradeDateOrContractMonth="",
        tradingClass="",
        multiplier="",
        comboLegs=[
            SimpleNamespace(conId=111, ratio=1, action="SELL", exchange="SMART"),
            SimpleNamespace(conId=222, ratio=1, action="BUY", exchange="SMART"),
        ],
    )


def test_build_bracket_tp_only_allows_missing_sl():
    c = make_contract()
    req = build_bracket(c, OrderSide.BUY, 1, take_profit_price=110)
    assert isinstance(req, BracketRequest)
    assert req.stop_loss_price is None


def test_build_bracket_tp_only_skips_geometry_check():
    # Without an SL there is no TP/SL geometry to validate — must not raise.
    c = make_contract()
    req = build_bracket(c, OrderSide.BUY, 1, take_profit_price=100)
    assert req.take_profit_price == 100


def test_bracket_to_orders_tp_only_returns_two_orders():
    req = BracketRequest(
        contract=make_contract(),
        side=OrderSide.BUY,
        quantity=1,
        take_profit_price=110,
        stop_loss_price=None,
        entry_limit_price=100,
    )
    orders = bracket_to_orders(req, "p", "tp", "sl", parent_order_id=5, oca_group="g")
    assert len(orders) == 2
    parent, tp = orders
    assert isinstance(parent, LimitOrder) and parent.transmit is False
    assert isinstance(tp, LimitOrder) and tp.action == "SELL"
    assert tp.parentId == 5
    # TP transmits the group when there is no SL; no OCA group needed.
    assert tp.transmit is True
    assert not getattr(tp, "ocaGroup", "")


def test_build_bracket_combo_allows_signed_prices():
    # A credit spread: BUY BAG @ -credit, TP = SELL BAG @ -tp_debit (negative).
    bag = _bag_contract()
    req = build_bracket(
        bag,
        OrderSide.BUY,
        1,
        take_profit_price=-0.60,
        entry_limit_price=-1.70,
    )
    assert req.stop_loss_price is None
    assert req.take_profit_price == -0.60


def test_bracket_to_orders_combo_tp_only():
    bag = _bag_contract()
    req = BracketRequest(
        contract=bag,
        side=OrderSide.BUY,
        quantity=1,
        take_profit_price=-0.60,
        stop_loss_price=None,
        entry_limit_price=-1.70,
    )
    parent, tp = bracket_to_orders(req, "p", "tp", "sl", parent_order_id=9, oca_group="g")
    assert parent.lmtPrice == -1.70
    assert tp.lmtPrice == -0.60
    assert tp.action == "SELL"
    assert tp.transmit is True
