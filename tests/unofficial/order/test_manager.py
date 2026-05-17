"""Tests for ``ibtws.unofficial.order.manager``."""

from __future__ import annotations

import asyncio

import pytest

from ibtws.unofficial.order import (
    Cancelled,
    Filled,
    MarketRequest,
    OrderManager,
    OrderSide,
    OrderState,
    PositionChanged,
    Rejected,
    RequestSubmitted,
    StatusChanged,
    build_bracket,
    build_limit,
)

from .conftest import make_contract, make_fill, make_position, make_trade


@pytest.fixture
async def manager(fake_client, tmp_store):
    mgr = OrderManager(fake_client, tmp_store, max_concurrency=10, pace_per_sec=0)
    await mgr.start()
    yield mgr
    await mgr.stop()


# ---------------------------------------------------------------------------
# Lifecycle / safety
# ---------------------------------------------------------------------------


async def test_paper_guard_refuses_live(fake_client, tmp_store):
    fake_client.ib.managedAccounts.return_value = ["U999999"]
    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    with pytest.raises(RuntimeError, match="live account"):
        await mgr.start()


async def test_paper_guard_allows_live_when_overridden(fake_client, tmp_store):
    fake_client.ib.managedAccounts.return_value = ["U999999"]
    mgr = OrderManager(fake_client, tmp_store, allow_live=True, pace_per_sec=0)
    await mgr.start()
    await mgr.stop()


async def test_double_start_raises(fake_client, tmp_store):
    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    await mgr.start()
    with pytest.raises(RuntimeError, match="already started"):
        await mgr.start()
    await mgr.stop()


async def test_methods_require_start(fake_client, tmp_store):
    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    req = MarketRequest(contract=make_contract(), side=OrderSide.BUY, quantity=1)
    with pytest.raises(RuntimeError, match="start"):
        await mgr.place(req)


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------


async def test_place_happy_path(manager, fake_client, tmp_store):
    contract = make_contract()
    trade = make_trade("placeholder", perm_id=42)
    fake_client.ib.placeOrder.return_value = trade

    req = build_limit(contract, OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)

    assert tracked.uuid
    assert tracked.state == OrderState.PENDING_SUBMIT
    assert tracked.remaining == 1.0

    submitted_order = fake_client.ib.placeOrder.call_args.args[1]
    assert submitted_order.orderRef == tracked.uuid
    assert submitted_order.lmtPrice == 100.0

    events = list(tmp_store.replay())
    assert len(events) == 1
    assert isinstance(events[0], RequestSubmitted)
    assert events[0].uuid == tracked.uuid


async def test_place_status_event_updates_tracked(manager, fake_client, tmp_store):
    fake_client.ib.placeOrder.return_value = make_trade("ignored", perm_id=42)
    req = build_limit(make_contract(), OrderSide.BUY, 2, 100.0)
    tracked = await manager.place(req)

    status_trade = make_trade(tracked.uuid, perm_id=42, status="Submitted", filled=0, remaining=2)
    fake_client.ib.orderStatusEvent.fire(status_trade)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert tracked.state == OrderState.SUBMITTED
    # last event in store should be a StatusChanged
    events = list(tmp_store.replay())
    assert any(isinstance(e, StatusChanged) and e.uuid == tracked.uuid for e in events)


async def test_place_fill_event_emits_filled(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=7)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)

    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    trade = make_trade(tracked.uuid, perm_id=7)
    fill = make_fill("E1", price=100.5, shares=1)
    fake_client.ib.execDetailsEvent.fire(trade, fill)
    await asyncio.sleep(0)

    fills = [e for e in seen if isinstance(e, Filled)]
    assert len(fills) == 1
    assert fills[0].uuid == tracked.uuid
    assert fills[0].price == 100.5
    assert fills[0].exec_id == "E1"


async def test_status_event_without_orderref_ignored(manager, fake_client):
    seen: list = []
    manager.on_event(lambda e: seen.append(e))
    fake_client.ib.orderStatusEvent.fire(make_trade("", perm_id=1))
    await asyncio.sleep(0)
    assert [e for e in seen if isinstance(e, StatusChanged)] == []


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------


async def test_place_bracket_submits_three(manager, fake_client):
    fake_client.ib.placeOrder.side_effect = [
        make_trade("p"),
        make_trade("tp"),
        make_trade("sl"),
    ]
    req = build_bracket(make_contract(), OrderSide.BUY, 1, take_profit_price=110, stop_loss_price=90)
    tracked = await manager.place_bracket(req)

    assert len(tracked) == 3
    assert fake_client.ib.placeOrder.call_count == 3
    group = tracked[0].bracket_group
    assert all(t.bracket_group == group for t in tracked)
    assert tracked[0].uuid.endswith("_parent")
    assert tracked[1].uuid.endswith("_tp")
    assert tracked[2].uuid.endswith("_sl")

    submitted_orders = [c.args[1] for c in fake_client.ib.placeOrder.call_args_list]
    assert submitted_orders[0].transmit is False
    assert submitted_orders[1].transmit is False
    assert submitted_orders[2].transmit is True


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


async def test_cancel_calls_ib(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)

    await manager.cancel(tracked.uuid)
    assert fake_client.ib.cancelOrder.called


async def test_cancel_emits_cancelled_on_status_event(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)

    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    cancel_trade = make_trade(tracked.uuid, perm_id=1, status="Cancelled", filled=0, remaining=1)
    fake_client.ib.orderStatusEvent.fire(cancel_trade)
    await asyncio.sleep(0)

    assert any(isinstance(e, Cancelled) and e.uuid == tracked.uuid for e in seen)
    assert tracked.state == OrderState.CANCELLED


async def test_cancel_unknown_uuid_raises(manager):
    with pytest.raises(KeyError):
        await manager.cancel("no-such-uuid")


async def test_cancel_idempotent_after_terminal(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)
    tracked.state = OrderState.FILLED

    await manager.cancel(tracked.uuid)
    assert not fake_client.ib.cancelOrder.called


# ---------------------------------------------------------------------------
# Reject path
# ---------------------------------------------------------------------------


async def test_inactive_status_emits_rejected(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)

    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    trade = make_trade(tracked.uuid, perm_id=1, status="Inactive", filled=0, remaining=1)
    fake_client.ib.orderStatusEvent.fire(trade)
    await asyncio.sleep(0)

    assert any(isinstance(e, Rejected) for e in seen)
    assert tracked.state == OrderState.REJECTED


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


async def test_position_event_caches_and_publishes(manager, fake_client):
    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    pos = make_position(account="DU1", contract=make_contract(con_id=99), quantity=10, avg_cost=50)
    fake_client.ib.positionEvent.fire(pos)
    await asyncio.sleep(0)

    pos_events = [e for e in seen if isinstance(e, PositionChanged)]
    assert len(pos_events) == 1
    assert pos_events[0].quantity == 10
    assert manager.positions[0].quantity == 10


# ---------------------------------------------------------------------------
# Read-only views
# ---------------------------------------------------------------------------


async def test_open_orders_excludes_terminal(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    t1 = await manager.place(req)

    fake_client.ib.placeOrder.return_value = make_trade("p2", perm_id=2)
    t2 = await manager.place(build_limit(make_contract(), OrderSide.SELL, 1, 105.0))
    t2.state = OrderState.FILLED

    opens = manager.open_orders
    assert t1 in opens
    assert t2 not in opens


# ---------------------------------------------------------------------------
# Reconnect / rehydration
# ---------------------------------------------------------------------------


async def test_start_rehydrates_matched_open_trades(fake_client, tmp_store):
    # Seed local store with a previously-submitted order
    await tmp_store.append(
        RequestSubmitted(
            uuid="u-rehydrate",
            request_kind="limit",
            contract={"conId": 1},
            side="BUY",
            quantity=1.0,
            tif="DAY",
            account=None,
            extra={"limit_price": 100.0},
        )
    )
    open_trade = make_trade("u-rehydrate", perm_id=999, status="Submitted", filled=0, remaining=1)
    fake_client.ib.reqOpenOrdersAsync.return_value = [open_trade]
    fake_client.ib.openTrades.return_value = [open_trade]

    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    report = await mgr.start()

    assert "u-rehydrate" in report.matched
    opens = mgr.open_orders
    assert any(t.uuid == "u-rehydrate" for t in opens)
    await mgr.stop()


async def test_orderstatus_event_routes_after_rehydration(fake_client, tmp_store):
    await tmp_store.append(
        RequestSubmitted(
            uuid="u-rh",
            request_kind="limit",
            contract={"conId": 1},
            side="BUY",
            quantity=1.0,
            tif="DAY",
            account=None,
            extra={"limit_price": 100.0},
        )
    )
    open_trade = make_trade("u-rh", perm_id=11, status="Submitted", filled=0, remaining=1)
    fake_client.ib.reqOpenOrdersAsync.return_value = [open_trade]
    fake_client.ib.openTrades.return_value = [open_trade]

    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    await mgr.start()

    seen: list = []
    mgr.on_event(lambda e: seen.append(e))

    fill_trade = make_trade("u-rh", perm_id=11, status="Filled", filled=1, remaining=0, avg_fill_price=100.5)
    fake_client.ib.orderStatusEvent.fire(fill_trade)
    await asyncio.sleep(0)

    status_events = [e for e in seen if isinstance(e, StatusChanged)]
    assert len(status_events) >= 1
    assert status_events[0].state == OrderState.FILLED.value
    await mgr.stop()
