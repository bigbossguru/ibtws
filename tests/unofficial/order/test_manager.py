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


async def test_paper_guard_refuses_empty_managed_accounts(fake_client, tmp_store):
    fake_client.ib.managedAccounts.return_value = []
    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    with pytest.raises(RuntimeError, match="managedAccounts"):
        await mgr.start()


async def test_place_refuses_unknown_per_request_account(manager, fake_client):
    contract = make_contract()
    req = build_limit(contract, OrderSide.BUY, 1, 100.0, account="U999999")
    with pytest.raises(RuntimeError, match="not in the session"):
        await manager.place(req)
    fake_client.ib.placeOrder.assert_not_called()


async def test_place_refuses_live_per_request_account_without_override(fake_client, tmp_store):
    # Paper-primary session that nevertheless has visibility into a live
    # sub-account (e.g. FA setup). The per-request account guard must refuse.
    fake_client.ib.managedAccounts.return_value = ["DU123", "U999999"]
    mgr = OrderManager(fake_client, tmp_store, pace_per_sec=0)
    await mgr.start()
    try:
        req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0, account="U999999")
        with pytest.raises(RuntimeError, match="live account"):
            await mgr.place(req)
        fake_client.ib.placeOrder.assert_not_called()
    finally:
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


async def test_exec_details_dedup_by_exec_id(manager, fake_client):
    """IB replays execDetails on reconnect — we must only publish each fill once."""
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=7)
    req = build_limit(make_contract(), OrderSide.BUY, 1, 100.0)
    tracked = await manager.place(req)

    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    trade = make_trade(tracked.uuid, perm_id=7)
    fill = make_fill("E1", price=100.5, shares=1)
    fake_client.ib.execDetailsEvent.fire(trade, fill)
    fake_client.ib.execDetailsEvent.fire(trade, fill)  # replay
    await asyncio.sleep(0)

    fills = [e for e in seen if isinstance(e, Filled)]
    assert len(fills) == 1
    assert fills[0].exec_id == "E1"


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
    # IB bracket wiring: children point to parent's orderId, TP/SL share OCA group
    parent_id = submitted_orders[0].orderId
    assert parent_id  # non-zero, allocated via getReqId
    assert submitted_orders[1].parentId == parent_id
    assert submitted_orders[2].parentId == parent_id
    assert submitted_orders[1].ocaGroup == submitted_orders[2].ocaGroup
    assert submitted_orders[1].ocaGroup  # non-empty
    assert submitted_orders[1].ocaType == 1 and submitted_orders[2].ocaType == 1


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


async def test_cancel_all_skips_terminal(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1)
    t1 = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    fake_client.ib.placeOrder.return_value = make_trade("p2", perm_id=2)
    t2 = await manager.place(build_limit(make_contract(), OrderSide.SELL, 1, 105.0))
    t2.state = OrderState.FILLED

    cancelled = await manager.cancel_all()
    assert cancelled == [t1.uuid]
    assert fake_client.ib.cancelOrder.call_count == 1


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
    # Inactive is not terminal: IB may revive the order, so it must stay
    # cancellable and visible as open.
    assert tracked.state == OrderState.INACTIVE
    assert tracked in manager.open_orders
    await manager.cancel(tracked.uuid)
    assert fake_client.ib.cancelOrder.called


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
# Position close helpers
# ---------------------------------------------------------------------------


async def test_close_position_flattens_long(manager, fake_client):
    contract = make_contract(con_id=42)
    pos = make_position(account="DU123", contract=contract, quantity=3, avg_cost=100)
    fake_client.ib.reqPositionsAsync.return_value = [pos]

    fake_client.ib.qualifyContractsAsync.return_value = [contract]
    fake_client.ib.placeOrder.return_value = make_trade("close", perm_id=99)

    closed = await manager.close_position(42)

    assert closed is not None
    submitted = fake_client.ib.placeOrder.call_args.args[1]
    assert submitted.action == "SELL"
    assert submitted.totalQuantity == 3


async def test_close_position_flattens_short(manager, fake_client):
    contract = make_contract(con_id=7)
    pos = make_position(contract=contract, quantity=-2, avg_cost=100)
    fake_client.ib.reqPositionsAsync.return_value = [pos]

    fake_client.ib.qualifyContractsAsync.return_value = [contract]
    fake_client.ib.placeOrder.return_value = make_trade("close", perm_id=1)

    await manager.close_position(7)
    submitted = fake_client.ib.placeOrder.call_args.args[1]
    assert submitted.action == "BUY"
    assert submitted.totalQuantity == 2


async def test_close_position_unknown_conid_returns_none(manager):
    assert await manager.close_position(9999) is None


async def test_close_position_zero_quantity_returns_none(manager, fake_client):
    pos = make_position(contract=make_contract(con_id=5), quantity=0, avg_cost=0)
    fake_client.ib.reqPositionsAsync.return_value = [pos]
    assert await manager.close_position(5) is None


async def test_close_position_cancels_working_orders_first(manager, fake_client):
    contract = make_contract(con_id=11)
    # Place a working limit on the same contract.
    fake_client.ib.placeOrder.return_value = make_trade("working", perm_id=1)
    working = await manager.limit(contract, OrderSide.BUY, 1, 100.0)

    pos = make_position(contract=contract, quantity=1, avg_cost=100)
    fake_client.ib.reqPositionsAsync.return_value = [pos]

    # IB confirms the cancel asynchronously via orderStatus.
    def confirm_cancel(order):
        trade = make_trade(working.uuid, perm_id=1, status="Cancelled", remaining=1)
        asyncio.get_running_loop().call_soon(fake_client.ib.orderStatusEvent.fire, trade)

    fake_client.ib.cancelOrder.side_effect = confirm_cancel
    fake_client.ib.qualifyContractsAsync.return_value = [contract]
    fake_client.ib.placeOrder.return_value = make_trade("close", perm_id=2)

    closed = await manager.close_position(11)

    assert fake_client.ib.cancelOrder.called
    cancelled_order = fake_client.ib.cancelOrder.call_args.args[0]
    assert cancelled_order is working.trade.order
    assert closed is not None
    assert working.state == OrderState.CANCELLED


async def test_close_position_refuses_when_cancel_not_confirmed(manager, fake_client):
    contract = make_contract(con_id=12)
    fake_client.ib.placeOrder.return_value = make_trade("working", perm_id=1)
    await manager.limit(contract, OrderSide.SELL, 1, 105.0)
    fake_client.ib.reqPositionsAsync.return_value = [make_position(contract=contract, quantity=1)]
    fake_client.ib.placeOrder.reset_mock()

    with pytest.raises(RuntimeError, match="not confirmed"):
        await manager.close_position(12, cancel_timeout=0.05)

    fake_client.ib.placeOrder.assert_not_called()


async def test_close_position_cancels_rehydrated_and_combo_orders(manager, fake_client):
    from types import SimpleNamespace

    from ibtws.unofficial.order.models import TrackedOrder

    # An order rehydrated after restart has request=None; a combo order
    # carries the leg only inside comboLegs. Both must be cancelled.
    rehydrated_trade = make_trade("rehydrated", perm_id=5)
    rehydrated_trade.contract = make_contract(con_id=33)
    combo_trade = make_trade("combo", perm_id=6)
    combo_trade.contract = SimpleNamespace(conId=0, secType="BAG", comboLegs=[SimpleNamespace(conId=33)])
    unrelated_trade = make_trade("other", perm_id=7)
    unrelated_trade.contract = make_contract(con_id=99)
    for uuid, trade in (("rehydrated", rehydrated_trade), ("combo", combo_trade), ("other", unrelated_trade)):
        manager._tracked[uuid] = TrackedOrder(uuid=uuid, request=None, trade=trade, state=OrderState.SUBMITTED)

    def confirm_cancel(order):
        trade = make_trade(order.orderRef, perm_id=order.permId, status="Cancelled")
        asyncio.get_running_loop().call_soon(fake_client.ib.orderStatusEvent.fire, trade)

    fake_client.ib.cancelOrder.side_effect = confirm_cancel
    fake_client.ib.reqPositionsAsync.return_value = [make_position(contract=make_contract(con_id=33), quantity=2)]
    fake_client.ib.placeOrder.return_value = make_trade("close", perm_id=8)

    await manager.close_position(33)

    cancelled_refs = {c.args[0].orderRef for c in fake_client.ib.cancelOrder.call_args_list}
    assert cancelled_refs == {"rehydrated", "combo"}


async def test_close_position_limit_requires_price(manager, fake_client):
    contract = make_contract(con_id=22)
    fake_client.ib.reqPositionsAsync.return_value = [make_position(contract=contract, quantity=1, avg_cost=100)]
    fake_client.ib.qualifyContractsAsync.return_value = [contract]

    with pytest.raises(ValueError, match="limit_price"):
        await manager.close_position(22, kind="limit")


async def test_close_all_positions(manager, fake_client):
    c1 = make_contract(con_id=1)
    c2 = make_contract(con_id=2)
    fake_client.ib.reqPositionsAsync.return_value = [
        make_position(contract=c1, quantity=1),
        make_position(contract=c2, quantity=-1),
        make_position(contract=make_contract(con_id=3), quantity=0),
    ]

    fake_client.ib.qualifyContractsAsync.side_effect = lambda c: [c]
    fake_client.ib.placeOrder.side_effect = [
        make_trade("c1", perm_id=11),
        make_trade("c2", perm_id=22),
    ]

    closed = await manager.close_all_positions()
    assert len(closed) == 2  # zero-quantity skipped


async def test_close_all_positions_respects_conid_filter(manager, fake_client):
    fake_client.ib.reqPositionsAsync.return_value = [
        make_position(contract=make_contract(con_id=1), quantity=1),
        make_position(contract=make_contract(con_id=2), quantity=-1),
    ]
    fake_client.ib.placeOrder.return_value = make_trade("c2", perm_id=22)

    closed = await manager.close_all_positions(con_ids=[2])

    assert len(closed) == 1
    assert fake_client.ib.placeOrder.call_args.args[0].conId == 2


async def test_close_position_uses_fresh_quantity(manager, fake_client):
    contract = make_contract(con_id=44)
    # Cached event says 5, IB now reports 2 (a partial exit already filled).
    fake_client.ib.positionEvent.fire(make_position(contract=contract, quantity=5))
    await asyncio.sleep(0)
    fake_client.ib.reqPositionsAsync.return_value = [make_position(contract=contract, quantity=2)]
    fake_client.ib.placeOrder.return_value = make_trade("close", perm_id=1)

    await manager.close_position(44)

    assert fake_client.ib.placeOrder.call_args.args[1].totalQuantity == 2


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


# ---------------------------------------------------------------------------
# current_pnl
# ---------------------------------------------------------------------------


async def test_current_pnl_computes_pnl(manager, fake_client):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    contract = make_contract(con_id=42)
    pos = make_position(account="DU123", contract=contract, quantity=10, avg_cost=150.0)
    fake_client.ib.positionEvent.fire(pos)
    await asyncio.sleep(0)

    ticker = SimpleNamespace(contract=SimpleNamespace(conId=42), bid=160.0, ask=160.40, last=160.10, close=159.0)
    fake_client.ib.reqTickersAsync = AsyncMock(return_value=[ticker])

    pnls = await manager.current_pnl()

    assert len(pnls) == 1
    p = pnls[0]
    assert p.quantity == 10.0
    assert p.avg_cost == 150.0
    assert p.multiplier == 1.0
    # mid = 160.20 → market_value = 1602; cost basis = 1500; unrealized = 102.
    assert p.market_price == pytest.approx(160.20)
    assert p.market_value == pytest.approx(1602.0)
    assert p.cost_basis == pytest.approx(1500.0)
    assert p.unrealized_pnl == pytest.approx(102.0)


async def test_current_pnl_skips_flat_positions(manager, fake_client):
    from unittest.mock import AsyncMock

    pos = make_position(contract=make_contract(con_id=11), quantity=0, avg_cost=0.0)
    fake_client.ib.positionEvent.fire(pos)
    await asyncio.sleep(0)
    fake_client.ib.reqTickersAsync = AsyncMock(return_value=[])

    pnls = await manager.current_pnl()
    assert pnls == []
    fake_client.ib.reqTickersAsync.assert_not_called()


async def test_current_pnl_marks_unknown_when_no_quote(manager, fake_client):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    contract = make_contract(con_id=77)
    fake_client.ib.positionEvent.fire(make_position(account="DU123", contract=contract, quantity=5, avg_cost=100.0))
    await asyncio.sleep(0)

    # Ticker with no bid/ask/last/close — all None → price unavailable.
    blank = SimpleNamespace(contract=SimpleNamespace(conId=77), bid=None, ask=None, last=None, close=None)
    fake_client.ib.reqTickersAsync = AsyncMock(return_value=[blank])

    pnls = await manager.current_pnl()
    assert len(pnls) == 1
    assert pnls[0].market_price is None
    assert pnls[0].market_value is None
    assert pnls[0].unrealized_pnl is None
    assert pnls[0].cost_basis == pytest.approx(500.0)


async def test_current_pnl_filters_by_conid(manager, fake_client):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    fake_client.ib.positionEvent.fire(
        make_position(account="DU123", contract=make_contract(con_id=1), quantity=1, avg_cost=10)
    )
    fake_client.ib.positionEvent.fire(
        make_position(account="DU123", contract=make_contract(con_id=2), quantity=1, avg_cost=20)
    )
    await asyncio.sleep(0)

    ticker = SimpleNamespace(contract=SimpleNamespace(conId=2), bid=25, ask=25.20, last=25, close=24)
    fake_client.ib.reqTickersAsync = AsyncMock(return_value=[ticker])

    pnls = await manager.current_pnl(con_ids=[2])
    assert len(pnls) == 1
    assert pnls[0].contract["conId"] == 2


# ---------------------------------------------------------------------------
# Reliability: failures, readonly, wait_for, resync, start rollback
# ---------------------------------------------------------------------------


async def test_place_failure_records_rejection(manager, fake_client, tmp_store):
    fake_client.ib.placeOrder.side_effect = ConnectionError("socket closed")
    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    with pytest.raises(ConnectionError):
        await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))

    events = list(tmp_store.replay())
    assert [type(e) for e in events] == [RequestSubmitted, Rejected]
    assert events[1].uuid == events[0].uuid
    assert "socket closed" in events[1].reason
    assert any(isinstance(e, Rejected) for e in seen)


async def test_place_failure_is_not_reported_local_only_on_restart(fake_client, tmp_store):
    mgr = OrderManager(fake_client, tmp_store)
    await mgr.start()
    fake_client.ib.placeOrder.side_effect = ConnectionError("socket closed")
    with pytest.raises(ConnectionError):
        await mgr.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    await mgr.stop()

    report = await OrderManager(fake_client, tmp_store).start()
    assert report.local_only == []


async def test_place_bracket_persists_every_member(manager, fake_client, tmp_store):
    fake_client.ib.placeOrder.side_effect = [make_trade("p"), make_trade("tp"), make_trade("sl")]
    req = build_bracket(make_contract(), OrderSide.BUY, 1, take_profit_price=110, stop_loss_price=90)

    tracked = await manager.place_bracket(req)

    submitted = [e for e in tmp_store.replay() if isinstance(e, RequestSubmitted)]
    assert [e.uuid for e in submitted] == [t.uuid for t in tracked]
    assert [e.extra["leg"] for e in submitted] == ["parent", "tp", "sl"]


async def test_place_bracket_failure_cancels_placed_members(manager, fake_client, tmp_store):
    fake_client.ib.placeOrder.side_effect = [make_trade("p"), ConnectionError("socket closed")]
    req = build_bracket(make_contract(), OrderSide.BUY, 1, take_profit_price=110, stop_loss_price=90)

    with pytest.raises(ConnectionError):
        await manager.place_bracket(req)

    assert fake_client.ib.cancelOrder.call_count == 1
    rejected = [e for e in tmp_store.replay() if isinstance(e, Rejected)]
    assert len(rejected) == 3


async def test_readonly_config_blocks_orders(fake_client, tmp_store):
    from types import SimpleNamespace

    fake_client.config = SimpleNamespace(readonly=True)
    mgr = OrderManager(fake_client, tmp_store)
    await mgr.start()
    try:
        with pytest.raises(RuntimeError, match="readonly"):
            await mgr.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
        fake_client.ib.placeOrder.assert_not_called()
    finally:
        await mgr.stop()


async def test_wait_for_wakes_on_status_event(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=3)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))

    filled = make_trade(tracked.uuid, perm_id=3, status="Filled", filled=1, remaining=0)
    asyncio.get_running_loop().call_later(0.01, fake_client.ib.orderStatusEvent.fire, filled)

    assert await manager.wait_for(lambda: tracked.state == OrderState.FILLED, timeout=2.0)


async def test_wait_for_times_out(manager):
    assert await manager.wait_for(lambda: False, timeout=0.02) is False


async def test_start_failure_unbinds_events(fake_client, tmp_path):
    from ibtws.unofficial.order import JsonStore

    path = tmp_path / "corrupt.jsonl"
    path.write_text("not json\n{}\n")
    mgr = OrderManager(fake_client, JsonStore(path, fsync=False))

    with pytest.raises(ValueError):
        await mgr.start()

    assert fake_client.ib.orderStatusEvent.handlers == []
    assert fake_client.ib.execDetailsEvent.handlers == []
    assert fake_client.ib.positionEvent.handlers == []


async def test_start_survives_torn_last_line(fake_client, tmp_store):
    await tmp_store.append(Cancelled(uuid="u-old", perm_id=1))
    tmp_store.close()
    with tmp_store.path.open("a", encoding="utf-8") as f:
        f.write('{"kind":"status_changed","uuid":"u-tor')

    mgr = OrderManager(fake_client, tmp_store)
    await mgr.start()
    await mgr.stop()


async def test_resync_applies_status_missed_while_disconnected(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=9)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    fake_client.ib.trades.return_value = [make_trade(tracked.uuid, perm_id=9, status="Filled", filled=1)]
    await manager.resync()

    assert tracked.state == OrderState.FILLED
    assert any(isinstance(e, StatusChanged) and e.state == OrderState.FILLED.value for e in seen)


async def test_start_registers_reconnect_listener(fake_client, tmp_store):
    from unittest.mock import MagicMock

    fake_client.add_reconnect_listener = MagicMock()
    fake_client.remove_reconnect_listener = MagicMock()
    mgr = OrderManager(fake_client, tmp_store)
    await mgr.start()
    fake_client.add_reconnect_listener.assert_called_once_with(mgr.resync)
    await mgr.stop()
    fake_client.remove_reconnect_listener.assert_called_once_with(mgr.resync)


async def test_fill_event_carries_contract(manager, fake_client):
    from types import SimpleNamespace

    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=7)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    fill = make_fill("E9", price=1.0, shares=1)
    fill.contract = SimpleNamespace(conId=555, secType="OPT")
    fake_client.ib.execDetailsEvent.fire(make_trade(tracked.uuid, perm_id=7), fill)

    [ev] = [e for e in seen if isinstance(e, Filled)]
    assert (ev.con_id, ev.sec_type) == (555, "OPT")


async def test_events_stream_fans_out_to_every_subscriber(manager, fake_client):
    a = manager.events()
    b = manager.events()
    first_a = asyncio.ensure_future(a.__anext__())
    first_b = asyncio.ensure_future(b.__anext__())
    await asyncio.sleep(0)

    fake_client.ib.positionEvent.fire(make_position(contract=make_contract(con_id=1), quantity=1))

    ev_a, ev_b = await asyncio.wait_for(asyncio.gather(first_a, first_b), timeout=1.0)
    assert isinstance(ev_a, PositionChanged) and isinstance(ev_b, PositionChanged)


async def test_ib_rejection_error_makes_inactive_order_terminal(manager, fake_client, tmp_store):
    # Live TWS sequence for e.g. a MiFID/KID rejection: status Inactive, then error 201.
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1, order_id=41)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))

    fake_client.ib.orderStatusEvent.fire(make_trade(tracked.uuid, perm_id=1, status="Inactive", order_id=41))
    assert tracked.state == OrderState.INACTIVE
    fake_client.ib.errorEvent.fire(41, 201, "Order rejected - reason:No Trading Permission", None)

    assert tracked.state == OrderState.REJECTED
    assert tracked not in manager.open_orders
    # A late Inactive status must not reopen it, and cancel_all leaves it alone.
    fake_client.ib.orderStatusEvent.fire(make_trade(tracked.uuid, perm_id=1, status="Inactive", order_id=41))
    assert tracked.state == OrderState.REJECTED
    assert await manager.cancel_all() == []

    await manager.stop()
    report = await OrderManager(fake_client, tmp_store).start()
    assert tracked.uuid not in report.local_only


async def test_rejection_error_before_status_wins(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1, order_id=42)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))

    fake_client.ib.errorEvent.fire(42, 201, "Order rejected", None)
    fake_client.ib.orderStatusEvent.fire(make_trade(tracked.uuid, perm_id=1, status="Inactive", order_id=42))

    assert tracked.state == OrderState.REJECTED


async def test_reject_error_ignored_for_working_order(manager, fake_client):
    # e.g. a rejected modification: the original order is still live.
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1, order_id=43)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    fake_client.ib.orderStatusEvent.fire(make_trade(tracked.uuid, perm_id=1, status="Submitted", order_id=43))

    fake_client.ib.errorEvent.fire(43, 201, "Order rejected", None)

    assert tracked.state == OrderState.SUBMITTED


async def test_cancel_of_unknown_inactive_order_closes_it(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=1, order_id=44)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    fake_client.ib.orderStatusEvent.fire(make_trade(tracked.uuid, perm_id=1, status="Inactive", order_id=44))

    await manager.cancel(tracked.uuid)
    fake_client.ib.errorEvent.fire(44, 10147, "OrderId 44 that needs to be cancelled is not found.", None)

    assert await manager.wait_for(lambda: tracked.state in (OrderState.REJECTED,), timeout=0.1)


async def test_resync_replays_fills_missed_while_disconnected(manager, fake_client):
    fake_client.ib.placeOrder.return_value = make_trade("p", perm_id=9)
    tracked = await manager.place(build_limit(make_contract(), OrderSide.BUY, 1, 100.0))
    seen: list = []
    manager.on_event(lambda e: seen.append(e))

    # After a reconnect ib_async rebuilds the trade from the startup fetch:
    # status Filled and the execution in trade.fills, but no execDetailsEvent.
    filled_trade = make_trade(tracked.uuid, perm_id=9, status="Filled", filled=1)
    filled_trade.fills = [make_fill("E-missed", price=100.0, shares=1)]
    fake_client.ib.trades.return_value = [filled_trade]
    await manager.resync()
    await manager.resync()  # a second resync must not duplicate the fill

    fills = [e for e in seen if isinstance(e, Filled)]
    assert [f.exec_id for f in fills] == ["E-missed"]
    assert tracked.state == OrderState.FILLED


async def test_current_pnl_survives_one_unsubscribed_contract(manager, fake_client):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    # Live TWS: error 10090 on one contract used to blank every position's price.
    fake_client.ib.positionEvent.fire(make_position(contract=make_contract(con_id=1), quantity=1, avg_cost=10))
    fake_client.ib.positionEvent.fire(make_position(contract=make_contract(con_id=2), quantity=1, avg_cost=20))
    await asyncio.sleep(0)

    async def tickers(contract, **_kw):
        if contract.conId == 1:
            raise RuntimeError("API error: 10090: Part of requested market data is not subscribed")
        return [SimpleNamespace(contract=contract, bid=25.0, ask=25.2, last=25.0, close=24.0)]

    fake_client.ib.reqTickersAsync = AsyncMock(side_effect=tickers)

    by_conid = {p.contract["conId"]: p for p in await manager.current_pnl()}

    assert by_conid[1].market_price is None
    assert by_conid[2].market_price == pytest.approx(25.1)
