"""Tests for entry confirmation, confirmed exits and the monitor loop of
:class:`CreditSpreadStrategy`, plus exchange-time DTE and tick rounding.

A deterministic fake order manager stands in for :class:`OrderManager`: each
closing order is resolved according to a scripted outcome (fill, partial fill,
or stay working), and ``wait_for`` simply evaluates the predicate.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from ib_async import Ticker

from ibtws.unofficial.helpers import days_to_expiry
from ibtws.unofficial.order.models import OrderSide, OrderState, TrackedOrder
from ibtws.unofficial.strategies import CreditSpreadError, CreditSpreadParams, CreditSpreadStrategy, SpreadType
from ibtws.unofficial.strategies.utils import _parse_expiry_to_dte, _round_to_tick

from .conftest import make_quote

ET = ZoneInfo("America/New_York")
_TERMINAL = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED}


class FakeOM:
    """Scripted stand-in for OrderManager.

    ``outcomes`` is consumed one entry per ``limit()`` call:
    ``"fill"`` fills completely, a float fills that much and leaves the rest
    working, ``"work"`` leaves the order working.
    """

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.orders: list[tuple[TrackedOrder, OrderSide, float, float]] = []
        self.cancelled: list[str] = []
        self.cancel_confirms = True

    async def limit(self, bag, side, qty, price, **_kw):
        t = TrackedOrder(uuid=f"o{len(self.orders)}", request=None, trade=None, state=OrderState.SUBMITTED)
        t.remaining = qty
        outcome = self.outcomes.pop(0) if self.outcomes else "work"
        if outcome == "fill":
            t.state, t.filled, t.remaining = OrderState.FILLED, qty, 0.0
        elif isinstance(outcome, float):
            t.filled, t.remaining = outcome, qty - outcome
        self.orders.append((t, side, qty, price))
        return t

    async def wait_for(self, predicate, timeout=None):
        return predicate()

    async def cancel(self, uuid):
        self.cancelled.append(uuid)
        if not self.cancel_confirms:
            return
        for t, *_ in self.orders:
            if t.uuid == uuid and t.state not in _TERMINAL:
                t.state = OrderState.CANCELLED
        for t in getattr(self, "entries", []):
            if t.uuid == uuid and t.state not in _TERMINAL:
                t.state = OrderState.CANCELLED


def _plan(strat, fake_fetcher_quotes=None, **overrides):
    params = CreditSpreadParams(
        underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
        spread_type=SpreadType.BULL_PUT,
        wing_width=5.0,
        quantity=overrides.pop("quantity", 2),
        limit_slippage=0.10,
        take_profit_pct=0.5,
        stop_loss_multiplier=2.0,
        **overrides,
    )
    short = make_quote(strike=150.0, con_id=150, delta=-0.30, bid=1.10, ask=1.30)
    long = make_quote(strike=145.0, con_id=145, delta=-0.20, bid=0.70, ask=0.90)
    return strat._materialise_plan(params, "20260619", short, long, "100", 155.0)


@pytest.fixture
def strat(fake_client, fake_fetcher):
    om = FakeOM()
    s = CreditSpreadStrategy(fake_client, om, fetcher=fake_fetcher, exit_max_attempts=3)
    s.om = om
    return s


# ---------------------------------------------------------------------------
# DTE in exchange time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "when, expected_today, expected_tomorrow",
    [
        (dt.datetime(2026, 10, 8, 9, 30, tzinfo=ET), 0, 1),
        (dt.datetime(2026, 10, 8, 12, 1, tzinfo=ET), 0, 1),  # used to flip to -1 / 0 at noon
        (dt.datetime(2026, 10, 8, 15, 59, tzinfo=ET), 0, 1),
        (dt.datetime(2026, 10, 8, 16, 0, tzinfo=ET), -1, 1),  # after the close today has expired
        (dt.datetime(2026, 12, 8, 13, 0, tzinfo=ET), None, None),  # EST: same rules in winter
    ],
)
def test_same_day_expiry_counts_in_exchange_time(when, expected_today, expected_tomorrow):
    today = when.strftime("%Y%m%d")
    tomorrow = (when + dt.timedelta(days=1)).strftime("%Y%m%d")
    now = when.timestamp()
    if expected_today is None:
        assert _parse_expiry_to_dte(today, now=now) == 0
        assert _parse_expiry_to_dte(tomorrow, now=now) == 1
    else:
        assert _parse_expiry_to_dte(today, now=now) == expected_today
        assert _parse_expiry_to_dte(tomorrow, now=now) == expected_tomorrow


def test_dte_ignores_host_timezone():
    # 23:30 in Prague on Oct 8 is 17:30 ET the same day: Oct 9 is 1 DTE.
    prague = dt.datetime(2026, 10, 8, 23, 30, tzinfo=ZoneInfo("Europe/Prague"))
    assert days_to_expiry("20261009", now=prague.timestamp()) == 1


# ---------------------------------------------------------------------------
# Tick rounding
# ---------------------------------------------------------------------------


def test_round_to_tick_directions():
    assert _round_to_tick(0.36, 0.05, mode="down") == pytest.approx(0.35)
    assert _round_to_tick(0.36, 0.05, mode="up") == pytest.approx(0.40)
    assert _round_to_tick(0.20, 0.05, mode="up") == pytest.approx(0.20)  # no float-noise bump
    assert _round_to_tick(0.35, 0.05, mode="down") == pytest.approx(0.35)
    with pytest.raises(ValueError):
        _round_to_tick(1.0, 0.05, mode="sideways")


async def test_place_rounds_credit_down_and_close_rounds_debit_up(strat):
    plan = _plan(strat)
    await strat.place(plan, limit_credit=0.37)
    await strat.close(plan, limit_debit=0.21)
    (_, _, _, entry_px), (_, _, _, close_px) = [o[:4] for o in strat.om.orders]
    assert entry_px == pytest.approx(-0.35)
    assert close_px == pytest.approx(-0.25)


async def test_close_caps_debit_at_width(strat):
    plan = _plan(strat)
    await strat.close(plan, limit_debit=99.0)
    assert strat.om.orders[0][3] == pytest.approx(-plan.width)


# ---------------------------------------------------------------------------
# await_entry
# ---------------------------------------------------------------------------


async def test_await_entry_returns_full_fill(strat):
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.FILLED, filled=2.0)
    assert await strat.await_entry(entry, max_wait=1) == 2.0


async def test_await_entry_timeout_cancels_and_returns_partial(strat):
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.SUBMITTED, filled=1.0, remaining=1.0)
    strat.om.entries = [entry]

    filled = await strat.await_entry(entry, max_wait=0)

    assert strat.om.cancelled == ["e"]
    assert entry.state == OrderState.CANCELLED
    assert filled == 1.0


async def test_await_entry_cancels_inactive(strat):
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.INACTIVE)
    strat.om.entries = [entry]
    assert await strat.await_entry(entry, max_wait=5) == 0.0
    assert strat.om.cancelled == ["e"]


# ---------------------------------------------------------------------------
# close_and_confirm
# ---------------------------------------------------------------------------


async def test_close_and_confirm_fills_first_attempt(strat):
    plan = _plan(strat)
    strat.om.outcomes = ["fill"]

    result = await strat.close_and_confirm(plan, 2, urgent=False, mid_debit=0.20)

    assert result.complete and result.closed_quantity == 2
    assert len(strat.om.orders) == 1
    assert strat.om.orders[0][1] == OrderSide.SELL


async def test_close_and_confirm_chases_remaining_after_partial(strat):
    plan = _plan(strat)
    strat.om.outcomes = [1.0, "fill"]
    strat.current_mid_debit = AsyncMock(return_value=0.30)

    result = await strat.close_and_confirm(plan, 2, urgent=False, mid_debit=0.20)

    assert result.complete
    (first, _, q1, _), (second, _, q2, _) = strat.om.orders
    assert (q1, q2) == (2, 1.0)  # second attempt only for the unfilled remainder
    assert strat.om.cancelled == [first.uuid]


async def test_urgent_close_escalates_to_width(strat):
    plan = _plan(strat)
    strat.om.outcomes = ["work", "work", "fill"]
    strat.current_mid_debit = AsyncMock(return_value=None)

    result = await strat.close_and_confirm(plan, 2, urgent=True, mid_debit=1.0)

    assert result.complete
    prices = [-o[3] for o in strat.om.orders]
    assert prices[-1] == pytest.approx(plan.width)
    assert prices[0] < prices[-1]


async def test_take_profit_close_does_not_escalate(strat):
    plan = _plan(strat)
    strat.om.outcomes = ["work", "work", "work"]
    strat.current_mid_debit = AsyncMock(return_value=0.20)

    result = await strat.close_and_confirm(plan, 2, urgent=False, mid_debit=0.20)

    assert not result.complete
    assert max(-o[3] for o in strat.om.orders) < 0.5


async def test_close_stops_when_cancel_unconfirmed(strat):
    plan = _plan(strat)
    strat.om.outcomes = ["work", "fill"]
    strat.om.cancel_confirms = False

    result = await strat.close_and_confirm(plan, 2, urgent=True, mid_debit=1.0)

    assert len(strat.om.orders) == 1  # never risks a second, overlapping close
    assert not result.complete


# ---------------------------------------------------------------------------
# monitor_and_exit
# ---------------------------------------------------------------------------


async def test_monitor_stop_loss_closes_filled_quantity(strat):
    plan = _plan(strat, quantity=3)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.CANCELLED, filled=2.0)
    strat.om.outcomes = ["fill"]
    strat.current_mid_debit = AsyncMock(side_effect=[0.50, plan.stop_loss_debit + 0.05])

    closed = await strat.monitor_and_exit(plan, entry, poll_interval=0)

    assert closed is not None
    assert strat.om.orders[0][2] == 2.0  # partial entry → close only what filled


async def test_monitor_returns_none_when_entry_never_fills(strat):
    plan = _plan(strat)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.CANCELLED)
    assert await strat.monitor_and_exit(plan, entry, poll_interval=0) is None
    assert strat.om.orders == []


async def test_monitor_closes_on_timeout(strat):
    plan = _plan(strat)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.FILLED, filled=2.0)
    strat.om.outcomes = ["fill"]
    strat.current_mid_debit = AsyncMock(return_value=0.40)

    closed = await strat.monitor_and_exit(plan, entry, poll_interval=0.01, max_wait=0.03)

    assert closed is not None
    assert strat.om.orders[0][2] == 2.0


async def test_monitor_can_leave_position_on_timeout(strat):
    plan = _plan(strat)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.FILLED, filled=2.0)
    strat.current_mid_debit = AsyncMock(return_value=0.40)

    closed = await strat.monitor_and_exit(plan, entry, poll_interval=0.01, max_wait=0.03, close_on_timeout=False)

    assert closed is None
    assert strat.om.orders == []


async def test_monitor_closes_after_repeated_quote_loss(strat, fake_client):
    plan = _plan(strat)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.FILLED, filled=2.0)
    strat.om.outcomes = ["fill"]
    strat.current_mid_debit = AsyncMock(return_value=None)
    fake_client.ib.isConnected.return_value = True

    closed = await strat.monitor_and_exit(plan, entry, poll_interval=0, max_quote_failures=3)

    assert closed is not None
    assert strat.current_mid_debit.await_count >= 3


async def test_monitor_take_profit_partial_keeps_monitoring(strat):
    plan = _plan(strat)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.FILLED, filled=2.0)
    # TP attempt 1 fills 1, attempts 2-3 stay working; next TP round fills the rest.
    strat.om.outcomes = [1.0, "work", "work", "fill"]
    strat.current_mid_debit = AsyncMock(return_value=0.10)

    closed = await strat.monitor_and_exit(plan, entry, poll_interval=0)

    assert closed is not None
    assert sum(1 for o in strat.om.orders if o[0].state == OrderState.FILLED) == 1
    assert strat.om.orders[-1][2] == 1.0


async def test_monitor_subscribes_and_releases_streams(strat, fake_client):
    plan = _plan(strat)
    entry = TrackedOrder(uuid="e", request=None, trade=None, state=OrderState.FILLED, filled=2.0)
    strat.om.outcomes = ["fill"]
    strat.current_mid_debit = AsyncMock(return_value=plan.stop_loss_debit)

    await strat.monitor_and_exit(plan, entry, poll_interval=0)

    assert fake_client.ib.reqMktData.call_count == 2
    assert fake_client.ib.cancelMktData.call_count == 2
    assert strat._streams == {}


# ---------------------------------------------------------------------------
# Streaming quotes
# ---------------------------------------------------------------------------


def _ticker(contract, bid, ask, *, age=0.0, data_type=1):
    t = Ticker()
    t.contract = contract
    t.bid, t.ask = bid, ask
    t.marketDataType = data_type
    t.time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age)
    return t


async def test_streaming_mid_debit_and_staleness(strat, fake_client):
    plan = _plan(strat)
    short_c, long_c = plan.short_leg.quote.contract, plan.long_leg.quote.contract
    fresh = {150: _ticker(short_c, 1.10, 1.30), 145: _ticker(long_c, 0.70, 0.90)}
    fake_client.ib.reqMktData.side_effect = lambda c, *a, **k: fresh[c.conId]

    strat.watch(plan)
    assert await strat.current_mid_debit(plan) == pytest.approx(0.40)
    fake_client.ib.reqTickersAsync.assert_not_called()

    fresh[150].time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=3600)
    assert await strat.current_mid_debit(plan) is None
    strat.unwatch(plan)


async def test_streams_are_reference_counted_and_resubscribed(strat, fake_client):
    plan = _plan(strat)
    strat.watch(plan)
    strat.watch(plan)
    assert fake_client.ib.reqMktData.call_count == 2

    strat._resubscribe()
    assert fake_client.ib.reqMktData.call_count == 4

    strat.unwatch(plan)
    fake_client.ib.cancelMktData.assert_not_called()
    strat.unwatch(plan)
    assert fake_client.ib.cancelMktData.call_count == 2


async def test_require_live_quotes_rejects_frozen_data(strat, fake_client):
    plan = _plan(strat, require_live_quotes=True)
    short_c, long_c = plan.short_leg.quote.contract, plan.long_leg.quote.contract
    frozen = {150: _ticker(short_c, 1.10, 1.30, data_type=2), 145: _ticker(long_c, 0.70, 0.90, data_type=2)}
    fake_client.ib.reqTickersAsync = AsyncMock(side_effect=lambda c, **_kw: [frozen[c.conId]])
    assert await strat.current_mid_debit(plan) is None


async def test_build_plan_require_live_quotes(fake_client, fake_fetcher, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, FakeOM(), fetcher=fake_fetcher)
    chain_def = SimpleNamespace(
        underlying_conId=1,
        underlying_symbol="AAPL",
        trading_class="AAPL",
        multiplier="100",
        exchange="SMART",
        expirations=("20260619",),
        strikes=(145.0, 150.0),
    )
    frozen = [
        make_quote(strike=145.0, con_id=145, delta=-0.20, bid=0.70, ask=0.90),
        make_quote(strike=150.0, con_id=150, delta=-0.30, bid=1.10, ask=1.30),
    ]
    for q in frozen:
        q.market_data_type = 2
    strat._fetcher.fetch_chain_definition = AsyncMock(return_value=chain_def)
    strat._fetcher.fetch_snapshot = AsyncMock(return_value=frozen)
    monkeypatch.setattr("ibtws.unofficial.strategies.utils._parse_expiry_to_dte", lambda exp, now=None: 30)

    params = CreditSpreadParams(
        underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
        spread_type=SpreadType.BULL_PUT,
        require_live_quotes=True,
    )
    with pytest.raises(CreditSpreadError, match="no live quotes"):
        await strat.build_plan(params)
