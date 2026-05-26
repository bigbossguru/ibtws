"""Tests for ``ibtws.unofficial.strategies.credit_spread``.

Coverage:

* pure selectors (`select_expiry`, `select_short_leg`, `select_long_leg`)
* `_parse_expiry_to_dte` timezone correctness
* `CreditSpreadParams` validation
* `_materialise_plan` math (credit, max loss, breakeven, TP/SL)
* BAG-aware ``validate_request`` in the order layer
* End-to-end `build_plan` / `place` / `close` via mocked
  `OptionChainFetcher` + `OrderManager`.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ib_async import Bag, ComboLeg

from ibtws.unofficial.order.models import LimitRequest, OrderSide, OrderState, TimeInForce, TrackedOrder
from ibtws.unofficial.order.utils import validate_request
from ibtws.unofficial.strategies import (
    CreditSpreadError,
    CreditSpreadParams,
    CreditSpreadStrategy,
    SpreadType,
    select_expiry,
    select_long_leg,
    select_short_leg,
)
from ibtws.unofficial.strategies.utils import (
    _parse_expiry_to_dte,
    _quote_mid,
    _round_to_tick,
)

from .conftest import make_quote, make_ticker_for


# ---------------------------------------------------------------------------
# _parse_expiry_to_dte
# ---------------------------------------------------------------------------


def test_parse_expiry_to_dte_basic():
    # Reference at noon UTC on 2026-05-19. Expiry 2026-06-18 close (20:00 UTC).
    ref = dt.datetime(2026, 5, 19, 12, tzinfo=dt.timezone.utc).timestamp()
    assert _parse_expiry_to_dte("20260618", now=ref) == 30


def test_parse_expiry_to_dte_monthly_format():
    ref = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc).timestamp()
    # YYYYMM expands to YYYYMM15.
    assert _parse_expiry_to_dte("202606", now=ref) == _parse_expiry_to_dte("20260615", now=ref)


def test_parse_expiry_to_dte_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_expiry_to_dte("not-a-date")


# ---------------------------------------------------------------------------
# select_expiry
# ---------------------------------------------------------------------------


def test_select_expiry_picks_closest_within_tolerance():
    now = dt.datetime(2026, 5, 19, 12, tzinfo=dt.timezone.utc).timestamp()
    chosen = select_expiry(
        ["20260522", "20260612", "20260619", "20260717"],
        target_dte=30,
        dte_tolerance=14,
        now=now,
    )
    assert chosen == "20260619"  # exactly 31d away — closer than 12 Jun (24d) and 17 Jul (59d)


def test_select_expiry_filters_expired():
    now = dt.datetime(2026, 5, 19, 12, tzinfo=dt.timezone.utc).timestamp()
    chosen = select_expiry(["20260101", "20260619"], target_dte=30, dte_tolerance=30, now=now)
    assert chosen == "20260619"


def test_select_expiry_raises_when_none_in_window():
    now = dt.datetime(2026, 5, 19, 12, tzinfo=dt.timezone.utc).timestamp()
    with pytest.raises(CreditSpreadError, match="No expiry within"):
        select_expiry(["20271231"], target_dte=30, dte_tolerance=14, now=now)


# ---------------------------------------------------------------------------
# select_short_leg / select_long_leg
# ---------------------------------------------------------------------------


def _put_chain():
    """A modest put chain: strikes 140..160 step 5, all with conIds + greeks."""
    deltas = {140.0: -0.10, 145.0: -0.20, 150.0: -0.30, 155.0: -0.45, 160.0: -0.60}
    return [
        make_quote(
            strike=s, right="P", con_id=int(s), delta=d, bid=max(0.05, abs(d) * 4), ask=max(0.10, abs(d) * 4 + 0.20)
        )
        for s, d in deltas.items()
    ]


def test_select_short_leg_picks_closest_delta():
    chain = _put_chain()
    chosen = select_short_leg(
        chain,
        target_short_delta=0.30,
        max_short_delta=0.50,
        min_open_interest=0,
        min_volume=0,
    )
    assert chosen.contract.strike == 150.0


def test_select_short_leg_respects_max_delta():
    chain = _put_chain()
    # Target 0.50, but cap 0.40 — must skip strike 155 (Δ -0.45) and 160 (-0.60).
    chosen = select_short_leg(
        chain,
        target_short_delta=0.50,
        max_short_delta=0.40,
        min_open_interest=0,
        min_volume=0,
    )
    assert chosen.contract.strike == 150.0  # 0.30 is the closest within the cap


def test_select_short_leg_rejects_when_all_filtered():
    chain = _put_chain()
    with pytest.raises(CreditSpreadError, match="exceed max_short_delta"):
        select_short_leg(chain, target_short_delta=0.50, max_short_delta=0.05, min_open_interest=0, min_volume=0)


def test_select_short_leg_requires_delta_and_conid():
    chain = [make_quote(strike=150.0, delta=None, con_id=1), make_quote(strike=155.0, delta=-0.3, con_id=0)]
    with pytest.raises(CreditSpreadError, match="No tradeable quotes"):
        select_short_leg(chain, target_short_delta=0.3, max_short_delta=None, min_open_interest=0, min_volume=0)


def test_select_long_leg_snaps_to_width_bull_put():
    chain = _put_chain()
    short = next(q for q in chain if q.contract.strike == 150.0)
    long_leg = select_long_leg(
        chain,
        short=short,
        wing_width=5.0,
        spread_type=SpreadType.BULL_PUT,
        min_open_interest=0,
        min_volume=0,
    )
    assert long_leg.contract.strike == 145.0  # 5 below short


def test_select_long_leg_rejects_when_chain_too_narrow():
    short = make_quote(strike=150.0, right="P", con_id=10, delta=-0.30)
    long_only = make_quote(strike=149.0, right="P", con_id=11, delta=-0.25)  # 1 wide vs requested 10
    with pytest.raises(CreditSpreadError, match="too narrow"):
        select_long_leg(
            [short, long_only],
            short=short,
            wing_width=10.0,
            spread_type=SpreadType.BULL_PUT,
            min_open_interest=0,
            min_volume=0,
        )


def test_select_long_leg_no_protective_side():
    short = make_quote(strike=140.0, right="P", con_id=10, delta=-0.10)
    chain = [short, make_quote(strike=145.0, right="P", con_id=11, delta=-0.20)]
    with pytest.raises(CreditSpreadError, match="No protective leg"):
        select_long_leg(
            chain,
            short=short,
            wing_width=5.0,
            spread_type=SpreadType.BULL_PUT,  # need a strike BELOW 140
            min_open_interest=0,
            min_volume=0,
        )


def test_select_long_leg_bear_call_picks_higher_strike():
    chain = [
        make_quote(strike=150.0, right="C", con_id=1, delta=0.50),
        make_quote(strike=155.0, right="C", con_id=2, delta=0.30),
        make_quote(strike=160.0, right="C", con_id=3, delta=0.20),
    ]
    short = chain[1]  # 155
    long_leg = select_long_leg(
        chain,
        short=short,
        wing_width=5.0,
        spread_type=SpreadType.BEAR_CALL,
        min_open_interest=0,
        min_volume=0,
    )
    assert long_leg.contract.strike == 160.0


# ---------------------------------------------------------------------------
# Misc pure helpers
# ---------------------------------------------------------------------------


def test_round_to_tick():
    assert _round_to_tick(1.23, 0.05) == pytest.approx(1.25)
    assert _round_to_tick(1.22, 0.05) == pytest.approx(1.20)
    assert _round_to_tick(1.0, 0) == 1.0  # tick disabled


def test_quote_mid_handles_missing():
    assert _quote_mid(make_quote(bid=None, ask=1.0)) is None
    assert _quote_mid(make_quote(bid=1.0, ask=None)) is None
    assert _quote_mid(make_quote(bid=1.0, ask=0.9)) is None  # crossed
    assert _quote_mid(make_quote(bid=1.0, ask=1.2)) == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# CreditSpreadParams validation
# ---------------------------------------------------------------------------


def _params(**overrides):
    base = dict(
        underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
        spread_type=SpreadType.BULL_PUT,
    )
    base.update(overrides)
    return CreditSpreadParams(**base)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("target_short_delta", 0.0, "target_short_delta"),
        ("target_short_delta", 1.5, "target_short_delta"),
        ("wing_width", 0, "wing_width"),
        ("target_dte", -1, "target_dte"),
        ("dte_tolerance", -1, "dte_tolerance"),
        ("quantity", 0, "quantity"),
        ("limit_slippage", 1.0, "limit_slippage"),
        ("take_profit_pct", 0.0, "take_profit_pct"),
        ("take_profit_pct", 1.5, "take_profit_pct"),
        ("stop_loss_multiplier", 0, "stop_loss_multiplier"),
    ],
)
def test_params_validation(field, value, match):
    with pytest.raises(ValueError, match=match):
        _params(**{field: value})


# ---------------------------------------------------------------------------
# validate_request: BAG combo support
# ---------------------------------------------------------------------------


def test_validate_request_accepts_well_formed_bag():
    bag = Bag(symbol="AAPL", exchange="SMART", currency="USD")
    bag.comboLegs = [
        ComboLeg(conId=1, ratio=1, action="SELL", exchange="SMART"),
        ComboLeg(conId=2, ratio=1, action="BUY", exchange="SMART"),
    ]
    req = LimitRequest(contract=bag, side=OrderSide.SELL, quantity=1, limit_price=0.50)
    validate_request(req)  # must not raise


def test_validate_request_rejects_bag_without_legs():
    bag = Bag(symbol="AAPL", exchange="SMART", currency="USD")
    bag.comboLegs = []
    req = LimitRequest(contract=bag, side=OrderSide.SELL, quantity=1, limit_price=0.50)
    with pytest.raises(ValueError, match="BAG contract"):
        validate_request(req)


def test_validate_request_rejects_bag_with_zero_conid_leg():
    bag = Bag(symbol="AAPL", exchange="SMART", currency="USD")
    bag.comboLegs = [
        ComboLeg(conId=1, ratio=1, action="SELL", exchange="SMART"),
        ComboLeg(conId=0, ratio=1, action="BUY", exchange="SMART"),
    ]
    req = LimitRequest(contract=bag, side=OrderSide.SELL, quantity=1, limit_price=0.50)
    with pytest.raises(ValueError, match="BAG contract"):
        validate_request(req)


# ---------------------------------------------------------------------------
# CreditSpreadStrategy: build_plan + place + close
# ---------------------------------------------------------------------------


def _patch_fetcher_for_chain(monkeypatch, strategy, *, expirations, snapshot_quotes):
    """Stub the fetcher's chain-definition + snapshot calls."""
    chain_def = SimpleNamespace(
        underlying_conId=1,
        underlying_symbol="AAPL",
        trading_class="AAPL",
        multiplier="100",
        exchange="SMART",
        expirations=tuple(expirations),
        strikes=(140.0, 145.0, 150.0, 155.0, 160.0),
    )
    strategy._fetcher.fetch_chain_definition = AsyncMock(return_value=chain_def)
    strategy._fetcher.fetch_snapshot = AsyncMock(return_value=snapshot_quotes)
    return chain_def


async def test_build_plan_happy_path(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=140.0, right="P", con_id=140, delta=-0.10, bid=0.30, ask=0.40, expiry=expiry),
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.70, ask=0.90, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=1.10, ask=1.30, expiry=expiry),
        make_quote(strike=155.0, right="P", con_id=155, delta=-0.45, bid=2.00, ask=2.30, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 31,
    )

    params = _params(
        underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
        target_short_delta=0.30,
        wing_width=5.0,
        target_dte=30,
        dte_tolerance=14,
    )
    plan = await strat.build_plan(params)

    assert plan.short_leg.strike == 150.0
    assert plan.long_leg.strike == 145.0
    assert plan.width == 5.0
    # mid credit = (1.20 short - 0.80 long) * 100 = 40.0
    assert plan.net_credit == pytest.approx(40.0)
    assert plan.max_loss == pytest.approx(5 * 100 - 40.0)
    assert plan.max_profit == pytest.approx(40.0)
    # bull-put breakeven = short_strike - credit/multiplier
    assert plan.breakeven == pytest.approx(150.0 - 0.40)
    assert plan.short_delta == pytest.approx(-0.30)
    # TP debit = (1 - 0.5) * credit_per_share = 0.20
    assert plan.take_profit_debit == pytest.approx(0.20)
    # SL debit = (1 + 2) * 0.40 = 1.20, capped at width=5 → 1.20
    assert plan.stop_loss_debit == pytest.approx(1.20)
    # BAG wiring
    assert plan.bag.secType == "BAG"
    assert [(leg.conId, leg.action) for leg in plan.bag.comboLegs] == [(150, "SELL"), (145, "BUY")]


async def test_build_plan_enforces_min_credit(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.10, ask=0.15, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=0.20, ask=0.25, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )

    params = _params(
        underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
        wing_width=5.0,
        min_credit=20.0,  # mid credit here is only (0.225 - 0.125)*100 = 10
    )
    with pytest.raises(CreditSpreadError, match="below min_credit"):
        await strat.build_plan(params)


async def test_build_plan_enforces_min_credit_width_ratio(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.10, ask=0.15, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=0.20, ask=0.25, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )

    # width = 5*100 = 500; credit = 10; ratio = 0.02 → fails 0.10 floor.
    params = _params(
        underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
        wing_width=5.0,
        min_credit_width_ratio=0.10,
    )
    with pytest.raises(CreditSpreadError, match="Credit/width ratio"):
        await strat.build_plan(params)


async def test_place_routes_through_order_manager(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.70, ask=0.90, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=1.10, ask=1.30, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )

    plan = await strat.build_plan(
        _params(
            underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
            wing_width=5.0,
            quantity=2,
            limit_slippage=0.10,
        )
    )

    tracked = TrackedOrder(uuid="abc", request=None, trade=None, state=OrderState.SUBMITTED)
    fake_manager.limit.return_value = tracked

    result = await strat.place(plan)

    assert result is tracked
    args, kwargs = fake_manager.limit.call_args
    bag, side, qty, price = args
    assert bag is plan.bag
    # IB combo convention: BUY action + signed net cost (negative = credit).
    assert side == OrderSide.BUY
    assert qty == 2
    # net credit/share = 0.40; slippage 10% → 0.36; rounded to 0.05 tick → 0.35;
    # negated for the wire as the signed net cost.
    assert price == pytest.approx(-0.35)
    assert kwargs["tif"] == TimeInForce.DAY


async def test_close_uses_take_profit_debit_by_default(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.70, ask=0.90, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=1.10, ask=1.30, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )

    plan = await strat.build_plan(
        _params(
            underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
            wing_width=5.0,
            take_profit_pct=0.5,
        )
    )
    # tp_debit per share = (1-0.5) * (40/100) = 0.20 → rounded 0.20
    tracked = TrackedOrder(uuid="close-xyz", request=None, trade=None, state=OrderState.SUBMITTED)
    fake_manager.limit.return_value = tracked

    await strat.close(plan)

    args, _ = fake_manager.limit.call_args
    _, side, _, price = args
    assert side == OrderSide.SELL
    assert price == pytest.approx(-0.20)


async def test_constructor_requires_order_manager(fake_client, fake_fetcher):
    with pytest.raises(ValueError, match="order_manager"):
        CreditSpreadStrategy(fake_client, None, fetcher=fake_fetcher)  # type: ignore[arg-type]


async def test_current_mid_debit_returns_none_on_missing_quote(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.70, ask=0.90, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=1.10, ask=1.30, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )
    plan = await strat.build_plan(
        _params(
            underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
            wing_width=5.0,
        )
    )

    # Both legs return with a missing bid → mid unavailable.
    short_t = make_ticker_for(plan.short_leg.quote.contract, bid=0, ask=1.30)
    long_t = make_ticker_for(plan.long_leg.quote.contract, bid=0, ask=0.90)
    fake_client.ib.reqTickersAsync = AsyncMock(return_value=[short_t, long_t])

    assert await strat._current_mid_debit(plan) is None


async def test_current_mid_debit_computes_value(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.70, ask=0.90, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=1.10, ask=1.30, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )
    plan = await strat.build_plan(
        _params(
            underlying=SimpleNamespace(conId=1, symbol="AAPL", secType="STK"),
            wing_width=5.0,
        )
    )

    # Short mid 1.20, long mid 0.80 → debit 0.40
    short_t = make_ticker_for(plan.short_leg.quote.contract, bid=1.10, ask=1.30)
    long_t = make_ticker_for(plan.long_leg.quote.contract, bid=0.70, ask=0.90)
    fake_client.ib.reqTickersAsync = AsyncMock(return_value=[short_t, long_t])

    mid = await strat._current_mid_debit(plan)
    assert mid == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# outside_rth plumbing
# ---------------------------------------------------------------------------


def test_credit_spread_params_outside_rth_default_false():
    assert _params().outside_rth is False


def test_build_limit_threads_outside_rth():
    from ib_async import Stock
    from ibtws.unofficial.order.factory import build_limit, request_to_order

    contract = Stock(conId=1, symbol="AAPL", exchange="SMART", currency="USD")
    req = build_limit(contract, OrderSide.BUY, 1, 100.0, outside_rth=True)
    assert req.outside_rth is True
    order = request_to_order(req, order_ref="ref")
    assert order.outsideRth is True


def test_bracket_to_orders_threads_outside_rth():
    from ib_async import Stock
    from ibtws.unofficial.order.factory import build_bracket, bracket_to_orders

    contract = Stock(conId=1, symbol="AAPL", exchange="SMART", currency="USD")
    req = build_bracket(
        contract,
        OrderSide.BUY,
        1,
        take_profit_price=110.0,
        stop_loss_price=90.0,
        entry_limit_price=100.0,
        outside_rth=True,
    )
    parent, tp, sl = bracket_to_orders(
        req,
        parent_order_id=1,
        parent_ref="p",
        tp_ref="tp",
        sl_ref="sl",
        oca_group="g",
    )
    assert parent.outsideRth is True
    assert tp.outsideRth is True
    assert sl.outsideRth is True


async def test_strategy_place_forwards_outside_rth(fake_client, fake_fetcher, fake_manager, monkeypatch):
    strat = CreditSpreadStrategy(fake_client, fake_manager, fetcher=fake_fetcher)
    expiry = "20260619"
    chain = [
        make_quote(strike=145.0, right="P", con_id=145, delta=-0.20, bid=0.70, ask=0.90, expiry=expiry),
        make_quote(strike=150.0, right="P", con_id=150, delta=-0.30, bid=1.10, ask=1.30, expiry=expiry),
    ]
    _patch_fetcher_for_chain(monkeypatch, strat, expirations=[expiry], snapshot_quotes=chain)
    monkeypatch.setattr(
        "ibtws.unofficial.strategies.utils._parse_expiry_to_dte",
        lambda exp, now=None: 30,
    )

    plan = await strat.build_plan(_params(outside_rth=True))

    tracked = TrackedOrder(uuid="abc", request=None, trade=None, state=OrderState.SUBMITTED)
    fake_manager.limit.return_value = tracked

    await strat.place(plan)

    _, kwargs = fake_manager.limit.call_args
    assert kwargs["outside_rth"] is True
