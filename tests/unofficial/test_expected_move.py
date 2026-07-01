"""Unit tests for ibtws.unofficial.analysis.expected_move."""

from __future__ import annotations

import asyncio
import datetime as _dt
import math

import pytest
from ib_async import Option

from ibtws.unofficial.analysis.expected_move import ExpectedMoveCalculator, ExpectedMoveResult
from ibtws.unofficial.option.models import OptionQuote


def _make_quote(
    strike: float, right: str, *, bid=None, ask=None, iv=None, underlying_price=None, expiration="20260620"
) -> OptionQuote:
    return OptionQuote(
        contract=Option(symbol="SPY", lastTradeDateOrContractMonth=expiration, strike=strike, right=right),
        bid=bid,
        ask=ask,
        iv=iv,
        underlying_price=underlying_price,
    )


def _future_expiration(days: int = 30) -> str:
    return (_dt.date.today() + _dt.timedelta(days=days)).strftime("%Y%m%d")


class TestQuoteMid:
    def test_valid(self):
        q = _make_quote(100, "C", bid=5.0, ask=5.5)
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(q) == 5.25

    def test_none_when_bid_missing(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(_make_quote(100, "C", bid=None, ask=5.0)) is None

    def test_none_when_ask_missing(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(_make_quote(100, "C", bid=5.0, ask=None)) is None

    def test_none_when_bid_zero(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(_make_quote(100, "C", bid=0.0, ask=5.0)) is None

    def test_none_when_ask_less_than_bid(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(_make_quote(100, "C", bid=6.0, ask=5.0)) is None


class TestFindAtmPair:
    def test_finds_closest_to_spot(self):
        calc = ExpectedMoveCalculator()
        quotes = [
            _make_quote(540, "C"),
            _make_quote(550, "C"),
            _make_quote(560, "C"),
            _make_quote(540, "P"),
            _make_quote(550, "P"),
            _make_quote(560, "P"),
        ]
        call, put = calc._find_atm_pair(quotes, 551.0)
        assert call.contract.strike == 550
        assert put.contract.strike == 550

    def test_empty_quotes(self):
        calc = ExpectedMoveCalculator()
        call, put = calc._find_atm_pair([], 100.0)
        assert call is None
        assert put is None


class TestStraddleExpectedMove:
    def test_valid(self):
        calc = ExpectedMoveCalculator()
        call = _make_quote(550, "C", bid=5.0, ask=5.5)
        put = _make_quote(550, "P", bid=4.0, ask=4.5)
        assert calc._straddle_expected_move(call, put) == 5.25 + 4.25

    def test_none_when_call_missing(self):
        calc = ExpectedMoveCalculator()
        assert calc._straddle_expected_move(None, _make_quote(550, "P", bid=4.0, ask=4.5)) is None

    def test_none_when_mid_unavailable(self):
        calc = ExpectedMoveCalculator()
        call = _make_quote(550, "C", bid=None, ask=5.5)
        put = _make_quote(550, "P", bid=4.0, ask=4.5)
        assert calc._straddle_expected_move(call, put) is None


class TestExtractAtmIv:
    def test_averages_both(self):
        calc = ExpectedMoveCalculator()
        call = _make_quote(550, "C", iv=0.28)
        put = _make_quote(550, "P", iv=0.32)
        assert calc._extract_atm_iv(call, put) == pytest.approx(0.30)

    def test_single_side(self):
        calc = ExpectedMoveCalculator()
        call = _make_quote(550, "C", iv=0.28)
        put = _make_quote(550, "P", iv=None)
        assert calc._extract_atm_iv(call, put) == 0.28

    def test_none_when_both_missing(self):
        calc = ExpectedMoveCalculator()
        assert calc._extract_atm_iv(_make_quote(550, "C"), _make_quote(550, "P")) is None

    def test_none_when_both_none(self):
        calc = ExpectedMoveCalculator()
        assert calc._extract_atm_iv(None, None) is None


class TestIvExpectedMove:
    def test_formula(self):
        calc = ExpectedMoveCalculator()
        result = calc._iv_expected_move(100.0, 0.30, _future_expiration(30))
        expected = 100.0 * 0.30 * math.sqrt(31.0 / 365.0)
        assert result == pytest.approx(expected)

    def test_none_when_iv_zero(self):
        calc = ExpectedMoveCalculator()
        assert calc._iv_expected_move(100.0, 0.0, _future_expiration(30)) is None


class TestExpectedMoveCalculator:
    def test_calculate_returns_expected_move_result(self):
        calc = ExpectedMoveCalculator()
        quotes = [
            _make_quote(548, "C", bid=7.0, ask=7.5, iv=0.25, underlying_price=550.0, expiration=_future_expiration(30)),
            _make_quote(550, "C", bid=5.0, ask=5.5, iv=0.24, underlying_price=550.0, expiration=_future_expiration(30)),
            _make_quote(552, "C", bid=3.5, ask=4.0, iv=0.23, underlying_price=550.0, expiration=_future_expiration(30)),
            _make_quote(548, "P", bid=5.5, ask=6.0, iv=0.26, underlying_price=550.0, expiration=_future_expiration(30)),
            _make_quote(550, "P", bid=4.0, ask=4.5, iv=0.25, underlying_price=550.0, expiration=_future_expiration(30)),
            _make_quote(552, "P", bid=2.5, ask=3.0, iv=0.24, underlying_price=550.0, expiration=_future_expiration(30)),
        ]

        result = asyncio.run(calc.calculate(quotes))

        assert isinstance(result, ExpectedMoveResult)
        assert result.underlying_symbol == "SPY"
        assert result.spot == 550.0
        assert result.straddle_move == pytest.approx(9.5)
        assert result.straddle_pct == pytest.approx(9.5 / 550.0 * 100)
        assert result.atm_iv == pytest.approx(0.245)
        expected_iv_move = 550.0 * 0.245 * math.sqrt(31.0 / 365.0)
        assert result.iv_move == pytest.approx(expected_iv_move)
        assert result.iv_pct == pytest.approx(expected_iv_move / 550.0 * 100)
