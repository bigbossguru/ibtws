"""Unit tests for ibtws.unofficial.analysis.expected_move."""

from __future__ import annotations

import datetime as _dt
import math

import pandas as pd
import pytest

from ibtws.unofficial.analysis.expected_move import ExpectedMoveCalculator, ExpectedMoveResult


def _make_df(rows: list[dict], symbol: str = "SPY") -> pd.DataFrame:
    """Build an options chain DataFrame from simplified row dicts."""
    for row in rows:
        row.setdefault("symbol", symbol)
        row.setdefault("volume", None)
        row.setdefault("open_interest", None)
        row.setdefault("delta", None)
        row.setdefault("gamma", None)
        row.setdefault("vega", None)
        row.setdefault("theta", None)
        row.setdefault("timestamp", 0.0)
    return pd.DataFrame(rows)


def _future_expiration(days: int = 30) -> str:
    return (_dt.date.today() + _dt.timedelta(days=days)).strftime("%Y%m%d")


class TestQuoteMid:
    def test_valid(self):
        calc = ExpectedMoveCalculator()
        row = pd.Series({"bid": 5.0, "ask": 5.5})
        assert calc._quote_mid(row) == 5.25

    def test_none_when_bid_missing(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(pd.Series({"bid": None, "ask": 5.0})) is None

    def test_none_when_ask_missing(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(pd.Series({"bid": 5.0, "ask": None})) is None

    def test_none_when_bid_zero(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(pd.Series({"bid": 0.0, "ask": 5.0})) is None

    def test_none_when_ask_less_than_bid(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(pd.Series({"bid": 6.0, "ask": 5.0})) is None

    def test_none_when_bid_nan(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(pd.Series({"bid": float("nan"), "ask": 5.0})) is None

    def test_none_when_ask_nan(self):
        calc = ExpectedMoveCalculator()
        assert calc._quote_mid(pd.Series({"bid": 5.0, "ask": float("nan")})) is None


class TestFindAtmPair:
    def test_finds_closest_to_spot(self):
        calc = ExpectedMoveCalculator()
        df = _make_df([
            {
                "strike": 540,
                "right": "C",
                "bid": 1.0,
                "ask": 2.0,
                "iv": 0.2,
                "underlying_price": 551.0,
                "expiry": "20260620",
            },
            {
                "strike": 550,
                "right": "C",
                "bid": 1.0,
                "ask": 2.0,
                "iv": 0.2,
                "underlying_price": 551.0,
                "expiry": "20260620",
            },
            {
                "strike": 560,
                "right": "C",
                "bid": 1.0,
                "ask": 2.0,
                "iv": 0.2,
                "underlying_price": 551.0,
                "expiry": "20260620",
            },
            {
                "strike": 540,
                "right": "P",
                "bid": 1.0,
                "ask": 2.0,
                "iv": 0.2,
                "underlying_price": 551.0,
                "expiry": "20260620",
            },
            {
                "strike": 550,
                "right": "P",
                "bid": 1.0,
                "ask": 2.0,
                "iv": 0.2,
                "underlying_price": 551.0,
                "expiry": "20260620",
            },
            {
                "strike": 560,
                "right": "P",
                "bid": 1.0,
                "ask": 2.0,
                "iv": 0.2,
                "underlying_price": 551.0,
                "expiry": "20260620",
            },
        ])
        call, put = calc._find_atm_pair(df, 551.0)
        assert call["strike"] == 550
        assert put["strike"] == 550

    def test_empty_dataframe(self):
        calc = ExpectedMoveCalculator()
        df = pd.DataFrame(columns=["strike", "right", "bid", "ask", "iv", "underlying_price", "expiry"])
        call, put = calc._find_atm_pair(df, 100.0)
        assert call is None
        assert put is None

    def test_only_calls(self):
        calc = ExpectedMoveCalculator()
        df = _make_df([
            {
                "strike": 550,
                "right": "C",
                "bid": 5.0,
                "ask": 5.5,
                "iv": 0.24,
                "underlying_price": 550.0,
                "expiry": "20260620",
            },
        ])
        call, put = calc._find_atm_pair(df, 550.0)
        assert call is not None
        assert put is None


class TestStraddleExpectedMove:
    def test_valid(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"bid": 5.0, "ask": 5.5})
        put = pd.Series({"bid": 4.0, "ask": 4.5})
        assert calc._straddle_expected_move(call, put) == 5.25 + 4.25

    def test_none_when_call_missing(self):
        calc = ExpectedMoveCalculator()
        put = pd.Series({"bid": 4.0, "ask": 4.5})
        assert calc._straddle_expected_move(None, put) is None

    def test_none_when_mid_unavailable(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"bid": None, "ask": 5.5})
        put = pd.Series({"bid": 4.0, "ask": 4.5})
        assert calc._straddle_expected_move(call, put) is None


class TestExtractAtmIv:
    def test_averages_both(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"iv": 0.28})
        put = pd.Series({"iv": 0.32})
        assert calc._extract_atm_iv(call, put) == pytest.approx(0.30)

    def test_single_side(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"iv": 0.28})
        put = pd.Series({"iv": None})
        assert calc._extract_atm_iv(call, put) == 0.28

    def test_none_when_both_missing(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"iv": None})
        put = pd.Series({"iv": None})
        assert calc._extract_atm_iv(call, put) is None

    def test_none_when_both_none(self):
        calc = ExpectedMoveCalculator()
        assert calc._extract_atm_iv(None, None) is None

    def test_nan_treated_as_missing(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"iv": float("nan")})
        put = pd.Series({"iv": 0.30})
        assert calc._extract_atm_iv(call, put) == 0.30

    def test_zero_iv_treated_as_missing(self):
        calc = ExpectedMoveCalculator()
        call = pd.Series({"iv": 0.0})
        put = pd.Series({"iv": 0.30})
        assert calc._extract_atm_iv(call, put) == 0.30


class TestIvExpectedMove:
    def test_formula(self):
        calc = ExpectedMoveCalculator()
        result = calc._iv_expected_move(100.0, 0.30, _future_expiration(30))
        expected = 100.0 * 0.30 * math.sqrt(31.0 / 365.0)
        assert result == pytest.approx(expected)

    def test_none_when_iv_zero(self):
        calc = ExpectedMoveCalculator()
        assert calc._iv_expected_move(100.0, 0.0, _future_expiration(30)) is None

    def test_none_when_spot_zero(self):
        calc = ExpectedMoveCalculator()
        assert calc._iv_expected_move(0.0, 0.30, _future_expiration(30)) is None


class TestExpectedMoveCalculator:
    def test_calculate_returns_expected_move_result(self):
        calc = ExpectedMoveCalculator()
        exp = _future_expiration(30)
        df = _make_df([
            {"strike": 548, "right": "C", "bid": 7.0, "ask": 7.5, "iv": 0.25, "underlying_price": 550.0, "expiry": exp},
            {"strike": 550, "right": "C", "bid": 5.0, "ask": 5.5, "iv": 0.24, "underlying_price": 550.0, "expiry": exp},
            {"strike": 552, "right": "C", "bid": 3.5, "ask": 4.0, "iv": 0.23, "underlying_price": 550.0, "expiry": exp},
            {"strike": 548, "right": "P", "bid": 5.5, "ask": 6.0, "iv": 0.26, "underlying_price": 550.0, "expiry": exp},
            {"strike": 550, "right": "P", "bid": 4.0, "ask": 4.5, "iv": 0.25, "underlying_price": 550.0, "expiry": exp},
            {"strike": 552, "right": "P", "bid": 2.5, "ask": 3.0, "iv": 0.24, "underlying_price": 550.0, "expiry": exp},
        ])

        result = calc.calculate(df)

        assert isinstance(result, ExpectedMoveResult)
        assert result.underlying_symbol == "SPY"
        assert result.spot == 550.0
        assert result.straddle_move == pytest.approx(9.5)
        assert result.straddle_pct == pytest.approx(9.5 / 550.0 * 100)
        assert result.atm_iv == pytest.approx(0.245)
        expected_iv_move = 550.0 * 0.245 * math.sqrt(31.0 / 365.0)
        assert result.iv_move == pytest.approx(expected_iv_move)
        assert result.iv_pct == pytest.approx(expected_iv_move / 550.0 * 100)
        assert result.avg_move == pytest.approx((9.5 + expected_iv_move) / 2.0)

    def test_raises_on_empty_dataframe(self):
        calc = ExpectedMoveCalculator()
        df = pd.DataFrame(columns=["strike", "right", "bid", "ask", "iv", "underlying_price", "expiry"])
        with pytest.raises(ValueError, match="empty DataFrame"):
            calc.calculate(df)

    def test_raises_on_missing_columns(self):
        calc = ExpectedMoveCalculator()
        df = pd.DataFrame({"strike": [100], "right": ["C"]})
        with pytest.raises(ValueError, match="missing required columns"):
            calc.calculate(df)

    def test_raises_when_underlying_price_all_nan(self):
        calc = ExpectedMoveCalculator()
        df = _make_df([
            {
                "strike": 550,
                "right": "C",
                "bid": 5.0,
                "ask": 5.5,
                "iv": 0.24,
                "underlying_price": None,
                "expiry": "20260620",
            },
            {
                "strike": 550,
                "right": "P",
                "bid": 4.0,
                "ask": 4.5,
                "iv": 0.25,
                "underlying_price": None,
                "expiry": "20260620",
            },
        ])
        with pytest.raises(ValueError, match="spot price"):
            calc.calculate(df)

    def test_straddle_none_when_only_calls(self):
        calc = ExpectedMoveCalculator()
        exp = _future_expiration(30)
        df = _make_df([
            {"strike": 550, "right": "C", "bid": 5.0, "ask": 5.5, "iv": 0.24, "underlying_price": 550.0, "expiry": exp},
        ])
        result = calc.calculate(df)
        assert result.straddle_move is None
        assert result.iv_move is not None  # IV still works from the call side

    def test_no_symbol_column(self):
        calc = ExpectedMoveCalculator()
        exp = _future_expiration(30)
        df = pd.DataFrame([
            {"strike": 550, "right": "C", "bid": 5.0, "ask": 5.5, "iv": 0.24, "underlying_price": 550.0, "expiry": exp},
            {"strike": 550, "right": "P", "bid": 4.0, "ask": 4.5, "iv": 0.25, "underlying_price": 550.0, "expiry": exp},
        ])
        result = calc.calculate(df)
        assert result.underlying_symbol == ""
        assert result.straddle_move == pytest.approx(9.5)

    def test_avg_move_none_when_straddle_unavailable(self):
        calc = ExpectedMoveCalculator()
        exp = _future_expiration(30)
        # bid=None makes straddle unavailable
        df = _make_df([
            {
                "strike": 550,
                "right": "C",
                "bid": None,
                "ask": 5.5,
                "iv": 0.24,
                "underlying_price": 550.0,
                "expiry": exp,
            },
            {
                "strike": 550,
                "right": "P",
                "bid": None,
                "ask": 4.5,
                "iv": 0.25,
                "underlying_price": 550.0,
                "expiry": exp,
            },
        ])
        result = calc.calculate(df)
        assert result.straddle_move is None
        assert result.iv_move is not None
        assert result.avg_move is None

    def test_avg_move_none_when_iv_unavailable(self):
        calc = ExpectedMoveCalculator()
        exp = _future_expiration(30)
        df = _make_df([
            {"strike": 550, "right": "C", "bid": 5.0, "ask": 5.5, "iv": None, "underlying_price": 550.0, "expiry": exp},
            {"strike": 550, "right": "P", "bid": 4.0, "ask": 4.5, "iv": None, "underlying_price": 550.0, "expiry": exp},
        ])
        result = calc.calculate(df)
        assert result.straddle_move == pytest.approx(9.5)
        assert result.iv_move is None
        assert result.avg_move is None
