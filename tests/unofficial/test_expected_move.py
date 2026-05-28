"""Unit tests for ibtws.unofficial.analysis.expected_move."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from ib_async import Option

from ibtws.unofficial.helpers import calc_dte
from ibtws.unofficial.analysis.expected_move import (
    ExpectedMoveCalculator,
    ExpectedMoveResult,
    _extract_atm_iv,
    _find_atm_pair,
    _iv_expected_move,
    _quote_mid,
    _straddle_expected_move,
)
from ibtws.unofficial.option.models import OptionQuote


def _make_quote(strike: float, right: str, *, bid=None, ask=None, iv=None, underlying_price=None) -> OptionQuote:
    return OptionQuote(
        contract=Option(symbol="SPY", lastTradeDateOrContractMonth="20260620", strike=strike, right=right),
        bid=bid,
        ask=ask,
        iv=iv,
        underlying_price=underlying_price,
    )


class TestQuoteMid:
    def test_valid(self):
        q = _make_quote(100, "C", bid=5.0, ask=5.5)
        assert _quote_mid(q) == 5.25

    def test_none_when_bid_missing(self):
        assert _quote_mid(_make_quote(100, "C", bid=None, ask=5.0)) is None

    def test_none_when_ask_missing(self):
        assert _quote_mid(_make_quote(100, "C", bid=5.0, ask=None)) is None

    def test_none_when_bid_zero(self):
        assert _quote_mid(_make_quote(100, "C", bid=0.0, ask=5.0)) is None

    def test_none_when_ask_less_than_bid(self):
        assert _quote_mid(_make_quote(100, "C", bid=6.0, ask=5.0)) is None


class TestCalcDte:
    def test_future_date(self):
        import datetime as _dt

        future = (_dt.date.today() + _dt.timedelta(days=10)).strftime("%Y%m%d")
        assert calc_dte(future) == 10.0

    def test_past_date_returns_zero(self):
        assert calc_dte("20200101") == 0.0


class TestFindAtmPair:
    def test_finds_closest_to_spot(self):
        quotes = [
            _make_quote(540, "C"),
            _make_quote(550, "C"),
            _make_quote(560, "C"),
            _make_quote(540, "P"),
            _make_quote(550, "P"),
            _make_quote(560, "P"),
        ]
        call, put = _find_atm_pair(quotes, 551.0)
        assert call.contract.strike == 550
        assert put.contract.strike == 550

    def test_empty_quotes(self):
        call, put = _find_atm_pair([], 100.0)
        assert call is None
        assert put is None


class TestStraddleExpectedMove:
    def test_valid(self):
        call = _make_quote(550, "C", bid=5.0, ask=5.5)
        put = _make_quote(550, "P", bid=4.0, ask=4.5)
        assert _straddle_expected_move(call, put) == 5.25 + 4.25

    def test_none_when_call_missing(self):
        assert _straddle_expected_move(None, _make_quote(550, "P", bid=4.0, ask=4.5)) is None

    def test_none_when_mid_unavailable(self):
        call = _make_quote(550, "C", bid=None, ask=5.5)
        put = _make_quote(550, "P", bid=4.0, ask=4.5)
        assert _straddle_expected_move(call, put) is None


class TestExtractAtmIv:
    def test_averages_both(self):
        call = _make_quote(550, "C", iv=0.28)
        put = _make_quote(550, "P", iv=0.32)
        assert abs(_extract_atm_iv(call, put) - 0.30) < 1e-10

    def test_single_side(self):
        call = _make_quote(550, "C", iv=0.28)
        put = _make_quote(550, "P", iv=None)
        assert _extract_atm_iv(call, put) == 0.28

    def test_none_when_both_missing(self):
        assert _extract_atm_iv(_make_quote(550, "C"), _make_quote(550, "P")) is None

    def test_none_when_both_none(self):
        assert _extract_atm_iv(None, None) is None


class TestIvExpectedMove:
    def test_formula(self):
        result = _iv_expected_move(100.0, 0.30, 30.0)
        expected = 100.0 * 0.30 * math.sqrt(30.0 / 365.0)
        assert abs(result - expected) < 1e-10

    def test_none_when_dte_zero(self):
        assert _iv_expected_move(100.0, 0.30, 0.0) is None

    def test_none_when_iv_zero(self):
        assert _iv_expected_move(100.0, 0.0, 30.0) is None


class TestExpectedMoveCalculator:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.get_market_data = AsyncMock(return_value=MagicMock(last=550.0, close=550.0, bid=549.0, ask=551.0))
        client.get_historical_data = AsyncMock(return_value=pd.DataFrame({"close": np.linspace(100, 110, 30)}))
        return client

    @pytest.fixture
    def mock_chain(self):
        chain = MagicMock()
        chain.fetch_snapshot = AsyncMock(
            return_value=[
                _make_quote(548, "C", bid=7.0, ask=7.5, iv=0.25, underlying_price=550.0),
                _make_quote(550, "C", bid=5.0, ask=5.5, iv=0.24, underlying_price=550.0),
                _make_quote(552, "C", bid=3.5, ask=4.0, iv=0.23, underlying_price=550.0),
                _make_quote(548, "P", bid=5.5, ask=6.0, iv=0.26, underlying_price=550.0),
                _make_quote(550, "P", bid=4.0, ask=4.5, iv=0.25, underlying_price=550.0),
                _make_quote(552, "P", bid=2.5, ask=3.0, iv=0.24, underlying_price=550.0),
            ]
        )
        return chain

    async def test_returns_all_three_methods(self, mock_client, mock_chain):
        import datetime as _dt

        exp = (_dt.date.today() + _dt.timedelta(days=30)).strftime("%Y%m%d")
        calc = ExpectedMoveCalculator(mock_client, mock_chain)
        contract = MagicMock(symbol="SPY")

        result = await calc.calculate(contract, exp)

        assert isinstance(result, ExpectedMoveResult)
        assert result.underlying_symbol == "SPY"
        assert result.spot == 550.0
        assert result.dte == 30.0

        # Straddle: ATM call mid (5.25) + ATM put mid (4.25) = 9.5
        assert result.straddle_move == 9.5
        assert abs(result.straddle_pct - 9.5 / 550.0 * 100) < 1e-10

        # IV: avg of 0.24 and 0.25 = 0.245
        expected_iv_move = 550.0 * 0.245 * math.sqrt(30.0 / 365.0)
        assert abs(result.iv_move - expected_iv_move) < 1e-10
        assert result.atm_iv == pytest.approx(0.245)

        # HV: should be non-None since we provided historical data
        assert result.hv_move is not None
        assert result.hv is not None
        assert result.hv > 0

    async def test_raises_when_no_spot(self, mock_client, mock_chain):
        mock_chain.fetch_snapshot.return_value = [_make_quote(550, "C", underlying_price=None)]
        calc = ExpectedMoveCalculator(mock_client, mock_chain)

        with pytest.raises(ValueError, match="Cannot determine spot"):
            await calc.calculate(MagicMock(symbol="XYZ"), "20260620")

    async def test_hv_none_when_insufficient_data(self, mock_client, mock_chain):
        import datetime as _dt

        mock_client.get_historical_data.return_value = pd.DataFrame({"close": [100.0, 101.0]})
        exp = (_dt.date.today() + _dt.timedelta(days=30)).strftime("%Y%m%d")
        calc = ExpectedMoveCalculator(mock_client, mock_chain)

        result = await calc.calculate(MagicMock(symbol="SPY"), exp)

        assert result.hv_move is None
        assert result.hv is None
