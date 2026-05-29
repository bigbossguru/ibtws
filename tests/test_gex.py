"""Unit tests for GEX pure calculation functions."""

import pytest
from ib_async import Option

from ibtws.unofficial.analysis.gex import (
    StrikeGEX,
    _build_strike_gex,
    _compute_contract_gex,
    _find_zero_gamma,
)
from ibtws.unofficial.option.models import OptionQuote


def _make_quote(
    strike: float,
    right: str,
    gamma: float | None = 0.05,
    open_interest: float | None = 1000,
    volume: float | None = 500,
) -> OptionQuote:
    contract = Option(
        symbol="SPX",
        lastTradeDateOrContractMonth="20260601",
        strike=strike,
        right=right,
        exchange="SMART",
        multiplier="100",
    )
    return OptionQuote(
        contract=contract,
        gamma=gamma,
        open_interest=open_interest,
        volume=volume,
        bid=1.0,
        ask=1.5,
    )


class TestComputeContractGEX:
    def test_call_positive_gex(self):
        q = _make_quote(5000, "C", gamma=0.01, open_interest=2000)
        gex = _compute_contract_gex(q, spot=5000.0, use_open_interest=True)
        # gamma * OI * multiplier * spot^2 * 0.01
        expected = 0.01 * 2000 * 100 * 5000 * 5000 * 0.01
        assert gex == pytest.approx(expected)
        assert gex > 0

    def test_put_negative_gex(self):
        q = _make_quote(5000, "P", gamma=0.01, open_interest=2000)
        gex = _compute_contract_gex(q, spot=5000.0, use_open_interest=True)
        assert gex < 0

    def test_zero_gamma_returns_zero(self):
        q = _make_quote(5000, "C", gamma=None)
        assert _compute_contract_gex(q, spot=5000.0, use_open_interest=True) == 0.0

    def test_zero_oi_returns_zero(self):
        q = _make_quote(5000, "C", gamma=0.05, open_interest=0)
        assert _compute_contract_gex(q, spot=5000.0, use_open_interest=True) == 0.0

    def test_use_volume_instead_of_oi(self):
        q = _make_quote(5000, "C", gamma=0.01, open_interest=0, volume=100)
        gex = _compute_contract_gex(q, spot=5000.0, use_open_interest=False)
        expected = 0.01 * 100 * 100 * 5000 * 5000 * 0.01
        assert gex == pytest.approx(expected)


class TestBuildStrikeGEX:
    def test_aggregates_by_strike(self):
        quotes = [
            _make_quote(5000, "C", gamma=0.01, open_interest=1000),
            _make_quote(5000, "P", gamma=0.01, open_interest=800),
            _make_quote(5050, "C", gamma=0.008, open_interest=500),
        ]
        result = _build_strike_gex(quotes, spot=5000.0, use_open_interest=True)
        assert 5000 in result
        assert 5050 in result
        assert result[5000].call_gex > 0
        assert result[5000].put_gex < 0

    def test_empty_quotes(self):
        result = _build_strike_gex([], spot=5000.0, use_open_interest=True)
        assert result == {}


class TestFindZeroGamma:
    def test_sign_change(self):
        strikes = [
            StrikeGEX(strike=4900, call_gex=100, put_gex=-200, net_gex=-100),
            StrikeGEX(strike=5000, call_gex=200, put_gex=-100, net_gex=100),
        ]
        zero = _find_zero_gamma(strikes)
        assert zero is not None
        assert 4900 < zero < 5000
        # Linear interpolation: 4900 + (100/(100+100)) * 100 = 4950
        assert zero == pytest.approx(4950.0)

    def test_no_sign_change(self):
        strikes = [
            StrikeGEX(strike=4900, call_gex=200, put_gex=-50, net_gex=150),
            StrikeGEX(strike=5000, call_gex=300, put_gex=-100, net_gex=200),
        ]
        assert _find_zero_gamma(strikes) is None

    def test_single_strike(self):
        strikes = [StrikeGEX(strike=5000, call_gex=100, put_gex=-50, net_gex=50)]
        assert _find_zero_gamma(strikes) is None
