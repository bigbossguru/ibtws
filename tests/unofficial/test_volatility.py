"""Unit tests for ibtws.unofficial.analysis.volatility."""

from __future__ import annotations


from ibtws.unofficial.analysis.volatility import premarket_vol_regime, VolRegimeResult


class TestPremarketVolRegime:
    """Tests for the pure scoring function."""

    def test_spec_example_yellow(self):
        """Verify the example from the spec produces score=36, YELLOW."""
        result = premarket_vol_regime(
            vix1d=11.50,
            vix9d=14.30,
            vix=16.20,
            vx_front=17.80,
            vix_prev_close=15.40,
            vix_52w_high=23.50,
            vix_52w_low=12.10,
            rv_20d=10.80,
            is_pre_market=True,
        )
        assert result.score == 36
        assert result.regime == "YELLOW"
        assert result.trade is True
        assert result.vix1d_absolute.score == 8
        assert result.vix1d_vix_ratio.score == 7
        assert result.term_structure.score == 7
        assert result.vix9d_vix_ratio.score == 5
        assert result.overnight_vix_chg.score == 7
        assert result.iv_rank.score == 1
        assert result.iv_rv_spread.score == 1

    def test_green_regime(self):
        """Low vol environment -> GREEN."""
        result = premarket_vol_regime(
            vix1d=9.0,
            vix9d=10.0,
            vix=13.0,
            vx_front=16.0,
            vix_prev_close=13.5,
            vix_52w_high=25.0,
            vix_52w_low=11.0,
            rv_20d=8.0,
        )
        assert result.regime == "GREEN"
        assert result.score < 30
        assert result.trade is True

    def test_red_regime(self):
        """High vol / stress environment -> RED."""
        result = premarket_vol_regime(
            vix1d=28.0,
            vix9d=30.0,
            vix=27.0,
            vx_front=25.0,
            vix_prev_close=20.0,
            vix_52w_high=30.0,
            vix_52w_low=12.0,
            rv_20d=30.0,
        )
        assert result.regime == "RED"
        assert result.score >= 55
        assert result.trade is False

    def test_zero_division_safety(self):
        """Zero inputs should not raise."""
        result = premarket_vol_regime(
            vix1d=0.0,
            vix9d=0.0,
            vix=0.0,
            vx_front=0.0,
            vix_prev_close=0.0,
            vix_52w_high=0.0,
            vix_52w_low=0.0,
            rv_20d=0.0,
        )
        assert isinstance(result, VolRegimeResult)
        assert 0 <= result.score <= 100

    def test_data_note_pre_market(self):
        result = premarket_vol_regime(
            vix1d=12.0,
            vix9d=13.0,
            vix=14.0,
            vx_front=15.0,
            vix_prev_close=13.5,
            vix_52w_high=25.0,
            vix_52w_low=11.0,
            rv_20d=10.0,
            is_pre_market=True,
        )
        assert "prev close" in result.data_note

    def test_data_note_market_hours(self):
        result = premarket_vol_regime(
            vix1d=12.0,
            vix9d=13.0,
            vix=14.0,
            vx_front=15.0,
            vix_prev_close=13.5,
            vix_52w_high=25.0,
            vix_52w_low=11.0,
            rv_20d=10.0,
            is_pre_market=False,
        )
        assert "all live" in result.data_note
