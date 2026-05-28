"""Unit tests for ibtws.unofficial.analysis.volatility."""

from __future__ import annotations

from ibtws.unofficial.analysis.volatility import premarket_vol_regime, VolRegimeResult


def _defaults(**overrides):
    """Base inputs for a calm market — GREEN regime."""
    base = dict(
        vix1d=10.0,
        vix=14.0,
        vix3m=16.5,
        vvix=88.0,
        vvix_prev_close=87.0,
        vx_front=16.5,
        vix_prev_close=13.8,
        rv_20d=9.0,
        is_pre_market=True,
    )
    base.update(overrides)
    return base


class TestPremarketVolRegime:
    """Tests for the pure scoring function."""

    def test_green_regime_calm_market(self):
        """Low vol, contango, low VVIX, positive VRP -> GREEN."""
        result = premarket_vol_regime(**_defaults())
        assert result.regime == "GREEN"
        assert result.score < 25
        assert result.trade is True
        assert not result.vrp_override

    def test_yellow_regime_elevated(self):
        """Moderately elevated signals -> YELLOW."""
        result = premarket_vol_regime(
            **_defaults(
                vix=19.0,
                vix1d=15.0,
                vix3m=20.0,
                vvix=112.0,
                vvix_prev_close=110.0,
                vix_prev_close=18.0,
                rv_20d=14.0,
            )
        )
        assert result.regime == "YELLOW"
        assert 25 <= result.score < 50
        assert result.trade is True

    def test_red_regime_stress(self):
        """High vol, backwardation, high VVIX -> RED."""
        result = premarket_vol_regime(
            **_defaults(
                vix=30.0,
                vix1d=32.0,
                vix3m=27.0,
                vvix=145.0,
                vvix_prev_close=130.0,
                vx_front=28.0,
                vix_prev_close=22.0,
                rv_20d=32.0,
            )
        )
        assert result.regime == "RED"
        assert result.score >= 50
        assert result.trade is False

    def test_vrp_override_forces_yellow(self):
        """Negative VRP overrides GREEN to YELLOW even with low score."""
        result = premarket_vol_regime(
            **_defaults(
                vix=14.0,
                rv_20d=16.0,  # negative VRP: 14 - 16 = -2
            )
        )
        assert result.vrp_override is True
        assert result.regime != "GREEN"
        assert "VRP" in result.action

    def test_vrp_override_does_not_downgrade_red(self):
        """VRP override doesn't interfere with RED — RED stays RED."""
        result = premarket_vol_regime(
            **_defaults(
                vix=30.0,
                vix1d=32.0,
                vix3m=27.0,
                vvix=145.0,
                vvix_prev_close=130.0,
                vx_front=28.0,
                vix_prev_close=22.0,
                rv_20d=35.0,
            )
        )
        assert result.regime == "RED"

    def test_term_structure_backwardation_max(self):
        """VIX > VIX3M by >8% -> max score 25."""
        result = premarket_vol_regime(
            **_defaults(
                vix=25.0,
                vix3m=22.0,  # ratio = 1.136
            )
        )
        assert result.term_structure.score == 25

    def test_term_structure_deep_contango(self):
        """VIX << VIX3M -> score 0."""
        result = premarket_vol_regime(
            **_defaults(
                vix=12.0,
                vix3m=16.0,  # ratio = 0.75
            )
        )
        assert result.term_structure.score == 0

    def test_vvix_panic_level(self):
        """VVIX >= 140 -> max score 20."""
        result = premarket_vol_regime(**_defaults(vvix=145.0))
        assert result.vvix_level.score == 20

    def test_vvix_calm(self):
        """VVIX < 90 -> score 0."""
        result = premarket_vol_regime(**_defaults(vvix=85.0))
        assert result.vvix_level.score == 0

    def test_vvix_divergence_detected(self):
        """VVIX rising +10% while VIX flat = max divergence."""
        result = premarket_vol_regime(
            **_defaults(
                vvix=110.0,
                vvix_prev_close=100.0,
                vix=14.0,
                vix_prev_close=14.0,
            )
        )
        assert result.vvix_divergence.score == 10

    def test_vvix_divergence_not_triggered_when_both_rise(self):
        """Both VVIX and VIX rising = no divergence."""
        result = premarket_vol_regime(
            **_defaults(
                vvix=110.0,
                vvix_prev_close=100.0,
                vix=16.0,
                vix_prev_close=14.0,
            )
        )
        assert result.vvix_divergence.score == 0

    def test_vix1d_vix_ratio_hot_day(self):
        """VIX1D >= VIX -> max score 20."""
        result = premarket_vol_regime(**_defaults(vix1d=15.0, vix=14.0))
        assert result.vix1d_vix_ratio.score == 20

    def test_vix1d_vix_ratio_quiet_day(self):
        """VIX1D << VIX -> score 0."""
        result = premarket_vol_regime(**_defaults(vix1d=9.0, vix=14.0))
        assert result.vix1d_vix_ratio.score == 0

    def test_iv_rv_spread_wide_is_safe(self):
        """Wide VRP (IV >> RV) = score 0."""
        result = premarket_vol_regime(**_defaults(vix=20.0, rv_20d=10.0))
        assert result.iv_rv_spread.score == 0

    def test_iv_rv_spread_negative_is_max(self):
        """Deeply negative VRP = max score 5."""
        result = premarket_vol_regime(**_defaults(vix=14.0, rv_20d=18.0))
        assert result.iv_rv_spread.score == 5

    def test_zero_division_safety(self):
        """Zero inputs should not raise."""
        result = premarket_vol_regime(
            vix1d=0.0,
            vix=0.0,
            vix3m=0.0,
            vvix=0.0,
            vvix_prev_close=0.0,
            vx_front=0.0,
            vix_prev_close=0.0,
            rv_20d=0.0,
        )
        assert isinstance(result, VolRegimeResult)
        assert 0 <= result.score <= 100

    def test_max_score_is_100(self):
        """All components at max should sum to exactly 100."""
        result = premarket_vol_regime(
            vix1d=35.0,
            vix=35.0,
            vix3m=30.0,
            vvix=150.0,
            vvix_prev_close=130.0,
            vx_front=30.0,
            vix_prev_close=35.0,
            rv_20d=40.0,
        )
        assert result.score == 100

    def test_term_structure_falls_back_to_vx_front(self):
        """When VIX3M is 0, use VX front for term structure."""
        result = premarket_vol_regime(**_defaults(vix=14.0, vix3m=0.0, vx_front=16.0))
        # 14/16 = 0.875 -> score 5
        assert result.term_structure.score == 5
        assert result.term_structure.value == 0.875

    def test_data_note_pre_market(self):
        result = premarket_vol_regime(**_defaults(is_pre_market=True))
        assert "prev close" in result.data_note

    def test_data_note_market_hours(self):
        result = premarket_vol_regime(**_defaults(is_pre_market=False))
        assert "all live" in result.data_note

    def test_data_note_vrp_override(self):
        result = premarket_vol_regime(**_defaults(vix=14.0, rv_20d=17.0))
        assert "VRP override" in result.data_note
