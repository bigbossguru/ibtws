"""Unit tests for ibtws.unofficial.analysis.volatility_regime.

The scenario tests mirror the worked examples A-G of section 4 of
``VOLATILITY_REGIME_CONCEPT.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ibtws.unofficial.analysis.volatility_regime import (
    NO_BASE_LEVEL,
    TRADING_DAYS_PER_YEAR,
    VolatilityRegimeConfig,
    _expected_move_pct,
    _percentile_rank,
    _realised_vol,
    detect_volatility_regime,
)

# ----------------------------------------------------------------------
# Fixtures / builders
# ----------------------------------------------------------------------


def _vix1d_history(n: int = 60, low: float = 8.0, high: float = 18.0) -> pd.Series:
    """Evenly spread VIX1D opens so that a target percentile rank is easy to hit."""
    return pd.Series(np.linspace(low, high, n), dtype="float64")


def _vix1d_at_rank(rank: float, history: pd.Series) -> float:
    """A VIX1D value whose percentile rank inside *history* is ~*rank*."""
    value = float(np.quantile(history.to_numpy(), rank / 100.0)) + 1e-9
    assert abs(_percentile_rank(history, value) - rank) <= 2.0
    return value


def _spx_closes(n: int = 21, daily_vol: float = 0.006, start: float = 5000.0) -> pd.Series:
    """Deterministic closes with a known close-to-close volatility.

    Alternating ±*daily_vol* log returns give a sample std of exactly
    *daily_vol* (up to the ddof correction), so RV20 is predictable.
    """
    logs = [math.log(start)]
    for i in range(n - 1):
        logs.append(logs[-1] + (daily_vol if i % 2 == 0 else -daily_vol))
    return pd.Series(np.exp(logs), dtype="float64")


def _inputs(**overrides) -> dict:
    """Example A of the concept: a median, entirely normal day."""
    history = _vix1d_history()
    vix1d = _vix1d_at_rank(45.0, history)
    base = {
        "vix1d_open": vix1d,
        "vix1d_history": history,
        "vix_open": vix1d / 0.59,  # term structure at the sample median
        "vix_prev_close": vix1d / 0.59 / 1.005,  # ROC ≈ +0.5%
        "spx_closes": _spx_closes(),
        "spx_price": 5000.0,
        "zero_gamma_level": 4970.0,  # well above -> price above ZGL
        "zgl_source": "test-provider/v1",
    }
    base.update(overrides)
    return base


def _run(**overrides):
    return detect_volatility_regime(**_inputs(**overrides))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class TestHelpers:
    def test_percentile_rank_counts_strictly_below(self):
        history = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert _percentile_rank(history, 3.0) == 50.0
        assert _percentile_rank(history, 0.5) == 0.0
        assert _percentile_rank(history, 9.0) == 100.0

    def test_realised_vol_matches_annualised_std(self):
        closes = _spx_closes(n=21, daily_vol=0.006)
        rv = _realised_vol(closes, 20)
        expected = 0.006 * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
        # ddof=1 on perfectly alternating returns inflates std by sqrt(n/(n-1)).
        assert rv == pytest.approx(expected * math.sqrt(20 / 19), rel=1e-6)

    def test_realised_vol_requires_full_window(self):
        assert _realised_vol(_spx_closes(n=15), 20) is None

    def test_realised_vol_ignores_non_positive_and_nan(self):
        closes = pd.Series([float("nan"), -1.0, *_spx_closes(n=21).tolist()])
        assert _realised_vol(closes, 20) is not None

    def test_expected_move_pct(self):
        assert _expected_move_pct(16.0) == pytest.approx(16.0 / math.sqrt(TRADING_DAYS_PER_YEAR))
        assert _expected_move_pct(None) is None


class TestConfigValidation:
    def test_rejects_inverted_roc_thresholds(self):
        with pytest.raises(ValueError, match="roc_soft_pct"):
            VolatilityRegimeConfig(roc_soft_pct=20.0, roc_hard_pct=15.0)

    def test_rejects_inverted_premium_thresholds(self):
        with pytest.raises(ValueError, match="premium_hard"):
            VolatilityRegimeConfig(premium_soft=-10.0, premium_hard=-6.0)

    def test_rejects_inverted_term_thresholds(self):
        with pytest.raises(ValueError, match="term_soft"):
            VolatilityRegimeConfig(term_soft=1.1, term_hard=1.0)

    def test_rejects_bad_windows(self):
        with pytest.raises(ValueError, match="base_window"):
            VolatilityRegimeConfig(base_window=0)

    def test_rejects_rv_window_too_small_for_a_std(self):
        # One return cannot yield a sample std, which would make RV20 forever
        # unavailable and premium_richness a standing hard flag.
        with pytest.raises(ValueError, match="rv_window"):
            VolatilityRegimeConfig(rv_window=1)

    def test_rejects_bad_base_rank_order(self):
        with pytest.raises(ValueError, match="base rank"):
            VolatilityRegimeConfig(base_rank_soft=99.0, base_rank_hard=95.0)

    def test_rejects_negative_gex_buffer(self):
        with pytest.raises(ValueError, match="gex_buffer_em"):
            VolatilityRegimeConfig(gex_buffer_em=-0.1)


# ----------------------------------------------------------------------
# Section 4 examples
# ----------------------------------------------------------------------


class TestConceptExamples:
    def test_a_ordinary_day_is_favorable(self):
        result = _run()
        assert result.favorable is True
        assert result.flags == ()
        assert result.base_regime == "NORMAL"
        assert result.zgl_source == "test-provider/v1"

    def test_b_low_volatility_does_not_block(self):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(6.0, history)
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 0.59,
            vix_prev_close=vix1d / 0.59,
        )
        assert result.base_regime == "LOW"
        assert result.favorable is True
        assert result.hard_count == 0
        assert result.soft_count == 0

    def test_c_post_cpi_morning_is_blocked(self):
        history = _vix1d_history()
        vix1d = 26.0  # absolute hard cut-off, and rank far above 98
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 1.12,  # term structure > 1.00 -> hard
            vix_prev_close=vix1d / 1.12 / 1.18,  # ROC ≈ +18% -> hard
            spx_price=5000.0,
            zero_gamma_level=5300.0,  # price far below ZGL -> hard
        )
        assert result.favorable is False
        assert result.base_regime == "EXTREME"
        triggered = {f.metric: f.severity for f in result.flags}
        assert triggered["base_level"] == "hard"
        assert triggered["vix1d_absolute"] == "hard"
        assert triggered["vix_roc"] == "hard"
        assert triggered["term_structure"] == "hard"
        assert triggered["gex"] == "hard"
        assert not result.missing_metrics

    def test_d_two_soft_flags_block(self):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(93.0, history)  # soft
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 0.59,
            vix_prev_close=vix1d / 0.59 / 1.12,  # ROC ≈ +12% -> soft
        )
        assert result.hard_count == 0
        assert result.soft_count == 2
        assert result.favorable is False
        assert {f.metric for f in result.flags} == {"base_level", "vix_roc"}

    def test_e_single_soft_flag_is_tolerated(self):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(40.0, history)
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 0.91,  # term structure 0.91 -> soft
            vix_prev_close=vix1d / 0.91,
        )
        assert result.soft_count == 1
        assert result.hard_count == 0
        assert result.favorable is True

    def test_f_negative_gamma_alone_blocks(self):
        result = _run(spx_price=5000.0, zero_gamma_level=5050.0)
        assert result.hard_count == 1
        assert result.favorable is False
        gex = next(f for f in result.flags if f.metric == "gex")
        assert gex.severity == "hard"
        assert gex.missing is False

    def test_g_missing_zgl_is_a_hard_missing_data_flag(self):
        result = _run(zero_gamma_level=None)
        assert result.favorable is False
        assert result.missing_metrics == ("gex",)
        assert result.hard_count == 1


# ----------------------------------------------------------------------
# Individual metrics
# ----------------------------------------------------------------------


class TestBaseLevel:
    def test_high_band_is_soft_not_hard(self):
        history = _vix1d_history()
        result = _run(
            vix1d_history=history,
            vix1d_open=_vix1d_at_rank(95.0, history),
            vix_open=_vix1d_at_rank(95.0, history) / 0.59,
            vix_prev_close=_vix1d_at_rank(95.0, history) / 0.59,
        )
        flag = next(f for f in result.flags if f.metric == "base_level")
        assert flag.severity == "soft"
        assert result.base_regime == "HIGH"
        assert result.favorable is True  # a single soft flag is allowed

    def test_short_history_falls_back_to_no_base_level(self):
        result = _run(vix1d_history=_vix1d_history(n=30), vix_history=None)
        assert result.reason == NO_BASE_LEVEL
        assert result.favorable is False
        assert result.flags == ()  # no per-metric evaluation happened

    def test_vix_fallback_marks_base_as_degraded(self):
        vix_history = pd.Series(np.linspace(12.0, 24.0, 60), dtype="float64")
        result = _run(
            vix1d_open=None,
            vix1d_history=None,
            vix_history=vix_history,
            vix_open=15.0,
            vix_prev_close=15.0,
        )
        assert result.degraded_base is True
        assert result.base_rank is not None
        # Everything downstream of VIX1D is unavailable -> missing hard flags.
        assert set(result.missing_metrics) == {
            "vix1d_absolute",
            "term_structure",
            "premium_richness",
            "gex",
        }
        assert result.favorable is False

    def test_only_last_window_observations_are_used(self):
        history = pd.Series([100.0] * 60 + list(np.linspace(8.0, 18.0, 60)), dtype="float64")
        config = VolatilityRegimeConfig()
        result = detect_volatility_regime(**_inputs(vix1d_history=history), config=config)
        # The stale 100.0 block is outside the 60-day window, so a normal VIX1D
        # is not pushed to a LOW rank by ancient outliers.
        assert result.base_regime == "NORMAL"


class TestAbsoluteVix1d:
    def test_above_25_is_hard(self):
        history = pd.Series(np.linspace(20.0, 40.0, 60), dtype="float64")
        result = _run(
            vix1d_history=history,
            vix1d_open=26.0,
            vix_open=26.0 / 0.59,
            vix_prev_close=26.0 / 0.59,
        )
        flag = next(f for f in result.flags if f.metric == "vix1d_absolute")
        assert flag.severity == "hard"

    def test_at_threshold_does_not_flag(self):
        history = pd.Series(np.linspace(20.0, 40.0, 60), dtype="float64")
        result = _run(
            vix1d_history=history,
            vix1d_open=25.0,
            vix_open=25.0 / 0.59,
            vix_prev_close=25.0 / 0.59,
        )
        assert "vix1d_absolute" not in {f.metric for f in result.flags}


class TestVixRoc:
    @pytest.mark.parametrize(
        ("roc_pct", "expected"),
        # Boundaries are probed just inside each band: an exact hit on a
        # threshold is not reproducible in floating point and has no meaning for
        # market data.
        [(0.5, None), (9.9, None), (12.0, "soft"), (14.9, "soft"), (18.0, "hard")],
    )
    def test_bands(self, roc_pct: float, expected: str | None):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(45.0, history)
        vix_open = vix1d / 0.59
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix_open,
            vix_prev_close=vix_open / (1.0 + roc_pct / 100.0),
        )
        flags = {f.metric: f.severity for f in result.flags}
        assert flags.get("vix_roc") == expected

    def test_vix_compression_is_not_flagged(self):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(45.0, history)
        vix_open = vix1d / 0.59
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix_open,
            vix_prev_close=vix_open * 1.30,  # ROC = -23%, relief not repricing
        )
        assert "vix_roc" not in {f.metric for f in result.flags}


class TestTermStructure:
    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.59, None), (0.85, None), (0.91, "soft"), (1.00, "soft"), (1.12, "hard")],
    )
    def test_bands(self, ratio: float, expected: str | None):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(45.0, history)
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / ratio,
            vix_prev_close=vix1d / ratio,
        )
        flags = {f.metric: f.severity for f in result.flags}
        assert flags.get("term_structure") == expected


class TestPremiumRichness:
    @pytest.mark.parametrize(
        ("spread", "expected"),
        [(-1.9, None), (-5.9, None), (-8.0, "soft"), (-9.9, "soft"), (-12.0, "hard")],
    )
    def test_bands(self, spread: float, expected: str | None):
        history = _vix1d_history(low=4.0, high=30.0)
        vix1d = _vix1d_at_rank(45.0, history)
        # Choose the daily vol so that VIX1D - RV20 lands on *spread*.
        target_rv = vix1d - spread
        daily_vol = target_rv / 100.0 / math.sqrt(TRADING_DAYS_PER_YEAR) / math.sqrt(20 / 19)
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 0.59,
            vix_prev_close=vix1d / 0.59,
            spx_closes=_spx_closes(daily_vol=daily_vol),
        )
        flags = {f.metric: f.severity for f in result.flags}
        assert flags.get("premium_richness") == expected

    def test_missing_closes_is_a_missing_hard_flag(self):
        result = _run(spx_closes=None)
        assert result.missing_metrics == ("premium_richness",)
        assert result.hard_count == 1
        assert result.favorable is False


class TestGex:
    @pytest.mark.parametrize(
        ("dist_em", "expected"),
        [(1.0, None), (0.26, None), (0.24, "soft"), (0.0, "soft"), (-0.24, "soft"), (-0.5, "hard")],
    )
    def test_bands(self, dist_em: float, expected: str | None):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(45.0, history)
        spot = 5000.0
        em_pct = vix1d / math.sqrt(TRADING_DAYS_PER_YEAR)
        zgl = spot * (1.0 - dist_em * em_pct / 100.0)
        result = _run(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 0.59,
            vix_prev_close=vix1d / 0.59,
            spx_price=spot,
            zero_gamma_level=zgl,
        )
        flags = {f.metric: f.severity for f in result.flags}
        assert flags.get("gex") == expected

    def test_buffer_width_is_configurable(self):
        history = _vix1d_history()
        vix1d = _vix1d_at_rank(45.0, history)
        spot = 5000.0
        em_pct = vix1d / math.sqrt(TRADING_DAYS_PER_YEAR)
        zgl = spot * (1.0 - 0.40 * em_pct / 100.0)  # price 0.40 EM above ZGL
        kwargs = _inputs(
            vix1d_history=history,
            vix1d_open=vix1d,
            vix_open=vix1d / 0.59,
            vix_prev_close=vix1d / 0.59,
            spx_price=spot,
            zero_gamma_level=zgl,
        )
        assert detect_volatility_regime(**kwargs).favorable is True
        wide = detect_volatility_regime(**kwargs, config=VolatilityRegimeConfig(gex_buffer_em=0.5))
        assert {f.metric: f.severity for f in wide.flags}.get("gex") == "soft"


class TestFailSafe:
    def test_all_data_missing_yields_no_base_level(self):
        result = detect_volatility_regime(
            vix1d_open=None,
            vix1d_history=None,
            vix_open=None,
            vix_prev_close=None,
            spx_closes=None,
            spx_price=None,
            zero_gamma_level=None,
        )
        assert result.favorable is False
        assert result.reason == NO_BASE_LEVEL

    def test_missing_metric_is_hard_not_soft(self):
        result = _run(vix_prev_close=None)
        flag = next(f for f in result.flags if f.metric == "vix_roc")
        assert flag.severity == "hard"
        assert flag.missing is True
        assert "missing_data" in str(flag)

    def test_non_positive_inputs_are_treated_as_missing(self):
        result = _run(spx_price=0.0)
        assert result.missing_metrics == ("gex",)

    def test_nan_inputs_are_treated_as_missing(self):
        result = _run(zero_gamma_level=float("nan"))
        assert result.missing_metrics == ("gex",)

    def test_metrics_dict_is_populated_for_logging(self):
        result = _run()
        assert result.metrics["base_rank"] == pytest.approx(result.base_rank)
        assert result.metrics["term_structure"] == pytest.approx(0.59, abs=0.01)
        assert result.metrics["zgl_distance_em"] > 0

    def test_summary_is_single_line(self):
        result = _run(zero_gamma_level=None)
        summary = result.summary()
        assert "\n" not in summary
        assert "SKIP" in summary
