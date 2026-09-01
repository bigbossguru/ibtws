"""Unit tests for ibtws.unofficial.analysis.volatility_regime.

Scenario tests mirror the worked examples A-L of section 9 of
``VOLATILITY_REGIME_CONCEPT.md`` (version 5).

The key difference from the version-2 suite: it asserted that a high volatility
level *blocks* an entry. These assert the opposite — that it does not block but
trims the position size — because ``IC(r_level, P&L) = +0.366`` means the old
rule was removing the best-paid entries (concept 14.4).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ibtws.unofficial.analysis.volatility_regime import (
    BUCKET_VARIANCE_SHARE,
    FULL_SESSION_BUCKETS,
    NO_VIX1D,
    OPENING_WINDOW,
    OUTSIDE_RTH,
    RTH_BUCKETS,
    SESSION_MINUTES,
    SHORT_SESSION,
    TRADING_DAYS_PER_YEAR,
    VolatilityRegimeConfig,
    _realised_vol,
    bucket_of,
    detect_volatility_regime,
    expected_move_pct,
    legacy_v4_config,
    percentile_rank,
    remaining_variance_share,
)

# ----------------------------------------------------------------------
# Fixtures / builders
# ----------------------------------------------------------------------


def _history(n: int = 60, low: float = 8.0, high: float = 18.0) -> pd.Series:
    """Evenly spread readings so that a target percentile rank is easy to hit."""
    return pd.Series(np.linspace(low, high, n), dtype="float64")


def _at_rank(rank: float, history: pd.Series) -> float:
    """A value whose percentile rank inside *history* is ~*rank*."""
    value = float(np.quantile(history.to_numpy(), rank / 100.0)) + 1e-9
    assert abs(percentile_rank(history, value) - rank) <= 2.0
    return value


def _spx_closes(n: int = 21, daily_vol: float = 0.006, start: float = 5000.0) -> pd.Series:
    """Deterministic closes with a known close-to-close volatility.

    Alternating ±*daily_vol* log returns give a sample std of exactly *daily_vol*
    (up to the ddof correction), so RV20 is predictable.
    """
    logs = [math.log(start)]
    for i in range(n - 1):
        logs.append(logs[-1] + (daily_vol if i % 2 == 0 else -daily_vol))
    return pd.Series(np.exp(logs), dtype="float64")


def _inputs(**overrides) -> dict:
    """Example A of the concept: an ordinary midday bar, everything normal."""
    level_history = _history()
    vix1d = _at_rank(45.0, level_history)
    base = {
        "bucket": "12:30",
        "minutes_left": 225.0,
        "minutes_since_open": 180.0,
        "vix1d": vix1d,
        "vix1d_bucket_history": level_history,
        "vix": vix1d / 0.65,  # term structure at the sample median for midday
        "term_structure_history": _history(low=0.50, high=0.80),
        "realized_range_pct": 0.60,
        "realized_vs_em_history": _history(low=0.3, high=3.0),
        "spx_closes": _spx_closes(),
        "premium_spread_history": pd.Series(np.linspace(-6.0, 2.0, 60), dtype="float64"),
        "spx_price": 5000.0,
        "zgl_source": "test-provider/v1",
    }
    base.update(overrides)
    return base


def _run(**overrides):
    return detect_volatility_regime(**_inputs(**overrides))


def _flag(result, metric):
    return next((f for f in result.flags if f.metric == metric), None)


# ----------------------------------------------------------------------
# Configuration validation
# ----------------------------------------------------------------------


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bucket_lookback": 0},
            {"rv_window": 1},
            {"rank_soft": 99.0, "rank_hard": 95.0},
            {"rank_hard_low": 20.0, "rank_soft_low": 10.0},
            {"vix1d_absolute_soft": 40.0, "vix1d_absolute_hard": 35.0},
            {"vix1d_absolute_soft": 0.0},
            {"vix_absolute_hard": -1.0},
            {"gex_buffer_em": -0.1},
            {"max_soft_flags": -1},
            {"skip_first_minutes": -1.0},
            {"roll_degraded_below": 1.5},
        ],
    )
    def test_rejects_inconsistent_configuration(self, kwargs):
        with pytest.raises(ValueError):
            VolatilityRegimeConfig(**kwargs)

    def test_defaults_are_valid(self):
        assert VolatilityRegimeConfig().gate_on_realized is True

    def test_legacy_config_gates_on_every_metric(self):
        legacy = legacy_v4_config()
        assert legacy.gate_on_base_level
        assert legacy.gate_on_term_structure
        assert legacy.gate_on_premium_richness
        assert legacy.skip_first_minutes == 0.0
        assert legacy.use_variance_profile is False


# ----------------------------------------------------------------------
# Session clock and the intraday variance profile
# ----------------------------------------------------------------------


class TestSessionClock:
    def test_bucket_labels_cover_rth(self):
        assert len(RTH_BUCKETS) == FULL_SESSION_BUCKETS == 26
        assert RTH_BUCKETS[0] == "09:30"
        assert RTH_BUCKETS[-1] == "15:45"

    @pytest.mark.parametrize(
        ("stamp", "expected"),
        [("2026-08-31 12:37", "12:30"), ("2026-08-31 09:30", "09:30"), ("2026-08-31 15:44", "15:30")],
    )
    def test_bucket_of_floors_to_the_quarter_hour(self, stamp, expected):
        assert bucket_of(pd.Timestamp(stamp)) == expected

    def test_variance_shares_are_normalised(self):
        assert sum(BUCKET_VARIANCE_SHARE.values()) == pytest.approx(1.0, abs=1e-4)

    def test_range_is_front_loaded(self):
        """The opening bucket carries ~1.97x the uniform share (concept 14.7)."""
        uniform = 1.0 / FULL_SESSION_BUCKETS
        assert BUCKET_VARIANCE_SHARE["09:30"] / uniform > 1.9
        assert BUCKET_VARIANCE_SHARE["14:30"] < BUCKET_VARIANCE_SHARE["09:30"]

    def test_remaining_share_decreases_monotonically(self):
        shares = [remaining_variance_share(b) for b in RTH_BUCKETS]
        assert shares == sorted(shares, reverse=True)
        assert shares[0] == pytest.approx(1.0, abs=1e-4)

    def test_unknown_bucket_has_no_remaining_share(self):
        assert remaining_variance_share("08:00") == 0.0

    def test_profile_undercuts_sqrt_of_time_in_the_afternoon(self):
        """√t overstates residual risk late in the day (concept 15.5)."""
        profile = expected_move_pct(12.0, 105.0, "14:30", use_profile=True)
        sqrt_t = expected_move_pct(12.0, 105.0, "14:30", use_profile=False)
        assert profile < sqrt_t

    def test_expected_move_matches_full_day_at_the_open(self):
        assert expected_move_pct(11.4, SESSION_MINUTES, "09:30") == pytest.approx(
            11.4 / math.sqrt(TRADING_DAYS_PER_YEAR), rel=1e-6
        )

    def test_expected_move_shrinks_through_the_session(self):
        moves = [expected_move_pct(11.4, m, b) for b, m in (("09:30", 405.0), ("12:30", 225.0), ("15:30", 45.0))]
        assert moves == sorted(moves, reverse=True)

    def test_expected_move_requires_a_level(self):
        assert expected_move_pct(None, 225.0, "12:30") is None


# ----------------------------------------------------------------------
# Unconditional refusals
# ----------------------------------------------------------------------


class TestRefusals:
    def test_outside_rth_is_not_assessed(self):
        result = _run(bucket="08:00")
        assert result.reason == OUTSIDE_RTH
        assert result.favorable is False
        assert result.flags == ()

    def test_half_day_is_not_assessed(self):
        """Bucket alignment breaks on a shortened session (concept 6.5)."""
        result = _run(session_buckets=14)
        assert result.reason == SHORT_SESSION
        assert result.favorable is False

    def test_full_session_is_assessed(self):
        assert _run(session_buckets=FULL_SESSION_BUCKETS).reason != SHORT_SESSION

    def test_opening_window_is_skipped(self):
        result = _run(bucket="09:30", minutes_left=405.0, minutes_since_open=5.0)
        assert result.reason == OPENING_WINDOW
        assert result.favorable is False

    def test_entry_allowed_once_the_opening_window_has_passed(self):
        assert _run(bucket="10:15", minutes_left=360.0, minutes_since_open=46.0).reason != OPENING_WINDOW

    def test_missing_vix1d_stops_the_evaluation(self):
        """No VIX1D means no expected move, hence no width and nothing to size."""
        result = _run(vix1d=None)
        assert result.reason == NO_VIX1D
        assert result.favorable is False

    def test_legacy_config_does_not_skip_the_opening_window(self):
        result = _run(bucket="09:30", minutes_left=405.0, minutes_since_open=5.0, config=legacy_v4_config())
        assert result.reason != OPENING_WINDOW


# ----------------------------------------------------------------------
# What gates and what does not
# ----------------------------------------------------------------------


class TestGatingMetric:
    """``realized_vs_em`` is the only ranked metric that blocks (concept 15.1)."""

    def test_normal_bar_is_favorable(self):
        result = _run()
        assert result.hard_count == 0
        assert result.soft_count == 0
        assert result.favorable is True

    def test_extreme_realized_range_blocks(self):
        """Example C: calm level, but the market has already run."""
        history = _history(low=0.3, high=3.0)
        result = _run(realized_range_pct=8.0, realized_vs_em_history=history)
        flag = _flag(result, "realized_vs_em")
        assert flag is not None
        assert flag.severity == "hard"
        assert result.favorable is False

    def test_elevated_realized_range_is_soft(self):
        history = _history(n=60, low=0.3, high=1.0)
        vix1d = _at_rank(45.0, _history())
        em = expected_move_pct(vix1d, 225.0, "12:30")
        # Land just inside the p90-p98 band.
        target = float(np.quantile(history.to_numpy(), 0.94))
        result = _run(realized_range_pct=target * em, realized_vs_em_history=history)
        flag = _flag(result, "realized_vs_em")
        assert flag is not None
        assert flag.severity == "soft"
        assert result.favorable is True  # one soft flag is allowed

    def test_missing_gating_history_fails_safe(self):
        """Absent data is not evidence of a calm regime (concept 6.4)."""
        result = _run(realized_vs_em_history=_history(n=30))
        flag = _flag(result, "realized_vs_em")
        assert flag is not None
        assert flag.missing is True
        assert flag.severity == "hard"
        assert result.favorable is False
        assert "30/60" in flag.detail


class TestNonGatingMetrics:
    """The level metrics report and size; they never block (concept 15.1)."""

    @pytest.mark.parametrize("metric", ["base_level", "term_structure", "premium_richness"])
    def test_extreme_level_metric_does_not_block(self, metric):
        overrides: dict = {}
        if metric == "base_level":
            history = _history()
            overrides = {"vix1d": _at_rank(99.5, history) + 5.0, "vix1d_bucket_history": history}
        elif metric == "term_structure":
            history = _history(low=0.50, high=0.80)
            overrides = {"vix": _at_rank(45.0, _history()) / 0.95, "term_structure_history": history}
        else:
            overrides = {"premium_spread_history": pd.Series(np.linspace(3.0, 12.0, 60), dtype="float64")}
        result = _run(**overrides)
        flag = _flag(result, metric)
        assert flag is not None
        assert flag.severity == "info"
        assert result.hard_count == 0
        assert result.favorable is True

    def test_three_extreme_level_metrics_still_favorable(self):
        history = _history()
        result = _run(
            vix1d=_at_rank(99.5, history) + 5.0,
            vix1d_bucket_history=history,
            premium_spread_history=pd.Series(np.linspace(3.0, 12.0, 60), dtype="float64"),
        )
        assert result.hard_count == 0
        assert result.favorable is True
        assert len(result.info_flags) >= 2

    def test_missing_non_gating_metric_does_not_block(self):
        """Blocking on a metric that has no right to block would be incoherent."""
        result = _run(vix1d_bucket_history=_history(n=10))
        flag = _flag(result, "base_level")
        assert flag is not None
        assert flag.severity == "info"
        assert result.hard_count == 0
        assert result.favorable is True

    @pytest.mark.parametrize(
        ("flag_name", "config_field"),
        [
            ("base_level", "gate_on_base_level"),
            ("term_structure", "gate_on_term_structure"),
            ("premium_richness", "gate_on_premium_richness"),
        ],
    )
    def test_gating_is_reversible_through_config(self, flag_name, config_field):
        history = _history()
        overrides: dict = {"config": VolatilityRegimeConfig(**{config_field: True})}
        if flag_name == "base_level":
            overrides |= {"vix1d": _at_rank(99.5, history) + 5.0, "vix1d_bucket_history": history}
        elif flag_name == "term_structure":
            overrides |= {"vix": _at_rank(45.0, history) / 0.95, "term_structure_history": _history(low=0.5, high=0.8)}
        else:
            overrides |= {"premium_spread_history": pd.Series(np.linspace(3.0, 12.0, 60), dtype="float64")}
        result = _run(**overrides)
        flag = _flag(result, flag_name)
        assert flag is not None
        assert flag.severity == "hard"
        assert result.favorable is False


# ----------------------------------------------------------------------
# Absolute cut-offs
# ----------------------------------------------------------------------


class TestAbsoluteCutoffs:
    """Not bucket-scaled: being absolute is the point (concept 15.6)."""

    def test_moderate_absolute_level_is_soft(self):
        result = _run(vix1d=26.0, vix1d_bucket_history=_history(low=20.0, high=30.0))
        flag = _flag(result, "vix1d_absolute")
        assert flag is not None
        assert flag.severity == "soft"

    def test_extreme_absolute_level_is_hard(self):
        result = _run(vix1d=36.0, vix1d_bucket_history=_history(low=30.0, high=40.0))
        flag = _flag(result, "vix1d_absolute")
        assert flag is not None
        assert flag.severity == "hard"
        assert result.favorable is False

    def test_below_the_soft_threshold_nothing_fires(self):
        assert _flag(_run(), "vix1d_absolute") is None

    @pytest.mark.parametrize("bucket", ["10:30", "15:30"])
    def test_absolute_cutoff_is_not_bucket_scaled(self, bucket):
        minutes = {"10:30": 345.0, "15:30": 45.0}[bucket]
        result = _run(
            bucket=bucket,
            minutes_left=minutes,
            minutes_since_open=SESSION_MINUTES - minutes,
            vix1d=36.0,
            vix1d_bucket_history=_history(low=30.0, high=40.0),
        )
        flag = _flag(result, "vix1d_absolute")
        assert flag is not None
        assert flag.severity == "hard"

    def test_vix_regime_shift_guard(self):
        result = _run(vix=31.0)
        flag = _flag(result, "vix_absolute")
        assert flag is not None
        assert flag.severity == "hard"
        assert result.favorable is False

    def test_vix_below_the_guard_does_not_fire(self):
        assert _flag(_run(vix=29.0), "vix_absolute") is None


# ----------------------------------------------------------------------
# GEX — opt-in
# ----------------------------------------------------------------------


class TestGex:
    def test_absent_zgl_is_a_degradation_not_a_flag(self):
        """Version 2 turned "no ZGL" into "never trade"; version 5 does not."""
        result = _run()
        assert _flag(result, "gex") is None
        assert "gex_not_evaluated" in result.degraded
        assert result.favorable is True

    def test_price_above_zgl_is_clean(self):
        result = _run(zero_gamma_level=4900.0)
        assert _flag(result, "gex") is None
        assert result.favorable is True

    def test_price_below_zgl_is_hard(self):
        result = _run(zero_gamma_level=5100.0)
        flag = _flag(result, "gex")
        assert flag is not None
        assert flag.severity == "hard"

    def test_price_inside_the_buffer_is_soft(self):
        em_pct = expected_move_pct(_at_rank(45.0, _history()), 225.0, "12:30")
        # 0.1 EM above the level: inside the ±0.25 EM unstable-sign zone.
        zgl = 5000.0 * (1 - 0.1 * em_pct / 100.0)
        result = _run(zero_gamma_level=zgl)
        flag = _flag(result, "gex")
        assert flag is not None
        assert flag.severity == "soft"

    def test_supplied_zgl_without_spot_fails_safe(self):
        result = _run(zero_gamma_level=4900.0, spx_price=None)
        flag = _flag(result, "gex")
        assert flag is not None
        assert flag.missing is True
        assert flag.severity == "hard"


# ----------------------------------------------------------------------
# Realised volatility helper
# ----------------------------------------------------------------------


class TestRealisedVol:
    def test_matches_the_known_construction(self):
        closes = _spx_closes(n=21, daily_vol=0.006)
        rv = _realised_vol(closes, 20)
        assert rv == pytest.approx(0.006 * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0, rel=0.05)

    def test_short_history_is_unavailable(self):
        assert _realised_vol(_spx_closes(n=10), 20) is None

    def test_no_history_is_unavailable(self):
        assert _realised_vol(None, 20) is None

    def test_non_positive_closes_are_rejected(self):
        closes = _spx_closes(n=21)
        closes.iloc[5] = -1.0
        assert _realised_vol(closes, 20) is None


# ----------------------------------------------------------------------
# Reporting surface
# ----------------------------------------------------------------------


class TestReporting:
    def test_summary_reports_the_verdict(self):
        assert "SKIP" in _run(vix=31.0).summary()
        assert "FAVORABLE" in _run().summary()

    def test_metrics_carry_every_input_and_derived_value(self):
        metrics = _run().metrics
        for key in (
            "vix1d",
            "vix",
            "expected_move_pct",
            "base_rank",
            "term_structure",
            "realized_vs_em",
            "rv20",
            "premium_spread",
            "variance_share_left",
        ):
            assert key in metrics

    def test_zgl_source_is_recorded(self):
        assert _run().zgl_source == "test-provider/v1"

    def test_expected_move_is_exposed_for_the_risk_layer(self):
        """Sizing lives with the caller, but it needs this scale (concept 15.2)."""
        result = _run()
        assert result.expected_move_pct is not None
        assert result.expected_move_pct == pytest.approx(result.metrics["expected_move_pct"])

    def test_no_sizing_surface_is_exposed(self):
        """Position sizing is out of scope; the result must not imply otherwise."""
        result = _run()
        for attribute in ("sizing", "size_ok", "tradeable", "contracts"):
            assert not hasattr(result, attribute)

    def test_elevated_level_is_reported_as_degraded(self):
        history = _history()
        result = _run(vix1d=_at_rank(94.0, history), vix1d_bucket_history=history)
        assert "base_level_elevated" in result.degraded

    def test_missing_metrics_are_listed(self):
        result = _run(realized_vs_em_history=_history(n=30))
        assert "realized_vs_em" in result.missing_metrics
