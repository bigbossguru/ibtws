"""Unit tests for ibtws.unofficial.analysis.volatility_risk."""

from __future__ import annotations

import pandas as pd
import pytest

from ibtws.unofficial.analysis.volatility_risk import (
    MAX_POSSIBLE_SCORE,
    _lookup_bracket,
    common_volatility_risk,
)


def _vix_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


def _calm_series(n: int = 21, level: float = 14.0) -> pd.Series:
    """A flat, calm VIX history of *n* points."""
    return _vix_series([level] * n)


class TestLookupBracket:
    def test_returns_first_matching_bracket(self):
        brackets = [(0.5, "a"), (1.0, "b"), (float("inf"), "c")]
        assert _lookup_bracket(0.4, brackets) == (0.5, "a")
        assert _lookup_bracket(0.7, brackets) == (1.0, "b")

    def test_returns_last_bracket_when_above_all(self):
        brackets = [(0.5, "a"), (1.0, "b"), (float("inf"), "c")]
        assert _lookup_bracket(999.0, brackets) == (float("inf"), "c")


class TestGuards:
    def test_raises_when_series_too_short(self):
        with pytest.raises(ValueError, match="need at least"):
            common_volatility_risk(_vix_series([14.0] * 10), vx1d_current=13.0, lookback_days=20)


class TestScoring:
    def test_calm_market_is_trade(self):
        result = common_volatility_risk(
            _calm_series(),
            vx1d_current=8.0,  # ratio ~0.57 -> very calm
            vix3m_current=16.0,  # contango -> low term score
            risk_threshold=50,
        )
        assert result["decision"] == "TRADE"
        assert result["risk_score"] < 50
        assert set(result["component_scores"]) == {
            "vix_deviation",
            "vx_ratio",
            "vix_level",
            "term_structure",
        }

    def test_stressed_market_is_no_trade(self):
        # Sharp spike on the last day: high level, rising momentum, inversion.
        values = [14.0] * 20 + [35.0]
        result = common_volatility_risk(
            _vix_series(values),
            vx1d_current=42.0,  # ratio > 1.05 -> extreme intraday stress
            vix3m_current=28.0,  # backwardation/inversion
            risk_threshold=50,
        )
        assert result["decision"] == "NO TRADE"
        assert result["risk_score"] >= 50

    def test_term_structure_skipped_when_vix3m_none(self):
        result = common_volatility_risk(
            _calm_series(),
            vx1d_current=8.0,
            vix3m_current=None,
        )
        assert result["component_scores"]["term_structure"] == 0
        assert "N/A" in result["overall_structure"]

    def test_score_never_exceeds_max(self):
        values = [14.0] * 20 + [40.0]
        result = common_volatility_risk(
            _vix_series(values),
            vx1d_current=50.0,
            vix3m_current=25.0,
        )
        assert result["risk_score"] <= MAX_POSSIBLE_SCORE

    def test_debug_includes_metrics(self):
        result = common_volatility_risk(
            _calm_series(),
            vx1d_current=8.0,
            vix3m_current=16.0,
            debug=True,
        )
        assert "metrics" in result
        assert result["metrics"]["max_possible"] == MAX_POSSIBLE_SCORE
        assert "vix_z_score" in result["metrics"]

    def test_no_metrics_without_debug(self):
        result = common_volatility_risk(_calm_series(), vx1d_current=8.0, vix3m_current=16.0)
        assert "metrics" not in result
