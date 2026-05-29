"""Unit tests for market_bias pure scoring functions."""

from typing import Literal

import numpy as np
import pytest

from ibtws.unofficial.analysis.market_bias import (
    Bias,
    Structure,
    classify_structure,
    compute_bias,
    compute_signal,
    detect_market_bias,
    detect_swing_points,
    score_gap_vwap,
    score_structure,
    score_vol_regime,
    _neutral_fallback,
)
from ibtws.unofficial.analysis.volatility import VolRegimeResult


def _make_vol_regime(
    regime: Literal["GREEN", "YELLOW", "RED"] = "GREEN",
    signal: Literal["TRADE", "NOTRADE"] = "TRADE",
) -> VolRegimeResult:
    """Helper to build a minimal VolRegimeResult for testing."""
    return VolRegimeResult(
        score=50,
        regime=regime,
        trade=signal == "TRADE",
        signal=signal,
        action="test",
    )


# ---------------------------------------------------------------------------
# detect_swing_points
# ---------------------------------------------------------------------------


class TestDetectSwingPoints:
    def test_basic_swing_detection(self):
        highs = np.array([1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1], dtype=float)
        lows = np.array([10, 9, 8, 7, 6, 1, 6, 7, 8, 9, 10], dtype=float)
        sh, sl = detect_swing_points(highs, lows, pivot_len=5)
        assert 10.0 in sh
        assert 1.0 in sl

    def test_insufficient_data(self):
        highs = np.array([1, 2, 3], dtype=float)
        lows = np.array([1, 2, 3], dtype=float)
        sh, sl = detect_swing_points(highs, lows, pivot_len=5)
        assert len(sh) == 0
        assert len(sl) == 0

    def test_flat_data(self):
        highs = np.ones(20)
        lows = np.ones(20)
        sh, sl = detect_swing_points(highs, lows, pivot_len=5)
        assert len(sh) > 0
        assert len(sl) > 0


# ---------------------------------------------------------------------------
# classify_structure
# ---------------------------------------------------------------------------


class TestClassifyStructure:
    def test_bullish_hh_hl(self):
        highs = np.array([100, 105])
        lows = np.array([90, 92])
        structure, inv = classify_structure(highs, lows)
        assert structure == Structure.BULLISH
        assert inv == 92.0

    def test_bearish_lh_ll(self):
        highs = np.array([105, 100])
        lows = np.array([92, 88])
        structure, inv = classify_structure(highs, lows)
        assert structure == Structure.BEARISH
        assert inv == 100.0

    def test_neutral_mixed(self):
        highs = np.array([100, 105])
        lows = np.array([92, 88])
        structure, inv = classify_structure(highs, lows)
        assert structure == Structure.NEUTRAL
        assert inv is None

    def test_insufficient_points(self):
        highs = np.array([100])
        lows = np.array([90])
        structure, inv = classify_structure(highs, lows)
        assert structure == Structure.NEUTRAL
        assert inv is None


# ---------------------------------------------------------------------------
# score_structure
# ---------------------------------------------------------------------------


class TestScoreStructure:
    def test_bullish(self):
        assert score_structure(Structure.BULLISH) == 2

    def test_bearish(self):
        assert score_structure(Structure.BEARISH) == -2

    def test_neutral(self):
        assert score_structure(Structure.NEUTRAL) == 0


# ---------------------------------------------------------------------------
# score_vol_regime
# ---------------------------------------------------------------------------


class TestScoreVolRegime:
    def test_green(self):
        assert score_vol_regime(_make_vol_regime("GREEN")) == 1

    def test_red(self):
        assert score_vol_regime(_make_vol_regime("RED")) == -1

    def test_yellow(self):
        assert score_vol_regime(_make_vol_regime("YELLOW")) == 0


# ---------------------------------------------------------------------------
# score_gap_vwap
# ---------------------------------------------------------------------------


class TestScoreGapVwap:
    def test_gap_up_above_vwap(self):
        assert score_gap_vwap(spot=101, prior_close=100, vwap=100.5) == 1

    def test_gap_down_below_vwap(self):
        assert score_gap_vwap(spot=99, prior_close=100, vwap=99.5) == -1

    def test_small_gap_neutral(self):
        assert score_gap_vwap(spot=100.1, prior_close=100, vwap=100) == 0

    def test_no_vwap_gap_up(self):
        assert score_gap_vwap(spot=100.5, prior_close=100, vwap=None) == 1

    def test_no_vwap_gap_down(self):
        assert score_gap_vwap(spot=99.5, prior_close=100, vwap=None) == -1

    def test_no_vwap_small_gap(self):
        assert score_gap_vwap(spot=100.2, prior_close=100, vwap=None) == 0

    def test_zero_prior_close(self):
        assert score_gap_vwap(spot=100, prior_close=0, vwap=100) == 0

    def test_zero_vwap(self):
        assert score_gap_vwap(spot=100.5, prior_close=100, vwap=0) == 1


# ---------------------------------------------------------------------------
# compute_bias
# ---------------------------------------------------------------------------


class TestComputeBias:
    @pytest.mark.parametrize(
        "score,expected_bias,expected_conf",
        [
            (5, Bias.STRONG_BULLISH, "HIGH"),
            (4, Bias.STRONG_BULLISH, "HIGH"),
            (3, Bias.STRONG_BULLISH, "MEDIUM"),
            (2, Bias.LEAN_BULLISH, "MEDIUM"),
            (1, Bias.LEAN_BULLISH, "LOW"),
            (0, Bias.NEUTRAL, "LOW"),
            (-1, Bias.LEAN_BEARISH, "LOW"),
            (-2, Bias.LEAN_BEARISH, "MEDIUM"),
            (-3, Bias.STRONG_BEARISH, "MEDIUM"),
            (-4, Bias.STRONG_BEARISH, "HIGH"),
            (-5, Bias.STRONG_BEARISH, "HIGH"),
        ],
    )
    def test_all_scores(self, score, expected_bias, expected_conf):
        bias, conf = compute_bias(score)
        assert bias == expected_bias
        assert conf == expected_conf


# ---------------------------------------------------------------------------
# compute_signal
# ---------------------------------------------------------------------------


class TestComputeSignal:
    def test_vol_regime_notrade_overrides(self):
        vol = _make_vol_regime("RED", "NOTRADE")
        assert compute_signal(Bias.STRONG_BULLISH, "HIGH", vol) == "NOTRADE"

    def test_neutral_low_is_notrade(self):
        assert compute_signal(Bias.NEUTRAL, "LOW", None) == "NOTRADE"

    def test_lean_bullish_is_trade(self):
        assert compute_signal(Bias.LEAN_BULLISH, "LOW", None) == "TRADE"

    def test_strong_bearish_green_is_trade(self):
        vol = _make_vol_regime("GREEN", "TRADE")
        assert compute_signal(Bias.STRONG_BEARISH, "HIGH", vol) == "TRADE"


# ---------------------------------------------------------------------------
# detect_market_bias (integration of pure functions)
# ---------------------------------------------------------------------------


class TestDetectMarketBias:
    def test_strong_bullish(self):
        # Bullish structure (+2), GREEN vol (+1), gap up above vwap (+1) = +4
        result = detect_market_bias(
            swing_highs=np.array([100, 105]),
            swing_lows=np.array([90, 92]),
            spot=101,
            prior_close=100,
            vwap=100.5,
            vol_regime=_make_vol_regime("GREEN"),
        )
        assert result.bias == Bias.STRONG_BULLISH
        assert result.score == 4
        assert result.confidence == "HIGH"
        assert result.signal == "TRADE"
        assert result.invalidation_level == 92.0

    def test_strong_bearish(self):
        # Bearish structure (-2), RED vol (-1), gap down below vwap (-1) = -4
        vol = _make_vol_regime("RED", "NOTRADE")
        result = detect_market_bias(
            swing_highs=np.array([105, 100]),
            swing_lows=np.array([92, 88]),
            spot=99,
            prior_close=100,
            vwap=99.5,
            vol_regime=vol,
        )
        assert result.bias == Bias.STRONG_BEARISH
        assert result.score == -4
        assert result.signal == "NOTRADE"

    def test_neutral_no_data(self):
        result = detect_market_bias(
            swing_highs=np.array([100]),
            swing_lows=np.array([90]),
            spot=100,
            prior_close=100,
            vwap=None,
            vol_regime=None,
        )
        assert result.bias == Bias.NEUTRAL
        assert result.score == 0
        assert result.signal == "NOTRADE"


# ---------------------------------------------------------------------------
# MarketBiasResult properties
# ---------------------------------------------------------------------------


class TestMarketBiasResultProperties:
    def test_high_vol_yellow(self):
        result = detect_market_bias(
            swing_highs=np.array([100, 105]),
            swing_lows=np.array([90, 92]),
            spot=100,
            prior_close=100,
            vol_regime=_make_vol_regime("YELLOW"),
        )
        assert result.high_vol is True

    def test_high_vol_green(self):
        result = detect_market_bias(
            swing_highs=np.array([100, 105]),
            swing_lows=np.array([90, 92]),
            spot=100,
            prior_close=100,
            vol_regime=_make_vol_regime("GREEN"),
        )
        assert result.high_vol is False

    def test_action_notrade(self):
        vol = _make_vol_regime("RED", "NOTRADE")
        result = detect_market_bias(
            swing_highs=np.array([100, 105]),
            swing_lows=np.array([90, 92]),
            spot=100,
            prior_close=100,
            vol_regime=vol,
        )
        assert "NO TRADE" in result.action


# ---------------------------------------------------------------------------
# _neutral_fallback
# ---------------------------------------------------------------------------


class TestNeutralFallback:
    def test_returns_notrade(self):
        result = _neutral_fallback()
        assert result.bias == Bias.NEUTRAL
        assert result.signal == "NOTRADE"
        assert result.score == 0
        assert result.spot == 0.0
