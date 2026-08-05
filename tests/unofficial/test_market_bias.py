"""Unit tests for ibtws.unofficial.analysis.market_bias."""

from __future__ import annotations

import pandas as pd
import pytest

from ibtws.unofficial.analysis.market_bias import determine_market_bias


def _series_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


class TestGuards:
    def test_none_returns_not_neutral(self):
        result = determine_market_bias(None)
        assert result["bias"] == "!neutral"
        assert result["details"]["reason"] == "no_data_provided"

    def test_insufficient_data_returns_not_neutral(self):
        # default min_len = max(slow_window=10, volume_window=20) = 20
        df = _series_df([1.0] * 5)
        result = determine_market_bias(df)
        assert result["bias"] == "!neutral"
        assert result["details"]["reason"] == "insufficient_data"

    def test_missing_close_column_returns_not_neutral(self):
        df = pd.DataFrame({"open": [1.0] * 25})
        result = determine_market_bias(df)
        assert result["bias"] == "!neutral"
        assert result["details"]["reason"] == "missing_close_column"

    def test_nan_in_window_returns_not_neutral(self):
        closes = [float(i) for i in range(1, 25)] + [float("nan")]
        result = determine_market_bias(_series_df(closes))
        assert result["bias"] == "!neutral"
        assert result["details"]["reason"] == "nan_in_window"

    def test_raises_on_non_positive_window(self):
        with pytest.raises(ValueError, match="window lengths must be >= 1"):
            determine_market_bias(_series_df([1.0] * 25), fast_window=0)

    def test_raises_when_fast_not_smaller_than_slow(self):
        with pytest.raises(ValueError, match="strictly smaller"):
            determine_market_bias(_series_df([1.0] * 25), fast_window=10, slow_window=10)


class TestBias:
    def test_bullish_when_trend_and_momentum_agree(self):
        closes = [float(i) for i in range(1, 26)]  # strictly rising
        result = determine_market_bias(_series_df(closes))
        assert result["bias"] == "bullish"
        assert result["details"]["trend"] == "bullish"
        assert result["details"]["momentum"] == "bullish"

    def test_bearish_when_trend_and_momentum_agree(self):
        closes = [float(i) for i in range(25, 0, -1)]  # strictly falling
        result = determine_market_bias(_series_df(closes))
        assert result["bias"] == "bearish"
        assert result["details"]["trend"] == "bearish"
        assert result["details"]["momentum"] == "bearish"

    def test_neutral_when_flat(self):
        result = determine_market_bias(_series_df([100.0] * 25))
        assert result["bias"] == "neutral"
        assert result["details"]["trend"] == "neutral"

    def test_custom_windows_allow_shorter_series(self):
        # With small windows the guard threshold drops, so 6 points suffice.
        result = determine_market_bias(
            _series_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            fast_window=2,
            slow_window=3,
            volume_window=3,
        )
        assert result["bias"] == "bullish"
