"""Unit tests for ibtws.unofficial.analysis.market_bias."""

from __future__ import annotations

import pandas as pd

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
