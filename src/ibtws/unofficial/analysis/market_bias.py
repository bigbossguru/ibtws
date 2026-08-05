"""Directional market-bias classification from a price DataFrame.

Data loading (I/O) is intentionally out of scope — that belongs to an
infrastructure adapter. This module is a pure transformation over a
DataFrame of prices, so it has no dependency on ``ib_async`` and is trivially
testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def determine_market_bias(
    market_data: pd.DataFrame | None,
    *,
    fast_window: int = 5,
    slow_window: int = 10,
    volume_window: int = 20,
) -> dict:
    """Classify directional bias from close prices via fast/slow moving averages.

    A directional bias is reported only when both the trend (fast MA vs slow MA)
    and the momentum (last close vs slow MA) agree and are non-neutral.

    Parameters
    ----------
    market_data:
        DataFrame with at least a ``close`` column. May be ``None``.
    fast_window:
        Window length for the fast moving average (default 5).
    slow_window:
        Window length for the slow moving average (default 10).
    volume_window:
        Minimum-length guard used together with ``slow_window`` (default 20).

    Returns
    -------
    dict
        ``{"bias": "bullish" | "bearish" | "neutral", "details": {...}}``.
        A ``"!neutral"`` bias signals a data problem (no data / insufficient
        data) rather than a genuine neutral reading.
    """
    if market_data is None:
        return {"bias": "!neutral", "details": {"reason": "no_data_provided"}}

    min_len = max(slow_window, volume_window)
    if len(market_data) < min_len:
        return {"bias": "!neutral", "details": {"reason": "insufficient_data"}}

    df = market_data.copy()
    df["ma_fast"] = df["close"].rolling(fast_window).mean()
    df["ma_slow"] = df["close"].rolling(slow_window).mean()
    last = df.iloc[-1]

    if last["ma_fast"] > last["ma_slow"]:
        trend = "bullish"
    elif last["ma_fast"] < last["ma_slow"]:
        trend = "bearish"
    else:
        trend = "neutral"

    if last["close"] > last["ma_slow"]:
        momentum = "bullish"
    elif last["close"] < last["ma_slow"]:
        momentum = "bearish"
    else:
        momentum = "neutral"

    directional = trend == momentum and trend != "neutral"
    bias = trend if directional else "neutral"
    return {"bias": bias, "details": {"trend": trend, "momentum": momentum}}
