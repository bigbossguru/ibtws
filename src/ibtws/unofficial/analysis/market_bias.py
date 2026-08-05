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
    and the momentum (last close vs slow MA) agree and are non-neutral. This
    trend+momentum confirmation is deliberate: a raw moving-average crossover is
    prone to false signals in choppy markets, so it should be used as a
    confirmation filter rather than a standalone trigger.

    Parameters
    ----------
    market_data:
        DataFrame with at least a ``close`` column. May be ``None``.
    fast_window:
        Window length for the fast moving average (default 5). Must be a
        positive integer strictly smaller than ``slow_window``.
    slow_window:
        Window length for the slow moving average (default 10).
    volume_window:
        Minimum-length guard used together with ``slow_window`` (default 20).
        Must be a positive integer.

    Returns
    -------
    dict
        ``{"bias": "bullish" | "bearish" | "neutral", "details": {...}}``.
        A ``"!neutral"`` bias signals a *data* problem (no data, insufficient
        data, missing ``close`` column, or NaN in the evaluated window) rather
        than a genuine neutral reading, so callers can distinguish "flat market"
        from "cannot tell".

    Raises
    ------
    ValueError
        If the window parameters are misconfigured (non-positive, or
        ``fast_window >= slow_window``). This is a caller programming error, as
        opposed to a data problem which is reported via the ``"!neutral"``
        sentinel.
    """
    if fast_window < 1 or slow_window < 1 or volume_window < 1:
        raise ValueError(
            f"window lengths must be >= 1 (fast_window={fast_window}, "
            f"slow_window={slow_window}, volume_window={volume_window})."
        )
    if fast_window >= slow_window:
        raise ValueError(f"fast_window ({fast_window}) must be strictly smaller than slow_window ({slow_window}).")

    if market_data is None:
        return {"bias": "!neutral", "details": {"reason": "no_data_provided"}}

    if "close" not in market_data.columns:
        return {"bias": "!neutral", "details": {"reason": "missing_close_column"}}

    min_len = max(slow_window, volume_window)
    if len(market_data) < min_len:
        return {"bias": "!neutral", "details": {"reason": "insufficient_data"}}

    df = market_data.copy()
    df["ma_fast"] = df["close"].rolling(fast_window).mean()
    df["ma_slow"] = df["close"].rolling(slow_window).mean()
    last = df.iloc[-1]

    # NaN in the evaluated values (e.g. gaps in the input close series) would
    # make every comparison below False and silently masquerade as a genuine
    # "neutral". Surface it as a data problem instead.
    if last[["close", "ma_fast", "ma_slow"]].isna().any():
        return {"bias": "!neutral", "details": {"reason": "nan_in_window"}}

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
