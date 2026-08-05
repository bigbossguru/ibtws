"""Pre-market volatility risk scoring for short-premium / 0DTE strategies.

Scores market conditions on a 0-100 scale from four complementary components:

    Component 1 — VIX deviation   (0-35)  z-score vs a rolling window
    Component 2 — VX1D / VIX ratio (0-25)  intraday vs 30-day implied vol
    Component 3 — Absolute VIX     (0-20)  raw level of fear
    Component 4 — Term structure   (0-20)  VIX slope vs VIX3M

This is a pure function over pandas data with no dependency on ``ib_async``,
so it is reusable across any short-premium bot and trivially testable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Momentum adjustments (Component 1 only)
MOMENTUM_FALLING_MULTIPLIER = 0.7  # applied when the 1-day VIX change < -0.5
MOMENTUM_RISING_DELTA = 8  # added when the 1-day VIX change > +1.0
MOMENTUM_FALLING_THRESHOLD = -0.5
MOMENTUM_RISING_THRESHOLD = 1.0

# True maximum achievable score (35 + 25 + 20 + 20 = 100)
MAX_POSSIBLE_SCORE = 100


# VIX z-deviation brackets -> score_delta (Component 1, max 35).
# Steps are graduated evenly to avoid over-penalising borderline z-values.
Z_SCORE_BRACKETS = [
    (0.0, 0),
    (0.5, 6),
    (1.0, 14),
    (1.5, 25),
    (float("inf"), 35),
]

# VX1D / VIX ratio brackets -> (score, flag)  (Component 2, max 25).
# Capped at 25 (reduced from 35) — this is the noisiest signal, so it should
# support rather than dominate the z-score component.
VX_RATIO_BRACKETS = [
    (0.60, 0, "🟢 VERY CALM (DECAY FAVORABLE)"),
    (0.75, 4, "🟢 CALM"),
    (0.85, 8, "🟡 NORMAL"),
    (0.95, 15, "⚠️ ELEVATED INTRADAY RISK"),
    (1.05, 21, "🚨 HIGH INTRADAY VOL EXPECTED"),
    (float("inf"), 25, "🚨 EXTREME INTRADAY STRESS"),
]

# Absolute VIX level brackets -> (score, flag)  (Component 3, max 20).
VIX_LEVEL_BRACKETS = [
    (13, 0, "🟢 VERY LOW"),
    (16, 3, "🟢 LOW"),
    (20, 8, "🟡 MODERATE"),
    (25, 14, "⚠️ ELEVATED"),
    (30, 18, "⚠️ HIGH"),
    (float("inf"), 20, "🚨 VERY HIGH"),
]

# Term-structure slope (%) brackets -> (score, flag)  (Component 4, max 20).
# Raised to 20 (from 10) — inversion is historically the most reliable leading
# indicator of acute volatility stress and was previously underweighted.
TERM_SLOPE_BRACKETS = [
    (-10, 20, "🚨 STEEP INVERSION"),
    (-5, 16, "⚠️ INVERTED"),
    (-2, 12, "⚠️ SLIGHT INVERSION"),
    (2, 8, "🟢 FLAT"),
    (5, 4, "🟢 NORMAL"),
    (10, 2, "🟢 NORMAL CONTANGO"),
    (float("inf"), 0, "🟢 STEEP CONTANGO"),
]


def _lookup_bracket(value: float, brackets: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Return the first bracket whose threshold exceeds *value*.

    Guards against NaN: a NaN *value* never satisfies ``value < threshold`` and
    would otherwise silently fall through to the most severe bracket, so it is
    rejected explicitly.
    """
    if isinstance(value, float) and math.isnan(value):
        raise ValueError("_lookup_bracket received NaN; upstream metric is undefined.")
    for entry in brackets:
        if value < entry[0]:
            return entry
    return brackets[-1]


def _validate_positive_finite(**values: float) -> None:
    """Raise ``ValueError`` if any named value is not a positive, finite number."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value!r}.")


def common_volatility_risk(
    vix_series: pd.Series,
    vx1d_current: float,
    vix3m_current: float | None = None,
    lookback_days: int = 20,
    risk_threshold: int = 50,
    debug: bool = False,
) -> dict:
    """Pre-market volatility risk score.

    Evaluates market conditions on a 0-100 scale across four components:

        Component 1 — VIX deviation    (0-35)  z-score vs a rolling window
        Component 2 — VX1D / VIX ratio  (0-25)  intraday vs 30-day implied vol
        Component 3 — Absolute VIX      (0-20)  raw level of fear
        Component 4 — Term structure    (0-20)  VIX slope vs VIX3M

    Maximum achievable score: 100.

    Parameters
    ----------
    vix_series:
        Daily VIX close prices; needs at least ``lookback_days + 1``
        observations.
    vx1d_current:
        Current VIX1D value (1-day implied volatility).
    vix3m_current:
        Current VIX3M value (93-day implied volatility). If ``None``,
        Component 4 is skipped (score = 0).
    lookback_days:
        Rolling window used for the z-score calculation (default 20).
    risk_threshold:
        Score at which and above the decision flips to NO TRADE.
            Conservative  40-50  (avoids medium-high risk)
            Moderate       50-60  (balanced)
            Aggressive     60-70  (halts only in extreme conditions)
    debug:
        If ``True``, include raw diagnostic metrics in the returned dict.

    Returns
    -------
    dict
        Keys:
            decision          : "TRADE" or "NO TRADE"
            risk_score        : int  0-100
            risk_threshold    : int
            overall_structure : str  human-readable summary of flags
            component_scores  : dict of individual component scores
            metrics           : dict of raw values (only when debug=True)

    Raises
    ------
    ValueError
        If ``vix_series`` has fewer than ``lookback_days + 1`` rows, contains
        NaNs in the window used for scoring, or if any of the current VIX /
        VIX1D / VIX3M inputs are non-positive or non-finite. Failing loudly is
        intentional: a caller relying on this as a trade gate should treat bad
        data as a hard block (fail-closed) rather than receive a silently
        maxed-out score.
    """
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}.")

    if len(vix_series) < lookback_days + 1:
        raise ValueError(
            f"vix_series has {len(vix_series)} rows; need at least {lookback_days + 1} (lookback_days + 1)."
        )

    logger.info("Analyzing pre-market volatility risk score...")

    # ------------------------------------------------------------------
    # Raw metrics
    # ------------------------------------------------------------------
    # Only the tail actually used for scoring (current + prev + lookback window)
    # needs to be clean; earlier history is irrelevant and may legitimately
    # contain gaps.
    window = vix_series.iloc[-(lookback_days + 1) :]
    if window.isna().any():
        raise ValueError(
            "vix_series contains NaN in the scoring window "
            f"(last {lookback_days + 1} observations); cannot compute a reliable score."
        )

    vix_current = float(vix_series.iloc[-1])
    vix_prev = float(vix_series.iloc[-2])
    _validate_positive_finite(vix_current=vix_current, vx1d_current=vx1d_current)
    if vix3m_current is not None:
        _validate_positive_finite(vix3m_current=vix3m_current)

    vix_hist = vix_series.iloc[-(lookback_days + 1) : -1]
    vix_mean = vix_hist.mean()
    vix_std = vix_hist.std()
    vix_change_1d = vix_current - vix_prev
    z_score = (vix_current - vix_mean) / vix_std if vix_std > 0 else 0.0
    vx_ratio = vx1d_current / vix_current

    # ------------------------------------------------------------------
    # Component 1: VIX deviation (0-35)
    # ------------------------------------------------------------------
    _, score_delta = _lookup_bracket(z_score, Z_SCORE_BRACKETS)[:2]

    # Momentum adjustment — applied regardless of the sign of z_score so that a
    # sharp spike from a low base is still captured.
    if vix_change_1d < MOMENTUM_FALLING_THRESHOLD and z_score > 0:
        # VIX is retreating from an elevated level — reduce urgency
        score_delta = int(score_delta * MOMENTUM_FALLING_MULTIPLIER)
    elif vix_change_1d > MOMENTUM_RISING_THRESHOLD:
        # VIX is rising regardless of current level — add urgency
        score_delta = min(35, score_delta + MOMENTUM_RISING_DELTA)

    # ------------------------------------------------------------------
    # Component 2: VX1D / VIX ratio (0-25)
    # ------------------------------------------------------------------
    _, score_vx, vx_flag = _lookup_bracket(vx_ratio, VX_RATIO_BRACKETS)

    # ------------------------------------------------------------------
    # Component 3: Absolute VIX level (0-20)
    # ------------------------------------------------------------------
    _, score_level, level_flag = _lookup_bracket(vix_current, VIX_LEVEL_BRACKETS)

    # ------------------------------------------------------------------
    # Component 4: Term structure (0-20) — skipped if vix3m is unavailable
    # ------------------------------------------------------------------
    if vix3m_current is None:
        score_term = 0
        term_structure_flag = "⚪ TERM STRUCTURE N/A"
    else:
        vix_term_slope = (vix3m_current - vix_current) / vix_current * 100
        _, score_term, term_structure_flag = _lookup_bracket(vix_term_slope, TERM_SLOPE_BRACKETS)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    total_score: int = score_delta + score_vx + score_level + score_term
    decision: Literal["TRADE", "NO TRADE"] = "TRADE" if total_score < risk_threshold else "NO TRADE"

    result = {
        "decision": decision,
        "risk_score": total_score,
        "risk_threshold": risk_threshold,
        "overall_structure": f"Term Struct: {term_structure_flag}\nVIX1D/VIX:   {vx_flag}\nVIX Level:   {level_flag}",
        "component_scores": {
            "vix_deviation": score_delta,
            "vx_ratio": score_vx,
            "vix_level": score_level,
            "term_structure": score_term,
        },
    }

    if debug:
        result["metrics"] = {
            "vix_current": round(float(vix_current), 2),
            "vix_z_score": round(z_score, 2),
            "vix_change_1d": round(float(vix_change_1d), 2),
            "vx1d_vix_ratio": round(vx_ratio, 2),
            "vix3m_current": vix3m_current,
            "max_possible": MAX_POSSIBLE_SCORE,
        }

    return result
