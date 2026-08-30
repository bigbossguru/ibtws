"""Pre-market volatility-regime gate for 0DTE SPX credit spreads / iron condors.

Implementation of ``VOLATILITY_REGIME_CONCEPT.md`` (version 2). The gate is a
**tail-regime cut-off, not a good-day selector**: it rejects mornings where
volatility sits at an extreme or is being repriced right now, and lets
everything else through (~88% of days on the calibration sample).

Six metrics, each contributing at most one flag:

    base_level        percentile rank of VIX1D_open over the last 60 sessions
    vix1d_absolute    raw VIX1D_open level
    vix_roc           overnight VIX repricing, (open - prev_close) / prev_close
    term_structure    VIX1D / VIX
    premium_richness  VIX1D_open - RV20 (realised close-to-close vol)
    gex               signed distance from the Zero Gamma Level, in expected moves

Decision (concept 3.2)::

    favorable = (hard == 0) and (soft <= 1)

Thresholds are percentiles (p90 -> soft, p98 -> hard) of the *actual* observed
distributions on 500 sessions (2024-08-29 … 2026-08-27), never transplanted
from equity-option practice. Keep that property when editing them.

Fail-safe (concept 3.3): an unavailable metric counts as a **hard** flag and is
logged with the ``missing_data`` code so infrastructure gaps stay
distinguishable from genuine market flags. An unavailable base level is a
special case — no evaluation happens at all, reason ``no_base_level``.

The macro calendar (FOMC / CPI / NFP) is deliberately **out of scope**: it is a
separate hard-skip module that overrides this detector (concept 5). This module
knows nothing about it.

Like the rest of :mod:`ibtws.unofficial.analysis`, this is a pure computation
over pandas / floats with no dependency on ``ib_async`` — data loading belongs
to an infrastructure adapter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

Severity = Literal["soft", "hard"]

MetricName = Literal[
    "base_level",
    "vix1d_absolute",
    "vix_roc",
    "term_structure",
    "premium_richness",
    "gex",
]

BaseRegime = Literal["LOW", "NORMAL", "HIGH", "EXTREME"]

#: Reason reported when the base level cannot be established (concept 3.3).
NO_BASE_LEVEL = "no_base_level"


@dataclass(frozen=True)
class VolatilityRegimeConfig:
    """Thresholds and windows of the regime gate.

    Defaults are the version-2 calibration. Every threshold is a percentile of
    the observed distribution of its own metric: ``*_soft`` ≈ p90, ``*_hard``
    ≈ p98.
    """

    # 2.1 base level — percentile rank of VIX1D_open
    base_window: int = 60
    base_rank_soft: float = 90.0  # 90-98 -> HIGH, soft
    base_rank_hard: float = 98.0  # > 98  -> EXTREME, hard
    base_rank_low: float = 10.0  # < 10 -> LOW, informational only (no flag)

    # 2.1 absolute cut-off — percentile rank is relative, this is not
    vix1d_absolute_hard: float = 25.0

    # 2.2 overnight VIX repricing, percent
    roc_soft_pct: float = 10.0
    roc_hard_pct: float = 15.0

    # 2.3 term structure VIX1D / VIX
    term_soft: float = 0.85
    term_hard: float = 1.00

    # 2.4 premium richness VIX1D_open - RV20, volatility points
    rv_window: int = 20
    premium_soft: float = -6.0
    premium_hard: float = -10.0

    # 2.5 GEX — half-width of the unstable-sign zone, in expected moves
    gex_buffer_em: float = 0.25

    # 3.2 decision
    max_soft_flags: int = 1

    def __post_init__(self) -> None:
        """Validate the configuration; misordered thresholds are a caller bug."""
        if self.base_window < 1:
            raise ValueError(f"base_window must be >= 1, got {self.base_window}.")
        # A sample standard deviation needs two returns, i.e. three closes. A
        # smaller window would make RV20 permanently unavailable and silently
        # turn premium_richness into a standing hard flag.
        if self.rv_window < 2:
            raise ValueError(f"rv_window must be >= 2, got {self.rv_window}.")
        if not 0.0 <= self.base_rank_low <= self.base_rank_soft <= self.base_rank_hard <= 100.0:
            raise ValueError(
                "base rank thresholds must satisfy 0 <= low <= soft <= hard <= 100 "
                f"(low={self.base_rank_low}, soft={self.base_rank_soft}, hard={self.base_rank_hard})."
            )
        if self.roc_soft_pct > self.roc_hard_pct:
            raise ValueError(f"roc_soft_pct ({self.roc_soft_pct}) must be <= roc_hard_pct ({self.roc_hard_pct}).")
        if self.term_soft > self.term_hard:
            raise ValueError(f"term_soft ({self.term_soft}) must be <= term_hard ({self.term_hard}).")
        if self.premium_hard > self.premium_soft:
            raise ValueError(f"premium_hard ({self.premium_hard}) must be <= premium_soft ({self.premium_soft}).")
        if self.gex_buffer_em < 0:
            raise ValueError(f"gex_buffer_em must be >= 0, got {self.gex_buffer_em}.")
        if self.vix1d_absolute_hard <= 0:
            raise ValueError(f"vix1d_absolute_hard must be > 0, got {self.vix1d_absolute_hard}.")
        if self.max_soft_flags < 0:
            raise ValueError(f"max_soft_flags must be >= 0, got {self.max_soft_flags}.")


DEFAULT_CONFIG = VolatilityRegimeConfig()


@dataclass(frozen=True)
class RegimeFlag:
    """One triggered flag. ``missing=True`` marks a data gap, not a market state."""

    metric: MetricName
    severity: Severity
    value: float | None
    detail: str
    missing: bool = False

    def __str__(self) -> str:  # pragma: no cover - logging convenience
        code = "missing_data" if self.missing else "risk_flag"
        return f"{code}: {self.metric} [{self.severity}] {self.detail}"


@dataclass(frozen=True)
class VolatilityRegimeResult:
    """Outcome of one pre-market regime evaluation.

    ``favorable`` is the only field a caller must honour. Everything else exists
    for logging and for the flag-frequency monitoring described in concept 7.2.
    """

    favorable: bool
    flags: tuple[RegimeFlag, ...] = ()
    base_rank: float | None = None
    base_regime: BaseRegime | None = None
    degraded_base: bool = False  # VIX1D unavailable -> VIX percentile fallback
    reason: str | None = None  # set only when no evaluation happened
    metrics: dict[str, float | None] = field(default_factory=dict)
    zgl_source: str | None = None

    @property
    def hard_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "hard")

    @property
    def soft_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "soft")

    @property
    def missing_metrics(self) -> tuple[MetricName, ...]:
        return tuple(f.metric for f in self.flags if f.missing)

    def summary(self) -> str:
        """One-line human-readable verdict."""
        verdict = "FAVORABLE" if self.favorable else "SKIP"
        head = f"{verdict} hard={self.hard_count} soft={self.soft_count}"
        if self.reason:
            head = f"{head} reason={self.reason}"
        if self.degraded_base:
            head = f"{head} (degraded base level)"
        if not self.flags:
            return head
        return head + " | " + "; ".join(str(f) for f in self.flags)


def detect_volatility_regime(
    *,
    vix1d_open: float | None,
    vix1d_history: pd.Series | None,
    vix_open: float | None,
    vix_prev_close: float | None,
    spx_closes: pd.Series | None,
    spx_price: float | None,
    zero_gamma_level: float | None,
    vix_history: pd.Series | None = None,
    zgl_source: str | None = None,
    config: VolatilityRegimeConfig = DEFAULT_CONFIG,
) -> VolatilityRegimeResult:
    """Evaluate the pre-market volatility regime for a 0DTE short-premium entry.

    Parameters
    ----------
    vix1d_open:
        Today's VIX1D open. Basis of the base level (2.1), numerator of the term
        structure (2.3), premium-richness input (2.4) and the expected move used
        to normalise the GEX distance (2.5).
    vix1d_history:
        VIX1D opens of the **previous** sessions, oldest first, today excluded.
        Only the last ``config.base_window`` usable observations are used.
    vix_open:
        Today's VIX open. ROC (2.2) and term-structure denominator (2.3).
    vix_prev_close:
        Previous VIX close, for the overnight ROC (2.2).
    spx_closes:
        SPX closes of **completed** sessions, oldest first; needs
        ``config.rv_window + 1`` usable values to produce RV20 (2.4).
    spx_price:
        Current SPX price for the GEX distance (2.5).
    zero_gamma_level:
        Zero Gamma Level from an external GEX calculation (e.g.
        :class:`~ibtws.unofficial.analysis.gex.GexCalculator`).
    vix_history:
        VIX opens of previous sessions, used only as the degraded fallback for
        the base level when VIX1D is unavailable (2.1). Term structure, premium
        richness and the GEX normalisation have **no** fallback and become
        missing-data hard flags in that case.
    zgl_source:
        Provider / version of the ZGL, recorded for reproducibility (concept 6).
    config:
        Thresholds and windows; see :class:`VolatilityRegimeConfig`.

    Returns
    -------
    VolatilityRegimeResult
        ``favorable=True`` only when no hard flag fired and at most
        ``config.max_soft_flags`` soft flags did. A data gap yields
        ``favorable=False`` with the affected metric flagged ``missing`` — an
        unavailable base level short-circuits the whole evaluation with
        ``reason="no_base_level"``.

    Raises
    ------
    ValueError
        Never for market data — only via :class:`VolatilityRegimeConfig`
        validation, which is a caller programming error.

    Notes
    -----
    The macro calendar overrides this result and is checked by a separate module
    (concept 5). GEX is the only metric without historical validation (concept
    7.1): treat it as a hypothesis under observation.
    """
    vix1d = _clean_value(vix1d_open)
    vix = _clean_value(vix_open)
    vix_prev = _clean_value(vix_prev_close)
    spot = _clean_value(spx_price)
    zgl = _clean_value(zero_gamma_level)

    # ------------------------------------------------------------------
    # 2.1 Base level — percentile rank, with the degraded VIX fallback
    # ------------------------------------------------------------------
    base_rank, degraded = _base_rank(vix1d, vix1d_history, vix, vix_history, config)
    if base_rank is None:
        result = VolatilityRegimeResult(
            favorable=False,
            reason=NO_BASE_LEVEL,
            metrics={"vix1d_open": vix1d, "vix_open": vix},
            zgl_source=zgl_source,
        )
        logger.warning(f"VolatilityRegime: missing_data: base_level — {result.summary()}")
        return result
    if degraded:
        logger.warning("VolatilityRegime: VIX1D unavailable — base level degraded to a VIX percentile (2.1 fallback).")

    base_regime = _base_regime(base_rank, config)
    flags: list[RegimeFlag] = []

    if base_rank > config.base_rank_hard:
        flags.append(
            RegimeFlag("base_level", "hard", base_rank, f"base_rank={base_rank:.1f} > {config.base_rank_hard}")
        )
    elif base_rank >= config.base_rank_soft:
        flags.append(
            RegimeFlag("base_level", "soft", base_rank, f"base_rank={base_rank:.1f} in HIGH band ({base_regime})")
        )

    # ------------------------------------------------------------------
    # 2.1 Absolute cut-off — VIX1D_open > 25
    # ------------------------------------------------------------------
    # No fallback: a VIX level cannot be compared against a VIX1D threshold, so
    # in degraded mode this metric is simply unavailable (3.3 -> hard).
    if vix1d is None:
        flags.append(_missing("vix1d_absolute", "VIX1D open unavailable"))
    elif vix1d > config.vix1d_absolute_hard:
        flags.append(RegimeFlag("vix1d_absolute", "hard", vix1d, f"VIX1D={vix1d:.2f} > {config.vix1d_absolute_hard}"))

    # ------------------------------------------------------------------
    # 2.2 Overnight VIX repricing — flag on the upside only
    # ------------------------------------------------------------------
    roc = (vix - vix_prev) / vix_prev * 100.0 if vix is not None and vix_prev is not None else None
    if roc is None:
        flags.append(_missing("vix_roc", "VIX open or previous close unavailable"))
    elif roc > config.roc_hard_pct:
        flags.append(RegimeFlag("vix_roc", "hard", roc, f"ROC={roc:.1f}% > {config.roc_hard_pct}%"))
    elif roc > config.roc_soft_pct:
        flags.append(RegimeFlag("vix_roc", "soft", roc, f"ROC={roc:.1f}% > {config.roc_soft_pct}%"))

    # ------------------------------------------------------------------
    # 2.3 Term structure — VIX1D / VIX (normal level ≈ 0.6, not 1.0)
    # ------------------------------------------------------------------
    term = vix1d / vix if vix1d is not None and vix is not None else None
    if term is None:
        flags.append(_missing("term_structure", "VIX1D or VIX unavailable (no VIX9D fallback by design)"))
    elif term > config.term_hard:
        flags.append(RegimeFlag("term_structure", "hard", term, f"VIX1D/VIX={term:.2f} > {config.term_hard}"))
    elif term > config.term_soft:
        flags.append(RegimeFlag("term_structure", "soft", term, f"VIX1D/VIX={term:.2f} > {config.term_soft}"))

    # ------------------------------------------------------------------
    # 2.4 Premium richness — VIX1D_open - RV20
    # ------------------------------------------------------------------
    rv20 = _realised_vol(spx_closes, config.rv_window)
    spread = vix1d - rv20 if vix1d is not None and rv20 is not None else None
    if spread is None:
        flags.append(_missing("premium_richness", "VIX1D or RV20 unavailable"))
    elif spread < config.premium_hard:
        flags.append(RegimeFlag("premium_richness", "hard", spread, f"VIX1D-RV20={spread:.1f} < {config.premium_hard}"))
    elif spread < config.premium_soft:
        flags.append(RegimeFlag("premium_richness", "soft", spread, f"VIX1D-RV20={spread:.1f} < {config.premium_soft}"))

    # ------------------------------------------------------------------
    # 2.5 GEX — signed distance from ZGL, measured in expected moves
    # ------------------------------------------------------------------
    em_pct = _expected_move_pct(vix1d)
    if spot is not None and zgl is not None and em_pct is not None:
        dist: float | None = (spot - zgl) / spot * 100.0 / em_pct
    else:
        dist = None
    buffer_em = config.gex_buffer_em
    if dist is None:
        flags.append(_missing("gex", "spot, ZGL or expected move unavailable"))
    elif dist < -buffer_em:
        flags.append(RegimeFlag("gex", "hard", dist, f"price {dist:.2f} EM below ZGL (< -{buffer_em} EM)"))
    elif dist <= buffer_em:
        flags.append(RegimeFlag("gex", "soft", dist, f"price within ±{buffer_em} EM of ZGL (dist={dist:.2f} EM)"))

    # ------------------------------------------------------------------
    # 3.2 Decision
    # ------------------------------------------------------------------
    result = VolatilityRegimeResult(
        favorable=False,
        flags=tuple(flags),
        base_rank=base_rank,
        base_regime=base_regime,
        degraded_base=degraded,
        metrics={
            "vix1d_open": vix1d,
            "vix_open": vix,
            "vix_prev_close": vix_prev,
            "base_rank": base_rank,
            "vix_roc_pct": roc,
            "term_structure": term,
            "rv20": rv20,
            "premium_spread": spread,
            "expected_move_pct": em_pct,
            "spx_price": spot,
            "zero_gamma_level": zgl,
            "zgl_distance_em": dist,
        },
        zgl_source=zgl_source,
    )
    favorable = result.hard_count == 0 and result.soft_count <= config.max_soft_flags
    result = replace(result, favorable=favorable)

    for flag in flags:
        if flag.missing:
            logger.warning(f"VolatilityRegime: {flag}")
        else:
            logger.info(f"VolatilityRegime: {flag}")
    logger.info(f"VolatilityRegime: {result.summary()}")
    return result


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _clean_value(value: float | None) -> float | None:
    """Return *value* as a positive finite float, or ``None`` if unusable."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0:
        return None
    return out


def _usable_window(series: pd.Series | None, window: int) -> pd.Series | None:
    """Last *window* usable observations, or ``None`` when there are not enough.

    Unusable values (NaN, non-numeric, non-positive) are dropped and the window
    is filled from older history instead, so an isolated bad print does not
    shrink the sample. What is *not* papered over is an outright short history:
    computing a percentile over fewer sessions would silently change the scale's
    meaning, so it is reported as a data gap (and hence a hard flag) instead.
    """
    if series is None or len(series) == 0:
        return None
    values = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]
    if len(values) < window:
        return None
    return values.iloc[-window:]


def _percentile_rank(history: pd.Series, current: float) -> float:
    """Share of *history* strictly below *current*, in percent (concept 2.1)."""
    return float((history < current).mean() * 100.0)


def _base_rank(
    vix1d: float | None,
    vix1d_history: pd.Series | None,
    vix: float | None,
    vix_history: pd.Series | None,
    config: VolatilityRegimeConfig,
) -> tuple[float | None, bool]:
    """Percentile rank of today's 0DTE vol level; second element flags degradation."""
    window = _usable_window(vix1d_history, config.base_window)
    if vix1d is not None and window is not None:
        return _percentile_rank(window, vix1d), False

    # Fallback (2.1): same window and thresholds on VIX, logged as degraded.
    fallback = _usable_window(vix_history, config.base_window)
    if vix is not None and fallback is not None:
        return _percentile_rank(fallback, vix), True

    return None, False


def _base_regime(base_rank: float, config: VolatilityRegimeConfig) -> BaseRegime:
    """Descriptive label of the base level (2.1). Informational: LOW never blocks."""
    if base_rank < config.base_rank_low:
        return "LOW"
    if base_rank > config.base_rank_hard:
        return "EXTREME"
    if base_rank >= config.base_rank_soft:
        return "HIGH"
    return "NORMAL"


def _realised_vol(closes: pd.Series | None, window: int) -> float | None:
    """Annualised close-to-close realised volatility over *window* sessions.

    Uses log returns of the last ``window + 1`` closes (i.e. ``window``
    completed daily returns), sample standard deviation, annualised by
    ``√252`` and expressed in volatility points to match VIX1D.
    """
    values = _usable_window(closes, window + 1)
    if values is None:
        return None
    returns = np.diff(np.log(values.to_numpy(dtype="float64")))
    if len(returns) < 2:
        return None
    std = float(np.std(returns, ddof=1))
    if not math.isfinite(std):
        return None
    return std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


def _expected_move_pct(vix1d: float | None) -> float | None:
    """One-day expected move in percent of spot: ``VIX1D / 100 / √252 × 100``."""
    if vix1d is None:
        return None
    return vix1d / 100.0 / math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


def _missing(metric: MetricName, detail: str) -> RegimeFlag:
    """Fail-safe flag for an unavailable metric (concept 3.3): missing == hard."""
    return RegimeFlag(metric, "hard", None, detail, missing=True)
