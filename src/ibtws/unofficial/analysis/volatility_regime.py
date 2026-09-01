"""Intraday volatility-regime assessment for 0DTE SPX credit spreads / iron condors.

Implementation of ``VOLATILITY_REGIME_CONCEPT.md`` (version 5), gate only.
Replaces the version-2 pre-market gate: the entry decision is no longer tied to
the open and the gate is far narrower.

Why the previous gate was replaced
----------------------------------
Version 2 blocked entries on four volatility metrics. Pricing the actual
position — iron condor, real strikes, take-profit, stop, walked forward on
15-minute bars — showed that rule to be *negative* in expectancy, delta -0.016
against trading unconditionally (concept 14.5). The cause is structural
(concept 14.4): with the strike width fixed in points, a vertical's maximum loss
does not move with implied volatility but the credit for a given delta does, so
``IC(r_level, P&L) = +0.366``. Blocking high volatility was removing the
best-paid entries.

What version 5 does instead
---------------------------
Everything is a percentile rank inside a 15-minute time-of-day bucket, because
VIX1D is a time-weighted blend of today's and tomorrow's SPX expiries whose
median climbs from 9.8 at 09:30 ET to 12.8 at 15:30 ET with no change in risk
(concept 2). Comparing against a pooled history labels every morning calm and
every afternoon stressed.

    realized_vs_em     rank(realised range so far / EM remaining)   -> GATES
    base_level         rank(VIX1D)                                  -> reports
    term_structure     rank(VIX1D / VIX)                            -> reports
    premium_richness   rank(VIX1D - RV20)                           -> reports
    absolute           VIX1D > 25 soft, VIX1D > 35 hard, VIX > 30 hard
    gex                signed distance from the Zero Gamma Level, in expected moves

``realized_vs_em`` is the only ranked metric that gates: it is the one whose
numerator carries no implied-volatility level, the only rule with a positive
delta (+0.040) and the best CVaR at 89% coverage (concept 14.5).

Decision (concept 6.1)::

    favorable = (hard == 0) and (soft <= max_soft_flags)

Scope: this module answers "is the regime acceptable", nothing else. Position
sizing is **out of scope** and lives with the caller. That boundary matters more
than it looks: concept 15.8 measures sizing as the only component that changes
the probability of ruin, while the gate moves expectancy by fractions of a
percent. A caller that honours ``favorable`` and sizes carelessly has solved the
smaller half of the problem — see the module notes below for what to carry over.

The three level metrics are computed and reported but never block, because their
direction is established while their magnitude is not (concept 15.1). With
sizing removed they are purely diagnostic: log them, monitor their frequency
(concept 12, item 10), and let the risk layer decide what a HIGH or EXTREME
reading is worth.

Fail-safe (concept 6.4): an unavailable **gating** metric counts as a hard flag,
logged with the ``missing_data`` code. An unavailable non-gating metric is only
recorded: blocking on the absence of a metric that has no right to block would
be incoherent. GEX stays opt-in — it is the one metric with no historical
validation at all (concept 4.7).

The macro calendar (FOMC / CPI / NFP) remains out of scope: it is a separate
hard-skip module that overrides this detector (concept 10).

Notes for the risk layer
------------------------
Two measured facts the caller should not rediscover the hard way:

* **Strike width belongs in expected-move units, not points** (concept 15.2).
  With the width fixed in points, maximum loss is a constant while the credit
  for a given delta grows with implied volatility — which is precisely why
  gating on the level destroyed expectancy. Scaling the width to
  :func:`expected_move_pct` restores the intended relationship.
* **Position size decides survival, not entry quality** (concept 15.8). At a 94%
  win rate with the average loss ~6x the average win, an SPX condor at ~30
  points of width risks ~$3,000 per contract, so a 0.5% risk limit needs ~$600k
  of equity for a single contract. That is the strategy's risk asymmetry, not a
  tuning problem.

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

#: RTH minutes from 09:30 ET to the 16:15 ET VIX1D roll horizon (concept 3).
SESSION_MINUTES = 405

#: Width of one time-of-day bucket, in minutes.
BUCKET_MINUTES = 15

Severity = Literal["soft", "hard", "info"]

MetricName = Literal[
    "realized_vs_em",
    "base_level",
    "term_structure",
    "premium_richness",
    "vix1d_absolute",
    "vix_absolute",
    "gex",
]

BaseRegime = Literal["LOW", "NORMAL", "HIGH", "EXTREME"]

#: Reasons reported when no evaluation happens at all (concept 6.1, 6.5).
NO_VIX1D = "no_vix1d"
OUTSIDE_RTH = "outside_rth"
SHORT_SESSION = "short_session"
OPENING_WINDOW = "opening_window"

#: Measured share of the session's range carried by each 15-minute bucket,
#: median over 251 sessions (concept 14.7, ``audit_economics.py``). The range is
#: front-loaded: the opening bucket carries 1.97x the uniform share of 3.85%,
#: the 14:30 bucket 0.70x. Square-root-of-time assumes uniformity and therefore
#: overstates residual risk in the afternoon — the ratio of realised remaining
#: excursion to ``EM_remaining`` falls from 0.92 at 09:30 to 0.47 at 15:30.
#:
#: These are the measured medians, not a smooth fit: the small bumps at 14:00
#: and 15:00 are in the data, and smoothing them would substitute a belief about
#: the shape for the observation. Using this profile reduces the drift of the
#: scale — coefficient of variation across buckets 0.211 -> 0.141 — but does not
#: remove it; the residual is most likely the VIX1D roll, which no reweighting of
#: today's clock can fix (concept 15.5).
_MEASURED_VARIANCE_SHARE: dict[str, float] = {
    "09:30": 0.07595,
    "09:45": 0.06459,
    "10:00": 0.05513,
    "10:15": 0.05081,
    "10:30": 0.05022,
    "10:45": 0.04719,
    "11:00": 0.04152,
    "11:15": 0.03905,
    "11:30": 0.04066,
    "11:45": 0.03692,
    "12:00": 0.03431,
    "12:15": 0.03227,
    "12:30": 0.03344,
    "12:45": 0.03415,
    "13:00": 0.03318,
    "13:15": 0.02988,
    "13:30": 0.02998,
    "13:45": 0.02817,
    "14:00": 0.03001,
    "14:15": 0.02693,
    "14:30": 0.02699,
    "14:45": 0.02641,
    "15:00": 0.03137,
    "15:15": 0.02608,
    "15:30": 0.03052,
    "15:45": 0.04426,
}

#: The 26 RTH buckets, ascending. ``09:30`` … ``15:45``.
RTH_BUCKETS: tuple[str, ...] = tuple(sorted(_MEASURED_VARIANCE_SHARE))

#: Normalised to sum to exactly 1.0. The measured medians sum to 0.99999, and
#: leaving that in place would make ``remaining_variance_share("09:30")`` differ
#: from a full session in the fifth decimal — small, but it would silently shift
#: every expected move away from the closed-form ``VIX1D / √252`` at the open.
_SHARE_TOTAL = sum(_MEASURED_VARIANCE_SHARE.values())
BUCKET_VARIANCE_SHARE: dict[str, float] = {
    label: share / _SHARE_TOTAL for label, share in _MEASURED_VARIANCE_SHARE.items()
}

#: A full session's worth of buckets; fewer means a half-day (concept 6.5).
FULL_SESSION_BUCKETS = len(RTH_BUCKETS)


@dataclass(frozen=True)
class VolatilityRegimeConfig:
    """Thresholds and windows of the intraday regime assessment.

    Defaults are the version-5 calibration on 251 sessions of 15-minute bars
    (2025-08-29 … 2026-08-28). Rank thresholds are uniform across metrics and
    buckets — that is the point of working in rank space — while the absolute
    cut-offs are deliberately *not* bucket-scaled.
    """

    # ── Rank space (concept 4) ────────────────────────────────────────────
    #: Observations of the SAME bucket used as the comparison base. Strictly
    #: prior sessions: including the one being assessed caps the attainable rank
    #: at 59/60 = 98.3 and makes a p98 threshold fire about half as often as
    #: calibrated.
    bucket_lookback: int = 60
    rank_hard: float = 98.0
    rank_soft: float = 90.0
    rank_hard_low: float = 2.0  # lower-tail metrics (premium richness)
    rank_soft_low: float = 10.0
    base_rank_low: float = 10.0  # LOW label; informational only

    # ── What gates (concept 15.1) ─────────────────────────────────────────
    #: Only ``realized_vs_em``. The level metrics correlate POSITIVELY with
    #: trade P&L (+0.37 / +0.36 / +0.23), so gating on them removed the
    #: best-paid entries. Flip these back on to reproduce the version-4 rule —
    #: see :func:`legacy_v4_config`.
    gate_on_realized: bool = True
    gate_on_base_level: bool = False
    gate_on_term_structure: bool = False
    gate_on_premium_richness: bool = False
    max_soft_flags: int = 1

    # ── Absolute cut-offs (concept 15.6) ──────────────────────────────────
    #: Not bucket-scaled, and deliberately redundant with the ranks. A rolling
    #: 60-session base carries only ~11 effective independent observations for
    #: VIX1D (concept 14.3), so after a calm quarter p98 sits at a level that is
    #: not dangerous. Percentiles adapt to a regime; these do not. The
    #: calibration sample contains no stress event at all, so these are the only
    #: tail protection that does not depend on it.
    vix1d_absolute_soft: float = 25.0  # was the sole hard cut-off in version 2
    vix1d_absolute_hard: float = 35.0  # never reached in-sample (max 34.0)
    vix_absolute_hard: float = 30.0  # 30-day vol as a regime-shift guard

    # ── Realised volatility (concept 4.5) ─────────────────────────────────
    rv_window: int = 20

    # ── Expected move (concept 15.5) ──────────────────────────────────────
    #: Use the measured intraday variance profile instead of √t.
    use_variance_profile: bool = True

    # ── Opening window (concept 15.3) ─────────────────────────────────────
    #: On risk-normalised returns the strongest relationship in the data is
    #: minutes remaining, IC -0.285, and the sign says EARLY is worse: E[RoR]
    #: -0.0002 with >360 minutes left against +0.0510 with <90, CVaR5 -0.319
    #: against -0.109. The opening window is skipped outright.
    skip_first_minutes: float = 45.0

    # ── GEX (concept 4.7) ─────────────────────────────────────────────────
    gex_buffer_em: float = 0.25

    # ── Reporting (concept 11) ────────────────────────────────────────────
    #: Past this weight on today's expiry the base level reflects tomorrow's
    #: session more than the rest of today's.
    roll_degraded_below: float = 0.26

    def __post_init__(self) -> None:
        """Validate the configuration; misordered thresholds are a caller bug."""
        if self.bucket_lookback < 1:
            raise ValueError(f"bucket_lookback must be >= 1, got {self.bucket_lookback}.")
        # A sample standard deviation needs two returns, i.e. three closes. A
        # smaller window would make RV20 permanently unavailable.
        if self.rv_window < 2:
            raise ValueError(f"rv_window must be >= 2, got {self.rv_window}.")
        if not 0.0 <= self.base_rank_low <= self.rank_soft <= self.rank_hard <= 100.0:
            raise ValueError(
                "rank thresholds must satisfy 0 <= base_rank_low <= rank_soft <= rank_hard <= 100 "
                f"(low={self.base_rank_low}, soft={self.rank_soft}, hard={self.rank_hard})."
            )
        if not 0.0 <= self.rank_hard_low <= self.rank_soft_low <= 100.0:
            raise ValueError(
                "lower-tail thresholds must satisfy 0 <= rank_hard_low <= rank_soft_low <= 100 "
                f"(hard_low={self.rank_hard_low}, soft_low={self.rank_soft_low})."
            )
        if self.vix1d_absolute_soft > self.vix1d_absolute_hard:
            raise ValueError(
                f"vix1d_absolute_soft ({self.vix1d_absolute_soft}) must be "
                f"<= vix1d_absolute_hard ({self.vix1d_absolute_hard})."
            )
        if self.vix1d_absolute_soft <= 0:
            raise ValueError(f"vix1d_absolute_soft must be > 0, got {self.vix1d_absolute_soft}.")
        if self.vix_absolute_hard <= 0:
            raise ValueError(f"vix_absolute_hard must be > 0, got {self.vix_absolute_hard}.")
        if self.gex_buffer_em < 0:
            raise ValueError(f"gex_buffer_em must be >= 0, got {self.gex_buffer_em}.")
        if self.max_soft_flags < 0:
            raise ValueError(f"max_soft_flags must be >= 0, got {self.max_soft_flags}.")
        if self.skip_first_minutes < 0:
            raise ValueError(f"skip_first_minutes must be >= 0, got {self.skip_first_minutes}.")
        if not 0 <= self.roll_degraded_below <= 1:
            raise ValueError(f"roll_degraded_below must be in [0, 1], got {self.roll_degraded_below}.")


DEFAULT_CONFIG = VolatilityRegimeConfig()


def legacy_v4_config(**overrides) -> VolatilityRegimeConfig:
    """The version-4 rule, for side-by-side comparison only.

    Kept so the change can be measured rather than asserted. Do **not** trade
    it: concept 14.5 measures it at -0.016 expectancy against trading
    unconditionally. All four ranked metrics gate, the single absolute cut-off
    sits at 25 and there is no opening-window skip.
    """
    defaults: dict[str, object] = {
        "gate_on_base_level": True,
        "gate_on_term_structure": True,
        "gate_on_premium_richness": True,
        "gate_on_realized": True,
        "vix1d_absolute_soft": 25.0,
        "vix1d_absolute_hard": 25.0,
        "vix_absolute_hard": math.inf,
        "use_variance_profile": False,
        "skip_first_minutes": 0.0,
    }
    defaults.update(overrides)
    return VolatilityRegimeConfig(**defaults)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RegimeFlag:
    """One recorded observation.

    ``severity="info"`` is a regime note that does not gate — the version-5
    treatment of the level metrics. ``missing=True`` marks a data gap rather than
    a market state.
    """

    metric: MetricName
    severity: Severity
    value: float | None
    detail: str
    missing: bool = False

    def __str__(self) -> str:  # pragma: no cover - logging convenience
        code = "missing_data" if self.missing else "risk_flag"
        return f"{code}: {self.metric} [{self.severity}] {self.detail}"


@dataclass(frozen=True)
class RankedMetric:
    """A raw value together with its percentile rank inside its bucket."""

    raw: float | None = None
    rank: float | None = None
    observations: int = 0

    @property
    def available(self) -> bool:
        return self.raw is not None and self.rank is not None


@dataclass(frozen=True)
class VolatilityRegimeResult:
    """Outcome of one intraday regime assessment.

    ``favorable`` is the only field a caller must honour. Everything else exists
    for logging, for the risk layer, and for the flag-frequency monitoring of
    concept 12.
    """

    favorable: bool
    flags: tuple[RegimeFlag, ...] = ()
    bucket: str | None = None
    minutes_left: float | None = None
    roll_weight_today: float | None = None
    variance_share_left: float | None = None
    expected_move_pct: float | None = None
    base_rank: float | None = None
    base_regime: BaseRegime | None = None
    reason: str | None = None  # set when no evaluation happened
    degraded: tuple[str, ...] = ()
    metrics: dict[str, float | None] = field(default_factory=dict)
    zgl_source: str | None = None

    @property
    def hard_count(self) -> int:
        """Hard flags, including missing gating metrics (concept 6.4)."""
        return sum(1 for f in self.flags if f.severity == "hard")

    @property
    def soft_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "soft")

    @property
    def info_flags(self) -> tuple[RegimeFlag, ...]:
        """Regime observations that are reported but do not gate (concept 15.1)."""
        return tuple(f for f in self.flags if f.severity == "info")

    @property
    def missing_metrics(self) -> tuple[MetricName, ...]:
        return tuple(f.metric for f in self.flags if f.missing)

    def summary(self) -> str:
        """One-line human-readable verdict."""
        verdict = "FAVORABLE" if self.favorable else "SKIP"
        head = f"{verdict} hard={self.hard_count} soft={self.soft_count}"
        if self.bucket:
            head = f"{head} bucket={self.bucket}"
        if self.reason:
            head = f"{head} reason={self.reason}"
        if self.degraded:
            head = f"{head} degraded={','.join(self.degraded)}"
        if not self.flags:
            return head
        return head + " | " + "; ".join(str(f) for f in self.flags)


def detect_volatility_regime(
    *,
    bucket: str,
    minutes_left: float,
    minutes_since_open: float,
    vix1d: float | None,
    vix1d_bucket_history: pd.Series | None,
    vix: float | None,
    term_structure_history: pd.Series | None,
    realized_range_pct: float | None,
    realized_vs_em_history: pd.Series | None,
    spx_closes: pd.Series | None,
    premium_spread_history: pd.Series | None,
    spx_price: float | None = None,
    zero_gamma_level: float | None = None,
    zgl_source: str | None = None,
    session_buckets: int | None = None,
    config: VolatilityRegimeConfig = DEFAULT_CONFIG,
) -> VolatilityRegimeResult:
    """Assess the intraday volatility regime for a 0DTE short-premium entry.

    Every ranked metric is compared against its own history **at the same clock
    time**, because VIX1D is not homogeneous intraday (concept 2). The caller
    supplies those bucket histories; building them from bars is an
    infrastructure concern.

    Parameters
    ----------
    bucket:
        Current 15-minute RTH bucket label, ``"HH:MM"`` ET, one of
        :data:`RTH_BUCKETS`. Anything else is reported as ``outside_rth``.
    minutes_left:
        Business minutes to the 16:15 ET VIX1D roll horizon. Used for the
        expected move under the √t convention and for the roll-degradation note.
    minutes_since_open:
        Business minutes since 09:30 ET. The first
        ``config.skip_first_minutes`` are not assessed (concept 15.3).
    vix1d:
        Current VIX1D. Basis of the base level, numerator of the term structure,
        premium-richness input and the expected move that normalises both
        ``realized_vs_em`` and the GEX distance. Without it there is no expected
        move and nothing to rank: reason ``no_vix1d``.
    vix1d_bucket_history:
        VIX1D readings of **previous** sessions in the same bucket, oldest
        first, today excluded. Only the last ``config.bucket_lookback`` usable
        observations are used.
    vix:
        Current VIX. Term-structure denominator and the absolute regime-shift
        guard.
    term_structure_history:
        Historical ``VIX1D / VIX`` in the same bucket, previous sessions only.
    realized_range_pct:
        ``(session high - session low) / session open × 100`` so far today.
    realized_vs_em_history:
        Historical ``realized_range_pct / EM_remaining`` in the same bucket,
        previous sessions only. This is the comparison base of the **only**
        ranked metric that gates, so its absence blocks the entry.
    spx_closes:
        SPX closes of **completed** sessions, oldest first; needs
        ``config.rv_window + 1`` usable values to produce RV20. Today's close
        does not exist at decision time and must not be included.
    premium_spread_history:
        Historical ``VIX1D - RV20`` in the same bucket, previous sessions only.
    spx_price:
        Current SPX price, for the GEX distance.
    zero_gamma_level:
        Zero Gamma Level from an external GEX calculation (e.g.
        :class:`~ibtws.unofficial.analysis.gex.GexCalculator`). Optional: GEX is
        opt-in, and every coverage figure in the concept was measured without it.
    zgl_source:
        Provider / version of the ZGL, recorded for reproducibility.
    session_buckets:
        Number of RTH buckets this session produced. Fewer than
        ``FULL_SESSION_BUCKETS - 2`` marks a half-day, where bucket alignment
        breaks and no assessment is produced (concept 6.5). ``None`` skips the
        check.
    config:
        Thresholds and windows; see :class:`VolatilityRegimeConfig`.

    Returns
    -------
    VolatilityRegimeResult
        ``favorable=True`` only when no hard flag fired and at most
        ``config.max_soft_flags`` soft flags did. :attr:`expected_move_pct` is
        exposed on the result because the risk layer needs it to scale the
        strike width (concept 15.2).

    Raises
    ------
    ValueError
        Never for market data — only via :class:`VolatilityRegimeConfig`
        validation, which is a caller programming error.

    Notes
    -----
    The calibration sample (251 sessions) contains no volatility stress event:
    VIX peaked at 30.4 and the worst day was -2.74% (concept 14.1). A tail filter
    calibrated on a sample without tails is an extrapolation, which is why the
    absolute cut-offs exist alongside the ranks. Break-even round-trip cost is
    0.89 on a 4.71 credit (concept 14.8), so execution quality dominates this
    signal by an order of magnitude, and position size dominates both
    (concept 15.8).
    """
    level = _clean_value(vix1d)
    vix_now = _clean_value(vix)
    spot = _clean_value(spx_price)
    zgl = _clean_value(zero_gamma_level)

    roll_weight = _roll_weight(minutes_left)
    variance_left = remaining_variance_share(bucket) or None
    base_metrics: dict[str, float | None] = {
        "vix1d": level,
        "vix": vix_now,
        "spx_price": spot,
        "minutes_left": float(minutes_left),
        "roll_weight_today": roll_weight,
        "variance_share_left": variance_left,
    }

    # ------------------------------------------------------------------
    # Unconditional refusals, ordered by how little they depend on data
    # ------------------------------------------------------------------
    if bucket not in BUCKET_VARIANCE_SHARE:
        return _refuse(OUTSIDE_RTH, bucket, minutes_left, roll_weight, variance_left, base_metrics, zgl_source)

    if session_buckets is not None and 0 < session_buckets < FULL_SESSION_BUCKETS - 2:
        # On a half-day the roll weight at a given clock time differs from a
        # regular session, so the bucket comparison base does not apply.
        return _refuse(SHORT_SESSION, bucket, minutes_left, roll_weight, variance_left, base_metrics, zgl_source)

    if config.skip_first_minutes > 0 and minutes_since_open < config.skip_first_minutes:
        return _refuse(OPENING_WINDOW, bucket, minutes_left, roll_weight, variance_left, base_metrics, zgl_source)

    if level is None:
        return _refuse(NO_VIX1D, bucket, minutes_left, roll_weight, variance_left, base_metrics, zgl_source)

    em_pct = expected_move_pct(level, minutes_left, bucket, use_profile=config.use_variance_profile)

    # ------------------------------------------------------------------
    # Ranked metrics — all compared inside the current time-of-day bucket
    # ------------------------------------------------------------------
    base = _ranked(level, vix1d_bucket_history, config)
    term_raw = level / vix_now if vix_now is not None else None
    term = _ranked(term_raw, term_structure_history, config)

    realized_raw = None
    if realized_range_pct is not None and em_pct:
        cleaned = _clean_value(realized_range_pct)
        realized_raw = cleaned / em_pct if cleaned is not None else None
    realized = _ranked(realized_raw, realized_vs_em_history, config)

    rv20 = _realised_vol(spx_closes, config.rv_window)
    spread_raw = level - rv20 if rv20 is not None else None
    premium = _ranked(spread_raw, premium_spread_history, config)

    flags: list[RegimeFlag] = []

    # ``realized_vs_em`` is the only ranked metric that gates (concept 15.1).
    ranked_metrics: tuple[tuple[MetricName, RankedMetric, bool, bool], ...] = (
        ("realized_vs_em", realized, False, config.gate_on_realized),
        ("base_level", base, False, config.gate_on_base_level),
        ("term_structure", term, False, config.gate_on_term_structure),
        ("premium_richness", premium, True, config.gate_on_premium_richness),
    )
    for metric_name, metric, lower_tail, gating in ranked_metrics:
        flag = _rank_flag(metric_name, metric, config, lower_tail=lower_tail, gating=gating)
        if flag is not None:
            flags.append(flag)

    # ------------------------------------------------------------------
    # Absolute cut-offs — not bucket-scaled, being absolute is the point
    # ------------------------------------------------------------------
    if level > config.vix1d_absolute_hard:
        flags.append(RegimeFlag("vix1d_absolute", "hard", level, f"VIX1D={level:.2f} > {config.vix1d_absolute_hard}"))
    elif level > config.vix1d_absolute_soft:
        flags.append(RegimeFlag("vix1d_absolute", "soft", level, f"VIX1D={level:.2f} > {config.vix1d_absolute_soft}"))

    if vix_now is not None and vix_now > config.vix_absolute_hard:
        flags.append(
            RegimeFlag(
                "vix_absolute",
                "hard",
                vix_now,
                f"VIX={vix_now:.2f} > {config.vix_absolute_hard} (regime shift)",
            )
        )

    # ------------------------------------------------------------------
    # GEX — opt-in; once a ZGL is supplied, failing to use it fails safe
    # ------------------------------------------------------------------
    degraded: list[str] = []
    distance_em: float | None = None
    if zgl is None:
        degraded.append("gex_not_evaluated")
    elif spot is None or not em_pct:
        flags.append(_missing("gex", "ZGL supplied but spot or expected move unavailable"))
    else:
        distance_em = (spot - zgl) / spot * 100.0 / em_pct
        if distance_em < -config.gex_buffer_em:
            flags.append(RegimeFlag("gex", "hard", distance_em, f"price {distance_em:.2f} EM below ZGL"))
        elif distance_em <= config.gex_buffer_em:
            flags.append(
                RegimeFlag(
                    "gex",
                    "soft",
                    distance_em,
                    f"price within ±{config.gex_buffer_em} EM of ZGL (dist={distance_em:.2f} EM)",
                )
            )

    # ------------------------------------------------------------------
    # Decision (concept 6.1)
    # ------------------------------------------------------------------
    base_regime = _base_regime(base.rank, config) if base.rank is not None else None
    if roll_weight is not None and roll_weight <= config.roll_degraded_below:
        degraded.append("vix1d_roll")
    if base.rank is not None and base.rank >= config.rank_soft:
        degraded.append("base_level_elevated")

    result = VolatilityRegimeResult(
        favorable=False,
        flags=tuple(flags),
        bucket=bucket,
        minutes_left=float(minutes_left),
        roll_weight_today=roll_weight,
        variance_share_left=variance_left,
        expected_move_pct=em_pct,
        base_rank=base.rank,
        base_regime=base_regime,
        degraded=tuple(degraded),
        metrics={
            **base_metrics,
            "expected_move_pct": em_pct,
            "base_rank": base.rank,
            "term_structure": term_raw,
            "term_structure_rank": term.rank,
            "realized_range_pct": _clean_value(realized_range_pct),
            "realized_vs_em": realized_raw,
            "realized_vs_em_rank": realized.rank,
            "rv20": rv20,
            "premium_spread": spread_raw,
            "premium_spread_rank": premium.rank,
            "zero_gamma_level": zgl,
            "zgl_distance_em": distance_em,
        },
        zgl_source=zgl_source,
    )
    favorable = result.hard_count == 0 and result.soft_count <= config.max_soft_flags
    result = replace(result, favorable=favorable)

    for flag in flags:
        if flag.missing:
            logger.warning(f"VolatilityRegime: {flag}")
        elif flag.severity == "info":
            logger.debug(f"VolatilityRegime: {flag}")
        else:
            logger.info(f"VolatilityRegime: {flag}")
    logger.info(f"VolatilityRegime: {result.summary()}")
    return result


# ----------------------------------------------------------------------
# Session clock and scale helpers
# ----------------------------------------------------------------------


def bucket_of(timestamp: pd.Timestamp) -> str:
    """Floor an Eastern-time timestamp to its 15-minute RTH bucket label."""
    floored = timestamp.floor(f"{BUCKET_MINUTES}min")
    return f"{floored.hour:02d}:{floored.minute:02d}"


def remaining_variance_share(bucket: str) -> float:
    """Share of the session's variance still ahead, from *bucket* onward.

    Replaces ``minutes_left / 405``. At 09:30 both give ~1.0; by 14:30 the
    minutes ratio says 0.26 while the measured profile says less, because the
    afternoon is quieter than a uniform clock implies (concept 15.5). Returns
    ``0.0`` for an unknown bucket so callers can treat it as "outside RTH".
    """
    if bucket not in BUCKET_VARIANCE_SHARE:
        return 0.0
    return sum(share for label, share in BUCKET_VARIANCE_SHARE.items() if label >= bucket)


def expected_move_pct(
    vix1d: float | None,
    minutes_left: float,
    bucket: str | None = None,
    *,
    use_profile: bool = True,
) -> float | None:
    """Expected move over the REST of the session, in percent of spot.

    A full-day denominator inflates every normalisation intraday and thereby
    understates risk: at 15:00 ET the remaining move is a fraction of the day's.

    With *use_profile* the time fraction is the measured share of daily variance
    still ahead rather than ``minutes_left / 405``. The profile reduces the drift
    of the scale — the realised-excursion-to-EM ratio goes from 0.92 → 0.47 under
    √t to 0.94 → 0.59, coefficient of variation 0.211 → 0.141. It is an
    improvement, not a cure: about a quarter of the bias survives, and the likely
    cause is the VIX1D roll rather than the clock (concept 15.5).

    This is also the scale a risk layer needs to express the strike width in
    expected moves instead of points (concept 15.2).
    """
    level = _clean_value(vix1d)
    if level is None:
        return None
    if use_profile and bucket:
        share = remaining_variance_share(bucket)
        if share > 0:
            return level / 100.0 / math.sqrt(TRADING_DAYS_PER_YEAR) * math.sqrt(share) * 100.0
    fraction = max(float(minutes_left), 0.0) / SESSION_MINUTES
    if fraction <= 0:
        return None
    return level / 100.0 / math.sqrt(TRADING_DAYS_PER_YEAR) * math.sqrt(fraction) * 100.0


def percentile_rank(history: pd.Series, current: float) -> float:
    """Share of *history* strictly below *current*, in percent (concept 4).

    Percentile rank, not min-max: robust to single outliers. A min-max IV Rank
    kept the base level pinned at LOW for 251 sessions after one spike.
    """
    return float((history < current).mean() * 100.0)


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

    Unusable values (NaN, non-numeric) are dropped and the window is filled from
    older history instead, so an isolated bad print does not shrink the sample.
    What is *not* papered over is an outright short history: a percentile over
    fewer sessions would silently change the scale's meaning, so it is reported
    as a data gap instead.

    Non-positive values are kept: unlike a volatility level, the premium spread
    ``VIX1D - RV20`` is legitimately negative — that is exactly its signal side.
    """
    if series is None or len(series) == 0:
        return None
    values = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < window:
        return None
    return values.iloc[-window:]


def _ranked(current: float | None, history: pd.Series | None, config: VolatilityRegimeConfig) -> RankedMetric:
    """Percentile rank of *current* inside its bucket history.

    The history must contain only **prior** sessions. Including the session being
    assessed caps the attainable rank at ``(n-1)/n`` = 98.3 for n = 60, so a p98
    threshold silently turns into "above all 59 priors" and fires about half as
    often as calibrated (concept 4).

    Caveat worth knowing (concept 14.3): these bucket series are autocorrelated,
    ac(1) = 0.66 for VIX1D and 0.68 for the premium spread, so 60 observations
    carry roughly 11 independent ones. A p98 read off this window is an
    extrapolation, and it tracks the last quarter's level — which is why the
    absolute cut-offs exist alongside it.
    """
    if current is None:
        return RankedMetric()
    window = _usable_window(history, config.bucket_lookback)
    if window is None:
        observations = 0 if history is None else int(pd.to_numeric(pd.Series(history), errors="coerce").notna().sum())
        return RankedMetric(raw=current, observations=observations)
    return RankedMetric(raw=current, rank=percentile_rank(window, current), observations=len(window))


def _rank_flag(
    metric: MetricName,
    ranked: RankedMetric,
    config: VolatilityRegimeConfig,
    *,
    lower_tail: bool,
    gating: bool,
) -> RegimeFlag | None:
    """One flag from one metric's rank.

    ``gating=False`` downgrades hard and soft to ``info``: the metric is measured
    and reported, but it cannot block. That is the version-5 treatment of the
    level metrics — they describe the regime, and the measurement says they
    describe it in the direction opposite to a risk filter (concept 14.4).

    Missing data still fails safe for a gating metric. For a non-gating one it is
    only noted, since blocking on the absence of a metric that must not block
    anyway would be incoherent.
    """
    if not ranked.available:
        detail = (
            f"only {ranked.observations}/{config.bucket_lookback} bucket observations"
            if ranked.raw is not None
            else "value unavailable"
        )
        if gating:
            return _missing(metric, detail)
        return RegimeFlag(metric, "info", ranked.raw, detail, missing=True)

    # ``available`` guarantees both fields; narrow for the type checker.
    rank = float(ranked.rank) if ranked.rank is not None else 0.0
    if lower_tail:
        hard_hit, soft_hit = rank < config.rank_hard_low, rank < config.rank_soft_low
        bound = config.rank_hard_low if hard_hit else config.rank_soft_low
        comparison = "<"
    else:
        hard_hit, soft_hit = rank > config.rank_hard, rank > config.rank_soft
        bound = config.rank_hard if hard_hit else config.rank_soft
        comparison = ">"
    if not (hard_hit or soft_hit):
        return None

    severity: Severity = ("hard" if hard_hit else "soft") if gating else "info"
    return RegimeFlag(metric, severity, ranked.raw, f"rank={rank:.0f} {comparison} {bound:.0f}")


def _base_regime(base_rank: float, config: VolatilityRegimeConfig) -> BaseRegime:
    """Descriptive label of the base level.

    Reporting only: version 5 does not gate on it, and with sizing out of scope
    it carries no mechanical consequence inside this module. ``LOW`` in
    particular is not a warning — low volatility means a smaller credit, not a
    worse outcome, and premium sufficiency is an entry rule, not a regime
    (concept 4.2).
    """
    if base_rank < config.base_rank_low:
        return "LOW"
    if base_rank > config.rank_hard:
        return "EXTREME"
    if base_rank >= config.rank_soft:
        return "HIGH"
    return "NORMAL"


def _realised_vol(closes: pd.Series | None, window: int) -> float | None:
    """Annualised close-to-close realised volatility over *window* sessions.

    Uses log returns of the last ``window + 1`` closes (i.e. ``window`` completed
    daily returns), sample standard deviation, annualised by ``√252`` and
    expressed in volatility points to match VIX1D.

    Daily closes, not intraday bars: intraday realised variance measures a
    different quantity, and the thresholds were calibrated on close-to-close
    (concept 4.5).
    """
    values = _usable_window(closes, window + 1)
    if values is None:
        return None
    array = values.to_numpy(dtype="float64")
    if np.any(array <= 0):
        return None
    returns = np.diff(np.log(array))
    if len(returns) < 2:
        return None
    std = float(np.std(returns, ddof=1))
    if not math.isfinite(std):
        return None
    return std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


def _roll_weight(minutes_left: float) -> float | None:
    """Approximate share of VIX1D carried by today's expiry (concept 2).

    Linear in remaining session minutes. Cboe's white-paper formula was not
    verified; this is an estimate used for reporting and for the degradation
    note, never inside a threshold.
    """
    try:
        remaining = float(minutes_left)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(remaining):
        return None
    return max(0.0, min(1.0, remaining / SESSION_MINUTES))


def _missing(metric: MetricName, detail: str) -> RegimeFlag:
    """Fail-safe flag for an unavailable gating metric (concept 6.4).

    Absent data is not evidence of a calm regime.
    """
    return RegimeFlag(metric, "hard", None, detail, missing=True)


def _refuse(
    reason: str,
    bucket: str,
    minutes_left: float,
    roll_weight: float | None,
    variance_left: float | None,
    metrics: dict[str, float | None],
    zgl_source: str | None,
) -> VolatilityRegimeResult:
    """No assessment: return an unfavorable result carrying only the context."""
    result = VolatilityRegimeResult(
        favorable=False,
        bucket=bucket,
        minutes_left=float(minutes_left),
        roll_weight_today=roll_weight,
        variance_share_left=variance_left,
        reason=reason,
        metrics=metrics,
        zgl_source=zgl_source,
    )
    logger.info(f"VolatilityRegime: {result.summary()}")
    return result
