"""Market structure and bias detection for short-term options trading.

Determines market condition (Bullish/Bearish/Neutral) and confidence level
by scoring three independent components — all sourced from IBKR API:

  C1: Price Structure (swing HH/HL/LH/LL on daily bars)  — ±2 points
  C2: Volatility Regime (composite GREEN/YELLOW/RED)      — ±1 point
  C3: Gap & VWAP positioning (ES futures for volume)      — ±1 point

Total score range: -4 to +4
  +3 to +4  → Strong Bullish    (HIGH/MEDIUM confidence)
  +1 to +2  → Lean Bullish      (MEDIUM/LOW confidence)
       0    → Neutral           (LOW confidence)
  -2 to -1  → Lean Bearish      (MEDIUM/LOW confidence)
  -4 to -3  → Strong Bearish    (HIGH/MEDIUM confidence)

Signal: Inherits TRADE/NOTRADE from VolRegimeResult.
  - NOTRADE overrides all — vol regime RED means sit out regardless of bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from ib_async import ContFuture, Index

from ibtws.unofficial.analysis.volatility import VolRegimeDetector, VolRegimeResult
from ibtws.unofficial.client import IBKRClient

logger = logging.getLogger(__name__)


class Bias(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    LEAN_BULLISH = "LEAN_BULLISH"
    NEUTRAL = "NEUTRAL"
    LEAN_BEARISH = "LEAN_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class Structure(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class MarketBiasResult:
    """Full market bias assessment."""

    bias: Bias
    score: int
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    signal: Literal["TRADE", "NOTRADE"]

    structure_score: int
    vol_regime_score: int
    gap_vwap_score: int

    structure: Structure
    vol_regime: VolRegimeResult | None
    spot: float
    prior_close: float
    vwap: float | None
    invalidation_level: float | None = None

    @property
    def high_vol(self) -> bool:
        """True when vol regime is YELLOW or RED."""
        return self.vol_regime is not None and self.vol_regime.regime != "GREEN"

    @property
    def action(self) -> str:
        """Strategy recommendation factoring bias × vol regime."""
        if self.signal == "NOTRADE":
            return "NO TRADE — vol regime RED, sit out entirely"
        hv = self.high_vol
        match self.bias:
            case Bias.STRONG_BULLISH:
                return (
                    "Bull put credit spreads (sell elevated premium)" if hv else "Buy calls / bull call debit spreads"
                )
            case Bias.LEAN_BULLISH:
                return "Reduced size — bull put credit spreads" if hv else "Reduced size — bull call debit spreads"
            case Bias.NEUTRAL:
                return "Iron condors / strangles (wide wings)" if hv else "Iron condors / calendars"
            case Bias.LEAN_BEARISH:
                return "Reduced size — bear call credit spreads" if hv else "Reduced size — bear put debit spreads"
            case Bias.STRONG_BEARISH:
                return "Bear call credit spreads (sell elevated premium)" if hv else "Buy puts / bear put debit spreads"


# ---------------------------------------------------------------------------
# Pure scoring functions (no I/O)
# ---------------------------------------------------------------------------


def detect_swing_points(highs: np.ndarray, lows: np.ndarray, pivot_len: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Detect swing highs/lows using N-bar pivot rule."""
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    half = pivot_len // 2

    for i in range(half, len(highs) - half):
        window = slice(i - half, i + half + 1)
        if highs[i] == np.max(highs[window]):
            swing_highs.append(float(highs[i]))
        if lows[i] == np.min(lows[window]):
            swing_lows.append(float(lows[i]))

    return np.array(swing_highs), np.array(swing_lows)


def classify_structure(highs: np.ndarray, lows: np.ndarray) -> tuple[Structure, float | None]:
    """Classify market structure from swing points.

    Returns (structure, invalidation_level):
      - Bullish → invalidation = most recent swing low
      - Bearish → invalidation = most recent swing high
      - Neutral → None
    """
    if len(highs) < 2 or len(lows) < 2:
        return Structure.NEUTRAL, None

    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return Structure.BULLISH, float(lows[-1])
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return Structure.BEARISH, float(highs[-1])
    return Structure.NEUTRAL, None


def score_structure(structure: Structure) -> int:
    """±2 for directional structure, 0 for neutral."""
    if structure == Structure.BULLISH:
        return 2
    if structure == Structure.BEARISH:
        return -2
    return 0


def score_vol_regime(vol_result: VolRegimeResult) -> int:
    """Use composite vol regime (6 components, 0–100 scale).

    GREEN → +1 (calm, bullish tailwind)
    YELLOW → 0 (caution)
    RED → -1 (stress, bearish headwind)
    """
    if vol_result.regime == "GREEN":
        return 1
    if vol_result.regime == "RED":
        return -1
    return 0


def score_gap_vwap(spot: float, prior_close: float, vwap: float | None) -> int:
    """+1 gapped up & above VWAP. -1 gapped down & below VWAP."""
    if prior_close <= 0:
        return 0
    gap_pct = (spot - prior_close) / prior_close * 100

    if vwap is None or vwap <= 0:
        if gap_pct > 0.3:
            return 1
        if gap_pct < -0.3:
            return -1
        return 0

    if gap_pct > 0.2 and spot > vwap:
        return 1
    if gap_pct < -0.2 and spot < vwap:
        return -1
    return 0


def compute_bias(score: int) -> tuple[Bias, Literal["HIGH", "MEDIUM", "LOW"]]:
    """Map total score → (bias, confidence)."""
    if score >= 4:
        return Bias.STRONG_BULLISH, "HIGH"
    if score == 3:
        return Bias.STRONG_BULLISH, "MEDIUM"
    if score == 2:
        return Bias.LEAN_BULLISH, "MEDIUM"
    if score == 1:
        return Bias.LEAN_BULLISH, "LOW"
    if score == 0:
        return Bias.NEUTRAL, "LOW"
    if score == -1:
        return Bias.LEAN_BEARISH, "LOW"
    if score == -2:
        return Bias.LEAN_BEARISH, "MEDIUM"
    if score == -3:
        return Bias.STRONG_BEARISH, "MEDIUM"
    return Bias.STRONG_BEARISH, "HIGH"


def compute_signal(
    bias: Bias,
    confidence: Literal["HIGH", "MEDIUM", "LOW"],
    vol_regime: VolRegimeResult | None,
) -> Literal["TRADE", "NOTRADE"]:
    """Derive TRADE/NOTRADE from all components combined.

    NOTRADE when:
      - Vol regime says NOTRADE (RED), OR
      - Bias is NEUTRAL with LOW confidence (no edge from any component)
    """
    if vol_regime and vol_regime.signal == "NOTRADE":
        return "NOTRADE"
    if bias == Bias.NEUTRAL and confidence == "LOW":
        return "NOTRADE"
    return "TRADE"


def detect_market_bias(
    *,
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
    spot: float,
    prior_close: float,
    vwap: float | None = None,
    vol_regime: VolRegimeResult | None = None,
    es_spot: float | None = None,
    es_prior_close: float | None = None,
) -> MarketBiasResult:
    """Compose all scoring components into a single MarketBiasResult.

    Pure function — no I/O. All market data must be pre-fetched and passed in.

    When *es_spot* / *es_prior_close* are provided they are used for gap & VWAP
    scoring (C3) instead of the index spot/prior_close, because ES futures have
    real volume and trade pre-market — giving a more accurate gap assessment.
    """
    structure, invalidation = classify_structure(swing_highs, swing_lows)
    s1 = score_structure(structure)
    s2 = score_vol_regime(vol_regime) if vol_regime else 0
    s3 = score_gap_vwap(
        es_spot if es_spot is not None else spot,
        es_prior_close if es_prior_close is not None else prior_close,
        vwap,
    )

    total = s1 + s2 + s3
    bias, confidence = compute_bias(total)
    signal = compute_signal(bias, confidence, vol_regime)

    return MarketBiasResult(
        bias=bias,
        score=total,
        confidence=confidence,
        signal=signal,
        structure_score=s1,
        vol_regime_score=s2,
        gap_vwap_score=s3,
        structure=structure,
        vol_regime=vol_regime,
        spot=spot,
        prior_close=prior_close,
        vwap=vwap,
        invalidation_level=invalidation,
    )


# ---------------------------------------------------------------------------
# IBKR integration
# ---------------------------------------------------------------------------


class MarketBiasDetector:
    """Fetches all data from IBKR and computes market bias."""

    def __init__(self, client: IBKRClient) -> None:
        self._client = client
        self._vol_detector = VolRegimeDetector(client)

    async def detect(self, symbol: str = "SPX", exchange: str = "CBOE") -> MarketBiasResult:
        """Run full bias detection using live IBKR data."""
        contract = Index(symbol, exchange)

        # Structure from daily bars
        df = await self._client.get_historical_data(
            contract, duration="3 M", bar_size="1 day", what_to_show="TRADES", use_rth=True
        )
        if df is None or len(df) < 10:
            logger.warning("MarketBias: insufficient data for %s", symbol)
            return _neutral_fallback()

        swing_highs, swing_lows = detect_swing_points(df["high"].values, df["low"].values, pivot_len=5)

        # SPX spot + prior close (for display / structure context)
        spot = float(df["close"].iloc[-1])
        prior_close = float(df["close"].iloc[-2])

        # ES data for gap & VWAP scoring (has volume + pre-market trading)
        es_spot, es_prior_close, vwap = await self._get_es_data()
        vol_regime = await self._vol_detector.detect()

        return detect_market_bias(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            spot=spot,
            prior_close=prior_close,
            vwap=vwap,
            vol_regime=vol_regime,
            es_spot=es_spot,
            es_prior_close=es_prior_close,
        )

    async def _get_es_data(self) -> tuple[float | None, float | None, float | None]:
        """Fetch ES futures spot, prior close, and VWAP.

        ES continuous futures provide volume and pre-market data that SPX lacks,
        making them ideal for gap & VWAP assessment.
        """
        try:
            contract = ContFuture(symbol="ES", exchange="CME")

            # Daily bars for spot + prior close
            daily = await self._client.get_market_data(contract)
            es_spot: float | None = None
            es_prior_close: float | None = None
            if daily is not None:
                es_spot = daily.last if daily.last is not None else float(daily.close)
                es_prior_close = float(daily.close) if daily.close is not None else None

            # Intraday bars for VWAP
            intra = await self._client.get_historical_data(
                contract, duration="1 D", bar_size="5 mins", what_to_show="TRADES", use_rth=False
            )
            vwap: float | None = None
            if intra is not None and not intra.empty and "volume" in intra.columns:
                vol = intra["volume"].values
                if vol.sum() > 0:
                    typical = (intra["high"].values + intra["low"].values + intra["close"].values) / 3
                    vwap = float((typical * vol).sum() / vol.sum())

            return es_spot, es_prior_close, vwap
        except Exception as e:
            logger.debug("MarketBias: ES data fetch failed: %s", e)
            return None, None, None


def _neutral_fallback() -> MarketBiasResult:
    """Returned when insufficient data — defaults to NOTRADE for safety."""
    return MarketBiasResult(
        bias=Bias.NEUTRAL,
        score=0,
        confidence="LOW",
        signal="NOTRADE",
        structure_score=0,
        vol_regime_score=0,
        gap_vwap_score=0,
        structure=Structure.NEUTRAL,
        vol_regime=None,
        spot=0.0,
        prior_close=0.0,
        vwap=None,
    )
