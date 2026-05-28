"""Pre-market volatility regime detection for 0DTE SPX premium-selling strategies.

Scores the current volatility environment on a 0–100 scale across 6 independent
components and classifies into GREEN / YELLOW / RED regimes.

Components (no redundancy — each measures something independent):
  C1: VIX absolute level (20 pts) — baseline risk magnitude
  C2: VIX/VIX3M term structure (25 pts) — THE leading regime indicator
  C3: VIX1D/VIX ratio (20 pts) — is today priced hot? (0DTE-specific)
  C4: VVIX level (20 pts) — vol-of-vol, can VIX spike unpredictably?
  C5: VVIX divergence (10 pts) — hidden institutional hedging
  C6: IV/RV spread / VRP (5 pts) — direct edge measurement

Hard override: negative VRP forces YELLOW minimum (no edge = no full-size trade).
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from ib_async import Index, Future

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import safe_pick_value

logger = logging.getLogger(__name__)

Regime = Literal["GREEN", "YELLOW", "RED"]

PRE_MARKET_CUTOFF = _dt.time(9, 30)
_EST = _dt.timezone(_dt.timedelta(hours=-5))


@dataclass(frozen=True)
class ComponentScore:
    """Single scoring component result."""

    value: float
    score: int


@dataclass(frozen=True)
class VolRegimeResult:
    """Full volatility regime assessment."""

    score: int
    regime: Regime
    trade: bool
    signal: Literal["TRADE", "NOTRADE"]
    action: str
    vrp_override: bool = False

    # Individual components (6 total, sum to 100 max)
    vix_absolute: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    term_structure: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    vix1d_vix_ratio: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    vvix_level: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    vvix_divergence: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    iv_rv_spread: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))

    # Inputs used
    vix: float = 0.0
    vix1d: float = 0.0
    vix3m: float = 0.0
    vvix: float = 0.0
    vvix_prev_close: float = 0.0
    vx_front: float = 0.0
    vix_prev_close: float = 0.0
    rv_20d: float = 0.0
    is_pre_market: bool = True

    @property
    def data_note(self) -> str:
        parts = []
        if self.is_pre_market:
            parts.append("VIX1D/VVIX: prev close (pre-market)")
        else:
            parts.append("all live")
        if self.vrp_override:
            parts.append("⚠️ VRP override active — negative edge")
        return " | ".join(parts)


def premarket_vol_regime(
    vix1d: float,
    vix: float,
    vix3m: float,
    vvix: float,
    vvix_prev_close: float,
    vx_front: float,
    vix_prev_close: float,
    rv_20d: float,
    is_pre_market: bool = True,
) -> VolRegimeResult:
    """Score the volatility environment for 0DTE SPX premium-selling.

    Higher score = more dangerous for premium selling.
    """
    # C1: VIX absolute level (0–20)
    if vix < 14:
        c1 = 0
    elif vix < 18:
        c1 = 5
    elif vix < 22:
        c1 = 10
    elif vix < 28:
        c1 = 15
    else:
        c1 = 20

    # C2: Term structure — VIX / VIX3M ratio (0–25)
    # Backwardation (ratio > 1) precedes every major vol event.
    # Falls back to VX front if VIX3M unavailable.
    r_ts = vix / vix3m if vix3m > 0 else vix / vx_front if vx_front > 0 else 1.0
    if r_ts < 0.85:
        c2 = 0
    elif r_ts < 0.92:
        c2 = 5
    elif r_ts < 0.98:
        c2 = 10
    elif r_ts < 1.02:
        c2 = 15
    elif r_ts < 1.08:
        c2 = 20
    else:
        c2 = 25

    # C3: VIX1D / VIX ratio (0–20)
    # Is today priced hotter than the month? Directly relevant to 0DTE.
    r_1d_vix = vix1d / vix if vix > 0 else 1.0
    if r_1d_vix < 0.70:
        c3 = 0
    elif r_1d_vix < 0.85:
        c3 = 5
    elif r_1d_vix < 1.00:
        c3 = 12
    else:
        c3 = 20

    # C4: VVIX level (0–20)
    # Vol-of-vol. When VVIX is high, VIX can spike unpredictably.
    if vvix < 90:
        c4 = 0
    elif vvix < 105:
        c4 = 5
    elif vvix < 120:
        c4 = 10
    elif vvix < 140:
        c4 = 15
    else:
        c4 = 20

    # C5: VVIX divergence (0–10)
    # Rising VVIX while VIX flat/falling = institutions buying protection quietly.
    vvix_chg = ((vvix - vvix_prev_close) / vvix_prev_close * 100) if vvix_prev_close > 0 else 0.0
    vix_chg = ((vix - vix_prev_close) / vix_prev_close * 100) if vix_prev_close > 0 else 0.0
    if vvix_chg > 8 and vix_chg < 2:
        c5 = 10
    elif vvix_chg > 5 and vix_chg < 3:
        c5 = 6
    elif vvix_chg > 3 and vix_chg < 1:
        c5 = 3
    else:
        c5 = 0

    # C6: IV / RV spread — Variance Risk Premium (0–5)
    # Direct measurement of your edge. Wide = safe, negative = no edge.
    iv_rv = vix - rv_20d
    if iv_rv > 8:
        c6 = 0
    elif iv_rv > 4:
        c6 = 1
    elif iv_rv >= 0:
        c6 = 2
    elif iv_rv >= -2:
        c6 = 4
    else:
        c6 = 5

    total = c1 + c2 + c3 + c4 + c5 + c6

    # Hard override: negative VRP forces YELLOW minimum.
    # No statistical edge = no full-size premium selling regardless of score.
    vrp_override = iv_rv < 0

    if total < 25 and not vrp_override:
        regime: Regime = "GREEN"
        trade = True
        action = "Full size — standard wing width, positive VRP environment"
    elif total < 50:
        regime = "YELLOW"
        trade = True
        if vrp_override:
            action = "Reduced size — VRP negative, edge is thin or absent"
        else:
            action = "Reduced size — widen wings 20-30%, defined-risk only"
    else:
        regime = "RED"
        trade = False
        action = "No premium selling — sit out or minimal defined-risk with 50%+ wing width"

    return VolRegimeResult(
        score=total,
        regime=regime,
        trade=trade,
        signal="TRADE" if trade else "NOTRADE",
        action=action,
        vrp_override=vrp_override,
        vix_absolute=ComponentScore(vix, c1),
        term_structure=ComponentScore(round(r_ts, 3), c2),
        vix1d_vix_ratio=ComponentScore(round(r_1d_vix, 3), c3),
        vvix_level=ComponentScore(vvix, c4),
        vvix_divergence=ComponentScore(round(vvix_chg, 2), c5),
        iv_rv_spread=ComponentScore(round(iv_rv, 2), c6),
        vix=vix,
        vix1d=vix1d,
        vix3m=vix3m,
        vvix=vvix,
        vvix_prev_close=vvix_prev_close,
        vx_front=vx_front,
        vix_prev_close=vix_prev_close,
        rv_20d=rv_20d,
        is_pre_market=is_pre_market,
    )


class VolRegimeDetector:
    """Fetches all data from IBKR and computes the pre-market volatility regime."""

    def __init__(self, client: IBKRClient) -> None:
        self._client = client

    async def detect(self) -> VolRegimeResult:
        """Run full regime detection using live IBKR data."""
        now = _dt.datetime.now(tz=_dt.timezone.utc).astimezone(_EST)
        is_pre_market = now.time() < PRE_MARKET_CUTOFF

        vix_live, vix_prev_close = await self._get_vix()
        vx_front = await self._get_vx_front()
        vix1d = await self._get_prev_close_or_last_price(Index("VIX1D", "CBOE"))
        vix3m = await self._get_prev_close_or_last_price(Index("VIX3M", "CBOE"))
        vvix, vvix_prev_close = await self._get_vvix()
        rv_20d = await self._get_rv_20d()

        return premarket_vol_regime(
            vix1d=vix1d,
            vix=vix_live,
            vix3m=vix3m,
            vvix=vvix,
            vvix_prev_close=vvix_prev_close,
            vx_front=vx_front,
            vix_prev_close=vix_prev_close,
            rv_20d=rv_20d,
            is_pre_market=is_pre_market,
        )

    async def _get_vix(self) -> tuple[float, float]:
        """Return (vix_live, vix_prev_close)."""
        contract = Index("VIX", "CBOE")
        ticker = await self._client.get_market_data(contract)
        vix_live = ticker.last if not math.isnan(ticker.last) else ticker.close
        vix_prev_close = ticker.close if not math.isnan(ticker.close) else vix_live
        return float(vix_live), float(vix_prev_close)

    async def _get_vvix(self) -> tuple[float, float]:
        """Return (vvix_current, vvix_prev_close)."""
        contract = Index("VVIX", "CBOE")
        try:
            ticker = await self._client.get_market_data(contract)
            current = ticker.last if not math.isnan(ticker.last) else ticker.close
            prev_close = ticker.close if not math.isnan(ticker.close) else current
            if not math.isnan(current) and current > 0:
                return float(current), float(prev_close)
        except Exception as e:
            logger.warning("VolRegime: market data failed for VVIX: %s", e)

        try:
            df = await self._client.get_historical_data(
                contract,
                duration="5 D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
            )
            if df is not None and len(df) >= 2:
                return float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
            if df is not None and not df.empty:
                val = float(df["close"].iloc[-1])
                return val, val
        except Exception as e:
            logger.warning("VolRegime: historical data failed for VVIX: %s", e)

        logger.warning("VolRegime: no VVIX data, using default 95")
        return 95.0, 95.0

    async def _get_prev_close_or_last_price(self, contract: Index) -> float:
        """Return previous day's closing value for an index."""
        try:
            ticker = await self._client.get_market_data(contract)
            close = ticker.close
            for attr in ("marketPrice", "last", "close"):
                v = safe_pick_value(ticker, attr)
                if v is not None:
                    close = v
                    break
            if not math.isnan(close) and close > 0:
                return float(close)
        except Exception as e:
            logger.warning("VolRegime: market data failed for %s: %s", contract.symbol, e)

        try:
            df = await self._client.get_historical_data(
                contract,
                duration="5 D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
            )
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("VolRegime: historical data failed for %s: %s", contract.symbol, e)

        logger.warning("VolRegime: no data for %s, using 0", contract.symbol)
        return 0.0

    async def _get_vx_front(self) -> float:
        """Return front-month VX futures price."""
        query = Future(symbol="VX", exchange="CFE", tradingClass="VX")
        details = await self._client.ib.reqContractDetailsAsync(query)
        if not details:
            logger.warning("VolRegime: no VX futures found, falling back to VIX")
            return await self._vix_fallback()

        front = min(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
        try:
            self._client.ib.reqMarketDataType(3)
            ticker = await self._client.get_market_data(front.contract)
            self._client.ib.reqMarketDataType(1)
            price = ticker.last if not math.isnan(ticker.last) else ticker.close
            return float(price)
        except Exception as e:
            self._client.ib.reqMarketDataType(1)
            logger.warning("VolRegime: VX data unavailable (%s), falling back to VIX", e)
            return await self._vix_fallback()

    async def _vix_fallback(self) -> float:
        contract = Index("VIX", "CBOE")
        ticker = await self._client.get_market_data(contract)
        price = ticker.last if not math.isnan(ticker.last) else ticker.close
        return float(price)

    async def _get_rv_20d(self) -> float:
        """Return 20-day annualized realized volatility of SPX (as %)."""
        contract = Index("SPX", "CBOE")
        df = await self._client.get_historical_data(
            contract,
            duration="30 D",
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )
        if df is None or df.empty or len(df) < 5:
            logger.warning("VolRegime: insufficient SPX data for RV calc")
            return 15.0
        closes = df["close"].values[-21:]
        log_returns = np.diff(np.log(closes))
        return float(np.std(log_returns, ddof=1) * math.sqrt(252) * 100)
