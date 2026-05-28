"""Pre-market volatility regime detection for 0DTE SPX Iron Condor strategies.

Scores the current volatility environment on a 0–100 scale across 7 components
and classifies it into GREEN / YELLOW / RED regimes to guide trading decisions.
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
from ibtws.unofficial.option.iv_rank import IVRankCalculator

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
    action: str

    # Individual components
    vix1d_absolute: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    vix1d_vix_ratio: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    term_structure: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    vix9d_vix_ratio: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    overnight_vix_chg: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    iv_rank: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))
    iv_rv_spread: ComponentScore = field(default_factory=lambda: ComponentScore(0, 0))

    # Inputs used
    vix: float = 0.0
    vix1d: float = 0.0
    vix9d: float = 0.0
    vx_front: float = 0.0
    vix_prev_close: float = 0.0
    ivr: float = 0.0
    rv_20d: float = 0.0
    is_pre_market: bool = True

    @property
    def data_note(self) -> str:
        return "VIX1D/VIX9D: prev close (pre-market)" if self.is_pre_market else "all live"


def premarket_vol_regime(
    vix1d: float,
    vix9d: float,
    vix: float,
    vx_front: float,
    vix_prev_close: float,
    ivr: float,
    rv_20d: float,
    is_pre_market: bool = True,
) -> VolRegimeResult:
    """Score the volatility environment for 0DTE SPX strategies.

    Parameters
    ----------
    vix1d:
        VIX1D value (prev close if pre-market).
    vix9d:
        VIX9D value (prev close if pre-market).
    vix:
        VIX live value.
    vx_front:
        Front-month VX futures price (live).
    vix_prev_close:
        Yesterday's VIX closing value.
    ivr:
        IV Rank (0–100) from IVRankCalculator.
    rv_20d:
        20-day realized volatility of SPX (annualized %).
    is_pre_market:
        True if running before 09:30 EST.

    Returns
    -------
    VolRegimeResult with score (0–100), regime, and component breakdown.
    """
    # C1: VIX1D absolute level (0–25)
    if vix1d < 10:
        c1 = 0
    elif vix1d < 15:
        c1 = 8
    elif vix1d < 20:
        c1 = 16
    elif vix1d < 25:
        c1 = 21
    else:
        c1 = 25

    # C2: VIX1D / VIX ratio (0–20)
    r_1d_vix = vix1d / vix if vix > 0 else 1.0
    if r_1d_vix < 0.70:
        c2 = 0
    elif r_1d_vix < 0.85:
        c2 = 7
    elif r_1d_vix < 1.00:
        c2 = 13
    else:
        c2 = 20

    # C3: Term structure VIX / VX_front (0–20)
    r_ts = vix / vx_front if vx_front > 0 else 1.0
    if r_ts < 0.85:
        c3 = 0
    elif r_ts < 0.95:
        c3 = 7
    elif r_ts < 1.00:
        c3 = 13
    else:
        c3 = 20

    # C4: VIX9D / VIX ratio (0–15)
    r_9d_vix = vix9d / vix if vix > 0 else 1.0
    if r_9d_vix < 0.80:
        c4 = 0
    elif r_9d_vix < 0.95:
        c4 = 5
    elif r_9d_vix < 1.10:
        c4 = 10
    else:
        c4 = 15

    # C5: Overnight VIX change (0–10)
    overnight_chg = ((vix - vix_prev_close) / vix_prev_close * 100) if vix_prev_close > 0 else 0.0
    if overnight_chg < -10:
        c5 = 0
    elif overnight_chg < -3:
        c5 = 2
    elif overnight_chg < 5:
        c5 = 4
    elif overnight_chg < 15:
        c5 = 7
    else:
        c5 = 10

    # C6: IV Rank (0–5)
    if ivr < 30:
        c6 = 0
    elif ivr < 50:
        c6 = 1
    elif ivr < 70:
        c6 = 2
    elif ivr < 85:
        c6 = 4
    else:
        c6 = 5

    # C7: IV / RV spread (0–5)
    iv_rv = vix - rv_20d
    if iv_rv > 8:
        c7 = 0
    elif iv_rv > 4:
        c7 = 1
    elif iv_rv >= 0:
        c7 = 2
    elif iv_rv >= -2:
        c7 = 4
    else:
        c7 = 5

    total = c1 + c2 + c3 + c4 + c5 + c6 + c7

    if total < 30:
        regime: Regime = "GREEN"
        trade = True
        action = "Trade — standard wing width"
    elif total < 55:
        regime = "YELLOW"
        trade = True
        action = "Trade cautiously — widen wings by 20-30%"
    else:
        regime = "RED"
        trade = False
        action = "Skip the day or take a very wide IC"

    return VolRegimeResult(
        score=total,
        regime=regime,
        trade=trade,
        action=action,
        vix1d_absolute=ComponentScore(vix1d, c1),
        vix1d_vix_ratio=ComponentScore(round(r_1d_vix, 3), c2),
        term_structure=ComponentScore(round(r_ts, 3), c3),
        vix9d_vix_ratio=ComponentScore(round(r_9d_vix, 3), c4),
        overnight_vix_chg=ComponentScore(round(overnight_chg, 2), c5),
        iv_rank=ComponentScore(round(ivr, 1), c6),
        iv_rv_spread=ComponentScore(round(iv_rv, 2), c7),
        vix=vix,
        vix1d=vix1d,
        vix9d=vix9d,
        vx_front=vx_front,
        vix_prev_close=vix_prev_close,
        ivr=ivr,
        rv_20d=rv_20d,
        is_pre_market=is_pre_market,
    )


class VolRegimeDetector:
    """Fetches all data from IBKR and computes the pre-market volatility regime."""

    def __init__(self, client: IBKRClient) -> None:
        self._client = client

    async def detect(self) -> VolRegimeResult:
        """Run full regime detection using live IBKR data.

        All inputs (VIX, VIX1D, VIX9D, VX futures, 52w range, RV) are fetched
        from IBKR. VIX1D and VIX9D use previous close from historical data
        since they are not available live pre-market.
        """
        now = _dt.datetime.now(tz=_dt.timezone.utc).astimezone(_EST)
        is_pre_market = now.time() < PRE_MARKET_CUTOFF

        vix_live, vix_prev_close = await self._get_vix()
        vx_front = await self._get_vx_front()
        vix1d = await self._get_prev_close_or_last_price(Index("VIX1D", "CBOE"))
        vix9d = await self._get_prev_close_or_last_price(Index("VIX9D", "CBOE"))
        ivr = await self._get_iv_rank()
        rv_20d = await self._get_rv_20d()

        return premarket_vol_regime(
            vix1d=vix1d,
            vix9d=vix9d,
            vix=vix_live,
            vx_front=vx_front,
            vix_prev_close=vix_prev_close,
            ivr=ivr,
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

    async def _get_prev_close_or_last_price(self, contract: Index) -> float:
        """Return previous day's closing value for an index (VIX1D, VIX9D)."""
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

        # Fallback: try historical with TRADES
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
            logger.warning("VolRegime: no VX futures found, falling back to VIX value")
            return await self._vix_fallback()

        # Pick nearest expiration (front month)
        front = min(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
        try:
            self._client.ib.reqMarketDataType(3)  # delayed — VX requires subscription
            ticker = await self._client.get_market_data(front.contract)
            self._client.ib.reqMarketDataType(1)  # restore live
            price = ticker.last if not math.isnan(ticker.last) else ticker.close
            return float(price)
        except Exception as e:
            self._client.ib.reqMarketDataType(1)
            logger.warning("VolRegime: VX data unavailable (%s), falling back to VIX", e)
            return await self._vix_fallback()

    async def _vix_fallback(self) -> float:
        """Use VIX index as proxy for VX front when futures data unavailable."""
        contract = Index("VIX", "CBOE")
        ticker = await self._client.get_market_data(contract)
        price = ticker.last if not math.isnan(ticker.last) else ticker.close
        return float(price)

    async def _get_iv_rank(self) -> float:
        """Return IV Rank (0–100) for SPX using IVRankCalculator."""
        calculator = IVRankCalculator(self._client)
        contract = Index("SPX", "CBOE", "USD")
        result = await calculator.calculate(contract, lookback_days=252)
        if result.iv_rank is not None:
            return result.iv_rank
        logger.warning("VolRegime: IVRankCalculator returned no data, using 50")
        return 50.0

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
            return 15.0  # conservative fallback
        closes = df["close"].values[-21:]
        log_returns = np.diff(np.log(closes))
        return float(np.std(log_returns, ddof=1) * math.sqrt(252) * 100)
