"""SPX Volatility Analysis System for Options Premium Selling.

Implements the full automated volatility analysis pipeline:
  Metric 1: VIX Z-Score (20-day)
  Metric 2: Expected Move (EM)
  Metric 3: Term Structure Indicators
  Metric 4: Volatility Risk Premium (VRP)
  Metric 5: Volatility Skew Slope Ratio

Rules Engine produces GREEN / YELLOW / RED signals for 0DTE, Weekly, and Monthly
strategies, plus a global circuit breaker.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from ib_async import Index, Future, Contract

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import safe_pick_value
from ibtws.unofficial.analysis.expected_move import ExpectedMoveCalculator
from ibtws.unofficial.option.chains import OptionChainFetcher

logger = logging.getLogger(__name__)

Signal = Literal["GREEN", "YELLOW", "RED"]
_EST = _dt.timezone(_dt.timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TermStructure:
    ratio_macro: float  # VIX / VIX3M
    ratio_weekly: float  # VIX9D / VIX (short over long)
    ratio_intraday: float  # VIX1D / VIX
    slope_futures: float  # VIX_F2 / VIX_F1

    @property
    def macro_state(self) -> str:
        return "BACKWARDATION" if self.ratio_macro > 1.0 else "CONTANGO"

    @property
    def weekly_state(self) -> str:
        return "LOCAL BACKWARDATION" if self.ratio_weekly > 1.0 else "CONTANGO"

    @property
    def intraday_state(self) -> str:
        return "BACKWARDATION" if self.ratio_intraday > 1.0 else "CONTANGO"


@dataclass(frozen=True)
class StrategySignal:
    signal: Signal
    reason: str
    recommended_strike: float | None = None


@dataclass(frozen=True)
class SPXVolReport:
    """Complete system output."""

    # Market context
    spx: float
    vix: float
    vix_zscore: float
    vvix: float
    vvix_declining: bool

    # Term structure
    term_structure: TermStructure

    # Metrics
    expected_move: float  # points
    vrp: float  # VIX - RV20
    rv_20: float
    skew_slope: float | None  # IV(15Δ put) - IV(50Δ ATM)
    skew_slope_20d_avg: float | None
    skew_ratio: float | None  # current / 20d avg

    # Strategy signals
    signal_0dte: StrategySignal
    signal_weekly: StrategySignal
    signal_monthly: StrategySignal

    # Global circuit breaker
    circuit_breaker: bool
    circuit_breaker_reasons: list[str] = field(default_factory=list)

    # Timestamp
    timestamp: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(tz=_EST))

    @property
    def regime(self) -> str:
        if self.vix_zscore > 2.0:
            return "High Volatility / Panic"
        elif self.vix_zscore > 1.0:
            return "Elevated Volatility"
        elif self.vix_zscore < -1.0:
            return "Low Volatility / Complacency"
        return "Normal"


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SPXVolAnalyzer:
    """Fetches live data from IBKR and produces the full SPX volatility report."""

    def __init__(self, client: IBKRClient) -> None:
        self._client = client
        self._chain = OptionChainFetcher(client)
        self._em_calc = ExpectedMoveCalculator(client, self._chain)

    async def analyze(self, dte_weekly: int = 7) -> SPXVolReport:
        """Run the full analysis pipeline and return the report."""
        # Fetch all market data
        spx = await self._get_price(Index("SPX", "CBOE"))
        vix = await self._get_price(Index("VIX", "CBOE"))
        vix1d = await self._get_price(Index("VIX1D", "CBOE"))
        vix9d = await self._get_price(Index("VIX9D", "CBOE"))
        vix3m = await self._get_price(Index("VIX3M", "CBOE"))
        vvix, vvix_prev = await self._get_vvix()
        vix_f1, vix_f2 = await self._get_vx_futures()

        # Previous day values for reversal detection
        vix1d_prev = await self._get_prev_close(Index("VIX1D", "CBOE"))
        vix9d_prev = await self._get_prev_close(Index("VIX9D", "CBOE"))
        vix_prev = await self._get_prev_close(Index("VIX", "CBOE"))

        # Metric 1: VIX Z-Score (20-day)
        vix_history = await self._get_vix_history(20)
        vix_zscore = self._calc_zscore(vix, vix_history)

        # Metric 2: Expected Move (average of straddle, IV, and HV methods)
        em = await self._calc_expected_move(dte_weekly)

        # Metric 3: Term Structure (short-over-long ratios)
        ts = TermStructure(
            ratio_macro=vix / vix3m if vix3m > 0 else 1.0,
            ratio_weekly=vix9d / vix if vix > 0 else 1.0,
            ratio_intraday=vix1d / vix if vix > 0 else 1.0,
            slope_futures=vix_f2 / vix_f1 if vix_f1 > 0 else 1.0,
        )

        # Reversal detection: current ratio declining from previous
        prev_ratio_0dte = vix1d_prev / vix_prev if vix_prev > 0 else 1.0
        ratio_0dte_reversing = ts.ratio_intraday < prev_ratio_0dte

        prev_ratio_weekly = vix9d_prev / vix_prev if vix_prev > 0 else 1.0
        ratio_weekly_reversing = ts.ratio_weekly < prev_ratio_weekly

        # Metric 4: VRP
        rv_20 = await self._calc_rv20()
        vrp = vix - rv_20

        # Metric 5: Skew (best-effort, may be None if chain unavailable)
        skew_slope, skew_avg, skew_ratio = await self._calc_skew(spx)

        # VVIX declining check
        vvix_declining = vvix < vvix_prev

        # Rules Engine
        circuit_breaker, cb_reasons = self._check_circuit_breaker(ts, vix_zscore)

        signal_0dte = self._eval_0dte(ts, vvix_declining, ratio_0dte_reversing, spx, em)
        signal_weekly = self._eval_weekly(vix_zscore, ts, vrp, ratio_weekly_reversing, spx, em)
        signal_monthly = self._eval_monthly(vix_zscore, ts, vrp, spx, em)

        if circuit_breaker:
            signal_0dte = StrategySignal("RED", "Circuit breaker active")
            signal_weekly = StrategySignal("RED", "Circuit breaker active")
            signal_monthly = StrategySignal("RED", "Circuit breaker active")

        return SPXVolReport(
            spx=spx,
            vix=vix,
            vix_zscore=vix_zscore,
            vvix=vvix,
            vvix_declining=vvix_declining,
            term_structure=ts,
            expected_move=em,
            vrp=vrp,
            rv_20=rv_20,
            skew_slope=skew_slope,
            skew_slope_20d_avg=skew_avg,
            skew_ratio=skew_ratio,
            signal_0dte=signal_0dte,
            signal_weekly=signal_weekly,
            signal_monthly=signal_monthly,
            circuit_breaker=circuit_breaker,
            circuit_breaker_reasons=cb_reasons,
        )

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_zscore(current: float, history: np.ndarray) -> float:
        # Match Pine Script behavior: include current value in the window
        # This gives ta.sma/ta.stdev equivalent (in-sample Z-score)
        if len(history) < 2:
            return 0.0
        window = np.append(history, current)[-len(history) :]
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=0))  # Pine uses population std (ddof=0)
        if sigma == 0:
            return 0.0
        return (current - mu) / sigma

    async def _calc_expected_move(self, dte: int) -> float:
        """Average of straddle, IV-based, and HV-based expected move methods."""
        underlying = Index("SPX", "CBOE")
        await self._client.ib.qualifyContractsAsync(underlying)
        # Find expiration closest to target DTE
        target = (_dt.date.today() + _dt.timedelta(days=dte)).strftime("%Y%m%d")
        try:
            defn = await self._chain.fetch_chain_definition(underlying, exchange="SMART", trading_class="SPXW")
            today_str = _dt.date.today().strftime("%Y%m%d")
            future_exps = [e for e in defn.expirations if e > today_str]
            if future_exps:
                expiration = min(future_exps, key=lambda e: abs(int(e) - int(target)))
            else:
                expiration = target
            result = await self._em_calc.calculate(underlying, expiration, trading_class="SPXW")
            values = [v for v in (result.straddle_move, result.iv_move, result.hv_move) if v is not None]
            if values:
                return sum(values) / len(values)
        except Exception as e:
            logger.warning("SPXVol: ExpectedMoveCalculator failed: %s", e)
        # Fallback formula: Price × (VIX/100) × √(DTE/365)
        # VIX is annualized over calendar days (365), not trading days (252)
        spx = await self._get_price(Index("SPX", "CBOE"))
        vix = await self._get_price(Index("VIX", "CBOE"))
        return spx * (vix / 100.0) * math.sqrt(max(dte, 1) / 365.0)

    async def _calc_rv20(self) -> float:
        """20-day annualized realized volatility of SPX (as %)."""
        contract = Index("SPX", "CBOE")
        df = await self._client.get_historical_data(
            contract, duration="30 D", bar_size="1 day", what_to_show="TRADES", use_rth=True
        )
        if df is None or df.empty or len(df) < 5:
            return 15.0
        closes = df["close"].values[-21:]
        log_returns = np.diff(np.log(closes))
        return float(np.std(log_returns, ddof=1) * math.sqrt(252) * 100)

    async def _calc_skew(self, spot: float) -> tuple[float | None, float | None, float | None]:
        """Skew = IV(15Δ put) - IV(50Δ ATM). Returns (current, 20d_avg, ratio)."""
        try:
            underlying = Index("SPX", "CBOE")
            await self._client.ib.qualifyContractsAsync(underlying)
            defn = await self._chain.fetch_chain_definition(underlying, exchange="SMART", trading_class="SPXW")
            # Pick nearest weekly expiration
            today = _dt.date.today().strftime("%Y%m%d")
            future_exps = [e for e in defn.expirations if e > today]
            if not future_exps:
                return None, None, None
            target_exp = future_exps[0]

            quotes = await self._chain.fetch_snapshot(
                underlying, expirations=[target_exp], strike_window_pct=0.10, rights=("C", "P"), trading_class="SPXW"
            )
            if not quotes:
                return None, None, None

            # Find 50Δ (ATM) and ~15Δ put
            puts = [q for q in quotes if q.contract.right == "P" and q.delta is not None]
            if not puts:
                return None, None, None

            atm_put = min(puts, key=lambda q: abs((q.delta or 0) + 0.50))
            d15_put = min(puts, key=lambda q: abs((q.delta or 0) + 0.15))

            iv_atm = atm_put.iv
            iv_15d = d15_put.iv
            if iv_atm is None or iv_15d is None:
                return None, None, None

            skew_slope = (iv_15d - iv_atm) * 100  # in vol points
            # 20d average would require historical chain data; approximate as None
            return skew_slope, None, None
        except Exception as e:
            logger.warning("Skew calculation failed: %s", e)
            return None, None, None

    # ------------------------------------------------------------------
    # Rules Engine
    # ------------------------------------------------------------------

    @staticmethod
    def _check_circuit_breaker(ts: TermStructure, zscore: float) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if ts.ratio_macro > 1.15:
            reasons.append(f"VIX/VIX3M={ts.ratio_macro:.2f} > 1.15 (deep backwardation)")
        if zscore < -2.0:
            reasons.append(f"VIX Z-Score={zscore:.2f} < -2.0 (extreme squeeze)")
        if ts.slope_futures < 0.95:
            reasons.append(f"F2/F1={ts.slope_futures:.2f} < 0.95 (futures inverted)")
        return len(reasons) > 0, reasons

    def _eval_0dte(
        self, ts: TermStructure, vvix_declining: bool, ratio_reversing: bool, spx: float, em: float
    ) -> StrategySignal:
        now = _dt.datetime.now(tz=_EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        min_after_open = (now - market_open).total_seconds() / 60.0

        if min_after_open < 20:
            return StrategySignal("YELLOW", "Wait — less than 20 min after open")

        ratio = ts.ratio_intraday
        if ratio > 1.25 and ratio_reversing and vvix_declining:
            strike = spx - 1.5 * em
            return StrategySignal("GREEN", f"VIX1D/VIX={ratio:.2f} reversal confirmed + VVIX declining", round(strike))

        if ratio > 1.25 and not ratio_reversing:
            return StrategySignal("YELLOW", f"VIX1D/VIX={ratio:.2f} spiking but no reversal yet")

        if ratio > 1.10:
            return StrategySignal("YELLOW", f"VIX1D/VIX={ratio:.2f} elevated, waiting for spike >1.25")

        return StrategySignal("YELLOW", "No intraday panic spike detected")

    def _eval_weekly(
        self, zscore: float, ts: TermStructure, vrp: float, ratio_reversing: bool, spx: float, em: float
    ) -> StrategySignal:
        # VIX9D/VIX > 1.15 with downward reversal + VRP > 0
        ratio = ts.ratio_weekly
        if ratio > 1.15 and ratio_reversing and vrp > 0:
            strike = spx - 2.0 * em
            return StrategySignal("GREEN", f"VIX9D/VIX={ratio:.2f} reversal, VRP={vrp:.1f}", round(strike))

        reasons = []
        if ratio <= 1.15:
            reasons.append(f"VIX9D/VIX={ratio:.2f} (need >1.15 spike)")
        elif not ratio_reversing:
            reasons.append(f"VIX9D/VIX={ratio:.2f} still rising (need reversal)")
        if vrp <= 0:
            reasons.append(f"VRP={vrp:.1f} (need >0)")
        return StrategySignal("YELLOW", "; ".join(reasons))

    def _eval_monthly(self, zscore: float, ts: TermStructure, vrp: float, spx: float, em: float) -> StrategySignal:
        # Monthly needs: elevated vol, contango, positive VRP
        if zscore > 1.0 and ts.ratio_macro < 1.0 and vrp > 2.0:
            strike = spx - 2.5 * em
            return StrategySignal("GREEN", f"Z={zscore:.2f}, contango, VRP={vrp:.1f}", round(strike))

        reasons = []
        if zscore <= 1.0:
            reasons.append(f"Z-Score={zscore:.2f} (need >1.0)")
        if ts.ratio_macro >= 1.0:
            reasons.append("Macro backwardation")
        if vrp <= 2.0:
            reasons.append(f"VRP={vrp:.1f} (need >2.0)")
        return StrategySignal("YELLOW", "; ".join(reasons))

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------

    async def _get_price(self, contract: Contract) -> float:
        try:
            ticker = await self._client.get_market_data(contract)
            for attr in ("last", "close"):
                v = safe_pick_value(ticker, attr)
                if v is not None:
                    return v
        except Exception as e:
            logger.warning("SPXVol: market data failed for %s: %s", contract.symbol, e)
        # Fallback to historical
        try:
            df = await self._client.get_historical_data(
                contract, duration="5 D", bar_size="1 day", what_to_show="TRADES", use_rth=True
            )
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("SPXVol: historical fallback failed for %s: %s", contract.symbol, e)
        return 0.0

    async def _get_prev_close(self, contract: Contract) -> float:
        """Return previous day's closing value for reversal detection."""
        try:
            df = await self._client.get_historical_data(
                contract, duration="5 D", bar_size="1 day", what_to_show="TRADES", use_rth=True
            )
            if df is not None and len(df) >= 2:
                return float(df["close"].iloc[-2])
        except Exception as e:
            logger.warning("SPXVol: prev close failed for %s: %s", contract.symbol, e)
        return await self._get_price(contract)

    async def _get_vvix(self) -> tuple[float, float]:
        """Return (current, prev_close)."""
        contract = Index("VVIX", "CBOE")
        try:
            ticker = await self._client.get_market_data(contract)
            current = safe_pick_value(ticker, "last") or safe_pick_value(ticker, "close") or 0.0
            prev = safe_pick_value(ticker, "close") or current
            if current > 0:
                return current, prev
        except Exception:
            pass
        try:
            df = await self._client.get_historical_data(
                contract, duration="5 D", bar_size="1 day", what_to_show="TRADES", use_rth=True
            )
            if df is not None and len(df) >= 2:
                return float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        except Exception:
            pass
        return 95.0, 95.0

    async def _get_vx_futures(self) -> tuple[float, float]:
        """Return (front_month, second_month) VX futures prices."""
        query = Future(symbol="VX", exchange="CFE", tradingClass="VX")
        try:
            details = await self._client.ib.reqContractDetailsAsync(query)
            if not details or len(details) < 2:
                vix = await self._get_price(Index("VIX", "CBOE"))
                return vix, vix * 1.05  # rough fallback
            sorted_d = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
            f1 = await self._get_futures_price(sorted_d[0].contract)
            f2 = await self._get_futures_price(sorted_d[1].contract)
            return f1, f2
        except Exception as e:
            logger.warning("SPXVol: VX futures fetch failed: %s", e)
            vix = await self._get_price(Index("VIX", "CBOE"))
            return vix, vix * 1.05

    async def _get_futures_price(self, contract: Contract) -> float:
        self._client.ib.reqMarketDataType(3)
        try:
            ticker = await self._client.get_market_data(contract)
            for attr in ("last", "close"):
                v = safe_pick_value(ticker, attr)
                if v is not None:
                    return v
        finally:
            self._client.ib.reqMarketDataType(1)
        return 0.0

    async def _get_vix_history(self, days: int) -> np.ndarray:
        contract = Index("VIX", "CBOE")
        df = await self._client.get_historical_data(
            contract, duration="30 D", bar_size="1 day", what_to_show="TRADES", use_rth=True
        )
        if df is None or df.empty:
            return np.array([])
        return df["close"].values[-days:]


# ---------------------------------------------------------------------------
# Dashboard formatter
# ---------------------------------------------------------------------------


def format_report(r: SPXVolReport) -> str:
    """Format the report as the dashboard text block."""
    ts = r.term_structure
    lines = [
        "=" * 68,
        "               SPX VOLATILITY ANALYSIS SYSTEM REPORT",
        "=" * 68,
        "[MARKET CONTEXT]",
        f"SPX: {r.spx:.0f} | VIX: {r.vix:.2f} (Z-Score: {r.vix_zscore:+.2f}) | VVIX: {r.vvix:.2f} (Declining: {'YES' if r.vvix_declining else 'NO'})",
        f"Market Regime: {r.regime}",
        "",
        "[TERM STRUCTURE]",
        f"F2/F1 Spread: {ts.slope_futures:.2f} ({ts.intraday_state})",
        f"VIX/VIX3M:    {ts.ratio_macro:.2f} ({ts.macro_state})",
        f"VIX9D/VIX:    {ts.ratio_weekly:.2f} ({ts.weekly_state})",
        "",
        "[STRATEGY EVALUATION]",
        f"-> 0DTE Strategy:   [{r.signal_0dte.signal}] ({r.signal_0dte.reason})",
    ]

    weekly_extra = (
        f" Recommended short strike: < {r.signal_weekly.recommended_strike}"
        if r.signal_weekly.recommended_strike
        else ""
    )
    lines.append(f"-> Weekly (7 DTE):  [{r.signal_weekly.signal}]  ({r.signal_weekly.reason}.{weekly_extra})")

    monthly_extra = (
        f" Recommended short strike: < {r.signal_monthly.recommended_strike}"
        if r.signal_monthly.recommended_strike
        else ""
    )
    lines.append(f"-> Monthly (30 DTE):[{r.signal_monthly.signal}] ({r.signal_monthly.reason}.{monthly_extra})")

    lines += [
        "",
        "[RISK FILTER]",
        f"VRP: {r.vrp:+.1f}% (Seller Edge: {'PRESENT' if r.vrp > 0 else 'ABSENT'})",
        f"Expected Move (7 DTE): ±{r.expected_move:.0f} pts",
    ]
    if r.skew_slope is not None:
        lines.append(f"Skew Slope (15Δ-50Δ): {r.skew_slope:.2f} vol pts")

    cb_status = "ON ⚠️  TRADING HALTED" if r.circuit_breaker else "OFF (Trading Allowed)"
    lines.append(f"Emergency Circuit Breaker: {cb_status}")
    if r.circuit_breaker:
        for reason in r.circuit_breaker_reasons:
            lines.append(f"  ⚠️  {reason}")

    lines.append("=" * 68)
    return "\n".join(lines)
