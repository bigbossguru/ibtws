"""Tests for the SPX Volatility Analysis System."""

from __future__ import annotations


from ibtws.unofficial.analysis.spx_vol_system import (
    SPXVolAnalyzer,
    SPXVolReport,
    StrategySignal,
    TermStructure,
    format_report,
)


class TestCalcZscore:
    def test_normal(self):
        import numpy as np

        history = np.array([15.0, 16.0, 17.0, 18.0, 19.0])
        z = SPXVolAnalyzer._calc_zscore(20.0, history)
        # Pine-style: window=[16,17,18,19,20], mean=18, pop_std≈1.414
        # z = (20-18)/1.414 ≈ 1.41
        assert 1.3 < z < 1.5

    def test_empty_history(self):
        import numpy as np

        assert SPXVolAnalyzer._calc_zscore(18.0, np.array([])) == 0.0

    def test_zero_std(self):
        import numpy as np

        assert SPXVolAnalyzer._calc_zscore(15.0, np.array([15.0, 15.0, 15.0])) == 0.0


class TestCalcExpectedMove:
    def test_fallback_formula(self):
        """The fallback formula should produce a reasonable value."""
        import math

        spx, vix, dte = 5200, 18.5, 7
        em = spx * (vix / 100.0) * math.sqrt(max(dte, 1) / 365.0)
        # 5200 * 0.185 * sqrt(7/365) ≈ 133
        assert 130 < em < 140


class TestCircuitBreaker:
    def test_no_trigger(self):
        ts = TermStructure(ratio_macro=0.95, ratio_weekly=1.05, ratio_intraday=0.9, slope_futures=1.06)
        active, reasons = SPXVolAnalyzer._check_circuit_breaker(ts, 0.5)
        assert not active
        assert reasons == []

    def test_deep_backwardation(self):
        ts = TermStructure(ratio_macro=1.20, ratio_weekly=1.05, ratio_intraday=0.9, slope_futures=1.06)
        active, reasons = SPXVolAnalyzer._check_circuit_breaker(ts, 0.5)
        assert active
        assert any("backwardation" in r for r in reasons)

    def test_extreme_squeeze(self):
        ts = TermStructure(ratio_macro=0.85, ratio_weekly=1.05, ratio_intraday=0.9, slope_futures=1.06)
        active, reasons = SPXVolAnalyzer._check_circuit_breaker(ts, -2.5)
        assert active
        assert any("squeeze" in r for r in reasons)

    def test_futures_inverted(self):
        ts = TermStructure(ratio_macro=0.90, ratio_weekly=1.05, ratio_intraday=0.9, slope_futures=0.90)
        active, reasons = SPXVolAnalyzer._check_circuit_breaker(ts, 0.5)
        assert active
        assert any("futures" in r for r in reasons)


class TestTermStructure:
    def test_states(self):
        # ratio_weekly is VIX9D/VIX: < 1.0 = contango, > 1.0 = local backwardation
        ts = TermStructure(ratio_macro=0.92, ratio_weekly=0.98, ratio_intraday=1.10, slope_futures=1.06)
        assert ts.macro_state == "CONTANGO"
        assert ts.weekly_state == "CONTANGO"
        assert ts.intraday_state == "BACKWARDATION"

    def test_backwardation(self):
        ts = TermStructure(ratio_macro=1.05, ratio_weekly=1.05, ratio_intraday=0.9, slope_futures=0.93)
        assert ts.macro_state == "BACKWARDATION"
        assert ts.weekly_state == "LOCAL BACKWARDATION"
        assert ts.intraday_state == "CONTANGO"


class TestFormatReport:
    def test_produces_output(self):
        ts = TermStructure(ratio_macro=0.92, ratio_weekly=0.98, ratio_intraday=0.85, slope_futures=1.06)
        report = SPXVolReport(
            spx=5200,
            vix=18.5,
            vix_zscore=1.2,
            vvix=95.2,
            vvix_declining=True,
            term_structure=ts,
            expected_move=50.0,
            vrp=2.4,
            rv_20=16.1,
            skew_slope=3.5,
            skew_slope_20d_avg=None,
            skew_ratio=None,
            signal_0dte=StrategySignal("YELLOW", "Wait for VIX1D intraday spike"),
            signal_weekly=StrategySignal("GREEN", "Conditions met", 5110),
            signal_monthly=StrategySignal("YELLOW", "Insufficient premium"),
            circuit_breaker=False,
        )
        text = format_report(report)
        assert "SPX VOLATILITY ANALYSIS SYSTEM REPORT" in text
        assert "5200" in text
        assert "GREEN" in text
        assert "Trading Allowed" in text

    def test_circuit_breaker_shown(self):
        ts = TermStructure(ratio_macro=1.20, ratio_weekly=0.98, ratio_intraday=0.85, slope_futures=0.90)
        report = SPXVolReport(
            spx=5200,
            vix=30.0,
            vix_zscore=-2.5,
            vvix=140.0,
            vvix_declining=False,
            term_structure=ts,
            expected_move=80.0,
            vrp=-1.0,
            rv_20=31.0,
            skew_slope=None,
            skew_slope_20d_avg=None,
            skew_ratio=None,
            signal_0dte=StrategySignal("RED", "Circuit breaker active"),
            signal_weekly=StrategySignal("RED", "Circuit breaker active"),
            signal_monthly=StrategySignal("RED", "Circuit breaker active"),
            circuit_breaker=True,
            circuit_breaker_reasons=["VIX/VIX3M=1.20 > 1.15 (deep backwardation)"],
        )
        text = format_report(report)
        assert "TRADING HALTED" in text
        assert "backwardation" in text


class TestRegimeProperty:
    def test_panic(self):
        ts = TermStructure(ratio_macro=0.9, ratio_weekly=1.0, ratio_intraday=0.9, slope_futures=1.0)
        r = SPXVolReport(
            spx=5000,
            vix=35,
            vix_zscore=2.5,
            vvix=130,
            vvix_declining=False,
            term_structure=ts,
            expected_move=70,
            vrp=5,
            rv_20=30,
            skew_slope=None,
            skew_slope_20d_avg=None,
            skew_ratio=None,
            signal_0dte=StrategySignal("RED", ""),
            signal_weekly=StrategySignal("RED", ""),
            signal_monthly=StrategySignal("RED", ""),
            circuit_breaker=True,
        )
        assert r.regime == "High Volatility / Panic"

    def test_complacency(self):
        ts = TermStructure(ratio_macro=0.8, ratio_weekly=1.1, ratio_intraday=0.7, slope_futures=1.1)
        r = SPXVolReport(
            spx=5500,
            vix=11,
            vix_zscore=-1.5,
            vvix=80,
            vvix_declining=True,
            term_structure=ts,
            expected_move=30,
            vrp=1,
            rv_20=10,
            skew_slope=None,
            skew_slope_20d_avg=None,
            skew_ratio=None,
            signal_0dte=StrategySignal("YELLOW", ""),
            signal_weekly=StrategySignal("YELLOW", ""),
            signal_monthly=StrategySignal("YELLOW", ""),
            circuit_breaker=False,
        )
        assert r.regime == "Low Volatility / Complacency"
