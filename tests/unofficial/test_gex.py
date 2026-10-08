"""Tests for ``ibtws.unofficial.analysis.gex.GexCalculator``."""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ibtws.unofficial.analysis.gex import GexCalculator

ET = ZoneInfo("America/New_York")
SPOT = 6700.0
# Snapshot taken at 10:00 ET on the expiry day (0DTE).
SNAPSHOT = dt.datetime(2026, 10, 9, 10, 0, tzinfo=ET).timestamp()


def _chain(*, expiry: str = "20261009", strikes=None, timestamp: float = SNAPSHOT) -> pd.DataFrame:
    strikes = np.arange(6400, 7005, 5.0) if strikes is None else strikes
    rows = []
    for k in strikes:
        for right in ("C", "P"):
            rows.append({
                "strike": float(k),
                "right": right,
                "gamma": 0.002,
                "open_interest": 1000.0 + (k % 50) * 10,
                "underlying_price": SPOT,
                "iv": 0.12 + abs(k - SPOT) / 20000,
                "expiry": expiry,
                "timestamp": timestamp,
                "delta": 0.5 if right == "C" else -0.5,
            })
    return pd.DataFrame(rows)


def _reference_curve(calc: GexCalculator, sweep: np.ndarray) -> np.ndarray:
    """The original row-by-row implementation, kept as an oracle."""
    out = []
    for s in sweep:
        total = 0.0
        for i, row in calc._df.iterrows():
            g = GexCalculator._bs_gamma(s, row["strike"], calc._T[i], calc._r, row["iv"])
            gex = g * row["open_interest"] * s**2 * 0.01
            total += gex if row["right"] == "C" else -gex
        out.append(total)
    return np.array(out)


def test_vectorised_curve_matches_reference():
    calc = GexCalculator(sweep_points=40)
    result = calc.compute(_chain())
    expected = _reference_curve(calc, result.sweep_levels)
    np.testing.assert_allclose(result.sweep_gex, expected, rtol=1e-9, atol=1e-6)


def test_full_chain_computes_quickly():
    calc = GexCalculator(sweep_points=500)
    started = time.perf_counter()
    calc.compute(_chain())
    # Used to take ~50 s for this chain with iterrows.
    assert time.perf_counter() - started < 2.0


def test_time_to_expiry_uses_exchange_close_per_row():
    df = pd.concat([_chain(expiry="20261009"), _chain(expiry="20261016")], ignore_index=True)
    calc = GexCalculator(sweep_points=20)
    calc.compute(df)
    years = {e: calc._T[(calc._df["expiry"] == e).to_numpy()][0] for e in ("20261009", "20261016")}
    # 10:00 → 16:00 ET is 6 hours; the weekly is exactly 7 days further out.
    assert years["20261009"] * 365.25 * 24 == pytest.approx(6.0)
    assert (years["20261016"] - years["20261009"]) * 365.25 == pytest.approx(7.0)


def test_drops_unusable_and_expired_rows():
    df = _chain(strikes=[6650.0, 6700.0, 6750.0])
    df.loc[0, "iv"] = np.nan
    df.loc[1, "gamma"] = None
    df.loc[2, "iv"] = 0.0
    expired = _chain(expiry="20261008", strikes=[6700.0])
    calc = GexCalculator(sweep_points=20)

    result = calc.compute(pd.concat([df, expired], ignore_index=True))

    assert len(calc._df) == len(df) - 3
    assert np.isfinite(result.sweep_gex).all()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        GexCalculator().compute(_chain().drop(columns=["open_interest"]))


def test_all_rows_expired_raise():
    late = dt.datetime(2026, 10, 9, 16, 30, tzinfo=ET).timestamp()
    with pytest.raises(ValueError, match="No usable option rows"):
        GexCalculator().compute(_chain(timestamp=late))


def test_import_does_not_require_matplotlib():
    code = "import sys, ibtws.unofficial.analysis.gex; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_zero_gamma_found_between_put_and_call_walls():
    df = _chain()
    heavy_put = (df["right"] == "P") & (df["strike"] < SPOT)
    heavy_call = (df["right"] == "C") & (df["strike"] >= SPOT)
    df["open_interest"] = np.where(heavy_put | heavy_call, 5000.0, 200.0)

    result = GexCalculator(sweep_points=200).compute(df)

    assert result.zero_gamma_level is not None
    assert df["strike"].min() < result.zero_gamma_level < df["strike"].max()
    assert result.all_crossings[0].direction == "neg→pos"


def test_flat_profile_reports_no_crossing():
    # Calls and puts with identical OI/IV cancel exactly: the curve is flat
    # zero, which must not be reported as a zero-gamma level at the sweep edge.
    df = _chain()
    df["open_interest"] = 1000.0
    result = GexCalculator(sweep_points=50).compute(df)
    assert result.zero_gamma_level is None
    assert result.all_crossings == []
