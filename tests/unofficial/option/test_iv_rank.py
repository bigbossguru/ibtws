"""Tests for ``ibtws.unofficial.option.iv_rank.IVRankCalculator``.

Exercises the IV-history → (rank, percentile) reduction against a fake
``reqHistoricalDataAsync`` so the suite never touches TWS.
"""

from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ibtws.unofficial.option import IVRankCalculator, IVRankResult

from .conftest import make_underlying


def _bar(close: float, date: _dt.date | _dt.datetime | None = None) -> SimpleNamespace:
    """Minimal stand-in for an ``ib_async.BarData`` row."""
    return SimpleNamespace(close=close, date=date if date is not None else _dt.date(2026, 1, 1))


@pytest.fixture
def calc(fake_client) -> IVRankCalculator:
    return IVRankCalculator(fake_client, request_timeout=1.0)


@pytest.fixture
def hist_mock(fake_client) -> AsyncMock:
    """Attach a fresh ``reqHistoricalDataAsync`` AsyncMock and return it."""
    fake_client.ib.reqHistoricalDataAsync = AsyncMock()
    return fake_client.ib.reqHistoricalDataAsync


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_calculate_happy_path(calc, hist_mock):
    # Bars in chronological order — last bar is "current".
    bars = [
        _bar(0.10, _dt.date(2025, 5, 19)),
        _bar(0.20, _dt.date(2025, 8, 19)),
        _bar(0.30, _dt.date(2025, 11, 19)),
        _bar(0.40, _dt.date(2026, 2, 19)),
        _bar(0.25, _dt.date(2026, 5, 19)),  # current
    ]
    hist_mock.return_value = bars

    result = await calc.calculate(make_underlying(), lookback_days=252)

    assert isinstance(result, IVRankResult)
    assert result.underlying_symbol == "AAPL"
    assert result.sample_size == 5
    assert result.lookback_days == 252
    assert result.as_of == _dt.date(2026, 5, 19)
    assert result.current_iv == pytest.approx(0.25)
    assert result.min_iv == pytest.approx(0.10)
    assert result.max_iv == pytest.approx(0.40)
    # rank = (0.25 - 0.10) / (0.40 - 0.10) * 100 = 50
    assert result.iv_rank == pytest.approx(50.0)
    # percentile over the 4 prior bars: 0.10 and 0.20 are < 0.25 → 50%
    assert result.iv_percentile == pytest.approx(50.0)


async def test_calculate_passes_request_args(calc, hist_mock):
    hist_mock.return_value = [_bar(0.2)]

    await calc.calculate(make_underlying(), lookback_days=63, end_datetime="", use_rth=False)

    kwargs = hist_mock.await_args.kwargs
    assert kwargs["durationStr"] == "63 D"
    assert kwargs["barSizeSetting"] == "1 day"
    assert kwargs["whatToShow"] == "OPTION_IMPLIED_VOLATILITY"
    assert kwargs["useRTH"] is False
    assert kwargs["keepUpToDate"] is False


async def test_calculate_qualifies_unqualified_underlying(calc, fake_client, hist_mock):
    hist_mock.return_value = [_bar(0.2)]
    unqualified = SimpleNamespace(symbol="AAPL", conId=0, secType="STK")
    qualified = make_underlying()
    fake_client.qualify.return_value = [qualified]

    await calc.calculate(unqualified)

    fake_client.qualify.assert_awaited_once()
    # The qualified contract should be the one sent to IB.
    assert hist_mock.await_args.args[0] is qualified


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_calculate_empty_history_returns_none_metrics(calc, hist_mock):
    hist_mock.return_value = []

    result = await calc.calculate(make_underlying())

    assert result.sample_size == 0
    assert result.current_iv is None
    assert result.min_iv is None
    assert result.max_iv is None
    assert result.iv_rank is None
    assert result.iv_percentile is None
    assert result.as_of is None


async def test_calculate_swallows_request_failure(calc, hist_mock):
    hist_mock.side_effect = RuntimeError("pacing violation")

    result = await calc.calculate(make_underlying())

    assert result.sample_size == 0
    assert result.iv_rank is None


async def test_calculate_filters_nan_and_nonpositive_bars(calc, hist_mock):
    hist_mock.return_value = [
        _bar(float("nan")),
        _bar(-1.0),
        _bar(0.0),
        _bar(0.15, _dt.date(2026, 4, 1)),
        _bar(0.25, _dt.date(2026, 5, 19)),
    ]

    result = await calc.calculate(make_underlying())

    assert result.sample_size == 2
    assert result.current_iv == pytest.approx(0.25)
    assert result.min_iv == pytest.approx(0.15)
    assert result.iv_rank == pytest.approx(100.0)


async def test_calculate_flat_history_returns_none_rank(calc, hist_mock):
    hist_mock.return_value = [_bar(0.2), _bar(0.2), _bar(0.2)]

    result = await calc.calculate(make_underlying())

    assert result.min_iv == result.max_iv == pytest.approx(0.2)
    assert result.iv_rank is None  # degenerate band
    # No prior bars strictly below current → 0%.
    assert result.iv_percentile == pytest.approx(0.0)


async def test_calculate_single_bar_percentile_uses_self(calc, hist_mock):
    # With only one observation there is no prior history, so iv_percentile
    # is undefined: it must be None (not a false 0.0 that downstream callers
    # could mistake for "IV at the bottom of its range").
    hist_mock.return_value = [_bar(0.2, _dt.date(2026, 5, 19))]

    result = await calc.calculate(make_underlying())

    assert result.sample_size == 1
    assert result.current_iv == pytest.approx(0.2)
    assert result.iv_rank is None
    assert result.iv_percentile is None


async def test_calculate_accepts_datetime_bar_date(calc, hist_mock):
    hist_mock.return_value = [
        _bar(0.10, _dt.datetime(2026, 5, 18, 16, 0)),
        _bar(0.30, _dt.datetime(2026, 5, 19, 16, 0)),
    ]

    result = await calc.calculate(make_underlying())

    assert result.as_of == _dt.date(2026, 5, 19)
    assert result.iv_rank == pytest.approx(100.0)
