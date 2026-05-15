"""Unit tests for ibtws.unofficial.option.OptionChainFetcher.

The class is exercised against a fake IBKRClient backed by AsyncMocks so the
tests cover the full pipeline (definition cache, contract qualification,
snapshot/streaming flow, throttling, error tolerance) without TWS.
"""

from __future__ import annotations

import asyncio
import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from ib_async import Option, Ticker

from ibtws.unofficial import option as option_module
from ibtws.unofficial.option import (
    ChainDefinition,
    OptionChainFetcher,
    OptionQuote,
    _chunked,
    _filter_expirations,
    _filter_strikes,
    _safe_float,
    _ticker_to_quote,
    quotes_to_dataframe,
)


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


def _make_underlying(symbol: str = "AAPL", con_id: int = 265598) -> SimpleNamespace:
    """A minimal stand-in for a qualified underlying ``Contract``."""
    return SimpleNamespace(symbol=symbol, conId=con_id, secType="STK")


def _make_chain_param(
    *,
    exchange: str = "SMART",
    trading_class: str = "AAPL",
    multiplier: str = "100",
    expirations=("20260116", "20260220", "20260320"),
    strikes=(140.0, 150.0, 160.0, 170.0, 180.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        exchange=exchange,
        tradingClass=trading_class,
        multiplier=multiplier,
        expirations=list(expirations),
        strikes=list(strikes),
    )


def _make_ticker(
    contract: Option,
    *,
    bid: float = 1.0,
    ask: float = 1.2,
    last: float = 1.1,
    volume: float = 500.0,
    call_oi: float = 1000.0,
    put_oi: float = 1100.0,
    iv: float | None = 0.25,
    delta: float | None = 0.5,
) -> Ticker:
    t = Ticker()
    t.contract = contract
    t.bid = bid
    t.ask = ask
    t.last = last
    t.volume = volume
    t.callOpenInterest = call_oi
    t.putOpenInterest = put_oi
    if iv is not None:
        t.modelGreeks = SimpleNamespace(impliedVol=iv, delta=delta, gamma=0.01, vega=0.05, theta=-0.02, undPrice=155.0)
    return t


@pytest.fixture
def fake_client():
    """Stand-in for IBKRClient that exposes only what OptionChainFetcher uses."""
    ib = MagicMock()
    ib.reqSecDefOptParamsAsync = AsyncMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqTickersAsync = AsyncMock()
    ib.reqMktData = MagicMock()
    ib.cancelMktData = MagicMock()
    client = SimpleNamespace(ib=ib, qualify=AsyncMock())
    return client


@pytest.fixture
def fetcher(fake_client) -> OptionChainFetcher:
    # Disable pacing so tests do not pay 25 ms per call.
    return OptionChainFetcher(
        fake_client,
        max_concurrency=10,
        pace_per_sec=0,
        cache_ttl=60.0,
        snapshot_timeout=1.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_chunked_splits_evenly_and_preserves_order():
    assert list(_chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty_returns_nothing():
    assert list(_chunked([], 10)) == []


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, 1.5),
        (None, None),
        (float("nan"), None),
        ("not a number", None),
        ("3.14", 3.14),
    ],
)
def test_safe_float(value, expected):
    assert _safe_float(value) == expected


def test_filter_expirations_explicit_wins_over_range():
    avail = ("20260116", "20260220", "20260320")
    out = _filter_expirations(avail, explicit=["20260220", "20260320"], expiry_from=None, expiry_to=None)
    assert out == ["20260220", "20260320"]


def test_filter_expirations_range():
    avail = ("20260116", "20260220", "20260320")
    out = _filter_expirations(avail, explicit=None, expiry_from="20260201", expiry_to="20260301")
    assert out == ["20260220"]


def test_filter_strikes_range_inclusive():
    avail = (140.0, 150.0, 160.0, 170.0, 180.0)
    assert _filter_strikes(avail, explicit=None, strike_from=150.0, strike_to=170.0) == [150.0, 160.0, 170.0]


def test_filter_strikes_explicit():
    avail = (140.0, 150.0, 160.0)
    assert _filter_strikes(avail, explicit=[140.0, 160.0], strike_from=None, strike_to=None) == [140.0, 160.0]


def test_ticker_to_quote_uses_call_oi_for_calls():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = _make_ticker(contract, call_oi=222, put_oi=999)
    q = _ticker_to_quote(t)
    assert q.open_interest == 222
    assert q.iv == 0.25
    assert q.delta == 0.5


def test_ticker_to_quote_uses_put_oi_for_puts():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="P")
    t = _make_ticker(contract, call_oi=222, put_oi=999)
    q = _ticker_to_quote(t)
    assert q.open_interest == 999


def test_ticker_to_quote_handles_missing_greeks():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = _make_ticker(contract, iv=None)
    t.modelGreeks = None
    q = _ticker_to_quote(t)
    assert q.iv is None
    assert q.delta is None
    assert q.underlying_price is None


def test_ticker_to_quote_scrubs_nan():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = _make_ticker(contract)
    t.bid = float("nan")
    t.ask = float("nan")
    q = _ticker_to_quote(t)
    assert q.bid is None and q.ask is None


# ---------------------------------------------------------------------------
# fetch_chain_definition
# ---------------------------------------------------------------------------


async def test_fetch_chain_definition_happy_path(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param()]
    underlying = _make_underlying()

    definition = await fetcher.fetch_chain_definition(underlying)

    assert isinstance(definition, ChainDefinition)
    assert definition.underlying_symbol == "AAPL"
    assert definition.exchange == "SMART"
    assert definition.expirations == ("20260116", "20260220", "20260320")
    assert definition.strikes == (140.0, 150.0, 160.0, 170.0, 180.0)
    fake_client.ib.reqSecDefOptParamsAsync.assert_awaited_once_with(
        underlyingSymbol="AAPL",
        futFopExchange="",
        underlyingSecType="STK",
        underlyingConId=265598,
    )


async def test_fetch_chain_definition_caches(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param()]
    underlying = _make_underlying()

    await fetcher.fetch_chain_definition(underlying)
    await fetcher.fetch_chain_definition(underlying)

    assert fake_client.ib.reqSecDefOptParamsAsync.await_count == 1


async def test_fetch_chain_definition_force_refresh(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param()]
    underlying = _make_underlying()

    await fetcher.fetch_chain_definition(underlying)
    await fetcher.fetch_chain_definition(underlying, force_refresh=True)

    assert fake_client.ib.reqSecDefOptParamsAsync.await_count == 2


async def test_fetch_chain_definition_cache_expires(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0, cache_ttl=0.05)
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param()]
    underlying = _make_underlying()

    await f.fetch_chain_definition(underlying)
    await asyncio.sleep(0.1)
    await f.fetch_chain_definition(underlying)

    assert fake_client.ib.reqSecDefOptParamsAsync.await_count == 2


async def test_fetch_chain_definition_prefers_requested_exchange(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(exchange="CBOE"),
        _make_chain_param(exchange="SMART"),
    ]
    definition = await fetcher.fetch_chain_definition(_make_underlying(), exchange="SMART")
    assert definition.exchange == "SMART"


async def test_fetch_chain_definition_falls_back_when_exchange_missing(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param(exchange="CBOE")]
    definition = await fetcher.fetch_chain_definition(_make_underlying(), exchange="SMART")
    assert definition.exchange == "CBOE"


async def test_fetch_chain_definition_raises_when_empty(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = []
    with pytest.raises(LookupError):
        await fetcher.fetch_chain_definition(_make_underlying())


async def test_fetch_chain_definition_requires_qualified_underlying(fetcher):
    with pytest.raises(ValueError, match="qualified"):
        await fetcher.fetch_chain_definition(_make_underlying(con_id=0))


# ---------------------------------------------------------------------------
# fetch_snapshot
# ---------------------------------------------------------------------------


async def test_fetch_snapshot_full_pipeline(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0, 160.0))
    ]

    def qualify_side_effect(*contracts):
        for i, c in enumerate(contracts):
            c.conId = 1000 + i
        return list(contracts)

    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(qualify_side_effect)
    fake_client.ib.reqMktData.side_effect = lambda c, *_a, **_k: _make_ticker(c)

    quotes = await fetcher.fetch_snapshot(
        _make_underlying(),
        expirations=["20260116"],
        strikes=[150.0, 160.0],
        rights=("C", "P"),
    )

    # 1 expiry × 2 strikes × 2 rights = 4 quotes.
    assert len(quotes) == 4
    assert all(isinstance(q, OptionQuote) for q in quotes)
    assert {q.contract.right for q in quotes} == {"C", "P"}
    assert {q.contract.strike for q in quotes} == {150.0, 160.0}
    # Every subscription must have been cancelled.
    assert fake_client.ib.cancelMktData.call_count == 4


async def test_fetch_snapshot_returns_empty_when_filters_exclude_everything(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param()]
    quotes = await fetcher.fetch_snapshot(
        _make_underlying(),
        expirations=["20990101"],  # not in chain
    )
    assert quotes == []
    fake_client.ib.qualifyContractsAsync.assert_not_called()
    fake_client.ib.reqMktData.assert_not_called()


async def test_fetch_snapshot_drops_qualify_failures(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = RuntimeError("IB no like")

    quotes = await fetcher.fetch_snapshot(_make_underlying(), expirations=["20260116"], strikes=[150.0])

    assert quotes == []
    fake_client.ib.reqMktData.assert_not_called()


async def test_fetch_snapshot_drops_unresolved_contracts(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0, 160.0))
    ]

    def qualify(*contracts):
        # Only first contract gets a conId.
        contracts[0].conId = 1
        return list(contracts)

    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(qualify)
    fake_client.ib.reqMktData.side_effect = lambda c, *_a, **_k: _make_ticker(c)

    quotes = await fetcher.fetch_snapshot(
        _make_underlying(),
        expirations=["20260116"],
        strikes=[150.0, 160.0],
        rights=("C",),
    )
    # Only the qualified one survives.
    assert len(quotes) == 1


async def test_fetch_snapshot_handles_snapshot_timeout(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0, snapshot_timeout=0.2)
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]

    def qualify(*contracts):
        for c in contracts:
            c.conId = 7
        return list(contracts)

    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(qualify)

    # Ticker that will never become "complete" — no Greeks, no OI.
    def make_incomplete(c, *_a, **_k):
        t = Ticker()
        t.contract = c
        t.bid = float("nan")
        t.ask = float("nan")
        t.last = float("nan")
        return t

    fake_client.ib.reqMktData.side_effect = make_incomplete

    quotes = await f.fetch_snapshot(
        _make_underlying(),
        expirations=["20260116"],
        strikes=[150.0],
        rights=("C",),
    )
    # Partial data is still returned; just empty fields.
    assert len(quotes) == 1
    assert quotes[0].iv is None
    assert quotes[0].open_interest is None
    fake_client.ib.cancelMktData.assert_called_once()


async def test_fetch_snapshot_auto_windows_strikes_around_spot(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(
            expirations=("20260116",),
            strikes=(50.0, 100.0, 140.0, 150.0, 160.0, 200.0, 300.0),
        )
    ]

    underlying_ticker = SimpleNamespace(last=150.0, close=150.0, bid=149.5, ask=150.5, contract=_make_underlying())

    fake_client.ib.reqTickersAsync.side_effect = _async_side(lambda *_cs, **_kw: [underlying_ticker])
    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(
        lambda *cs: [_with_conid(c, 1000 + i) for i, c in enumerate(cs)]
    )
    fake_client.ib.reqMktData.side_effect = lambda c, *_a, **_k: _make_ticker(c)

    quotes = await fetcher.fetch_snapshot(
        _make_underlying(),
        expirations=["20260116"],
        rights=("C",),
        strike_window_pct=0.2,
    )

    # Spot 150 ±20% => [120, 180] — only 140, 150, 160 should survive.
    assert {q.contract.strike for q in quotes} == {140.0, 150.0, 160.0}


async def test_fetch_snapshot_skips_window_when_explicit_strikes(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(140.0, 150.0, 300.0))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(lambda *cs: [_with_conid(c, 1) for c in cs])
    fake_client.ib.reqMktData.side_effect = lambda c, *_a, **_k: _make_ticker(c)

    quotes = await fetcher.fetch_snapshot(
        _make_underlying(),
        expirations=["20260116"],
        strikes=[300.0],  # explicit — must bypass auto-window
        rights=("C",),
    )
    assert {q.contract.strike for q in quotes} == {300.0}
    # Spot was never fetched.
    fake_client.ib.reqTickersAsync.assert_not_called()


async def test_fetch_snapshot_qualifies_underlying_if_needed(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0, cache_ttl=60.0)
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(
        lambda *cs: [_with_conid(c, 1000 + i) for i, c in enumerate(cs)]
    )
    fake_client.ib.reqMktData.side_effect = lambda c, *_a, **_k: _make_ticker(c)

    qualified = _make_underlying(con_id=12345)
    fake_client.qualify.return_value = [qualified]

    unqualified = _make_underlying(con_id=0)
    quotes = await f.fetch_snapshot(unqualified, expirations=["20260116"], strikes=[150.0], rights=("C",))

    fake_client.qualify.assert_awaited_once_with(unqualified)
    assert len(quotes) == 1


# ---------------------------------------------------------------------------
# subscribe (streaming)
# ---------------------------------------------------------------------------


async def test_subscribe_subscribes_and_cancels_on_exit(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]

    def qualify(*contracts):
        for c in contracts:
            c.conId = 99
        return list(contracts)

    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(qualify)
    fake_client.ib.reqMktData.side_effect = lambda contract, *_a, **_k: _make_ticker(contract)

    async with fetcher.subscribe(
        _make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C", "P")
    ) as sub:
        assert len(sub.tickers) == 2
        assert fake_client.ib.reqMktData.call_count == 2

    # On exit, every subscribed contract must be cancelled.
    assert fake_client.ib.cancelMktData.call_count == 2


async def test_subscribe_cancels_on_exception(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(lambda *cs: [_with_conid(c, 1) for c in cs])
    fake_client.ib.reqMktData.side_effect = lambda contract, *_a, **_k: _make_ticker(contract)

    with pytest.raises(RuntimeError, match="user code"):
        async with fetcher.subscribe(_make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",)):
            raise RuntimeError("user code")

    fake_client.ib.cancelMktData.assert_called_once()


async def test_subscribe_swallows_cancel_errors(fetcher, fake_client, caplog):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(lambda *cs: [_with_conid(c, 1) for c in cs])
    fake_client.ib.reqMktData.side_effect = lambda contract, *_a, **_k: _make_ticker(contract)
    fake_client.ib.cancelMktData.side_effect = RuntimeError("IB closed")

    # Must not raise even though cancel fails.
    async with fetcher.subscribe(_make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",)):
        pass


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


async def test_pacing_enforces_minimum_interval(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=20)  # 50 ms apart
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [_make_chain_param()]

    underlying = _make_underlying()
    start = time.monotonic()
    # Three definition fetches with force_refresh to bypass cache.
    await f.fetch_chain_definition(underlying)
    await f.fetch_chain_definition(underlying, force_refresh=True)
    await f.fetch_chain_definition(underlying, force_refresh=True)
    elapsed = time.monotonic() - start

    # Two intervals between three calls => at least 100 ms (allow scheduler slack).
    assert elapsed >= 0.09


async def test_pacing_disabled_when_pace_zero(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0)
    assert f._min_interval == 0
    # Should not even sleep.
    await f._await_next_slot()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _aresult(value):
    """Wrap a sync result so AsyncMock-style side_effects can return synchronously."""

    async def _coro():
        return value

    return _coro()


def _async_side(fn):
    """Adapt a sync function into an ``AsyncMock.side_effect`` that returns its result."""

    async def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return _wrapped


def _with_conid(contract, con_id: int):
    contract.conId = con_id
    return contract


def test_module_exposes_constants():
    # Sanity: the generic tick list must include IV + OI codes.
    assert "106" in option_module._OPTION_GENERIC_TICKS  # implied vol
    assert "101" in option_module._OPTION_GENERIC_TICKS  # open interest
    assert math.isnan(float("nan"))  # sanity, environment supports nan


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------


def test_quotes_to_dataframe_columns_and_rows():
    contract_c = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    contract_p = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="P")
    quotes = [
        _ticker_to_quote(_make_ticker(contract_c, bid=1.0, ask=1.2, iv=0.25)),
        _ticker_to_quote(_make_ticker(contract_p, bid=0.8, ask=1.0, iv=0.30)),
    ]

    df = quotes_to_dataframe(quotes)

    assert list(df.columns) == list(option_module.DATAFRAME_COLUMNS)
    assert len(df) == 2
    assert set(df["right"]) == {"C", "P"}
    assert df.loc[df["right"] == "C", "iv"].iloc[0] == 0.25


def test_quotes_to_dataframe_empty_returns_typed_empty():
    df = quotes_to_dataframe([])
    assert df.empty
    assert list(df.columns) == list(option_module.DATAFRAME_COLUMNS)


def test_quotes_to_dataframe_preserves_none_for_missing_fields():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = _make_ticker(contract)
    t.modelGreeks = None  # simulate Greeks never arriving
    df = quotes_to_dataframe([_ticker_to_quote(t)])
    assert df["iv"].iloc[0] is None
    assert df["delta"].iloc[0] is None


async def test_fetch_snapshot_df_returns_dataframe(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        _make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]

    def qualify(*contracts):
        for c in contracts:
            c.conId = 42
        return list(contracts)

    fake_client.ib.qualifyContractsAsync.side_effect = _async_side(qualify)
    fake_client.ib.reqMktData.side_effect = lambda c, *_a, **_k: _make_ticker(c)

    df = await fetcher.fetch_snapshot_df(
        _make_underlying(),
        expirations=["20260116"],
        strikes=[150.0],
        rights=("C",),
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"
    assert df["strike"].iloc[0] == 150.0
    assert df["right"].iloc[0] == "C"
