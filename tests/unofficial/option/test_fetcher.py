"""Tests for ``ibtws.unofficial.option.fetcher.OptionChainFetcher``.

Exercises the full pipeline (chain discovery, contract qualification,
snapshot flow, throttling, error tolerance, DataFrame projection) against
the fake client defined in ``conftest.py``.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pandas as pd
import pytest
from ib_async import Option

from ibtws.unofficial.option import ChainDefinition, OptionChainFetcher, OptionQuote

from .conftest import async_side, make_chain_param, make_ticker, make_underlying, with_conid


def _qualify_options(*contracts):
    """Stamp a ``conId`` onto each contract — used as ``client.qualify`` side_effect."""
    for i, c in enumerate(contracts):
        if not getattr(c, "conId", 0):
            c.conId = 1000 + i
    return list(contracts)


# ---------------------------------------------------------------------------
# fetch_chain_definition
# ---------------------------------------------------------------------------


async def test_fetch_chain_definition_happy_path(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [make_chain_param()]
    underlying = make_underlying()

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


async def test_fetch_chain_definition_prefers_requested_exchange(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(exchange="CBOE"),
        make_chain_param(exchange="SMART"),
    ]
    definition = await fetcher.fetch_chain_definition(make_underlying(), exchange="SMART")
    assert definition.exchange == "SMART"


async def test_fetch_chain_definition_falls_back_when_exchange_missing(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [make_chain_param(exchange="CBOE")]
    definition = await fetcher.fetch_chain_definition(make_underlying(), exchange="SMART")
    assert definition.exchange == "CBOE"


async def test_fetch_chain_definition_raises_when_empty(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = []
    with pytest.raises(LookupError):
        await fetcher.fetch_chain_definition(make_underlying())


async def test_fetch_chain_definition_requires_qualified_underlying(fetcher):
    with pytest.raises(ValueError, match="qualified"):
        await fetcher.fetch_chain_definition(make_underlying(con_id=0))


# ---------------------------------------------------------------------------
# fetch_snapshot
# ---------------------------------------------------------------------------


async def test_fetch_snapshot_full_pipeline(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0, 160.0))
    ]
    fake_client.qualify.side_effect = async_side(_qualify_options)
    fake_client.ib.reqTickersAsync.side_effect = async_side(lambda *cs, **_kw: [make_ticker(c) for c in cs])

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[150.0, 160.0],
        rights=("C", "P"),
    )

    # 1 expiry × 2 strikes × 2 rights = 4 quotes.
    assert len(quotes) == 4
    assert all(isinstance(q, OptionQuote) for q in quotes)
    assert {q.contract.right for q in quotes} == {"C", "P"}
    assert {q.contract.strike for q in quotes} == {150.0, 160.0}


async def test_fetch_snapshot_returns_empty_when_filters_exclude_everything(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [make_chain_param()]
    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20990101"],  # not in chain
    )
    assert quotes == []
    fake_client.qualify.assert_not_called()
    fake_client.ib.reqTickersAsync.assert_not_called()


async def test_fetch_snapshot_drops_qualify_failures(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.qualify.side_effect = RuntimeError("IB no like")

    quotes = await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0])

    assert quotes == []
    fake_client.ib.reqTickersAsync.assert_not_called()


async def test_fetch_snapshot_drops_unresolved_contracts(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0, 160.0))
    ]

    def qualify(*contracts):
        # Only first contract gets a conId.
        contracts[0].conId = 1
        return list(contracts)

    fake_client.qualify.side_effect = async_side(qualify)
    fake_client.ib.reqTickersAsync.side_effect = async_side(lambda *cs, **_kw: [make_ticker(c) for c in cs])

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[150.0, 160.0],
        rights=("C",),
    )
    # Only the qualified one survives.
    assert len(quotes) == 1


async def test_fetch_snapshot_handles_snapshot_timeout(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0, snapshot_timeout=0.05)
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.qualify.side_effect = async_side(_qualify_options)

    async def never_returns(*_cs, **_kw):
        await asyncio.sleep(10)
        return []

    fake_client.ib.reqTickersAsync.side_effect = never_returns

    quotes = await f.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[150.0],
        rights=("C",),
    )
    # Snapshot timed out — batch is dropped entirely.
    assert quotes == []


async def test_fetch_snapshot_auto_windows_strikes_around_spot(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(
            expirations=("20260116",),
            strikes=(50.0, 100.0, 140.0, 150.0, 160.0, 200.0, 300.0),
        )
    ]

    underlying_ticker = SimpleNamespace(last=150.0, close=150.0, bid=149.5, ask=150.5, contract=make_underlying())

    async def req_tickers(*cs, **_kw):
        # Spot lookup: single non-Option contract → return underlying ticker.
        if len(cs) == 1 and not isinstance(cs[0], Option):
            return [underlying_ticker]
        return [make_ticker(c) for c in cs]

    fake_client.ib.reqTickersAsync.side_effect = req_tickers
    fake_client.qualify.side_effect = async_side(_qualify_options)

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        rights=("C",),
        strike_window_pct=0.2,
    )

    # Spot 150 ±20% => [120, 180] — only 140, 150, 160 should survive.
    assert {q.contract.strike for q in quotes} == {140.0, 150.0, 160.0}


async def test_fetch_snapshot_skips_window_when_explicit_strikes(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(140.0, 150.0, 300.0))
    ]
    fake_client.qualify.side_effect = async_side(lambda *cs: [with_conid(c, 1) for c in cs])

    spot_calls = 0

    async def req_tickers(*cs, **_kw):
        nonlocal spot_calls
        if len(cs) == 1 and not isinstance(cs[0], Option):
            spot_calls += 1
            return []
        return [make_ticker(c) for c in cs]

    fake_client.ib.reqTickersAsync.side_effect = req_tickers

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[300.0],  # explicit — must bypass auto-window
        rights=("C",),
    )
    assert {q.contract.strike for q in quotes} == {300.0}
    assert spot_calls == 0


async def test_fetch_snapshot_qualifies_underlying_if_needed(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0)
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]

    qualified_underlying = make_underlying(con_id=12345)

    def qualify(*contracts):
        # Single STK contract = underlying-qualify; otherwise stamp option conIds.
        if len(contracts) == 1 and getattr(contracts[0], "secType", None) == "STK":
            return [qualified_underlying]
        return _qualify_options(*contracts)

    fake_client.qualify.side_effect = async_side(qualify)
    fake_client.ib.reqTickersAsync.side_effect = async_side(lambda *cs, **_kw: [make_ticker(c) for c in cs])

    unqualified = make_underlying(con_id=0)
    quotes = await f.fetch_snapshot(unqualified, expirations=["20260116"], strikes=[150.0], rights=("C",))

    fake_client.qualify.assert_any_await(unqualified)
    assert len(quotes) == 1


async def test_fetch_snapshot_as_dataframe_returns_dataframe(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.qualify.side_effect = async_side(_qualify_options)
    fake_client.ib.reqTickersAsync.side_effect = async_side(lambda *cs, **_kw: [make_ticker(c) for c in cs])

    df = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[150.0],
        rights=("C",),
        as_dataframe=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"
    assert df["strike"].iloc[0] == 150.0
    assert df["right"].iloc[0] == "C"


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


async def test_pacing_enforces_minimum_interval(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=20)  # 50 ms apart
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [make_chain_param()]

    underlying = make_underlying()
    start = time.monotonic()
    await f.fetch_chain_definition(underlying)
    await f.fetch_chain_definition(underlying)
    await f.fetch_chain_definition(underlying)
    elapsed = time.monotonic() - start

    # Two intervals between three calls => at least 100 ms (allow scheduler slack).
    assert elapsed >= 0.09


async def test_pacing_disabled_when_pace_zero(fake_client):
    f = OptionChainFetcher(fake_client, pace_per_sec=0)
    assert f._min_interval == 0
    await f._await_next_slot()
