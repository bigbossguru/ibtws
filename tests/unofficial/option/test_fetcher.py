"""Tests for ``ibtws.unofficial.option.chains.OptionChainFetcher``.

Exercises chain discovery, contract qualification, the snapshot flow,
filter behaviour, error tolerance, and DataFrame projection against the
fake client defined in ``conftest.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import asyncio
import time

import pandas as pd
import pytest
from ib_async import Ticker

from ibtws.unofficial.option import ChainDefinition, OptionChainFetcher, OptionQuote

from .conftest import async_side, make_chain_param, make_ticker, make_underlying, with_conid


def _qualify_options(*contracts):
    """Stamp a ``conId`` onto each contract — used as ``qualifyContractsAsync.side_effect``."""
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


async def test_fetch_chain_definition_filters_by_trading_class(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(trading_class="SPX"),
        make_chain_param(trading_class="SPXW"),
    ]
    definition = await fetcher.fetch_chain_definition(make_underlying(), trading_class="SPXW")
    assert definition.trading_class == "SPXW"


async def test_fetch_chain_definition_raises_when_empty(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = []
    with pytest.raises(LookupError):
        await fetcher.fetch_chain_definition(make_underlying())


async def test_fetch_chain_definition_raises_when_trading_class_missing(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [make_chain_param(trading_class="SPX")]
    with pytest.raises(LookupError):
        await fetcher.fetch_chain_definition(make_underlying(), trading_class="SPXW")


async def test_fetch_chain_definition_requires_qualified_underlying(fetcher):
    with pytest.raises(ValueError, match="qualified"):
        await fetcher.fetch_chain_definition(make_underlying(con_id=0))


# ---------------------------------------------------------------------------
# fetch_snapshot
# ---------------------------------------------------------------------------


async def test_fetch_snapshot_full_pipeline(fetcher, fake_client, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0, 160.0))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

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
    fake_client.ib.qualifyContractsAsync.assert_not_called()
    fake_client.ib.reqMktData.assert_not_called()


async def test_fetch_snapshot_filters_out_empty_tickers(fetcher, fake_client, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0, 160.0))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)

    def ticker_or_empty(c, **_kw):
        if c.strike == 150.0:
            return make_ticker(c)  # valid
        # Empty ticker — no bid/ask/iv
        return SimpleNamespace(contract=c, bid=-1, ask=-1, last=-1, close=-1, volume=0, modelGreeks=None)

    fake_client.ib.reqMktData.side_effect = stream(ticker_or_empty)

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[150.0, 160.0],
        rights=("C",),
    )
    # Only the ticker with actual data survives.
    assert len(quotes) == 1


async def test_fetch_snapshot_drops_qualify_failures(fetcher, fake_client):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = RuntimeError("IB no like")

    quotes = await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0])

    assert quotes == []


async def test_fetch_snapshot_subscribes_and_cancels_market_data(fetcher, fake_client, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[150.0],
        rights=("C", "P"),
    )

    assert len(quotes) == 2
    # Each qualified contract gets a reqMktData and a matching cancelMktData.
    assert fake_client.ib.reqMktData.call_count == 2
    assert fake_client.ib.cancelMktData.call_count == 2


async def test_fetch_snapshot_batches_subscriptions(fetcher, fake_client, stream):
    # 3 strikes × 2 rights = 6 contracts; batch_size=2 ⇒ 3 batches.
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(140.0, 150.0, 160.0))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[140.0, 150.0, 160.0],
        rights=("C", "P"),
        batch_size=2,
    )

    assert len(quotes) == 6
    # Batches bound the streaming lines; every subscription is cancelled.
    assert fake_client.ib.reqMktData.call_count == 6
    assert fake_client.ib.cancelMktData.call_count == 6


async def test_fetch_snapshot_survives_one_unsubscribed_contract(fetcher, fake_client, stream):
    """IB reports 10090 asynchronously: the stream for that contract just never fills."""
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(140.0, 150.0))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)

    def ticker(c, **_kw):
        if c.strike == 140.0:
            t = Ticker()
            t.contract = c
            return t  # never receives a tick
        return make_ticker(c)

    fake_client.ib.reqMktData.side_effect = stream(ticker)

    quotes = await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[140.0, 150.0])

    assert {q.contract.strike for q in quotes} == {150.0}
    assert fake_client.ib.cancelMktData.call_count == 4


async def test_fetch_snapshot_auto_windows_strikes_around_spot(fetcher, fake_client, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(
            expirations=("20260116",),
            strikes=(50.0, 100.0, 140.0, 150.0, 160.0, 200.0, 300.0),
        )
    ]
    # Spot lookup hits IBKRClient.get_market_data.
    fake_client.get_market_data.return_value = SimpleNamespace(
        last=150.0, close=150.0, bid=149.5, ask=150.5, contract=make_underlying()
    )
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        rights=("C",),
        strike_window_pct=0.2,
    )

    # Spot 150 ±20% => [120, 180] — only 140, 150, 160 should survive.
    assert {q.contract.strike for q in quotes} == {140.0, 150.0, 160.0}
    fake_client.get_market_data.assert_awaited_once()


async def test_fetch_snapshot_skips_window_when_explicit_strikes(fetcher, fake_client, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(140.0, 150.0, 300.0))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(lambda *cs: [with_conid(c, 1) for c in cs])
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

    quotes = await fetcher.fetch_snapshot(
        make_underlying(),
        expirations=["20260116"],
        strikes=[300.0],  # explicit — must bypass auto-window
        rights=("C",),
    )
    assert {q.contract.strike for q in quotes} == {300.0}
    fake_client.get_market_data.assert_not_called()


async def test_fetch_snapshot_qualifies_underlying_if_needed(fake_client, fetcher, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    qualified_underlying = make_underlying(con_id=12345)

    def qualify(*contracts):
        # Single STK contract = underlying-qualify; otherwise stamp option conIds.
        if len(contracts) == 1 and getattr(contracts[0], "secType", None) == "STK":
            return [qualified_underlying]
        return _qualify_options(*contracts)

    fake_client.ib.qualifyContractsAsync.side_effect = async_side(qualify)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

    unqualified = make_underlying(con_id=0)
    quotes = await fetcher.fetch_snapshot(unqualified, expirations=["20260116"], strikes=[150.0], rights=("C",))

    assert len(quotes) == 1
    # The first qualify call gets the unqualified underlying.
    first_call_args = fake_client.ib.qualifyContractsAsync.await_args_list[0].args
    assert first_call_args == (unqualified,)


async def test_fetch_snapshot_as_dataframe_returns_dataframe(fetcher, fake_client, stream):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [
        make_chain_param(expirations=("20260116",), strikes=(150.0,))
    ]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

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
# streamed quotes — no per-contract snapshots
# ---------------------------------------------------------------------------
#
# A per-contract snapshot took ~16 s per 83-contract SPXW batch on a paper
# account, while the streaming subscription it sat on top of had already
# delivered quotes, OI and model greeks for 83/83 contracts in 3.6-3.8 s
# (live check 2026-10-13). Snapshots cannot carry generic ticks, so the
# snapshot added nothing but the wait.


def _chain(fake_client, strikes=(150.0,)):
    fake_client.ib.reqSecDefOptParamsAsync.return_value = [make_chain_param(expirations=("20260116",), strikes=strikes)]
    fake_client.ib.qualifyContractsAsync.side_effect = async_side(_qualify_options)


def _bare_ticker(contract, **fields) -> Ticker:
    t = Ticker()
    t.contract = contract
    for name, value in fields.items():
        setattr(t, name, value)
    return t


async def test_fetch_snapshot_reads_streamed_tickers_without_snapshots(fetcher, fake_client, stream):
    _chain(fake_client)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)

    quotes = await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",))

    assert len(quotes) == 1
    fake_client.ib.reqTickersAsync.assert_not_called()


async def test_fetch_snapshot_waits_for_greeks_and_open_interest(fake_client, monkeypatch):
    """Quotes arrive first; model greeks and OI a little later. Wait for all of it."""
    import ibtws.unofficial.option.chains as chains_module

    _chain(fake_client)
    streamed: list[Ticker] = []

    def subscribe(c, **_kw):
        t = _bare_ticker(c)
        streamed.append(t)
        return t

    fake_client.ib.reqMktData.side_effect = subscribe
    polls = {"n": 0}

    async def tick_arrives(_seconds):
        polls["n"] += 1
        if polls["n"] == 1:
            streamed[0].bid, streamed[0].ask = 1.0, 1.2  # quote only, no greeks / OI yet
        if polls["n"] == 3:
            full = make_ticker(streamed[0].contract)
            streamed[0].modelGreeks = full.modelGreeks
            streamed[0].callOpenInterest = full.callOpenInterest

    monkeypatch.setattr(chains_module.asyncio, "sleep", tick_arrives)
    fetcher = OptionChainFetcher(fake_client, quote_timeout=30.0)

    [quote] = await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",))

    assert quote.delta == pytest.approx(0.5)
    assert quote.open_interest == pytest.approx(1000.0)


async def test_fetch_snapshot_returns_as_soon_as_tickers_are_complete(fake_client, stream):
    _chain(fake_client)
    fake_client.ib.reqMktData.side_effect = stream(make_ticker)
    fetcher = OptionChainFetcher(fake_client, quote_timeout=30.0)

    started = time.monotonic()
    await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C", "P"))

    assert time.monotonic() - started < 5.0


async def test_fetch_snapshot_gives_up_on_incomplete_tickers_at_quote_timeout(fake_client, stream):
    """A contract that never gets greeks must not hold the chain past quote_timeout."""
    _chain(fake_client, strikes=(150.0, 160.0))

    def subscribe(c, **_kw):
        if c.strike == 160.0:
            return _bare_ticker(c, bid=0.05, ask=0.10)  # quote but never greeks / OI
        return make_ticker(c)

    fake_client.ib.reqMktData.side_effect = stream(subscribe)
    fetcher = OptionChainFetcher(fake_client, quote_timeout=0.2)

    started = time.monotonic()
    quotes = await fetcher.fetch_snapshot(
        make_underlying(), expirations=["20260116"], strikes=[150.0, 160.0], rights=("C",)
    )

    assert time.monotonic() - started < 5.0
    # The partial ticker still carries a quote, so it is returned, not dropped.
    assert {q.contract.strike for q in quotes} == {150.0, 160.0}
    assert fake_client.ib.cancelMktData.call_count == 2


async def test_subscriptions_are_cancelled_when_waiting_is_interrupted(fake_client, monkeypatch):
    import ibtws.unofficial.option.chains as chains_module

    _chain(fake_client)
    fake_client.ib.reqMktData.side_effect = lambda c, **_kw: _bare_ticker(c)

    async def cancelled(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(chains_module.asyncio, "sleep", cancelled)
    fetcher = OptionChainFetcher(fake_client, quote_timeout=30.0)

    with pytest.raises(asyncio.CancelledError):
        await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",))
    assert fake_client.ib.cancelMktData.call_count == 1


async def test_reused_ticker_does_not_pass_stale_values_as_fresh(fake_client, monkeypatch):
    """ib_async keeps one Ticker per contract and never clears it on re-subscribe.

    The previous fetch's values must not satisfy the wait: a bot polling the
    same strikes every tick would otherwise get an instant answer with quotes
    one tick old.
    """
    import ibtws.unofficial.option.chains as chains_module

    _chain(fake_client)
    shared: dict[str, Ticker] = {}

    def subscribe(c, **_kw):
        shared.setdefault("t", _bare_ticker(c))
        return shared["t"]  # same object on every subscription, values kept

    def fill(bid):
        fresh = make_ticker(shared["t"].contract, bid=bid)
        for field in ("bid", "ask", "modelGreeks", "callOpenInterest", "putOpenInterest"):
            setattr(shared["t"], field, getattr(fresh, field))

    async def first_ticks(_seconds):
        fill(1.0)

    fake_client.ib.reqMktData.side_effect = subscribe
    monkeypatch.setattr(chains_module.asyncio, "sleep", first_ticks)
    fetcher = OptionChainFetcher(fake_client, quote_timeout=30.0)
    await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",))

    polls = {"n": 0}

    async def fresh_ticks_arrive(_seconds):
        polls["n"] += 1
        if polls["n"] == 2:
            fill(2.0)

    monkeypatch.setattr(chains_module.asyncio, "sleep", fresh_ticks_arrive)
    [quote] = await fetcher.fetch_snapshot(make_underlying(), expirations=["20260116"], strikes=[150.0], rights=("C",))

    assert quote.bid == pytest.approx(2.0)
