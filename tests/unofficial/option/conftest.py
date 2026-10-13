"""Shared fakes/fixtures for ``ibtws.unofficial.option`` tests.

Exposes a stand-in :class:`IBKRClient` backed by ``AsyncMock`` so each test
module can exercise the option pipeline without touching TWS.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ib_async import Option, Ticker

from ibtws.unofficial.option import OptionChainFetcher


def make_underlying(symbol: str = "AAPL", con_id: int = 265598) -> SimpleNamespace:
    """A minimal stand-in for a qualified underlying ``Contract``."""
    return SimpleNamespace(symbol=symbol, conId=con_id, secType="STK")


def make_chain_param(
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


def make_ticker(
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


def async_side(fn):
    """Adapt a sync function into an ``AsyncMock.side_effect`` returning its result."""

    async def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return _wrapped


def with_conid(contract, con_id: int):
    contract.conId = con_id
    return contract


@pytest.fixture
def fake_client():
    """Stand-in for IBKRClient exposing only what the option package uses."""
    ib = MagicMock()
    ib.reqSecDefOptParamsAsync = AsyncMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqTickersAsync = AsyncMock()
    ib.reqMktData = MagicMock()
    ib.cancelMktData = MagicMock()
    return SimpleNamespace(
        ib=ib,
        # IVRankCalculator currently calls ``self._client.qualify(...)``;
        # keep an AsyncMock here so iv_rank tests can wire a side_effect.
        qualify=AsyncMock(),
        # OptionChainFetcher._fetch_spot uses IBKRClient.get_market_data().
        get_market_data=AsyncMock(),
    )


_STREAM_FIELDS = ("bid", "ask", "last", "volume", "callOpenInterest", "putOpenInterest", "modelGreeks")


class FakeStream:
    """``reqMktData`` double whose values arrive on the next poll, like a real stream.

    The fetcher clears the fields it waits for right after subscribing (ib_async
    reuses tickers), so a ticker handed back already filled would never count
    as fresh. ``stream(factory)`` returns the ``side_effect``: the factory's
    values are captured and delivered by the next ``asyncio.sleep`` poll.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[object, dict]] = []

    def __call__(self, factory):
        def subscribe(contract, **_kw):
            ticker = factory(contract)
            values = {f: getattr(ticker, f) for f in _STREAM_FIELDS if hasattr(ticker, f)}
            self._pending.append((ticker, values))
            return ticker

        return subscribe

    def deliver(self) -> None:
        for ticker, values in self._pending:
            for field, value in values.items():
                setattr(ticker, field, value)
        self._pending.clear()


@pytest.fixture
def stream() -> FakeStream:
    return FakeStream()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch, stream):
    """Replace the ``asyncio.sleep`` polls in chains.py: no wait, deliver pending ticks."""
    import ibtws.unofficial.option.chains as chains_module

    async def _instant(_seconds):
        stream.deliver()

    monkeypatch.setattr(chains_module.asyncio, "sleep", _instant)


@pytest.fixture
def fetcher(fake_client) -> OptionChainFetcher:
    # Short quote_timeout: tests whose tickers never complete must not spin for
    # the production default.
    return OptionChainFetcher(fake_client, quote_timeout=0.2)
