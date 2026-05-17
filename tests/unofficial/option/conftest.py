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
    """Stand-in for IBKRClient exposing only what OptionChainFetcher uses."""
    ib = MagicMock()
    ib.reqSecDefOptParamsAsync = AsyncMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqTickersAsync = AsyncMock()
    return SimpleNamespace(ib=ib, qualify=AsyncMock())


@pytest.fixture
def fetcher(fake_client) -> OptionChainFetcher:
    """Fetcher with pacing disabled so tests don't pay the 25 ms throttle."""
    return OptionChainFetcher(
        fake_client,
        max_concurrency=10,
        pace_per_sec=0,
        snapshot_timeout=1.0,
    )
