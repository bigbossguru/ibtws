"""Shared fakes for ``ibtws.unofficial.strategies`` tests.

Lightweight stand-ins for option quotes and the IBKR client — exactly what
the credit-spread selectors and the strategy class actually touch. No
network, no event loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ib_async import Option, Ticker

from ibtws.unofficial.option import OptionChainFetcher, OptionQuote


def make_option(
    *,
    symbol: str = "AAPL",
    expiry: str = "20260619",
    strike: float = 150.0,
    right: str = "P",
    con_id: int = 0,
    exchange: str = "SMART",
    currency: str = "USD",
    trading_class: str = "AAPL",
    multiplier: str = "100",
) -> Option:
    opt = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        exchange=exchange,
        currency=currency,
        tradingClass=trading_class,
        multiplier=multiplier,
    )
    if con_id:
        opt.conId = con_id
    return opt


def make_quote(
    *,
    strike: float = 150.0,
    right: str = "P",
    expiry: str = "20260619",
    con_id: int = 1,
    delta: float | None = -0.30,
    bid: float | None = 1.10,
    ask: float | None = 1.30,
    iv: float | None = 0.25,
    volume: float | None = 500.0,
    open_interest: float | None = 1000.0,
    underlying_price: float | None = 155.0,
) -> OptionQuote:
    contract = make_option(strike=strike, right=right, expiry=expiry, con_id=con_id)
    return OptionQuote(
        contract=contract,
        bid=bid,
        ask=ask,
        volume=volume,
        open_interest=open_interest,
        iv=iv,
        delta=delta,
        gamma=0.01,
        vega=0.05,
        theta=-0.02,
        underlying_price=underlying_price,
    )


def make_ticker_for(contract, *, bid: float = 1.0, ask: float = 1.2) -> Ticker:
    """Stand-in ticker for `_current_mid_debit` re-quote calls."""
    t = Ticker()
    t.contract = contract
    t.bid = bid
    t.ask = ask
    return t


@pytest.fixture
def fake_client():
    """Stand-in IBKRClient sufficient for OptionChainFetcher + strategy paths."""
    ib = MagicMock()
    ib.reqSecDefOptParamsAsync = AsyncMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqTickersAsync = AsyncMock()
    ib.placeOrder = MagicMock()
    return SimpleNamespace(ib=ib, qualify=AsyncMock())


@pytest.fixture
def fake_fetcher(fake_client):
    """Fetcher used by strategy tests; selection tests bypass it via patching."""
    return OptionChainFetcher(fake_client)


@pytest.fixture
def fake_manager():
    """Stand-in OrderManager — only the methods the strategy calls are wired."""
    m = MagicMock()
    m.limit = AsyncMock()
    return m
