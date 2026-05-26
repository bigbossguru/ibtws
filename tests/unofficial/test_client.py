"""Unit tests for ibtws.unofficial.client.IBKRClient.

The real ``ib_async.IB`` is replaced with a lightweight fake so the tests
exercise the simplified client surface (connect, disconnect, market data,
historical data) without touching the network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from ibtws.config import IBKRConfig
from ibtws.unofficial import client as client_module
from ibtws.unofficial.client import IBKRClient


# ---------------------------------------------------------------------------
# Fakes & fixtures
# ---------------------------------------------------------------------------


class FakeIB:
    """Minimal stand-in for ``ib_async.IB`` covering only what the client uses."""

    def __init__(self) -> None:
        self.RequestTimeout: float | None = None
        self.RaiseRequestErrors: bool | None = None

        self._connected = False
        self.disconnect_calls = 0

        self.connectAsync = AsyncMock(side_effect=self._connect_side_effect)
        self.qualifyContractsAsync = AsyncMock()
        self.reqTickersAsync = AsyncMock(return_value=[MagicMock(name="ticker")])
        self.reqHistoricalDataAsync = AsyncMock(return_value=[])
        self.reqMktData = MagicMock(return_value=MagicMock(name="stream-ticker"))
        self.cancelMktData = MagicMock()

    async def _connect_side_effect(self, **_kwargs):
        self._connected = True

    def isConnected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


@pytest.fixture
def config() -> IBKRConfig:
    return IBKRConfig(
        host="127.0.0.1",
        port=4002,
        client_id=7,
        connect_timeout=1.0,
        request_timeout=1.0,
    )


@pytest.fixture
def patch_ib(monkeypatch):
    """Replace ``IB`` in the client module with our fake; return the instance."""
    holder: dict[str, FakeIB] = {}

    def factory():
        ib = FakeIB()
        holder["ib"] = ib
        return ib

    monkeypatch.setattr(client_module, "IB", factory)
    return holder


@pytest.fixture
def client(config, patch_ib) -> IBKRClient:
    return IBKRClient(config)


@pytest.fixture(autouse=True)
def _skip_sleep(monkeypatch):
    """Make asyncio.sleep a no-op so tests don't actually wait."""

    async def _noop(_seconds):
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", _noop)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_applies_config(client, patch_ib, config):
    ib = patch_ib["ib"]
    assert client.config is config
    assert ib.RequestTimeout == config.request_timeout
    assert ib.RaiseRequestErrors is True


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_passes_config(client, patch_ib, config):
    await client.connect()
    patch_ib["ib"].connectAsync.assert_awaited_once_with(
        host=config.host,
        port=config.port,
        clientId=config.client_id,
        timeout=config.connect_timeout,
        readonly=config.readonly,
        account=config.account,
        fetchFields=config.fetch_fields,
    )
    assert patch_ib["ib"].isConnected() is True


async def test_connect_is_idempotent_when_already_connected(client, patch_ib):
    ib = patch_ib["ib"]
    await client.connect()
    ib.connectAsync.reset_mock()

    await client.connect()  # second call must be a no-op
    ib.connectAsync.assert_not_called()


async def test_connect_propagates_exceptions(client, patch_ib):
    patch_ib["ib"].connectAsync.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await client.connect()


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------


async def test_disconnect_closes_socket_when_connected(client, patch_ib):
    await client.connect()
    await client.disconnect()
    assert patch_ib["ib"].disconnect_calls == 1
    assert patch_ib["ib"].isConnected() is False


async def test_disconnect_noop_when_not_connected(client, patch_ib):
    await client.disconnect()
    assert patch_ib["ib"].disconnect_calls == 0


# ---------------------------------------------------------------------------
# async context manager
# ---------------------------------------------------------------------------


async def test_async_context_manager_connects_and_disconnects(client, patch_ib):
    ib = patch_ib["ib"]
    async with client as ctx:
        assert ctx is client
        ib.connectAsync.assert_awaited_once()
        assert ib.isConnected() is True
    assert ib.disconnect_calls == 1
    assert ib.isConnected() is False


# ---------------------------------------------------------------------------
# get_market_data
# ---------------------------------------------------------------------------


async def test_get_market_data_qualifies_and_returns_first_ticker(client, patch_ib):
    ib = patch_ib["ib"]
    expected_ticker = MagicMock(name="expected")
    ib.reqTickersAsync.return_value = [expected_ticker]

    contract = MagicMock(name="contract")
    result = await client.get_market_data(contract)

    ib.qualifyContractsAsync.assert_awaited_once_with(contract)
    ib.reqMktData.assert_called_once_with(contract, genericTickList="100,101,104,106")
    ib.reqTickersAsync.assert_awaited_once_with(contract)
    ib.cancelMktData.assert_called_once_with(contract)
    assert result is expected_ticker


# ---------------------------------------------------------------------------
# get_historical_data
# ---------------------------------------------------------------------------


async def test_get_historical_data_passes_arguments(client, patch_ib, monkeypatch):
    ib = patch_ib["ib"]
    bars = [MagicMock(name="bar")]
    ib.reqHistoricalDataAsync.return_value = bars

    expected_df = pd.DataFrame({"close": [1.0]})
    monkeypatch.setattr(client_module.util, "df", lambda data: expected_df if data is bars else None)

    contract = MagicMock(name="contract")
    df = await client.get_historical_data(
        contract,
        duration="1 D",
        bar_size="5 mins",
        use_rth=False,
        what_to_show="MIDPOINT",
    )

    ib.qualifyContractsAsync.assert_awaited_once_with(contract)
    ib.reqHistoricalDataAsync.assert_awaited_once_with(
        contract,
        endDateTime=None,
        durationStr="1 D",
        barSizeSetting="5 mins",
        whatToShow="MIDPOINT",
        useRTH=False,
    )
    assert df is expected_df


async def test_get_historical_data_returns_empty_dataframe_when_util_df_none(client, patch_ib, monkeypatch):
    patch_ib["ib"].reqHistoricalDataAsync.return_value = []
    monkeypatch.setattr(client_module.util, "df", lambda _data: None)

    df = await client.get_historical_data(
        MagicMock(name="contract"),
        duration="1 D",
        bar_size="1 day",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty
