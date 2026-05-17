"""Unit tests for ibtws.unofficial.client.IBKRClient.

The real ``ib_async.IB`` is replaced with a lightweight fake so the tests
exercise the full lifecycle (connect, disconnect, reconnect, watchdog, error
classification) without touching the network or pulling heavy dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ibtws import unofficial as unofficial_pkg  # noqa: F401  (ensure subpackage import)
from ibtws.config import IBKRConfig
from ibtws.unofficial import client as client_module
from ibtws.unofficial.client import IBKRClient


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEvent:
    """ib_async events expose ``+=`` to register handlers and are callable."""

    def __init__(self) -> None:
        self._handlers: list = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def __contains__(self, handler) -> bool:
        return handler in self._handlers

    def emit(self, *args):
        for handler in list(self._handlers):
            handler(*args)


class FakeIB:
    """Minimal stand-in for ``ib_async.IB`` covering only what the client uses."""

    def __init__(self) -> None:
        self.connectedEvent = FakeEvent()
        self.disconnectedEvent = FakeEvent()
        self.errorEvent = FakeEvent()

        self.RequestTimeout: float | None = None
        self.RaiseRequestErrors: bool | None = None

        self._connected = False
        self.disconnect_calls = 0

        self.connectAsync = AsyncMock(side_effect=self._connect_side_effect)
        self.qualifyContractsAsync = AsyncMock()
        self.reqCurrentTimeAsync = AsyncMock()

        self.client = MagicMock()
        self.client.serverVersion.return_value = 176
        self.wrapper = SimpleNamespace(accounts=["DU123"])

    async def _connect_side_effect(self, **_kwargs):
        self._connected = True
        self.connectedEvent.emit()

    def isConnected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        was_connected = self._connected
        self._connected = False
        if was_connected:
            self.disconnectedEvent.emit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_config() -> IBKRConfig:
    """Config tuned for tests: tiny back-off, watchdog off by default."""
    return IBKRConfig(
        connect_timeout=1.0,
        request_timeout=1.0,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.05,
        reconnect_max_attempts=3,
        watchdog_interval=0.05,
        watchdog_enabled=False,
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
def client(fast_config, patch_ib) -> IBKRClient:
    return IBKRClient(fast_config)


# ---------------------------------------------------------------------------
# __init__ / property
# ---------------------------------------------------------------------------


def test_init_wires_events_and_settings(client, patch_ib, fast_config):
    ib = patch_ib["ib"]
    assert ib.RequestTimeout == fast_config.request_timeout
    assert ib.RaiseRequestErrors is True
    assert client._handle_connected in ib.connectedEvent
    assert client._handle_disconnected in ib.disconnectedEvent
    assert client._handle_error in ib.errorEvent
    assert client._reconnect_attempt == 0
    assert client._shutting_down is False


def test_is_connected_proxies(client, patch_ib):
    ib = patch_ib["ib"]
    assert client.is_connected is False
    ib._connected = True
    assert client.is_connected is True


# ---------------------------------------------------------------------------
# connect / _connect_once
# ---------------------------------------------------------------------------


async def test_connect_passes_config(client, patch_ib, fast_config):
    await client.connect()
    patch_ib["ib"].connectAsync.assert_awaited_once_with(
        host=fast_config.host,
        port=fast_config.port,
        clientId=fast_config.client_id,
        timeout=fast_config.connect_timeout,
        readonly=fast_config.readonly,
        account=fast_config.account,
        fetchFields=fast_config.fetch_fields,
    )
    assert client.is_connected is True


async def test_connect_translates_timeout(client, patch_ib):
    patch_ib["ib"].connectAsync.side_effect = asyncio.TimeoutError()
    with pytest.raises(ConnectionError, match="Timed out"):
        await client.connect()


async def test_connect_translates_generic_error(client, patch_ib):
    patch_ib["ib"].connectAsync.side_effect = RuntimeError("boom")
    with pytest.raises(ConnectionError, match="boom"):
        await client.connect()


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------


async def test_disconnect_shuts_down_and_closes_socket(client, patch_ib):
    await client.connect()
    await client.disconnect()
    assert client._shutting_down is True
    assert patch_ib["ib"].disconnect_calls == 1
    assert client.is_connected is False


async def test_disconnect_no_socket_call_when_already_down(client, patch_ib):
    await client.disconnect()  # never connected
    assert patch_ib["ib"].disconnect_calls == 0
    assert client._shutting_down is True


async def test_disconnect_does_not_spawn_reconnect(client, patch_ib):
    await client.connect()
    await client.disconnect()
    assert client._reconnect_task is None


# ---------------------------------------------------------------------------
# qualify
# ---------------------------------------------------------------------------


async def test_qualify_returns_resolved(client, patch_ib):
    c1 = SimpleNamespace(conId=1)
    c2 = SimpleNamespace(conId=2)
    patch_ib["ib"].qualifyContractsAsync.return_value = [c1, c2]
    out = await client.qualify(c1, c2)
    assert out == [c1, c2]


async def test_qualify_raises_on_dropped(client, patch_ib):
    c1 = SimpleNamespace(conId=1)
    c2 = SimpleNamespace(conId=0)
    patch_ib["ib"].qualifyContractsAsync.return_value = [c1]  # dropped one
    with pytest.raises(ValueError, match="1 of 2"):
        await client.qualify(c1, c2)


async def test_qualify_raises_on_missing_conid(client, patch_ib):
    c1 = SimpleNamespace(conId=1)
    c2 = SimpleNamespace(conId=0)
    patch_ib["ib"].qualifyContractsAsync.return_value = [c1, c2]
    with pytest.raises(ValueError):
        await client.qualify(c1, c2)


# ---------------------------------------------------------------------------
# Async context manager + run_sync
# ---------------------------------------------------------------------------


async def test_aexit_calls_disconnect(client, patch_ib):
    async with client:
        await client.connect()
    assert patch_ib["ib"].disconnect_calls == 1


def test_run_sync_disconnects_on_success(fast_config, patch_ib, monkeypatch):
    # Build the client *inside* run_sync so the fake IB is created there.
    c = IBKRClient(fast_config)

    async def work():
        assert c.is_connected is True

    c.run_sync(work())
    assert patch_ib["ib"].disconnect_calls == 1


def test_run_sync_disconnects_on_error(fast_config, patch_ib):
    c = IBKRClient(fast_config)

    async def work():
        raise RuntimeError("user error")

    with pytest.raises(RuntimeError, match="user error"):
        c.run_sync(work())
    assert patch_ib["ib"].disconnect_calls == 1


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected_level",
    [
        (1100, logging.WARNING),
        (1300, logging.WARNING),
        (2100, logging.WARNING),
        (2200, logging.WARNING),
        # Codes 2104/2106/2158 fall inside the 2100-2200 warning range,
        # so they are classified as WARNING (the explicit DEBUG set is for
        # codes outside that range, e.g. nothing currently matches it).
        (2104, logging.WARNING),
        (2158, logging.WARNING),
        (200, logging.ERROR),
        (321, logging.ERROR),
    ],
)
def test_handle_error_log_level(client, caplog, code, expected_level):
    caplog.set_level(logging.DEBUG, logger=client_module.logger.name)
    client._handle_error(req_id=1, error_code=code, error_str="msg", advanced_order_reject="")
    matching = [r for r in caplog.records if r.levelno == expected_level]
    assert matching, f"expected level {expected_level} for code {code}, got {[r.levelno for r in caplog.records]}"


def test_handle_error_invokes_hook(fast_config, patch_ib):
    hook = MagicMock()
    c = IBKRClient(fast_config, on_error=hook)
    c._handle_error(7, 200, "bad", "adv")
    hook.assert_called_once_with(7, 200, "bad", "adv")


def test_handle_error_hook_exception_is_swallowed(fast_config, patch_ib, caplog):
    hook = MagicMock(side_effect=RuntimeError("hook boom"))
    c = IBKRClient(fast_config, on_error=hook)
    caplog.set_level(logging.ERROR, logger=client_module.logger.name)
    c._handle_error(1, 200, "x", "")  # must not raise
    assert any("on_error hook raised" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Connected / disconnected handlers
# ---------------------------------------------------------------------------


async def test_handle_connected_invokes_hook_and_starts_watchdog(fast_config, patch_ib):
    fast_config.watchdog_enabled = True
    hook = MagicMock()
    c = IBKRClient(fast_config, on_connected=hook)
    await c.connect()
    try:
        assert c._watchdog_task is not None
        assert not c._watchdog_task.done()
        hook.assert_called_once_with(c)
    finally:
        await c.disconnect()


async def test_handle_connected_hook_exception_swallowed(fast_config, patch_ib, caplog):
    hook = MagicMock(side_effect=RuntimeError("nope"))
    c = IBKRClient(fast_config, on_connected=hook)
    caplog.set_level(logging.ERROR, logger=client_module.logger.name)
    await c.connect()  # must not raise
    await c.disconnect()
    assert any("on_connected hook raised" in r.message for r in caplog.records)


async def test_handle_disconnected_schedules_reconnect_loop(client, patch_ib):
    await client.connect()
    # Force connectAsync to fail on every reconnect attempt so the loop spins.
    patch_ib["ib"].connectAsync.side_effect = ConnectionError("nope")
    patch_ib["ib"].disconnect()  # triggers _handle_disconnected
    # Give the event loop a tick to schedule the task.
    await asyncio.sleep(0)
    assert client._reconnect_task is not None
    # Tear down without waiting for backoff to finish.
    await client.disconnect()


async def test_handle_disconnected_skips_reconnect_when_shutting_down(client, patch_ib):
    await client.connect()
    client._shutting_down = True
    client._handle_disconnected()
    assert client._reconnect_task is None


async def test_handle_disconnected_invokes_hook(fast_config, patch_ib):
    hook = MagicMock()
    c = IBKRClient(fast_config, on_disconnected=hook)
    await c.connect()
    await c.disconnect()
    hook.assert_called_once_with(c)


# ---------------------------------------------------------------------------
# Reconnect loop
# ---------------------------------------------------------------------------


async def test_reconnect_loop_succeeds_and_resets_attempts(client, patch_ib):
    calls = {"n": 0}

    async def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        patch_ib["ib"]._connected = True
        patch_ib["ib"].connectedEvent.emit()

    patch_ib["ib"].connectAsync.side_effect = flaky

    await client._reconnect_loop()
    assert calls["n"] == 3
    assert client._reconnect_attempt == 0


async def test_reconnect_loop_gives_up_after_max_attempts(client, patch_ib, caplog):
    patch_ib["ib"].connectAsync.side_effect = ConnectionError("permanent")
    caplog.set_level(logging.ERROR, logger=client_module.logger.name)

    await client._reconnect_loop()

    # config.reconnect_max_attempts=3, so 3 failed tries then bail.
    assert patch_ib["ib"].connectAsync.await_count == 3
    assert any("exceeded maximum reconnect attempts" in r.message for r in caplog.records)


async def test_reconnect_loop_exits_when_shutting_down(client, patch_ib):
    patch_ib["ib"].connectAsync.side_effect = ConnectionError("x")

    async def shutdown_soon():
        await asyncio.sleep(0.02)
        client._shutting_down = True

    await asyncio.gather(client._reconnect_loop(), shutdown_soon())
    # Should exit promptly; no real upper bound but well under max attempts.


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


async def test_watchdog_skips_ping_when_disconnected(fast_config, patch_ib, caplog):
    fast_config.watchdog_enabled = True
    fast_config.watchdog_interval = 0.01
    c = IBKRClient(fast_config)
    caplog.set_level(logging.WARNING, logger=client_module.logger.name)

    task = asyncio.ensure_future(c._watchdog())
    await asyncio.sleep(0.03)
    c._shutting_down = True
    await task
    # Ping should NOT have been issued because IB is not connected.
    patch_ib["ib"].reqCurrentTimeAsync.assert_not_called()
    assert any("not connected" in r.message for r in caplog.records)


async def test_watchdog_forces_disconnect_on_ping_failure(fast_config, patch_ib):
    fast_config.watchdog_enabled = True
    fast_config.watchdog_interval = 0.01
    c = IBKRClient(fast_config)
    await c.connect()  # this will ALSO start a watchdog via connected handler
    # Replace the watchdog with one we control to avoid double-task races.
    c._watchdog_task.cancel()
    try:
        await c._watchdog_task
    except BaseException:
        pass

    patch_ib["ib"].reqCurrentTimeAsync.side_effect = TimeoutError("ping timeout")

    task = asyncio.ensure_future(c._watchdog())
    await asyncio.sleep(0.05)
    c._shutting_down = True
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    # Watchdog should have called ib.disconnect() at least once.
    assert patch_ib["ib"].disconnect_calls >= 1


async def test_watchdog_pings_when_connected(fast_config, patch_ib):
    fast_config.watchdog_enabled = True
    fast_config.watchdog_interval = 0.01
    c = IBKRClient(fast_config)
    patch_ib["ib"]._connected = True

    task = asyncio.ensure_future(c._watchdog())
    await asyncio.sleep(0.05)
    c._shutting_down = True
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    assert patch_ib["ib"].reqCurrentTimeAsync.await_count >= 1


# ---------------------------------------------------------------------------
# Background-task cancellation
# ---------------------------------------------------------------------------


async def test_cancel_background_tasks_clears_refs(client):
    async def forever():
        await asyncio.sleep(10)

    client._watchdog_task = asyncio.ensure_future(forever())
    client._reconnect_task = asyncio.ensure_future(forever())

    await client._cancel_background_tasks()

    assert client._watchdog_task is None
    assert client._reconnect_task is None


async def test_cancel_background_tasks_swallows_errors(client):
    async def boom():
        raise RuntimeError("boom")

    t = asyncio.ensure_future(boom())
    await asyncio.sleep(0)  # let it run
    client._reconnect_task = t
    await client._cancel_background_tasks()  # must not raise
    assert client._reconnect_task is None
