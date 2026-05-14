# Non-official package. This is not an official package and may not be maintained by the original authors of ib_async.

"""
A simple, robust and resilient IBKR TWS/Gateway client built on ib_async 2.1.0.

Design goals
------------
* Single responsibility  – thin connection/lifecycle layer only.
* Auto-reconnect         – exponential back-off with configurable ceiling.
* Health monitoring      – periodic watchdog that detects silent disconnects.
* Structured logging     – every state transition is logged, never silently swallowed.
* Async-first            – all public methods are coroutines; a blocking helper is
                           provided for scripts that do not manage their own event loop.
* Resource-safe          – context-manager support guarantees clean teardown.

Usage (async)
-------------
    async with IBKRClient() as client:
        await client.connect()
        positions = await client.ib.reqPositionsAsync()

Usage (sync / script)
---------------------
    client = IBKRClient()
    client.run_sync(my_async_main(client))
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Callable, Optional

from ib_async import IB, Contract

from ibtws.config import IBKRConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class IBKRClient:
    """
    Lifecycle wrapper around ``ib_async.IB`` with auto-reconnect and a watchdog.

    The raw ``IB`` instance is exposed as ``self.ib`` so callers can use the full
    ib_async API without any friction.
    """

    def __init__(
        self,
        config: IBKRConfig,
        *,
        on_connected: Optional[Callable[["IBKRClient"], None]] = None,
        on_disconnected: Optional[Callable[["IBKRClient"], None]] = None,
        on_error: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> None:
        self.config = config
        self.ib = IB()
        self.ib.RequestTimeout = self.config.request_timeout
        self.ib.RaiseRequestErrors = True

        # User-supplied hooks (called from the event loop thread).
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_error = on_error

        # Internal state.
        self._reconnect_attempt: int = 0
        self._connected_at: Optional[float] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutting_down: bool = False

        # Wire up ib_async events.
        self.ib.connectedEvent += self._handle_connected
        self.ib.disconnectedEvent += self._handle_disconnected
        self.ib.errorEvent += self._handle_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.ib.isConnected()

    async def connect(self) -> None:
        """
        Establish a connection to TWS/Gateway.

        Blocks until the connection is live and fully synchronised, or raises
        ``ConnectionError`` after all retry attempts are exhausted.
        """
        await self._connect_once()

    async def disconnect(self) -> None:
        """Gracefully tear down the connection and stop background tasks."""
        self._shutting_down = True
        await self._cancel_background_tasks()
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKRClient: disconnected cleanly.")

    async def qualify(self, *contracts: Contract) -> list[Contract]:
        """
        Qualify one or more contracts (resolves conId, exchange, etc.).

        Raises ``ValueError`` if any contract cannot be resolved.
        """
        resolved = await self.ib.qualifyContractsAsync(*contracts)
        unresolved = [c for c in resolved if not c.conId]
        if unresolved:
            raise ValueError(f"Could not qualify contracts: {unresolved}")
        return resolved

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "IBKRClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Blocking helper (for scripts without an explicit event loop)
    # ------------------------------------------------------------------

    def run_sync(self, coro) -> None:
        """
        Run *coro* inside a fresh event loop.  Ensures the client is always
        disconnected cleanly even if the coroutine raises.

        Example::

            client = IBKRClient()
            client.run_sync(main(client))
        """

        async def _runner():
            await self.connect()
            try:
                await coro
            finally:
                await self.disconnect()

        asyncio.run(_runner())

    # ------------------------------------------------------------------
    # Connection internals
    # ------------------------------------------------------------------

    async def _connect_once(self) -> None:
        """Single connection attempt. Raises on failure."""
        cfg = self.config
        logger.info(
            "IBKRClient: connecting to %s:%d (clientId=%d) …",
            cfg.host,
            cfg.port,
            cfg.client_id,
        )
        try:
            await self.ib.connectAsync(
                host=cfg.host,
                port=cfg.port,
                clientId=cfg.client_id,
                timeout=cfg.connect_timeout,
                readonly=cfg.readonly,
                account=cfg.account,
                fetchFields=cfg.fetch_fields,
            )
        except asyncio.TimeoutError as exc:
            raise ConnectionError(
                f"Timed out connecting to {cfg.host}:{cfg.port} after {cfg.connect_timeout}s"
            ) from exc
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to {cfg.host}:{cfg.port}: {exc}") from exc

    async def _reconnect_loop(self) -> None:
        """
        Exponential back-off reconnect loop.

        Runs in the background after an unexpected disconnect.
        Stops when:
          - connection is re-established, or
          - ``_shutting_down`` is True, or
          - max attempts exceeded (if configured).
        """
        cfg = self.config
        while not self._shutting_down:
            self._reconnect_attempt += 1
            if cfg.reconnect_max_attempts and self._reconnect_attempt > cfg.reconnect_max_attempts:
                logger.error(
                    "IBKRClient: exceeded maximum reconnect attempts (%d). Giving up.",
                    cfg.reconnect_max_attempts,
                )
                return

            delay = min(
                cfg.reconnect_base_delay * (2 ** (self._reconnect_attempt - 1)),
                cfg.reconnect_max_delay,
            )
            # Add ±10 % jitter to avoid thundering herd if multiple clients restart.
            import random

            delay *= 0.9 + 0.2 * random.random()
            delay = math.ceil(delay)

            logger.info(
                "IBKRClient: reconnect attempt %d in %ds …",
                self._reconnect_attempt,
                delay,
            )
            await asyncio.sleep(delay)

            if self._shutting_down:
                return

            try:
                await self._connect_once()
                logger.info("IBKRClient: reconnected on attempt %d.", self._reconnect_attempt)
                self._reconnect_attempt = 0
                return
            except ConnectionError as exc:
                logger.warning("IBKRClient: reconnect attempt %d failed – %s", self._reconnect_attempt, exc)

    async def _cancel_background_tasks(self) -> None:
        for task in (self._watchdog_task, self._reconnect_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._watchdog_task = None
        self._reconnect_task = None

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """
        Periodically pings TWS by requesting the current time.

        ib_async's ``disconnectedEvent`` fires on clean TCP disconnects, but a
        silent network failure may leave the socket in a half-open state.  The
        watchdog detects that scenario.
        """
        interval = self.config.watchdog_interval
        while not self._shutting_down:
            await asyncio.sleep(interval)
            if self._shutting_down:
                return
            if not self.ib.isConnected():
                logger.warning("IBKRClient: watchdog – not connected, skipping ping.")
                continue
            try:
                await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), timeout=10.0)
                logger.debug("IBKRClient: watchdog ping OK.")
            except Exception as exc:
                logger.warning("IBKRClient: watchdog ping failed – %s. Forcing reconnect.", exc)
                self.ib.disconnect()
                # _handle_disconnected will schedule the reconnect loop.

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_connected(self) -> None:
        self._connected_at = time.monotonic()
        logger.info(
            "IBKRClient: connected – account=%s  server_version=%s",
            self.ib.client.serverVersion(),
            self.ib.wrapper.accounts,
        )
        # Cancel any in-flight reconnect loop.
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Start / restart the watchdog.
        if self.config.watchdog_enabled:
            if self._watchdog_task and not self._watchdog_task.done():
                self._watchdog_task.cancel()
            self._watchdog_task = asyncio.ensure_future(self._watchdog())

        if self._on_connected:
            try:
                self._on_connected(self)
            except Exception:
                logger.exception("IBKRClient: on_connected hook raised.")

    def _handle_disconnected(self) -> None:
        uptime = f"{time.monotonic() - self._connected_at:.1f}s" if self._connected_at else "n/a"
        logger.warning("IBKRClient: disconnected (uptime %s).", uptime)
        self._connected_at = None

        if self._on_disconnected:
            try:
                self._on_disconnected(self)
            except Exception:
                logger.exception("IBKRClient: on_disconnected hook raised.")

        if not self._shutting_down:
            # Launch reconnect loop if not already running.
            if not self._reconnect_task or self._reconnect_task.done():
                self._reconnect_task = asyncio.ensure_future(self._reconnect_loop())

    def _handle_error(self, req_id: int, error_code: int, error_str: str, advanced_order_reject: str) -> None:
        """
        Translate IBKR error codes into Python log levels.

        Codes 1100–1300 are connection/system messages.
        Codes 2100–2200 are warnings.
        Codes ≥ 2000 that are not warnings are informational.
        The rest are genuine errors.
        """
        # System / connection messages – treat as warnings, not errors.
        if 1100 <= error_code <= 1300:
            logger.warning("IBKRClient [sys %d]: reqId=%d  %s", error_code, req_id, error_str)
        elif 2100 <= error_code <= 2200:
            logger.warning("IBKRClient [warn %d]: reqId=%d  %s", error_code, req_id, error_str)
        elif error_code in {
            # Benign informational codes – log at DEBUG.
            2104,
            2106,
            2107,
            2108,
            2119,
            2158,
        }:
            logger.debug("IBKRClient [info %d]: reqId=%d  %s", error_code, req_id, error_str)
        else:
            logger.error(
                "IBKRClient [err %d]: reqId=%d  %s  advanced=%s",
                error_code,
                req_id,
                error_str,
                advanced_order_reject or "",
            )

        if self._on_error:
            try:
                self._on_error(req_id, error_code, error_str, advanced_order_reject)
            except Exception:
                logger.exception("IBKRClient: on_error hook raised.")
