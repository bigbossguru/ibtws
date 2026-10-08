# Non-official package. This is not an official package and may not be maintained by the original authors of ib_async.
from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
from typing import Any, Awaitable, Callable, Literal

import pandas as pd
from ib_async import IB, Contract, Ticker, util

from ibtws.config import IBKRConfig


logger = logging.getLogger(__name__)

# Bar size strings accepted by IBKR's reqHistoricalData.
BarSize = Literal[
    "1 secs",
    "5 secs",
    "10 secs",
    "15 secs",
    "30 secs",
    "1 min",
    "2 mins",
    "3 mins",
    "5 mins",
    "10 mins",
    "15 mins",
    "20 mins",
    "30 mins",
    "1 hour",
    "2 hours",
    "3 hours",
    "4 hours",
    "8 hours",
    "1 day",
    "1 week",
    "1 month",
]

# Common duration presets. Any `"<int> <S|D|W|M|Y>"` string is also valid.
Duration = Literal[
    "60 S",
    "300 S",
    "1800 S",
    "3600 S",
    "1 D",
    "5 D",
    "10 D",
    "30 D",
    "1 W",
    "4 W",
    "1 M",
    "3 M",
    "6 M",
    "1 Y",
    "2 Y",
    "5 Y",
    "10 Y",
]


# IB system messages that describe the TWS <-> IB server link (not the socket).
_CONNECTIVITY_LOST = 1100
_CONNECTIVITY_RESTORED_DATA_LOST = 1101
_CONNECTIVITY_RESTORED_DATA_KEPT = 1102

ReconnectListener = Callable[[], "Awaitable[None] | None"]


class IBKRClient:
    def __init__(self, config: IBKRConfig) -> None:
        self.config = config
        self.ib = IB()
        self.ib.RequestTimeout = self.config.request_timeout
        self.ib.RaiseRequestErrors = True

        # Set by disconnect() so the disconnect it causes is not mistaken for a drop.
        self._closing = False
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_listeners: list[ReconnectListener] = []
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error

    async def connect(self) -> None:
        self._closing = False
        if self.ib.isConnected():
            return
        await self._connect_once()

    async def _connect_once(self) -> None:
        cfg = self.config
        logger.info(
            "IBKRClient: connecting to %s:%s (clientId=%s)...",
            cfg.host,
            cfg.port,
            cfg.client_id,
        )
        await self.ib.connectAsync(
            host=cfg.host,
            port=cfg.port,
            clientId=cfg.client_id,
            timeout=cfg.connect_timeout,
            readonly=cfg.readonly,
            account=cfg.account,
            fetchFields=cfg.fetch_fields,
        )
        logger.info("IBKRClient: connected successfully.")

    async def disconnect(self) -> None:
        self._closing = True
        task, self._reconnect_task = self._reconnect_task, None
        if task is not None and not task.done():
            task.cancel()
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKRClient: disconnected cleanly.")

    # ------------------------------------------------------------------
    # Connection supervision
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def add_reconnect_listener(self, fn: ReconnectListener) -> None:
        """Call ``fn`` after every successful reconnect and after IB reports a
        restored link that lost market-data subscriptions (code 1101).

        Use it to resubscribe streaming data and resync order state. ``fn`` may
        be a plain function or a coroutine function; exceptions are logged.
        """
        if fn not in self._reconnect_listeners:
            self._reconnect_listeners.append(fn)

    def remove_reconnect_listener(self, fn: ReconnectListener) -> None:
        if fn in self._reconnect_listeners:
            self._reconnect_listeners.remove(fn)

    def _on_disconnected(self) -> None:
        if self._closing:
            return
        logger.error("IBKRClient: connection to TWS lost.")
        if not self.config.auto_reconnect:
            return
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.ensure_future(self._reconnect_loop())

    def _on_error(self, req_id: int, code: int, message: str, *_args: Any) -> None:
        if code == _CONNECTIVITY_LOST:
            logger.error(f"IBKRClient: TWS lost connectivity to IB servers ({message}).")
        elif code == _CONNECTIVITY_RESTORED_DATA_LOST:
            logger.warning("IBKRClient: TWS connectivity restored, market data subscriptions lost; resubscribing.")
            asyncio.ensure_future(self._notify_reconnected())
        elif code == _CONNECTIVITY_RESTORED_DATA_KEPT:
            logger.info("IBKRClient: TWS connectivity restored, data maintained.")

    async def _reconnect_loop(self) -> None:
        cfg = self.config
        delay = max(cfg.reconnect_initial_delay, 0.0)
        attempt = 0
        while not self._closing:
            attempt += 1
            if cfg.reconnect_max_attempts is not None and attempt > cfg.reconnect_max_attempts:
                logger.error(f"IBKRClient: giving up after {cfg.reconnect_max_attempts} reconnect attempt(s).")
                return
            await asyncio.sleep(delay)
            if self._closing:
                return
            try:
                await self._connect_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"IBKRClient: reconnect attempt {attempt} failed: {exc}")
                delay = min(max(delay * 2, 1.0), cfg.reconnect_max_delay)
                continue
            logger.info(f"IBKRClient: reconnected after {attempt} attempt(s).")
            await self._notify_reconnected()
            return

    async def _notify_reconnected(self) -> None:
        for fn in list(self._reconnect_listeners):
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception(f"IBKRClient: reconnect listener {fn!r} failed")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_market_data(
        self,
        contract: Contract,
        *,
        generic_ticks: str = "100,101,104,106",
        settle: float = 1.0,
        timeout: float | None = 15.0,
    ) -> Ticker:
        """Snapshot one contract.

        With ``generic_ticks`` set, a streaming subscription runs for ``settle``
        seconds alongside the snapshot so generic ticks (volume, open interest,
        IV) can arrive; it is always cancelled, even on error. Pass
        ``generic_ticks=""`` for a plain price snapshot without the extra wait.
        """
        await self.ib.qualifyContractsAsync(contract)
        streaming = bool(generic_ticks)
        if streaming:
            self.ib.reqMktData(contract, genericTickList=generic_ticks)
        try:
            if streaming and settle > 0:
                await asyncio.sleep(settle)  # give IB time to stream the generic ticks
            tickers = await asyncio.wait_for(self.ib.reqTickersAsync(contract), timeout=timeout)
        finally:
            if streaming:
                self.ib.cancelMktData(contract)
        if not tickers:
            raise LookupError(f"No ticker data returned for {contract.symbol} (conId={contract.conId})")
        return tickers[0]

    async def get_historical_data(
        self,
        contract: Contract,
        duration: Duration,
        bar_size: BarSize,
        *,
        use_rth: bool = True,
        end_datetime: datetime.datetime | datetime.date | str | None = None,
        what_to_show: Literal[
            "TRADES",
            "MIDPOINT",
            "BID",
            "ASK",
            "BID_ASK",
            "ADJUSTED_LAST",
            "HISTORICAL_VOLATILITY",
            "OPTION_IMPLIED_VOLATILITY",
            "YIELD_BID",
            "YIELD_ASK",
            "YIELD_BID_ASK",
            "YIELD_LAST",
        ] = "TRADES",
    ) -> pd.DataFrame:
        """Fetch historical bars for ``contract``.

        Thin async wrapper around :meth:`ib_async.IB.reqHistoricalDataAsync`.
        """
        await self.ib.qualifyContractsAsync(contract)
        data = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )
        df = util.df(data)
        return df if df is not None else pd.DataFrame()

    async def __aenter__(self) -> "IBKRClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect()
