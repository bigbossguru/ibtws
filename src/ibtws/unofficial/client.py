# Non-official package. This is not an official package and may not be maintained by the original authors of ib_async.
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Literal

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


class IBKRClient:
    def __init__(self, config: IBKRConfig) -> None:
        self.config = config
        self.ib = IB()
        self.ib.RequestTimeout = self.config.request_timeout
        self.ib.RaiseRequestErrors = True

    async def connect(self) -> None:
        if self.ib.isConnected():
            return
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
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IBKRClient: disconnected cleanly.")

    async def get_market_data(self, contract: Contract) -> Ticker:
        await self.ib.qualifyContractsAsync(contract)
        self.ib.reqMktData(contract, genericTickList="100,101,104,106")
        await asyncio.sleep(1.0)  # give IB some time to respond with the tick data
        tickers = await self.ib.reqTickersAsync(contract)
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
    ) -> pd.DataFrame | None:
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
