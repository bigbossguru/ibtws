"""01 — Basic snapshot.

Smallest possible end-to-end usage: connect, fetch a tight slice of the SPX
index option chain, print the resulting quotes.

SPX is an Index (not a Stock), and its options are listed on CBOE — the
chain definition will pick a trading class (SPX = AM-settled monthlies, or
SPXW = PM-settled weeklies/dailies) automatically.

Sets ``reqMarketDataType(2)`` (frozen) so the example works when the live
feed is locked by another session (TWS error 10197).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pandas as pd
from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def on_connected(client: IBKRClient) -> None:
    print(f"[hook] connected. Server v{client.ib.client.serverVersion()}")


def on_disconnected(client: IBKRClient) -> None:
    print("[hook] disconnected — reconnect loop will spin up automatically.")


def on_error(req_id: int, code: int, msg: str, advanced: str) -> None:
    if code >= 1000:  # filter out genuine errors only
        print(f"[hook] err code={code} reqId={req_id} msg={msg}")


async def main() -> None:
    config = IBKRConfig(port=7497, client_id=14)  # TWS paper

    async with IBKRClient(config) as client:
        await client.connect()

        # 1 = live, 2 = frozen, 3 = delayed, 4 = delayed-frozen
        client.ib.reqMarketDataType(2)

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.qualify(underlying)

        fetcher = OptionChainFetcher(client)

        # SPX has hundreds of listed strikes — auto-window to ±5% of spot to
        # keep the request bounded.
        df = await fetcher.fetch_snapshot(
            underlying,
            expirations=["20260518"],
            trading_class="SPXW",
            strike_window_pct=0.02,
            as_dataframe=True,
        )
        if isinstance(df, pd.DataFrame):
            print(df.head(20))
            df.to_csv(f"spx_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
