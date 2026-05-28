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
import datetime as _dt

import pandas as pd
from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)  # TWS paper

    async with IBKRClient(config) as client:
        # 1 = live, 2 = frozen, 3 = delayed, 4 = delayed-frozen
        client.ib.reqMarketDataType(2)

        # underlying = ContFuture("ES", "CME", "USD")
        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        await client.get_market_data(underlying)
        historical_data = await client.get_historical_data(  # noqa: F841
            underlying,
            duration="1 D",
            bar_size="5 mins",
            use_rth=False,
        )
        logging.info(historical_data)

        optchain = OptionChainFetcher(client)

        # Pick the nearest monthly expiration (~30 DTE)
        chain_def = await optchain.fetch_chain_definition(underlying)
        today = _dt.date.today()
        target_dte = 30
        expiration = min(  # noqa: F841
            chain_def.expirations,
            key=lambda e: abs((_dt.datetime.strptime(e, "%Y%m%d").date() - today).days - target_dte),
        )

        logging.info(f"Fetching SPX chain snapshot for {underlying}...")
        df = await optchain.fetch_snapshot(
            underlying,
            expirations=["20260528"],
            trading_class="SPXW",
            strike_window_pct=0.05,
            as_dataframe=True,
        )
        logging.info(f"Successfully fetched SPX chain snapshot for {underlying}...")

        if isinstance(df, pd.DataFrame):
            logging.info(df.head(20))
            df.to_csv(f"output/spx_chain_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
