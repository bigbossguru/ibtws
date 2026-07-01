"""09 — Gamma Exposure (GEX) Analysis.

Compute dealer gamma exposure across the SPX option chain to identify:

- Total GEX (aggregate dealer gamma)
- Call Wall (key resistance level)
- Put Wall (key support level)
- Net GEX (directional bias)
- Zero Gamma Level (flip point where dealer hedging reverses)
- Per-strike histogram breakdown with matplotlib visualization
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging

import pandas as pd
from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.analysis.gex import GexCalculator
from ibtws.unofficial.option.chains import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)  # TWS paper

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)  # frozen

        chain_fetcher = OptionChainFetcher(client)
        calculator = GexCalculator()

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        quotes = await chain_fetcher.fetch_snapshot(
            underlying,
            expirations=["20260701"],
            strike_window_pct=0.03,
            trading_class="SPXW",
            rights=("C", "P"),
            as_dataframe=True,
        )

        spot = quotes.iloc[0].underlying_price if isinstance(quotes, pd.DataFrame) and not quotes.empty else None
        if not spot or spot <= 0:
            raise ValueError(f"Cannot determine spot price for {underlying.symbol}")

        if isinstance(quotes, pd.DataFrame):
            quotes.to_csv(f"spx_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            calculator.compute(quotes)
            calculator.summary()
            calculator.plot(save_path=f"spx_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == "__main__":
    asyncio.run(main())
