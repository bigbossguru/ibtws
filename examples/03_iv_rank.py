"""03 — IV Rank.

Compute IV Rank and IV Percentile for an underlying using IB's daily
``OPTION_IMPLIED_VOLATILITY`` historical series (the 30-day ATM IV that TWS
surfaces in the option-trader pane).

Defaults to a 1-year lookback (~252 trading days). Uses frozen market data
(``reqMarketDataType(2)``) so the example works when the live feed is locked
by another session.
"""

from __future__ import annotations

import asyncio
import logging

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import IVRankCalculator, IVRankResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def _fmt_pct(v: float | None) -> str:
    return f"{v:6.2f}%" if v is not None else "    n/a"


def _fmt_iv(v: float | None) -> str:
    return f"{v * 100:6.2f}%" if v is not None else "    n/a"


def _print(result: IVRankResult) -> None:
    print(f"\n=== {result.underlying_symbol} (as of {result.as_of}) ===")
    print(f"  samples       : {result.sample_size} / {result.lookback_days}d lookback")
    print(f"  current IV    : {_fmt_iv(result.current_iv)}")
    print(f"  min / max IV  : {_fmt_iv(result.min_iv)}  /  {_fmt_iv(result.max_iv)}")
    print(f"  IV Rank       : {_fmt_pct(result.iv_rank)}")
    print(f"  IV Percentile : {_fmt_pct(result.iv_percentile)}")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)  # TWS paper

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)  # frozen

        calculator = IVRankCalculator(client)

        underlyings = [
            Index("SPX", "CBOE", "USD"),
        ]

        for u in underlyings:
            [u] = await client.qualify(u)
            result = await calculator.calculate(u, lookback_days=252, use_rth=False)
            _print(result)


if __name__ == "__main__":
    asyncio.run(main())
