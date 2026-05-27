"""04 — Expected Move.

Calculate the expected move for an underlying using all three methods:

1. ATM Straddle price (market-implied)
2. ATM Implied Volatility × √(DTE/365)
3. Historical Volatility × √(DTE/365)

Uses frozen market data (``reqMarketDataType(2)``) so the example works when
the live feed is locked by another session.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.analysis.expected_move import ExpectedMoveCalculator, ExpectedMoveResult
from ibtws.unofficial.option.chains import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def _print(result: ExpectedMoveResult) -> None:
    print(f"\n=== {result.underlying_symbol} — expiry {result.expiration} ({result.dte:.0f} DTE) ===")
    print(f"  Spot: ${result.spot:.2f}")
    print()
    print("  Method 1 — ATM Straddle (market-implied):")
    if result.straddle_move:
        print(f"    EM: ±${result.straddle_move:.2f}  ({result.straddle_pct:.2f}%)")
        print(f"    Range: ${result.spot - result.straddle_move:.2f} – ${result.spot + result.straddle_move:.2f}")
    else:
        print("    n/a (quotes unavailable)")
    print()
    print("  Method 2 — IV-based 1σ:")
    if result.iv_move and result.atm_iv:
        print(f"    ATM IV: {result.atm_iv * 100:.1f}%")
        print(f"    EM: ±${result.iv_move:.2f}  ({result.iv_pct:.2f}%)")
        print(f"    Range: ${result.spot - result.iv_move:.2f} – ${result.spot + result.iv_move:.2f}")
    else:
        print("    n/a (IV unavailable)")
    print()
    print("  Method 3 — Historical Volatility 1σ:")
    if result.hv_move and result.hv:
        print(f"    HV (30d): {result.hv * 100:.1f}%")
        print(f"    EM: ±${result.hv_move:.2f}  ({result.hv_pct:.2f}%)")
        print(f"    Range: ${result.spot - result.hv_move:.2f} – ${result.spot + result.hv_move:.2f}")
    else:
        print("    n/a (historical data unavailable)")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)  # TWS paper

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)  # frozen

        chain_fetcher = OptionChainFetcher(client)
        calculator = ExpectedMoveCalculator(client, chain_fetcher)

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        # Pick the nearest monthly expiration (~30 DTE)
        chain_def = await chain_fetcher.fetch_chain_definition(underlying)
        today = _dt.date.today()
        target_dte = 30
        expiration = min(
            chain_def.expirations,
            key=lambda e: abs((_dt.datetime.strptime(e, "%Y%m%d").date() - today).days - target_dte),
        )

        result = await calculator.calculate(underlying, expiration, hv_lookback_days=30)
        _print(result)


if __name__ == "__main__":
    asyncio.run(main())
