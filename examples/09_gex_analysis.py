"""09 — Gamma Exposure (GEX) Analysis.

Compute dealer gamma exposure across the SPX option chain to identify:

- Total GEX (aggregate dealer gamma)
- Call Wall (key resistance level)
- Put Wall (key support level)
- Net GEX (directional bias)
- Zero Gamma Level (flip point where dealer hedging reverses)
- Per-strike histogram breakdown
"""

from __future__ import annotations

import asyncio
import logging

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.analysis.gex import GEXCalculator, GEXResult
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option.chains import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def _print(result: GEXResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  GEX Analysis — {result.underlying_symbol} (spot: ${result.spot:.2f})")
    print(f"{'=' * 60}")
    print(f"  Total GEX:        ${result.total_gex:>16,.0f}")
    print(f"  Call GEX Total:   ${result.call_gex_total:>16,.0f}")
    print(f"  Put GEX Total:    ${result.put_gex_total:>16,.0f}")
    print(f"  Net GEX:          ${result.net_gex:>16,.0f}")
    print()
    print(f"  Call Wall:         {result.call_wall}")
    print(f"  Put Wall:          {result.put_wall}")
    print(
        f"  Zero Gamma Level:  {result.zero_gamma_level:.2f}" if result.zero_gamma_level else "  Zero Gamma Level:  n/a"
    )
    print()

    # Top strikes by absolute net GEX
    print("  Top 10 Strikes by |Net GEX|:")
    print(f"  {'Strike':>8}  {'Call GEX':>14}  {'Put GEX':>14}  {'Net GEX':>14}")
    print(f"  {'-' * 8}  {'-' * 14}  {'-' * 14}  {'-' * 14}")
    top = sorted(result.strikes, key=lambda s: abs(s.net_gex), reverse=True)[:10]
    for s in top:
        print(f"  {s.strike:>8.1f}  {s.call_gex:>14,.0f}  {s.put_gex:>14,.0f}  {s.net_gex:>14,.0f}")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=19)  # TWS paper

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)  # frozen

        chain_fetcher = OptionChainFetcher(client)
        calculator = GEXCalculator(client, chain_fetcher)

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        result = await calculator.calculate(
            underlying,
            strike_window_pct=0.10,  # ±10% around spot
        )
        _print(result)


if __name__ == "__main__":
    asyncio.run(main())
