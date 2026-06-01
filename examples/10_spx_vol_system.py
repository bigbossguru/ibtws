"""10 — SPX Volatility Analysis System.

Full automated volatility analysis for options premium selling (SPX).
Calculates VIX Z-Score, Expected Move, Term Structure, VRP, and Skew,
then evaluates 0DTE / Weekly / Monthly strategies via a rules engine.

Run during market hours for live signals.
"""

from __future__ import annotations

import asyncio
import logging

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.analysis.spx_vol_system import SPXVolAnalyzer, format_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)

    async with IBKRClient(config) as client:
        await client.connect()

        analyzer = SPXVolAnalyzer(client)
        report = await analyzer.analyze(dte_weekly=7)
        print(format_report(report))


if __name__ == "__main__":
    asyncio.run(main())
