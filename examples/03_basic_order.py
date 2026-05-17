"""03 — Basic order placement.

Connects to TWS paper, submits a far-from-market limit order on AAPL stock
(so it sits in the book without filling), streams every state transition for
30 seconds, then cancels and exits cleanly. The full audit log lives in
``orders.jsonl`` next to this script.

Rerun after a hard kill mid-flight: ``manager.start()`` reconciles the local
log against IB's open orders and rehydrates the previously-submitted order.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ib_async import Stock

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.order import (
    JsonStore,
    OrderManager,
    OrderSide,
    build_limit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def main() -> None:
    config = IBKRConfig(port=7497, client_id=14)
    store = JsonStore(Path(__file__).parent / "orders.jsonl")

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)

        underlying = Stock("AAPL", "SMART", "USD")
        [underlying] = await client.qualify(underlying)

        manager = OrderManager(client, store)
        report = await manager.start()
        print(
            f"reconcile: matched={len(report.matched)} local_only={len(report.local_only)} "
            f"ib_only={len(report.ib_only)} positions={len(report.positions)}"
        )

        manager.on_event(lambda e: print(f"  event: {type(e).__name__} {e}"))

        # Limit 30% below last close — should not fill in any sane session.
        req = build_limit(underlying, OrderSide.BUY, 1, limit_price=50.0)
        tracked = await manager.place(req)
        print(f"submitted uuid={tracked.uuid}")

        await asyncio.sleep(30)

        await manager.cancel(tracked.uuid)
        await asyncio.sleep(2)
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
