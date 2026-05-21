"""02 — Full tour of ``ibtws.unofficial.order``.

Each section below is independently togglable. Uncomment the ones you want to
exercise, leave the rest commented. The default config just calls
``manager.start()``, prints the reconciliation report, and exits cleanly.

Sections:
  1. Convenience placement: market / limit / stop / bracket
  2. Pure builders + ``manager.place()`` (when you want to inspect a request first)
  3. Event subscription (sync callback)
  4. Async event stream (``async for``)
  5. Read-only views (open_orders, positions)
  6. Cancel — single + cancel_all
  7. Refresh positions from IB (positionEvent can lag a fill)
  8. Close positions — by conId + close_all_positions
  9. Persistence — replay the audit log
 10. Current PnL — on-demand unrealized PnL for open positions

Audit log: ``orders.jsonl`` (created next to this file).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ib_async import ContFuture

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.order import (
    JsonStore,
    OrderManager,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)
    store = JsonStore(Path(__file__).parent.parent / "output" / "orders.jsonl")

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)  # delayed-frozen — fine for paper

        underlying = ContFuture("MES", exchange="CME", currency="USD")
        [underlying] = await client.qualify(underlying)

        manager = OrderManager(client, store)

        # ── Section 3 — Event callback (register BEFORE start to catch reconcile) ──
        # manager.on_event(lambda e: print(f"  event: {type(e).__name__} {e}"))

        # ── start() runs reconciliation against IB; IB is source of truth ─────────
        report = await manager.start()
        print(
            f"reconcile: matched={len(report.matched)} local_only={len(report.local_only)} "
            f"ib_only={len(report.ib_only)} positions={len(report.positions)}"
        )

        # ── Section 4 — async event stream (remember to cancel the task) ─────────
        # async def consume_events() -> None:
        #     async for e in manager.events():
        #         print(f"  stream: {type(e).__name__} {e}")
        # stream_task = asyncio.create_task(consume_events())

        # ── Section 1 — Convenience placement (build + place in one call) ────────
        # mkt = await manager.market(underlying, OrderSide.BUY, 1)
        # print(f"market   uuid={mkt.uuid} state={mkt.state}")

        # lim = await manager.limit(underlying, OrderSide.BUY, 1, limit_price=3000.0)
        # print(f"limit    uuid={lim.uuid} state={lim.state}")

        # stp = await manager.stop_order(underlying, OrderSide.SELL, 1, stop_price=2900.0)
        # print(f"stop     uuid={stp.uuid} state={stp.state}")

        # parent, tp, sl = await manager.bracket(
        #     underlying,
        #     OrderSide.BUY,
        #     1,
        #     entry_limit_price=3000.0,
        #     take_profit_price=3100.0,
        #     stop_loss_price=2950.0,
        # )
        # print(f"bracket  parent={parent.uuid} tp={tp.uuid} sl={sl.uuid} group={parent.bracket_group}")

        # ── Section 2 — Pure builder + place() (inspect/modify the request first) ─
        # req = build_limit(underlying, OrderSide.BUY, 1, limit_price=2950.0)
        # print(f"preview: {req}")
        # tracked = await manager.place(req)
        # print(f"placed:  uuid={tracked.uuid}")

        await asyncio.sleep(5)

        # ── Section 5 — Read-only views ─────────────────────────────────────────
        print(f"open orders: {len(manager.open_orders)}")
        for t in manager.open_orders:
            print(f"  {t.uuid} {t.state} filled={t.filled} remaining={t.remaining}")

        print(f"positions:   {len(manager.positions)}")
        for pos in manager.positions:
            print(
                f"  {pos.contract.get('symbol')} qty={pos.quantity} "
                f"conId={pos.contract.get('conId')} account={pos.account}"
            )

        # ── Section 6 — Cancel ──────────────────────────────────────────────────
        # Single uuid (idempotent: already-terminal orders are no-ops):
        # await manager.cancel(lim.uuid)
        #
        # All non-terminal tracked orders in one call:
        # cancelled_uuids = await manager.cancel_all()
        # print(f"cancel_all: {len(cancelled_uuids)} cancelled")

        # ── Section 7 — Refresh positions (positionEvent can lag a fill) ────────
        # snaps = await manager.refresh_positions()
        # print(f"refresh: {len(snaps)} live positions")

        # ── Section 8 — Close positions (flatten) ───────────────────────────────
        # Close one specific position by conId, market default:
        # closed = await manager.close_position(649180678, kind="market")
        # if closed:
        #     print(f"close: {closed.uuid} state={closed.state}")
        #
        # Limit-priced close:
        # await manager.close_position(649180678, kind="limit", limit_price=7500.0)
        #
        # Flatten everything — cancels working orders on each contract first:
        # closed_all = await manager.close_all_positions()
        # print(f"close_all: submitted {len(closed_all)} closing orders")
        # for t in closed_all:
        #     print(f"  {t.uuid} state={t.state}")

        await asyncio.sleep(2)

        # ── Section 9 — Replay the persisted audit log ──────────────────────────
        # events = list(store.replay())
        # print(f"audit log: {len(events)} events")
        # for e in events[-5:]:
        #     print(f"  {type(e).__name__}: {e}")

        # ── Section 10 — Current PnL (on-demand unrealized PnL) ─────────────────
        # One batched snapshot of every open position; returns PositionPnL list.
        # market_price / market_value / unrealized_pnl are None when no quote is
        # available (after-hours, missing market-data sub, snapshot timeout) —
        # treat None as "unknown", never as 0.0.
        pnls = await manager.current_pnl()
        # # Filter to a subset by conId if you don't want to quote everything:
        # # pnls = await manager.current_pnl(con_ids=[underlying.conId])
        total = 0.0
        for p in pnls:
            sym = p.contract.get("localSymbol") or p.contract.get("symbol")
            if p.unrealized_pnl is None:
                print(f"  {sym:<20s} qty={p.quantity:+g}  mark=unpriced  cost_basis={p.cost_basis:.2f}")
            else:
                total += p.unrealized_pnl
                print(
                    f"  {sym:<20s} qty={p.quantity:+g}  mark={p.market_price:.4f}  "
                    f"value={p.market_value:.2f}  uPnL={p.unrealized_pnl:+.2f}"
                )
        print(f"current_pnl total uPnL: {total:+.2f}")

        # If Section 4's stream task is running, cancel it before stop():
        # stream_task.cancel()
        # try:
        #     await stream_task
        # except asyncio.CancelledError:
        #     pass

        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
