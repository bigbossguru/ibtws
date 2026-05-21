"""09 — Exact 0DTE SPX put credit spread (bull-put on SPXW, same-day expiry).

0DTE put-credit-spread checklist:

* Underlying must be ``Index("SPX","CBOE","USD")``; SMART won't route index
  options.
* Pin ``trading_class="SPXW"`` — that's the PM-settled series with daily
  expirations. The plain ``SPX`` series is AM-settled monthly and will not
  include today's expiry.
* ``target_dte=0`` + ``dte_tolerance=0`` forces an exact same-day match.
  If today's SPXW expiry is missing from the chain (rare — early close,
  pre-market run, holiday) the build raises ``CreditSpreadError`` instead
  of silently falling back to tomorrow.
* Short delta around 0.05–0.15 is typical: 0DTE gamma is enormous, so you
  trade further OTM than a 30 DTE spread of equivalent risk.
* Wing widths of 5–25 are typical on SPX. We use 10 here.
* TP is small (25 %) and time-stopped — 0DTEs don't usually round-trip a
  50 % decay before assignment risk dominates.
* Strict ``min_credit_width_ratio`` rejects spreads whose mid credit is so
  thin the slippage round-trip would eat the edge.

CAUTION: this places real-money risk on a paper account by default. Read
every line before uncommenting the ``strat.place(plan)`` block.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher
from ibtws.unofficial.order import JsonStore, OrderManager
from ibtws.unofficial.strategies import (
    CreditSpreadError,
    CreditSpreadParams,
    CreditSpreadStrategy,
    SpreadType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)  # TWS paper
    store = JsonStore(Path(__file__).parent / "orders.jsonl")

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)  # delayed-frozen — fine for paper / outside RTH

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.qualify(underlying)

        manager = OrderManager(client, store)
        await manager.start()

        # Tight freshness window — 0DTE moves fast and stale option quotes
        # mis-price the spread. The fetcher drops tickers older than this
        # before they reach the selector.
        fetcher = OptionChainFetcher(client, quote_max_age_s=5.0)
        strat = CreditSpreadStrategy(client, manager, fetcher=fetcher)

        # Belt-and-braces: pin the expiry to today's date string. Combined
        # with target_dte=0/dte_tolerance=0 this gives a tight, single-day
        # selection window — if today's SPXW expiry is not yet listed (e.g.
        # over a long weekend) the build fails loud.
        today = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")

        params = CreditSpreadParams(
            underlying=underlying,
            spread_type=SpreadType.BULL_PUT,
            target_short_delta=0.10,  # ~10 delta short put — far OTM for 0DTE
            max_short_delta=0.20,  # hard cap: never sell deeper than 20Δ
            wing_width=10.0,  # $10 wide
            target_dte=0,
            dte_tolerance=0,
            expirations=[today],  # exact-day filter
            trading_class="SPXW",  # PM-settled daily series
            exchange="SMART",
            currency="USD",
            strike_window_pct=0.03,  # 0DTEs only live near spot — narrow window
            min_credit_width_ratio=0.10,  # collect at least 10 % of width
            take_profit_pct=0.25,  # exit at 25 % of credit captured
            stop_loss_multiplier=2.0,  # stop out at 2x credit loss
            limit_slippage=0.05,
            quantity=1,
            min_open_interest=100,  # avoid stale strikes
            outside_rth=True,
        )

        try:
            plan = await strat.build_plan(params)
        except CreditSpreadError as exc:
            print(f"could not build 0DTE plan: {exc}")
            await manager.stop()
            return

        print("\n=== 0DTE SPX put credit spread ===")
        print(f"  {plan.describe()}")
        print(f"  spot         : {plan.spot_price}")
        print(f"  short strike : {plan.short_leg.strike:g}  (Δ={plan.short_delta:+.3f})")
        print(f"  long  strike : {plan.long_leg.strike:g}")
        print(f"  net credit   : {plan.net_credit:.2f}")
        print(f"  max profit   : {plan.max_profit:.2f}")
        print(f"  max loss     : {plan.max_loss:.2f}")
        print(f"  R:R          : {plan.risk_reward:.2f}")
        print(f"  breakeven    : {plan.breakeven:.2f}")
        print(f"  TP debit     : {plan.take_profit_debit:.2f} (per-share)")
        print(f"  SL debit     : {plan.stop_loss_debit:.2f} (per-share)")

        # ── UNCOMMENT to submit ─────────────────────────────────────────────
        # tracked = await strat.place(plan)
        # print(f"placed uuid={tracked.uuid} state={tracked.state}")
        #
        # # 0DTE: poll mid every 30s. Hard ceiling of 5 hours covers a full
        # # session start-to-finish; the monitor will exit early on TP or SL.
        # closed = await strat.monitor_and_exit(
        #     plan, tracked, poll_interval=30.0, max_wait=5 * 3600
        # )
        # if closed:
        #     print(f"closed uuid={closed.uuid} state={closed.state}")
        # else:
        #     # No TP/SL hit by EOD — flatten so the position doesn't pin/assign.
        #     print("no TP/SL hit; flattening before expiry")
        #     await manager.close_all_positions()

        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
