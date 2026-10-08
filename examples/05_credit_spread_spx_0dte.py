"""09 — Exact 0DTE SPX put credit spread (bull-put on SPXW, same-day expiry).

0DTE put-credit-spread checklist:

* Underlying must be ``Index("SPX","CBOE","USD")``; SMART won't route index
  options.
* Pin ``trading_class="SPXW"`` — that's the PM-settled series with daily
  expirations. The plain ``SPX`` series is AM-settled monthly and will not
  include today's expiry.
* ``target_dte=0`` + ``dte_tolerance=0`` forces an exact same-day match.
  DTE is counted in exchange time (America/New_York): today's expiry stays
  0 DTE until the 16:00 ET close wherever this script runs. If today's SPXW
  expiry is missing from the chain (holiday, pre-listing run) the build
  raises ``CreditSpreadError`` instead of silently falling back to tomorrow.
* Short delta around 0.05–0.15 is typical: 0DTE gamma is enormous, so you
  trade further OTM than a 30 DTE spread of equivalent risk.
* Wing widths of 5–25 are typical on SPX. We use 10 here.
* TP is small (25 %) and time-stopped — 0DTEs don't usually round-trip a
  50 % decay before assignment risk dominates.
* Strict ``min_credit_width_ratio`` rejects spreads whose mid credit is so
  thin the slippage round-trip would eat the edge.
* Trading decisions need live quotes. ``LIVE_QUOTES=True`` requests market
  data type 1 and makes the strategy refuse frozen or delayed quotes; set it
  to False only to build plans outside market hours.

Connection details come from ``IBKR_HOST`` / ``IBKR_PORT`` / ``IBKR_CLIENT_ID``
(see :meth:`IBKRConfig.from_env`).

CAUTION: this places real-money risk on a paper account by default. Read
every line before uncommenting the ``strat.place(plan)`` block.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from pathlib import Path

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import MARKET_TZ
from ibtws.unofficial.option import OptionChainFetcher
from ibtws.unofficial.order import JsonStore, OrderManager
from ibtws.unofficial.strategies import (
    CreditSpreadError,
    CreditSpreadParams,
    CreditSpreadStrategy,
    SpreadType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

LIVE_QUOTES = True  # False → frozen data (type 2), only for building plans outside RTH
FLATTEN_AT_ET = time(15, 55)  # close whatever is still open before the 16:00 ET expiry


async def main() -> None:
    config = IBKRConfig.from_env(client_id=14)  # defaults to TWS paper on 127.0.0.1:7497
    store = JsonStore(Path(__file__).parent / "orders.jsonl")

    async with IBKRClient(config) as client:
        client.ib.reqMarketDataType(1 if LIVE_QUOTES else 2)

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        manager = OrderManager(client, store)
        await manager.start()

        fetcher = OptionChainFetcher(client)
        strat = CreditSpreadStrategy(client, manager, fetcher=fetcher)

        # Belt-and-braces: pin the expiry to today's date in exchange time.
        # Combined with target_dte=0/dte_tolerance=0 this gives a tight,
        # single-day selection window — if today's SPXW expiry is not listed
        # the build fails loud.
        now_et = datetime.now(MARKET_TZ)
        today = now_et.strftime("%Y%m%d")

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
            require_live_quotes=LIVE_QUOTES,
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
        # # Manage until TP / SL, or until 15:55 ET. When the deadline passes
        # # with the spread still open, monitor_and_exit closes it as one combo
        # # (escalating to a marketable price) — the entry is cancelled if it
        # # never filled, and a partial fill is managed at its filled size.
        # flatten_at = datetime.combine(now_et.date(), FLATTEN_AT_ET, tzinfo=MARKET_TZ)
        # budget = max((flatten_at - datetime.now(MARKET_TZ)).total_seconds(), 0)
        # closed = await strat.monitor_and_exit(plan, tracked, poll_interval=2.0, max_wait=budget)
        # if closed:
        #     print(f"closed uuid={closed.uuid} state={closed.state}")
        #
        # # Backstop: flatten only this spread's legs, never the whole account.
        # legs = [plan.short_leg.conId, plan.long_leg.conId]
        # await manager.close_all_positions(con_ids=legs)

        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
