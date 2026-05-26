"""10 — Scheduled 0DTE SPX iron condors: open one every 15 min from 9:00 ET.

What this example does
----------------------
* Connects to TWS (paper by default — read every line before flipping the
  ``DRY_RUN`` switch below).
* From the configured ``OPEN_AT_ET`` (default 09:00 ET) until ``STOP_OPENING_AT_ET``
  (default 15:30 ET), every ``INTERVAL_MIN`` minutes (default 15), builds a
  fresh symmetric SPX 0DTE iron condor (bull-put + bear-call on SPXW with
  exact same-day expiry) and submits both sides.
* Each condor runs its own background monitor task with **per-side SL** and
  **one combined take-profit** (close both sides when their combined
  remaining mid-debit falls to ``(1 - COMBINED_TP_PCT) * total_credit``).
* At ``FLATTEN_AT_ET`` (default 15:55 ET) every still-open position is
  market-closed via ``OrderManager.close_all_positions()`` — 0DTE SPX is
  cash-settled but you really don't want to ride the last 5 minutes of
  gamma if the monitor missed an exit.

Caveats — read before going live
--------------------------------
* SPX cash open is 09:30 ET; 09:00 falls in GTH (Global Trading Hours,
  08:15–09:25 ET). ``outside_rth=True`` keeps orders alive across the
  GTH→RTH transition. If you don't want any GTH fills, set
  ``OPEN_AT_ET=time(9, 30)``.
* No spacing between condors beyond the 15 min cadence — you'll be running
  N parallel condors by mid-day. Cap with ``MAX_CONCURRENT_CONDORS``.
* The build can legitimately fail (skew too thin, spread below
  ``min_credit_width_ratio``, today's expiry not yet listed). We log and
  skip that slot rather than abort the day.
* ``DRY_RUN=True`` by default — plans are built and printed but nothing is
  sent to IB.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option import OptionChainFetcher
from ibtws.unofficial.order import JsonStore, OrderManager
from ibtws.unofficial.strategies import (
    CreditSpreadError,
    CreditSpreadParams,
    CreditSpreadPlan,
    CreditSpreadStrategy,
    SpreadType,
)
from ibtws.unofficial.order.models import TrackedOrder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("spx_0dte_ic")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")

DRY_RUN = False  # set False to actually submit
OPEN_AT_ET = time(9, 0)  # first condor at 09:00 ET (GTH — see caveat)
STOP_OPENING_AT_ET = time(15, 30)  # last condor no later than 15:30 ET
FLATTEN_AT_ET = time(15, 55)  # market-close everything by 15:55 ET
INTERVAL_MIN = 15  # one condor every 15 minutes
MAX_CONCURRENT_CONDORS = 8  # safety cap

# Per-condor knobs (kept symmetric — that's what makes it a condor).
TARGET_SHORT_DELTA = 0.199  # ~10Δ short legs — far OTM for 0DTE
MAX_SHORT_DELTA = 0.20
WING_WIDTH = 10.0  # $10 wings
STOP_LOSS_MULTIPLIER = 2.0  # per-side: stop at 2x credit loss
COMBINED_TP_PCT = 0.5  # close BOTH at 50 % of total credit captured
QUANTITY = 1
MIN_CREDIT_WIDTH_RATIO = 0.05  # 0DTE skew is thin; accept ≥5 % of width
LIMIT_SLIPPAGE = 0.05
MONITOR_POLL_SEC = 30.0


# ---------------------------------------------------------------------------
# Condor monitor (per-side SL + combined TP)
# ---------------------------------------------------------------------------


@dataclass
class CondorPosition:
    label: str
    put_plan: CreditSpreadPlan
    call_plan: CreditSpreadPlan
    put_entry: TrackedOrder
    call_entry: TrackedOrder


async def monitor_condor(
    strat: CreditSpreadStrategy,
    condor: CondorPosition,
    *,
    combined_take_profit_pct: float,
    poll_interval: float,
    max_wait: Optional[float] = None,
) -> None:
    """Per-side SL + combined TP loop for one iron condor.

    See example 06 for the same logic inline. Quote drop-outs skip the
    iteration; the combined-TP check only runs when we have a fresh mid on
    every still-open side.
    """
    if not await strat._await_entry_fill(condor.put_entry, max_wait=max_wait):
        logger.warning(f"[{condor.label}] put entry did not fill; aborting monitor")
        return
    if not await strat._await_entry_fill(condor.call_entry, max_wait=max_wait):
        logger.warning(f"[{condor.label}] call entry did not fill; aborting monitor")
        return

    mult = condor.put_plan.multiplier
    total_credit_per_share = (condor.put_plan.net_credit + condor.call_plan.net_credit) / mult
    tp_combined_debit = (1.0 - combined_take_profit_pct) * total_credit_per_share

    open_sides: dict[str, CreditSpreadPlan] = {
        "put": condor.put_plan,
        "call": condor.call_plan,
    }

    started = asyncio.get_event_loop().time()
    while open_sides:
        if max_wait is not None and asyncio.get_event_loop().time() - started > max_wait:
            logger.info(f"[{condor.label}] monitor timeout, {len(open_sides)} side(s) still open")
            return

        await asyncio.sleep(poll_interval)

        mids: dict[str, float] = {}
        for name, plan in open_sides.items():
            mid = await strat._current_mid_debit(plan)
            if mid is not None:
                mids[name] = mid

        if len(mids) == len(open_sides):
            combined = sum(mids.values())
            if combined <= tp_combined_debit:
                logger.info(
                    f"[{condor.label}] combined TP hit — debit {combined:.2f} <= "
                    f"target {tp_combined_debit:.2f} (of {total_credit_per_share:.2f} credit)"
                )
                for name, plan in list(open_sides.items()):
                    mid = mids[name]
                    closed = await strat.close(plan, limit_debit=mid * (1.0 + plan.params.limit_slippage))
                    logger.info(f"[{condor.label}]   closed {name} uuid={closed.uuid}")
                return

        for name in list(open_sides.keys()):
            mid = mids.get(name)
            if mid is None:
                continue
            plan = open_sides[name]
            if plan.stop_loss_debit is not None and mid >= plan.stop_loss_debit:
                logger.warning(f"[{condor.label}] {name} SL hit — mid {mid:.2f} >= SL {plan.stop_loss_debit:.2f}")
                closed = await strat.close(plan, limit_debit=mid * (1.0 + plan.params.limit_slippage))
                logger.info(f"[{condor.label}]   closed {name} uuid={closed.uuid}")
                open_sides.pop(name)


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


def _next_slot_after(now_et: datetime, first_slot: datetime, interval: timedelta) -> datetime:
    """Return the next scheduled slot on/after ``now_et``."""
    if now_et <= first_slot:
        return first_slot
    elapsed = now_et - first_slot
    n = int(elapsed / interval) + 1
    return first_slot + n * interval


async def _sleep_until(target_et: datetime) -> None:
    delay = (target_et - datetime.now(ET)).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)


def _build_params(underlying, today_yyyymmdd: str, spread_type: SpreadType) -> CreditSpreadParams:
    return CreditSpreadParams(
        underlying=underlying,
        spread_type=spread_type,
        target_short_delta=TARGET_SHORT_DELTA,
        max_short_delta=MAX_SHORT_DELTA,
        wing_width=WING_WIDTH,
        target_dte=0,
        dte_tolerance=0,
        expirations=[today_yyyymmdd],
        trading_class="SPXW",
        exchange="SMART",
        currency="USD",
        strike_window_pct=0.03,
        min_credit_width_ratio=MIN_CREDIT_WIDTH_RATIO,
        # Per-side TP off — combined TP enforced in monitor_condor.
        take_profit_pct=None,
        stop_loss_multiplier=STOP_LOSS_MULTIPLIER,
        limit_slippage=LIMIT_SLIPPAGE,
        quantity=QUANTITY,
        min_open_interest=0,
        min_volume=0,
        outside_rth=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=14)
    store = JsonStore(Path(__file__).parent / "orders.jsonl")

    async with IBKRClient(config) as client:
        await client.connect()
        client.ib.reqMarketDataType(2)

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        manager = OrderManager(client, store)
        await manager.start()
        # Shared fetcher with a tight freshness window so the 15-minute scheduler
        # never acts on a stale 0DTE quote; subscriptions are released by
        # CreditSpreadStrategy.close() to stay under IB's per-session cap.
        fetcher = OptionChainFetcher(client)
        strat = CreditSpreadStrategy(client, manager, fetcher=fetcher)

        # Schedule anchors — pin all times to today in ET.
        today_et = datetime.now(ET).date()
        first_slot = datetime.combine(today_et, OPEN_AT_ET, tzinfo=ET)
        stop_opening = datetime.combine(today_et, STOP_OPENING_AT_ET, tzinfo=ET)
        flatten_at = datetime.combine(today_et, FLATTEN_AT_ET, tzinfo=ET)
        interval = timedelta(minutes=INTERVAL_MIN)
        today_yyyymmdd = today_et.strftime("%Y%m%d")

        logger.info(
            f"schedule: first={first_slot.strftime('%H:%M')} ET, "
            f"stop_open={stop_opening.strftime('%H:%M')} ET, "
            f"flatten={flatten_at.strftime('%H:%M')} ET, every {INTERVAL_MIN} min, "
            f"DRY_RUN={DRY_RUN}"
        )

        monitor_tasks: list[asyncio.Task] = []
        opened = 0

        # ---------------- scheduling loop ----------------
        slot = _next_slot_after(datetime.now(ET), first_slot, interval)
        while slot <= stop_opening:
            # await _sleep_until(slot)
            slot_label = slot.strftime("%H:%M")
            logger.info(f"=== slot {slot_label} ET ===")

            # Drop slots once we've hit the cap (still wait for monitors).
            live = sum(1 for t in monitor_tasks if not t.done())
            if live >= MAX_CONCURRENT_CONDORS:
                logger.warning(
                    f"[{slot_label}] {live} condors already running (cap {MAX_CONCURRENT_CONDORS}); skipping this slot"
                )
                slot += interval
                continue

            try:
                put_plan = await strat.build_plan(_build_params(underlying, today_yyyymmdd, SpreadType.BULL_PUT))
                call_plan = await strat.build_plan(_build_params(underlying, today_yyyymmdd, SpreadType.BEAR_CALL))
            except CreditSpreadError as exc:
                logger.warning(f"[{slot_label}] build failed: {exc}")
                slot += interval
                continue

            total_credit = put_plan.net_credit + call_plan.net_credit
            logger.info(
                f"[{slot_label}] PUT  {put_plan.short_leg.strike:g}/{put_plan.long_leg.strike:g} "
                f"credit={put_plan.net_credit:.2f}"
            )
            logger.info(
                f"[{slot_label}] CALL {call_plan.short_leg.strike:g}/{call_plan.long_leg.strike:g} "
                f"credit={call_plan.net_credit:.2f}"
            )
            logger.info(
                f"[{slot_label}] total credit {total_credit:.2f}, "
                f"inner range ({put_plan.short_leg.strike:g}, {call_plan.short_leg.strike:g})"
            )

            if DRY_RUN:
                logger.info(f"[{slot_label}] DRY_RUN — not placing")
                slot += interval
                continue

            try:
                put_entry = await strat.place(put_plan)
                call_entry = await strat.place(call_plan)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[{slot_label}] placement failed: {exc}")
                # If only one side placed, cancel any working order on that side.
                # (cancel_all is cheap and idempotent.)
                await manager.cancel_all()
                slot += interval
                continue

            opened += 1
            condor = CondorPosition(
                label=f"{slot_label}#{opened}",
                put_plan=put_plan,
                call_plan=call_plan,
                put_entry=put_entry,
                call_entry=call_entry,
            )
            logger.info(f"[{condor.label}] placed put={put_entry.uuid} call={call_entry.uuid}")

            # Monitor runs until combined TP, per-side SL on both, or flatten time.
            wait_budget = (flatten_at - datetime.now(ET)).total_seconds()
            monitor_tasks.append(
                asyncio.create_task(
                    monitor_condor(
                        strat,
                        condor,
                        combined_take_profit_pct=COMBINED_TP_PCT,
                        poll_interval=MONITOR_POLL_SEC,
                        max_wait=max(wait_budget, 0),
                    ),
                    name=f"monitor-{condor.label}",
                )
            )
            slot += interval

        # ---------------- post-schedule: drain monitors ----------------
        if monitor_tasks:
            remaining = (flatten_at - datetime.now(ET)).total_seconds()
            if remaining > 0:
                logger.info(
                    f"all slots scheduled; {len(monitor_tasks)} monitor(s) running "
                    f"for up to {remaining:.0f}s until flatten"
                )
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*monitor_tasks, return_exceptions=True),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    pass

        # ---------------- flatten anything still open ----------------
        await _sleep_until(flatten_at)
        if DRY_RUN:
            logger.info("DRY_RUN — skipping flatten + cancel_all")
        else:
            logger.info("flatten window reached — cancelling working orders and closing positions")
            await manager.cancel_all()
            closed = await manager.close_all_positions(kind="market")
            logger.info(f"flatten: submitted {len(closed)} closing market orders")

        # Cancel any monitor tasks still hanging on (they should be done already).
        for t in monitor_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*monitor_tasks, return_exceptions=True)

        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
