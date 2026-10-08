"""10 — Scheduled 0DTE SPX iron condors: open one every 15 min from 9:00 ET.

What this example does
----------------------
* Connects to TWS (paper by default — read every line before flipping the
  ``DRY_RUN`` switch below). Connection details come from ``IBKR_HOST`` /
  ``IBKR_PORT`` / ``IBKR_CLIENT_ID`` (see :meth:`IBKRConfig.from_env`).
* From the configured ``OPEN_AT_ET`` (default 09:00 ET) until ``STOP_OPENING_AT_ET``
  (default 15:30 ET), every ``INTERVAL_MIN`` minutes (default 15), builds a
  fresh symmetric SPX 0DTE iron condor (bull-put + bear-call on SPXW with
  exact same-day expiry) and submits both sides.
* Each condor runs its own background monitor task with **per-side SL** and
  **one combined take-profit** (close both sides when their combined
  remaining mid-debit falls to ``(1 - COMBINED_TP_PCT) * total_credit``).
  Every exit is confirmed: an unfilled close is cancelled, re-priced and
  re-sent, and stop-loss exits escalate to a marketable price.
* Each side is managed at the size that actually filled. If one side never
  fills its order is cancelled and the other side is still managed.
* At ``FLATTEN_AT_ET`` (default 15:55 ET) every monitor closes what is left
  of its condor as combos; afterwards ``close_all_positions`` runs as a
  backstop restricted to the legs this script traded.

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

from ib_async import Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import MARKET_TZ as ET
from ibtws.unofficial.option import OptionChainFetcher
from ibtws.unofficial.order import JsonStore, OrderManager
from ibtws.unofficial.order.models import TrackedOrder
from ibtws.unofficial.strategies import (
    CreditSpreadError,
    CreditSpreadParams,
    CreditSpreadPlan,
    CreditSpreadStrategy,
    SpreadType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("spx_0dte_ic")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DRY_RUN = True  # set False to actually submit
OPEN_AT_ET = time(9, 0)  # first condor at 09:00 ET (GTH — see caveat)
STOP_OPENING_AT_ET = time(15, 30)  # last condor no later than 15:30 ET
FLATTEN_AT_ET = time(15, 55)  # close everything by 15:55 ET
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
MONITOR_POLL_SEC = 2.0  # streaming quotes: polling is a local read, no IB round trip
MAX_QUOTE_FAILURES = 60  # ~2 min without a usable quote → close the condor
LIVE_QUOTES = True  # require real-time quotes for every decision


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
    client: IBKRClient,
    strat: CreditSpreadStrategy,
    condor: CondorPosition,
    *,
    combined_take_profit_pct: float,
    poll_interval: float,
    deadline: datetime,
) -> None:
    """Per-side SL + combined TP loop for one iron condor.

    Both entries are awaited together; each side is then managed at its
    filled size. Quote drop-outs skip the iteration, and the combined-TP
    check only runs when every still-open side has a fresh mid. At the
    deadline, after repeated quote loss while connected, or on any error,
    whatever is open is closed urgently. While TWS is disconnected the loop
    just waits: the client reconnects and the strategy resubscribes quotes.
    """

    def seconds_left() -> float:
        return max((deadline - datetime.now(ET)).total_seconds(), 0.0)

    put_qty, call_qty = await asyncio.gather(
        strat.await_entry(condor.put_entry, max_wait=seconds_left(), quantity=QUANTITY),
        strat.await_entry(condor.call_entry, max_wait=seconds_left(), quantity=QUANTITY),
    )
    open_sides: dict[str, tuple[CreditSpreadPlan, float]] = {
        name: (plan, qty)
        for name, plan, qty in (("put", condor.put_plan, put_qty), ("call", condor.call_plan, call_qty))
        if qty > 0
    }
    if not open_sides:
        logger.warning(f"[{condor.label}] neither side filled; nothing to manage")
        return
    if len(open_sides) == 1:
        logger.warning(f"[{condor.label}] only the {next(iter(open_sides))} side filled; managing it alone")

    mult = condor.put_plan.multiplier
    total_credit_per_share = sum(plan.net_credit for plan, _ in open_sides.values()) / mult
    tp_combined_debit = (1.0 - combined_take_profit_pct) * total_credit_per_share

    async def close_side(name: str, *, urgent: bool, mid: Optional[float] = None) -> None:
        plan, qty = open_sides[name]
        result = await strat.close_and_confirm(plan, qty, urgent=urgent, mid_debit=mid)
        logger.info(f"[{condor.label}]   {name}: closed {result.closed_quantity:g}/{qty:g}")
        if result.complete:
            open_sides.pop(name)
        else:
            open_sides[name] = (plan, result.remaining_quantity)

    watched = [plan for plan, _ in open_sides.values()]
    for plan in watched:
        strat.watch(plan)
    failures = 0
    try:
        while open_sides:
            if seconds_left() <= 0:
                logger.info(f"[{condor.label}] flatten time — closing {len(open_sides)} side(s)")
                for name in list(open_sides):
                    await close_side(name, urgent=True)
                return

            await asyncio.sleep(min(poll_interval, seconds_left()))

            mids: dict[str, float] = {}
            for name, (plan, _) in open_sides.items():
                mid = await strat.current_mid_debit(plan)
                if mid is not None:
                    mids[name] = mid

            if len(mids) < len(open_sides):
                failures += 1
                if failures >= MAX_QUOTE_FAILURES and client.is_connected():
                    logger.error(f"[{condor.label}] no usable quotes for {failures} polls — closing")
                    for name in list(open_sides):
                        await close_side(name, urgent=True)
                    return
            else:
                failures = 0

            if len(mids) == len(open_sides):
                combined = sum(mids.values())
                if combined <= tp_combined_debit:
                    logger.info(
                        f"[{condor.label}] combined TP hit — debit {combined:.2f} <= "
                        f"target {tp_combined_debit:.2f} (of {total_credit_per_share:.2f} credit)"
                    )
                    for name in list(open_sides):
                        await close_side(name, urgent=False, mid=mids[name])
                    continue

            for name in list(open_sides):
                mid = mids.get(name)
                plan, _ = open_sides[name]
                if mid is not None and plan.stop_loss_debit is not None and mid >= plan.stop_loss_debit:
                    logger.warning(f"[{condor.label}] {name} SL hit — mid {mid:.2f} >= SL {plan.stop_loss_debit:.2f}")
                    await close_side(name, urgent=True, mid=mid)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(f"[{condor.label}] monitor failed — closing what is open")
        for name in list(open_sides):
            await close_side(name, urgent=True)
    finally:
        for plan in watched:
            strat.unwatch(plan)


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
        require_live_quotes=LIVE_QUOTES,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    config = IBKRConfig.from_env(client_id=14)  # defaults to TWS paper on 127.0.0.1:7497
    store = JsonStore(Path(__file__).parent / "orders.jsonl")

    async with IBKRClient(config) as client:
        client.ib.reqMarketDataType(1 if LIVE_QUOTES else 2)

        underlying = Index("SPX", "CBOE", "USD")
        [underlying] = await client.ib.qualifyContractsAsync(underlying)

        manager = OrderManager(client, store)
        await manager.start()
        # One shared fetcher: it caches the SPXW chain definition and the
        # qualified option contracts across the day's slots.
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
        traded_legs: set[int] = set()
        opened = 0

        # ---------------- scheduling loop ----------------
        slot = _next_slot_after(datetime.now(ET), first_slot, interval)
        while slot <= stop_opening:
            await _sleep_until(slot)
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

            placed: list[TrackedOrder] = []
            try:
                placed.append(await strat.place(put_plan))
                placed.append(await strat.place(call_plan))
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[{slot_label}] placement failed: {exc}")
                # Cancel only this slot's order(s); other condors keep running.
                for order in placed:
                    await manager.cancel(order.uuid)
                slot += interval
                continue

            for plan in (put_plan, call_plan):
                traded_legs.update((plan.short_leg.conId, plan.long_leg.conId))
            opened += 1
            condor = CondorPosition(
                label=f"{slot_label}#{opened}",
                put_plan=put_plan,
                call_plan=call_plan,
                put_entry=placed[0],
                call_entry=placed[1],
            )
            logger.info(f"[{condor.label}] placed put={condor.put_entry.uuid} call={condor.call_entry.uuid}")

            # Monitor runs until combined TP, per-side SL on both, or flatten time.
            monitor_tasks.append(
                asyncio.create_task(
                    monitor_condor(
                        client,
                        strat,
                        condor,
                        combined_take_profit_pct=COMBINED_TP_PCT,
                        poll_interval=MONITOR_POLL_SEC,
                        deadline=flatten_at,
                    ),
                    name=f"monitor-{condor.label}",
                )
            )
            slot += interval

        # ---------------- post-schedule: let monitors finish ----------------
        # Each monitor closes its own condor at flatten time, so just wait.
        if monitor_tasks:
            logger.info(f"all slots scheduled; waiting for {len(monitor_tasks)} monitor(s)")
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

        # ---------------- backstop: flatten this script's legs ----------------
        await _sleep_until(flatten_at)
        if DRY_RUN:
            logger.info("DRY_RUN — skipping flatten backstop")
        elif traded_legs:
            await manager.cancel_all()
            closed = await manager.close_all_positions(kind="market", con_ids=traded_legs)
            logger.info(f"flatten backstop: submitted {len(closed)} closing market order(s)")

        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
