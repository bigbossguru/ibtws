"""End-to-end verification against a live TWS / IB Gateway **paper** session.

Places, fills, cancels and closes real (paper) orders to exercise the whole
stack: order lifecycle, IB rejections, brackets, close_position, restart
rehydration, disconnect/reconnect with resync, quote stream resubscription,
and the credit-spread strategy's entry / take-profit / stop-loss / chase
exits on today's SPXW 0DTE chain, plus the example 06 condor monitor.

Safety:
* OrderManager refuses to start on a non-paper (non ``DU``) account.
* Pre-existing positions are never closed: cleanup only restores the
  contracts this run touched to their starting quantity.
* Run during US regular trading hours on a session with SPX options data.

Usage::

    TWS_HOST=127.0.0.1 TWS_PORT=7497 TWS_CLIENT_ID=27 poetry run python scripts/live_paper_check.py

Exits with status 1 when any check fails.
"""

import asyncio
import dataclasses
import sys
import tempfile
import importlib.util
import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from ib_async import Index, Stock

from ibtws.config import IBKRConfig
from ibtws.unofficial.analysis.expected_move import ExpectedMoveCalculator
from ibtws.unofficial.analysis.gex import GexCalculator
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import MARKET_TZ
from ibtws.unofficial.option import IVRankCalculator, OptionChainFetcher, quotes_to_dataframe
from ibtws.unofficial.order import (
    Filled,
    JsonStore,
    OrderManager,
    OrderSide,
    OrderState,
    Rejected,
    RequestSubmitted,
)
from ibtws.unofficial.strategies import CreditSpreadParams, CreditSpreadStrategy, SpreadType
from ibtws.unofficial.strategies.utils import _round_to_tick

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logging.getLogger("ibtws").setLevel(logging.WARNING)
logging.getLogger("ib_async").setLevel(logging.CRITICAL)
REPO = Path(__file__).resolve().parents[1]
TERMINAL = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED}
results: list[tuple[str, bool, str]] = []
touched: set[int] = set()


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


class Ctx:
    pass


async def section(title, coro_fn, ctx):
    print(f"\n=== {title}", flush=True)
    t0 = time.monotonic()
    try:
        await coro_fn(ctx)
    except Exception as exc:  # noqa: BLE001
        check(f"{title}: no exception", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    print(f"  ({time.monotonic() - t0:.0f}s)", flush=True)


async def wait_reconnect(ctx, before, timeout=30):
    for _ in range(int(timeout / 0.25)):
        if len(ctx.reconnects) > before and ctx.client.is_connected():
            return True
        await asyncio.sleep(0.25)
    return False


def positions_by_conid(om):
    out: dict[int, float] = {}
    for p in om.positions:
        cid = p.contract.get("conId")
        out[cid] = out.get(cid, 0.0) + p.quantity
    return out


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def s_read_only(ctx):
    bars = await ctx.client.get_historical_data(ctx.aapl, "1 D", "5 mins")
    check("historical bars returned", len(bars) > 10, f"{len(bars)} bars")
    ivr = await IVRankCalculator(ctx.client).calculate(ctx.spx, lookback_days=60)
    check(
        "IV rank computed", ivr.sample_size > 10 and ivr.iv_rank is not None, f"rank={ivr.iv_rank} n={ivr.sample_size}"
    )
    t = time.perf_counter()
    quotes = await ctx.fetcher.fetch_snapshot(
        ctx.spx, trading_class="SPXW", expirations=[ctx.today], strike_window_pct=0.01
    )
    check(
        "per-contract snapshot of the 0DTE window",
        len(quotes) > 20,
        f"{len(quotes)} quotes in {time.perf_counter() - t:.1f}s",
    )
    df = quotes_to_dataframe(quotes)
    em = ExpectedMoveCalculator().calculate(df)
    check(
        "expected move computed",
        em.straddle_move and em.iv_move,
        f"straddle={em.straddle_move:.1f} iv={em.iv_move:.1f}",
    )
    t = time.perf_counter()
    gex = GexCalculator().compute(df)
    check("GEX on live chain", gex.spot > 0, f"{time.perf_counter() - t:.3f}s zgl={gex.zero_gamma_level}")
    pnl = await ctx.om.current_pnl()
    priced = [p for p in pnl if p.market_price is not None]
    check("current_pnl prices existing positions", len(priced) == len(pnl) and pnl, f"{len(priced)}/{len(pnl)}")


async def s_rejection(ctx):
    # Any product the account may not trade works here; EU retail accounts
    # are refused US ETFs such as SPY (no KID) with error 201.
    [etf] = await ctx.client.ib.qualifyContractsAsync(Stock("SPY", "SMART", "USD"))
    order = await ctx.om.limit(etf, OrderSide.BUY, 1, 1.00)
    ok = await ctx.om.wait_for(lambda: order.state in TERMINAL, timeout=15)
    if order.state == OrderState.REJECTED:
        check("IB rejection closes the order", ok, order.state.value)
        check("rejected order not open", order not in ctx.om.open_orders)
    else:
        # Account is allowed to trade it: just clean up.
        await ctx.om.cancel(order.uuid)
        await ctx.om.wait_for(lambda: order.state in TERMINAL, timeout=15)
        check("SPY accepted (no rejection to test on this account)", True, order.state.value)


async def s_stop_and_streams(ctx):
    got = {"a": [], "b": []}

    async def consume(name, stream):
        async for ev in stream:
            got[name].append(ev)

    tasks = [asyncio.create_task(consume(n, ctx.om.events())) for n in ("a", "b")]
    await asyncio.sleep(0)
    stop = await ctx.om.stop_order(ctx.aapl, OrderSide.BUY, 1, round(ctx.px * 1.5, 2))
    ok = await ctx.om.wait_for(lambda: stop.state == OrderState.SUBMITTED, timeout=15)
    check("stop order working", ok, stop.state.value)
    await ctx.om.cancel(stop.uuid)
    ok = await ctx.om.wait_for(lambda: stop.state == OrderState.CANCELLED, timeout=15)
    check("stop order cancel confirmed", ok, stop.state.value)
    await asyncio.sleep(0.5)
    for t in tasks:
        t.cancel()
    a = [e.uuid for e in got["a"] if getattr(e, "uuid", None) == stop.uuid]
    b = [e.uuid for e in got["b"] if getattr(e, "uuid", None) == stop.uuid]
    check("both events() subscribers got every event", a and a == b, f"a={len(a)} b={len(b)}")


async def s_bracket_fill_then_close(ctx):
    touched.add(ctx.aapl.conId)
    legs = await ctx.om.bracket(
        ctx.aapl, OrderSide.BUY, 1, take_profit_price=round(ctx.px * 1.5, 2), stop_loss_price=round(ctx.px * 0.5, 2)
    )
    parent, tp, sl = legs
    ok = await ctx.om.wait_for(lambda: parent.state == OrderState.FILLED, timeout=30)
    check("bracket market entry filled", ok, parent.state.value)
    ok = await ctx.om.wait_for(
        lambda: tp.state == OrderState.SUBMITTED and sl.state == OrderState.SUBMITTED, timeout=15
    )
    check("TP and SL working after entry", ok, f"tp={tp.state.value} sl={sl.state.value}")
    await asyncio.sleep(1)
    pnl = await ctx.om.current_pnl()
    by = {p.contract["conId"]: p for p in pnl}
    a = by.get(ctx.aapl.conId)
    others = [p for c, p in by.items() if c != ctx.aapl.conId]
    check(
        "current_pnl: AAPL row present (price may be None without a NASDAQ subscription)",
        a is not None,
        f"{a.quantity}@{a.market_price}" if a else None,
    )
    check(
        "current_pnl: other positions still priced despite AAPL error",
        others and all(p.market_price for p in others),
        f"{sum(1 for p in others if p.market_price)}/{len(others)}",
    )
    closing = await ctx.om.close_position(ctx.aapl.conId)
    check(
        "close_position cancelled TP+SL first",
        tp.state == OrderState.CANCELLED and sl.state == OrderState.CANCELLED,
        f"tp={tp.state.value} sl={sl.state.value}",
    )
    ok = await ctx.om.wait_for(lambda: closing.state == OrderState.FILLED, timeout=30)
    check("closing order filled", ok, closing.state.value)
    await asyncio.sleep(1)
    await ctx.om.refresh_positions()
    check("AAPL back to start", positions_by_conid(ctx.om).get(ctx.aapl.conId, 0) == ctx.start.get(ctx.aapl.conId, 0))


async def s_disconnect_mid_order(ctx):
    seen = []
    ctx.om.on_event(seen.append)
    before = len(ctx.reconnects)
    m = await ctx.om.market(ctx.aapl, OrderSide.BUY, 1)
    ctx.client.ib.disconnect()  # drop right after sending; the fill happens while we are offline
    # An order sent while disconnected must fail cleanly and be journaled as Rejected.
    try:
        await ctx.om.limit(ctx.aapl, OrderSide.BUY, 1, round(ctx.px * 0.5, 2))
        check("placing while disconnected raises", False)
    except Exception as exc:  # noqa: BLE001
        check("placing while disconnected raises", True, type(exc).__name__)
    ok = await wait_reconnect(ctx, before)
    check("auto-reconnected", ok, f"listener calls={len(ctx.reconnects) - before}")
    ok = await ctx.om.wait_for(lambda: m.state == OrderState.FILLED, timeout=20)
    check("fill during disconnect reached the tracked order", ok, m.state.value)
    await asyncio.sleep(1)
    fills = [e for e in seen if isinstance(e, Filled) and e.uuid == m.uuid]
    check("Filled event delivered exactly once", len(fills) == 1, f"{len(fills)}")
    rejected = [e for e in ctx.store.replay() if isinstance(e, Rejected) and "placeOrder failed" in e.reason]
    check("offline placement journaled as Rejected", rejected, rejected[-1].reason[:60] if rejected else "")
    s = await ctx.om.market(ctx.aapl, OrderSide.SELL, 1)
    ok = await ctx.om.wait_for(lambda: s.state == OrderState.FILLED, timeout=30)
    check("sell back filled", ok, s.state.value)


async def s_restart_with_bracket(ctx):
    legs = await ctx.om.bracket(
        ctx.aapl,
        OrderSide.BUY,
        1,
        entry_limit_price=round(ctx.px * 0.5, 2),
        take_profit_price=round(ctx.px * 0.6, 2),
        stop_loss_price=round(ctx.px * 0.4, 2),
    )
    await ctx.om.wait_for(lambda: legs[0].state == OrderState.SUBMITTED, timeout=15)
    await asyncio.sleep(1)
    await ctx.om.stop()
    om2 = OrderManager(ctx.client, ctx.store)
    report = await om2.start()
    uuids = [t.uuid for t in legs]
    check(
        "restart matched all bracket members", all(u in report.matched for u in uuids), f"{len(report.matched)} matched"
    )
    check("no phantom local_only orders", report.local_only == [], f"{report.local_only}")
    ctx.om = om2
    await om2.cancel(legs[0].uuid)
    ok = await om2.wait_for(
        lambda: all(om2._tracked[u].state in TERMINAL for u in uuids if u in om2._tracked), timeout=20
    )
    check("cancel through new manager cascades", ok, [om2._tracked[u].state.value for u in uuids if u in om2._tracked])


async def s_build_plans(ctx):
    def params(spread_type, **kw):
        base = dict(
            underlying=ctx.spx,
            spread_type=spread_type,
            target_short_delta=0.15,
            max_short_delta=0.25,
            wing_width=5.0,
            target_dte=0,
            dte_tolerance=0,
            trading_class="SPXW",
            strike_window_pct=0.03,
            min_credit_width_ratio=0.01,
            take_profit_pct=0.5,
            stop_loss_multiplier=2.0,
            limit_slippage=0.0,
            quantity=1,
            require_live_quotes=True,
        )
        base.update(kw)
        return CreditSpreadParams(**base)

    ctx.put_plan = await ctx.strat.build_plan(params(SpreadType.BULL_PUT))
    ctx.call_plan = await ctx.strat.build_plan(params(SpreadType.BEAR_CALL))
    for plan in (ctx.put_plan, ctx.call_plan):
        touched.update((plan.short_leg.conId, plan.long_leg.conId))
    mid = await ctx.strat.current_mid_debit(ctx.put_plan)  # unwatched -> snapshot path
    check("snapshot mid debit (unwatched)", mid is not None, mid)
    check("put plan", ctx.put_plan.expiry == ctx.today, ctx.put_plan.describe().encode("ascii", "replace").decode())
    check("call plan", ctx.call_plan.expiry == ctx.today, ctx.call_plan.describe().encode("ascii", "replace").decode())


def fill_credit(plan):
    """A credit a little under mid so paper fills the entry promptly."""
    return _round_to_tick(plan.net_credit / plan.multiplier * 0.8, 0.05, mode="down")


async def legs_flat(ctx, plan):
    await asyncio.sleep(2)
    await ctx.om.refresh_positions()
    pos = positions_by_conid(ctx.om)
    diffs = {c: pos.get(c, 0) - ctx.start.get(c, 0) for c in (plan.short_leg.conId, plan.long_leg.conId)}
    return all(abs(d) < 1e-9 for d in diffs.values()), diffs


async def s_streams_survive_reconnect(ctx):
    plan = ctx.put_plan
    ctx.strat.watch(plan)
    try:
        await asyncio.sleep(2)
        m1 = await ctx.strat.current_mid_debit(plan)
        check("streaming mid before drop", m1 is not None, m1)
        before = len(ctx.reconnects)
        ctx.client.ib.disconnect()
        ok = await wait_reconnect(ctx, before)
        check("reconnected", ok)
        m2 = None
        for _ in range(20):
            await asyncio.sleep(0.5)
            m2 = await ctx.strat.current_mid_debit(plan)
            if m2 is not None:
                break
        check("streaming mid after reconnect (resubscribed)", m2 is not None, m2)
    finally:
        ctx.strat.unwatch(plan)


async def s_entry_never_fills(ctx):
    plan = ctx.put_plan
    entry = await ctx.strat.place(plan, limit_credit=round(plan.width * 0.9, 2))
    closed = await ctx.strat.monitor_and_exit(plan, entry, poll_interval=1, max_wait=15)
    check("unfilled entry: monitor returns None", closed is None)
    check("unfilled entry cancelled at deadline", entry.state == OrderState.CANCELLED, entry.state.value)


async def s_take_profit(ctx):
    plan = dataclasses.replace(ctx.put_plan, take_profit_debit=ctx.put_plan.width, stop_loss_debit=None)
    entry = await ctx.strat.place(plan, limit_credit=fill_credit(plan))
    closed = await ctx.strat.monitor_and_exit(plan, entry, poll_interval=1, max_wait=120)
    check("TP: entry filled", entry.state == OrderState.FILLED, f"{entry.state.value} filled={entry.filled}")
    check(
        "TP: closed via take-profit",
        closed is not None and closed.state == OrderState.FILLED,
        closed.state.value if closed else None,
    )
    ok, diffs = await legs_flat(ctx, plan)
    check("TP: legs flat", ok, diffs)


async def s_stop_loss(ctx):
    plan = dataclasses.replace(ctx.put_plan, take_profit_debit=None, stop_loss_debit=0.05)
    entry = await ctx.strat.place(plan, limit_credit=fill_credit(plan))
    closed = await ctx.strat.monitor_and_exit(plan, entry, poll_interval=1, max_wait=120)
    check(
        "SL: closed urgently",
        closed is not None and closed.state == OrderState.FILLED,
        closed.state.value if closed else None,
    )
    ok, diffs = await legs_flat(ctx, plan)
    check("SL: legs flat", ok, diffs)


async def s_chase_and_escalate(ctx):
    plan = ctx.put_plan
    entry = await ctx.strat.place(plan, limit_credit=fill_credit(plan))
    filled = await ctx.strat.await_entry(entry, max_wait=60, quantity=1)
    check("chase: entry filled", filled == 1, filled)
    if filled <= 0:
        return
    sent_before = len([e for e in ctx.store.replay() if isinstance(e, RequestSubmitted)])
    # First attempt priced at one tick: cannot fill, must be cancelled, re-quoted and escalated.
    result = await ctx.strat.close_and_confirm(plan, filled, urgent=True, mid_debit=0.01)
    sent = len([e for e in ctx.store.replay() if isinstance(e, RequestSubmitted)]) - sent_before
    check("chase: exit completed", result.complete, f"closed={result.closed_quantity}")
    check("chase: needed more than one attempt", sent >= 2, f"{sent} closing orders")
    ok, diffs = await legs_flat(ctx, plan)
    check("chase: legs flat", ok, diffs)


async def s_condor_example(ctx):
    spec = importlib.util.spec_from_file_location("ex06", REPO / "examples" / "06_iron_condor_spx_0dte.py")
    ex06 = importlib.util.module_from_spec(spec)
    sys.modules["ex06"] = ex06  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(ex06)
    put_plan, call_plan = ctx.put_plan, ctx.call_plan
    put_entry = await ctx.strat.place(put_plan, limit_credit=fill_credit(put_plan))
    call_entry = await ctx.strat.place(call_plan, limit_credit=fill_credit(call_plan))
    condor = ex06.CondorPosition("live", put_plan, call_plan, put_entry, call_entry)
    deadline = datetime.now(MARKET_TZ) + timedelta(seconds=45)
    await ex06.monitor_condor(
        ctx.client, ctx.strat, condor, combined_take_profit_pct=0.99, poll_interval=1.0, deadline=deadline
    )
    check(
        "condor: both entries filled",
        put_entry.state == call_entry.state == OrderState.FILLED,
        f"put={put_entry.state.value} call={call_entry.state.value}",
    )
    ok1, d1 = await legs_flat(ctx, put_plan)
    ok2, d2 = await legs_flat(ctx, call_plan)
    check("condor: all four legs flat after monitor", ok1 and ok2, {**d1, **d2})
    check("condor: no streams left", ctx.strat._streams == {}, list(ctx.strat._streams))


async def s_journal(ctx):
    events = list(ctx.store.replay())
    kinds = {type(e).__name__ for e in events}
    check("journal replays cleanly", events, f"{len(events)} events, kinds={sorted(kinds)}")
    report_store = JsonStore(ctx.store.path)
    om3 = OrderManager(ctx.client, report_store)
    await ctx.om.stop()
    report = await om3.start()
    check("fresh start: no local_only", report.local_only == [], report.local_only)
    check("fresh start: nothing of ours still open at IB", not [u for u in report.matched], report.matched)
    ctx.om = om3


# ---------------------------------------------------------------------------


async def main():
    ctx = Ctx()
    ctx.client = IBKRClient(IBKRConfig.from_env(prefix="TWS_"))
    ctx.reconnects = []
    ctx.client.add_reconnect_listener(lambda: ctx.reconnects.append(time.monotonic()))
    await ctx.client.connect()
    ctx.client.ib.reqMarketDataType(1)
    path = Path(tempfile.mkdtemp(prefix="ibtws-live-")) / "orders.jsonl"
    ctx.store = JsonStore(path)
    ctx.om = OrderManager(ctx.client, ctx.store)
    await ctx.om.start()
    await ctx.om.refresh_positions()
    ctx.start = positions_by_conid(ctx.om)
    print("start positions:", ctx.start)
    [ctx.aapl] = await ctx.client.ib.qualifyContractsAsync(Stock("AAPL", "SMART", "USD"))
    [ctx.spx] = await ctx.client.ib.qualifyContractsAsync(Index("SPX", "CBOE", "USD"))
    [tk] = await ctx.client.ib.reqTickersAsync(ctx.aapl)
    ctx.px = tk.marketPrice()
    ctx.today = datetime.now(MARKET_TZ).strftime("%Y%m%d")
    ctx.fetcher = OptionChainFetcher(ctx.client)
    ctx.strat = CreditSpreadStrategy(ctx.client, ctx.om, fetcher=ctx.fetcher, exit_fill_timeout=8, exit_max_attempts=3)
    print("AAPL", ctx.px, "ET", datetime.now(MARKET_TZ).strftime("%H:%M"))

    try:
        await section("read-only analytics", s_read_only, ctx)
        await section("IB rejection", s_rejection, ctx)
        await section("stop order + two event streams", s_stop_and_streams, ctx)
        await section("bracket fill -> close_position", s_bracket_fill_then_close, ctx)
        await section("disconnect right after a market order", s_disconnect_mid_order, ctx)
        await section("restart with a working bracket", s_restart_with_bracket, ctx)
        ctx.strat._om = ctx.om
        await section("build 0DTE plans", s_build_plans, ctx)
        if getattr(ctx, "put_plan", None):
            await section("quote streams survive reconnect", s_streams_survive_reconnect, ctx)
            ctx.strat._om = ctx.om
            await section("strategy: entry never fills", s_entry_never_fills, ctx)
            await section("strategy: take-profit exit", s_take_profit, ctx)
            await section("strategy: stop-loss exit", s_stop_loss, ctx)
            await section("strategy: chase + escalate", s_chase_and_escalate, ctx)
            await section("example 06 condor monitor", s_condor_example, ctx)
        await section("journal + fresh restart", s_journal, ctx)
    finally:
        print("\n=== cleanup", flush=True)
        try:
            await ctx.om.cancel_all()
            await asyncio.sleep(2)
            await ctx.om.refresh_positions()
            now = positions_by_conid(ctx.om)
            for cid in touched:
                diff = now.get(cid, 0) - ctx.start.get(cid, 0)
                if abs(diff) < 1e-9:
                    continue
                contract = ctx.om._position_contracts.get(cid)
                if contract is not None and not contract.exchange:
                    [contract] = await ctx.client.ib.qualifyContractsAsync(contract)
                print(f"  restoring conId={cid} by {-diff}")
                o = await ctx.om.market(contract, OrderSide.SELL if diff > 0 else OrderSide.BUY, abs(diff))
                await ctx.om.wait_for(lambda: o.state in TERMINAL, timeout=30)
            await asyncio.sleep(2)
            await ctx.om.refresh_positions()
            end = positions_by_conid(ctx.om)
            same = {c: q for c, q in end.items() if q} == {c: q for c, q in ctx.start.items() if q}
            check("account positions identical to start", same, {c: q for c, q in end.items() if q})
            await ctx.om.stop()
        finally:
            ctx.store.close()
            await ctx.client.disconnect()
        failed = [r for r in results if not r[1]]
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
        for name, _, detail in failed:
            print(f"  FAILED: {name}  {detail}")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
