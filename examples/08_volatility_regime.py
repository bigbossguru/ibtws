"""08 — Intraday volatility regime for 0DTE SPX credit spreads.

Runs the version-5 assessment of ``analysis/VOLATILITY_REGIME_CONCEPT.md``
end-to-end against TWS, using only :class:`IBKRClient`:

1. ``get_historical_data`` — 15-minute VIX1D / VIX / SPX bars over ``1 Y``, from
   which the time-of-day bucket histories are built. Every metric is a percentile
   rank inside its own 15-minute bucket, because VIX1D is a time-weighted blend
   of today's and tomorrow's SPX expiries whose median climbs from 9.8 at 09:30
   ET to 12.8 at 15:30 ET with no change in risk (§2). A pooled history would
   label every morning calm and every afternoon stressed.
2. ``OptionChainFetcher`` + ``GexCalculator`` — the 0DTE chain reduced to a Zero
   Gamma Level. Optional: GEX is the one metric with no historical validation
   (§4.7), so without it the assessment simply records a degradation.
3. ``detect_volatility_regime`` — the gate.

Position sizing is deliberately absent. The gate answers "is the regime
acceptable"; how much to trade is a risk-layer decision, and §15.8 measures it as
the only component that changes the probability of ruin — the gate itself moves
expectancy by fractions of a percent. A caller wiring this into a strategy should
carry over two measured facts: the strike width belongs in expected-move units
rather than points (§15.2), and ``result.expected_move_pct`` is the scale for it.

What changed against the version-2 example
------------------------------------------
That example read *daily* bars and evaluated once at the open. This one reads
15-minute bars and can evaluate at any point of RTH after the opening window,
which is what the concept requires: the entry is not tied to the open, and the
first 45 minutes are skipped outright as the worst part of the session on
risk-normalised returns (§15.3).

Two things to keep in mind about the data:

* **Depth of history matters.** Each bucket needs 60 prior observations, so the
  ``1 Y`` request has to come back reasonably complete. IBKR serves 15-minute
  bars for about 251 sessions and 5-minute bars for only ~90, which is why the
  granularity is 15 minutes and not finer. A short series is reported as a data
  gap rather than computed on a smaller window.
* **Today is excluded from every window.** Including the session being assessed
  caps the attainable rank at 59/60 = 98.3, so a p98 threshold would fire about
  half as often as calibrated (§4).

The macro calendar (FOMC / CPI / NFP) has priority over this assessment and is a
separate module by design (§10); it is deliberately not implemented here.
"""

from __future__ import annotations

import asyncio
import logging
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ib_async import Contract, Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.analysis.gex import GexCalculator
from ibtws.unofficial.analysis.volatility_regime import (
    DEFAULT_CONFIG,
    FULL_SESSION_BUCKETS,
    SESSION_MINUTES,
    TRADING_DAYS_PER_YEAR,
    VolatilityRegimeResult,
    bucket_of,
    detect_volatility_regime,
    expected_move_pct,
)
from ibtws.unofficial.client import BarSize, Duration, IBKRClient
from ibtws.unofficial.option import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

logger = logging.getLogger(__name__)

# Trading day is an Eastern-time notion, not the host's local one.
MARKET_TZ = ZoneInfo("America/New_York")

# 15-minute bars over a year: the deepest intraday window IBKR serves, and the
# only one that covers 60 observations across 26 buckets.
HISTORY_DURATION: Duration = "1 Y"
BAR_SIZE: BarSize = "15 mins"

# ~126 trading days of daily closes: comfortably covers RV20.
DAILY_DURATION: Duration = "6 M"

# Strike window for the ZGL sweep. Wide enough to bracket the flip point,
# narrow enough to stay inside snapshot pacing limits.
GEX_STRIKE_WINDOW_PCT = 0.03

# The VIX1D roll completes at 16:15 ET, so that is the horizon the remaining
# expected move refers to (§3).
ROLL_HORIZON_MINUTES = 16 * 60 + 15
RTH_OPEN_MINUTES = 9 * 60 + 30


def _today_expiration() -> str:
    """Today's date in IBKR ``YYYYMMDD`` form — the 0DTE expiration."""
    return pd.Timestamp.now(tz=MARKET_TZ).strftime("%Y%m%d")


def _bucketed(bars: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Reshape intraday bars into a ``(date, bucket)`` frame.

    ``date`` and ``bucket`` are the two axes every rank is taken along: the
    bucket selects the comparison population, the date orders it.
    """
    frame = bars.copy()
    stamps = pd.to_datetime(frame["date"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize("UTC")
    stamps = stamps.dt.tz_convert(MARKET_TZ)
    frame["session"] = stamps.dt.normalize()
    frame["bucket"] = stamps.map(bucket_of)
    keep = ["session", "bucket", column]
    if column != "close" and "close" in frame.columns:
        keep.append("close")
    return frame[[c for c in keep if c in frame.columns]]


def _prior_bucket_series(frame: pd.DataFrame, bucket: str, today: pd.Timestamp, column: str) -> pd.Series | None:
    """Readings of *bucket* on sessions strictly before *today*, oldest first."""
    if frame is None or frame.empty:
        return None
    same = frame[(frame["bucket"] == bucket) & (frame["session"] < today)]
    if same.empty:
        return None
    return same.sort_values("session")[column].astype("float64").reset_index(drop=True)


def _session_realized_range_pct(spx: pd.DataFrame, today: pd.Timestamp, bucket: str) -> float | None:
    """``(high - low) / open × 100`` from today's open up to and including *bucket*."""
    if spx is None or spx.empty:
        return None
    stamps = pd.to_datetime(spx["date"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize("UTC")
    stamps = stamps.dt.tz_convert(MARKET_TZ)
    rows = spx[(stamps.dt.normalize() == today) & (stamps.map(bucket_of) <= bucket)]
    if rows.empty:
        return None
    day_open = float(rows.iloc[0]["open"])
    if day_open <= 0:
        return None
    return (float(rows["high"].max()) - float(rows["low"].min())) / day_open * 100.0


def _realized_vs_em_history(
    spx: pd.DataFrame,
    vix1d: pd.DataFrame,
    bucket: str,
    today: pd.Timestamp,
) -> pd.Series | None:
    """History of ``realised range so far / EM remaining`` in *bucket*.

    Reconstructed from bars rather than logged, which is possible only because
    both inputs are index series — the reason the gating metric works on day one
    instead of after a 60-session warm-up.
    """
    if spx is None or spx.empty or vix1d is None or vix1d.empty:
        return None

    stamps = pd.to_datetime(spx["date"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize("UTC")
    stamps = stamps.dt.tz_convert(MARKET_TZ)
    frame = spx.copy()
    frame["session"] = stamps.dt.normalize()
    frame["bucket"] = stamps.map(bucket_of)
    frame = frame[(frame["session"] < today) & (frame["bucket"] <= bucket)]
    if frame.empty:
        return None

    grouped = frame.sort_values(["session", "bucket"]).groupby("session")
    ranges = (grouped["high"].max() - grouped["low"].min()) / grouped["open"].first() * 100.0

    levels = _bucketed(vix1d)
    at_bucket = levels[(levels["bucket"] == bucket) & (levels["session"] < today)]
    if at_bucket.empty:
        return None
    minutes_left = _minutes_left_for(bucket)
    values: list[float] = []
    for session, level in at_bucket.set_index("session")["close"].items():
        realised = ranges.get(session)
        em = expected_move_pct(float(level), minutes_left, bucket, use_profile=DEFAULT_CONFIG.use_variance_profile)
        if realised is None or not em:
            continue
        values.append(float(realised) / em)
    return pd.Series(values, dtype="float64") if values else None


def _session_key(value) -> pd.Timestamp:
    """Normalise a timestamp to a tz-naive midnight, so both sides of a join match.

    Bars come back tz-aware from IBKR and the ``_bucketed`` frames keep Eastern
    time, while a daily RV series indexed off ``date`` may or may not. Comparing
    the two directly silently produces empty joins — the premium-spread history
    came back as ``None`` for exactly that reason.
    """
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _rv_by_session(spx_daily: pd.DataFrame, window: int) -> pd.Series | None:
    """Annualised close-to-close RV over the *window* sessions ENDING BEFORE each date.

    The value stored under session D uses closes up to D-1 only, which is what a
    live run can know intraday. Mapping RV that includes close[D] onto session D
    was one of the three leaks fixed in version 4 (§8), hence the trailing
    ``shift(1)``.
    """
    if spx_daily is None or spx_daily.empty:
        return None
    daily = spx_daily.copy()
    daily["session"] = daily["date"].map(_session_key)
    closes = daily.sort_values("session").set_index("session")["close"].astype("float64")
    if len(closes) < window + 2:
        return None
    log_returns = np.log(closes).diff()
    rv = log_returns.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    return rv.shift(1)


def _premium_spread_history(
    vix1d: pd.DataFrame,
    spx_daily: pd.DataFrame,
    bucket: str,
    today: pd.Timestamp,
    window: int,
) -> pd.Series | None:
    """History of ``VIX1D - RV20`` in *bucket*, RV from completed sessions only."""
    levels = _bucketed(vix1d)
    at_bucket = levels[(levels["bucket"] == bucket) & (levels["session"] < today)]
    if at_bucket.empty:
        return None
    rv = _rv_by_session(spx_daily, window)
    if rv is None:
        return None

    values: list[float] = []
    for session, level in at_bucket.sort_values("session").set_index("session")["close"].items():
        realised = rv.get(_session_key(session))
        if realised is None or pd.isna(realised):
            continue
        values.append(float(level) - float(realised))
    return pd.Series(values, dtype="float64") if values else None


def _minutes_left_for(bucket: str) -> float:
    """Business minutes from *bucket* to the 16:15 ET roll horizon."""
    hour, minute = (int(part) for part in bucket.split(":"))
    return float(max(15, ROLL_HORIZON_MINUTES - (hour * 60 + minute)))


async def resolve_zero_gamma_level(
    client: IBKRClient,
    underlying: Contract,
    expiration: str,
) -> tuple[float | None, float | None]:
    """Return ``(spot, zero_gamma_level)`` for *expiration*, either may be ``None``.

    GEX is the one metric of the concept with no historical validation (§4.7) —
    it is included on theory alone and is opt-in: without a ZGL the assessment
    records a degradation instead of blocking, which is the fix for version 2's
    behaviour of refusing every entry when the chain was unavailable.
    """
    quotes = await OptionChainFetcher(client).fetch_snapshot(
        underlying,
        expirations=[expiration],
        trading_class="SPXW",
        rights=("C", "P"),
        strike_window_pct=GEX_STRIKE_WINDOW_PCT,
        as_dataframe=True,
    )
    if not isinstance(quotes, pd.DataFrame) or quotes.empty:
        logger.warning(f"Empty {underlying.symbol} chain snapshot for {expiration}; ZGL unavailable.")
        return None, None

    missing = GexCalculator.REQUIRED_COLUMNS - set(quotes.columns)
    if missing:
        logger.warning(f"Chain snapshot missing columns for GEX: {sorted(missing)}")
        return None, None

    try:
        gex = GexCalculator()
        gex.compute(quotes)
    except Exception as exc:  # noqa: BLE001 - a failed sweep is a data gap, not a crash
        logger.warning(f"GEX computation failed: {exc}")
        return None, None

    logger.info(
        f"GEX: spot={gex.spot:.2f} ZGL={gex.zero_gamma_level} regime={gex.regime} total_gex={gex.total_gex:,.0f}"
    )
    return gex.spot, gex.zero_gamma_level


def print_report(result: VolatilityRegimeResult) -> None:
    verdict = "FAVORABLE" if result.favorable else "SKIP"

    print("\n=== Intraday volatility regime — 0DTE SPX ===")
    print(f"  verdict        : {verdict}")
    if result.reason:
        print(f"  reason         : {result.reason}")
    if result.bucket:
        print(f"  bucket         : {result.bucket} ET   {result.minutes_left:.0f} min to the roll horizon")
    if result.variance_share_left is not None:
        print(f"  variance left  : {result.variance_share_left:.2f} of the session (measured profile, not √t)")
    if result.roll_weight_today is not None:
        degraded = "  [DEGRADED]" if "vix1d_roll" in result.degraded else ""
        print(f"  VIX1D roll     : {result.roll_weight_today:.2f} weight on today's expiry{degraded}")
    if result.base_rank is not None:
        print(f"  base level     : rank {result.base_rank:5.1f} → {result.base_regime}   (reports, does not gate)")
    if result.expected_move_pct is not None:
        print(f"  expected move  : {result.expected_move_pct:.2f}% of spot over the rest of the session")
    print(f"  flags          : {result.hard_count} hard / {result.soft_count} soft")
    print(f"  allowance      : hard == 0 and soft <= {DEFAULT_CONFIG.max_soft_flags}, gating on realized_vs_em")

    if result.flags:
        print("\n  recorded:")
        for flag in result.flags:
            code = "missing_data" if flag.missing else "observation" if flag.severity == "info" else "risk_flag"
            print(f"    [{flag.severity:4}] {code:12} {flag.metric:17} {flag.detail}")

    print("\n  metrics:")
    for name, value in result.metrics.items():
        print(f"    {name:22} {'n/a' if value is None else f'{value:10.4f}'}")
    if result.degraded:
        print(f"\n  degraded       : {', '.join(result.degraded)}")
    if result.zgl_source:
        print(f"  ZGL source     : {result.zgl_source}")
    print()


async def load_bars(
    client: IBKRClient,
    contract: Contract,
    duration: Duration,
    bar_size: BarSize,
) -> pd.DataFrame | None:
    """Load bars for one contract; a failure degrades the assessment, never crashes it."""
    try:
        bars = await client.get_historical_data(
            contract,
            duration=duration,
            bar_size=bar_size,
            use_rth=True,
        )
    except Exception as exc:  # noqa: BLE001 - a missing index becomes a data gap downstream
        logger.warning(f"Historical bars unavailable for {contract.symbol} ({bar_size}): {exc}")
        return None
    if bars is None or bars.empty:
        logger.warning(f"Historical bars empty for {contract.symbol} ({bar_size}).")
        return None
    return bars


async def main() -> None:
    # Read-only: this example places no orders, so remove the possibility.
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=15, readonly=True)

    async with IBKRClient(config) as client:
        client.ib.reqMarketDataType(1)

        contracts: dict[str, Contract] = {
            "VIX": Index("VIX", "CBOE", "USD"),
            "VIX1D": Index("VIX1D", "CBOE", "USD"),
            "SPX": Index("SPX", "CBOE", "USD"),
        }
        intraday: dict[str, pd.DataFrame | None] = {}
        for symbol, underlying in contracts.items():
            qualified = await client.ib.qualifyContractsAsync(underlying)
            resolved = qualified[0] if qualified else None
            if not isinstance(resolved, Contract):
                logger.warning(f"Failed to qualify {symbol}; its metrics become data gaps.")
                intraday[symbol] = None
                continue
            contracts[symbol] = resolved
            intraday[symbol] = await load_bars(client, resolved, HISTORY_DURATION, BAR_SIZE)

        spx_daily = await load_bars(client, contracts["SPX"], DAILY_DURATION, "1 day")

        now = pd.Timestamp.now(tz=MARKET_TZ)
        today = pd.Timestamp(now.date()).tz_localize(MARKET_TZ)
        bucket = bucket_of(now)
        minutes_left = float(max(15, ROLL_HORIZON_MINUTES - (now.hour * 60 + now.minute)))
        minutes_since_open = float(now.hour * 60 + now.minute - RTH_OPEN_MINUTES)

        vix1d_bars, vix_bars, spx_bars = intraday["VIX1D"], intraday["VIX"], intraday["SPX"]
        vix1d_frame = _bucketed(vix1d_bars) if vix1d_bars is not None else None
        vix_frame = _bucketed(vix_bars) if vix_bars is not None else None

        # Latest reading of each index in the current bucket.
        def _latest(frame: pd.DataFrame | None) -> float | None:
            if frame is None or frame.empty:
                return None
            rows = frame[(frame["session"] == today) & (frame["bucket"] == bucket)]
            return float(rows.iloc[-1]["close"]) if not rows.empty else None

        vix1d_now, vix_now = _latest(vix1d_frame), _latest(vix_frame)

        term_history = None
        if vix1d_frame is not None and vix_frame is not None:
            merged = vix1d_frame.merge(vix_frame, on=["session", "bucket"], suffixes=("_v1d", "_vix"))
            merged["ratio"] = merged["close_v1d"] / merged["close_vix"].replace(0.0, pd.NA)
            term_history = _prior_bucket_series(merged, bucket, today, "ratio")

        session_buckets = None
        if spx_bars is not None:
            stamps = pd.to_datetime(spx_bars["date"])
            if stamps.dt.tz is None:
                stamps = stamps.dt.tz_localize("UTC")
            session_buckets = int((stamps.dt.tz_convert(MARKET_TZ).dt.normalize() == today).sum())

        spot, zgl = await resolve_zero_gamma_level(client, contracts["SPX"], _today_expiration())

        result = detect_volatility_regime(
            bucket=bucket,
            minutes_left=minutes_left,
            minutes_since_open=minutes_since_open,
            vix1d=vix1d_now,
            vix1d_bucket_history=(
                _prior_bucket_series(vix1d_frame, bucket, today, "close") if vix1d_frame is not None else None
            ),
            vix=vix_now,
            term_structure_history=term_history,
            realized_range_pct=(_session_realized_range_pct(spx_bars, today, bucket) if spx_bars is not None else None),
            realized_vs_em_history=_realized_vs_em_history(spx_bars, vix1d_bars, bucket, today),
            spx_closes=(spx_daily["close"] if spx_daily is not None else None),
            premium_spread_history=(
                _premium_spread_history(vix1d_bars, spx_daily, bucket, today, DEFAULT_CONFIG.rv_window)
                if vix1d_bars is not None and spx_daily is not None
                else None
            ),
            spx_price=spot,
            zero_gamma_level=zgl,
            zgl_source="GexCalculator/ibtws (Black-Scholes sweep, Brent root-find)",
            session_buckets=session_buckets if session_buckets else FULL_SESSION_BUCKETS,
        )

        print_report(result)
        logger.info(f"Session minutes: {SESSION_MINUTES}, bucket {bucket}, {minutes_left:.0f} min left.")


if __name__ == "__main__":
    asyncio.run(main())
