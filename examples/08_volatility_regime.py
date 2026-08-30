"""08 — Pre-market volatility-regime gate for 0DTE SPX credit spreads.

Runs the tail-regime cut-off of ``analysis/VOLATILITY_REGIME_CONCEPT.md``
end-to-end against TWS, using only :class:`IBKRClient`:

1. ``get_historical_data`` — daily VIX1D / VIX / SPX bars for the rolling
   windows (60-session percentile base, RV20, overnight ROC).
2. ``get_market_data`` — today's live index opens, which the daily bars do not
   yet carry in the first minutes of the session.
3. ``OptionChainFetcher`` + ``GexCalculator`` — the 0DTE chain reduced to a Zero
   Gamma Level.
4. ``detect_volatility_regime`` — the gate itself.

The connection is read-only (``IBKRConfig(readonly=True)``): this example never
submits an order, so the interlock costs nothing and removes the possibility.

``Index("VIX1D", "CBOE", "USD")`` resolves on TWS, so the primary path needs no
external data source. Two things to keep in mind about the values it returns:

* **Depth of history matters.** The base level is a 60-session percentile and
  RV20 needs 21 completed sessions, so the ``6 M`` request has to come back
  reasonably complete. A short series is reported as a data gap (hard flag)
  rather than computed on a smaller window — if that happens, check the index
  subscription before widening the duration. Cboe's ``VIX1D_History.csv``
  remains the source of record per §6 and is the natural cross-check.
* **Pre-open index values are noisy.** Cboe opens SPX/VIX option order
  acceptance at 07:30 ET, so an 08:30 ET reading exists but sits on wide
  spreads. The concept flags comparing 08:30 against post-09:30 values as
  unverified (§6).

The macro calendar (FOMC / CPI / NFP) has priority over this gate and is a
separate module by design (§5); ``macro_calendar_clear`` below is an explicit
placeholder, not an implementation.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
from ib_async import Contract, Index

from ibtws.config import IBKRConfig
from ibtws.unofficial.analysis.gex import GexCalculator
from ibtws.unofficial.analysis.volatility_regime import (
    DEFAULT_CONFIG,
    VolatilityRegimeResult,
    detect_volatility_regime,
)
from ibtws.unofficial.client import Duration, IBKRClient
from ibtws.unofficial.helpers import safe_pick_value
from ibtws.unofficial.option import OptionChainFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

logger = logging.getLogger(__name__)

# Trading day is an Eastern-time notion, not the host's local one.
MARKET_TZ = ZoneInfo("America/New_York")

# ~126 trading days: comfortably covers the 60-session base window and RV20.
HISTORY_DURATION: Duration = "6 M"

# Strike window for the ZGL sweep. Wide enough to bracket the flip point,
# narrow enough to stay inside snapshot pacing limits.
GEX_STRIKE_WINDOW_PCT = 0.03


@dataclass(frozen=True)
class IndexSeries:
    """Daily history of one index, split at today's session boundary."""

    symbol: str
    today_open: float | None
    prev_close: float | None
    opens: pd.Series  # completed sessions only, oldest first
    closes: pd.Series

    def describe(self) -> str:
        open_txt = f"{self.today_open:.2f}" if self.today_open is not None else "n/a"
        prev_txt = f"{self.prev_close:.2f}" if self.prev_close is not None else "n/a"
        return f"{self.symbol}: open={open_txt} prev_close={prev_txt} history={len(self.closes)} sessions"


def _today_et() -> _dt.date:
    return _dt.datetime.now(tz=MARKET_TZ).date()


def _bar_dates(df: pd.DataFrame) -> pd.Series:
    """Normalise the ``date`` column of ``util.df`` output to ``datetime.date``."""
    return pd.to_datetime(df["date"]).dt.date


async def _live_open(client: IBKRClient, contract: Contract) -> float | None:
    """Today's index open from a live/frozen ticker.

    Daily bars do not carry today's session until it closes, so the open that
    every metric of §2 is defined on has to come from a snapshot. Falls back
    through ``open`` → ``last`` → ``close``: for an index the first is what we
    want, the others keep the example usable on a frozen feed.
    """
    try:
        ticker = await client.get_market_data(contract)
    except Exception as exc:  # noqa: BLE001 - absent subscription must degrade, not abort
        logger.warning(f"No live snapshot for {contract.symbol}: {exc}")
        return None

    for attr in ("open", "last", "close"):
        value = safe_pick_value(ticker, attr)
        if value is not None:
            if attr != "open":
                logger.warning(f"{contract.symbol}: session open unavailable, using '{attr}' ({value:.2f}) instead.")
            return value

    logger.warning(f"{contract.symbol}: ticker carried no usable price.")
    return None


async def load_index_series(
    client: IBKRClient,
    contract: Contract,
    *,
    duration: Duration = HISTORY_DURATION,
) -> IndexSeries | None:
    """Load daily bars + today's open for one index. ``None`` when unavailable.

    History is deliberately cut to *completed* sessions: the detector expects
    today excluded from the percentile window and from RV20, and an in-progress
    bar would contaminate both.
    """
    symbol = contract.symbol
    try:
        bars = await client.get_historical_data(
            contract,
            duration=duration,
            bar_size="1 day",
            use_rth=True,
            what_to_show="TRADES",
        )
    except Exception as exc:  # noqa: BLE001 - a missing index must degrade the gate, not crash it
        logger.warning(f"Historical bars unavailable for {symbol}: {exc}")
        return None

    if bars is None or bars.empty:
        logger.warning(f"Historical bars empty for {symbol}.")
        return None

    completed = bars[_bar_dates(bars) < _today_et()]
    if completed.empty:
        logger.warning(f"No completed sessions in the {symbol} history.")
        return None

    today_open = await _live_open(client, contract)
    if today_open is None:
        # Late in the day IBKR may already report today's bar; its open is the
        # same number the snapshot would have given.
        todays_bar = bars[_bar_dates(bars) == _today_et()]
        if not todays_bar.empty:
            today_open = safe_pick_value(todays_bar.iloc[-1], "open")

    series = IndexSeries(
        symbol=symbol,
        today_open=today_open,
        prev_close=float(completed["close"].iloc[-1]),
        opens=completed["open"].astype("float64").reset_index(drop=True),
        closes=completed["close"].astype("float64").reset_index(drop=True),
    )
    logger.info(series.describe())
    return series


async def resolve_zero_gamma_level(
    client: IBKRClient,
    underlying: Contract,
    expiration: str,
) -> tuple[float | None, float | None]:
    """Return ``(spot, zero_gamma_level)`` for *expiration*, either may be ``None``.

    GEX is the one metric of the concept with no historical validation (§7.1) —
    it is included on theory alone and should be treated as a hypothesis under
    observation, not as a proven filter.
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
        gex.plot("./gex.png")
    except Exception as exc:  # noqa: BLE001 - a failed sweep is a data gap for the gate
        logger.warning(f"GEX computation failed: {exc}")
        return None, None

    logger.info(
        f"GEX: spot={gex.spot:.2f} ZGL={gex.zero_gamma_level} regime={gex.regime} total_gex={gex.total_gex:,.0f}"
    )
    return gex.spot, gex.zero_gamma_level


def resolve_0dte_expiration(expirations: tuple[str, ...]) -> str | None:
    """Today's expiry, or ``None`` when SPXW has no 0DTE listed today."""
    today = _today_et().strftime("%Y%m%d")
    return today if today in expirations else "20260831"


def macro_calendar_clear(day: _dt.date) -> bool:
    """Placeholder for the macro-calendar module (§5).

    A real implementation is a hard skip on FOMC / CPI / NFP mornings and takes
    priority over the detector. Returning ``True`` unconditionally means this
    example performs **no** macro check — do not read the verdict below as a
    complete entry decision.
    """
    logger.warning(f"Macro calendar not checked for {day}: wire this to your event source before trading.")
    return True


def print_report(result: VolatilityRegimeResult) -> None:
    print("\n=== Volatility regime — 0DTE SPX ===")
    print(f"  verdict        : {'FAVORABLE' if result.favorable else 'SKIP'}")
    if result.reason:
        print(f"  reason         : {result.reason}")
    if result.base_rank is not None:
        degraded = "  (degraded: VIX percentile fallback)" if result.degraded_base else ""
        print(f"  base level     : rank {result.base_rank:5.1f} → {result.base_regime}{degraded}")
    print(f"  flags          : {result.hard_count} hard / {result.soft_count} soft")
    print(f"  allowance      : hard == 0 and soft <= {DEFAULT_CONFIG.max_soft_flags}")

    if result.flags:
        print("\n  triggered:")
        for flag in result.flags:
            code = "missing_data" if flag.missing else "risk_flag"
            print(f"    [{flag.severity:4}] {code:12} {flag.metric:17} {flag.detail}")

    print("\n  metrics:")
    for name, value in result.metrics.items():
        print(f"    {name:20} {'n/a' if value is None else f'{value:10.4f}'}")
    if result.zgl_source:
        print(f"\n  ZGL source     : {result.zgl_source}")
    print()


async def qualify_index(client: IBKRClient, symbol: str) -> Contract | None:
    """Qualify one CBOE index, or ``None`` when TWS does not carry it.

    A missing index must degrade the gate (fail-safe hard flags), not abort it —
    only SPX is indispensable, and that is enforced by the caller.
    """
    try:
        contracts = await client.ib.qualifyContractsAsync(Index(symbol, "CBOE", "USD"))
    except Exception as exc:  # noqa: BLE001 - an absent index is a data gap, not a crash
        logger.warning(f"Cannot qualify {symbol}: {exc}")
        return None
    if not contracts:
        logger.warning(f"Cannot qualify {symbol}: TWS returned no contract.")
        return None
    return contracts[0]


async def main() -> None:
    # Read-only: this example places no orders, so remove the possibility.
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=15, readonly=True)

    async with IBKRClient(config) as client:
        # 1 = live, 2 = frozen, 3 = delayed, 4 = delayed-frozen.
        # Frozen keeps the example usable when the live feed is held by another
        # session; for a real gate use live data — a stale open is a wrong open.
        client.ib.reqMarketDataType(2)

        spx_contract, vix_contract, vix1d_contract = await asyncio.gather(
            qualify_index(client, "SPX"),
            qualify_index(client, "VIX"),
            qualify_index(client, "VIX1D"),
        )

        if spx_contract is None:
            raise LookupError("Cannot qualify SPX — there is nothing to evaluate.")

        if vix1d_contract is None:
            logger.warning(
                "VIX1D could not be qualified on this connection, though the index does exist at "
                "IBKR — check the market-data subscription. The gate now falls back to the VIX "
                "percentile (concept §2.1) and every VIX1D-derived metric becomes a missing-data "
                "hard flag, so expect SKIP."
            )

        # Index series first: they are cheap, and a missing base level makes the
        # far more expensive chain snapshot pointless.
        vix1d_series = await load_index_series(client, vix1d_contract) if vix1d_contract else None
        vix_series = await load_index_series(client, vix_contract) if vix_contract else None
        spx_series = await load_index_series(client, spx_contract)

        chain_def = await OptionChainFetcher(client).fetch_chain_definition(spx_contract, trading_class="SPXW")
        expiration = resolve_0dte_expiration(chain_def.expirations)

        if expiration is None:
            logger.warning(f"No 0DTE SPXW expiry listed for {_today_et()} — ZGL will be reported as missing.")
            spot, zgl = None, None
        else:
            spot, zgl = await resolve_zero_gamma_level(client, spx_contract, expiration)

        result = detect_volatility_regime(
            vix1d_open=vix1d_series.today_open if vix1d_series else None,
            vix1d_history=vix1d_series.opens if vix1d_series else None,
            vix_open=vix_series.today_open if vix_series else None,
            vix_prev_close=vix_series.prev_close if vix_series else None,
            spx_closes=spx_series.closes if spx_series else None,
            spx_price=spot,
            zero_gamma_level=zgl,
            vix_history=vix_series.opens if vix_series else None,
            zgl_source="GexCalculator/ibtws (Black-Scholes sweep, Brent root-find)",
        )

        print_report(result)

        # The macro calendar overrides the detector, so it is checked last and
        # separately: two independent decisions, neither aware of the other.
        if result.favorable and macro_calendar_clear(_today_et()):
            logger.info("Gate passed — proceed to strike selection and credit checks.")
        else:
            logger.info("Gate blocked — no entry today.")


if __name__ == "__main__":
    asyncio.run(main())
