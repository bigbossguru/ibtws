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
import logging
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


async def main() -> None:
    # Read-only: this example places no orders, so remove the possibility.
    config = IBKRConfig(host="192.168.0.129", port=7497, client_id=15, readonly=True)

    async with IBKRClient(config) as client:
        client.ib.reqMarketDataType(2)

        contracts = {
            "VIX": Index("VIX", "CBOE", "USD"),
            "VIX1D": Index("VIX1D", "CBOE", "USD"),
            "SPX": Index("SPX", "CBOE", "USD"),
        }

        market_data = {}
        historical_data = {}
        for symbol, underlying in contracts.items():
            [underlying] = await client.ib.qualifyContractsAsync(underlying)
            contracts[symbol] = underlying
            market_data[symbol] = await client.get_market_data(underlying)
            historical_data[symbol] = await client.get_historical_data(
                underlying,
                duration="60 D",
                bar_size="1 day",
                use_rth=False,
            )

        spot, zgl = await resolve_zero_gamma_level(client, contracts["SPX"], "20260831")

        result = detect_volatility_regime(
            vix1d_open=historical_data["VIX1D"].iloc[-1].open,
            vix1d_history=historical_data["VIX1D"]["close"],
            vix_open=historical_data["VIX"].iloc[-1].open,
            vix_prev_close=historical_data["VIX"].iloc[-2].close,
            spx_closes=historical_data["SPX"]["close"],
            spx_price=historical_data["SPX"].iloc[-1]["close"],
            zero_gamma_level=zgl,
            vix_history=historical_data["VIX"].open,
            zgl_source="GexCalculator/ibtws (Black-Scholes sweep, Brent root-find)",
        )

        print_report(result)


if __name__ == "__main__":
    asyncio.run(main())
