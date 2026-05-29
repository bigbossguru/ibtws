"""Gamma Exposure (GEX) analysis and calculation.

Computes dealer gamma exposure across the option chain to identify key
support/resistance levels and hedging flows:

- **Total GEX** — aggregate dealer gamma exposure (calls positive, puts negative).
- **Call Wall** — strike with the highest call gamma exposure (resistance).
- **Put Wall** — strike with the highest put gamma exposure (support).
- **Net GEX** — total call GEX minus total put GEX (directional bias).
- **Zero Gamma Level** — interpolated price where net GEX flips sign.
- **GEX by Strike** — histogram data for visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from ib_async import Contract

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.option.chains import OptionChainFetcher
from ibtws.unofficial.option.models import OptionQuote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrikeGEX:
    """GEX breakdown for a single strike."""

    strike: float
    call_gex: float
    put_gex: float
    net_gex: float


@dataclass(frozen=True)
class GEXResult:
    """Aggregated GEX metrics for one underlying + expiration set."""

    underlying_symbol: str
    spot: float

    total_gex: float
    call_gex_total: float
    put_gex_total: float
    net_gex: float

    call_wall: float | None  # strike with max call GEX
    put_wall: float | None  # strike with max |put GEX|
    zero_gamma_level: float | None  # interpolated flip point

    strikes: list[StrikeGEX] = field(default_factory=list)


class GEXCalculator:
    """Compute Gamma Exposure from live option chain data."""

    def __init__(self, client: IBKRClient, chain_fetcher: OptionChainFetcher) -> None:
        self._client = client
        self._chain = chain_fetcher

    async def calculate(
        self,
        underlying: Contract,
        *,
        expirations: list[str] | None = None,
        expiry_from: str | None = None,
        expiry_to: str | None = None,
        strike_window_pct: float = 0.15,
        use_open_interest: bool = True,
        trading_class: str | None = None,
    ) -> GEXResult:
        """Calculate GEX for *underlying*.

        Parameters
        ----------
        underlying:
            Underlying contract (qualified or will be qualified).
        expirations:
            Specific expirations (YYYYMMDD). If None, uses all within range.
        expiry_from / expiry_to:
            Filter expirations to this date range.
        strike_window_pct:
            Fraction of spot to include strikes (e.g. 0.15 = ±15%).
        use_open_interest:
            Weight gamma by open interest (True) or volume (False).
        trading_class:
            Option trading class (e.g. "SPXW" for SPX weeklies).
        """
        quotes: list[OptionQuote] = await self._chain.fetch_snapshot(
            underlying,
            expirations=expirations,
            expiry_from=expiry_from,
            expiry_to=expiry_to,
            strike_window_pct=strike_window_pct,
            trading_class=trading_class,
            rights=("C", "P"),
        )

        spot = quotes[0].underlying_price if quotes else None
        if not spot or spot <= 0:
            raise ValueError(f"Cannot determine spot price for {underlying.symbol}")

        strike_map = _build_strike_gex(quotes, spot, use_open_interest)
        strikes_sorted = sorted(strike_map.values(), key=lambda s: s.strike)

        call_gex_total = sum(s.call_gex for s in strikes_sorted)
        put_gex_total = sum(s.put_gex for s in strikes_sorted)
        total_gex = call_gex_total + put_gex_total
        net_gex = call_gex_total - abs(put_gex_total)

        call_wall = max(strikes_sorted, key=lambda s: s.call_gex).strike if strikes_sorted else None
        put_wall = max(strikes_sorted, key=lambda s: abs(s.put_gex)).strike if strikes_sorted else None
        zero_gamma = _find_zero_gamma(strikes_sorted)

        return GEXResult(
            underlying_symbol=underlying.symbol,
            spot=spot,
            total_gex=total_gex,
            call_gex_total=call_gex_total,
            put_gex_total=put_gex_total,
            net_gex=net_gex,
            call_wall=call_wall,
            put_wall=put_wall,
            zero_gamma_level=zero_gamma,
            strikes=strikes_sorted,
        )


def _compute_contract_gex(
    quote: OptionQuote,
    spot: float,
    use_open_interest: bool,
) -> float:
    """GEX for one contract = gamma × OI × spot² × 0.01 × multiplier.

    Dealer is short calls (positive gamma → positive GEX) and long puts
    (negative gamma → negative GEX from dealer perspective).
    """
    gamma = quote.gamma
    if gamma is None or gamma <= 0:
        return 0.0

    weight = (quote.open_interest or 0.0) if use_open_interest else (quote.volume or 0.0)
    if weight <= 0:
        return 0.0

    multiplier = float(quote.contract.multiplier or 100)
    # GEX = gamma × OI × contract_multiplier × spot² × 0.01
    gex = gamma * weight * multiplier * spot * spot * 0.01

    # Dealer is short calls → positive GEX; dealer is short puts → negative GEX
    if quote.contract.right == "P":
        gex = -gex

    return gex


def _build_strike_gex(
    quotes: Sequence[OptionQuote],
    spot: float,
    use_open_interest: bool,
) -> dict[float, StrikeGEX]:
    """Aggregate GEX per strike from all quotes."""
    call_by_strike: dict[float, float] = {}
    put_by_strike: dict[float, float] = {}

    for q in quotes:
        strike = q.contract.strike
        gex = _compute_contract_gex(q, spot, use_open_interest)
        if q.contract.right == "C":
            call_by_strike[strike] = call_by_strike.get(strike, 0.0) + gex
        else:
            put_by_strike[strike] = put_by_strike.get(strike, 0.0) + gex

    all_strikes = sorted(set(call_by_strike) | set(put_by_strike))
    result: dict[float, StrikeGEX] = {}
    for strike in all_strikes:
        c = call_by_strike.get(strike, 0.0)
        p = put_by_strike.get(strike, 0.0)
        result[strike] = StrikeGEX(strike=strike, call_gex=c, put_gex=p, net_gex=c + p)

    return result


def _find_zero_gamma(strikes: list[StrikeGEX]) -> float | None:
    """Linearly interpolate the strike where cumulative net GEX crosses zero."""
    if len(strikes) < 2:
        return None

    net_values = np.array([s.net_gex for s in strikes])
    strike_values = np.array([s.strike for s in strikes])

    for i in range(len(net_values) - 1):
        if net_values[i] * net_values[i + 1] < 0:
            # Linear interpolation between sign-change points
            frac = abs(net_values[i]) / (abs(net_values[i]) + abs(net_values[i + 1]))
            return float(strike_values[i] + frac * (strike_values[i + 1] - strike_values[i]))

    return None
