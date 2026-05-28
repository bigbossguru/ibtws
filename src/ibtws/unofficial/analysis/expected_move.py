"""Expected-move calculation using three complementary methods.

1. **ATM Straddle** — sum of ATM call + put mid prices (market-implied).
2. **IV-based 1σ** — ``spot × IV × √(DTE/365)`` from ATM implied volatility.
3. **Historical Volatility 1σ** — same formula but using realized vol from price history.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ib_async import Contract

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import calc_dte
from ibtws.unofficial.option.chains import OptionChainFetcher
from ibtws.unofficial.option.models import OptionQuote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpectedMoveResult:
    """Aggregated expected-move estimates for one underlying + expiration."""

    underlying_symbol: str
    spot: float
    expiration: str  # YYYYMMDD
    dte: float  # calendar days to expiration

    # Method 1: ATM straddle
    straddle_move: float | None = None
    straddle_pct: float | None = None

    # Method 2: IV-based 1σ
    iv_move: float | None = None
    iv_pct: float | None = None
    atm_iv: float | None = None

    # Method 3: Historical volatility 1σ
    hv_move: float | None = None
    hv_pct: float | None = None
    hv: float | None = None


class ExpectedMoveCalculator:
    """Compute expected move using straddle, IV, and historical volatility methods."""

    def __init__(self, client: IBKRClient, chain_fetcher: OptionChainFetcher) -> None:
        self._client = client
        self._chain = chain_fetcher

    async def calculate(
        self,
        underlying: Contract,
        expiration: str,
        *,
        hv_lookback_days: int = 30,
    ) -> ExpectedMoveResult:
        """Calculate expected move for *underlying* at *expiration* (YYYYMMDD).

        Parameters
        ----------
        underlying:
            Underlying contract (will be qualified if needed).
        expiration:
            Target expiration in YYYYMMDD format.
        hv_lookback_days:
            Number of trading days for historical volatility calculation.
        """
        quotes = await self._chain.fetch_snapshot(
            underlying,
            expirations=[expiration],
            strike_window_pct=0.05,
            rights=("C", "P"),
        )

        spot = quotes[0].underlying_price if quotes else None
        if spot is None:
            raise ValueError(f"Cannot determine spot price for {underlying.symbol}")

        dte = calc_dte(expiration)

        # Method 1 & 2: from option quotes
        atm_call, atm_put = _find_atm_pair(quotes, spot)
        straddle_move = _straddle_expected_move(atm_call, atm_put)
        atm_iv = _extract_atm_iv(atm_call, atm_put)
        iv_move = _iv_expected_move(spot, atm_iv, dte) if atm_iv else None

        # Method 3: historical volatility
        hv = await self._calc_hv(underlying, hv_lookback_days)
        hv_move = _iv_expected_move(spot, hv, dte) if hv else None

        return ExpectedMoveResult(
            underlying_symbol=underlying.symbol,
            spot=spot,
            expiration=expiration,
            dte=dte,
            straddle_move=straddle_move,
            straddle_pct=straddle_move / spot * 100 if straddle_move else None,
            iv_move=iv_move,
            iv_pct=iv_move / spot * 100 if iv_move else None,
            atm_iv=atm_iv,
            hv_move=hv_move,
            hv_pct=hv_move / spot * 100 if hv_move else None,
            hv=hv,
        )

    async def _calc_hv(self, underlying: Contract, lookback_days: int) -> float | None:
        """Annualized historical volatility from daily close-to-close returns."""
        from ibtws.unofficial.client import Duration

        duration_map: dict[int, Duration] = {5: "5 D", 10: "10 D", 30: "30 D"}
        duration: Duration = duration_map.get(lookback_days, "30 D")
        df = await self._client.get_historical_data(
            underlying,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
        )
        if df is None or df.empty or len(df) < 5:
            logger.warning(f"ExpectedMove: insufficient historical data for {underlying.symbol}")
            return None

        closes = df["close"].values
        log_returns = np.diff(np.log(closes))
        return float(np.std(log_returns, ddof=1) * math.sqrt(252))


def _find_atm_pair(quotes: Sequence[OptionQuote], spot: float) -> tuple[OptionQuote | None, OptionQuote | None]:
    """Find the call and put closest to spot."""
    calls = [q for q in quotes if q.contract.right == "C"]
    puts = [q for q in quotes if q.contract.right == "P"]

    atm_call = min(calls, key=lambda q: abs(q.contract.strike - spot), default=None)
    atm_put = min(puts, key=lambda q: abs(q.contract.strike - spot), default=None)
    return atm_call, atm_put


def _quote_mid(q: OptionQuote) -> float | None:
    if q.bid is None or q.ask is None or q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
        return None
    return (q.bid + q.ask) / 2.0


def _straddle_expected_move(atm_call: OptionQuote | None, atm_put: OptionQuote | None) -> float | None:
    """Expected move = ATM call mid + ATM put mid."""
    if atm_call is None or atm_put is None:
        return None
    call_mid = _quote_mid(atm_call)
    put_mid = _quote_mid(atm_put)
    if call_mid is None or put_mid is None:
        return None
    return call_mid + put_mid


def _extract_atm_iv(atm_call: OptionQuote | None, atm_put: OptionQuote | None) -> float | None:
    """Average IV of ATM call and put."""
    ivs = [q.iv for q in (atm_call, atm_put) if q and q.iv and q.iv > 0]
    return sum(ivs) / len(ivs) if ivs else None


def _iv_expected_move(spot: float, iv: float, dte: float) -> float | None:
    """1σ expected move = spot × IV × √(DTE/365)."""
    if dte <= 0 or iv <= 0:
        return None
    return spot * iv * math.sqrt(dte / 365.0)
