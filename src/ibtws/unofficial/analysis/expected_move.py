"""Expected-move calculation using three complementary methods.

1. **ATM Straddle** — sum of ATM call + put mid prices (market-implied).
2. **IV-based 1σ** — ``spot × IV × √(DTE/365)`` from ATM implied volatility.
3. **Historical Volatility 1σ** — same formula but using realized vol from price history.
"""

from __future__ import annotations

import math
import logging
from typing import Sequence
from dataclasses import dataclass

from ibtws.unofficial.option.models import OptionQuote
from ibtws.unofficial.helpers import calc_dte

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpectedMoveResult:
    """Aggregated expected-move estimates for one underlying + expiration."""

    underlying_symbol: str
    spot: float
    expiration: str  # YYYYMMDD

    # Method 1: ATM straddle
    straddle_move: float | None = None
    straddle_pct: float | None = None

    # Method 2: IV-based 1σ
    iv_move: float | None = None
    iv_pct: float | None = None
    atm_iv: float | None = None

    avg_move: float | None = None  # average of straddle_move and iv_move


class ExpectedMoveCalculator:
    """Compute expected move using straddle, IV, and historical volatility methods."""

    async def calculate(self, quotes: Sequence[OptionQuote]) -> ExpectedMoveResult:
        """Calculate expected move for *underlying* at *expiration* (YYYYMMDD).

        Parameters
        ----------
        quotes:
            List of option quotes for the underlying at the target expiration.
        """
        first_quote = quotes[0] if quotes else None
        if first_quote is None:
            raise ValueError("Cannot determine contract details from empty quotes list")

        spot = first_quote.underlying_price
        if spot is None:
            symbol = getattr(first_quote.contract, "symbol", None)
            raise ValueError(f"Cannot determine spot price for {symbol or 'unknown'}")

        expiration = getattr(first_quote.contract, "lastTradeDateOrContractMonth", None)
        if not expiration:
            raise ValueError("Cannot determine expiration from quote")

        # Method 1 & 2: from option quotes
        atm_call, atm_put = self._find_atm_pair(quotes, spot)
        straddle_move = self._straddle_expected_move(atm_call, atm_put)
        atm_iv = self._extract_atm_iv(atm_call, atm_put)
        iv_move = self._iv_expected_move(spot, atm_iv, expiration) if atm_iv else None

        return ExpectedMoveResult(
            underlying_symbol=getattr(first_quote.contract, "symbol", None) or "",
            spot=spot,
            expiration=expiration,
            straddle_move=straddle_move,
            straddle_pct=straddle_move / spot * 100 if straddle_move else None,
            iv_move=iv_move,
            iv_pct=iv_move / spot * 100 if iv_move else None,
            atm_iv=atm_iv,
            avg_move=(straddle_move + iv_move) / 2.0 if straddle_move is not None and iv_move is not None else None,
        )

    def _find_atm_pair(
        self, quotes: Sequence[OptionQuote], spot: float
    ) -> tuple[OptionQuote | None, OptionQuote | None]:
        """Find the call and put closest to spot."""
        calls = [q for q in quotes if q.contract.right == "C"]
        puts = [q for q in quotes if q.contract.right == "P"]

        atm_call = min(calls, key=lambda q: abs(q.contract.strike - spot), default=None)
        atm_put = min(puts, key=lambda q: abs(q.contract.strike - spot), default=None)
        return atm_call, atm_put

    def _quote_mid(self, quote: OptionQuote) -> float | None:
        if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
            return None
        return (quote.bid + quote.ask) / 2.0

    def _straddle_expected_move(self, atm_call: OptionQuote | None, atm_put: OptionQuote | None) -> float | None:
        """Expected move = ATM call mid + ATM put mid."""
        if atm_call is None or atm_put is None:
            return None
        call_mid = self._quote_mid(atm_call)
        put_mid = self._quote_mid(atm_put)
        if call_mid is None or put_mid is None:
            return None
        return call_mid + put_mid

    def _extract_atm_iv(self, atm_call: OptionQuote | None, atm_put: OptionQuote | None) -> float | None:
        """Average IV of ATM call and put."""
        ivs = [q.iv for q in (atm_call, atm_put) if q and q.iv and q.iv > 0]
        return sum(ivs) / len(ivs) if ivs else None

    def _iv_expected_move(self, spot: float, iv: float, expiration: str) -> float | None:
        """1σ expected move = spot × IV × √(DTE/365)."""
        if spot <= 0 or iv <= 0:
            return None
        dte = calc_dte(expiration) + 1  # include today
        return spot * iv * math.sqrt(dte / 365.0)
