"""Expected-move calculation using two complementary methods.

1. **ATM Straddle** — sum of ATM call + put mid prices (market-implied).
2. **IV-based 1σ** — ``spot × IV × √(DTE/365)`` from ATM implied volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ibtws.unofficial.helpers import calc_dte

REQUIRED_COLUMNS = {"strike", "right", "bid", "ask", "iv", "underlying_price", "expiry"}


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
    """Compute expected move from an options chain DataFrame.

    The DataFrame must contain at least the columns defined in
    :data:`REQUIRED_COLUMNS`: ``strike``, ``right``, ``bid``, ``ask``, ``iv``,
    ``underlying_price``, ``expiry``.  An optional ``symbol`` column is used
    for the result's ``underlying_symbol``.

    This aligns with the DataFrame produced by
    :func:`~ibtws.unofficial.option.utils.quotes_to_dataframe`.
    """

    REQUIRED_COLUMNS = REQUIRED_COLUMNS

    def calculate(self, df: pd.DataFrame) -> ExpectedMoveResult:
        """Calculate expected move from an options chain DataFrame.

        Parameters
        ----------
        df:
            Options chain DataFrame with columns matching
            :data:`REQUIRED_COLUMNS`.

        Returns
        -------
        ExpectedMoveResult
            Aggregated expected-move estimates.

        Raises
        ------
        ValueError
            If *df* is empty or missing required columns/data.
        """
        if df.empty:
            raise ValueError("Cannot calculate expected move from empty DataFrame")

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        # Derive spot & expiration from the DataFrame
        spot = self._extract_spot(df)
        expiration = self._extract_expiration(df)
        symbol = self._extract_symbol(df)

        # Method 1 & 2: from option quotes
        atm_call, atm_put = self._find_atm_pair(df, spot)
        straddle_move = self._straddle_expected_move(atm_call, atm_put)
        atm_iv = self._extract_atm_iv(atm_call, atm_put)
        iv_move = self._iv_expected_move(spot, atm_iv, expiration) if atm_iv else None

        return ExpectedMoveResult(
            underlying_symbol=symbol,
            spot=spot,
            expiration=expiration,
            straddle_move=straddle_move,
            straddle_pct=straddle_move / spot * 100 if straddle_move else None,
            iv_move=iv_move,
            iv_pct=iv_move / spot * 100 if iv_move else None,
            atm_iv=atm_iv,
            avg_move=(straddle_move + iv_move) / 2.0 if straddle_move is not None and iv_move is not None else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_spot(self, df: pd.DataFrame) -> float:
        """Get spot price from underlying_price column."""
        prices = df["underlying_price"].dropna()
        if prices.empty:
            raise ValueError("Cannot determine spot price: all underlying_price values are NaN")
        spot = float(prices.iloc[0])
        if spot <= 0:
            raise ValueError(f"Invalid spot price: {spot}")
        return spot

    def _extract_expiration(self, df: pd.DataFrame) -> str:
        """Get expiration from expiry column (first non-null value)."""
        expiries = df["expiry"].dropna()
        if expiries.empty:
            raise ValueError("Cannot determine expiration: all expiry values are NaN")
        return str(expiries.iloc[0])

    def _extract_symbol(self, df: pd.DataFrame) -> str:
        """Get symbol if available."""
        if "symbol" not in df.columns:
            return ""
        symbols = df["symbol"].dropna()
        return str(symbols.iloc[0]) if not symbols.empty else ""

    def _find_atm_pair(self, df: pd.DataFrame, spot: float) -> tuple[pd.Series | None, pd.Series | None]:
        """Find the call and put row closest to spot."""
        calls = df[df["right"] == "C"]
        puts = df[df["right"] == "P"]

        atm_call = None
        atm_put = None

        if not calls.empty:
            idx = (calls["strike"] - spot).abs().idxmin()
            atm_call = calls.loc[idx]

        if not puts.empty:
            idx = (puts["strike"] - spot).abs().idxmin()
            atm_put = puts.loc[idx]

        return atm_call, atm_put

    def _quote_mid(self, row: pd.Series) -> float | None:
        """Calculate mid price from a row; returns None if invalid."""
        bid = row.get("bid")
        ask = row.get("ask")
        if bid is None or ask is None or pd.isna(bid) or pd.isna(ask):
            return None
        bid, ask = float(bid), float(ask)
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        return (bid + ask) / 2.0

    def _straddle_expected_move(self, atm_call: pd.Series | None, atm_put: pd.Series | None) -> float | None:
        """Expected move = ATM call mid + ATM put mid."""
        if atm_call is None or atm_put is None:
            return None
        call_mid = self._quote_mid(atm_call)
        put_mid = self._quote_mid(atm_put)
        if call_mid is None or put_mid is None:
            return None
        return call_mid + put_mid

    def _extract_atm_iv(self, atm_call: pd.Series | None, atm_put: pd.Series | None) -> float | None:
        """Average IV of ATM call and put."""
        ivs: list[float] = []
        for row in (atm_call, atm_put):
            if row is None:
                continue
            iv = row.get("iv")
            if iv is not None and not pd.isna(iv) and float(iv) > 0:
                ivs.append(float(iv))
        return sum(ivs) / len(ivs) if ivs else None

    def _iv_expected_move(self, spot: float, iv: float, expiration: str) -> float | None:
        """1σ expected move = spot × IV × √(DTE/365)."""
        if spot <= 0 or iv <= 0:
            return None
        dte = calc_dte(expiration) + 1  # include today
        return spot * iv * math.sqrt(dte / 365.0)
