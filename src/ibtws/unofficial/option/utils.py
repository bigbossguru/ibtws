"""Pure helpers + DataFrame projection used across the option package."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import pandas as pd
from ib_async import Ticker

from .models import OptionQuote


def _chunked(seq: Sequence, size: int):
    """Yield successive ``size``-length slices of *seq* (preserves element type)."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _filter_expirations(
    available: tuple[str, ...],
    explicit: Optional[Iterable[str]],
    expiry_from: Optional[str],
    expiry_to: Optional[str],
) -> list[str]:
    """Narrow chain expirations: explicit whitelist wins over inclusive range."""
    if explicit is not None:
        wanted = set(explicit)
        return [e for e in available if e in wanted]
    return [e for e in available if (expiry_from is None or e >= expiry_from) and (expiry_to is None or e <= expiry_to)]


def _filter_strikes(
    available: tuple[float, ...],
    explicit: Optional[Iterable[float]],
    strike_from: Optional[float],
    strike_to: Optional[float],
) -> list[float]:
    """Narrow chain strikes: explicit whitelist wins over inclusive range."""
    if explicit is not None:
        wanted = set(explicit)
        return [s for s in available if s in wanted]
    return [s for s in available if (strike_from is None or s >= strike_from) and (strike_to is None or s <= strike_to)]


def _safe_pick_value(obj: object, attr: str, *, allow_negative: bool = False) -> Optional[float]:
    """Return the price or any value at ``attr``, scrubbing IB's ``-1`` / NaN sentinels.

    IB uses ``-1.0`` (and sometimes other negative values) to signal "no data"
    on price fields (bid, ask, last, close, volume, OI). By default these are
    filtered out. Pass ``allow_negative=True`` for fields that are legitimately
    negative (e.g. delta, theta).
    """
    value = getattr(obj, attr, None)
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    if not allow_negative and f < 0:
        return None
    return f


def _ticker_to_quote(ticker: Ticker, underlying_price: Optional[float] = None) -> OptionQuote:
    """Map an ib_async ``Ticker`` snapshot into an :class:`OptionQuote`."""
    greeks = getattr(ticker, "modelGreeks", None)
    return OptionQuote(
        contract=ticker.contract,
        bid=_safe_pick_value(ticker, "bid"),
        ask=_safe_pick_value(ticker, "ask"),
        volume=_safe_pick_value(ticker, "volume"),
        open_interest=_safe_pick_value(
            ticker, "callOpenInterest" if ticker.contract.right == "C" else "putOpenInterest"
        ),
        iv=_safe_pick_value(greeks, "impliedVol") if greeks else None,
        delta=_safe_pick_value(greeks, "delta", allow_negative=True) if greeks else None,
        gamma=_safe_pick_value(greeks, "gamma") if greeks else None,
        vega=_safe_pick_value(greeks, "vega") if greeks else None,
        theta=_safe_pick_value(greeks, "theta", allow_negative=True) if greeks else None,
        underlying_price=underlying_price,
    )


DATAFRAME_COLUMNS = (
    "symbol",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "vega",
    "theta",
    "underlying_price",
    "timestamp",
)


def quotes_to_dataframe(quotes: Sequence[OptionQuote]) -> "pd.DataFrame":
    """Convert a sequence of :class:`OptionQuote` into a pandas ``DataFrame``.

    Returns an empty DataFrame with :data:`DATAFRAME_COLUMNS` when *quotes* is
    empty, so downstream ``df["strike"]`` / ``df.empty`` checks always work.
    """
    if not quotes:
        return pd.DataFrame(columns=list(DATAFRAME_COLUMNS))

    return pd.DataFrame(
        [
            {
                "symbol": q.contract.symbol,
                "expiry": q.contract.lastTradeDateOrContractMonth,
                "strike": q.contract.strike,
                "right": q.contract.right,
                "bid": q.bid,
                "ask": q.ask,
                "volume": q.volume,
                "open_interest": q.open_interest,
                "iv": q.iv,
                "delta": q.delta,
                "gamma": q.gamma,
                "vega": q.vega,
                "theta": q.theta,
                "underlying_price": q.underlying_price,
                "timestamp": q.timestamp,
            }
            for q in quotes
        ],
        columns=list(DATAFRAME_COLUMNS),
    )
