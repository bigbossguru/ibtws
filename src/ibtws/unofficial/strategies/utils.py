# Non-official package. Not affiliated with ib_async upstream.

"""Stateless helpers and exceptions for the strategies package."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from ibtws.unofficial.option import OptionQuote

from .models import SpreadType

logger = logging.getLogger(__name__)


class CreditSpreadError(RuntimeError):
    """Raised when a credit spread cannot be built or placed.

    The message is always actionable – includes which constraint failed and
    what numbers were observed – so callers can surface it to a user or log
    pipeline without further inspection.
    """


def _parse_expiry_to_dte(expiry: str, *, now: Optional[float] = None) -> int:
    """Convert a ``YYYYMMDD`` expiration string into days-to-expiry.

    IBKR also returns ``YYYYMM`` for monthlies – treated as the third-Friday
    convention (just use day 15 as a reasonable proxy). Negative DTEs (already
    expired) are returned as-is so the caller can filter them out.
    """
    if len(expiry) == 6:
        expiry = expiry + "15"
    if len(expiry) != 8 or not expiry.isdigit():
        raise ValueError(f"Unrecognised expiry format: {expiry!r}")

    import datetime as _dt

    y, m, d = int(expiry[:4]), int(expiry[4:6]), int(expiry[6:])
    target = _dt.datetime(y, m, d, 16, 0, 0, tzinfo=_dt.timezone.utc)  # 4pm ET close as proxy
    ref = (
        _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc) if now is not None else _dt.datetime.now(_dt.timezone.utc)
    )
    return (target - ref).days


def select_expiry(
    expirations: Iterable[str],
    *,
    target_dte: int,
    dte_tolerance: int,
    now: Optional[float] = None,
) -> str:
    """Pick the expiry closest to ``target_dte`` within ``dte_tolerance``.

    Negative-DTE (expired) entries are ignored. Raises
    :class:`CreditSpreadError` if no expiry survives the tolerance window.
    """
    candidates: list[tuple[int, str]] = []
    for exp in expirations:
        try:
            dte = _parse_expiry_to_dte(exp, now=now)
        except ValueError:
            continue
        if dte < 0:
            continue
        if abs(dte - target_dte) <= dte_tolerance:
            candidates.append((abs(dte - target_dte), exp))
    if not candidates:
        raise CreditSpreadError(
            f"No expiry within {dte_tolerance}d of target_dte={target_dte} in {list(expirations)!r}"
        )
    candidates.sort()
    chosen = candidates[0][1]
    logger.info(f"CreditSpread: chose expiry {chosen} (Δdte={candidates[0][0]}d) from {len(candidates)} candidate(s)")
    return chosen


def _quote_is_tradeable(
    quote: OptionQuote,
    *,
    min_open_interest: float,
    min_volume: float,
) -> bool:
    """Liquidity / data-quality filter applied before strike selection.

    A leg is rejected only when IB explicitly returned a too-low value. Legs
    whose OI or volume is ``None`` (IB simply did not include it in the
    snapshot) are *kept* — they would otherwise be discarded en masse for
    freshly-listed expiries.
    """
    if quote.delta is None:
        return False
    if quote.contract.conId == 0:
        return False
    if quote.open_interest is not None and quote.open_interest < min_open_interest:
        return False
    if quote.volume is not None and quote.volume < min_volume:
        return False
    return True


def select_short_leg(
    quotes: Sequence[OptionQuote],
    *,
    target_short_delta: float,
    max_short_delta: Optional[float],
    min_open_interest: float,
    min_volume: float,
) -> OptionQuote:
    """Pick the option whose ``|delta|`` is closest to the target.

    Filters out quotes that fail the liquidity / data-quality check first.
    Enforces ``max_short_delta`` as a hard ceiling — if no remaining quote
    fits, the spread is rejected (rather than silently widening risk).
    """
    candidates = [
        q for q in quotes if _quote_is_tradeable(q, min_open_interest=min_open_interest, min_volume=min_volume)
    ]
    if not candidates:
        raise CreditSpreadError(f"No tradeable quotes (need delta + conId; got {len(quotes)} raw)")
    if max_short_delta is not None:
        candidates = [q for q in candidates if q.delta is not None and abs(q.delta) <= max_short_delta]
        if not candidates:
            raise CreditSpreadError(f"All candidate strikes exceed max_short_delta={max_short_delta}")
    candidates.sort(key=lambda q: abs(abs(q.delta or 0.0) - target_short_delta))
    return candidates[0]


def select_long_leg(
    quotes: Sequence[OptionQuote],
    *,
    short: OptionQuote,
    wing_width: float,
    spread_type: SpreadType,
    min_open_interest: float,
    min_volume: float,
) -> OptionQuote:
    """Pick the protective leg at (or just beyond) the requested wing width.

    For a bull-put spread the long is *lower* than the short; for a bear-call
    spread the long is *higher*. We snap to the strike whose distance from
    the short is at least ``wing_width`` and minimised. Falls back to the
    farthest available strike when none meets the minimum width (e.g.
    chain truncated by ``strike_window_pct``).
    """
    short_strike = float(short.contract.strike)
    target = short_strike - wing_width if spread_type is SpreadType.BULL_PUT else short_strike + wing_width

    tradeable = [
        q
        for q in quotes
        if _quote_is_tradeable(q, min_open_interest=min_open_interest, min_volume=min_volume)
        and q.contract.conId != short.contract.conId
        and q.contract.right == short.contract.right
        and q.contract.lastTradeDateOrContractMonth == short.contract.lastTradeDateOrContractMonth
    ]
    if spread_type is SpreadType.BULL_PUT:
        tradeable = [q for q in tradeable if q.contract.strike < short_strike]
    else:
        tradeable = [q for q in tradeable if q.contract.strike > short_strike]
    if not tradeable:
        raise CreditSpreadError(
            f"No protective leg available on the {('lower' if spread_type is SpreadType.BULL_PUT else 'higher')} "
            f"side of strike {short_strike}"
        )

    tradeable.sort(key=lambda q: abs(q.contract.strike - target))
    chosen = tradeable[0]
    actual_width = abs(chosen.contract.strike - short_strike)
    if actual_width < wing_width * 0.5:
        # Final guard — refusing to trade a 1-strike-wide spread when user
        # asked for 10 wide. Better to fail loud than fill a tiny credit.
        raise CreditSpreadError(
            f"Closest protective strike only {actual_width:g} wide (requested {wing_width:g}) — chain too narrow"
        )
    return chosen


def _quote_mid(q: OptionQuote) -> Optional[float]:
    """Bid/ask midpoint with NaN-safe handling. ``None`` when either side is missing."""
    if q.bid is None or q.ask is None:
        return None
    if q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
        return None
    return (q.bid + q.ask) / 2.0


def _round_to_tick(price: float, tick: float = 0.05) -> float:
    """Round to the nearest IB-acceptable tick (default 5¢ for options)."""
    if tick <= 0:
        return price
    return round(price / tick) * tick
