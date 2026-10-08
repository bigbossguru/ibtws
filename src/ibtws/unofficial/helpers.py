from __future__ import annotations

import asyncio
import logging
import math
import datetime as _dt
from typing import Any, Sequence
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

# US option expirations are defined in exchange time, not the host's local time.
MARKET_TZ = ZoneInfo("America/New_York")
# Regular-session close. PM-settled options (SPXW, equity options) stop trading here.
MARKET_CLOSE = _dt.time(16, 0)


def market_now(now: float | None = None) -> _dt.datetime:
    """Current time (or the epoch ``now``) in exchange time (America/New_York)."""
    if now is None:
        return _dt.datetime.now(MARKET_TZ)
    return _dt.datetime.fromtimestamp(now, tz=MARKET_TZ)


def parse_expiry(expiry: str) -> _dt.date:
    """Parse an IB expiry string. ``YYYYMM`` monthlies map to the 15th as a proxy."""
    if len(expiry) == 6:
        expiry = expiry + "15"
    if len(expiry) != 8 or not expiry.isdigit():
        raise ValueError(f"Unrecognised expiry format: {expiry!r}")
    return _dt.date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:]))


def expiry_close(expiry: str) -> _dt.datetime:
    """Moment an expiry stops trading: 16:00 America/New_York on the expiry date."""
    return _dt.datetime.combine(parse_expiry(expiry), MARKET_CLOSE, tzinfo=MARKET_TZ)


def days_to_expiry(expiry: str, *, now: float | None = None) -> int:
    """Calendar days to expiry, counted in exchange time.

    Same-day expiries are ``0`` until the 16:00 ET close and ``-1`` after it,
    so a 0DTE selection keeps working all session long regardless of the
    host's timezone.
    """
    ref = market_now(now)
    days = (parse_expiry(expiry) - ref.date()).days
    if days == 0 and ref.time() >= MARKET_CLOSE:
        return -1
    return days


def safe_pick_value(obj: object, attr: str, *, allow_negative: bool = False) -> float | None:
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


def calc_dte(expiration: str) -> float:
    """Calendar days from today (exchange time) to expiration (YYYYMMDD), floored at 0."""
    exp_date = _dt.datetime.strptime(expiration, "%Y%m%d").date()
    delta = exp_date - market_now().date()
    return max(delta.days, 0.0)


async def snapshot_each(ib: Any, contracts: Sequence[Any], *, timeout: float | None, **kwargs: Any) -> list[Any]:
    """Snapshot every contract with its own ``reqTickersAsync`` call.

    ``reqTickersAsync(*contracts)`` fails as a whole when IB errors on any one
    contract (e.g. 10090 "market data is not subscribed"), losing the quotes
    of all the others. Here each contract succeeds or fails on its own: the
    result is positional, with ``None`` for a contract whose snapshot failed
    or timed out. ``kwargs`` are passed through (e.g. ``regulatorySnapshot``).
    """

    async def one(contract: Any) -> Any:
        try:
            tickers = await asyncio.wait_for(ib.reqTickersAsync(contract, **kwargs), timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"snapshot failed for {getattr(contract, 'symbol', '?')} conId={getattr(contract, 'conId', 0)}: {exc}"
            )
            return None
        return tickers[0] if tickers else None

    return list(await asyncio.gather(*(one(c) for c in contracts)))


def chunked(seq: Sequence, size: int):
    """Yield successive ``size``-length slices of *seq* (preserves element type)."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
