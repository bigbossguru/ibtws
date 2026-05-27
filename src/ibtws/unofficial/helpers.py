from __future__ import annotations

import logging
import math
import datetime as _dt
from typing import Sequence

from ib_async import Contract

from ibtws.unofficial.client import IBKRClient

logger = logging.getLogger(__name__)


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


async def fetch_spot(underlying: Contract, client: IBKRClient) -> float | None:
    t = await client.get_market_data(underlying)
    for attr in ("last", "close"):
        v = safe_pick_value(t, attr)
        if v is not None:
            return v
    bid = safe_pick_value(t, "bid")
    ask = safe_pick_value(t, "ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    logger.warning(f"OptionChainFetcher: spot for {underlying.symbol} unavailable — skipping auto-window")
    return None


def calc_dte(expiration: str) -> float:
    """Calendar days from now to expiration (YYYYMMDD)."""
    exp_date = _dt.datetime.strptime(expiration, "%Y%m%d").date()
    delta = exp_date - _dt.date.today()
    return max(delta.days, 0.0)


def chunked(seq: Sequence, size: int):
    """Yield successive ``size``-length slices of *seq* (preserves element type)."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
