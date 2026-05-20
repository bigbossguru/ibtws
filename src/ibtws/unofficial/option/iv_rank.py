"""IV Rank / IV Percentile calculator built on :class:`IBKRClient`.

Uses IB's daily ``OPTION_IMPLIED_VOLATILITY`` historical series (the 30-day
ATM IV that TWS surfaces in the option-trader pane) as the input signal.

Definitions
-----------
* **IV Rank** — where today's IV sits inside the lookback ``[min, max]``
  band, expressed as a percentage::

      iv_rank = (current - min) / (max - min) * 100

  ``None`` when ``max == min`` (degenerate flat history).

* **IV Percentile** — fraction of historical observations *strictly below*
  today's IV, expressed as a percentage. Robust to outliers because it
  ignores the magnitude of the extremes.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Optional

from ib_async import Contract

from ibtws.unofficial.client import IBKRClient

from .utils import _safe_float


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IVRankResult:
    """Result of an IV Rank / IV Percentile computation for one underlying."""

    underlying_symbol: str
    as_of: Optional[_dt.date]
    current_iv: Optional[float]
    min_iv: Optional[float]
    max_iv: Optional[float]
    iv_rank: Optional[float]  # 0..100, or None when the band is degenerate
    iv_percentile: Optional[float]  # 0..100
    sample_size: int
    lookback_days: int


class IVRankCalculator:
    """Compute IV Rank / IV Percentile from IB's historical 30-day ATM IV series.

    The calculator does not manage the IB connection — the caller owns
    ``connect()`` / ``disconnect()`` on the supplied :class:`IBKRClient`.
    """

    def __init__(
        self,
        client: IBKRClient,
        *,
        request_timeout: float = 60.0,
    ) -> None:
        self._client = client
        self._timeout = request_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def calculate(
        self,
        underlying: Contract,
        *,
        lookback_days: int = 252,
        end_datetime: str = "",
        use_rth: bool = True,
    ) -> IVRankResult:
        """Fetch the IV history and reduce it to an :class:`IVRankResult`.

        Parameters
        ----------
        underlying:
            The underlying contract (stock / index). Qualified on the fly
            when ``conId`` is missing.
        lookback_days:
            Trading-day window for the historical IV series. ``252`` ≈ 1y.
        end_datetime:
            Right edge of the window. Empty string = "now" (IB convention).
        use_rth:
            Restrict to Regular Trading Hours bars (recommended).
        """
        if not underlying.conId:
            (underlying,) = await self._client.qualify(underlying)

        duration = f"{max(1, int(lookback_days))} D"
        logger.info(f"IVRankCalculator: requesting {duration} of OPTION_IMPLIED_VOLATILITY for {underlying.symbol}")

        try:
            bars = await self._client.ib.reqHistoricalDataAsync(
                underlying,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="OPTION_IMPLIED_VOLATILITY",
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"IVRankCalculator: historical IV request failed for {underlying.symbol}: {exc}")
            bars = []

        ivs: list[float] = []
        dates: list[_dt.date] = []
        for bar in bars or []:
            v = _safe_float(getattr(bar, "close", None))
            if v is None or v <= 0:
                continue
            ivs.append(v)
            d = getattr(bar, "date", None)
            if isinstance(d, _dt.datetime):
                dates.append(d.date())
            elif isinstance(d, _dt.date):
                dates.append(d)

        if not ivs:
            logger.warning(f"IVRankCalculator: no usable IV bars returned for {underlying.symbol}")
            return IVRankResult(
                underlying_symbol=underlying.symbol,
                as_of=None,
                current_iv=None,
                min_iv=None,
                max_iv=None,
                iv_rank=None,
                iv_percentile=None,
                sample_size=0,
                lookback_days=lookback_days,
            )

        current = ivs[-1]
        lo = min(ivs)
        hi = max(ivs)
        spread = hi - lo
        iv_rank = ((current - lo) / spread * 100.0) if spread > 0 else None
        # Percentile: share of *historical* observations strictly below current.
        # Always exclude the current bar; if that leaves no history (single-bar
        # series) the percentile is undefined — do NOT silently return 0.0,
        # which downstream strategies could misread as "IV at bottom of range".
        history = ivs[:-1]
        below = sum(1 for v in history if v < current)
        iv_percentile = (below / len(history) * 100.0) if history else None

        return IVRankResult(
            underlying_symbol=underlying.symbol,
            as_of=dates[-1] if dates else None,
            current_iv=current,
            min_iv=lo,
            max_iv=hi,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            sample_size=len(ivs),
            lookback_days=lookback_days,
        )
