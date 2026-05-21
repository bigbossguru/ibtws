"""Shared concurrency + pacing primitive for IBKR API calls.

The order manager and option-chain fetcher both have to stay under IB's
~50 msg/s ceiling. Previously each maintained its own semaphore + token
bucket — independent buckets meant the aggregate rate could exceed the
limit. This module centralises both knobs into one reusable executor that
can be shared across subsystems or instantiated per-subsystem when isolation
matters more than aggregate fairness.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator


class ThrottledExecutor:
    """Concurrency-bounded, rate-limited slot allocator.

    ``max_concurrency`` caps in-flight calls. ``pace_per_sec`` enforces a
    minimum interval between successive ``slot()`` acquisitions via a
    monotonic token bucket. Setting ``pace_per_sec=0`` disables pacing
    (semaphore still applies). The same executor instance can be shared
    by multiple callers — the bucket then governs the aggregate rate.
    """

    def __init__(self, *, max_concurrency: int, pace_per_sec: float) -> None:
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency!r}")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._min_interval = 1.0 / pace_per_sec if pace_per_sec > 0 else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()

    @property
    def min_interval(self) -> float:
        """Minimum seconds between slot acquisitions; 0 when pacing disabled."""
        return self._min_interval

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire a concurrency + pacing slot. Releases on exit."""
        async with self._semaphore:
            await self._await_next_slot()
            yield

    async def _await_next_slot(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._min_interval
