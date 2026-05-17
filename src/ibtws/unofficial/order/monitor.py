# Non-official package. Not affiliated with ib_async upstream.

"""Fan-out event bus for :class:`OrderEvent` — both async iterator and sync callbacks."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable

from .models import OrderEvent

logger = logging.getLogger(__name__)


class OrderMonitor:
    """Single source of truth for order/position events.

    Two consumption styles, both safe to mix:

    * ``async for event in monitor.stream():`` — backpressure-friendly queue.
    * ``monitor.register(callback)`` — sync hook fired inline. Callback
      exceptions are logged and swallowed so a bad subscriber cannot poison
      the bus.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[OrderEvent] = asyncio.Queue()
        self._callbacks: list[Callable[[OrderEvent], None]] = []

    def publish(self, event: OrderEvent) -> None:
        self._queue.put_nowait(event)
        for fn in list(self._callbacks):
            try:
                fn(event)
            except Exception:
                logger.exception(f"OrderMonitor: callback {fn!r} raised on {type(event).__name__}")

    async def stream(self) -> AsyncIterator[OrderEvent]:
        while True:
            event = await self._queue.get()
            yield event

    def register(self, fn: Callable[[OrderEvent], None]) -> None:
        self._callbacks.append(fn)

    def unregister(self, fn: Callable[[OrderEvent], None]) -> None:
        if fn in self._callbacks:
            self._callbacks.remove(fn)
