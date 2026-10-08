# Non-official package. Not affiliated with ib_async upstream.

"""Fan-out event bus for :class:`OrderEvent` — both async iterators and sync callbacks."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncIterator, Callable

from .models import OrderEvent

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUEUE = 10_000

# Sentinel pushed into subscriber queues by close() to end their streams.
_CLOSED = object()


class OrderMonitor:
    """Single source of truth for order/position events.

    Two consumption styles, both safe to mix:

    * ``async for event in monitor.stream():`` — every stream gets its own
      bounded queue, so each subscriber sees every event published after it
      subscribed. The first subscriber also receives the backlog buffered
      while nobody was listening.
    * ``monitor.register(callback)`` — sync hook fired inline. Callback
      exceptions are logged and swallowed so a bad subscriber cannot poison
      the bus.

    Queues are bounded by ``max_queue``: a subscriber that falls that far
    behind loses its oldest events (counted in :attr:`dropped`) instead of
    growing memory for the rest of the session.
    """

    def __init__(self, *, max_queue: int = DEFAULT_MAX_QUEUE) -> None:
        if max_queue <= 0:
            raise ValueError(f"max_queue must be positive, got {max_queue!r}")
        self._max_queue = max_queue
        self._subscribers: list[asyncio.Queue] = []
        self._backlog: deque[OrderEvent] = deque()
        self._callbacks: list[Callable[[OrderEvent], None]] = []
        self._closed = False
        self.dropped = 0

    def publish(self, event: OrderEvent) -> None:
        if self._subscribers:
            for q in self._subscribers:
                self._offer(q, event)
        else:
            if len(self._backlog) >= self._max_queue:
                self._backlog.popleft()
                self.dropped += 1
            self._backlog.append(event)
        for fn in list(self._callbacks):
            try:
                fn(event)
            except Exception:
                logger.exception(f"OrderMonitor: callback {fn!r} raised on {type(event).__name__}")

    def _offer(self, q: asyncio.Queue, item: object) -> None:
        if q.full():
            q.get_nowait()
            self.dropped += 1
        q.put_nowait(item)

    async def stream(self) -> AsyncIterator[OrderEvent]:
        if self._closed:
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        if not self._subscribers:
            while self._backlog:
                q.put_nowait(self._backlog.popleft())
        self._subscribers.append(q)
        try:
            while True:
                item = await q.get()
                if item is _CLOSED:
                    return
                yield item
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def close(self) -> None:
        """End every active stream; later ``stream()`` calls return immediately."""
        self._closed = True
        for q in list(self._subscribers):
            self._offer(q, _CLOSED)

    def reopen(self) -> None:
        """Allow new streams again after :meth:`close` (used on manager restart)."""
        self._closed = False

    def register(self, fn: Callable[[OrderEvent], None]) -> None:
        self._callbacks.append(fn)

    def unregister(self, fn: Callable[[OrderEvent], None]) -> None:
        if fn in self._callbacks:
            self._callbacks.remove(fn)
