# Non-official package. Not affiliated with ib_async upstream.

"""Append-only JSON Lines audit store for :class:`OrderEvent`.

Each ``append`` writes one line + ``flush`` + ``fsync`` so a crash mid-write
loses at most the line we were writing. ``replay`` streams the file back as
events — used by :func:`reconcile` on startup.

The :class:`OrderStore` ``Protocol`` exists so callers can swap in a SQLite
or remote-log backend without touching the manager. Only :class:`JsonStore`
ships today.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from .models import OrderEvent, event_from_dict

logger = logging.getLogger(__name__)


@runtime_checkable
class OrderStore(Protocol):
    async def append(self, event: OrderEvent) -> None: ...
    def replay(self) -> Iterator[OrderEvent]: ...


class JsonStore:
    """Crash-safe append-only JSONL store.

    Parameters
    ----------
    path:
        File path. Parent directory must exist. File is created on first
        write; missing file at ``replay`` time yields an empty iterator.
    fsync:
        When True (default), ``fsync`` after each write — durable across
        kernel crashes. Set False in tests for speed.
    """

    def __init__(self, path: Path | str, *, fsync: bool = True) -> None:
        self._path = Path(path)
        self._fsync = fsync
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def append(self, event: OrderEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"), default=str)
        async with self._lock:
            # Sync I/O inside a brief critical section. Order rates are slow
            # enough (humans / strategies, not market data) that blocking the
            # loop for a single fsync is acceptable in exchange for durability.
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                if self._fsync:
                    os.fsync(f.fileno())

    def replay(self) -> Iterator[OrderEvent]:
        if not self._path.exists():
            return iter(())
        return self._replay()

    def _replay(self) -> Iterator[OrderEvent]:
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield event_from_dict(data)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    raise ValueError(f"{self._path}:{line_no}: corrupt event line — {exc}") from exc
