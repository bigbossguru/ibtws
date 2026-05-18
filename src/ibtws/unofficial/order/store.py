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
    """Pluggable persistence backend for :class:`OrderEvent`.

    The contract is intentionally minimal: write events forward in time
    (``append``) and read them back in the order they were written
    (``replay``). No update, no delete, no query — the audit log is
    append-only by design so that crash recovery is just "replay from the
    start, the last good line wins."

    Implementations supply their own storage medium (JSONL file, SQLite,
    Kafka topic, …). :class:`JsonStore` is the only implementation that
    ships; swap it out by passing any other Protocol-conforming object to
    :class:`OrderManager`.
    """

    async def append(self, event: OrderEvent) -> None: ...
    def replay(self) -> Iterator[OrderEvent]: ...


class JsonStore:
    """Append-only JSON Lines :class:`OrderStore` with crash-safe writes.

    Every :class:`OrderEvent` becomes one ``json.dumps`` line in ``path``,
    flushed (and optionally ``fsync``-ed) before ``append`` returns — so a
    process kill, OOM, or kernel panic loses at most the line currently
    being written. The file is never rewritten or truncated; recovery is
    pure replay.

    Used by :class:`OrderManager` to (1) persist every state transition
    for audit, and (2) feed :func:`reconcile` on start-up so a restarted
    process can rejoin its previously-submitted orders by ``orderRef``.

    Parameters
    ----------
    path:
        File path. Parent directory must exist. The file is created on
        first ``append``; a missing file at ``replay`` time yields an
        empty iterator (no error).
    fsync:
        When True (default), call ``os.fsync`` after each write so the
        line survives a kernel crash, not just a process crash. Set False
        in tests for ~10× faster append throughput.
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
