# Non-official package. Not affiliated with ib_async upstream.

"""Append-only JSON Lines audit store for :class:`OrderEvent`.

Each ``append`` writes one line + ``flush`` + ``fsync`` so a crash mid-write
loses at most the line we were writing; such an unfinished last line is
skipped on ``replay`` and truncated before the next write. ``replay`` streams
the file back as events — used by :func:`reconcile` on startup.

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
from typing import IO, Iterator, Protocol, runtime_checkable

from .models import OrderEvent, event_from_dict

logger = logging.getLogger(__name__)

_NL = "\n"
_NL_BYTE = b"\n"


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
    being written. Recovery is pure replay.

    Used by :class:`OrderManager` to (1) persist every state transition
    for audit, and (2) feed :func:`reconcile` on start-up so a restarted
    process can rejoin its previously-submitted orders by ``orderRef``.

    Crash recovery: a final line without its trailing newline is a write the
    process did not finish. ``replay`` skips it with a warning, and the first
    ``append`` truncates it so new lines never get glued onto the fragment.
    A malformed line anywhere else is real corruption and still raises.

    The file stays open between appends, and the write + ``fsync`` run in a
    worker thread so a slow disk never stalls the event loop. Call
    :meth:`close` when done (optional; the handle is also closed on GC).

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
        self._fh: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    async def append(self, event: OrderEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"), default=str) + _NL
        async with self._lock:
            await asyncio.to_thread(self._write, line)

    def _write(self, line: str) -> None:
        if self._fh is None:
            self._repair_tail()
            self._fh = self._path.open("a", encoding="utf-8", newline=_NL)
        self._fh.write(line)
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())

    def _repair_tail(self) -> None:
        """Truncate an unterminated final line left by a crash mid-write."""
        if not self._path.exists():
            return
        with self._path.open("rb+") as f:
            size = f.seek(0, os.SEEK_END)
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == _NL_BYTE:
                return
            # Walk back to the last newline; everything after it is the torn write.
            keep = 0
            pos = size
            while pos > 0:
                start = max(0, pos - 4096)
                f.seek(start)
                idx = f.read(pos - start).rfind(_NL_BYTE)
                if idx != -1:
                    keep = start + idx + 1
                    break
                pos = start
            f.truncate(keep)
        logger.warning(f"JsonStore: truncated {size - keep} byte(s) of an unfinished write at the end of {self._path}")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def replay(self) -> Iterator[OrderEvent]:
        if not self._path.exists():
            return iter(())
        return self._replay()

    def _replay(self) -> Iterator[OrderEvent]:
        with self._path.open("r", encoding="utf-8", newline="") as f:
            lines = iter(f)
            line_no = 0
            pending = next(lines, None)
            while pending is not None:
                line_no += 1
                raw, pending = pending, next(lines, None)
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = event_from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    if pending is None and not raw.endswith(_NL):
                        logger.warning(f"JsonStore: ignoring unfinished last line {self._path}:{line_no} ({exc})")
                        return
                    raise ValueError(f"{self._path}:{line_no}: corrupt event line — {exc}") from exc
                yield event
