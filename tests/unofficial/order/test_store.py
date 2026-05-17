"""Tests for ``ibtws.unofficial.order.store``."""

from __future__ import annotations

import pytest

from ibtws.unofficial.order import (
    Cancelled,
    Filled,
    JsonStore,
    PositionChanged,
    Rejected,
    RequestSubmitted,
    StatusChanged,
)


def _all_events() -> list:
    contract = {"conId": 1, "symbol": "X", "secType": "STK"}
    return [
        RequestSubmitted(
            uuid="u1",
            request_kind="limit",
            contract=contract,
            side="BUY",
            quantity=1.0,
            tif="DAY",
            account=None,
            extra={"limit_price": 100.0},
        ),
        StatusChanged(uuid="u1", perm_id=1, state="Submitted", filled=0, remaining=1, avg_fill_price=0),
        Filled(uuid="u1", perm_id=1, exec_id="e1", price=100.0, quantity=1.0),
        Cancelled(uuid="u2", perm_id=2),
        Rejected(uuid="u3", perm_id=3, reason="Inactive"),
        PositionChanged(account="DU1", contract=contract, quantity=1.0, avg_cost=100.0),
    ]


async def test_append_and_replay_roundtrip(tmp_path):
    store = JsonStore(tmp_path / "log.jsonl", fsync=False)
    events = _all_events()
    for ev in events:
        await store.append(ev)

    replayed = list(store.replay())
    assert len(replayed) == len(events)
    for orig, rep in zip(events, replayed):
        assert type(orig) is type(rep)
        assert orig == rep


def test_replay_missing_file_is_empty(tmp_path):
    store = JsonStore(tmp_path / "absent.jsonl", fsync=False)
    assert list(store.replay()) == []


def test_replay_corrupt_line_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "cancelled", "uuid": "u", "perm_id": 1, "timestamp": 0}\nnot json at all\n')
    store = JsonStore(path, fsync=False)
    with pytest.raises(ValueError, match=":2:"):
        list(store.replay())


def test_replay_unknown_kind_raises(tmp_path):
    path = tmp_path / "unk.jsonl"
    path.write_text('{"kind": "weird_thing"}\n')
    store = JsonStore(path, fsync=False)
    with pytest.raises(ValueError, match="Unknown event kind"):
        list(store.replay())


async def test_append_preserves_order(tmp_path):
    store = JsonStore(tmp_path / "ord.jsonl", fsync=False)
    for i in range(5):
        await store.append(Cancelled(uuid=f"u{i}", perm_id=i))
    replayed = list(store.replay())
    assert [e.uuid for e in replayed] == [f"u{i}" for i in range(5)]
