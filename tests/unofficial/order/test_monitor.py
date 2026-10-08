"""Tests for ``ibtws.unofficial.order.monitor``."""

from __future__ import annotations

import asyncio

import pytest

from ibtws.unofficial.order import Cancelled, OrderMonitor


async def test_publish_fans_out_to_stream_and_callback():
    mon = OrderMonitor()
    seen: list = []
    mon.register(lambda e: seen.append(e))

    ev = Cancelled(uuid="u1", perm_id=1)
    mon.publish(ev)

    assert seen == [ev]

    stream = mon.stream()
    received = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
    assert received == ev


async def test_callback_exception_does_not_poison_bus(caplog):
    mon = OrderMonitor()
    seen: list = []

    def bad(_e):
        raise RuntimeError("boom")

    mon.register(bad)
    mon.register(lambda e: seen.append(e))

    ev = Cancelled(uuid="u", perm_id=1)
    mon.publish(ev)

    assert seen == [ev]
    received = await asyncio.wait_for(mon.stream().__anext__(), timeout=0.5)
    assert received == ev


def test_unregister_removes_callback():
    mon = OrderMonitor()
    seen: list = []
    fn = lambda e: seen.append(e)  # noqa: E731
    mon.register(fn)
    mon.unregister(fn)
    mon.publish(Cancelled(uuid="u", perm_id=1))
    assert seen == []


async def test_every_stream_receives_every_event():
    mon = OrderMonitor()
    a, b = mon.stream(), mon.stream()
    next_a = asyncio.ensure_future(a.__anext__())
    next_b = asyncio.ensure_future(b.__anext__())
    await asyncio.sleep(0)

    ev = Cancelled(uuid="u", perm_id=1)
    mon.publish(ev)

    assert await asyncio.wait_for(asyncio.gather(next_a, next_b), timeout=0.5) == [ev, ev]


def test_backlog_is_bounded_without_subscribers():
    mon = OrderMonitor(max_queue=3)
    for i in range(10):
        mon.publish(Cancelled(uuid=f"u{i}", perm_id=i))
    assert len(mon._backlog) == 3
    assert mon.dropped == 7


async def test_slow_subscriber_drops_oldest():
    mon = OrderMonitor(max_queue=2)
    stream = mon.stream()
    first = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0)
    mon.publish(Cancelled(uuid="u0", perm_id=0))
    assert (await first).uuid == "u0"

    for i in range(1, 5):
        mon.publish(Cancelled(uuid=f"u{i}", perm_id=i))
    got = [(await stream.__anext__()).uuid for _ in range(2)]
    assert got == ["u3", "u4"]
    assert mon.dropped == 2


async def test_close_ends_streams():
    mon = OrderMonitor()
    stream = mon.stream()
    pending = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0)

    mon.close()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, timeout=0.5)
