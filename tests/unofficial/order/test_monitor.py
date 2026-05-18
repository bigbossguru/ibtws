"""Tests for ``ibtws.unofficial.order.monitor``."""

from __future__ import annotations

import asyncio

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
