"""Tests for ``ibtws.unofficial.order.reconciler``."""

from __future__ import annotations

from ibtws.unofficial.order import Cancelled, RequestSubmitted, StatusChanged
from ibtws.unofficial.order.reconciler import reconcile

from .conftest import make_contract, make_position, make_trade


def _submitted(uuid: str) -> RequestSubmitted:
    return RequestSubmitted(
        uuid=uuid,
        request_kind="market",
        contract={"conId": 1},
        side="BUY",
        quantity=1.0,
        tif="DAY",
        account=None,
        extra={},
    )


async def test_reconcile_matched(fake_client, tmp_store):
    await tmp_store.append(_submitted("u-match"))
    fake_client.ib.reqOpenOrdersAsync.return_value = [make_trade("u-match")]

    report = await reconcile(fake_client, tmp_store)

    assert report.matched == ["u-match"]
    assert report.local_only == []
    assert report.ib_only == []


async def test_reconcile_local_only(fake_client, tmp_store):
    await tmp_store.append(_submitted("u-local"))
    fake_client.ib.reqOpenOrdersAsync.return_value = []

    report = await reconcile(fake_client, tmp_store)

    assert report.local_only == ["u-local"]
    assert report.matched == []


async def test_reconcile_ib_only(fake_client, tmp_store):
    fake_client.ib.reqOpenOrdersAsync.return_value = [make_trade("u-mystery")]
    report = await reconcile(fake_client, tmp_store)

    assert report.matched == []
    assert len(report.ib_only) == 1
    assert report.ib_only[0].order.orderRef == "u-mystery"


async def test_reconcile_terminal_state_excluded_from_open(fake_client, tmp_store):
    await tmp_store.append(_submitted("u-done"))
    await tmp_store.append(
        StatusChanged(uuid="u-done", perm_id=1, state="Filled", filled=1, remaining=0, avg_fill_price=100)
    )
    fake_client.ib.reqOpenOrdersAsync.return_value = []

    report = await reconcile(fake_client, tmp_store)

    assert report.matched == []
    assert report.local_only == []


async def test_reconcile_positions(fake_client, tmp_store):
    fake_client.ib.reqPositionsAsync.return_value = [
        make_position(account="DU1", contract=make_contract(con_id=42), quantity=5, avg_cost=100)
    ]
    report = await reconcile(fake_client, tmp_store)

    assert len(report.positions) == 1
    assert report.positions[0].account == "DU1"
    assert report.positions[0].quantity == 5
    assert report.positions[0].contract["conId"] == 42


async def test_reconcile_cancelled_locally_excluded(fake_client, tmp_store):
    await tmp_store.append(_submitted("u-cx"))
    await tmp_store.append(
        StatusChanged(uuid="u-cx", perm_id=1, state="Cancelled", filled=0, remaining=1, avg_fill_price=0)
    )
    await tmp_store.append(Cancelled(uuid="u-cx", perm_id=1))
    fake_client.ib.reqOpenOrdersAsync.return_value = []

    report = await reconcile(fake_client, tmp_store)
    assert report.local_only == []
    assert report.matched == []
