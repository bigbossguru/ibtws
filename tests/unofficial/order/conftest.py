"""Shared fakes/fixtures for ``ibtws.unofficial.order`` tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ibtws.unofficial.order import JsonStore


def make_contract(symbol: str = "AAPL", con_id: int = 265598, sec_type: str = "STK") -> SimpleNamespace:
    return SimpleNamespace(
        conId=con_id,
        symbol=symbol,
        secType=sec_type,
        exchange="SMART",
        currency="USD",
        strike=0.0,
        right="",
        lastTradeDateOrContractMonth="",
        tradingClass="",
        multiplier="",
    )


def make_order_status(
    status: str = "Submitted",
    *,
    filled: float = 0.0,
    remaining: float = 0.0,
    avg_fill_price: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        filled=filled,
        remaining=remaining,
        avgFillPrice=avg_fill_price,
    )


def make_trade(
    order_ref: str,
    *,
    perm_id: int = 1000,
    status: str = "Submitted",
    filled: float = 0.0,
    remaining: float = 0.0,
    avg_fill_price: float = 0.0,
) -> SimpleNamespace:
    order = SimpleNamespace(orderRef=order_ref, permId=perm_id)
    return SimpleNamespace(
        order=order,
        orderStatus=make_order_status(status, filled=filled, remaining=remaining, avg_fill_price=avg_fill_price),
    )


def make_fill(exec_id: str = "exec-1", *, price: float = 100.0, shares: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        execution=SimpleNamespace(execId=exec_id, price=price, shares=shares),
    )


def make_position(
    *,
    account: str = "DU123",
    contract=None,
    quantity: float = 1.0,
    avg_cost: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        account=account,
        contract=contract or make_contract(),
        position=quantity,
        avgCost=avg_cost,
    )


class _EventHook:
    """Minimal stand-in for ib_async event ``+=`` / ``-=`` semantics."""

    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def __isub__(self, fn):
        if fn in self.handlers:
            self.handlers.remove(fn)
        return self

    def fire(self, *args, **kwargs) -> None:
        for h in list(self.handlers):
            h(*args, **kwargs)


@pytest.fixture
def fake_client():
    """Stand-in IBKRClient for OrderManager tests."""
    from unittest.mock import AsyncMock

    ib = MagicMock()
    ib.managedAccounts = MagicMock(return_value=["DU123"])
    ib.client = MagicMock()
    ib.client.getReqId = MagicMock(side_effect=lambda: 10000)
    ib.orderStatusEvent = _EventHook()
    ib.execDetailsEvent = _EventHook()
    ib.positionEvent = _EventHook()
    ib.reqOpenOrdersAsync = AsyncMock(return_value=[])
    ib.reqPositionsAsync = AsyncMock(return_value=[])
    ib.openTrades = MagicMock(return_value=[])
    ib.placeOrder = MagicMock()
    ib.cancelOrder = MagicMock()
    ib.qualifyContractsAsync = AsyncMock()
    return SimpleNamespace(ib=ib)


@pytest.fixture
def tmp_store(tmp_path) -> JsonStore:
    return JsonStore(tmp_path / "orders.jsonl", fsync=False)
