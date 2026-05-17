# Non-official package. Not affiliated with ib_async upstream.

"""Startup reconciliation: trust IB, compare against the local audit log."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .models import OrderState, PositionSnapshot, RequestSubmitted, StatusChanged, serialise_contract
from .store import OrderStore

logger = logging.getLogger(__name__)


_TERMINAL_STATES = {OrderState.FILLED.value, OrderState.CANCELLED.value, OrderState.REJECTED.value}


@dataclass(frozen=True)
class ReconciliationReport:
    """Snapshot of state divergence between local store and IB at start-up."""

    matched: list[str] = field(default_factory=list)
    local_only: list[str] = field(default_factory=list)
    ib_only: list[Any] = field(default_factory=list)  # ib_async.Trade
    positions: list[PositionSnapshot] = field(default_factory=list)


async def reconcile(client: Any, store: OrderStore) -> ReconciliationReport:
    """Diff IB's open orders + positions against the local jsonl log.

    Returns a report — never mutates IB or the store. The manager rehydrates
    its in-memory ``TrackedOrder`` map from the ``matched`` set; ``ib_only``
    orders are surfaced as warnings (they exist in IB but we don't know what
    UUID they came from, e.g. submitted from the TWS UI or a prior client).
    """
    open_trades = await client.ib.reqOpenOrdersAsync()
    positions_raw = await client.ib.reqPositionsAsync()

    local_latest: dict[str, str] = {}
    for event in store.replay():
        uuid = getattr(event, "uuid", None)
        if uuid is None:
            continue
        if isinstance(event, RequestSubmitted):
            local_latest[uuid] = OrderState.PENDING_SUBMIT.value
        elif isinstance(event, StatusChanged):
            local_latest[uuid] = event.state

    local_open = {uuid for uuid, state in local_latest.items() if state not in _TERMINAL_STATES}
    ib_uuids = {t.order.orderRef for t in open_trades if getattr(t.order, "orderRef", None)}

    matched = sorted(local_open & ib_uuids)
    local_only = sorted(local_open - ib_uuids)
    ib_only = [t for t in open_trades if not getattr(t.order, "orderRef", None) or t.order.orderRef not in local_open]

    positions = [
        PositionSnapshot(
            account=p.account,
            contract=serialise_contract(p.contract),
            quantity=float(p.position),
            avg_cost=float(p.avgCost),
            timestamp=time.time(),
        )
        for p in positions_raw
    ]

    if local_only:
        logger.warning(f"reconcile: {len(local_only)} local-open orders missing from IB: {local_only}")
    if ib_only:
        refs = [getattr(t.order, "orderRef", None) or f"<no-ref permId={t.order.permId}>" for t in ib_only]
        logger.warning(f"reconcile: {len(ib_only)} IB-open orders unknown to local store: {refs}")
    logger.info(
        f"reconcile: matched={len(matched)} local_only={len(local_only)} "
        f"ib_only={len(ib_only)} positions={len(positions)}"
    )

    return ReconciliationReport(matched=matched, local_only=local_only, ib_only=ib_only, positions=positions)
