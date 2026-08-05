# Order Module

## Overview

The `ibtws.unofficial.order` package handles the full lifecycle of IBKR orders:
placement, tracking, persistence, reconciliation, and position management. It is
the only code path that talks to `ib.placeOrder` — strategies and application
code interact exclusively through `OrderManager`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application / Strategy                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                    OrderManager (orchestrator)
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    OrderStore         OrderMonitor        IBKRClient
   (persistence)       (event bus)        (IB connection)
```

- **OrderManager** — the single stateful class. Places/cancels orders, binds to
  IB events, persists state transitions, publishes events, reconciles on startup.
- **OrderStore** — append-only audit log (Protocol). Ships with `JsonStore` (JSONL file).
- **OrderMonitor** — fan-out event bus (async stream + sync callbacks).
- **Factory** — pure functions that build request dataclasses and translate them to `ib_async.Order`.

## Quick Start

```python
from pathlib import Path
from ib_async import Stock
from ibtws.config import IBKRConfig
from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.order import (
    JsonStore, OrderManager, OrderSide, TimeInForce
)

async def main():
    config = IBKRConfig(port=7497, client_id=1)
    store = JsonStore(Path("orders.jsonl"))

    async with IBKRClient(config) as client:
        await client.connect()

        contract = Stock("AAPL", "SMART", "USD")
        [contract] = await client.ib.qualifyContractsAsync(contract)

        manager = OrderManager(client, store)
        report = await manager.start()

        # Place a market order
        tracked = await manager.market(contract, OrderSide.BUY, 10)
        print(f"uuid={tracked.uuid} state={tracked.state}")

        # Place a limit order
        tracked = await manager.limit(contract, OrderSide.BUY, 10, 150.0)

        # Cancel
        await manager.cancel(tracked.uuid)

        # Flatten all positions
        await manager.close_all_positions()

        await manager.stop()
```

## Safety Interlocks

### Paper-vs-Live Guard

`OrderManager` refuses to start on a live account unless `allow_live=True` is
explicitly passed. Paper accounts are identified by the `DU` prefix (IB's
"Demo User" convention).

```python
# Paper (default) — works on DU* accounts
manager = OrderManager(client, store)

# Live — explicit opt-in required
manager = OrderManager(client, store, allow_live=True)
```

### Per-Request Account Routing

Even on a paper-primary session, individual requests can target sub-accounts
(FA setups). The manager refuses to route to a live sub-account unless
`allow_live=True`.

## Order Types

### Market

```python
tracked = await manager.market(contract, OrderSide.BUY, quantity=5)
```

### Limit

```python
tracked = await manager.limit(
    contract, OrderSide.BUY, quantity=5, limit_price=150.0,
    tif=TimeInForce.GTC, outside_rth=True
)
```

### Stop

```python
tracked = await manager.stop_order(
    contract, OrderSide.SELL, quantity=5, stop_price=140.0
)
```

### Bracket (Entry + Take-Profit + optional Stop-Loss)

```python
# Full bracket: entry + TP + SL
parent, tp, sl = await manager.bracket(
    contract, OrderSide.BUY, quantity=5,
    entry_limit_price=150.0,   # None → market entry
    take_profit_price=170.0,
    stop_loss_price=140.0,     # omit / None → TP-only bracket
    tif=TimeInForce.GTC,
)
# parent, tp, sl share parent.bracket_group

# TP-only bracket: entry + TP (no stop-loss)
parent, tp = await manager.bracket(
    contract, OrderSide.BUY, quantity=5,
    entry_limit_price=150.0,
    take_profit_price=170.0,
)
```

IB wiring: children have `parentId` pointing to the parent. When a stop-loss is
present, TP/SL share an OCA group (cancel-with-block: when one fills, IB cancels
the other) and the SL transmits the group; for a TP-only bracket the TP itself
transmits the group.

For **combo (BAG)** contracts the bracket prices are *signed net* values
(negative = credit): e.g. a credit spread enters `BUY BAG @ -credit` with a
take-profit `SELL BAG @ -tp_debit`. Positivity and TP/SL geometry validation is
skipped for BAG contracts since those checks assume single-leg positive prices.

### BAG / Combo Orders

Combo orders (e.g., credit spreads) use a `Bag` contract with `comboLegs`.
The limit price is a signed net cost:

```python
from ib_async import Bag, ComboLeg

bag = Bag(symbol="AAPL", currency="USD", exchange="SMART")
bag.comboLegs = [
    ComboLeg(conId=short_con_id, ratio=1, action="SELL", exchange="SMART"),
    ComboLeg(conId=long_con_id, ratio=1, action="BUY", exchange="SMART"),
]

# Open: BUY @ negative = collect credit
tracked = await manager.limit(bag, OrderSide.BUY, 1, -0.45)

# Close: SELL @ negative = pay debit (IB reverses leg actions)
tracked = await manager.limit(bag, OrderSide.SELL, 1, -0.20)
```

## Request Builders

For cases where you want to inspect or modify a request before submitting:

```python
from ibtws.unofficial.order import build_limit, build_market, build_stop, build_bracket

req = build_limit(contract, OrderSide.BUY, 5, 150.0, tif=TimeInForce.GTC)
# inspect req...
tracked = await manager.place(req)
```

Builders validate immediately (quantity > 0, prices positive for single-leg,
contract qualified, bracket geometry correct).

## TrackedOrder

Every submitted order returns a `TrackedOrder` — a mutable view kept in sync
by the manager's IB event handlers:

```python
@dataclass
class TrackedOrder:
    uuid: str                    # unique orderRef (32 hex chars)
    request: OrderRequest | None # original request (None after restart)
    trade: Any                   # ib_async.Trade
    state: OrderState            # PENDING_SUBMIT → SUBMITTED → FILLED/CANCELLED/REJECTED
    filled: float
    remaining: float
    avg_fill_price: float
    perm_id: int
    bracket_group: str | None
    last_update: float
```

## Order States

```
PENDING_SUBMIT → SUBMITTED → FILLED
                           → CANCELLED
                           → REJECTED
                           → INACTIVE (treated as REJECTED)
```

IB's `PendingCancel` and `PreSubmitted` are mapped to `SUBMITTED` (the order
is still live until a terminal status arrives).

## Cancellation

```python
# Single order (idempotent — terminal orders are no-ops)
await manager.cancel(tracked.uuid)

# All non-terminal tracked orders
cancelled_uuids = await manager.cancel_all()
```

## Position Management

```python
# Refresh from IB (positionEvent can lag fills by seconds)
positions = await manager.refresh_positions()

# Close one position by conId
await manager.close_position(con_id=12345, kind="market")
await manager.close_position(con_id=12345, kind="limit", limit_price=150.0)

# Flatten everything (refreshes first, cancels working orders per contract)
closed = await manager.close_all_positions()
```

## Unrealized PnL

```python
pnls = await manager.current_pnl()
# Or filter to specific contracts:
pnls = await manager.current_pnl(con_ids=[12345, 67890])

for p in pnls:
    # p.market_price / p.unrealized_pnl are None when no quote available
    print(f"{p.contract['symbol']} qty={p.quantity} uPnL={p.unrealized_pnl}")
```

Pricing priority: mid(bid/ask) → last → close. `None` means "unknown" (not zero).

## Event System

### Sync Callbacks

```python
def on_event(event: OrderEvent) -> None:
    print(f"{type(event).__name__}: {event}")

manager.on_event(on_event)
```

### Async Stream

```python
async for event in manager.events():
    if isinstance(event, Filled):
        print(f"Fill: {event.uuid} @ {event.price}")
```

Note: the stream uses a single queue — only one consumer can use `events()`.
For multiple consumers, use `on_event()` callbacks.

### Event Types

| Event | Fields | When |
|-------|--------|------|
| `RequestSubmitted` | uuid, contract, side, quantity, tif, extra | Order persisted + sent to IB |
| `StatusChanged` | uuid, state, filled, remaining, avg_fill_price | Any IB status update |
| `Filled` | uuid, exec_id, price, quantity | Each execution (partial or full) |
| `Cancelled` | uuid, perm_id | Order cancelled |
| `Rejected` | uuid, reason | Order rejected or inactive |
| `PositionChanged` | account, contract, quantity, avg_cost | Position update from IB |
| `LegMismatch` | uuid, expected, actual, detail | Combo/bracket position mismatch |

## Persistence

### Audit Log

Every event is appended to the `OrderStore` as a single JSON line, flushed and
fsync'd before the write returns. A crash loses at most the line being written.

```python
store = JsonStore("orders.jsonl", fsync=True)  # fsync=False for tests

# Replay the full history
for event in store.replay():
    print(event)
```

### Custom Backends

Implement the `OrderStore` protocol:

```python
class OrderStore(Protocol):
    async def append(self, event: OrderEvent) -> None: ...
    def replay(self) -> Iterator[OrderEvent]: ...
```

## Reconciliation

On `manager.start()`, the reconciler diffs IB's open orders against the local
audit log:

```python
report = await manager.start()
report.matched      # UUIDs open in both IB and local log
report.local_only   # Open locally but missing from IB (crashed before submit?)
report.ib_only      # In IB but unknown locally (submitted from TWS UI?)
report.positions    # Current position snapshots
```

Matched orders are rehydrated into `TrackedOrder` objects so `cancel()` and
`open_orders` work immediately after restart.

## Rate Limiting

The manager uses a `ThrottledExecutor` (token-bucket + semaphore) to stay under
IB's ~50 msg/s ceiling. Default: 10 msg/s with 10 concurrent slots.

```python
from ibtws.unofficial._pacing import ThrottledExecutor

# Share one executor across order manager and chain fetcher
executor = ThrottledExecutor(max_concurrency=10, pace_per_sec=40.0)
manager = OrderManager(client, store, executor=executor)
```

## Lifecycle

```python
manager = OrderManager(client, store)
report = await manager.start()   # bind events, reconcile, start persist worker
# ... use manager ...
await manager.stop()             # unbind events, drain persist queue, stop worker
```

`stop()` guarantees all pending events are flushed to disk before returning.
