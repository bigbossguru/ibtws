# ibtws — Module Reference

A resilient async Python layer over `ib_async` for IBKR TWS/Gateway. This
document is a per-module reference covering public surface, behaviour, and
key tunables. For runnable end-to-end usage see `examples/`.

## Architectural layers

```
ibtws/
├── config.py                     # IBKRConfig — single tunable dataclass
├── official/                     # placeholder (no impl yet)
└── unofficial/                   # production layer
    ├── client.py                 # IBKRClient — connection + reconnect + watchdog
    ├── _pacing.py                # ThrottledExecutor — shared rate-limit slot
    ├── _ib_errors.py             # classify(code) → ErrorCategory
    ├── option/                   # chain definition + quote snapshots + IV rank
    │   ├── chains.py
    │   ├── iv_rank.py
    │   ├── models.py
    │   └── utils.py
    ├── order/                    # placement, tracking, audit log, reconciliation
    │   ├── manager.py            # OrderManager — orchestrator
    │   ├── monitor.py            # event bus
    │   ├── store.py              # JsonStore — append-only audit log
    │   ├── reconciler.py         # IB-vs-local startup diff
    │   ├── factory.py            # build_market / build_limit / build_stop / build_bracket
    │   ├── models.py             # requests, events, runtime dataclasses
    │   └── utils.py              # validate_request, make_order_ref, is_paper_account
    └── strategies/
        └── credit_spread.py      # vertical credit-spread strategy (BAG combo)
```

**Dependency direction (clean):** strategies → order/option → client → ib_async.
Lower layers never import upward.

---

## `ibtws.config`

Single dataclass holding every tunable: connection, reconnect, watchdog,
quote-freshness, leg-mismatch behaviour, startup-fetch flags.

### `IBKRConfig`

| Field | Default | Notes |
|---|---|---|
| `host` | `"127.0.0.1"` | TWS / IB Gateway host |
| `port` | `7497` | 7497 paper TWS, 7496 live TWS, 4002 paper GW, 4001 live GW |
| `client_id` | `1` | Each concurrent connection needs a unique id |
| `readonly` | `False` | True blocks order submission at the API level |
| `account` | `""` | Specific account when the session has multiple |
| `connect_timeout` | `10.0` | Seconds for `connectAsync` to complete |
| `request_timeout` | `30.0` | Default per-request timeout |
| `reconnect_base_delay` | `2.0` | Doubles each attempt |
| `reconnect_max_delay` | `120.0` | Cap on backoff |
| `reconnect_max_attempts` | `0` | 0 = retry forever |
| `watchdog_interval` | `30.0` | Ping cadence |
| `watchdog_enabled` | `True` | Detect silent half-open sockets |
| `quote_max_age_s` | `5.0` | Drop option quotes older than this |
| `stale_quote_circuit_break` | `5` | Polls before strategies escalate |
| `auto_flatten_on_mismatch` | `True` | Flatten orphan legs on `LegMismatch` (when wired by caller) |
| `fetch_fields` | `POSITIONS \| ORDERS_OPEN \| ORDERS_COMPLETE \| ACCOUNT_UPDATES \| EXECUTIONS` | Startup state to pre-fetch |

Pure data; no I/O on construction.

---

## `ibtws.unofficial._pacing`

Shared concurrency + rate-limit primitive. Both `OrderManager` and
`OptionChainFetcher` accept an `executor` so a single bucket can govern the
aggregate request rate (keeping the session under IB's ~50 msg/s ceiling).

### `ThrottledExecutor`

```python
ThrottledExecutor(*, max_concurrency: int, pace_per_sec: float)
```

- `max_concurrency` — in-flight call cap. Raises `ValueError` if ≤ 0.
- `pace_per_sec` — minimum per-acquisition rate. `0` disables pacing (semaphore still applies).

| Method | Purpose |
|---|---|
| `async slot()` (asynccontextmanager) | Acquire concurrency + pacing slot; releases on exit |
| `min_interval` (property) | Seconds between slot acquisitions; 0 when pacing off |

```python
executor = ThrottledExecutor(max_concurrency=25, pace_per_sec=40.0)
async with executor.slot():
    await ib.reqTickersAsync(...)
```

Pacing bucket is `asyncio.Lock`-protected; safe to share across coroutines.

---

## `ibtws.unofficial._ib_errors`

Maps raw IBKR numeric codes into actionable buckets so callers can react
differently to pacing vs connection vs market-data vs order errors.

### `ErrorCategory` (str Enum)

`CONNECTION`, `MARKET_DATA`, `ORDER`, `PACING`, `INFO`, `UNKNOWN`.

### `classify(error_code: int) -> ErrorCategory`

- Specific codes win over ranges (e.g. `100` → `PACING`, `354` → `MARKET_DATA`, `1100` → `CONNECTION`, `2104` → `INFO`).
- Range fallbacks: `1100–1300` = `CONNECTION`, `2100–2200` = `INFO`,
  `300–399` = `MARKET_DATA`, `200–299` / `400–499` = `ORDER`.
- Anything else → `UNKNOWN`.

Pure lookup, no state. Reference:
<https://interactivebrokers.github.io/tws-api/message_codes.html>.

---

## `ibtws.unofficial.client`

### `IBKRClient`

Lifecycle wrapper around `ib_async.IB` with auto-reconnect, watchdog and
structured error logging. Raw `IB` instance exposed as `.ib` for direct use
of the full ib_async API.

```python
IBKRClient(
    config: IBKRConfig,
    *,
    on_connected:    Callable[[IBKRClient], None] | None = None,
    on_disconnected: Callable[[IBKRClient], None] | None = None,
    on_error:        Callable[[int, int, str, str], None] | None = None,
)
```

Hooks run on the event-loop thread; exceptions are logged and swallowed so
they cannot break the lifecycle. No network I/O at construction.

| Method | Purpose |
|---|---|
| `is_connected` (property) | Pass-through to `ib.isConnected()` |
| `async connect()` | Open socket + complete startup sync. Raises `ConnectionError` (chains `__cause__`) |
| `async disconnect()` | Idempotent. Sets `_shutting_down` *first* so the sync `disconnectedEvent` won't schedule a new reconnect. Cancels and awaits watchdog/reconnect tasks before closing |
| `async qualify(*contracts) -> list[Contract]` | Wraps `qualifyContractsAsync`; raises `ValueError` on length mismatch or missing `conId` |
| `async __aenter__/__aexit__` | Context manager does NOT auto-connect. `__aexit__` always calls `disconnect()` |
| `run_sync(coro)` | Spins up a fresh event loop, connects, runs `coro`, disconnects in `finally`. Do not call inside an existing loop |

**Reconnect:** delay doubles each attempt up to `reconnect_max_delay`, with
±10 % jitter. Resets counter on success. Honours `reconnect_max_attempts`
(0 = forever).

**Watchdog:** pings via `reqCurrentTimeAsync` every `watchdog_interval`
with a fixed 10 s timeout. On failure: forces `ib.disconnect()` so the
disconnect handler will reconnect — catches NAT timeouts and half-open
sockets that don't fire `disconnectedEvent`.

**Errors:** every `errorEvent` is routed through `classify()` and logged at
a level appropriate to the category (DEBUG for INFO, WARNING for
CONNECTION/MARKET_DATA/PACING, ERROR for ORDER and unknown).

```python
async with IBKRClient(cfg) as client:
    await client.connect()
    positions = await client.ib.reqPositionsAsync()
```

---

## `ibtws.unofficial.option`

Public re-exports: `ChainDefinition`, `OptionQuote`, `OptionChainFetcher`,
`IVRankCalculator`, `IVRankResult`, `quotes_to_dataframe`, `DATAFRAME_COLUMNS`.

### `option.models`

- **`ChainDefinition`** (frozen) — option universe for one underlying:
  `underlying_conId/symbol`, `trading_class`, `multiplier`, `exchange`,
  `expirations: tuple[str, ...]` (YYYYMMDD), `strikes: tuple[float, ...]`.
- **`OptionQuote`** (mutable) — single-contract snapshot: `contract: Option`,
  `bid/ask/volume/open_interest`, greeks `iv/delta/gamma/vega/theta`,
  `underlying_price`, `timestamp`. Every metric is `Optional[float]`;
  `None` means "IB returned no data" — never silently coerced to 0.

### `option.utils`

| Public helper | Purpose |
|---|---|
| `is_quote_fresh(ticker, max_age_s, *, now=None) -> bool` | True iff ticker's timestamp is within `max_age_s` of `now`. Missing timestamp → not fresh. `max_age_s<=0` disables (always True) |
| `quotes_to_dataframe(quotes) -> pd.DataFrame` | Flatten quotes; returns empty DataFrame with `DATAFRAME_COLUMNS` when input empty |
| `DATAFRAME_COLUMNS` | Canonical column order |

### `option.chains.OptionChainFetcher`

Throttled, fault-tolerant snapshot fetcher built on `IBKRClient`. Does NOT
own the connection — caller handles `connect()` / `disconnect()`.

```python
OptionChainFetcher(
    client: IBKRClient,
    *,
    max_concurrency:   int = 25,
    pace_per_sec:      float = 40.0,
    snapshot_timeout:  float = 20.0,
    quote_max_age_s:   float = 0.0,   # >0 enables the freshness filter
    executor:          ThrottledExecutor | None = None,
)
```

| Method | Purpose |
|---|---|
| `async fetch_chain_definition(underlying, *, exchange="SMART", trading_class=None) -> ChainDefinition` | Returns the (expirations × strikes) universe for one underlying. Raises `ValueError` if unqualified, `LookupError` if no params match |
| `async fetch_snapshot(underlying, *, rights=("C","P"), expirations=None, expiry_from/to=None, strikes=None, strike_from/to=None, strike_window_pct=0.2, batch_size=50, as_dataframe=False, ...) -> list[OptionQuote] \| DataFrame` | Resolve + quote a slice of the chain. **Fault-tolerant**: qualify or snapshot failures are logged at WARNING and excluded — caller always gets a partial answer. Auto-windows strikes around spot when no explicit strikes given |
| `release(contracts) -> int` | Cancel `reqMktData` subscriptions opened by previous snapshots. Returns count actually cancelled. **Important** — IB caps simultaneous market-data lines (~100); long-running strategies must release to avoid exhausting the quota |
| `subscribed_count` (property) | Number of contracts currently subscribed |

Quotes older than `quote_max_age_s` are dropped before returning (DEBUG log
of the dropped count). Backwards-compat shims (`_min_interval`,
`_await_next_slot`, `_slot`) delegate to the executor.

### `option.iv_rank.IVRankCalculator`

IV Rank / IV Percentile from IB's daily `OPTION_IMPLIED_VOLATILITY` series.

```python
IVRankCalculator(client, *, request_timeout=60.0)
async calculate(underlying, *, lookback_days=252, end_datetime="", use_rth=True) -> IVRankResult
```

`IVRankResult` fields: `current_iv`, `min_iv`, `max_iv`, `iv_rank` (0–100
or `None`), `iv_percentile` (0–100 or `None`), `sample_size`, `lookback_days`,
`as_of`, `underlying_symbol`.

- `iv_rank` is `None` when max == min (degenerate window).
- `iv_percentile` is `None` for single-bar history (never silently 0).
- Percentile excludes the current bar (strictly-below over history).
- Failed history requests are logged and treated as empty (all-None result).

---

## `ibtws.unofficial.order`

The only stateful subsystem. Everything besides `OrderManager` is a pure
dataclass, pure function, or thin I/O wrapper.

### `order.models`

Pure data — no I/O, no IB calls. Symmetric (de)serialisation:
`event.to_dict()` ↔ `event_from_dict(data)`.

**Enums.** `OrderSide` (BUY/SELL), `TimeInForce` (DAY/GTC/IOC/FOK),
`OrderState` (PendingSubmit, Submitted, Filled, Cancelled, Rejected, Inactive).

**Helpers.** `serialise_contract(contract) -> dict` — JSON-safe Contract
flattener used everywhere events touch disk.

**Frozen request dataclasses.** `MarketRequest`, `LimitRequest`,
`StopRequest`, `BracketRequest` (entry + TP + SL, OCA-grouped).

**Runtime.**
- `TrackedOrder` (mutable) — live view of one submitted order, kept in sync
  by the manager.
- `PositionSnapshot` (frozen) — point-in-time position record.
- `PositionPnL` (frozen) — on-demand mark-to-market. `market_price` /
  `market_value` / `unrealized_pnl` are `Optional` — `None` means "quote
  unavailable", never `0.0`.

**Events** (all frozen, all with `to_dict()`):
`RequestSubmitted`, `StatusChanged`, `Filled`, `Cancelled`, `Rejected`,
`PositionChanged`, `LegMismatch`. Unioned as `OrderEvent`.

`event_from_dict(data)` reverses `to_dict()`; raises `ValueError` on unknown
discriminator.

### `order.factory`

Builders that produce request dataclasses (validated in-line) and translate
them into raw `ib_async.Order` objects.

| Function | Purpose |
|---|---|
| `build_market(contract, side, qty, *, tif=DAY, account=None, outside_rth=False)` | `MarketRequest` |
| `build_limit(contract, side, qty, limit_price, *, ...)` | `LimitRequest` |
| `build_stop(contract, side, qty, stop_price, *, ...)` | `StopRequest` |
| `build_bracket(contract, side, qty, *, take_profit_price, stop_loss_price, entry_limit_price=None, ...)` | `BracketRequest` (`entry_limit_price=None` → market entry) |
| `request_to_order(request, order_ref)` | Translate one request → `Order` |
| `bracket_to_orders(req, parent_ref, tp_ref, sl_ref, *, parent_order_id, oca_group)` | Three wired `Order`s: parent + TP + SL. Parent/TP `transmit=False`, SL `transmit=True` (atomic group transmit), shared `ocaGroup`, `ocaType=1` |

### `order.utils`

| Function | Purpose |
|---|---|
| `make_order_ref() -> str` | 32-char hex UUID used as `orderRef` |
| `is_paper_account(account_id) -> bool` | True iff starts with `DU` |
| `validate_request(request) -> None` | Raises `ValueError` with a precise message. Rejects: unsupported type, missing contract, unqualified non-BAG contract, non-`OrderSide`, non-positive qty, non-positive limit/stop/TP/SL (BAG combo limits are allowed signed/zero/negative since they encode net credit), bracket TP/SL geometry inconsistent with side |

### `order.store`

`OrderStore` is a `@runtime_checkable` `Protocol` — the minimal contract is
`async append(event)` + `replay() -> Iterator[OrderEvent]`. No update / no
delete / no query.

#### `JsonStore(path, *, fsync=True)`

Append-only JSONL audit log.

| Method | Purpose |
|---|---|
| `path` (property) | File path |
| `async append(event)` | Serialise, write, flush, optional `os.fsync`. Under `asyncio.Lock` |
| `replay() -> Iterator[OrderEvent]` | Stream events back from disk. Missing file = empty iterator (no error). Raises `ValueError` with `path:line_no` on corruption |

Parent directory must exist; file is created on first append.
`fsync=False` is ~10× faster but loses crash-safety — fine for tests.
Recovery model is pure replay; the file is never rewritten or truncated.

### `order.monitor`

Fan-out event bus. Supports both async-iterator and sync-callback styles.

| Method | Purpose |
|---|---|
| `publish(event)` | Enqueue and fire registered callbacks. Callback exceptions logged + swallowed — one bad subscriber cannot poison the bus |
| `async stream() -> AsyncIterator[OrderEvent]` | Backpressure-friendly queue consumer |
| `register(fn)` / `unregister(fn)` | Sync inline callback |

### `order.reconciler`

```python
async reconcile(client, store) -> ReconciliationReport
```

Startup divergence diff between IB (source of truth) and the local audit
log. Never mutates IB or store. Logs WARNINGs on `local_only` (we
thought we had it, IB doesn't) and `ib_only` (IB has an open order that we
never persisted — e.g. placed from TWS directly).

`ReconciliationReport` (frozen): `matched: list[str]`, `local_only:
list[str]`, `ib_only: list[Trade]`, `positions: list[PositionSnapshot]`.

Terminal states for the local fold = FILLED / CANCELLED / REJECTED.

### `order.manager.OrderManager`

The orchestrator. Persist-first ordering, paper-vs-live interlock,
exec-id dedup, and event fan-out all live here.

```python
OrderManager(
    client,
    store,
    *,
    allow_live:      bool = False,
    max_concurrency: int = 10,
    pace_per_sec:    float = 10.0,
    executor:        ThrottledExecutor | None = None,
)
```

#### Lifecycle

| Method | Behaviour |
|---|---|
| `async start() -> ReconciliationReport` | Binds IB events; raises `RuntimeError` if `managedAccounts` empty or primary account is live without `allow_live=True`. Rehydrates `TrackedOrder` for UUIDs that matched the reconciler. Seeds position cache + live `Contract` refs |
| `async stop()` | Unbinds IB events. Idempotent |

#### Placement (persist-first)

| Method | Notes |
|---|---|
| `async place(request) -> TrackedOrder` | Persists `RequestSubmitted` **before** `placeOrder`. Per-UUID `asyncio.Lock` guards mutations |
| `async place_bracket(request) -> [parent, tp, sl]` | Three orders, shared `bracket_group` |
| `async market(...)` / `limit(...)` / `stop_order(...)` / `bracket(...)` | One-call build + place |

`stop_order` is named to avoid shadowing the `stop()` lifecycle method.

#### Position management

| Method | Notes |
|---|---|
| `async close_position(con_id, *, kind="market", limit_price=None, cancel_working=True) -> TrackedOrder \| None` | Cancels working orders on same contract first; qualifies and backfills empty exchange; submits opposite-side order. Returns `None` for missing/zero position |
| `async refresh_positions() -> list[PositionChanged]` | Pull fresh from IB (positionEvent can lag — always refresh before flattening) |
| `async close_all_positions(*, kind="market", cancel_working=True) -> list[TrackedOrder]` | Refreshes then flattens every non-zero position |
| `async cancel(uuid)` | Idempotent. `KeyError` for unknown uuid |
| `async cancel_all() -> list[str]` | Skips terminal orders |

#### Read-only views

| | |
|---|---|
| `open_orders` | List of non-terminal `TrackedOrder` |
| `positions` | List of cached `PositionChanged` |
| `async current_pnl(con_ids=None, *, snapshot_timeout=5.0) -> list[PositionPnL]` | On-demand mark-to-market. Pricing rule: mid > last > close. `None` = unknown, never `0.0`. Skips zero-quantity positions |
| `events() -> AsyncIterator[OrderEvent]` | Stream from the monitor |
| `on_event(fn)` | Sync callback registration |

#### Key behaviour

- **Persist-first** — `RequestSubmitted` hits the store before `placeOrder`,
  so a crash between persist and submit is detectable by the reconciler
  (local_only entry that IB never saw → we know we never sent it).
- **Paper interlock** — refuses live primary account unless `allow_live=True`;
  `_check_account_safety` polices per-request `account` against
  `managedAccounts` (also blocks live sub-accounts on paper-primary FA setups).
- **Exec dedup** — `_seen_exec_ids` set drops replayed `execDetails` after
  reconnect so downstream TP/SL logic doesn't double-fire.
- **Status mapping** — `PendingCancel` → SUBMITTED (still live until a
  terminal state arrives). `Inactive` → emits `Rejected` and flips state
  to REJECTED.
- **Persist failures logged, not raised** (`_persist_safely`) — the event
  bus still gets the publish even if disk I/O fails.
- **Concurrency / pacing** — single `ThrottledExecutor`. Pass an `executor`
  shared with the option fetcher to govern aggregate rate.

---

## `ibtws.unofficial.strategies`

### `strategies.credit_spread`

Vertical credit-spread strategy (bull-put / bear-call) — discover → select →
place (atomic BAG combo) → monitor & exit. Routed entirely through
`OrderManager`, so it inherits persistence, reconciliation, the paper
interlock, and the event stream uniformly with single-leg orders.

#### Public surface

- `CreditSpreadError(RuntimeError)` — actionable: includes which constraint
  failed and the observed numbers.
- `SpreadType(str, Enum)` — `BULL_PUT`, `BEAR_CALL`. Properties: `.right`
  (`"P"`/`"C"`), `.is_bullish`.
- `CreditSpreadParams` (frozen) — all tunables, validated in `__post_init__`.
- `SpreadLeg` (frozen) — one side of a vertical: `quote: OptionQuote`,
  `action: OrderSide`. Props: `conId`, `strike`.
- `CreditSpreadPlan` (frozen) — fully resolved spread, ready to place. All
  cash figures are *per spread* in account currency. `risk_reward` property,
  `describe()` returns a one-line log summary.

#### `CreditSpreadParams` defaults

| Knob | Default | Purpose |
|---|---|---|
| `target_short_delta` | `0.30` | `|Δ|` target for short leg |
| `wing_width` | `5.0` | Strike distance ($) — selector snaps to nearest available |
| `target_dte` / `dte_tolerance` | `30` / `14` | Pick expiry closest to target within tolerance |
| `max_short_delta` | `0.50` | Hard cap on chosen short `|Δ|`; `None` to disable |
| `min_credit` / `min_credit_width_ratio` | `None` / `None` | Economic floors |
| `min_open_interest` / `min_volume` | `0` / `0` | Liquidity filters (legs with `None` are kept) |
| `quantity` | `1` | Number of spreads |
| `limit_slippage` | `0.05` | Fraction below mid the entry limit can sit |
| `take_profit_pct` | `0.5` | Close at 50 % of credit captured |
| `stop_loss_multiplier` | `2.0` | Close on loss = N × credit (capped at width) |
| `tif` / `account` / `outside_rth` | DAY / `None` / `False` | Order knobs |
| `exchange` / `currency` / `trading_class` | `"SMART"` / `"USD"` / `None` | Universe selector (e.g. `"SPXW"` for PM-settled SPX dailies) |
| `expirations` / `expiry_from` / `expiry_to` | `None` | Pre-filters on the chain |
| `strike_window_pct` | `0.10` | ±10 % of spot bounds the snapshot |

#### Pure selectors (unit-testable)

| Function | Behaviour |
|---|---|
| `select_expiry(expirations, *, target_dte, dte_tolerance, now=None) -> str` | Closest expiry inside tolerance; raises `CreditSpreadError` if none qualifies. Ignores negative-DTE entries |
| `select_short_leg(quotes, *, target_short_delta, max_short_delta, min_open_interest, min_volume) -> OptionQuote` | Tradeability filter + closest `|Δ|`. Rejects if max-delta ceiling leaves nothing |
| `select_long_leg(quotes, *, short, wing_width, spread_type, ...) -> OptionQuote` | Snaps to nearest strike at or beyond requested width on the protective side. Refuses widths < 50 % of requested (chain too narrow) |

#### `CreditSpreadStrategy`

```python
CreditSpreadStrategy(
    client: IBKRClient,
    order_manager: OrderManager,
    *,
    fetcher:   OptionChainFetcher | None = None,
    tick_size: float = 0.05,
)
```

| Method | Purpose |
|---|---|
| `async build_plan(params) -> CreditSpreadPlan` | Qualify underlying → fetch chain → pick expiry → snapshot relevant right → select legs → enforce economic constraints |
| `async place(plan, *, limit_credit=None) -> TrackedOrder` | Submits as `BUY @ -credit` signed combo limit via `OrderManager.limit`. Rounds to tick |
| `async close(plan, *, limit_debit=None, tif=None) -> TrackedOrder` | BUY-back BAG limit. **Releases the two market-data subscriptions in `finally`** to free IB quota |
| `async monitor_and_exit(plan, entry, *, poll_interval=15.0, max_wait=None) -> TrackedOrder \| None` | Wait for entry fill, poll mid-debit, fire close on TP / SL. Tolerates quote drop-outs (continues polling). Returns `None` on timeout or pre-fill cancel |

#### Combo sign convention

The BAG is submitted with `action="BUY"` and a **signed net cost** as the
limit price: negative = credit collected, positive = debit paid. So a $0.45
credit is sent as `BUY @ -0.45`. This matches TWS combo display and avoids
the SELL/+price sign-flip ambiguity that bites SMART-routed combos. Leg
directions live in `plan.bag.comboLegs` (SELL short, BUY long).

```python
async with IBKRClient(cfg) as client:
    await client.connect()
    om = OrderManager(client, JsonStore("orders.jsonl"))
    await om.start()
    fetcher = OptionChainFetcher(client, quote_max_age_s=5.0)
    strat = CreditSpreadStrategy(client, om, fetcher=fetcher)

    plan = await strat.build_plan(
        CreditSpreadParams(
            underlying=Index("SPX", "CBOE", "USD"),
            spread_type=SpreadType.BULL_PUT,
            target_short_delta=0.10,
            wing_width=10.0,
            target_dte=0,
        )
    )
    tracked = await strat.place(plan)
    await strat.monitor_and_exit(plan, tracked, poll_interval=15.0)
```

---

## `ibtws.official`

`official/client.py` is a placeholder. No public surface yet — reserved for
a future officially-sanctioned client. Use `ibtws.unofficial.*` for all
real work.

---

## Cross-cutting design notes

### Persistence model — pure replay
`JsonStore` is append-only. There is no UPDATE / DELETE / query — recovery
is `replay()` then fold to current state. This means corruption is
detectable (replay raises on bad line) and writes are crash-safe under
`fsync=True`.

### Persist-first ordering
`OrderManager` always persists `RequestSubmitted` **before** calling
`placeOrder`. A crash in the window between persist and submit leaves a
local entry the reconciler will classify as `local_only` — you know the
order was never sent, and can decide whether to re-submit or drop it.

### Idempotency
- `cancel(uuid)` — terminal orders are a no-op (warn-log).
- `disconnect()` — safe to call multiple times.
- `JsonStore.replay()` — pure read, no side effects.
- `OptionChainFetcher.release()` — unknown contracts silently ignored.

### Fault tolerance
- `fetch_snapshot` — failed qualify or snapshot batches are logged at
  WARNING and excluded. Caller always gets a partial answer.
- `_persist_safely` — store failures are logged, never raised, so the event
  bus still publishes.
- `_handle_error` — IB error codes never crash the client; they route
  through `classify()` and pick an appropriate log level.

### Rate limiting
A single `ThrottledExecutor` can be shared between `OrderManager` and
`OptionChainFetcher` so the aggregate request rate respects IB's ~50 msg/s
ceiling. Each subsystem can still own its own concurrency cap.

### Market-data subscription quota
IB caps simultaneous market-data lines (~100 / session). Long-running
strategies that build and close many spreads must call
`fetcher.release(contracts)` after a position is closed.
`CreditSpreadStrategy.close()` does this automatically in `finally`.
