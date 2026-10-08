# ibtws — Module Reference

A thin, resilient async Python layer over `ib_async` for IBKR TWS/Gateway.
This document is a per-module reference covering public surface, behaviour, and
key tunables. For runnable end-to-end usage see `examples/`.

## Architectural layers

```
ibtws/
├── config.py                     # IBKRConfig — single tunable dataclass
└── unofficial/                   # production layer
    ├── client.py                 # IBKRClient — connect / market + historical data
    ├── helpers.py                # safe_pick_value / calc_dte / chunked
    ├── _pacing.py                # ThrottledExecutor — shared rate-limit slot
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
    ├── analysis/                 # pure analytics over chain / price DataFrames
    │   ├── gex.py                # GexCalculator — gamma exposure profile
    │   ├── expected_move.py      # ExpectedMoveCalculator
    │   ├── market_bias.py        # determine_market_bias
    │   └── volatility_risk.py    # common_volatility_risk
    └── strategies/
        └── credit_spread.py      # vertical credit-spread strategy (BAG combo)
```

**Dependency direction (clean):** strategies → order/option → client → ib_async.
Lower layers never import upward. The `analysis` package is pure (pandas / numpy /
scipy only) and does not import the client — it operates on DataFrames produced
by the option layer.

---

## `ibtws.config`

Single dataclass holding the connection + startup tunables.

### `IBKRConfig`

| Field | Default | Notes |
|---|---|---|
| `host` | `"127.0.0.1"` | TWS / IB Gateway host |
| `port` | `7497` | 7497 paper TWS, 7496 live TWS, 4002 paper GW, 4001 live GW |
| `client_id` | `1` | Each concurrent connection needs a unique id |
| `readonly` | `False` | True makes `OrderManager` refuse every place / cancel call |
| `account` | `""` | Specific account when the session has multiple |
| `connect_timeout` | `10.0` | Seconds for `connectAsync` to complete |
| `request_timeout` | `30.0` | Default per-request timeout (`ib.RequestTimeout`) |
| `fetch_fields` | `POSITIONS \| ORDERS_OPEN \| ORDERS_COMPLETE \| ACCOUNT_UPDATES \| EXECUTIONS` | `StartupFetch` bitmask of data to pull on connect |
| `auto_reconnect` | `True` | Reconnect automatically after an unexpected disconnect |
| `reconnect_initial_delay` / `reconnect_max_delay` | `1.0` / `60.0` | Exponential backoff bounds (seconds) |
| `reconnect_max_attempts` | `None` | Give up after N attempts; `None` retries until `disconnect()` |

Pure data; no I/O on construction.

`IBKRConfig.from_env(prefix="IBKR_", **overrides)` reads `{prefix}HOST`,
`{prefix}PORT`, `{prefix}CLIENT_ID`, `{prefix}ACCOUNT` and `{prefix}READONLY`
from the environment and applies `overrides` last, so scripts never need to
hard-code connection details (`IBKRConfig.from_env(prefix="TWS_")` works too).

---

## `ibtws.unofficial._pacing`

Shared concurrency + rate-limit primitive. `OrderManager` accepts an
`executor` so a single bucket can govern the aggregate request rate. Note that
ib_async already throttles every outgoing message to 45/s per connection, so
pacing here is only needed to slow a subsystem down further.

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
executor = ThrottledExecutor(max_concurrency=10, pace_per_sec=10.0)
async with executor.slot():
    await ib.reqTickersAsync(...)
```

Pacing bucket is `asyncio.Lock`-protected; safe to share across coroutines.

---

## `ibtws.unofficial.helpers`

Small stateless utilities shared across the unofficial layer.

| Function | Purpose |
|---|---|
| `safe_pick_value(obj, attr, *, allow_negative=False) -> float \| None` | Read a numeric attribute, scrubbing IB's `-1` / NaN "no data" sentinels. Pass `allow_negative=True` for fields that are legitimately negative (delta, theta) |
| `calc_dte(expiration) -> float` | Calendar days from today (exchange time) to a `YYYYMMDD` expiration (floored at 0) |
| `days_to_expiry(expiry, *, now=None) -> int` | Calendar days to expiry counted in America/New_York. A same-day expiry is `0` until the 16:00 ET close and `-1` after it, independent of the host's timezone |
| `expiry_close(expiry) -> datetime` | 16:00 America/New_York on the expiry date |
| `market_now(now=None)`, `parse_expiry(expiry)` | Exchange-time clock and `YYYYMMDD` / `YYYYMM` parser |
| `MARKET_TZ`, `MARKET_CLOSE` | `ZoneInfo("America/New_York")`, `time(16, 0)` |
| `chunked(seq, size)` | Yield successive `size`-length slices of a sequence |

---

## `ibtws.unofficial.client`

### `IBKRClient`

Thin lifecycle wrapper around `ib_async.IB`. The raw `IB` instance is exposed
as `.ib` for direct use of the full ib_async API. Applies
`config.request_timeout` and sets `RaiseRequestErrors = True` at construction.
No network I/O at construction.

```python
IBKRClient(config: IBKRConfig)
```

| Method | Purpose |
|---|---|
| `async connect()` | Idempotent. Opens the socket via `connectAsync` using the config (host/port/clientId/timeout/readonly/account/fetchFields) |
| `async disconnect()` | Idempotent. Stops any reconnect in progress; an explicit disconnect is never treated as a drop |
| `is_connected() -> bool` | Socket state |
| `add_reconnect_listener(fn)` / `remove_reconnect_listener(fn)` | `fn()` (sync or async) runs after every successful reconnect and after IB error 1101 (link restored, market data lost). `OrderManager` and `CreditSpreadStrategy` register themselves |
| `async get_market_data(contract, *, generic_ticks="100,101,104,106", settle=1.0, timeout=15.0) -> Ticker` | Qualify, stream `generic_ticks` for `settle` seconds alongside a `reqTickersAsync` snapshot (bounded by `timeout`); the stream is cancelled in `finally`. `generic_ticks=""` takes a plain snapshot with no wait. Raises `LookupError` if no ticker returned |
| `async get_historical_data(contract, duration, bar_size, *, use_rth=True, end_datetime=None, what_to_show="TRADES") -> pd.DataFrame` | Qualify + `reqHistoricalDataAsync`. Returns a DataFrame (empty when IB returns nothing) |
| `async __aenter__` / `__aexit__` | Context manager. **`__aenter__` calls `connect()`**; `__aexit__` always calls `disconnect()` |

`BarSize` and `Duration` are `Literal` string types enumerating the values IB's
`reqHistoricalData` accepts (any `"<int> <S|D|W|M|Y>"` duration is also valid).

```python
async with IBKRClient(cfg) as client:            # auto-connects
    client.ib.reqMarketDataType(2)               # 1 live, 2 frozen, 3 delayed
    ticker = await client.get_market_data(contract)
    bars = await client.get_historical_data(contract, "1 D", "5 mins")
```

> Note: calling `await client.connect()` inside an `async with` block is
> harmless (idempotent), which is why several examples do both.

**Connection supervision.** The client subscribes to `disconnectedEvent` and
`errorEvent`. An unexpected drop (TWS daily restart, network loss) starts a
reconnect loop with exponential backoff; after it succeeds the reconnect
listeners run, so order state is resynced and streaming quotes are
resubscribed. IB codes 1100 / 1101 / 1102 are logged; 1101 also triggers the
listeners because IB dropped the market-data subscriptions.

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
  `underlying_price`, `market_data_type` (1 live, 2 frozen, 3 delayed,
  4 delayed-frozen) and `timestamp` (epoch seconds of the ticker's last tick,
  not of object creation). Every metric is `Optional[float]`;
  `None` means "IB returned no data" — never silently coerced to 0.
- **`IVRankResult`** (frozen) — see `option.iv_rank` below.

### `option.utils`

| Public helper | Purpose |
|---|---|
| `quotes_to_dataframe(quotes) -> pd.DataFrame` | Flatten quotes; returns an empty DataFrame with `DATAFRAME_COLUMNS` when input is empty, so downstream `df["strike"]` / `df.empty` checks always work |
| `DATAFRAME_COLUMNS` | Canonical column order for the projection |

Internal helpers (`_filter_expirations`, `_filter_strikes`, `_ticker_to_quote`)
back `fetch_snapshot`.

### `option.chains.OptionChainFetcher`

Throttled, fault-tolerant snapshot fetcher built on `IBKRClient`. Does NOT
own the connection — the caller handles `connect()` / `disconnect()`.

```python
OptionChainFetcher(client: IBKRClient, *, chain_cache_ttl=900.0, snapshot_timeout=30.0)
```

Chain definitions are cached for `chain_cache_ttl` seconds (`0` disables the
cache) and qualified option contracts for the life of the fetcher, so repeated
plans on one underlying skip `reqSecDefOptParams` / `qualifyContracts`.
`clear_cache()` drops both.

| Method | Purpose |
|---|---|
| `async fetch_chain_definition(underlying, *, exchange="SMART", trading_class=None) -> ChainDefinition` | Returns the (expirations × strikes) universe for one underlying. Raises `ValueError` if the underlying is unqualified, `LookupError` if no option params match. Falling back to another exchange's chain (no `trading_class` given) logs a warning |
| `async fetch_snapshot(underlying, *, exchange="SMART", currency="USD", trading_class=None, rights=("C","P"), expirations=None, expiry_from/to=None, strikes=None, strike_from/to=None, strike_window_pct=0.2, batch_size=100, as_dataframe=False) -> list[OptionQuote] \| DataFrame` | Resolve + quote a slice of the chain. **Fault-tolerant**: qualify or snapshot failures are logged at WARNING and excluded, so the caller always gets a partial answer. Auto-windows strikes around spot (`strike_window_pct`) when no explicit strikes are given |

Selection precedence for both expirations and strikes: an explicit whitelist
(`expirations=` / `strikes=`) wins over an inclusive range (`*_from` / `*_to`).
Quotes with no `bid`, `ask` and `iv` are dropped before returning.

Market-data subscriptions are opened and cancelled inside each snapshot batch
(subscribe → settle 0.2 s → `reqTickersAsync` bounded by `snapshot_timeout` →
cancel in `finally`), so subscriptions are released even when a snapshot fails
or times out. Spot for the strike window comes from a plain snapshot
(`Ticker.marketPrice()`, then last / close / mid).

### `option.iv_rank.IVRankCalculator`

IV Rank / IV Percentile from IB's daily `OPTION_IMPLIED_VOLATILITY` series.

```python
IVRankCalculator(client, *, request_timeout=60.0)
async calculate(underlying, *, lookback_days=252, end_datetime="", use_rth=True) -> IVRankResult
```

`IVRankResult` fields: `underlying_symbol`, `as_of`, `current_iv`, `min_iv`,
`max_iv`, `iv_rank` (0–100 or `None`), `iv_percentile` (0–100 or `None`),
`sample_size`, `lookback_days`.

- `iv_rank = (current − min) / (max − min) × 100`; `None` when max == min
  (degenerate flat window).
- `iv_percentile` = share of *historical* observations strictly below current,
  excluding the current bar; `None` for single-bar history (never silently 0).
- Failed history requests are logged and treated as empty (all-`None` result).
- The underlying is qualified on the fly when `conId` is missing.

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
`StopRequest`, `BracketRequest` (entry + TP, plus optional OCA-grouped SL).
All carry `contract`, `side`, `quantity`, `tif`, `account`, `outside_rth`.

**Runtime.**
- `TrackedOrder` (mutable) — live view of one submitted order, kept in sync
  by the manager.
- `PositionSnapshot` (frozen) — point-in-time position record.
- `PositionPnL` (frozen) — on-demand mark-to-market. `market_price` /
  `market_value` / `unrealized_pnl` are `Optional` — `None` means "quote
  unavailable", never `0.0`.

**Events** (all frozen, all with `to_dict()`):
`RequestSubmitted`, `StatusChanged`, `Filled`, `Cancelled`, `Rejected`,
`PositionChanged`, `LegMismatch`. Unioned as `OrderEvent`. `Filled` carries
`con_id` / `sec_type` of the contract that traded: IB reports a BAG order's
fills per leg under the combo's `orderRef`, so group by `con_id` before
summing quantities.

`event_from_dict(data)` reverses `to_dict()`; raises `ValueError` on an unknown
discriminator.

### `order.factory`

Builders that produce request dataclasses (validated in-line) and translate
them into raw `ib_async.Order` objects.

| Function | Purpose |
|---|---|
| `build_market(contract, side, qty, *, tif=DAY, account=None, outside_rth=False)` | `MarketRequest` |
| `build_limit(contract, side, qty, limit_price, *, ...)` | `LimitRequest` |
| `build_stop(contract, side, qty, stop_price, *, ...)` | `StopRequest` |
| `build_bracket(contract, side, qty, *, take_profit_price, stop_loss_price=None, entry_limit_price=None, ...)` | `BracketRequest` (`stop_loss_price=None` → TP-only; `entry_limit_price=None` → market entry) |
| `request_to_order(request, order_ref)` | Translate one request → `Order` |
| `bracket_to_orders(req, parent_ref, tp_ref, sl_ref, *, parent_order_id, oca_group)` | Wired `Order`s: parent + TP (+ SL when set). With SL: parent/TP `transmit=False`, SL `transmit=True`, TP/SL share `ocaGroup`+`ocaType=1`. TP-only: returns `[parent, TP]`, TP transmits the group. BAG contracts skip positivity/geometry checks (signed net prices) |

### `order.utils`

| Function | Purpose |
|---|---|
| `make_order_ref() -> str` | 32-char hex UUID used as `orderRef` |
| `is_paper_account(account_id) -> bool` | True iff it starts with `DU` |
| `validate_request(request) -> None` | Raises `ValueError` with a precise message. Rejects: unsupported type, missing contract, unqualified non-BAG contract, non-`OrderSide`, non-positive qty, non-positive limit/stop/TP/SL (BAG combo limits are allowed signed/zero/negative since they encode net credit), bracket TP/SL geometry inconsistent with side. A BAG contract is accepted iff every `comboLeg` carries a non-zero `conId` |

### `order.store`

`OrderStore` is a `@runtime_checkable` `Protocol` — the minimal contract is
`async append(event)` + `replay() -> Iterator[OrderEvent]`. No update / no
delete / no query.

#### `JsonStore(path, *, fsync=True)`

Append-only JSONL audit log.

| Method | Purpose |
|---|---|
| `path` (property) | File path |
| `async append(event)` | Serialise, write, flush, optional `os.fsync` — in a worker thread, under `asyncio.Lock`, so a slow disk never blocks the event loop. The file handle stays open between appends |
| `replay() -> Iterator[OrderEvent]` | Stream events back from disk. Missing file = empty iterator (no error). An unfinished last line (no trailing newline: a crash mid-write) is skipped with a warning; a bad line anywhere else raises `ValueError` with `path:line_no` |
| `close()` | Close the file handle (also done on GC) |

Parent directory must exist; the file is created on first append.
`fsync=False` is ~10× faster but loses kernel-crash safety — fine for tests.
Recovery model is pure replay. The only rewrite is crash repair: before the
first append, an unfinished last line is truncated so new lines are never
glued onto the fragment.

### `order.monitor`

Fan-out event bus. Supports both async-iterator and sync-callback styles.

```python
OrderMonitor(*, max_queue=10_000)
```

| Method | Purpose |
|---|---|
| `publish(event)` | Deliver to every stream and fire registered callbacks. Callback exceptions are logged + swallowed — one bad subscriber cannot poison the bus |
| `async stream() -> AsyncIterator[OrderEvent]` | Independent subscriber with its own bounded queue: every stream sees every event published after it subscribed. The first stream also gets the backlog buffered while nobody listened |
| `register(fn)` / `unregister(fn)` | Sync inline callback (un)registration |
| `close()` / `reopen()` | End all streams (used by `OrderManager.stop()`) / allow new ones |
| `dropped` | Count of events dropped because a queue or the backlog hit `max_queue` (oldest first) |

### `order.reconciler`

```python
async reconcile(client, store, *, positions=None) -> ReconciliationReport
```

`positions` reuses an already-fetched `reqPositionsAsync` result.

Startup divergence diff between IB (source of truth) and the local audit
log. Never mutates IB or the store. Logs WARNINGs on `local_only` (we
thought we had it, IB doesn't) and `ib_only` (IB has an open order we
never persisted — e.g. placed from TWS directly).

`ReconciliationReport` (frozen): `matched: list[str]`, `local_only:
list[str]`, `ib_only: list[Trade]`, `positions: list[PositionSnapshot]`.

Terminal states for the local fold = FILLED / CANCELLED / REJECTED. A
`Cancelled` event, or a `Rejected` event whose reason is not a bare
`"Inactive"` (e.g. a `placeOrder` that raised, IB error 201), also closes an
order locally.

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
    pace_per_sec:    float = 0.0,   # ib_async already throttles to 45 msg/s
    executor:        ThrottledExecutor | None = None,
)
```

#### Lifecycle

| Method | Behaviour |
|---|---|
| `async start() -> ReconciliationReport` | Binds IB events; raises `RuntimeError` if `managedAccounts` is empty or the primary account is live without `allow_live=True`. Rehydrates `TrackedOrder` for UUIDs that matched the reconciler. Seeds the position cache + live `Contract` refs. Starts the background persist worker and registers `resync` as a reconnect listener. If anything fails after events are bound, they are unbound again |
| `async stop()` | Unbinds IB events, drains the persist queue, cancels the worker, ends `events()` streams. Idempotent |
| `async resync()` | After a reconnect: refresh positions and replay the latest status of every tracked order, so fills/cancels that happened while disconnected reach subscribers |

#### Placement (persist-first)

| Method | Notes |
|---|---|
| `async place(request) -> TrackedOrder` | Persists `RequestSubmitted` **before** `placeOrder`. If `placeOrder` raises, a `Rejected` event closes the audit entry and the error propagates. Per-UUID `asyncio.Lock` guards mutations |
| `async place_bracket(request) -> [parent, tp(, sl)]` | Shared `bracket_group`; two orders for TP-only, three with SL. Every member is persisted (`extra["leg"]` = parent / tp / sl). A failure part-way cancels the members already placed |
| `async market(...)` / `limit(...)` / `stop_order(...)` / `bracket(...)` | One-call build + place |

`stop_order` is named to avoid shadowing the `stop()` lifecycle method.

#### Position management

| Method | Notes |
|---|---|
| `async close_position(con_id, *, kind="market", limit_price=None, cancel_working=True, refresh=True, account=None, cancel_timeout=10.0) -> TrackedOrder \| None` | Cancels every working order that trades the contract — including combos that have it as a leg and orders rehydrated after a restart — and **waits for IB to confirm**; raises `RuntimeError` instead of closing if the cancels are not confirmed. Then re-reads positions, qualifies/backfills the exchange and submits an opposite-side order for the fresh quantity. Returns `None` for a missing/zero position |
| `async refresh_positions() -> list[PositionChanged]` | Pull fresh from IB and replace the cache (positions IB no longer reports are dropped). positionEvent can lag — always refresh before flattening |
| `async close_all_positions(*, kind="market", cancel_working=True, con_ids=None, accounts=None) -> list[TrackedOrder]` | Refreshes then flattens non-zero positions. **Without filters it closes every position on every account**, including ones this process never opened — pass `con_ids` to limit it to a strategy's legs |
| `async cancel(uuid)` | Idempotent. `KeyError` for an unknown uuid. Never paced |
| `async cancel_all() -> list[str]` | Sends cancels for all non-terminal orders concurrently; does not wait for confirmation |
| `async wait_for(predicate, timeout=None) -> bool` | Sleep until `predicate()` is true, woken by every order/fill/position event (1 s safety recheck). Returns the predicate's final value |
| `prune_terminal(max_age=3600.0) -> int` | Forget terminal orders older than `max_age` seconds |

#### Read-only views

| | |
|---|---|
| `open_orders` | List of non-terminal `TrackedOrder` |
| `positions` | List of cached `PositionChanged` |
| `async current_pnl(con_ids=None, *, snapshot_timeout=5.0) -> list[PositionPnL]` | On-demand mark-to-market. Pricing rule: mid > last > close. `None` = unknown, never `0.0`. Skips zero-quantity positions |
| `events() -> AsyncIterator[OrderEvent]` | Independent stream from the monitor (each call is its own subscriber) |
| `on_event(fn)` | Sync callback registration |

#### Key behaviour

- **Persist-first** — `RequestSubmitted` hits the store before `placeOrder`,
  so a crash between persist and submit is detectable by the reconciler
  (a `local_only` entry that IB never saw → you know it was never sent).
- **Paper interlock** — refuses a live primary account unless `allow_live=True`;
  `_check_account_safety` polices per-request `account` against
  `managedAccounts` (also blocks live sub-accounts on paper-primary FA setups).
- **Readonly** — with `config.readonly=True` every place / cancel call raises
  `RuntimeError` before reaching IB.
- **Exec dedup** — a bounded LRU of exec ids drops replayed `execDetails`
  after a reconnect so downstream TP/SL logic doesn't double-fire.
- **Status mapping** — `PendingCancel` / `PreSubmitted` → SUBMITTED (still live
  until a terminal state arrives). `Inactive` emits `Rejected` but keeps the
  order INACTIVE (not terminal, still cancellable): IB also uses it for orders
  it is merely holding.
- **IB rejections** — order errors 201 (rejected: permissions, margin,
  MiFID/KID…), 203 (security not allowed) and 10147 (cancel target not found)
  on a pending or Inactive order mark it REJECTED and emit `Rejected`; a late
  Inactive status cannot reopen it. Rejections of a working order (failed
  modification) are ignored — that order is still live.
- **Persist failures logged, not raised** — the persist worker logs and
  continues, so the event bus still publishes even if disk I/O fails.
- **Concurrency / pacing** — a single `ThrottledExecutor` (no pacing by
  default; ib_async throttles the socket). Pass a shared `executor` to
  govern the aggregate request rate across subsystems.

---

## `ibtws.unofficial.analysis`

Pure analytics over DataFrames (pandas / numpy / scipy). No `ib_async`
dependency — feed these the DataFrame from
`option.utils.quotes_to_dataframe` (or any equivalently-shaped frame) and a
VIX/price series. Trivially unit-testable.

### `analysis.gex.GexCalculator`

Gamma Exposure (GEX) calculator using Black-Scholes repricing. The profile
sweep is vectorised (one `sweep_points × options` gamma matrix), so a full
0DTE SPX window computes in milliseconds.

```python
GexCalculator(*, risk_free_rate=0.045, sweep_points=500, bar_width=4.0)
compute(df: pd.DataFrame) -> GexResult
```

Required DataFrame columns: `strike`, `right`, `gamma`, `open_interest`,
`underlying_price`, `iv`, `expiry`, `timestamp`.

| Member | Purpose |
|---|---|
| `compute(df)` | Validates columns, drops rows missing strike/gamma/OI/IV, with IV ≤ 0 or past their expiry, then runs the full pipeline: per-strike net GEX, BS repricing sweep, Zero Gamma Level via Brent root-finding (strict sign changes only), call/put walls, 25-delta skew. Returns and caches a `GexResult` |
| `summary()` | Formatted multi-line summary string (also prints). Raises if `compute()` wasn't called |
| `plot(save_path=None, title_suffix="") -> bytes` | Render the combined profile + histogram chart; returns PNG bytes and also saves to `save_path` when given. Never opens a window. Needs the optional extra: `pip install "ibtws[plot]"` (matplotlib is imported lazily, so `compute()` works without it) |
| `zero_gamma_level` / `regime` / `total_gex` (properties) | Convenience accessors on the cached result |

`GexResult` (dataclass) carries `spot`, `zero_gamma_level`, `regime`
(`"POSITIVE"`/`"NEGATIVE"`), `pts_from_zgl`, `total_gex`, `call_gex`,
`put_gex`, `call_wall`, `put_wall`, `all_crossings`, sweep arrays, the
per-strike net-GEX frame, and optional `skew` / `skew_ratio`.

Time to expiry is computed **per row** from that row's expiry at 16:00
America/New_York and its `timestamp`, so mixed expiries are priced correctly
and the result does not depend on the host's timezone. Rows at or after their
close are dropped; `ValueError` if nothing usable remains.

### `analysis.expected_move.ExpectedMoveCalculator`

Expected-move estimate from an option-chain DataFrame via two methods.

```python
ExpectedMoveCalculator().calculate(df: pd.DataFrame) -> ExpectedMoveResult
```

Required columns: `strike`, `right`, `bid`, `ask`, `iv`, `underlying_price`,
`expiry` (optional `symbol`). Raises `ValueError` on an empty frame or missing
columns/data.

Methods combined:
1. **ATM straddle** — ATM call mid + ATM put mid.
2. **IV-based 1σ** — `spot × IV × √(DTE/365)` from the average ATM IV.

`ExpectedMoveResult` (frozen): `underlying_symbol`, `spot`, `expiration`,
`straddle_move`/`straddle_pct`, `iv_move`/`iv_pct`/`atm_iv`, and `avg_move`
(average of the two methods; `None` unless both are available). Every derived
metric is `None` when its inputs are unavailable — never silently zero.

### `analysis.market_bias.determine_market_bias`

```python
determine_market_bias(market_data: pd.DataFrame | None, *, fast_window=5, slow_window=10, volume_window=20) -> dict
```

Classifies directional bias from `close` prices using fast/slow moving
averages. Returns a `{"bias": ..., "details": {...}}` dict where `bias` is
`"bullish"`, `"bearish"`, `"neutral"`, or the sentinel `"!neutral"`.

- A directional bias is reported **only when trend (fast MA vs slow MA) and
  momentum (last close vs slow MA) agree** and are non-neutral — a deliberate
  confirmation filter against choppy-market crossovers.
- `"!neutral"` signals a *data* problem (no data, insufficient rows, missing
  `close` column, or NaN in the evaluated window) so callers can distinguish
  "flat market" from "cannot tell".
- Raises `ValueError` for misconfigured windows (non-positive, or
  `fast_window >= slow_window`) — a caller programming error, distinct from a
  data problem.

### `analysis.volatility_risk.common_volatility_risk`

```python
common_volatility_risk(vix_series, vx1d_current, vix3m_current=None,
                       lookback_days=20, risk_threshold=50, debug=False) -> dict
```

Pre-market volatility-risk score on a 0–100 scale for short-premium / 0DTE
gating, from four components:

| Component | Range | Signal |
|---|---|---|
| VIX deviation | 0–35 | z-score vs a rolling window (+ momentum adjustment) |
| VX1D / VIX ratio | 0–25 | intraday vs 30-day implied vol |
| Absolute VIX | 0–20 | raw level of fear |
| Term structure | 0–20 | VIX slope vs VIX3M (skipped when `vix3m_current` is `None`) |

Returns `decision` (`"TRADE"` / `"NO TRADE"` against `risk_threshold`),
`risk_score`, `risk_threshold`, `overall_structure` (human-readable flags),
`component_scores`, and — when `debug=True` — a `metrics` dict of raw values.

**Fails loud**: raises `ValueError` when `vix_series` is too short, contains
NaN in the scoring window, or when any current VIX / VIX1D / VIX3M input is
non-positive or non-finite. A trade gate should treat bad data as a hard block
(fail-closed), not receive a silently maxed-out score.

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
| `target_short_delta` | `0.30` | `\|Δ\|` target for the short leg |
| `wing_width` | `5.0` | Strike distance ($) — selector snaps to nearest available |
| `target_dte` / `dte_tolerance` | `30` / `14` | Pick expiry closest to target within tolerance |
| `max_short_delta` | `0.50` | Hard cap on chosen short `\|Δ\|`; `None` to disable |
| `min_open_interest` / `min_volume` | `0` / `0` | Liquidity filters (legs with `None` are kept) |
| `min_credit` / `min_credit_width_ratio` | `None` / `None` | Economic floors |
| `quantity` | `1` | Number of spreads |
| `limit_slippage` | `0.05` | Fraction below mid the entry limit can sit |
| `tif` / `account` / `outside_rth` | DAY / `None` / `False` | Order knobs |
| `take_profit_pct` | `0.5` | Close at 50 % of credit captured |
| `stop_loss_multiplier` | `2.0` | Close on loss = N × credit (capped at width) |
| `exchange` / `currency` / `trading_class` | `"SMART"` / `"USD"` / `None` | Universe selector (e.g. `"SPXW"` for PM-settled SPX dailies) |
| `expirations` / `expiry_from` / `expiry_to` | `None` | Pre-filters on the chain |
| `strike_window_pct` | `0.10` | ±10 % of spot bounds the snapshot |
| `require_live_quotes` | `False` | Refuse to build a plan from, or decide exits on, non-live quotes (market data type ≠ 1) |

#### Pure selectors (unit-testable)

| Function | Behaviour |
|---|---|
| `select_expiry(expirations, *, target_dte, dte_tolerance, now=None) -> str` | Closest expiry inside tolerance; raises `CreditSpreadError` if none qualifies. Ignores negative-DTE entries. DTE is counted in exchange time: today's expiry is 0 DTE until 16:00 ET |
| `select_short_leg(quotes, *, target_short_delta, max_short_delta, min_open_interest, min_volume) -> OptionQuote` | Tradeability filter + closest `\|Δ\|`. Rejects if the max-delta ceiling leaves nothing |
| `select_long_leg(quotes, *, short, wing_width, spread_type, ...) -> OptionQuote` | Snaps to the nearest strike at or beyond the requested width on the protective side. Refuses widths < 50 % of requested (chain too narrow) |

#### `CreditSpreadStrategy`

```python
CreditSpreadStrategy(
    client: IBKRClient,
    order_manager: OrderManager,
    *,
    fetcher:           OptionChainFetcher | None = None,
    tick_size:         float = 0.05,
    exit_fill_timeout: float = 15.0,   # seconds a closing order may work before re-pricing
    exit_max_attempts: int = 4,        # stop-loss: the last attempt is priced at the width
    cancel_timeout:    float = 10.0,   # wait for IB to confirm a cancel
    quote_max_age:     float = 60.0,   # streaming quotes older than this count as missing
)
```

`order_manager` is required — combo placement always goes through it.

| Method | Purpose |
|---|---|
| `async build_plan(params) -> CreditSpreadPlan` | Qualify underlying → fetch chain → pick expiry → snapshot the relevant right → select legs → enforce economic constraints |
| `async place(plan, *, limit_credit=None) -> TrackedOrder` | Submits as `BUY @ -credit` signed combo limit via `OrderManager.limit`. Rounds the credit **down** to the tick; refuses a non-positive credit |
| `async close(plan, *, limit_debit=None, tif=None, quantity=None) -> TrackedOrder` | One buy-back order for the BAG (`SELL @ -debit`), fire-and-forget. Rounds the debit **up**, caps it at the width. `quantity` defaults to `params.quantity` |
| `async close_and_confirm(plan, quantity, *, urgent, mid_debit=None) -> ExitResult` | Close and wait for IB to confirm: each attempt waits `exit_fill_timeout`, cancels the remainder (waiting for the cancel), re-quotes and re-prices with more slippage. `urgent=True` makes the last attempt marketable (priced at the width). Stops instead of risking an over-close if a cancel is not confirmed. `ExitResult` has `order`, `closed_quantity`, `remaining_quantity`, `complete` |
| `async await_entry(entry, *, max_wait=None, quantity=None) -> float` | Wait for the entry and return the filled quantity. At the deadline, or when IB parks the order as Inactive, the unfilled remainder is cancelled and the cancel awaited, so it cannot fill later unmanaged |
| `async monitor_and_exit(plan, entry, *, poll_interval=2.0, max_wait=None, close_on_timeout=True, max_quote_failures=30) -> TrackedOrder \| None` | `await_entry` → stream both legs → close the **filled** size on TP (no escalation; any unfilled part stays monitored) or SL (urgent). `max_wait` is one deadline for the whole call; at the deadline the position is closed urgently unless `close_on_timeout=False`. After `max_quote_failures` polls without a usable quote it closes urgently while connected and waits for the reconnect otherwise. Returns the last closing order, or `None` |
| `watch(plan)` / `unwatch(plan)` | Reference-counted streaming subscriptions for both legs; resubscribed automatically after a reconnect |
| `async current_mid_debit(plan) -> float \| None` | Mid debit per share from the streams (stale quotes → `None`) or a one-off snapshot when not watched. With `require_live_quotes`, non-live quotes → `None` |

#### Combo sign convention

The BAG is submitted with `action="BUY"` and a **signed net cost** as the
limit price: negative = credit collected, positive = debit paid. So a $0.45
credit is sent as `BUY @ -0.45`. This matches TWS combo display and avoids
the SELL/+price sign-flip ambiguity that bites SMART-routed combos. Leg
directions live in `plan.bag.comboLegs` (SELL short, BUY long).

```python
async with IBKRClient(cfg) as client:
    om = OrderManager(client, JsonStore("orders.jsonl"))
    await om.start()
    fetcher = OptionChainFetcher(client)
    strat = CreditSpreadStrategy(client, om, fetcher=fetcher)

    plan = await strat.build_plan(
        CreditSpreadParams(
            underlying=Index("SPX", "CBOE", "USD"),
            spread_type=SpreadType.BULL_PUT,
            target_short_delta=0.10,
            wing_width=10.0,
            target_dte=0,
            trading_class="SPXW",
        )
    )
    tracked = await strat.place(plan)
    # Close at TP/SL, or urgently after 5 hours at the latest.
    await strat.monitor_and_exit(plan, tracked, max_wait=5 * 3600)
```

---

## Cross-cutting design notes

### Persistence model — pure replay
`JsonStore` is append-only. There is no UPDATE / DELETE / query — recovery
is `replay()` then fold to current state. A torn last line from a crash is
skipped (and truncated before the next write); corruption anywhere else is
detectable (replay raises on a bad line). Writes are crash-safe under
`fsync=True`.

### Persist-first ordering
`OrderManager` always persists `RequestSubmitted` **before** calling
`placeOrder`. A crash in the window between persist and submit leaves a
local entry the reconciler classifies as `local_only` — you know the order
was never sent and can decide whether to re-submit or drop it.

### Idempotency
- `cancel(uuid)` — terminal orders are a no-op (warn-log).
- `connect()` / `disconnect()` — safe to call multiple times.
- `JsonStore.replay()` — pure read, no side effects.

### Fault tolerance
- `IBKRClient` reconnects on its own after a drop; `OrderManager.resync` and
  the strategy's quote streams are restored through reconnect listeners.
- Exits are confirmed, not fire-and-forget: `close_and_confirm` re-prices
  unfilled closes and escalates stop-loss exits to a marketable price.
- `fetch_snapshot` — failed qualify or snapshot batches are logged at
  WARNING and excluded; the caller always gets a partial answer.
- The persist worker logs store failures instead of raising, so the event
  bus still publishes.
- `current_pnl` — a missing quote yields `None` fields, never a fabricated 0.

### Rate limiting
ib_async throttles every outgoing message to 45/s per connection, which keeps
a session under IB's ~50 msg/s ceiling. The order path additionally uses a
`ThrottledExecutor` (10 concurrent, no pacing by default); pass a shared
`executor` or a `pace_per_sec` to slow it down deliberately. Cancels are never
paced.
