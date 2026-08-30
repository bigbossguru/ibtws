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
    │   ├── volatility_risk.py    # common_volatility_risk
    │   └── volatility_regime.py  # detect_volatility_regime — 0DTE tail-regime gate
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
| `readonly` | `False` | True blocks order submission at the API level |
| `account` | `""` | Specific account when the session has multiple |
| `connect_timeout` | `10.0` | Seconds for `connectAsync` to complete |
| `request_timeout` | `30.0` | Default per-request timeout (`ib.RequestTimeout`) |
| `fetch_fields` | `POSITIONS \| ORDERS_OPEN \| ORDERS_COMPLETE \| ACCOUNT_UPDATES \| EXECUTIONS` | `StartupFetch` bitmask of data to pull on connect |

Pure data; no I/O on construction.

---

## `ibtws.unofficial._pacing`

Shared concurrency + rate-limit primitive. `OrderManager` accepts an
`executor` so a single bucket can govern the aggregate request rate (keeping the
session under IB's ~50 msg/s ceiling).

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
| `calc_dte(expiration) -> float` | Calendar days from today to a `YYYYMMDD` expiration (floored at 0) |
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
| `async disconnect()` | Idempotent — no-op when already disconnected |
| `async get_market_data(contract) -> Ticker` | Qualify, `reqMktData` (generic ticks `100,101,104,106`), settle ~1 s, snapshot via `reqTickersAsync`, cancel. Raises `LookupError` if no ticker returned |
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
OptionChainFetcher(client: IBKRClient)
```

| Method | Purpose |
|---|---|
| `async fetch_chain_definition(underlying, *, exchange="SMART", trading_class=None) -> ChainDefinition` | Returns the (expirations × strikes) universe for one underlying. Raises `ValueError` if the underlying is unqualified, `LookupError` if no option params match |
| `async fetch_snapshot(underlying, *, exchange="SMART", currency="USD", trading_class=None, rights=("C","P"), expirations=None, expiry_from/to=None, strikes=None, strike_from/to=None, strike_window_pct=0.2, batch_size=100, as_dataframe=False) -> list[OptionQuote] \| DataFrame` | Resolve + quote a slice of the chain. **Fault-tolerant**: qualify or snapshot failures are logged at WARNING and excluded, so the caller always gets a partial answer. Auto-windows strikes around spot (`strike_window_pct`) when no explicit strikes are given |

Selection precedence for both expirations and strikes: an explicit whitelist
(`expirations=` / `strikes=`) wins over an inclusive range (`*_from` / `*_to`).
Quotes with no `bid`, `ask` and `iv` are dropped before returning.

Market-data subscriptions are opened and cancelled inside each snapshot batch
(subscribe → settle 0.2 s → `reqTickersAsync` → cancel), so no long-lived
subscriptions are leaked.

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
`PositionChanged`, `LegMismatch`. Unioned as `OrderEvent`.

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
| `async append(event)` | Serialise, write, flush, optional `os.fsync`. Under `asyncio.Lock` |
| `replay() -> Iterator[OrderEvent]` | Stream events back from disk. Missing file = empty iterator (no error). Raises `ValueError` with `path:line_no` on corruption |

Parent directory must exist; the file is created on first append.
`fsync=False` is ~10× faster but loses kernel-crash safety — fine for tests.
Recovery model is pure replay; the file is never rewritten or truncated.

### `order.monitor`

Fan-out event bus. Supports both async-iterator and sync-callback styles.

| Method | Purpose |
|---|---|
| `publish(event)` | Enqueue and fire registered callbacks. Callback exceptions are logged + swallowed — one bad subscriber cannot poison the bus |
| `async stream() -> AsyncIterator[OrderEvent]` | Backpressure-friendly queue consumer |
| `register(fn)` / `unregister(fn)` | Sync inline callback (un)registration |

### `order.reconciler`

```python
async reconcile(client, store) -> ReconciliationReport
```

Startup divergence diff between IB (source of truth) and the local audit
log. Never mutates IB or the store. Logs WARNINGs on `local_only` (we
thought we had it, IB doesn't) and `ib_only` (IB has an open order we
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
| `async start() -> ReconciliationReport` | Binds IB events; raises `RuntimeError` if `managedAccounts` is empty or the primary account is live without `allow_live=True`. Rehydrates `TrackedOrder` for UUIDs that matched the reconciler. Seeds the position cache + live `Contract` refs. Starts the background persist worker |
| `async stop()` | Unbinds IB events, drains the persist queue, cancels the worker. Idempotent |

#### Placement (persist-first)

| Method | Notes |
|---|---|
| `async place(request) -> TrackedOrder` | Persists `RequestSubmitted` **before** `placeOrder`. Per-UUID `asyncio.Lock` guards mutations |
| `async place_bracket(request) -> [parent, tp(, sl)]` | Shared `bracket_group`; two orders for TP-only, three with SL |
| `async market(...)` / `limit(...)` / `stop_order(...)` / `bracket(...)` | One-call build + place |

`stop_order` is named to avoid shadowing the `stop()` lifecycle method.

#### Position management

| Method | Notes |
|---|---|
| `async close_position(con_id, *, kind="market", limit_price=None, cancel_working=True) -> TrackedOrder \| None` | Cancels working orders on the same contract first; qualifies and backfills an empty exchange; submits an opposite-side order. Returns `None` for a missing/zero position |
| `async refresh_positions() -> list[PositionChanged]` | Pull fresh from IB (positionEvent can lag — always refresh before flattening) |
| `async close_all_positions(*, kind="market", cancel_working=True) -> list[TrackedOrder]` | Refreshes then flattens every non-zero position |
| `async cancel(uuid)` | Idempotent. `KeyError` for an unknown uuid |
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
  (a `local_only` entry that IB never saw → you know it was never sent).
- **Paper interlock** — refuses a live primary account unless `allow_live=True`;
  `_check_account_safety` polices per-request `account` against
  `managedAccounts` (also blocks live sub-accounts on paper-primary FA setups).
- **Exec dedup** — `_seen_exec_ids` drops replayed `execDetails` after a
  reconnect so downstream TP/SL logic doesn't double-fire.
- **Status mapping** — `PendingCancel` / `PreSubmitted` → SUBMITTED (still live
  until a terminal state arrives). `Inactive` → emits `Rejected` and flips
  state to REJECTED.
- **Persist failures logged, not raised** — the persist worker logs and
  continues, so the event bus still publishes even if disk I/O fails.
- **Concurrency / pacing** — a single `ThrottledExecutor`. Pass a shared
  `executor` to govern the aggregate request rate across subsystems.

---

## `ibtws.unofficial.analysis`

Pure analytics over DataFrames (pandas / numpy / scipy). No `ib_async`
dependency — feed these the DataFrame from
`option.utils.quotes_to_dataframe` (or any equivalently-shaped frame) and a
VIX/price series. Trivially unit-testable.

### `analysis.gex.GexCalculator`

Gamma Exposure (GEX) calculator using Black-Scholes repricing.

```python
GexCalculator(*, risk_free_rate=0.045, sweep_points=500, bar_width=4.0)
compute(df: pd.DataFrame) -> GexResult
```

Required DataFrame columns: `strike`, `right`, `gamma`, `open_interest`,
`underlying_price`, `iv`, `expiry`, `timestamp`.

| Member | Purpose |
|---|---|
| `compute(df)` | Runs the full pipeline: per-strike net GEX, BS repricing sweep, Zero Gamma Level via Brent root-finding, call/put walls, 25-delta skew. Returns and caches a `GexResult` |
| `summary()` | Formatted multi-line summary string (also prints). Raises if `compute()` wasn't called |
| `plot(save_path=None, title_suffix="") -> bytes` | Render the combined profile + histogram chart; returns PNG bytes. Saves to `save_path` when given |
| `zero_gamma_level` / `regime` / `total_gex` (properties) | Convenience accessors on the cached result |

`GexResult` (dataclass) carries `spot`, `zero_gamma_level`, `regime`
(`"POSITIVE"`/`"NEGATIVE"`), `pts_from_zgl`, `total_gex`, `call_gex`,
`put_gex`, `call_wall`, `put_wall`, `all_crossings`, sweep arrays, the
per-strike net-GEX frame, and optional `skew` / `skew_ratio`.

`_derive_params` treats expiry as the **4pm (22:00 UTC) close** on the expiry
date and raises `ValueError` on non-positive time-to-expiry — critical for 0DTE
where midnight vs. close flips the sign of `T`.

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

### `analysis.volatility_regime.detect_volatility_regime`

```python
detect_volatility_regime(*, vix1d_open, vix1d_history, vix_open, vix_prev_close,
                         spx_closes, spx_price, zero_gamma_level,
                         vix_history=None, zgl_source=None,
                         config=DEFAULT_CONFIG) -> VolatilityRegimeResult
```

Pre-market regime gate for 0DTE SPX credit spreads / iron condors, implementing
`analysis/VOLATILITY_REGIME_CONCEPT.md` (v2). It is a **tail-regime cut-off, not
a good-day selector**: expected coverage is ~88 % of days, and it makes no claim
to improve the average outcome on the days it allows.

Complementary to `common_volatility_risk`: that one produces a graded 0–100
score, this one a binary gate with per-metric flag provenance.

| Metric | Soft (≈p90) | Hard (≈p98) |
|---|---|---|
| `base_level` — percentile rank of `VIX1D_open` over 60 sessions | rank 90–98 | rank > 98 |
| `vix1d_absolute` — raw VIX1D level | — | > 25 |
| `vix_roc` — `(VIX_open − VIX_prev_close) / VIX_prev_close` | 10–15 % | > 15 % |
| `term_structure` — `VIX1D / VIX` (normal level ≈ 0.6) | 0.85–1.00 | > 1.00 |
| `premium_richness` — `VIX1D_open − RV20` | −10 … −6 | < −10 |
| `gex` — signed distance from ZGL in expected moves | ±0.25 EM | < −0.25 EM |

Each metric contributes at most one flag; the decision is
`favorable = (hard == 0) and (soft <= 1)`. A low base level (`LOW`) is
informational and never blocks — minimum-credit adequacy is an entry rule, not a
regime question.

Returns a frozen `VolatilityRegimeResult`: `favorable`, `flags`
(`RegimeFlag(metric, severity, value, detail, missing)`), `base_rank`,
`base_regime` (`LOW`/`NORMAL`/`HIGH`/`EXTREME`), `degraded_base`, `reason`,
`metrics`, `zgl_source`, plus `hard_count` / `soft_count` / `missing_metrics`
properties and a one-line `summary()`.

**Fail-safe over fail-loud**: unlike the scorer, bad market data never raises —
an unavailable metric becomes a `missing`-tagged **hard** flag (logged as
`missing_data` rather than `risk_flag`, so infrastructure gaps stay separable in
flag-frequency statistics). An unavailable base level short-circuits the whole
evaluation with `reason="no_base_level"`. `ValueError` is reserved for
`VolatilityRegimeConfig` misconfiguration, which is a caller bug.

Thresholds live in the frozen `VolatilityRegimeConfig`. They are percentiles of
the observed distributions on 500 sessions — **not** transplanted from
equity-option practice. Preserve that property when retuning, and recheck flag
frequencies against §3.2 of the concept as the market regime shifts.

Two deliberate exclusions: the **macro calendar** (FOMC/CPI/NFP) is a separate
hard-skip module that overrides this result, and **put/call skew** is a
strike-selection modifier rather than a regime flag. **GEX is unvalidated** — the
only metric resting on theory alone, since historical option chains are not
available; treat it as a hypothesis under observation.

```python
from ibtws.unofficial.analysis.gex import GexCalculator
from ibtws.unofficial.analysis.volatility_regime import detect_volatility_regime

gex = GexCalculator().compute(chain_df)
regime = detect_volatility_regime(
    vix1d_open=8.4,
    vix1d_history=vix1d_opens,      # >= 60 prior sessions, today excluded
    vix_open=14.2,
    vix_prev_close=14.1,
    spx_closes=spx_daily_closes,    # >= 21 completed sessions
    spx_price=gex.spot,
    zero_gamma_level=gex.zero_gamma_level,
    zgl_source="GexCalculator/ibtws",
)
if regime.favorable and not macro_calendar_skip(today):
    ...  # proceed to strike selection
```

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

#### Pure selectors (unit-testable)

| Function | Behaviour |
|---|---|
| `select_expiry(expirations, *, target_dte, dte_tolerance, now=None) -> str` | Closest expiry inside tolerance; raises `CreditSpreadError` if none qualifies. Ignores negative-DTE entries |
| `select_short_leg(quotes, *, target_short_delta, max_short_delta, min_open_interest, min_volume) -> OptionQuote` | Tradeability filter + closest `\|Δ\|`. Rejects if the max-delta ceiling leaves nothing |
| `select_long_leg(quotes, *, short, wing_width, spread_type, ...) -> OptionQuote` | Snaps to the nearest strike at or beyond the requested width on the protective side. Refuses widths < 50 % of requested (chain too narrow) |

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

`order_manager` is required — combo placement always goes through it.

| Method | Purpose |
|---|---|
| `async build_plan(params) -> CreditSpreadPlan` | Qualify underlying → fetch chain → pick expiry → snapshot the relevant right → select legs → enforce economic constraints |
| `async place(plan, *, limit_credit=None) -> TrackedOrder` | Submits as `BUY @ -credit` signed combo limit via `OrderManager.limit`. Rounds to tick; refuses a non-positive credit |
| `async close(plan, *, limit_debit=None, tif=None) -> TrackedOrder` | Buy-back the BAG (`SELL @ -debit`). Derives the debit from `take_profit_debit` or a live re-quote when not supplied |
| `async monitor_and_exit(plan, entry, *, poll_interval=15.0, max_wait=None) -> TrackedOrder \| None` | Wait for entry fill, poll mid-debit, fire close on TP / SL. Tolerates quote drop-outs (continues polling). Returns `None` on timeout or pre-fill cancel |

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
    await strat.monitor_and_exit(plan, tracked, poll_interval=15.0)
```

---

## Cross-cutting design notes

### Persistence model — pure replay
`JsonStore` is append-only. There is no UPDATE / DELETE / query — recovery
is `replay()` then fold to current state. Corruption is detectable (replay
raises on a bad line) and writes are crash-safe under `fsync=True`.

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
- `fetch_snapshot` — failed qualify or snapshot batches are logged at
  WARNING and excluded; the caller always gets a partial answer.
- The persist worker logs store failures instead of raising, so the event
  bus still publishes.
- `current_pnl` — a missing quote yields `None` fields, never a fabricated 0.

### Rate limiting
A single `ThrottledExecutor` (semaphore + monotonic token bucket) governs the
order path (default 10 msg/s, 10 concurrent). Pass a shared `executor` to keep
the aggregate session rate under IB's ~50 msg/s ceiling.
