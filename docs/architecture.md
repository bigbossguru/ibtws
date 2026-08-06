# ibtws Architecture

Package functionality overview. Open this file in VS Code with the Markdown preview (`Cmd+Shift+V`) and the `bierner.markdown-mermaid` extension, or view on GitHub.

```mermaid
flowchart TB
    User([User Script])

    subgraph Config["⚙️ Configuration"]
        IBKRConfig[IBKRConfig<br/>host/port/timeouts<br/>startup fetch fields]
    end

    subgraph Core["🔌 Core Foundation (unofficial/)"]
        Client[IBKRClient<br/>connect/disconnect<br/>market + historical data]
        Pacing[_pacing<br/>ThrottledExecutor<br/>semaphore + token bucket]
        Helpers[helpers<br/>safe_pick_value<br/>calc_dte / chunked]
        IB[(ib_async.IB<br/>TWS/Gateway)]
    end

    subgraph Options["📈 option/"]
        ChainFetcher[OptionChainFetcher<br/>fetch_chain_definition<br/>fetch_snapshot]
        IVRank[IVRankCalculator]
        OptUtils[utils<br/>quotes_to_dataframe]
        OptModels[Models<br/>ChainDefinition<br/>OptionQuote / IVRankResult]
    end

    subgraph Orders["📝 order/"]
        OrderMgr[OrderManager<br/>place/cancel/monitor<br/>reconcile on startup]
        Factory[factory<br/>build_market/limit<br/>stop/bracket]
        Store[(JsonStore<br/>JSONL persistence)]
        Monitor[OrderMonitor<br/>event bus]
        Reconciler[reconciler<br/>IB vs local diff]
        OrdModels[Models<br/>OrderRequest variants<br/>OrderEvent variants<br/>OrderState/Side/TIF]
    end

    subgraph Analysis["🧮 analysis/ (pure, DataFrame-in)"]
        Gex[GexCalculator<br/>gamma exposure profile]
        ExpMove[ExpectedMoveCalculator]
        Bias[determine_market_bias]
        VolRisk[common_volatility_risk]
    end

    subgraph Strategies["🎯 strategies/"]
        CreditSpread[CreditSpreadStrategy<br/>build_plan / place<br/>monitor_and_exit]
        StratModels[CreditSpreadParams<br/>CreditSpreadPlan<br/>SpreadLeg / SpreadType]
    end

    User --> IBKRConfig
    IBKRConfig --> Client
    Client --> IB
    Client -.uses.-> Helpers

    User -->|fetch quotes| ChainFetcher
    User -->|place orders| OrderMgr
    User -->|run strategy| CreditSpread
    User -->|analyse frame| Analysis

    ChainFetcher --> Client
    ChainFetcher --> OptModels
    ChainFetcher --> OptUtils
    IVRank --> Client
    IVRank --> OptModels

    OrderMgr --> Client
    OrderMgr -.throttle.-> Pacing
    OrderMgr --> Factory
    OrderMgr --> Store
    OrderMgr --> Monitor
    OrderMgr --> Reconciler
    OrderMgr --> OrdModels
    Factory --> OrdModels

    CreditSpread --> ChainFetcher
    CreditSpread --> OrderMgr
    CreditSpread --> StratModels

    OptUtils -.DataFrame.-> Analysis

    Monitor -.events.-> User
    ChainFetcher -.OptionQuote[].-> User
```

## Layering

**Dependency direction (clean):** `strategies → order/option → client → ib_async`.
Lower layers never import upward.

The `analysis` package is intentionally decoupled: it depends only on
pandas / numpy / scipy and consumes the DataFrame produced by
`option.utils.quotes_to_dataframe` (plus VIX / price series). This keeps the
analytics pure and trivially unit-testable, independent of any IB connection.

## Typical flow

`IBKRConfig` → `IBKRClient.connect()` → spawn `OptionChainFetcher` +
`OrderManager` → compose into `CreditSpreadStrategy` for end-to-end
discover → select → place → monitor → exit. Chain snapshots can be flattened
to a DataFrame and fed into the `analysis` calculators (GEX, expected move,
market bias, volatility risk) independently of the order path.
