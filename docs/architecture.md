# ibtws Architecture

Package functionality overview. Open this file in VS Code with the Markdown preview (`Cmd+Shift+V`) and the `bierner.markdown-mermaid` extension, or view on GitHub.

```mermaid
flowchart TB
    User([User Script])

    subgraph Config["⚙️ Configuration"]
        IBKRConfig[IBKRConfig<br/>host/port/timeouts<br/>watchdog/reconnect]
    end

    subgraph Core["🔌 Core Foundation (unofficial/)"]
        Client[IBKRClient<br/>connect/disconnect<br/>auto-reconnect<br/>health watchdog]
        Errors[_ib_errors<br/>ErrorCategory<br/>classify]
        Pacing[_pacing<br/>ThrottledExecutor<br/>semaphore + token bucket]
        IB[(ib_async.IB<br/>TWS/Gateway)]
    end

    subgraph Options["📈 option/"]
        ChainFetcher[OptionChainFetcher<br/>fetch_snapshot]
        IVRank[IVRankCalculator]
        OptUtils[utils<br/>quotes_to_dataframe<br/>freshness filters]
        OptModels[Models<br/>ChainDefinition<br/>OptionQuote]
    end

    subgraph Orders["📝 order/"]
        OrderMgr[OrderManager<br/>place/cancel/monitor<br/>reconcile on startup]
        Factory[factory<br/>build_market/limit<br/>stop/bracket]
        Store[(JsonStore<br/>JSONL persistence)]
        Monitor[OrderMonitor<br/>event bus]
        Reconciler[reconciler<br/>IB vs local diff]
        OrdModels[Models<br/>OrderRequest variants<br/>OrderEvent variants<br/>OrderState/Side/TIF]
    end

    subgraph Strategies["🎯 strategies/"]
        CreditSpread[CreditSpreadStrategy<br/>plan_spread<br/>place_and_monitor<br/>TP/SL exit]
        StratModels[CreditSpreadParams<br/>CreditSpreadPlan<br/>SpreadLeg]
    end

    User --> IBKRConfig
    IBKRConfig --> Client
    Client --> IB
    Client -.uses.-> Errors

    User -->|fetch quotes| ChainFetcher
    User -->|place orders| OrderMgr
    User -->|run strategy| CreditSpread

    ChainFetcher --> Client
    ChainFetcher -.throttle.-> Pacing
    ChainFetcher --> OptModels
    ChainFetcher --> OptUtils
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

    Monitor -.events.-> User
    ChainFetcher -.OptionQuote[].-> User
```

## Typical Flow

`IBKRConfig` → `IBKRClient.connect()` → spawn `OptionChainFetcher` + `OrderManager` → compose into `CreditSpreadStrategy` for end-to-end discover → select → place → monitor → exit.
