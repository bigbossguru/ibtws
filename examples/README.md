# Examples — `OptionChainFetcher` usage

Each file in this directory is a self-contained, runnable example showing one
way to use :class:`ibtws.unofficial.option.OptionChainFetcher`.

## Prerequisites

1. A running TWS or IB Gateway with API access enabled.
2. The default config in `IBKRConfig` connects to TWS paper at
   `127.0.0.1:7497`. Override `host`, `port`, `client_id` as needed in each
   example, e.g.:
   ```python
   config = IBKRConfig(port=4002, client_id=42)  # Gateway paper
   ```
3. Install the project: `make install`.

## Run

```bash
poetry run python examples/01_basic_snapshot.py
```

## Index

| File | What it shows |
|---|---|
| `01_basic_snapshot.py` | Minimum viable usage: connect, fetch a small snapshot, print it. |
