from __future__ import annotations

import logging
from dataclasses import dataclass

from ib_async import StartupFetch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IBKRConfig:
    # TWS / IB Gateway host.
    host: str = "127.0.0.1"

    # TWS / IB Gateway port.
    port: int = 7497

    # TWS / IB Gateway client ID. Must be unique per simultaneous connection to the same TWS instance.
    client_id: int = 1

    # Set to True to disable all order submission methods in the OrderManager.
    # Useful for "read-only" applications that only want to monitor positions and market data.
    readonly: bool = False

    # Optional default account to use for API requests that require an account.
    # If left blank, TWS will pick the primary account automatically.
    account: str = ""

    # Time to wait for the initial connection to be established before giving up and raising an error.
    connect_timeout: float = 10.0

    # Time to wait for the initial startup fetch (positions, orders, etc.) to complete before giving up and raising an error.
    request_timeout: float = 30.0

    # Bitmask of data to fetch on startup. See StartupFetch for available options.
    fetch_fields: StartupFetch = (
        StartupFetch.POSITIONS
        | StartupFetch.ORDERS_OPEN
        | StartupFetch.ORDERS_COMPLETE
        | StartupFetch.ACCOUNT_UPDATES
        | StartupFetch.EXECUTIONS
    )
