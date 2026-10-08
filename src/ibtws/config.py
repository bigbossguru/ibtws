from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

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

    # Set to True to make the session read-only: the OrderManager refuses every
    # place / cancel call, and ib_async skips the order-related startup requests.
    # Useful for applications that only monitor positions and market data.
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

    # Reconnect automatically after an unexpected disconnect (TWS daily restart,
    # network drop). Delays grow exponentially from the initial to the max value.
    auto_reconnect: bool = True
    reconnect_initial_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    # None = retry until disconnect() is called.
    reconnect_max_attempts: int | None = None

    @classmethod
    def from_env(cls, prefix: str = "IBKR_", **overrides: Any) -> "IBKRConfig":
        """Build a config from environment variables, then apply ``overrides``.

        Reads ``{prefix}HOST``, ``{prefix}PORT``, ``{prefix}CLIENT_ID``,
        ``{prefix}ACCOUNT`` and ``{prefix}READONLY`` (``1/true/yes``). Unset
        variables keep the dataclass defaults, so connection details never
        have to be hard-coded in scripts.
        """
        values: dict[str, Any] = {}
        env = os.environ
        if f"{prefix}HOST" in env:
            values["host"] = env[f"{prefix}HOST"]
        if f"{prefix}PORT" in env:
            values["port"] = int(env[f"{prefix}PORT"])
        if f"{prefix}CLIENT_ID" in env:
            values["client_id"] = int(env[f"{prefix}CLIENT_ID"])
        if f"{prefix}ACCOUNT" in env:
            values["account"] = env[f"{prefix}ACCOUNT"]
        if f"{prefix}READONLY" in env:
            values["readonly"] = env[f"{prefix}READONLY"].strip().lower() in {"1", "true", "yes", "on"}
        values.update(overrides)
        return cls(**values)
