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
    """All tunable parameters in one place – no magic numbers scattered in code."""

    host: str = "127.0.0.1"
    """TWS / IB Gateway host."""

    port: int = 7497
    """7497 = TWS paper · 7496 = TWS live · 4002 = Gateway paper · 4001 = Gateway live."""

    client_id: int = 1
    """Must be unique per simultaneous connection to the same TWS instance."""

    readonly: bool = False
    """Set True to prevent any order submission."""

    account: str = ""
    """Leave blank to let TWS pick the primary account automatically."""

    connect_timeout: float = 10.0
    """Seconds to wait for the initial TCP handshake + sync."""

    request_timeout: float = 30.0
    """Default timeout for blocking API requests."""

    # --- reconnection back-off ---
    reconnect_base_delay: float = 2.0
    """Starting delay (seconds) before the first reconnect attempt."""

    reconnect_max_delay: float = 120.0
    """Upper cap for exponential back-off (seconds)."""

    reconnect_max_attempts: int = 0
    """0 = retry forever."""

    # --- watchdog ---
    watchdog_interval: float = 30.0
    """How often (seconds) the watchdog checks the connection."""

    watchdog_enabled: bool = True

    # --- startup fetch flags ---
    fetch_fields: StartupFetch = (
        StartupFetch.POSITIONS
        | StartupFetch.ORDERS_OPEN
        | StartupFetch.ORDERS_COMPLETE
        | StartupFetch.ACCOUNT_UPDATES
        | StartupFetch.EXECUTIONS
    )
