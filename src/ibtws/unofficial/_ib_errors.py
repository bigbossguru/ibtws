"""Structured classification of IBKR error codes.

ib_async surfaces TWS errors as numeric codes mixed across categories:
connection drops, market-data farm pings, order rejects, pacing throttles
and ordinary informational notices all flow through the same channel.
Strategies need to react differently to each — a market-data error should
trip a quote-staleness circuit breaker, a pacing error should back off the
token bucket, a connection error should defer to the reconnect loop.

Reference: https://interactivebrokers.github.io/tws-api/message_codes.html
"""

from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    CONNECTION = "connection"
    MARKET_DATA = "market_data"
    ORDER = "order"
    PACING = "pacing"
    INFO = "info"
    UNKNOWN = "unknown"


# Specific codes that override range-based classification below.
_SPECIFIC = {
    # benign farm-status / informational pings
    2104: ErrorCategory.INFO,
    2106: ErrorCategory.INFO,
    2107: ErrorCategory.INFO,
    2108: ErrorCategory.INFO,
    2119: ErrorCategory.INFO,
    2158: ErrorCategory.INFO,
    # explicit pacing violations
    100: ErrorCategory.PACING,  # Max rate of messages per second has been exceeded
    103: ErrorCategory.PACING,  # Duplicate order id
    420: ErrorCategory.PACING,  # The maximum number of historical data requests has been exceeded
    # market-data specific
    354: ErrorCategory.MARKET_DATA,  # Requested market data is not subscribed
    10089: ErrorCategory.MARKET_DATA,  # Requested market data requires additional subscription
    10090: ErrorCategory.MARKET_DATA,  # Part of requested market data is not subscribed
    10167: ErrorCategory.MARKET_DATA,  # Displaying delayed market data
    10168: ErrorCategory.MARKET_DATA,  # Requested market data is not subscribed (delayed)
    10197: ErrorCategory.MARKET_DATA,  # No market data during competing live session
    # connection-loss codes
    1100: ErrorCategory.CONNECTION,  # Connectivity between IB and TWS has been lost
    1101: ErrorCategory.CONNECTION,
    1102: ErrorCategory.CONNECTION,
    1300: ErrorCategory.CONNECTION,  # TWS socket port has been reset
    504: ErrorCategory.CONNECTION,  # Not connected
    507: ErrorCategory.CONNECTION,
}


def classify(error_code: int) -> ErrorCategory:
    """Bucket an IBKR error code into an actionable category.

    Specific codes win over ranges; falls through to coarse range buckets,
    then :attr:`ErrorCategory.UNKNOWN` for anything not recognised.
    """
    specific = _SPECIFIC.get(error_code)
    if specific is not None:
        return specific
    if 1100 <= error_code <= 1300:
        return ErrorCategory.CONNECTION
    if 2100 <= error_code <= 2200:
        return ErrorCategory.INFO
    if 300 <= error_code <= 399:
        # Many of these are market-data related (subscription, depth).
        return ErrorCategory.MARKET_DATA
    if 200 <= error_code <= 299:
        return ErrorCategory.ORDER
    if 400 <= error_code <= 499:
        return ErrorCategory.ORDER
    return ErrorCategory.UNKNOWN
