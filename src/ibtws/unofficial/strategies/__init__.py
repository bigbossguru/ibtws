"""High-level option strategies built on top of the order & option layers."""

from .credit_spread import (
    CreditSpreadError,
    CreditSpreadParams,
    CreditSpreadPlan,
    CreditSpreadStrategy,
    SpreadLeg,
    SpreadType,
    select_expiry,
    select_long_leg,
    select_short_leg,
)

__all__ = [
    "CreditSpreadError",
    "CreditSpreadParams",
    "CreditSpreadPlan",
    "CreditSpreadStrategy",
    "SpreadLeg",
    "SpreadType",
    "select_expiry",
    "select_long_leg",
    "select_short_leg",
]
