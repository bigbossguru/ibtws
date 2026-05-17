"""Option-chain retrieval and snapshot quoting for IBKR via :class:`IBKRClient`.

Public surface:

* :class:`ChainDefinition`    — metadata returned by IB for an underlying.
* :class:`OptionQuote`        — one resolved option contract + market metrics.
* :class:`OptionChainFetcher` — the orchestrator class.
* :func:`quotes_to_dataframe` — flatten quotes into a pandas DataFrame.
* :data:`DATAFRAME_COLUMNS`   — column order used by the DataFrame projection.
"""

from .fetcher import OptionChainFetcher
from .models import ChainDefinition, OptionQuote
from .utils import DATAFRAME_COLUMNS, quotes_to_dataframe

__all__ = [
    "ChainDefinition",
    "DATAFRAME_COLUMNS",
    "OptionChainFetcher",
    "OptionQuote",
    "quotes_to_dataframe",
]
