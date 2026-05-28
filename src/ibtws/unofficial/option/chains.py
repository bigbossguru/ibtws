from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Sequence

import pandas as pd
from ib_async import Contract, Option, Ticker

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import chunked, safe_pick_value

from .models import ChainDefinition, OptionQuote
from .utils import (
    _filter_expirations,
    _filter_strikes,
    _ticker_to_quote,
    quotes_to_dataframe,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_BATCH = 100
_SETTLE_SECS = 0.2


class OptionChainFetcher:
    """Fetch option chain definitions and live snapshots from IB."""

    def __init__(self, client: IBKRClient) -> None:
        self._client = client

    async def fetch_chain_definition(
        self,
        underlying: Contract,
        *,
        exchange: str = "SMART",
        trading_class: str | None = None,
    ) -> ChainDefinition:
        """Return the option universe (expirations + strikes) for an underlying."""
        if not underlying.conId:
            raise ValueError("Underlying contract must be qualified (conId is required).")

        params = await self._client.ib.reqSecDefOptParamsAsync(
            underlyingSymbol=underlying.symbol,
            futFopExchange="",
            underlyingSecType=underlying.secType or "STK",
            underlyingConId=underlying.conId,
        )

        def _match(p) -> bool:
            if p.exchange != exchange:
                return False
            if trading_class is not None and p.tradingClass != trading_class:
                return False
            return True

        chosen = next((p for p in params if _match(p)), None)
        if chosen is None and trading_class is None:
            chosen = params[0] if params else None
        if chosen is None:
            raise LookupError(
                f"No option parameters returned for {underlying.symbol} "
                f"(exchange={exchange}, trading_class={trading_class})."
            )

        definition = ChainDefinition(
            underlying_conId=underlying.conId,
            underlying_symbol=underlying.symbol,
            trading_class=chosen.tradingClass,
            multiplier=chosen.multiplier,
            exchange=chosen.exchange,
            expirations=tuple(sorted(chosen.expirations)),
            strikes=tuple(sorted(chosen.strikes)),
        )
        logger.info(
            f"OptionChainFetcher: chain for {underlying.symbol} @ {chosen.exchange} "
            f"({len(definition.expirations)} expiries × {len(definition.strikes)} strikes)"
        )
        return definition

    async def fetch_snapshot(
        self,
        underlying: Contract,
        *,
        exchange: str = "SMART",
        currency: str = "USD",
        trading_class: str | None = None,
        rights: Sequence[str] = ("C", "P"),
        expirations: Iterable[str] | None = None,
        expiry_from: str | None = None,
        expiry_to: str | None = None,
        strikes: Iterable[float] | None = None,
        strike_from: float | None = None,
        strike_to: float | None = None,
        strike_window_pct: float | None = 0.2,
        batch_size: int = _SNAPSHOT_BATCH,
        as_dataframe: bool = False,
    ) -> list[OptionQuote] | pd.DataFrame:
        """Fetch live option quotes for the filtered subset of the chain."""
        if not underlying.conId:
            [underlying] = await self._client.ib.qualifyContractsAsync(underlying)

        # Fetch chain definition and spot in parallel
        needs_spot = strikes is None and strike_from is None and strike_to is None and strike_window_pct
        definition_coro = self.fetch_chain_definition(underlying, exchange=exchange, trading_class=trading_class)

        if needs_spot:
            definition, spot = await asyncio.gather(definition_coro, self._fetch_spot(underlying))
        else:
            definition = await definition_coro
            spot = None

        # Filter expirations
        selected_exp = _filter_expirations(definition.expirations, expirations, expiry_from, expiry_to)

        # Auto-window strikes around spot
        if needs_spot and selected_exp:
            if spot is None or spot <= 0:
                logger.error(f"OptionChainFetcher: spot unavailable for {underlying.symbol}, aborting")
                return pd.DataFrame() if as_dataframe else []
            assert strike_window_pct is not None  # guaranteed by needs_spot
            strike_from = spot * (1.0 - strike_window_pct)
            strike_to = spot * (1.0 + strike_window_pct)
            logger.info(
                f"OptionChainFetcher: strike window [{strike_from:.2f}, {strike_to:.2f}] "
                f"(spot={spot:.2f} ±{strike_window_pct * 100:.0f}%)"
            )

        # Filter strikes and build contracts
        selected_str = _filter_strikes(definition.strikes, strikes, strike_from, strike_to)
        if not selected_exp or not selected_str:
            logger.warning(
                f"OptionChainFetcher: empty filter — {len(selected_exp)} expiries, {len(selected_str)} strikes"
            )
            return pd.DataFrame() if as_dataframe else []

        contracts = [
            Option(
                symbol=definition.underlying_symbol,
                lastTradeDateOrContractMonth=exp,
                strike=strike,
                right=right,
                exchange=definition.exchange,
                tradingClass=definition.trading_class,
                multiplier=definition.multiplier,
                currency=currency,
            )
            for exp in selected_exp
            for strike in selected_str
            for right in rights
        ]
        logger.info(f"OptionChainFetcher: requesting tickers for {len(contracts)} contracts")

        # Qualify contracts (reqTickersAsync requires conId for hashing)
        contracts = await self._qualify(contracts)
        if not contracts:
            return pd.DataFrame() if as_dataframe else []

        # Fetch market data
        quotes: list[OptionQuote] = []
        for batch in chunked(contracts, batch_size):
            tickers = await self._request_tickers(batch)
            for t in tickers:
                q = _ticker_to_quote(t, spot)
                if q.bid is not None or q.ask is not None or q.iv is not None:
                    quotes.append(q)

        dropped = len(contracts) - len(quotes)
        if dropped:
            logger.info(f"OptionChainFetcher: filtered out {dropped} empty quote(s)")

        return quotes_to_dataframe(quotes) if as_dataframe else quotes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _qualify(self, contracts: list[Option]) -> list[Option]:
        """Qualify in parallel batches, silently dropping invalid contracts."""
        ib = self._client.ib
        prev = ib.RaiseRequestErrors
        ib.RaiseRequestErrors = False
        try:
            results = await asyncio.gather(
                *[ib.qualifyContractsAsync(*batch) for batch in chunked(contracts, _SNAPSHOT_BATCH)],
                return_exceptions=True,
            )
        finally:
            ib.RaiseRequestErrors = prev

        resolved: list[Option] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning(f"OptionChainFetcher: qualify batch failed: {r}")
            else:
                resolved.extend(c for c in r if getattr(c, "conId", 0))

        dropped = len(contracts) - len(resolved)
        if dropped:
            logger.info(f"OptionChainFetcher: dropped {dropped} unresolved contract(s)")
        return resolved

    async def _request_tickers(self, contracts: list[Option]) -> list[Ticker]:
        """Subscribe, wait, snapshot, cancel."""
        ib = self._client.ib
        for c in contracts:
            ib.reqMktData(c, genericTickList="100,101,104,106")
        await asyncio.sleep(_SETTLE_SECS)
        tickers = await ib.reqTickersAsync(*contracts)
        for c in contracts:
            ib.cancelMktData(c)
        return list(tickers)

    async def _fetch_spot(self, underlying: Contract) -> float | None:
        """Best-effort spot price."""
        t = await self._client.get_market_data(underlying)
        for attr in ("marketPrice", "last", "close"):
            v = safe_pick_value(t, attr)
            if v is not None:
                return v
        bid = safe_pick_value(t, "bid")
        ask = safe_pick_value(t, "ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None
