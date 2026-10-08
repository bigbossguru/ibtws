from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Sequence

import pandas as pd
from ib_async import Contract, Option, Ticker

from ibtws.unofficial.client import IBKRClient
from ibtws.unofficial.helpers import chunked, safe_pick_value, snapshot_each

from .models import ChainDefinition, OptionQuote
from .utils import (
    _filter_expirations,
    _filter_strikes,
    _ticker_to_quote,
    quotes_to_dataframe,
)

logger = logging.getLogger(__name__)

# Snapshots in a batch run concurrently and each batch takes roughly the same
# few seconds regardless of size, so fewer, larger batches are faster. 100
# stays well inside what a default TWS session accepts (verified live with
# 150 contracts per batch without error 101).
_SNAPSHOT_BATCH = 100
_QUALIFY_BATCH = 100
_SETTLE_SECS = 0.2
_DEFAULT_SNAPSHOT_TIMEOUT = 30.0
_DEFAULT_CHAIN_TTL = 900.0

_QualifyKey = tuple[str, str, float, str, str, str, str, str]


def _qualify_key(c: Option) -> _QualifyKey:
    return (
        c.symbol,
        c.lastTradeDateOrContractMonth,
        float(c.strike),
        c.right,
        c.exchange,
        c.tradingClass,
        c.multiplier,
        c.currency,
    )


class OptionChainFetcher:
    """Fetch option chain definitions and live snapshots from IB.

    Chain definitions are cached for ``chain_cache_ttl`` seconds and
    qualified option contracts for the life of the fetcher, so repeated plans
    on the same underlying (e.g. a scheduler building spreads every few
    minutes) skip the slow ``reqSecDefOptParams`` / ``qualifyContracts``
    round-trips. Pass ``chain_cache_ttl=0`` to disable the definition cache.
    """

    def __init__(
        self,
        client: IBKRClient,
        *,
        chain_cache_ttl: float = _DEFAULT_CHAIN_TTL,
        snapshot_timeout: float = _DEFAULT_SNAPSHOT_TIMEOUT,
    ) -> None:
        self._client = client
        self._chain_ttl = chain_cache_ttl
        self._snapshot_timeout = snapshot_timeout
        self._chain_cache: dict[tuple, tuple[float, ChainDefinition]] = {}
        self._qualified: dict[_QualifyKey, Option] = {}

    def clear_cache(self) -> None:
        """Drop cached chain definitions and qualified contracts."""
        self._chain_cache.clear()
        self._qualified.clear()

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

        key = (underlying.conId, exchange, trading_class)
        cached = self._chain_cache.get(key)
        if cached is not None and self._chain_ttl > 0 and time.monotonic() - cached[0] < self._chain_ttl:
            return cached[1]

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
        if chosen is None and trading_class is None and params:
            chosen = params[0]
            logger.warning(
                f"OptionChainFetcher: no {exchange} chain for {underlying.symbol}; "
                f"falling back to {chosen.exchange}/{chosen.tradingClass}. "
                f"Pass exchange= / trading_class= explicitly to avoid surprises."
            )
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
        if self._chain_ttl > 0:
            self._chain_cache[key] = (time.monotonic(), definition)
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
        """Qualify in parallel batches, silently dropping invalid contracts.

        Contracts qualified earlier by this fetcher are served from cache.
        """
        # Keys are taken before qualifying: IB fills fields in place (e.g.
        # tradingClass), which would otherwise change the key under us.
        keys = [_qualify_key(c) for c in contracts]
        missing = [(k, c) for k, c in zip(keys, contracts) if k not in self._qualified]
        if missing:
            ib = self._client.ib
            prev = ib.RaiseRequestErrors
            ib.RaiseRequestErrors = False
            try:
                results = await asyncio.gather(
                    *[ib.qualifyContractsAsync(*(c for _, c in batch)) for batch in chunked(missing, _QUALIFY_BATCH)],
                    return_exceptions=True,
                )
            finally:
                ib.RaiseRequestErrors = prev

            for batch, r in zip(chunked(missing, _QUALIFY_BATCH), results):
                if isinstance(r, BaseException):
                    logger.warning(f"OptionChainFetcher: qualify batch failed: {r}")
                    continue
                # ib_async returns results positionally, with None for failures.
                for (key, _), qualified in zip(batch, r):
                    if getattr(qualified, "conId", 0):
                        self._qualified[key] = qualified

        resolved = [self._qualified[k] for k in keys if k in self._qualified]
        dropped = len(contracts) - len(resolved)
        if dropped:
            logger.info(f"OptionChainFetcher: dropped {dropped} unresolved contract(s)")
        return resolved

    async def _request_tickers(self, contracts: list[Option]) -> list[Ticker]:
        """Subscribe, wait, snapshot, cancel.

        The streaming subscriptions are always cancelled, even when the
        snapshot fails or times out, so market-data lines never leak.
        """
        ib = self._client.ib
        subscribed: list[Option] = []
        try:
            for c in contracts:
                ib.reqMktData(c, genericTickList="100,101,104,106")
                subscribed.append(c)
            await asyncio.sleep(_SETTLE_SECS)
            # Per-contract snapshots: one unsubscribed or erroring contract
            # no longer drops the whole batch.
            snapshots = await snapshot_each(ib, contracts, timeout=self._snapshot_timeout)
            tickers = [t for t in snapshots if t is not None]
        finally:
            for c in subscribed:
                try:
                    ib.cancelMktData(c)
                except Exception:  # noqa: BLE001
                    logger.exception(f"OptionChainFetcher: cancelMktData failed for conId={getattr(c, 'conId', 0)}")
        return list(tickers)

    async def _fetch_spot(self, underlying: Contract) -> float | None:
        """Best-effort spot price from a plain snapshot (no generic ticks, no settle wait)."""
        t = await self._client.get_market_data(underlying, generic_ticks="", settle=0.0)
        if isinstance(t, Ticker):
            market_price = t.marketPrice()  # last within the spread, else mid, else close
            if market_price == market_price and market_price > 0:
                return float(market_price)
        for attr in ("last", "close"):
            v = safe_pick_value(t, attr)
            if v is not None:
                return v
        bid = safe_pick_value(t, "bid")
        ask = safe_pick_value(t, "ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None
