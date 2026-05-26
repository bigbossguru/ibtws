from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Optional, Sequence

import pandas as pd
from ib_async import Contract, Option, Ticker

from ibtws.unofficial._pacing import ThrottledExecutor
from ibtws.unofficial.client import IBKRClient

from .models import ChainDefinition, OptionQuote
from .utils import (
    _chunked,
    _filter_expirations,
    _filter_strikes,
    _safe_pick_value,
    _ticker_to_quote,
    quotes_to_dataframe,
)


logger = logging.getLogger(__name__)


class OptionChainFetcher:
    def __init__(self, client: IBKRClient, *, executor: Optional[ThrottledExecutor] = None) -> None:
        self._client = client
        self._executor = executor or ThrottledExecutor(max_concurrency=5, pace_per_sec=40.0)

    async def fetch_chain_definition(
        self,
        underlying: Contract,
        *,
        exchange: str = "SMART",
        trading_class: str | None = None,
    ) -> ChainDefinition:
        """Return the option universe (expirations + strikes) for an underlying.

        IB returns one parameter set per ``(exchange, tradingClass)`` combo.
        Most equities have just one; index options often have multiple (e.g.
        SPX returns both ``SPX`` AM-settled monthlies and ``SPXW`` PM-settled
        weeklies/dailies). Use ``trading_class`` to pin the variant; leave it
        ``None`` to take the first match on ``exchange``.
        """
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
            f"OptionChainFetcher: fetched chain for {underlying.symbol} @ {chosen.exchange} "
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
        batch_size: int = 50,
        as_dataframe: bool = False,
    ) -> "list[OptionQuote] | pd.DataFrame":
        contracts, spot_price = await self._build_and_qualify(
            underlying,
            exchange=exchange,
            currency=currency,
            trading_class=trading_class,
            rights=rights,
            expirations=expirations,
            expiry_from=expiry_from,
            expiry_to=expiry_to,
            strikes=strikes,
            strike_from=strike_from,
            strike_to=strike_to,
            strike_window_pct=strike_window_pct,
        )

        quotes: list[OptionQuote] = []

        for batch in _chunked(contracts, batch_size):
            tickers = await self._snapshot_batch(batch)
            for t in tickers:
                quotes.append(_ticker_to_quote(t, spot_price))

        return quotes_to_dataframe(quotes) if as_dataframe else quotes

    async def _build_and_qualify(
        self,
        underlying: Contract,
        *,
        exchange: str,
        currency: str,
        trading_class: str | None,
        rights: Sequence[str],
        expirations: Iterable[str] | None,
        expiry_from: str | None,
        expiry_to: str | None,
        strikes: Iterable[float] | None,
        strike_from: float | None,
        strike_to: float | None,
        strike_window_pct: float | None = None,
    ) -> tuple[list[Option], float | None]:
        if not underlying.conId:
            [underlying] = await self._client.ib.qualifyContractsAsync(underlying)

        definition = await self.fetch_chain_definition(underlying, exchange=exchange, trading_class=trading_class)

        selected_exp = _filter_expirations(definition.expirations, expirations, expiry_from, expiry_to)

        spot: float | None = None
        no_explicit_strikes = strikes is None and strike_from is None and strike_to is None
        if selected_exp and no_explicit_strikes and strike_window_pct is not None and strike_window_pct > 0:
            spot = await self._fetch_spot(underlying)
            if spot is not None and spot > 0:
                lo = spot * (1.0 - strike_window_pct)
                hi = spot * (1.0 + strike_window_pct)
                strike_from, strike_to = lo, hi
                logger.info(
                    f"OptionChainFetcher: auto-windowing strikes for {underlying.symbol} "
                    f"to [{lo:.2f}, {hi:.2f}] (spot={spot:.2f} ±{strike_window_pct * 100:.0f}%)"
                )

        selected_str = _filter_strikes(definition.strikes, strikes, strike_from, strike_to)

        if not selected_exp or not selected_str:
            logger.warning(
                f"OptionChainFetcher: empty filter window — {len(selected_exp)} expiries, {len(selected_str)} strikes"
            )
            return [], spot

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

        logger.info(
            f"OptionChainFetcher: qualifying {len(contracts)} candidate contracts for {definition.underlying_symbol}"
        )
        try:
            qualified = await self._client.ib.qualifyContractsAsync(*contracts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"OptionChainFetcher: qualify failed for {definition.underlying_symbol}: {exc}")
            return [], spot
        resolved = [c for c in qualified if getattr(c, "conId", 0)]
        dropped = len(qualified) - len(resolved)
        if dropped:
            logger.warning(f"OptionChainFetcher: dropped {dropped} unresolved contract(s) after qualify")
        return resolved, spot

    async def _fetch_spot(self, underlying: Contract) -> float | None:
        t = await self._client.get_market_data(underlying)
        for attr in ("last", "close"):
            v = _safe_pick_value(t, attr)
            if v is not None:
                return v
        bid = _safe_pick_value(t, "bid")
        ask = _safe_pick_value(t, "ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        logger.warning(f"OptionChainFetcher: spot for {underlying.symbol} unavailable — skipping auto-window")
        return None

    async def _snapshot_batch(self, contracts: list[Option]) -> list[Ticker]:
        for c in contracts:
            async with self._executor.slot():
                self._client.ib.reqMktData(c, genericTickList="100,101,104,106")
        await asyncio.sleep(0.5)  # give IB a moment to process the market data requests
        tickers = await self._client.ib.reqTickersAsync(*contracts)
        for c in contracts:
            self._client.ib.cancelMktData(c)
        return list(tickers)
