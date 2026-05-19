"""Throttled, fault-tolerant option-chain snapshot fetcher built on :class:`IBKRClient`."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable, Optional, Sequence

import pandas as pd
from ib_async import Contract, Option, Ticker

from ibtws.unofficial.client import IBKRClient

from .models import ChainDefinition, OptionQuote
from .utils import (
    _chunked,
    _filter_expirations,
    _filter_strikes,
    _pick_price,
    _ticker_to_quote,
    quotes_to_dataframe,
)


logger = logging.getLogger(__name__)


class OptionChainFetcher:
    """Reliable, throttled option-chain quote fetcher built on :class:`IBKRClient`.

    The fetcher does not manage the IB connection — the caller owns
    ``connect()`` / ``disconnect()``. Pacing is enforced via a semaphore
    (``max_concurrency``) plus a token-bucket delay (``pace_per_sec``) to stay
    under IB's ~50 msg/s ceiling. Per-batch ``snapshot_timeout`` guarantees
    forward progress when IB silently drops a snapshot.
    """

    def __init__(
        self,
        client: IBKRClient,
        *,
        max_concurrency: int = 25,
        pace_per_sec: float = 40.0,
        snapshot_timeout: float = 20.0,
    ) -> None:
        self._client = client
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._min_interval = 1.0 / pace_per_sec if pace_per_sec > 0 else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()
        self._snapshot_timeout = snapshot_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_chain_definition(
        self,
        underlying: Contract,
        *,
        exchange: str = "SMART",
        trading_class: Optional[str] = None,
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

        async with self._slot():
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
        trading_class: Optional[str] = None,
        rights: Sequence[str] = ("C", "P"),
        expirations: Optional[Iterable[str]] = None,
        expiry_from: Optional[str] = None,
        expiry_to: Optional[str] = None,
        strikes: Optional[Iterable[float]] = None,
        strike_from: Optional[float] = None,
        strike_to: Optional[float] = None,
        strike_window_pct: Optional[float] = 0.2,
        batch_size: int = 50,
        as_dataframe: bool = False,
    ) -> "list[OptionQuote] | pd.DataFrame":
        """Resolve and quote a slice of the option chain in batched snapshots.

        Fault-tolerant: failed qualify or snapshot batches are logged at
        WARNING and excluded, so the caller always gets a partial answer
        rather than an exception. Set ``as_dataframe=True`` to return a
        ``pandas.DataFrame`` via :func:`quotes_to_dataframe`.
        """
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
            quotes.extend(_ticker_to_quote(t, spot_price) for t in tickers)

        return quotes_to_dataframe(quotes) if as_dataframe else quotes

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _build_and_qualify(
        self,
        underlying: Contract,
        *,
        exchange: str,
        currency: str,
        trading_class: Optional[str],
        rights: Sequence[str],
        expirations: Optional[Iterable[str]],
        expiry_from: Optional[str],
        expiry_to: Optional[str],
        strikes: Optional[Iterable[float]],
        strike_from: Optional[float],
        strike_to: Optional[float],
        strike_window_pct: Optional[float] = None,
    ) -> tuple[list[Option], Optional[float]]:
        if not underlying.conId:
            (underlying,) = await self._client.qualify(underlying)

        definition = await self.fetch_chain_definition(underlying, exchange=exchange, trading_class=trading_class)

        selected_exp = _filter_expirations(definition.expirations, expirations, expiry_from, expiry_to)

        spot: Optional[float] = None
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
            qualified = await self._client.qualify(*contracts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"OptionChainFetcher: qualify failed for {definition.underlying_symbol}: {exc}")
            return [], spot
        resolved = [c for c in qualified if getattr(c, "conId", 0)]
        dropped = len(qualified) - len(resolved)
        if dropped:
            logger.warning(f"OptionChainFetcher: dropped {dropped} unresolved contract(s) after qualify")
        return resolved, spot

    async def _fetch_spot(self, underlying: Contract) -> Optional[float]:
        """Best-effort one-shot fetch of the underlying's spot price."""
        async with self._slot():
            try:
                tickers = await asyncio.wait_for(
                    self._client.ib.reqTickersAsync(underlying, regulatorySnapshot=False),
                    timeout=self._snapshot_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"OptionChainFetcher: spot lookup for {underlying.symbol} failed: {exc}")
                return None
        if not tickers:
            return None
        t = tickers[0]
        for attr in ("last", "close"):
            v = _pick_price(t, attr)
            if v is not None:
                return v
        bid = _pick_price(t, "bid")
        ask = _pick_price(t, "ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        logger.warning(f"OptionChainFetcher: spot for {underlying.symbol} unavailable — skipping auto-window")
        return None

    async def _snapshot_batch(self, contracts: list[Option]) -> list[Ticker]:
        async with self._slot():
            try:
                [
                    self._client.ib.reqMktData(c, genericTickList="100,101,106") for c in contracts
                ]  # pre-warm market data subscriptions to reduce snapshot latency
                await asyncio.sleep(0.2)  # give IB a moment to process the market data requests
                tickers = await asyncio.wait_for(
                    self._client.ib.reqTickersAsync(*contracts, regulatorySnapshot=False),
                    timeout=self._snapshot_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"OptionChainFetcher: snapshot batch of {len(contracts)} "
                    f"timed out after {self._snapshot_timeout:.1f}s"
                )
                return []
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"OptionChainFetcher: snapshot batch failed: {exc}")
                return []
        return list(tickers)

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        """Acquire concurrency + pacing slot; release semaphore on exit."""
        async with self._semaphore:
            await self._await_next_slot()
            yield

    async def _await_next_slot(self) -> None:
        """Sleep until the next pacing slot opens; no-op when pacing disabled."""
        if self._min_interval <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._min_interval
