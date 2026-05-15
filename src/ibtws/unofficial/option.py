"""Option-chain retrieval and quote fetching for IBKR via ``IBKRClient``.

Covers the full pipeline a strategy needs:

1. Discover the option universe for an underlying (expirations + strikes) using
   ``reqSecDefOptParamsAsync`` and cache it with a TTL.
2. Build :class:`ib_async.Option` contracts for the requested filter window.
3. Qualify contracts in conservative batches.
4. Quote them either as one-shot snapshots (via ``reqTickersAsync``) or as a
   live streaming subscription (context-managed so the 100-line market-data
   quota is always released).

Designed around the realities of TWS / IB Gateway:

* Pacing — concurrent in-flight requests are capped by a semaphore; bursts are
  spaced by a token-bucket-style delay to stay under ~50 req/s.
* Greeks / IV / OI — these arrive as **model Greek** ticks, not bid/ask, so the
  generic tick list explicitly requests them and the snapshot path uses
  ``reqTickersAsync`` which blocks until the model tick has been delivered.
* Frozen data — when the market is closed, quotes flatline at NaN; this module
  does not switch ``MarketDataType`` automatically (caller should call
  ``client.ib.reqMarketDataType(...)`` once at startup).
* Cleanup — the streaming subscription is exposed as an ``async with`` context
  manager so cancellations happen even when the caller raises.

Public surface
--------------
* :class:`ChainDefinition`  — cached metadata returned by IB for an underlying.
* :class:`OptionQuote`      — one resolved option contract + its market metrics.
* :class:`OptionChainFetcher` — the orchestrator class.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Optional, Sequence

import pandas as pd
from ib_async import Contract, Option, Ticker

from ibtws.unofficial.client import IBKRClient


logger = logging.getLogger(__name__)


# Generic tick list for option Greeks, IV, open interest, and historical
# volatility. Without these the model Greeks never arrive.
#   100 — option volume
#   101 — option open interest
#   104 — historical volatility
#   106 — option implied volatility
#   165 — misc stats
#   221 — mark price
#   225 — auction values
#   233 — RT volume
#   236 — shortable
#   258 — fundamental ratios
_OPTION_GENERIC_TICKS = "100,101,104,106,165,221,225,233,236,258"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainDefinition:
    """Universe of option contracts available for an underlying on one exchange."""

    underlying_conId: int
    underlying_symbol: str
    trading_class: str
    multiplier: str
    exchange: str
    expirations: tuple[str, ...]  # YYYYMMDD strings
    strikes: tuple[float, ...]


@dataclass
class OptionQuote:
    """Snapshot of one option contract's market metrics at a point in time."""

    contract: Option
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mark: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None

    # From modelGreeks (preferred — uses IB's pricing model with current vol surface).
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    underlying_price: Optional[float] = None

    timestamp: float = field(default_factory=time.time)


@dataclass
class StreamingSubscription:
    """Handle returned by :meth:`OptionChainFetcher.subscribe`.

    ``tickers`` is the list of live ``Ticker`` objects; ib_async mutates their
    fields in place as new ticks arrive. Exit the surrounding ``async with``
    block to cancel all subscriptions.
    """

    tickers: list[Ticker]


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class OptionChainFetcher:
    """Reliable, throttled option-chain quote fetcher built on :class:`IBKRClient`.

    Parameters
    ----------
    client:
        A connected :class:`IBKRClient`. The fetcher does not manage the
        connection — the caller is responsible for ``connect()``/``disconnect()``.
    max_concurrency:
        Hard cap on concurrent qualify/quote requests in flight. Defaults to 25
        which is comfortably under IB's 50 req/s pacing limit.
    pace_per_sec:
        Soft request budget per second. Each acquire is throttled so the burst
        rate stays under this value.
    cache_ttl:
        Seconds to cache a :class:`ChainDefinition` per underlying. Defaults to
        one hour — chain definitions rarely change intraday.
    snapshot_timeout:
        Per-batch timeout for ``reqTickersAsync``. IB sometimes drops snapshots
        silently; the timeout guarantees forward progress.
    """

    def __init__(
        self,
        client: IBKRClient,
        *,
        max_concurrency: int = 25,
        pace_per_sec: float = 40.0,
        cache_ttl: float = 3600.0,
        snapshot_timeout: float = 15.0,
    ) -> None:
        """
        Build a fetcher bound to an existing :class:`IBKRClient`.

        No I/O happens here; all IB calls are deferred to the public methods.

        Parameters
        ----------
        client:
            A connected :class:`IBKRClient`. Connection lifecycle is the
            caller's responsibility — this class never calls ``connect`` or
            ``disconnect`` on the client.
        max_concurrency:
            Hard upper bound on the number of in-flight ``qualify`` /
            ``snapshot`` / ``subscribe`` requests at any moment, enforced via
            ``asyncio.Semaphore``. Default 25 leaves headroom under IB's
            documented 50 msg/s ceiling per connection.
        pace_per_sec:
            Soft request-rate budget. Each acquired slot is delayed so that
            successive acquires are spaced by at least ``1 / pace_per_sec``
            seconds. Set to 0 to disable pacing entirely.
        cache_ttl:
            How long (seconds) a fetched :class:`ChainDefinition` remains
            valid in the in-memory cache. Default 1 hour — chain definitions
            essentially never change intraday but can change overnight when
            new strikes/expiries are listed.
        snapshot_timeout:
            Timeout (seconds) applied to each ``reqTickersAsync`` call.
            Without this, a single hung snapshot can stall the whole batch.
        """
        self._client = client
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._min_interval = 1.0 / pace_per_sec if pace_per_sec > 0 else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()
        self._cache_ttl = cache_ttl
        self._snapshot_timeout = snapshot_timeout
        self._chain_cache: dict[tuple[str, str, str], tuple[float, ChainDefinition]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_chain_definition(
        self,
        underlying: Contract,
        *,
        exchange: str = "SMART",
        trading_class: Optional[str] = None,
        force_refresh: bool = False,
    ) -> ChainDefinition:
        """
        Return the option universe (expirations + strikes) for an underlying.

        Wraps ``IB.reqSecDefOptParamsAsync``. The result is cached per
        ``(symbol, exchange, trading_class)`` for ``cache_ttl`` seconds.

        IB returns one parameter set per ``(exchange, tradingClass)`` combo.
        For most equities there's only one (e.g. ``AAPL`` on SMART), but
        index options often list multiple — SPX returns both
        ``trading_class="SPX"`` (AM-settled monthlies) and ``"SPXW"``
        (PM-settled weeklies/dailies). Use ``trading_class`` to pick the one
        you want; leave it ``None`` to take the first match on ``exchange``.

        Parameters
        ----------
        underlying:
            A **qualified** ``Contract`` (must already have a ``conId``).
        exchange:
            Preferred option exchange. Defaults to ``"SMART"``.
        trading_class:
            Optional trading-class filter (e.g. ``"SPX"`` or ``"SPXW"``).
            When given, only parameter sets matching BOTH ``exchange`` AND
            ``tradingClass`` are considered.
        force_refresh:
            If True, bypass the cache and re-fetch from IB.

        Returns
        -------
        ChainDefinition

        Raises
        ------
        ValueError
            If ``underlying.conId`` is falsy.
        LookupError
            If IB returns no parameter sets matching the filters.
        """
        if not underlying.conId:
            raise ValueError("Underlying contract must be qualified (conId is required).")

        key = (underlying.symbol, exchange, trading_class or "")
        now = time.monotonic()

        if not force_refresh and key in self._chain_cache:
            ts, cached = self._chain_cache[key]
            if now - ts < self._cache_ttl:
                logger.debug("OptionChainFetcher: chain cache hit for %s @ %s/%s", *key)
                return cached

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
            # Fall back to anything IB returned (only when trading_class isn't pinned).
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
        self._chain_cache[key] = (now, definition)
        logger.info(
            "OptionChainFetcher: cached chain for %s @ %s (%d expiries × %d strikes)",
            underlying.symbol,
            chosen.exchange,
            len(definition.expirations),
            len(definition.strikes),
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
    ) -> list[OptionQuote]:
        """
        Resolve and quote a slice of the option chain in batched snapshots.

        Pipeline:

        1. Discover the chain via :meth:`fetch_chain_definition` (cached).
        2. Filter expirations and strikes by the supplied criteria.
        3. Build ``Option`` contracts for the cartesian product
           ``expirations × strikes × rights``.
        4. Qualify them in throttled batches; drop any that IB cannot resolve.
        5. Snapshot each batch via ``IB.reqTickersAsync`` (which natively
           waits until model Greeks have arrived) and convert to
           :class:`OptionQuote`.

        The method is fault-tolerant: a failed qualify or snapshot batch is
        logged at WARNING and excluded from the result, so the caller always
        gets a partial answer rather than an exception. Use the returned
        list's length to detect drop-outs.

        Parameters
        ----------
        underlying:
            Underlying ``Contract`` (will be auto-qualified if ``conId == 0``).
        exchange:
            Option exchange to draw the chain from. Defaults to ``"SMART"``.
        currency:
            Contract currency. Defaults to ``"USD"``.
        rights:
            Subset of ``("C", "P")`` — calls only, puts only, or both.
        expirations:
            Explicit expiry list (``YYYYMMDD`` strings). Mutually exclusive
            with ``expiry_from``/``expiry_to``: when given, the range params
            are ignored.
        expiry_from, expiry_to:
            Inclusive expiry bounds (``YYYYMMDD``). Either may be ``None``.
        strikes:
            Explicit strike list. Mutually exclusive with the strike range.
        strike_from, strike_to:
            Inclusive strike bounds. Either may be ``None``.
        strike_window_pct:
            Auto-narrow strikes to ``spot * (1 ± pct)`` when no explicit
            ``strikes`` / ``strike_from`` / ``strike_to`` is supplied. Default
            ``0.2`` (±20% of spot) avoids requesting hundreds of phantom deep
            OTM/ITM strikes that aren't actually listed for the requested
            expiries (which causes IB error 200 noise). Set to ``None`` to
            disable and request every strike in the chain definition.
        batch_size:
            Number of contracts per ``reqTickersAsync`` call. Default 50.
            Smaller = more round-trips but tighter pacing; larger = fewer
            round-trips but bigger blast radius if a batch times out.

        Returns
        -------
        list[OptionQuote]
            One quote per successfully qualified + quoted contract. Empty
            list if the filter window matches nothing.
        """
        contracts = await self._build_and_qualify(
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
        if not contracts:
            return []

        quotes: list[OptionQuote] = []
        for batch in _chunked(contracts, batch_size):
            tickers = await self._snapshot_batch(batch)
            quotes.extend(_ticker_to_quote(t) for t in tickers if t is not None)
        return quotes

    async def fetch_snapshot_df(
        self,
        underlying: Contract,
        **kwargs: Any,
    ) -> "pd.DataFrame":
        """
        Convenience wrapper around :meth:`fetch_snapshot` returning a DataFrame.

        Identical pipeline and filtering options as :meth:`fetch_snapshot`;
        the only difference is the return type — quotes are converted via
        :func:`quotes_to_dataframe` for direct use in pandas pipelines
        (screening, surface fitting, exporting to parquet/CSV, etc.).

        Parameters
        ----------
        underlying:
            See :meth:`fetch_snapshot`.
        **kwargs:
            Forwarded verbatim to :meth:`fetch_snapshot`. See that method's
            docstring for the full list (``exchange``, ``currency``, ``rights``,
            ``expirations``, ``expiry_from``, ``expiry_to``, ``strikes``,
            ``strike_from``, ``strike_to``, ``batch_size``).

        Returns
        -------
        pandas.DataFrame
            One row per :class:`OptionQuote` with columns
            ``symbol, expiry, strike, right, bid, ask, last, mark, volume,
            open_interest, iv, delta, gamma, vega, theta, underlying_price,
            timestamp``. Empty DataFrame (with the same columns) if no
            contracts qualified.
        """
        quotes = await self.fetch_snapshot(underlying, **kwargs)
        return quotes_to_dataframe(quotes)

    @asynccontextmanager
    async def subscribe(
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
    ) -> AsyncIterator[StreamingSubscription]:
        """
        Open a live streaming subscription for the requested option universe.

        Yields a :class:`StreamingSubscription` whose ``tickers`` field
        contains one ``ib_async.Ticker`` per contract. ib_async mutates the
        ticker fields in place as new ticks arrive, so reading
        ``ticker.bid``, ``ticker.modelGreeks.delta``, etc. is always live.

        **Must be used as ``async with``.** The context manager calls
        ``cancelMktData`` for every subscribed contract on exit (including on
        exception), which is critical because IB caps simultaneous market-data
        lines at 100 per connection. Leaking subscriptions silently is one of
        the most common ways to brick a TWS session.

        Parameters
        ----------
        underlying:
            Underlying ``Contract`` (auto-qualified if ``conId == 0``).
        exchange, currency, rights, expirations, expiry_from, expiry_to,
        strikes, strike_from, strike_to:
            Same semantics as :meth:`fetch_snapshot`.
        strike_window_pct:
            Same semantics as :meth:`fetch_snapshot` — auto-narrow strikes to
            ±pct around spot when no explicit strike filter is given.

        Yields
        ------
        StreamingSubscription
            Handle holding the live ticker objects. Inspect them inside the
            ``async with`` block; they become stale (and will be cancelled)
            on exit.

        Examples
        --------
        ::

            async with fetcher.subscribe(stk, expirations=[exp]) as sub:
                await asyncio.sleep(30)
                for t in sub.tickers:
                    print(t.contract.localSymbol, t.bid, t.ask)
        """
        contracts = await self._build_and_qualify(
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
        ib = self._client.ib
        tickers: list[Ticker] = []
        try:
            for c in contracts:
                async with self._slot():
                    tickers.append(ib.reqMktData(c, _OPTION_GENERIC_TICKS, snapshot=False, regulatorySnapshot=False))
            yield StreamingSubscription(tickers=tickers)
        finally:
            for t in tickers:
                try:
                    ib.cancelMktData(t.contract)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    logger.debug("OptionChainFetcher: cancelMktData failed for %s", t.contract, exc_info=True)

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
    ) -> list[Option]:
        """
        Build the cartesian product of contracts and qualify them.

        Shared engine behind :meth:`fetch_snapshot` and :meth:`subscribe`.
        Walks through:

        1. Auto-qualify the underlying if it lacks a ``conId``.
        2. Pull (or cache-hit) the chain definition.
        3. Apply expiry + strike filters.
        4. Build one ``Option`` instance per
           ``(expiry, strike, right)`` combination.
        5. Qualify the batch via :meth:`_qualify_in_batches`.

        Returns an empty list (and logs WARNING) if filtering excludes every
        expiry or every strike.

        Parameters
        ----------
        underlying:
            Underlying contract; mutated in place if auto-qualified.
        exchange, currency, rights, expirations, expiry_from, expiry_to,
        strikes, strike_from, strike_to:
            Same semantics as the public methods that call this.

        Returns
        -------
        list[Option]
            Fully qualified option contracts ready for quoting.
        """
        # Make sure the underlying is qualified before pulling its chain.
        if not underlying.conId:
            (underlying,) = await self._client.qualify(underlying)

        definition = await self.fetch_chain_definition(underlying, exchange=exchange, trading_class=trading_class)

        selected_exp = _filter_expirations(definition.expirations, expirations, expiry_from, expiry_to)

        no_explicit_strikes = strikes is None and strike_from is None and strike_to is None
        if selected_exp and no_explicit_strikes and strike_window_pct is not None and strike_window_pct > 0:
            spot = await self._fetch_spot(underlying)
            if spot is not None and spot > 0:
                lo = spot * (1.0 - strike_window_pct)
                hi = spot * (1.0 + strike_window_pct)
                strike_from, strike_to = lo, hi
                logger.info(
                    "OptionChainFetcher: auto-windowing strikes for %s to [%.2f, %.2f] (spot=%.2f ±%.0f%%)",
                    underlying.symbol,
                    lo,
                    hi,
                    spot,
                    strike_window_pct * 100,
                )

        selected_str = _filter_strikes(definition.strikes, strikes, strike_from, strike_to)

        if not selected_exp or not selected_str:
            logger.warning(
                "OptionChainFetcher: empty filter window — %d expiries, %d strikes",
                len(selected_exp),
                len(selected_str),
            )
            return []

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
            "OptionChainFetcher: qualifying %d candidate contracts for %s",
            len(contracts),
            definition.underlying_symbol,
        )
        return await self._qualify_in_batches(contracts)

    async def _qualify_in_batches(self, contracts: list[Option], batch_size: int = 50) -> list[Option]:
        """
        Resolve ``conId`` / ``primaryExchange`` for each contract in batches.

        Splits ``contracts`` into chunks of ``batch_size`` and runs each chunk
        through ``IB.qualifyContractsAsync`` under the throttling slot.
        Failures (network errors, ambiguous contracts, dropped entries) are
        logged at WARNING and excluded — the caller still gets back whatever
        succeeded.

        Parameters
        ----------
        contracts:
            Unqualified ``Option`` instances (must have at minimum symbol +
            expiry + strike + right + exchange).
        batch_size:
            Number of contracts per IB round-trip. Default 50 keeps each
            message comfortably small.

        Returns
        -------
        list[Option]
            Subset of *contracts* that came back with a non-zero ``conId``.
        """
        resolved: list[Option] = []
        dropped = 0
        for batch in _chunked(contracts, batch_size):
            async with self._slot():
                try:
                    await self._client.ib.qualifyContractsAsync(*batch)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("OptionChainFetcher: qualify batch failed (%s) — dropping batch", exc)
                    dropped += len(batch)
                    continue
            for c in batch:
                if isinstance(c, Option) and c.conId:
                    resolved.append(c)
                else:
                    dropped += 1
                    logger.debug(
                        "OptionChainFetcher: dropping unqualified %s %s %s%s (no conId)",
                        c.symbol,
                        c.lastTradeDateOrContractMonth,
                        c.strike,
                        c.right,
                    )
        logger.info(
            "OptionChainFetcher: qualified %d/%d contracts (%d dropped — strike/expiry not listed)",
            len(resolved),
            len(contracts),
            dropped,
        )
        return resolved

    async def _fetch_spot(self, underlying: Contract) -> Optional[float]:
        """
        Best-effort one-shot fetch of the underlying's spot price.

        Tries ``last``, then ``close``, then mid of ``bid``/``ask``. Returns
        ``None`` (and logs WARNING) if every field is NaN/missing or IB
        rejects the request — the caller should then fall back to requesting
        the whole strike list.
        """
        async with self._slot():
            try:
                tickers = await asyncio.wait_for(
                    self._client.ib.reqTickersAsync(underlying, regulatorySnapshot=False),
                    timeout=self._snapshot_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("OptionChainFetcher: spot lookup for %s failed: %s", underlying.symbol, exc)
                return None
        if not tickers:
            return None
        t = tickers[0]
        for raw in (
            getattr(t, "last", None),
            getattr(t, "close", None),
            t.marketPrice() if callable(getattr(t, "marketPrice", None)) else None,
        ):
            v = _safe_float(raw)
            if v is not None and v > 0:
                return v
        bid = _safe_float(getattr(t, "bid", None))
        ask = _safe_float(getattr(t, "ask", None))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        logger.warning("OptionChainFetcher: spot for %s unavailable — skipping auto-window", underlying.symbol)
        return None

    async def _snapshot_batch(self, contracts: list[Option]) -> list[Optional[Ticker]]:
        """
        Snapshot a batch of contracts via short-lived streaming subscriptions.

        IB's "true" snapshot mode (``reqTickersAsync`` / ``reqMktData(snapshot=True)``)
        silently drops generic ticks — meaning open interest (101) and a few
        other extended fields never arrive. To collect OI we instead start a
        regular streaming subscription with the generic tick list, wait until
        each ticker is "complete" (or a per-batch timeout fires), then cancel.

        A ticker is considered complete when it has at least one price field
        (bid or last) AND model Greeks AND open interest. Whatever has arrived
        when the timeout fires is returned anyway, so very illiquid contracts
        without OI still come back with bid/ask/Greeks.

        Always cancels every subscription on the way out, even on exception,
        so the 100-line market-data quota is never leaked.
        """
        ib = self._client.ib
        tickers: list[Ticker] = []
        try:
            for c in contracts:
                async with self._slot():
                    tickers.append(ib.reqMktData(c, _OPTION_GENERIC_TICKS, snapshot=False, regulatorySnapshot=False))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(self._wait_for_quote(t) for t in tickers)),
                    timeout=self._snapshot_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "OptionChainFetcher: snapshot batch of %d incomplete after %.1fs — returning partial data",
                    len(contracts),
                    self._snapshot_timeout,
                )
            return list(tickers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OptionChainFetcher: snapshot batch failed: %s", exc)
            return []
        finally:
            for t in tickers:
                try:
                    ib.cancelMktData(t.contract)
                except Exception:  # noqa: BLE001
                    logger.debug("OptionChainFetcher: cancelMktData failed for %s", t.contract, exc_info=True)

    async def _wait_for_quote(self, ticker: Ticker, poll_interval: float = 0.1) -> None:
        """
        Poll a streaming ticker until bid/last + Greeks + OI have all arrived.

        Returns as soon as the ticker is "complete" or, if it never gets
        there, on the next poll after the caller's outer ``wait_for`` cancels
        this coroutine.
        """
        while not _quote_is_complete(ticker):
            await asyncio.sleep(poll_interval)

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        """
        Acquire one concurrency slot AND a pacing slot, then yield.

        Combines two layers of throttling:

        * ``self._semaphore`` enforces ``max_concurrency`` in flight at once;
        * :meth:`_await_next_slot` ensures successive acquires are spaced by
          ``1 / pace_per_sec``.

        Used as ``async with self._slot(): ...`` around every IB call. On
        exit the semaphore is released; the pacing token is consumed
        regardless of whether the wrapped call succeeds.
        """
        async with self._semaphore:
            await self._await_next_slot()
            yield

    async def _await_next_slot(self) -> None:
        """
        Sleep, if necessary, until the next pacing-slot opens.

        Implements a single-token bucket guarded by a lock so concurrent
        callers don't all decide it's their turn simultaneously. The next
        permitted timestamp (``self._next_slot``) is advanced to ``now +
        min_interval`` after each call.

        No-op when ``pace_per_sec == 0`` (pacing disabled).
        """
        if self._min_interval <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._min_interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked(seq: Sequence, size: int):
    """
    Yield successive ``size``-length slices of *seq*.

    Equivalent to ``itertools.batched`` (3.12+) but works on any
    ``Sequence`` and returns slice views rather than tuples — preserving the
    original element type, which matters for ``list[Option]`` here.

    Parameters
    ----------
    seq:
        Any indexable sequence.
    size:
        Maximum chunk size; the final chunk may be shorter.
    """
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _filter_expirations(
    available: tuple[str, ...],
    explicit: Optional[Iterable[str]],
    expiry_from: Optional[str],
    expiry_to: Optional[str],
) -> list[str]:
    """
    Narrow the chain's expirations to the caller-requested subset.

    ``explicit`` wins when supplied — only expirations that appear in BOTH
    *available* and *explicit* are returned (set intersection by string
    equality, ``YYYYMMDD`` format).

    Otherwise applies an inclusive range filter using lexicographic comparison
    (which is equivalent to chronological for ``YYYYMMDD`` strings). Either
    bound may be ``None`` to leave that side open.

    Parameters
    ----------
    available:
        All expirations IB returned for the underlying.
    explicit:
        Caller-provided whitelist; ``None`` falls through to the range.
    expiry_from, expiry_to:
        Inclusive lower / upper bound (``YYYYMMDD``).

    Returns
    -------
    list[str]
        Surviving expirations in the original order from *available*.
    """
    if explicit is not None:
        wanted = set(explicit)
        return [e for e in available if e in wanted]
    return [e for e in available if (expiry_from is None or e >= expiry_from) and (expiry_to is None or e <= expiry_to)]


def _filter_strikes(
    available: tuple[float, ...],
    explicit: Optional[Iterable[float]],
    strike_from: Optional[float],
    strike_to: Optional[float],
) -> list[float]:
    """
    Narrow the chain's strikes to the caller-requested subset.

    Same semantics as :func:`_filter_expirations` but for floats. The explicit
    list match is exact float equality — be aware that IB returns strikes as
    floats and that direct equality on, say, ``150.0`` vs ``150`` is fine but
    ``0.1 + 0.2`` is not. Pass strike values as you see them in the chain.

    Parameters
    ----------
    available:
        All strikes IB returned, sorted ascending.
    explicit:
        Caller-provided whitelist; ``None`` falls through to the range.
    strike_from, strike_to:
        Inclusive lower / upper bound. Either may be ``None``.

    Returns
    -------
    list[float]
        Surviving strikes in ascending order.
    """
    if explicit is not None:
        wanted = set(explicit)
        return [s for s in available if s in wanted]
    return [s for s in available if (strike_from is None or s >= strike_from) and (strike_to is None or s <= strike_to)]


def _safe_float(value) -> Optional[float]:
    """
    Coerce IB's NaN-laden numeric fields into clean ``Optional[float]`` values.

    IB transmits "no data" as ``NaN`` (and occasionally as ``None`` or odd
    types), which propagates silently through arithmetic and breaks downstream
    code that expects either a real number or an explicit ``None``. This
    helper normalises every variant to ``None`` so consumers can do
    ``if quote.bid is not None: ...``.

    Parameters
    ----------
    value:
        Raw value from a ``Ticker`` field. Anything that isn't a finite float
        becomes ``None``.

    Returns
    -------
    Optional[float]
        ``None`` for ``None`` / ``NaN`` / non-numeric inputs; the float
        otherwise.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _quote_is_complete(ticker: Ticker) -> bool:
    """True once the streaming ticker has price + Greeks + OI populated."""
    has_price = (
        _safe_float(getattr(ticker, "bid", None)) is not None or _safe_float(getattr(ticker, "last", None)) is not None
    )
    has_greeks = getattr(ticker, "modelGreeks", None) is not None
    oi_raw = (
        getattr(ticker, "callOpenInterest", None)
        if getattr(ticker.contract, "right", None) == "C"
        else getattr(ticker, "putOpenInterest", None)
    )
    has_oi = _safe_float(oi_raw) is not None
    return has_price and has_greeks and has_oi


def _ticker_to_quote(ticker: Ticker) -> OptionQuote:
    """
    Map an ib_async ``Ticker`` snapshot into an :class:`OptionQuote`.

    Pulls bid / ask / last / mark / volume directly, then routes to the
    correct open-interest field (``callOpenInterest`` for calls,
    ``putOpenInterest`` for puts — IB stores them separately even on a single
    contract object). Greeks come from ``ticker.modelGreeks`` if present;
    each numeric is normalised through :func:`_safe_float` so missing values
    surface as ``None`` rather than ``NaN``.

    Parameters
    ----------
    ticker:
        A snapshot ticker returned from ``reqTickersAsync`` (or a live one
        from ``reqMktData``).

    Returns
    -------
    OptionQuote
        Plain dataclass safe to serialise / hand to a DataFrame.
    """
    greeks = getattr(ticker, "modelGreeks", None)
    return OptionQuote(
        contract=ticker.contract,
        bid=_safe_float(ticker.bid),
        ask=_safe_float(ticker.ask),
        last=_safe_float(ticker.last),
        mark=_safe_float(getattr(ticker, "markPrice", None)),
        volume=_safe_float(ticker.volume),
        open_interest=_safe_float(
            getattr(ticker, "callOpenInterest", None)
            if ticker.contract.right == "C"
            else getattr(ticker, "putOpenInterest", None)
        ),
        iv=_safe_float(getattr(greeks, "impliedVol", None)) if greeks else None,
        delta=_safe_float(getattr(greeks, "delta", None)) if greeks else None,
        gamma=_safe_float(getattr(greeks, "gamma", None)) if greeks else None,
        vega=_safe_float(getattr(greeks, "vega", None)) if greeks else None,
        theta=_safe_float(getattr(greeks, "theta", None)) if greeks else None,
        underlying_price=_safe_float(getattr(greeks, "undPrice", None)) if greeks else None,
    )


DATAFRAME_COLUMNS = (
    "symbol",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "last",
    "mark",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "vega",
    "theta",
    "underlying_price",
    "timestamp",
)


def quotes_to_dataframe(quotes: Sequence[OptionQuote]) -> "pd.DataFrame":
    """
    Convert a sequence of :class:`OptionQuote` into a pandas ``DataFrame``.

    Useful when you want to feed an option chain into pandas-based analytics
    (volatility surface fitting, screening, parquet export, etc.) without
    writing the boilerplate flattening yourself.

    Parameters
    ----------
    quotes:
        The list returned by :meth:`OptionChainFetcher.fetch_snapshot`.
        May be empty — an empty DataFrame with the standard columns is
        returned in that case (so downstream ``.empty`` /
        ``df["strike"]`` checks always work).

    Returns
    -------
    pandas.DataFrame
        Columns: ``symbol, expiry, strike, right, bid, ask, last, mark,
        volume, open_interest, iv, delta, gamma, vega, theta,
        underlying_price, timestamp``. ``timestamp`` is the snapshot's
        ``time.time()`` value (float, seconds since epoch).
    """
    if not quotes:
        return pd.DataFrame(columns=list(DATAFRAME_COLUMNS))

    return pd.DataFrame(
        [
            {
                "symbol": q.contract.symbol,
                "expiry": q.contract.lastTradeDateOrContractMonth,
                "strike": q.contract.strike,
                "right": q.contract.right,
                "bid": q.bid,
                "ask": q.ask,
                "last": q.last,
                "mark": q.mark,
                "volume": q.volume,
                "open_interest": q.open_interest,
                "iv": q.iv,
                "delta": q.delta,
                "gamma": q.gamma,
                "vega": q.vega,
                "theta": q.theta,
                "underlying_price": q.underlying_price,
                "timestamp": q.timestamp,
            }
            for q in quotes
        ],
        columns=list(DATAFRAME_COLUMNS),
    )
