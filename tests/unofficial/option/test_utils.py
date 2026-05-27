"""Tests for the pure helpers in ``ibtws.unofficial.option.utils``."""

from __future__ import annotations

import pytest
from ib_async import Option

from ibtws.unofficial.helpers import chunked, safe_pick_value
from ibtws.unofficial.option.utils import (
    _filter_expirations,
    _filter_strikes,
    _ticker_to_quote,
)

from .conftest import make_ticker


def test_chunked_splits_evenly_and_preserves_order():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty_returns_nothing():
    assert list(chunked([], 10)) == []


@pytest.mark.parametrize(
    "attr_value,expected",
    [
        (1.5, 1.5),
        (None, None),
        (float("nan"), None),
        ("not a number", None),
        ("3.14", 3.14),
    ],
)
def test_pick_price(attr_value, expected):
    obj = type("X", (), {"v": attr_value})()
    assert safe_pick_value(obj, "v") == expected


def test_pick_price_missing_attr_returns_none():
    obj = type("X", (), {})()
    assert safe_pick_value(obj, "missing") is None


def test_filter_expirations_explicit_wins_over_range():
    avail = ("20260116", "20260220", "20260320")
    out = _filter_expirations(avail, explicit=["20260220", "20260320"], expiry_from=None, expiry_to=None)
    assert out == ["20260220", "20260320"]


def test_filter_expirations_range():
    avail = ("20260116", "20260220", "20260320")
    out = _filter_expirations(avail, explicit=None, expiry_from="20260201", expiry_to="20260301")
    assert out == ["20260220"]


def test_filter_strikes_range_inclusive():
    avail = (140.0, 150.0, 160.0, 170.0, 180.0)
    assert _filter_strikes(avail, explicit=None, strike_from=150.0, strike_to=170.0) == [150.0, 160.0, 170.0]


def test_filter_strikes_explicit():
    avail = (140.0, 150.0, 160.0)
    assert _filter_strikes(avail, explicit=[140.0, 160.0], strike_from=None, strike_to=None) == [140.0, 160.0]


def test_ticker_to_quote_uses_call_oi_for_calls():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = make_ticker(contract, call_oi=222, put_oi=999)
    q = _ticker_to_quote(t)
    assert q.open_interest == 222
    assert q.iv == 0.25
    assert q.delta == 0.5


def test_ticker_to_quote_uses_put_oi_for_puts():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="P")
    t = make_ticker(contract, call_oi=222, put_oi=999)
    q = _ticker_to_quote(t)
    assert q.open_interest == 999


def test_ticker_to_quote_handles_missing_greeks():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = make_ticker(contract, iv=None)
    t.modelGreeks = None
    q = _ticker_to_quote(t)
    assert q.iv is None
    assert q.delta is None
    assert q.underlying_price is None


def test_ticker_to_quote_scrubs_nan():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = make_ticker(contract)
    t.bid = float("nan")
    t.ask = float("nan")
    q = _ticker_to_quote(t)
    assert q.bid is None and q.ask is None


def test_ticker_to_quote_propagates_underlying_price():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    q = _ticker_to_quote(make_ticker(contract), underlying_price=152.5)
    assert q.underlying_price == 152.5
