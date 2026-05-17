"""Tests for ``ibtws.unofficial.option.dataframe``."""

from __future__ import annotations

from ib_async import Option

from ibtws.unofficial.option import DATAFRAME_COLUMNS, quotes_to_dataframe
from ibtws.unofficial.option.utils import _ticker_to_quote

from .conftest import make_ticker


def test_quotes_to_dataframe_columns_and_rows():
    contract_c = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    contract_p = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="P")
    quotes = [
        _ticker_to_quote(make_ticker(contract_c, bid=1.0, ask=1.2, iv=0.25)),
        _ticker_to_quote(make_ticker(contract_p, bid=0.8, ask=1.0, iv=0.30)),
    ]

    df = quotes_to_dataframe(quotes)

    assert list(df.columns) == list(DATAFRAME_COLUMNS)
    assert len(df) == 2
    assert set(df["right"]) == {"C", "P"}
    assert df.loc[df["right"] == "C", "iv"].iloc[0] == 0.25


def test_quotes_to_dataframe_empty_returns_typed_empty():
    df = quotes_to_dataframe([])
    assert df.empty
    assert list(df.columns) == list(DATAFRAME_COLUMNS)


def test_quotes_to_dataframe_preserves_none_for_missing_fields():
    contract = Option(symbol="AAPL", lastTradeDateOrContractMonth="20260116", strike=150, right="C")
    t = make_ticker(contract)
    t.modelGreeks = None  # simulate Greeks never arriving
    df = quotes_to_dataframe([_ticker_to_quote(t)])
    assert df["iv"].iloc[0] is None
    assert df["delta"].iloc[0] is None
