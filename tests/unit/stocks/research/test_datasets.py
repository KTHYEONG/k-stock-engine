"""Point-in-time dataset validation tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.time import TemporalViolationError
from src.stocks.research.datasets import validate_stock_rows_available
from tests.fixtures.stocks.helpers import stock_instrument_df


def test_valid_panel_passes() -> None:
    df = stock_instrument_df(n_sessions=5, n_tickers=2)
    assert validate_stock_rows_available(df, df["available_time"].max()) is None


def test_row_observed_after_available_is_rejected() -> None:
    df = stock_instrument_df(n_sessions=5, n_tickers=1).with_columns(
        (pl.col("available_time") + pl.duration(hours=1)).alias("observation_time")
    )
    with pytest.raises(TemporalViolationError):
        validate_stock_rows_available(df, df["available_time"].max())


def test_missing_pit_columns_rejected() -> None:
    df = stock_instrument_df(n_sessions=5, n_tickers=1).drop("available_time")
    with pytest.raises(ValueError, match="available_time"):
        validate_stock_rows_available(df, datetime(2024, 1, 10, tzinfo=UTC))


def test_duplicate_instrument_session_rejected() -> None:
    df = stock_instrument_df(n_sessions=5, n_tickers=1)
    dup = pl.concat([df, df.filter(pl.col("session_index") == 2)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_stock_rows_available(dup, datetime(2024, 1, 20, tzinfo=UTC))
