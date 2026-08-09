"""Point-in-time dataset validation tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.time import TemporalViolationError
from src.stocks.data.repositories import read_provisional_legacy_panel
from src.stocks.research.datasets import (
    ELIGIBLE_STATUS,
    QUALITY_STATUS_COLUMN,
    QUARANTINED_STATUS,
    research_eligible_frame,
    validate_stock_rows_available,
)
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


def test_ohlc_quality_quarantine_invalid_row(tmp_path) -> None:
    # SCENARIO_STOCK_OHLC_QUALITY_QUARANTINE_01
    root = tmp_path / "features" / "year=2022"
    root.mkdir(parents=True)
    pl.DataFrame(
        {
            "date": [datetime(2022, 2, 9).date(), datetime(2022, 2, 8).date()],
            "ticker": ["215090", "215090"],
            "open": [0.0, 1695.0],
            "high": [0.0, 1695.0],
            "low": [0.0, 1485.0],
            "close": [1505.0, 1505.0],
            "volume": [1911.0, 11508652.0],
            "trading_value": [2876055.0, 18096968425.0],
        }
    ).write_parquet(root / "2022-02-09_feat.parquet")

    snapshot = read_provisional_legacy_panel(
        tmp_path / "features", datetime(2022, 1, 1).date(), datetime(2022, 12, 31).date(), ()
    )
    assert snapshot.frame.filter(pl.col(QUALITY_STATUS_COLUMN) == QUARANTINED_STATUS).height == 1
    assert research_eligible_frame(snapshot.frame).height == 1


def test_ohlc_quality_quarantine_eligible_row(tmp_path) -> None:
    # SCENARIO_STOCK_OHLC_QUALITY_QUARANTINE_02
    root = tmp_path / "features" / "year=2022"
    root.mkdir(parents=True)
    pl.DataFrame(
        {
            "date": [datetime(2022, 2, 8).date()],
            "ticker": ["215090"],
            "open": [1695.0],
            "high": [1695.0],
            "low": [1485.0],
            "close": [1505.0],
            "volume": [11508652.0],
            "trading_value": [18096968425.0],
        }
    ).write_parquet(root / "2022-02-08_feat.parquet")

    snapshot = read_provisional_legacy_panel(
        tmp_path / "features", datetime(2022, 1, 1).date(), datetime(2022, 12, 31).date(), ()
    )
    assert snapshot.frame[QUALITY_STATUS_COLUMN].to_list() == [ELIGIBLE_STATUS]
    assert research_eligible_frame(snapshot.frame).height == 1
