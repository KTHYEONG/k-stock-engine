"""ML snapshot integrity audit tests."""
from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.stocks.data.feature_contracts import (
    semantic_feature_contract_book,
)
from src.stocks.data.ml_integrity import validate_ml_snapshot
from src.stocks.data.quality import KRXSessionCalendar


def _calendar() -> KRXSessionCalendar:
    return KRXSessionCalendar(
        version="test",
        sessions=tuple(date(2024, 1, d) for d in range(1, 12)),
        generated_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _contracts() -> pl.DataFrame:
    return semantic_feature_contract_book(
        "stock_alpha_v3",
        (
            {
                "name": "momentum",
                "role": "ALPHA",
                "source_field": "momentum",
                "source_dataset_ids": ("base_panel",),
                "source_columns": ("momentum",),
                "formula_id": "stock_alpha_v3:momentum:v1",
                "lookback_sessions": 1,
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 0,
                "expected_frequency": "session",
            },
            {
                "name": "volatility",
                "role": "RISK",
                "source_field": "volatility",
                "source_dataset_ids": ("base_panel",),
                "source_columns": ("volatility",),
                "formula_id": "stock_alpha_v3:volatility:v1",
                "lookback_sessions": 5,
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 0,
                "expected_frequency": "session",
            },
        ),
    )


def _valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["A"] * 6 + ["B"] * 6,
            "session": [
                date(2024, 1, d) for d in range(1, 7)
            ] * 2,
            "observation_time": [
                datetime(2024, 1, d, 15, 0, tzinfo=UTC) for d in range(1, 7)
            ] * 2,
            "available_time": [
                datetime(2024, 1, d, 15, 30, tzinfo=UTC) for d in range(1, 7)
            ] * 2,
            "open": [100.0 + i for i in range(6)] * 2,
            "high": [101.0 + i for i in range(6)] * 2,
            "low": [99.0 + i for i in range(6)] * 2,
            "close": [100.5 + i for i in range(6)] * 2,
            "volume": [1_000_000.0] * 12,
            "momentum": [None, 0.1, 0.2, 0.3, 0.4, 0.5] * 2,
            "volatility": [None, None, 0.2, 0.3, 0.4, 0.5] * 2,
        }
    )


def test_validate_ml_snapshot_passes_on_valid_frame() -> None:
    decision_time = datetime(2024, 1, 10, tzinfo=UTC)
    audit = validate_ml_snapshot(
        _valid_frame(), _contracts(), decision_time, _calendar()
    )
    assert audit.passed is True
    assert audit.row_count == 12
    assert audit.to_json()["passed"] is True


def test_validate_ml_snapshot_rejects_duplicate_keys() -> None:
    frame = pl.concat([_valid_frame(), _valid_frame().tail(1)])
    audit = validate_ml_snapshot(
        frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
    )
    assert audit.passed is False
    names = {check.name: check for check in audit.checks}
    assert names["key_and_calendar"].passed is False


def test_validate_ml_snapshot_rejects_non_calendar_session() -> None:
    frame = _valid_frame().with_columns(pl.lit(date(2025, 1, 1)).alias("session"))
    audit = validate_ml_snapshot(
        frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
    )
    assert audit.passed is False


def test_validate_ml_snapshot_rejects_available_after_decision() -> None:
    frame = _valid_frame().with_columns(
        pl.lit(datetime(2025, 1, 1, tzinfo=UTC)).alias("available_time")
    )
    audit = validate_ml_snapshot(
        frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
    )
    assert audit.passed is False


def test_validate_ml_snapshot_rejects_ohlc_violations() -> None:
    frame = _valid_frame().with_columns((pl.col("low") - 1.0).alias("high"))
    audit = validate_ml_snapshot(
        frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
    )
    assert audit.passed is False


def test_validate_ml_snapshot_rejects_target_namespace_predictor() -> None:
    frame = _valid_frame().with_columns(pl.lit(1.0).alias("feature__label_return"))
    with pytest.raises(ValueError, match="predictor namespace"):
        validate_ml_snapshot(
            frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
        )


def test_validate_ml_snapshot_rejects_constant_feature() -> None:
    frame = _valid_frame().with_columns(pl.lit(0.5).alias("momentum"))
    audit = validate_ml_snapshot(
        frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
    )
    names = {check.name: check for check in audit.checks}
    assert names["contract_coverage"].passed is False


def test_validate_ml_snapshot_records_label_universe() -> None:
    frame = _valid_frame().with_columns(
        pl.Series([0.01, None, 0.02, 0.03, 0.04, 0.05] * 2).alias("residual_o2o_5d")
    )
    audit = validate_ml_snapshot(
        frame, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()
    )
    assert audit.label_universe["residual_o2o_5d"] == 10
