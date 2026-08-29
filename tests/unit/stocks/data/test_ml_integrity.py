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


def test_ML_INTEGRITY_01_missing_availability_nonfinite_duplicate_ohlc_stale() -> None:
    # ML-INTEGRITY-01: Missing availability, non-finite, duplicate keys, invalid OHLC, equal-value stale runs rejected; warm-up nulls explicit
    # non-finite
    frame_nf = _valid_frame().with_columns(pl.lit(float("inf")).alias("momentum"))
    audit = validate_ml_snapshot(frame_nf, _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar())
    assert audit.passed is False
    # stale equal-value run (stale_after_sessions >0)
    stale_contracts = semantic_feature_contract_book(
        "stale_v1",
        (
            {
                "name": "momentum",
                "role": "ALPHA",
                "source_field": "momentum",
                "source_dataset_ids": ("base_panel",),
                "source_columns": ("momentum",),
                "formula_id": "stale_v1:momentum:v1",
                "lookback_sessions": 0,
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 2,
                "expected_frequency": "session",
            },
        ),
    )
    # 3 consecutive equal values triggers stale
    frame_stale = pl.DataFrame(
        {
            "instrument_id": ["A"] * 4,
            "session": [date(2024, 1, d) for d in range(1, 5)],
            "available_time": [datetime(2024, 1, d, 15, 30, tzinfo=UTC) for d in range(1, 5)],
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.5] * 4,
            "volume": [1_000_000.0] * 4,
            "momentum": [0.1, 0.1, 0.1, 0.1],
        }
    )
    audit2 = validate_ml_snapshot(frame_stale, stale_contracts, datetime(2024, 1, 10, tzinfo=UTC), _calendar())
    assert audit2.passed is False
    names = {c.name: c for c in audit2.checks}
    assert names["warmup_and_stale"].passed is False
    # warm-up nulls remain explicit: valid frame already has warm-up nulls, passes
    assert validate_ml_snapshot(_valid_frame(), _contracts(), datetime(2024, 1, 10, tzinfo=UTC), _calendar()).passed is True


def test_ML_INTEGRITY_02_fundamental_requires_disclosure_lineage() -> None:
    # ML-INTEGRITY-02
    fund_contracts = semantic_feature_contract_book(
        "fund_v1",
        (
            {
                "name": "bp_ratio",
                "role": "CONTROL",
                "source_field": "bp_ratio",
                "source_dataset_ids": ("base_panel",),
                "source_columns": ("bp_ratio",),
                "formula_id": "fund_v1:bp_ratio:v1",
                "lookback_sessions": 0,
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 0,
                "expected_frequency": "session",
                "source_available_time_field": "available_time",
            },
        ),
    )
    frame = _valid_frame().with_columns(pl.lit(0.5).alias("bp_ratio"))
    audit = validate_ml_snapshot(frame, fund_contracts, datetime(2024, 1, 10, tzinfo=UTC), _calendar())
    assert audit.passed is False
    assert any(c.name == "pit_availability" and not c.passed for c in audit.checks)
    # generic market availability cannot certify fundamental
    # correct disclosure lineage passes
    fund_ok = semantic_feature_contract_book(
        "fund_v1_ok",
        (
            {
                "name": "bp_ratio",
                "role": "CONTROL",
                "source_field": "bp_ratio",
                "source_dataset_ids": ("base_panel",),
                "source_columns": ("bp_ratio",),
                "formula_id": "fund_v1:bp_ratio:v1",
                "lookback_sessions": 0,
                "adjustment_basis": "split_adjusted",
                "null_policy": "retain_null",
                "stale_after_sessions": 0,
                "expected_frequency": "session",
                "source_available_time_field": "disclosure_date",
            },
        ),
    )
    frame_ok = _valid_frame().with_columns(pl.lit(0.5).alias("bp_ratio"), pl.lit(datetime(2024, 1, 1, tzinfo=UTC)).alias("disclosure_date"), pl.lit(datetime(2024, 1, 1, tzinfo=UTC)).alias("available_time"))
    # add generic available_time still present but disclosure lineage is required
    audit_ok = validate_ml_snapshot(frame_ok, fund_ok, datetime(2024, 1, 10, tzinfo=UTC), _calendar())
    # pit_availability should pass when disclosure_date present and lineage is disclosure_date
    assert any(c.name == "pit_availability" and c.passed for c in audit_ok.checks)
