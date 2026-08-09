"""Stock ML point-in-time dataset contracts.

The ``DatasetManifest`` type and its generic validation contract are low-level
shared primitives that live in ``core.datasets``. This module owns the
stock-specific dataset validation on top of the shared manifest contract.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from src.core.datasets import DatasetManifest, make_manifest, schema_hash, validate_dataset_manifest
from src.core.time import TemporalViolationError

__all__ = [
    "ELIGIBLE_STATUS",
    "QUALITY_REASON_COLUMN",
    "QUALITY_STATUS_COLUMN",
    "QUARANTINED_STATUS",
    "DatasetManifest",
    "make_manifest",
    "research_eligible_frame",
    "schema_hash",
    "validate_dataset_manifest",
    "validate_stock_rows_available",
]

QUALITY_STATUS_COLUMN = "data_quality_status"
QUALITY_REASON_COLUMN = "data_quality_reason"
ELIGIBLE_STATUS = "eligible"
QUARANTINED_STATUS = "quarantined"


def validate_stock_rows_available(df: pl.DataFrame, decision_time: datetime) -> None:
    """Fail closed on any point-in-time violation.

    Requires explicit ``observation_time`` and ``available_time`` columns rather
    than treating the session date as availability. Raises
    ``TemporalViolationError`` when an observation is newer than its
    availability or when a row becomes available after ``decision_time``, and
    ``ValueError`` for duplicate or non-monotonic instrument sessions.
    """
    required = ("instrument_id", "session", "observation_time", "available_time")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"panel must carry {', '.join(missing)}")

    duplicates = (
        df.group_by(["instrument_id", "session"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ValueError("duplicate (instrument_id, session) rows")

    non_monotonic = (
        df.with_columns(
            pl.col("observation_time")
            .shift(1)
            .over("instrument_id")
            .alias("_prev_obs")
        )
        .filter(
            pl.col("_prev_obs").is_not_null()
            & (pl.col("observation_time") < pl.col("_prev_obs"))
        )
    )
    if not non_monotonic.is_empty():
        raise ValueError("instrument observation times must be non-decreasing")

    late = df.filter(pl.col("observation_time") > pl.col("available_time"))
    if not late.is_empty():
        raise TemporalViolationError(
            f"{late.height} rows observe after available_time"
        )
    late_decision = df.filter(pl.col("available_time") > decision_time)
    if not late_decision.is_empty():
        raise TemporalViolationError(
            f"{late_decision.height} rows available after decision_time {decision_time.isoformat()}"
        )


def research_eligible_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return only rows that passed the persisted deterministic quality gate."""
    if QUALITY_STATUS_COLUMN not in df.columns:
        return df
    eligible = df.filter(pl.col(QUALITY_STATUS_COLUMN) == ELIGIBLE_STATUS)
    if eligible.is_empty():
        raise ValueError("no research-eligible stock rows")
    return eligible
