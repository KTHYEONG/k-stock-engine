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
    "DatasetManifest",
    "make_manifest",
    "schema_hash",
    "validate_dataset_manifest",
    "validate_stock_rows_available",
]


def validate_stock_rows_available(df: pl.DataFrame, decision_time: datetime) -> None:
    """Fail closed if any stock row becomes available after ``decision_time``."""
    late = df.filter(pl.col("date") > decision_time)
    if not late.is_empty():
        raise TemporalViolationError(
            f"{late.height} rows available after decision_time {decision_time.isoformat()}"
        )
