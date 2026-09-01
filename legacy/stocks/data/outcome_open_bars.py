"""Compact, immutable open-price projection for outcome resolution."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
from src.core.instruments import AssetKind
from legacy.stocks.data.catalog import CatalogEntry, CatalogKind, CatalogStore, EvidenceCompleteness
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

OUTCOME_OPEN_BAR_FEATURE_SET = "stock_outcome_open_v1"
OUTCOME_OPEN_BAR_COLUMNS = ("instrument_id", "price_date", "open")


def build_outcome_open_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Return the exact valid open-price projection of raw KRX bar records."""
    required = set(OUTCOME_OPEN_BAR_COLUMNS)
    if not records or any(not isinstance(row, dict) or not required.issubset(row) for row in records):
        raise ValueError("raw bar records must carry instrument_id, price_date, and open")
    frame = pl.DataFrame(records).select(
        pl.col("instrument_id").cast(pl.Utf8), pl.col("price_date").cast(pl.Utf8).str.to_date(), pl.col("open").cast(pl.Float64)
    )
    return validate_outcome_open_frame(frame)


def validate_outcome_open_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate an already projected frame without materializing Python rows."""
    if frame.columns != list(OUTCOME_OPEN_BAR_COLUMNS):
        raise ValueError("outcome-open dataset has an invalid column contract")
    invalid = frame.filter(
        (pl.col("instrument_id").str.len_chars() == 0) | pl.col("price_date").is_null()
        | pl.col("open").is_null() | ~pl.col("open").is_finite() | (pl.col("open") <= 0)
    )
    if not invalid.is_empty():
        raise ValueError("outcome-open projection requires finite, strictly positive opens")
    if not frame.group_by(["instrument_id", "price_date"]).len().filter(pl.col("len") > 1).is_empty():
        raise ValueError("outcome-open projection contains duplicate instrument/date keys")
    return frame.sort(["instrument_id", "price_date"])


def publish_outcome_open_bar_dataset(
    raw_bar_entry: CatalogEntry, destination_root: Path, catalog_root: Path,
    dataset_id: str, generated_time: datetime,
) -> CatalogEntry:
    """Project one complete immutable RAW_BARS entry into verified Parquet."""
    if raw_bar_entry.kind is not CatalogKind.RAW_BARS or raw_bar_entry.completeness is not EvidenceCompleteness.COMPLETE:
        raise ValueError("outcome-open projection requires complete RAW_BARS evidence")
    if not dataset_id:
        raise ValueError("outcome-open dataset id must be non-empty")
    content = Path(raw_bar_entry.path).read_bytes()
    if hashlib.sha256(content).hexdigest() != raw_bar_entry.content_hash:
        raise ValueError("raw bar dataset hash mismatch")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("raw bar dataset is invalid JSON") from exc
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or payload.get("record_count") != len(records):
        raise ValueError("raw bar dataset record_count mismatch")
    if raw_bar_entry.coverage is None:
        raise ValueError("raw bar dataset requires coverage metadata")
    frame = build_outcome_open_frame(records)
    coverage = raw_bar_entry.coverage
    content_hash = canonical_content_hash(frame, list(OUTCOME_OPEN_BAR_COLUMNS))
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK, columns=list(OUTCOME_OPEN_BAR_COLUMNS),
        feature_set=OUTCOME_OPEN_BAR_FEATURE_SET, label_definition="verified_daily_open",
        label_horizon_sessions=1,
        time_start=datetime.combine(coverage.start, datetime.min.time(), UTC),
        time_end=datetime.combine(coverage.end, datetime.min.time(), UTC),
        provider_version="krx-daily-bars-projection-v1", universe_policy_version="raw-bars-exact-key-v1",
        row_count=frame.height, generated_time=generated_time, content_hash=content_hash,
        schema_version="v2", storage_layout=HIVE_PARTITION_LAYOUT,
    )
    dataset_path = ParquetDatasetStore(destination_root).write_partitioned(
        frame, dataset_id=dataset_id, manifest=manifest,
        expected_feature_set=OUTCOME_OPEN_BAR_FEATURE_SET, decision_time=generated_time,
        content_manifest={"raw_bar_content_hash": raw_bar_entry.content_hash},
    )
    entry = CatalogEntry(
        kind=CatalogKind.OUTCOME_OPEN_BARS, name=dataset_id, content_hash=content_hash,
        schema_hash=manifest.schema_hash, registered_at=generated_time, coverage=coverage,
        completeness=EvidenceCompleteness.COMPLETE, path=str(dataset_path), row_count=frame.height,
        references=((CatalogKind.RAW_BARS.value, raw_bar_entry.name),),
    )
    CatalogStore(catalog_root).register(entry)
    return entry


def load_outcome_open_bar_evidence(
    catalog: CatalogStore, dataset_id: str | None, decision_time: datetime,
) -> tuple[CatalogEntry | None, pl.DataFrame | None]:
    """Load a catalog-bound projection, failing closed on schema or hash mismatch."""
    if dataset_id is None:
        return None, None
    entry = catalog.get(CatalogKind.OUTCOME_OPEN_BARS, dataset_id)
    if entry is None or entry.completeness is not EvidenceCompleteness.COMPLETE:
        raise ValueError(f"outcome-open dataset is not complete: {dataset_id!r}")
    path = Path(entry.path)
    store = ParquetDatasetStore(path.parent)
    manifest = store.read_manifest(path.name)
    if manifest.content_hash != entry.content_hash:
        raise ValueError(f"outcome-open dataset hash mismatch: {dataset_id!r}")
    frame = store.read(path.name, AssetKind.STOCK, OUTCOME_OPEN_BAR_FEATURE_SET, decision_time)
    return entry, validate_outcome_open_frame(frame)
