"""Feature readiness gate against a feature dataset's exact content manifest.

A model may select features only after :func:`validate_selected_feature_readiness`
confirms each selected ``feature__<name>`` column exists in every declared
partition, carries at least one finite non-null value, and contains no
NaN/infinite value. Only partitions named in ``content_manifest.json`` are
read, each verified by SHA-256 and row count before scanning. Non-selected
fully-null feature columns are reported, never rejected, and nothing is
imputed, dropped, or rewritten.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.storage.parquet_datasets import CONTENT_MANIFEST_NAME, file_sha256

FEATURE_COLUMN_PREFIX = "feature__"


@dataclass(frozen=True, slots=True)
class FeatureReadiness:
    """Aggregate null/non-null/non-finite counts for one selected feature."""

    name: str
    null_count: int
    non_null_count: int
    non_finite_count: int

    @property
    def usable(self) -> bool:
        return self.non_null_count > 0 and self.non_finite_count == 0


class FeatureReadinessIndex(Mapping[str, FeatureReadiness]):
    """Immutable name-indexed readiness lookup; read-only by construction."""

    _entries: dict[str, FeatureReadiness]

    def __init__(self, entries: Iterable[FeatureReadiness]) -> None:
        object.__setattr__(self, "_entries", {entry.name: entry for entry in entries})

    def __getitem__(self, name: str) -> FeatureReadiness:
        return self._entries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class FeatureReadinessReport:
    """Immutable readiness verdict for one feature dataset and explicit selection."""

    dataset_dir: str
    total_rows: int
    selected: FeatureReadinessIndex
    fully_null_stored_columns_not_selected: tuple[str, ...]


def validate_selected_feature_readiness(
    dataset_dir: Path, selected_features: tuple[str, ...]
) -> FeatureReadinessReport:
    """Validate selected feature columns against the declared partitions.

    Reads only the partitions named in ``content_manifest.json``, verifies each
    partition's path/SHA-256/row count, aggregates counts with lazy Polars, and
    fails closed for a missing, fully-null, or non-finite selected feature.
    """
    dataset_dir = Path(dataset_dir)
    if not selected_features:
        raise ValueError("at least one selected feature is required")
    if not dataset_dir.is_dir():
        raise ValueError(f"feature dataset directory not found: {dataset_dir}")
    manifest_path = dataset_dir / CONTENT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"content manifest missing: {manifest_path}")

    partitions = _read_content_partitions(manifest_path)
    if not partitions:
        raise ValueError("content manifest declares no partitions")
    paths = [_verify_partition(dataset_dir, entry) for entry in partitions]
    schemas = [pl.scan_parquet(path).collect_schema() for path in paths]

    selected_set = frozenset(selected_features)
    for name in selected_features:
        column = f"{FEATURE_COLUMN_PREFIX}{name}"
        absent = [
            path
            for path, schema in zip(paths, schemas, strict=True)
            if column not in schema
        ]
        if absent:
            raise ValueError(
                f"selected feature {name!r} is missing from partitions: {', '.join(absent)}"
            )

    feature_columns = sorted(
        {
            column
            for schema in schemas
            for column in schema.names()
            if column.startswith(FEATURE_COLUMN_PREFIX)
        }
    )
    if not feature_columns:
        raise ValueError("feature dataset exposes no feature__* columns")

    lazy = pl.scan_parquet(paths)
    aggs: list[pl.Expr] = [pl.len().alias("total_rows")]
    for name in selected_features:
        column = f"{FEATURE_COLUMN_PREFIX}{name}"
        dtype = schemas[0][column]
        aggs.append(pl.col(column).is_null().sum().alias(f"{name}__null"))
        aggs.append(pl.col(column).is_not_null().sum().alias(f"{name}__non_null"))
        if dtype in (pl.Float32, pl.Float64):
            non_finite = pl.col(column).is_nan().sum() + pl.col(column).is_infinite().sum()
        else:
            non_finite = pl.lit(0)
        aggs.append(non_finite.alias(f"{name}__non_finite"))
    aggs.append(pl.col("^feature__.*$").is_null().sum())
    result = lazy.select(aggs).collect().row(0, named=True)

    total_rows = int(result["total_rows"])
    selected = tuple(
        FeatureReadiness(
            name=name,
            null_count=int(result[f"{name}__null"]),
            non_null_count=int(result[f"{name}__non_null"]),
            non_finite_count=int(result[f"{name}__non_finite"]),
        )
        for name in selected_features
    )
    fully_null_not_selected = tuple(
        column
        for column in feature_columns
        if int(result[column]) == total_rows
        and column.removeprefix(FEATURE_COLUMN_PREFIX) not in selected_set
    )

    problems = [
        (
            f"{name} (fully null)"
            if readiness.non_null_count == 0
            else f"{name} (contains non-finite values)"
        )
        for name, readiness in zip(selected_features, selected, strict=True)
        if readiness.non_null_count == 0 or readiness.non_finite_count > 0
    ]
    if problems:
        raise ValueError("selected feature readiness failed: " + ", ".join(problems))

    return FeatureReadinessReport(
        dataset_dir=str(dataset_dir),
        total_rows=total_rows,
        selected=FeatureReadinessIndex(selected),
        fully_null_stored_columns_not_selected=fully_null_not_selected,
    )


def _read_content_partitions(manifest_path: Path) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid content manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"content manifest must be an object: {manifest_path}")
    raw = payload.get("partitions")
    if not isinstance(raw, list):
        raise ValueError("content manifest partitions must be a list")
    return tuple(entry for entry in raw if isinstance(entry, dict))


def _verify_partition(dataset_dir: Path, entry: dict[str, object]) -> str:
    raw_path = str(entry.get("path", ""))
    if not raw_path:
        raise ValueError("content manifest partition entry has no path")
    part_path = dataset_dir / raw_path
    if not part_path.is_file():
        raise ValueError(f"declared partition missing: {raw_path}")
    if file_sha256(part_path) != entry.get("sha256"):
        raise ValueError(f"partition digest mismatch: {raw_path}")
    declared_rows = entry.get("row_count")
    if not isinstance(declared_rows, int) or isinstance(declared_rows, bool):
        raise ValueError(f"partition row count invalid: {raw_path}")
    height = int(pl.scan_parquet(part_path).select(pl.len()).collect().item())
    if height != declared_rows:
        raise ValueError(f"partition row count mismatch: {raw_path}")
    return str(part_path)
