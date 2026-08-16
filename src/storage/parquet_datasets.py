"""Asset-neutral, manifest-validated Parquet dataset adapter.

Replaces the former stock ``CuratedStore`` and ETF ``EtfCuratedStore`` with a
single adapter parameterized by ``AssetKind``. It loads no asset policy and
validates metadata before any Parquet read or write. Asset-specific callers
validate ``AssetKind`` at their boundary.

A dataset directory holds ``dataset_manifest.json`` plus either a legacy
single-file ``<dataset_id>.parquet`` (v1) or a manifest-declared Hive partition
layout ``partitions/year=YYYY/month=MM/part-*.parquet`` with a
``content_manifest.json`` (v2). Reads verify the declared content hash and
per-partition digests before materializing any frame, so tampered or incomplete
datasets fail closed.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    DatasetManifest,
    schema_hash,
    validate_dataset_manifest,
)
from src.core.instruments import AssetKind

MANIFEST_NAME = "dataset_manifest.json"
CONTENT_MANIFEST_NAME = "content_manifest.json"
_PARTITION_DIRNAME = "partitions"
_PARTITION_COLUMNS = ("year", "month")

# Per-process verification cache: (absolute dataset dir, content_hash) ->
# verified partition paths. A partition digest is checked once per process for
# a given physical dataset; the cache is never persisted across processes (spec
# serving rule: keyed only for a single process).
_VERIFIED_PARTITIONS: dict[tuple[str, str], frozenset[str]] = {}


def canonical_content_hash(frame: pl.DataFrame, ordered_columns: list[str]) -> str:
    """Deterministic fingerprint of the canonical rows of ``frame``.

    The digest binds the ordered column list and the sorted-by-identity rows,
    so it is invariant to source row order and partition layout while still
    detecting any change to content.
    """
    rows = (
        frame.select(ordered_columns)
        .sort(ordered_columns)
        .hash_rows(seed=0)
        .to_numpy()
        .tobytes()
    )
    return hashlib.sha256(
        schema_hash(ordered_columns).encode("utf-8") + b"\x00" + rows
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_dt(value: object) -> str | None:
    return value.isoformat() if isinstance(value, (datetime, date)) else None


class ParquetDatasetStore:
    """Manifest-validated, ``AssetKind``-parameterized Parquet IO.

    Reads fail closed on any manifest mismatch before materializing the
    Parquet table. A v2 manifest declares a Hive partition layout whose
    ``content_manifest.json`` must verify before any partition is scanned.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def write(
        self,
        frame: pl.DataFrame,
        *,
        dataset_id: str,
        manifest: DatasetManifest,
        expected_feature_set: str,
        decision_time: datetime,
    ) -> Path:
        """Persist a curated frame together with its validated manifest.

        Legacy single-file layout used by v1 consumers; new datasets should
        prefer :meth:`write_partitioned`.
        """
        if frame.is_empty():
            raise ValueError("cannot write an empty dataset")
        validate_dataset_manifest(
            manifest, manifest.asset_kind, expected_feature_set, decision_time
        )
        dataset_dir = self.root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with (dataset_dir / MANIFEST_NAME).open("w", encoding="utf-8") as fh:
            json.dump(_manifest_to_dict(manifest), fh, indent=2, default=str)
        table_path = dataset_dir / f"{dataset_id}.parquet"
        frame.write_parquet(table_path)
        return table_path

    def write_partitioned(
        self,
        frame: pl.DataFrame,
        *,
        dataset_id: str,
        manifest: DatasetManifest,
        expected_feature_set: str,
        decision_time: datetime,
        content_manifest: dict[str, object] | None = None,
    ) -> Path:
        """Atomically publish a canonical Hive-partitioned v2 dataset.

        Writes ``partitions/year=YYYY/month=MM/part-00000.parquet`` plus both
        manifests into a staging directory and publishes them under the final
        name only after everything is fully written. An existing ``dataset_id``
        is rejected, never overwritten.
        """
        if frame.is_empty():
            raise ValueError("cannot write an empty dataset")
        validate_dataset_manifest(
            manifest, manifest.asset_kind, expected_feature_set, decision_time
        )
        if manifest.schema_version != "v2":
            raise ValueError("partitioned write requires schema_version v2")
        if not manifest.content_hash:
            raise ValueError("partitioned write requires a manifest content_hash")
        if manifest.storage_layout != HIVE_PARTITION_LAYOUT:
            raise ValueError(
                f"partitioned write requires storage_layout {HIVE_PARTITION_LAYOUT!r}"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        dataset_dir = self.root / dataset_id
        if dataset_dir.exists():
            raise ValueError(f"dataset already exists: {dataset_id}")
        staging = self.root / f".{dataset_id}.{uuid.uuid4().hex}.staging"
        partitions_dir = staging / _PARTITION_DIRNAME
        partitions_dir.mkdir(parents=True)

        year_col, month_col = _PARTITION_COLUMNS
        session_column = (
            "session" if "session" in frame.columns else "decision_session"
            if "decision_session" in frame.columns else "price_date"
        )
        if session_column not in frame.columns:
            raise ValueError(
                "partitioned write requires a session, decision_session, or price_date column"
            )
        partitioned = frame.with_columns(
            pl.col(session_column).dt.strftime("%Y").alias(year_col),
            pl.col(session_column).dt.strftime("%m").alias(month_col),
        )
        entries: list[dict[str, object]] = []
        for sub in partitioned.sort([year_col, month_col]).partition_by(
            [year_col, month_col]
        ):
            year = str(sub[year_col][0])
            month = str(sub[month_col][0])
            part_dir = partitions_dir / f"{year_col}={year}" / f"{month_col}={month}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_path = part_dir / "part-00000.parquet"
            sub.drop([year_col, month_col]).write_parquet(part_path)
            entries.append(
                {
                    "path": str(part_path.relative_to(staging)),
                    "row_count": sub.height,
                    "session_start": _iso_dt(sub[session_column].min()),
                    "session_end": _iso_dt(sub[session_column].max()),
                    "sha256": file_sha256(part_path),
                }
            )

        content_hash = canonical_content_hash(frame, frame.columns)
        if content_hash != manifest.content_hash:
            raise ValueError("manifest content_hash does not match the curated frame")
        output = {
            "row_count": frame.height,
            "column_order": frame.columns,
            "schema_hash": schema_hash(frame.columns),
            "content_hash": content_hash,
        }
        manifest_content = {
            "content_manifest_version": 1,
            **dict(content_manifest or {}),
            "output": output,
            "partitions": entries,
        }
        with (staging / CONTENT_MANIFEST_NAME).open("w", encoding="utf-8") as fh:
            json.dump(manifest_content, fh, indent=2, default=str)
        with (staging / MANIFEST_NAME).open("w", encoding="utf-8") as fh:
            json.dump(_manifest_to_dict(manifest), fh, indent=2, default=str)

        dataset_dir.mkdir()
        try:
            for item in sorted(staging.iterdir()):
                os.replace(item, dataset_dir / item.name)
        except Exception:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return dataset_dir

    def read_manifest(self, dataset_id: str) -> DatasetManifest:
        """Return the manifest of ``dataset_id`` without materializing Parquet."""
        dataset_dir = self.root / dataset_id
        manifest_path = dataset_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest for dataset {dataset_id!r}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            return _manifest_from_dict(json.load(fh))

    def read(
        self,
        dataset_id: str,
        expected_asset_kind: AssetKind,
        expected_feature_set: str,
        decision_time: datetime,
    ) -> pl.DataFrame:
        """Read and return the validated dataset table for ``dataset_id``.

        Supports both the legacy single-file layout and the v2 Hive partition
        layout. For a partitioned dataset the declared content manifest and
        per-partition digests are verified before any Parquet is materialized.

        Raises:
            FileNotFoundError: if the manifest or Parquet tables are absent.
            ValueError: if the manifest asset kind, feature set, schema
                fingerprint, point-in-time availability, declared content hash,
                or any partition digest does not match.
        """
        dataset_dir = self.root / dataset_id
        manifest_path = dataset_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest for dataset {dataset_id!r}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = _manifest_from_dict(json.load(fh))
        validate_dataset_manifest(
            manifest, expected_asset_kind, expected_feature_set, decision_time
        )
        partitions_dir = dataset_dir / _PARTITION_DIRNAME
        if partitions_dir.exists():
            return self._read_partitioned(dataset_dir, partitions_dir, manifest)
        table_path = dataset_dir / f"{dataset_id}.parquet"
        if not table_path.exists():
            raise FileNotFoundError(f"no parquet for dataset {dataset_id!r}")
        return pl.read_parquet(table_path)

    def _read_partitioned(
        self,
        dataset_dir: Path,
        partitions_dir: Path,
        manifest: DatasetManifest,
    ) -> pl.DataFrame:
        content_path = dataset_dir / CONTENT_MANIFEST_NAME
        if not content_path.exists():
            raise ValueError(
                f"partitioned dataset {dataset_dir.name} has no content manifest"
            )
        with content_path.open("r", encoding="utf-8") as fh:
            content = json.load(fh)
        output = _content_output(content)
        if output.get("schema_hash") != manifest.schema_hash:
            raise ValueError("content manifest schema hash does not match dataset manifest")

        if output.get("content_hash") != manifest.content_hash:
            raise ValueError("content manifest content hash does not match dataset manifest")
        self._verify_entries(
            dataset_dir, _content_partitions(content), manifest.content_hash
        )

        columns = _content_column_order(content)
        frame = pl.read_parquet(partitions_dir, hive_partitioning=True)
        key_columns = [
            c
            for c in ("instrument_id", "session", "price_date", "horizon_sessions")
            if c in columns
        ]
        sort_columns = key_columns or columns[:1]
        return frame.select(columns).sort(sort_columns)

    def content_columns(self, dataset_id: str) -> list[str]:
        """Return the declared column order of a partitioned dataset."""
        _, _, content = self._load_verified_layout(dataset_id)
        return _content_column_order(content)

    def bounded_partition_paths(
        self,
        dataset_id: str,
        *,
        session_start: date,
        session_end: date,
    ) -> tuple[Path, ...]:
        """Return the exact partition files a bounded read would scan.

        The manifest and content manifest are validated and the per-partition
        digests of the selected partitions are verified before the paths are
        returned, so the caller can assert that only the intersecting
        ``year=YYYY/month=MM`` partitions are touched.
        """
        dataset_dir, manifest, content = self._load_verified_layout(dataset_id)
        entries = _select_entries(content, session_start, session_end)
        paths = tuple(dataset_dir / str(entry["path"]) for entry in entries)
        self._verify_entries(dataset_dir, entries, manifest.content_hash)
        return paths

    def read_bounded(
        self,
        dataset_id: str,
        expected_asset_kind: AssetKind,
        expected_feature_set: str,
        decision_time: datetime,
        *,
        session_start: date,
        session_end: date,
        columns: list[str],
    ) -> pl.DataFrame:
        """Lazily read only the partitions/columns that intersect the request.

        Only the monthly partitions whose declared session range intersects
        ``[session_start, session_end]`` are scanned, and only the requested
        ``columns`` are projected; the session range is filtered before
        collect. Per-partition digests of the selected partitions are verified
        (cached per process). The result equals a full read followed by the
        same projection/filter.
        """
        dataset_dir, manifest, content = self._load_verified_layout(dataset_id)
        validate_dataset_manifest(
            manifest, expected_asset_kind, expected_feature_set, decision_time
        )
        column_order = _content_column_order(content)
        missing = [c for c in columns if c not in column_order]
        if missing:
            raise ValueError(
                f"bounded read requests columns absent from dataset {dataset_id}: {missing}"
            )
        entries = _select_entries(content, session_start, session_end)
        if not entries:
            return pl.DataFrame({column: [] for column in columns})
        self._verify_entries(dataset_dir, entries, manifest.content_hash)

        paths = [dataset_dir / str(entry["path"]) for entry in entries]
        scan = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("session") >= datetime.combine(session_start, datetime.min.time(), UTC))
                & (pl.col("session") <= datetime.combine(session_end, datetime.min.time(), UTC))
            )
            .select(columns)
        )
        frame = scan.collect()
        key_columns = [c for c in ("instrument_id", "session") if c in columns]
        sort_columns = key_columns or columns[:1]
        return frame.sort(sort_columns)

    def _load_verified_layout(
        self, dataset_id: str
    ) -> tuple[Path, DatasetManifest, dict[str, object]]:
        dataset_dir = self.root / dataset_id
        manifest_path = dataset_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest for dataset {dataset_id!r}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = _manifest_from_dict(json.load(fh))
        partitions_dir = dataset_dir / _PARTITION_DIRNAME
        if not partitions_dir.exists():
            raise ValueError(
                f"bounded read requires a partitioned dataset, got {dataset_id!r}"
            )
        content_path = dataset_dir / CONTENT_MANIFEST_NAME
        if not content_path.exists():
            raise ValueError(
                f"partitioned dataset {dataset_id!r} has no content manifest"
            )
        with content_path.open("r", encoding="utf-8") as fh:
            content = json.load(fh)
        output = _content_output(content)
        if output.get("schema_hash") != manifest.schema_hash:
            raise ValueError("content manifest schema hash does not match dataset manifest")
        return dataset_dir, manifest, content

    def _verify_entries(
        self,
        dataset_dir: Path,
        entries: list[dict[str, object]],
        content_hash: str,
    ) -> None:
        cache_key = (str(dataset_dir.resolve()), content_hash)
        verified = _VERIFIED_PARTITIONS.get(cache_key, frozenset())
        for entry in entries:
            path = str(entry.get("path", ""))
            if path in verified:
                continue
            part_path = dataset_dir / path
            if not part_path.exists():
                raise FileNotFoundError(f"missing partition {path!r}")
            if file_sha256(part_path) != entry.get("sha256"):
                raise ValueError(f"tampered partition {path!r}")
        _VERIFIED_PARTITIONS[cache_key] = verified.union(str(e["path"]) for e in entries)


def _content_output(content: dict[str, object]) -> dict[str, object]:
    output = content.get("output")
    if not isinstance(output, dict):
        raise ValueError("content manifest has no output object")
    return output


def _content_column_order(content: dict[str, object]) -> list[str]:
    raw_order = _content_output(content).get("column_order")
    if not isinstance(raw_order, list):
        raise ValueError("content manifest column_order must be a list")
    return [str(column) for column in raw_order]


def _content_partitions(content: dict[str, object]) -> list[dict[str, object]]:
    raw = content.get("partitions")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("content manifest partitions must be a list")
    return [entry for entry in raw if isinstance(entry, dict)]


def _select_entries(
    content: dict[str, object],
    session_start: date,
    session_end: date,
) -> list[dict[str, object]]:
    """Select content-manifest partitions whose month intersects ``[start, end]``.

    Selection is by the partition's declared ``year=YYYY/month=MM`` bucket, so a
    reader requesting January 2024 scans ``year=2024/month=01`` even when the
    observed sessions within it are narrower; row-level session filtering still
    happens inside the lazy scan before collect.
    """
    selected: list[dict[str, object]] = []
    for entry in _content_partitions(content):
        year_month = _partition_year_month(str(entry.get("path", "")))
        if year_month is None:
            continue
        year, month = year_month
        part_start = date(year, month, 1)
        part_end = date(year, month, calendar.monthrange(year, month)[1])
        if part_start <= session_end and part_end >= session_start:
            selected.append(entry)
    return selected


def _partition_year_month(path: str) -> tuple[int, int] | None:
    """Parse ``year=YYYY/month=MM`` segments out of a partition path."""
    year = month = None
    for segment in path.split("/"):
        if segment.startswith("year="):
            year = int(segment[len("year=") :])
        elif segment.startswith("month="):
            month = int(segment[len("month=") :])
    if year is None or month is None:
        return None
    return year, month


def _manifest_to_dict(manifest: DatasetManifest) -> dict[str, object]:
    data = asdict(manifest)
    data["asset_kind"] = manifest.asset_kind.value
    return data


def _manifest_from_dict(data: dict[str, object]) -> DatasetManifest:
    certification = data.get("certification")
    if isinstance(certification, str):
        certification = DatasetCertification(certification)
    elif certification is None or not isinstance(certification, DatasetCertification):
        certification = DatasetCertification.PROVISIONAL
    return DatasetManifest(
        asset_kind=AssetKind(str(data["asset_kind"])),
        schema_version=str(data["schema_version"]),
        schema_hash=str(data["schema_hash"]),
        provider_version=str(data["provider_version"]),
        universe_policy_version=str(data["universe_policy_version"]),
        universe_policy_hash=str(data["universe_policy_hash"]),
        feature_set=str(data["feature_set"]),
        feature_set_hash=str(data["feature_set_hash"]),
        label_definition=str(data["label_definition"]),
        label_horizon_sessions=int(str(data["label_horizon_sessions"])),
        time_start=datetime.fromisoformat(str(data["time_start"])),
        time_end=datetime.fromisoformat(str(data["time_end"])),
        generated_time=datetime.fromisoformat(str(data["generated_time"])),
        row_count=int(str(data["row_count"])),
        certification=certification,
        calendar_hash=str(data.get("calendar_hash", "")),
        corporate_action_hash=str(data.get("corporate_action_hash", "")),
        cost_source_hash=str(data.get("cost_source_hash", "")),
        master_hash=str(data.get("master_hash", "")),
        quality_report_hash=str(data.get("quality_report_hash", "")),
        content_hash=str(data.get("content_hash", "")),
        storage_layout=str(data.get("storage_layout", "")),
    )
