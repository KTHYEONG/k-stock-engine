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

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime
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


def canonical_content_hash(frame: pl.DataFrame, ordered_columns: list[str]) -> str:
    """Deterministic fingerprint of the canonical rows of ``frame``.

    The digest binds the ordered column list and the sorted-by-identity rows,
    so it is invariant to source row order and partition layout while still
    detecting any change to content.
    """
    key_columns = [c for c in ("instrument_id", "session") if c in ordered_columns]
    sort_columns = key_columns or ordered_columns[:1]
    rows = frame.select(ordered_columns).sort(sort_columns).hash_rows(seed=0).to_numpy().tobytes()
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
    return value.isoformat() if isinstance(value, datetime) else None


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
        partitioned = frame.with_columns(
            pl.col("session").dt.strftime("%Y").alias(year_col),
            pl.col("session").dt.strftime("%m").alias(month_col),
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
                    "session_start": _iso_dt(sub["session"].min()),
                    "session_end": _iso_dt(sub["session"].max()),
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
        output = content.get("output") or {}
        if output.get("schema_hash") != manifest.schema_hash:
            raise ValueError("content manifest schema hash does not match dataset manifest")

        for entry in content.get("partitions") or []:
            part_path = dataset_dir / str(entry.get("path", ""))
            if not part_path.exists():
                raise FileNotFoundError(f"missing partition {entry.get('path')!r}")
            if file_sha256(part_path) != entry.get("sha256"):
                raise ValueError(f"tampered partition {entry.get('path')!r}")

        columns = [str(c) for c in output.get("column_order", [])]
        frame = pl.read_parquet(partitions_dir, hive_partitioning=True)
        recomputed = canonical_content_hash(frame, columns)
        if recomputed != manifest.content_hash:
            raise ValueError("dataset content hash mismatch")
        if recomputed != output.get("content_hash"):
            raise ValueError("content manifest content hash mismatch")
        key_columns = [c for c in ("instrument_id", "session") if c in columns]
        sort_columns = key_columns or columns[:1]
        return frame.select(columns).sort(sort_columns)


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
