"""Asset-neutral, manifest-validated Parquet dataset adapter.

Replaces the former stock ``CuratedStore`` and ETF ``EtfCuratedStore`` with a
single adapter parameterized by ``AssetKind``. It loads no asset policy and
validates metadata before any Parquet read or write. Asset-specific callers
validate ``AssetKind`` at their boundary.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import polars as pl

from src.core.datasets import DatasetManifest, validate_dataset_manifest
from src.core.instruments import AssetKind

MANIFEST_NAME = "dataset_manifest.json"


class ParquetDatasetStore:
    """Manifest-validated, ``AssetKind``-parameterized Parquet IO.

    A dataset directory holds ``dataset_manifest.json`` plus
    ``<dataset_id>.parquet``. Reads fail closed on any manifest mismatch before
    materializing the Parquet table.
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
        """Persist a curated frame together with its validated manifest."""
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

        Raises:
            FileNotFoundError: if the manifest or Parquet table is absent.
            ValueError: if the manifest asset kind, feature set, schema
                fingerprint, or point-in-time availability does not match the
                caller's expectations.
        """
        dataset_dir = self.root / dataset_id
        manifest_path = dataset_dir / MANIFEST_NAME
        table_path = dataset_dir / f"{dataset_id}.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest for dataset {dataset_id!r}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = _manifest_from_dict(json.load(fh))
        validate_dataset_manifest(
            manifest, expected_asset_kind, expected_feature_set, decision_time
        )
        if not table_path.exists():
            raise FileNotFoundError(f"no parquet for dataset {dataset_id!r}")
        return pl.read_parquet(table_path)


def _manifest_to_dict(manifest: DatasetManifest) -> dict[str, object]:
    data = asdict(manifest)
    data["asset_kind"] = manifest.asset_kind.value
    return data


def _manifest_from_dict(data: dict[str, object]) -> DatasetManifest:
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
    )
