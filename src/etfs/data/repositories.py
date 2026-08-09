"""ETF dataset repository: translates domain requests into the generic store.

The repository enforces ``AssetKind.ETF`` at its boundary and reuses the
asset-neutral ``ParquetDatasetStore`` for persistence; it is not a duplicate
store implementation.
"""
from __future__ import annotations

from datetime import datetime

from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.etfs.data.contracts import EtfDataset
from src.storage.parquet_datasets import ParquetDatasetStore


class EtfDatasetRepository:
    """ETF-only facade over ``ParquetDatasetStore``."""

    def __init__(self, store: ParquetDatasetStore):
        self.store = store
        self.asset_kind = AssetKind.ETF

    def read(
        self,
        index_dataset_id: str,
        etf_dataset_id: str,
        feature_set: str,
        decision_time: datetime,
    ) -> EtfDataset:
        """Read validated ETF dataset snapshots (ETF kind enforced)."""
        index_manifest = self.store.read_manifest(index_dataset_id)
        self._assert_etf(index_manifest)
        etf_manifest = self.store.read_manifest(etf_dataset_id)
        self._assert_etf(etf_manifest)
        index_frame = self.store.read(
            index_dataset_id, AssetKind.ETF, feature_set, decision_time
        )
        etf_frame = self.store.read(
            etf_dataset_id, AssetKind.ETF, feature_set, decision_time
        )
        return EtfDataset(
            manifest=etf_manifest, index_frame=index_frame, etf_frame=etf_frame
        )

    def _assert_etf(self, manifest: DatasetManifest) -> None:
        if manifest.asset_kind is not AssetKind.ETF:
            raise ValueError(
                f"etf repository rejects {manifest.asset_kind.value} manifest"
            )
