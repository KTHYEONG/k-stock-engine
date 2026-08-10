"""Bounded lazy read plan: partition pruning and projection equivalence."""
from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetManifest,
    make_manifest,
)
from src.core.instruments import AssetKind
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

FEATURE_SET = "stock_alpha_v1"
DECISION = datetime(2024, 3, 15, 8, 50, tzinfo=UTC)


def frame() -> pl.DataFrame:
    sessions = [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
        datetime(2024, 2, 2, tzinfo=UTC),
        datetime(2024, 2, 5, tzinfo=UTC),
    ]
    n = len(sessions)
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:1"] * n,
            "session": sessions,
            "observation_time": sessions,
            "available_time": sessions,
            "close": [100.0 + i for i in range(n)],
            "volume": [1_000_000.0] * n,
            "market_cap": [1e12] * n,
            "feature__a": [1.0 + i for i in range(n)],
            "feature__b": [2.0 + i for i in range(n)],
            "feature__c": [3.0 + i for i in range(n)],
        }
    ).sort(["instrument_id", "session"])


def manifest(data: pl.DataFrame) -> DatasetManifest:
    return make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=data.columns,
        feature_set=FEATURE_SET,
        label_definition="none",
        label_horizon_sessions=1,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 1, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="v1",
        row_count=data.height,
        schema_version="v2",
        content_hash=canonical_content_hash(data, data.columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )


@pytest.fixture
def store(tmp_path) -> ParquetDatasetStore:
    store = ParquetDatasetStore(tmp_path / "root")
    data = frame()
    store.write_partitioned(
        data,
        dataset_id="features_v1",
        manifest=manifest(data),
        expected_feature_set=FEATURE_SET,
        decision_time=DECISION,
    )
    return store


class TestBoundedRead:
    def test_scans_only_intersecting_month_partition(self, store) -> None:
        paths = store.bounded_partition_paths(
            "features_v1",
            session_start=date(2024, 1, 15),
            session_end=date(2024, 1, 31),
        )
        relative = {str(p).split("partitions/", 1)[1] for p in paths}
        assert relative == {"year=2024/month=01/part-00000.parquet"}
        assert "month=02" not in "".join(relative)

    def test_output_equals_full_read_with_projection_and_filter(self, store) -> None:
        start, end = date(2024, 1, 1), date(2024, 1, 31)
        columns = ["instrument_id", "session", "close", "feature__a", "feature__b"]
        bounded = store.read_bounded(
            "features_v1",
            AssetKind.STOCK,
            FEATURE_SET,
            DECISION,
            session_start=start,
            session_end=end,
            columns=columns,
        )
        full = store.read("features_v1", AssetKind.STOCK, FEATURE_SET, DECISION)
        expected = full.filter(
            (pl.col("session") >= datetime(2024, 1, 1, tzinfo=UTC))
            & (pl.col("session") <= datetime(2024, 1, 31, tzinfo=UTC))
        ).select(columns).sort(["instrument_id", "session"])
        assert bounded.equals(expected)
        assert bounded["session"].to_list() == [
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 4, tzinfo=UTC),
        ]

    def test_column_subset_is_projected(self, store) -> None:
        bounded = store.read_bounded(
            "features_v1",
            AssetKind.STOCK,
            FEATURE_SET,
            DECISION,
            session_start=date(2024, 1, 1),
            session_end=date(2024, 2, 28),
            columns=["instrument_id", "close", "feature__c"],
        )
        assert set(bounded.columns) == {"instrument_id", "close", "feature__c"}

    def test_requested_column_absent_from_dataset_is_rejected(self, store) -> None:
        with pytest.raises(ValueError, match="absent from dataset"):
            store.read_bounded(
                "features_v1",
                AssetKind.STOCK,
                FEATURE_SET,
                DECISION,
                session_start=date(2024, 1, 1),
                session_end=date(2024, 1, 31),
                columns=["instrument_id", "session", "no_such_column"],
            )

    def test_tampered_selected_partition_fails_closed(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path / "fresh")
        data = frame()
        store.write_partitioned(
            data,
            dataset_id="fresh_features_v1",
            manifest=manifest(data),
            expected_feature_set=FEATURE_SET,
            decision_time=DECISION,
        )
        jan_part = (
            tmp_path / "fresh" / "fresh_features_v1"
            / "partitions" / "year=2024" / "month=01" / "part-00000.parquet"
        )
        with jan_part.open("ab") as fh:
            fh.write(b"tamper")
        with pytest.raises(ValueError, match="tampered partition"):
            store.read_bounded(
                "fresh_features_v1",
                AssetKind.STOCK,
                FEATURE_SET,
                DECISION,
                session_start=date(2024, 1, 1),
                session_end=date(2024, 1, 31),
                columns=["instrument_id", "close"],
            )
