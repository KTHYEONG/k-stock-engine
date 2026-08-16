"""Asset-neutral Parquet dataset store: manifest validation and cross-kind rejection."""
from __future__ import annotations

from datetime import UTC, datetime

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
TIME_START = datetime(2024, 1, 1, tzinfo=UTC)
TIME_END = datetime(2024, 3, 1, tzinfo=UTC)
DECISION = datetime(2024, 3, 15, 8, 50, tzinfo=UTC)

COLUMNS = ["session_index", "date", "instrument_id", "close"]


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_index": [0, 1],
            "date": [TIME_START, TIME_START],
            "instrument_id": ["KRX:1", "KRX:2"],
            "close": [100.0, 200.0],
        }
    )


def stock_manifest() -> DatasetManifest:
    return make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=COLUMNS,
        feature_set=FEATURE_SET,
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        time_start=TIME_START,
        time_end=TIME_END,
        provider_version="fixture",
        universe_policy_version="v1",
        row_count=2,
    )


class TestParquetDatasetStore:
    def test_write_then_read_round_trips_same_frame(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        store.write(frame(), dataset_id="d1", manifest=stock_manifest(), expected_feature_set=FEATURE_SET, decision_time=DECISION)
        out = store.read("d1", AssetKind.STOCK, FEATURE_SET, DECISION)
        assert out.equals(frame())

    def test_cross_kind_read_is_rejected(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        store.write(frame(), dataset_id="d1", manifest=stock_manifest(), expected_feature_set=FEATURE_SET, decision_time=DECISION)
        with pytest.raises(ValueError, match="asset_kind"):
            store.read("d1", AssetKind.ETF, FEATURE_SET, DECISION)

    def test_feature_set_mismatch_is_rejected(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        store.write(frame(), dataset_id="d1", manifest=stock_manifest(), expected_feature_set=FEATURE_SET, decision_time=DECISION)
        with pytest.raises(ValueError, match="feature_set"):
            store.read("d1", AssetKind.STOCK, "etf_switch_v1", DECISION)

    def test_dataset_unavailable_at_decision_time_is_rejected(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        store.write(frame(), dataset_id="d1", manifest=stock_manifest(), expected_feature_set=FEATURE_SET, decision_time=DECISION)
        early = datetime(2024, 2, 1, 8, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="not available"):
            store.read("d1", AssetKind.STOCK, FEATURE_SET, early)

    def test_missing_manifest_is_file_not_found(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="no manifest"):
            store.read("ghost", AssetKind.STOCK, FEATURE_SET, DECISION)

    def test_missing_table_is_file_not_found(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        store.write(frame(), dataset_id="d1", manifest=stock_manifest(), expected_feature_set=FEATURE_SET, decision_time=DECISION)
        (tmp_path / "d1" / "d1.parquet").rename(tmp_path / "d1" / "other.parquet")
        with pytest.raises(FileNotFoundError, match="no parquet"):
            store.read("d1", AssetKind.STOCK, FEATURE_SET, DECISION)

    def test_empty_write_is_rejected(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            store.write(pl.DataFrame(), dataset_id="e", manifest=stock_manifest(), expected_feature_set=FEATURE_SET, decision_time=DECISION)

    def test_store_imports_no_asset_or_execution_package(self) -> None:
        import re
        from pathlib import Path

        text = Path(ParquetDatasetStore.__module__.replace(".", "/") + ".py").read_text(encoding="utf-8")
        imports = re.findall(r"^(?:from|import)\s+(src\.\S+)", text, re.MULTILINE)
        forbidden = ("src.stocks", "src.etfs", "src.execution")
        assert not any(i.startswith(forbidden) for i in imports)


def partitioned_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:1", "KRX:2", "KRX:1", "KRX:2"],
            "session": [
                datetime(2024, 1, 4, tzinfo=UTC),
                datetime(2024, 1, 4, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
            ],
            "observation_time": [
                datetime(2024, 1, 4, 6, 30, tzinfo=UTC),
                datetime(2024, 1, 4, 6, 30, tzinfo=UTC),
                datetime(2024, 2, 1, 6, 30, tzinfo=UTC),
                datetime(2024, 2, 1, 6, 30, tzinfo=UTC),
            ],
            "available_time": [
                datetime(2024, 1, 4, 6, 31, tzinfo=UTC),
                datetime(2024, 1, 4, 6, 31, tzinfo=UTC),
                datetime(2024, 2, 1, 6, 31, tzinfo=UTC),
                datetime(2024, 2, 1, 6, 31, tzinfo=UTC),
            ],
            "close": [100.0, 200.0, 110.0, 220.0],
        }
    ).sort(["instrument_id", "session"])


def multi_horizon_frame() -> pl.DataFrame:
    session = datetime(2024, 1, 4, tzinfo=UTC)
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:1", "KRX:1", "KRX:2", "KRX:2"],
            "session": [session, session, session, session],
            "horizon_sessions": [5, 3, 5, 3],
            "target": [0.5, 0.3, 0.6, 0.4],
        }
    )


def partitioned_manifest(frame: pl.DataFrame) -> DatasetManifest:
    return make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set=FEATURE_SET,
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        time_start=TIME_START,
        time_end=TIME_END,
        provider_version="fixture",
        universe_policy_version="v1",
        row_count=frame.height,
        schema_version="v2",
        content_hash=canonical_content_hash(frame, frame.columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )


class TestPartitionedRoundTrip:
    def test_multi_horizon_hash_is_order_invariant(self) -> None:
        # multi_horizon_hash_is_order_invariant
        frame = multi_horizon_frame()
        reversed_frame = frame.reverse()
        assert canonical_content_hash(frame, frame.columns) == canonical_content_hash(
            reversed_frame, reversed_frame.columns
        )

    def test_multi_horizon_partitioned_round_trip(self, tmp_path) -> None:
        # multi_horizon_partitioned_round_trip
        store = ParquetDatasetStore(tmp_path / "root")
        frame = multi_horizon_frame()
        manifest = partitioned_manifest(frame)
        store.write_partitioned(
            frame.reverse(),
            dataset_id="multi_horizon",
            manifest=manifest,
            expected_feature_set=FEATURE_SET,
            decision_time=DECISION,
        )
        out = store.read("multi_horizon", AssetKind.STOCK, FEATURE_SET, DECISION)
        assert out["horizon_sessions"].to_list() == [3, 5, 3, 5]

    def test_partitioned_write_read_round_trips(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path / "root")
        frame = partitioned_frame()
        manifest = partitioned_manifest(frame)
        store.write_partitioned(
            frame,
            dataset_id="d2",
            manifest=manifest,
            expected_feature_set=FEATURE_SET,
            decision_time=DECISION,
            content_manifest={"curation_version": "curation-v1", "source": {"file_count": 2}},
        )
        out = store.read("d2", AssetKind.STOCK, FEATURE_SET, DECISION)
        assert out.equals(frame)

        dataset_dir = tmp_path / "root" / "d2"
        assert (dataset_dir / "partitions" / "year=2024" / "month=01" / "part-00000.parquet").exists()
        assert (dataset_dir / "partitions" / "year=2024" / "month=02" / "part-00000.parquet").exists()
        import json

        content = json.loads((dataset_dir / "content_manifest.json").read_text())
        assert content["output"]["content_hash"] == manifest.content_hash
        assert len(content["partitions"]) == 2

    def test_existing_dataset_id_is_rejected(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path / "root")
        frame = partitioned_frame()
        manifest = partitioned_manifest(frame)
        store.write_partitioned(
            frame, dataset_id="d2", manifest=manifest,
            expected_feature_set=FEATURE_SET, decision_time=DECISION,
        )
        with pytest.raises(ValueError, match="already exists"):
            store.write_partitioned(
                frame, dataset_id="d2", manifest=manifest,
                expected_feature_set=FEATURE_SET, decision_time=DECISION,
            )

    def test_tampered_partition_fails_closed(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path / "root")
        frame = partitioned_frame()
        manifest = partitioned_manifest(frame)
        store.write_partitioned(
            frame, dataset_id="d2", manifest=manifest,
            expected_feature_set=FEATURE_SET, decision_time=DECISION,
        )
        part = next((tmp_path / "root" / "d2" / "partitions").rglob("*.parquet"))
        with part.open("ab") as fh:
            fh.write(b"tamper")
        with pytest.raises(ValueError, match="tampered partition"):
            store.read("d2", AssetKind.STOCK, FEATURE_SET, DECISION)

    def test_content_hash_mismatch_fails_closed(self, tmp_path) -> None:
        # manifest_content_hash_disagreement_fails_closed
        store = ParquetDatasetStore(tmp_path / "root")
        frame = partitioned_frame()
        manifest = partitioned_manifest(frame)
        store.write_partitioned(
            frame, dataset_id="d2", manifest=manifest,
            expected_feature_set=FEATURE_SET, decision_time=DECISION,
        )
        manifest_path = tmp_path / "root" / "d2" / "dataset_manifest.json"
        data = manifest_path.read_text().replace(manifest.content_hash, "0" * 64)
        manifest_path.write_text(data)
        with pytest.raises(ValueError, match="content hash"):
            store.read("d2", AssetKind.STOCK, FEATURE_SET, DECISION)

    def test_partitioned_write_requires_v2_manifest(self, tmp_path) -> None:
        store = ParquetDatasetStore(tmp_path / "root")
        frame = partitioned_frame()
        v1 = make_manifest(
            asset_kind=AssetKind.STOCK,
            columns=frame.columns,
            feature_set=FEATURE_SET,
            label_definition="fwd_ret_5d",
            label_horizon_sessions=5,
            time_start=TIME_START,
            time_end=TIME_END,
            provider_version="fixture",
            universe_policy_version="v1",
            row_count=frame.height,
        )
        with pytest.raises(ValueError, match="schema_version v2"):
            store.write_partitioned(
                frame, dataset_id="d3", manifest=v1,
                expected_feature_set=FEATURE_SET, decision_time=DECISION,
            )
