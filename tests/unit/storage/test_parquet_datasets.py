"""Asset-neutral Parquet dataset store: manifest validation and cross-kind rejection."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.datasets import DatasetManifest, make_manifest
from src.core.instruments import AssetKind
from src.storage.parquet_datasets import ParquetDatasetStore

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
