"""Feature readiness gate tests against a feature dataset content manifest."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from legacy.stocks.data.readiness import validate_selected_feature_readiness
from tests.fixtures.stocks.helpers import feature_readiness_dataset


def _partition_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "partitions").rglob("*.parquet"))


def test_readiness_reads_only_content_manifest_partitions(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    stray = dataset_dir / "partitions" / "year=2024" / "month=03"
    stray.mkdir(parents=True)
    pl.DataFrame(
        {
            "instrument_id": ["KRX:99999"],
            "session": [datetime(2024, 3, 1, tzinfo=UTC)],
            "feature__overnight_ret": [0.5],
            "feature__ret_21_60d": [0.5],
            "feature__inactive": [None],
            "feature__bad": [0.5],
        }
    ).write_parquet(stray / "part-00000.parquet")

    report = validate_selected_feature_readiness(dataset_dir, ("overnight_ret",))
    assert report.total_rows == 4
    assert report.selected["overnight_ret"].non_null_count == 4


def test_readiness_accepts_selected_non_null_feature(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    report = validate_selected_feature_readiness(dataset_dir, ("overnight_ret",))
    assert report.total_rows == 4
    assert report.selected["overnight_ret"].non_null_count == 4
    assert report.selected["overnight_ret"].null_count == 0
    assert report.selected["overnight_ret"].non_finite_count == 0


def test_readiness_rejects_missing_selected_column(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    with pytest.raises(ValueError, match="missing from partitions"):
        validate_selected_feature_readiness(dataset_dir, ("does_not_exist",))


def test_readiness_rejects_fully_null_selected_column(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    with pytest.raises(ValueError, match="fully null"):
        validate_selected_feature_readiness(dataset_dir, ("inactive",))


def test_readiness_rejects_non_finite_selected_values(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        validate_selected_feature_readiness(dataset_dir, ("bad",))


def test_readiness_reports_warm_up_nulls_without_rejecting(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    report = validate_selected_feature_readiness(
        dataset_dir, ("overnight_ret", "ret_21_60d")
    )
    assert report.total_rows == 4
    assert report.selected["ret_21_60d"].null_count == 2
    assert report.selected["ret_21_60d"].non_null_count == 2
    assert report.selected["ret_21_60d"].non_finite_count == 0
    assert report.selected["overnight_ret"].null_count == 0


def test_readiness_lists_unselected_fully_null_columns_without_rejecting(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    report = validate_selected_feature_readiness(dataset_dir, ("overnight_ret",))
    assert report.fully_null_stored_columns_not_selected == ("feature__inactive",)


def test_readiness_rejects_partition_digest_mismatch(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    part = _partition_files(dataset_dir)[0]
    with part.open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_selected_feature_readiness(dataset_dir, ("overnight_ret",))


def test_readiness_rejects_declared_partition_missing_on_disk(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    part = _partition_files(dataset_dir)[0]
    part.unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_selected_feature_readiness(dataset_dir, ("overnight_ret",))


def test_readiness_never_mutates_the_stored_panel(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    before = [path.read_bytes() for path in _partition_files(dataset_dir)]
    manifest_before = (dataset_dir / "content_manifest.json").read_bytes()

    validate_selected_feature_readiness(dataset_dir, ("overnight_ret",))
    with pytest.raises(ValueError, match="fully null"):
        validate_selected_feature_readiness(dataset_dir, ("inactive",))
    with pytest.raises(ValueError, match="non-finite"):
        validate_selected_feature_readiness(dataset_dir, ("bad",))

    assert [path.read_bytes() for path in _partition_files(dataset_dir)] == before
    assert (dataset_dir / "content_manifest.json").read_bytes() == manifest_before


def test_readiness_requires_at_least_one_selected_feature(tmp_path) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        validate_selected_feature_readiness(dataset_dir, ())
