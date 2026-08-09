"""Stock data repository tests: provisional legacy panel certification."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.stocks.data.repositories import read_provisional_legacy_panel
from src.stocks.research.datasets import QUALITY_STATUS_COLUMN, QUARANTINED_STATUS


def write_feature_file(root, year: str, filename: str, frame: pl.DataFrame) -> None:
    year_dir = root / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(year_dir / filename)


def legacy_frame(days: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2024, 1, d) for d in range(1, days + 1)],
            "ticker": ["000050"] * days,
            "open": [100.0] * days,
            "high": [110.0] * days,
            "low": [90.0] * days,
            "close": [105.0 + i for i in range(days)],
            "volume": [1_000_000.0] * days,
            "trading_value": [1.05e8] * days,
            "market_cap": [1e12] * days,
            "feature_momentum_5d": [0.1] * days,
            "target_return_5d": [0.05] * days,
            "target_rank": [1.0] * days,
            "label_should_be_dropped": [1.0] * days,
        }
    )


class TestReadProvisionalLegacyPanel:
    def test_panel_exposes_pit_columns_and_drops_targets(self, tmp_path) -> None:
        write_feature_file(tmp_path, "2024", "2024-01-05_feat.parquet", legacy_frame())
        snapshot = read_provisional_legacy_panel(
            tmp_path,
            date(2024, 1, 1),
            date(2024, 1, 31),
            allowed_features=("feature_momentum_5d",),
        )
        columns = set(snapshot.frame.columns)
        assert {"instrument_id", "session", "observation_time", "available_time"}.issubset(columns)
        assert "feature_momentum_5d" in columns
        assert not columns & {"target_return_5d", "target_rank", "label_should_be_dropped"}
        assert snapshot.manifest.provider_version == "provisional-legacy"
        assert snapshot.frame["instrument_id"].to_list() == ["KRX:000050"] * 5

    def test_duplicate_rows_are_rejected(self, tmp_path) -> None:
        frame = legacy_frame()
        dup = pl.concat([frame, frame.tail(1)])
        write_feature_file(tmp_path, "2024", "2024-01-05_feat.parquet", dup)
        with pytest.raises(ValueError, match="duplicate"):
            read_provisional_legacy_panel(
                tmp_path, date(2024, 1, 1), date(2024, 1, 31), allowed_features=()
            )

    def test_schema_variant_is_rejected(self, tmp_path) -> None:
        write_feature_file(tmp_path, "2024", "a_feat.parquet", legacy_frame(3))
        other = legacy_frame(2).with_columns(pl.lit(1.0).alias("extra_col"))
        write_feature_file(tmp_path, "2024", "b_feat.parquet", other)
        with pytest.raises(ValueError, match="schema variant"):
            read_provisional_legacy_panel(
                tmp_path, date(2024, 1, 1), date(2024, 1, 31), allowed_features=()
            )

    def test_missing_lineage_is_rejected(self, tmp_path) -> None:
        bad = legacy_frame().drop("ticker")
        write_feature_file(tmp_path, "2024", "2024-01-05_feat.parquet", bad)
        with pytest.raises(ValueError, match="lineage"):
            read_provisional_legacy_panel(
                tmp_path, date(2024, 1, 1), date(2024, 1, 31), allowed_features=()
            )

    def test_non_positive_price_is_quarantined(self, tmp_path) -> None:
        bad = legacy_frame().with_columns(pl.lit(0.0).alias("close"))
        write_feature_file(tmp_path, "2024", "2024-01-05_feat.parquet", bad)
        snapshot = read_provisional_legacy_panel(
            tmp_path, date(2024, 1, 1), date(2024, 1, 31), allowed_features=()
        )
        assert snapshot.frame[QUALITY_STATUS_COLUMN].to_list() == [QUARANTINED_STATUS] * 5

    def test_missing_files_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="feat.parquet"):
            read_provisional_legacy_panel(
                tmp_path, date(2024, 1, 1), date(2024, 1, 31), allowed_features=()
            )
