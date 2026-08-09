"""Canonical stock data curation: deterministic projection and certification."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    validate_production_manifest,
)
from src.stocks.data.curation import (
    StockCurationRequest,
    curate_legacy_feature_panel,
)
from src.stocks.research.datasets import (
    ELIGIBLE_STATUS,
    QUALITY_REASON_COLUMN,
    QUALITY_STATUS_COLUMN,
    QUARANTINED_STATUS,
)

DATES = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]


def legacy_row(day_index: int, ticker: str = "000050") -> dict[str, object]:
    close = 105.0 + float(day_index)
    return {
        "date": DATES[day_index],
        "ticker": ticker,
        "open": 100.0 + float(day_index),
        "high": 110.0 + float(day_index),
        "low": 90.0 + float(day_index),
        "close": close,
        "volume": 1_000_000.0,
        "trading_value": 1.05e8,
        "market_cap": 1e12,
        "sector": "S1",
        "log_return_5d": 0.1 + 0.01 * day_index,
        "volatility_20d": 0.2,
        "target_return_5d": 0.05,
        "target_rank": 1.0,
        "label_should_be_dropped": 1.0,
        "name": "fixture",
        "market": "KOSPI",
    }


def write_feature_files(
    root: Path, frames_by_day: dict[int, pl.DataFrame]
) -> None:
    for day_index, frame in frames_by_day.items():
        day = DATES[day_index]
        year_dir = root / f"year={day.year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(year_dir / f"{day.isoformat()}_feat.parquet")


def fixture_source(root: Path, n_days: int = 5, tickers: tuple[str, ...] = ("000050",)) -> Path:
    frames = {
        i: pl.DataFrame([legacy_row(i, ticker) for ticker in tickers])
        for i in range(n_days)
    }
    write_feature_files(root, frames)
    return root


def request(dataset_id: str = "krx_daily_research_v1", **overrides) -> StockCurationRequest:
    base = {
        "dataset_id": dataset_id,
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 31),
        "generated_time": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return StockCurationRequest(**base)


def canonical_frame(result) -> pl.DataFrame:
    return pl.read_parquet(result.partition_paths, hive_partitioning=True)


class TestCurateDeterminism:
    def test_repeated_migration_yields_same_content_and_layout(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        req = request()
        first = curate_legacy_feature_panel(source, tmp_path / "out1", req)
        second = curate_legacy_feature_panel(source, tmp_path / "out2", req)

        assert first.manifest.content_hash == second.manifest.content_hash
        assert first.manifest.schema_hash == second.manifest.schema_hash
        assert canonical_frame(first).equals(canonical_frame(second))
        first_json = (tmp_path / "out1" / "krx_daily_research_v1" / "content_manifest.json").read_text()
        second_json = (tmp_path / "out2" / "krx_daily_research_v1" / "content_manifest.json").read_text()
        assert first_json == second_json

    def test_existing_dataset_id_is_rejected(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        destination = tmp_path / "out"
        curate_legacy_feature_panel(source, destination, request())
        with pytest.raises(ValueError, match="already exists"):
            curate_legacy_feature_panel(source, destination, request())


class TestCurateLeakageAndQuarantine:
    def test_predictor_columns_are_namespaced_and_targets_dropped(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        result = curate_legacy_feature_panel(source, tmp_path / "out", request())
        frame = canonical_frame(result)
        columns = set(frame.columns)

        assert "target_return_5d" not in columns
        assert "target_rank" not in columns
        assert "label_should_be_dropped" not in columns
        assert "name" not in columns
        assert "market" not in columns
        assert "feature__log_return_5d" in columns
        assert "feature__volatility_20d" in columns
        assert {"instrument_id", "session", "observation_time", "available_time"}.issubset(columns)

    def test_invalid_ohlc_rows_are_quarantined(self, tmp_path) -> None:
        source = tmp_path / "src"
        bad = pl.DataFrame([legacy_row(0), legacy_row(1)])
        bad = bad.with_columns(pl.lit(0.0).alias("close"))
        write_feature_files(source, {0: bad, 2: pl.DataFrame([legacy_row(2)])})
        result = curate_legacy_feature_panel(source, tmp_path / "out", request())
        frame = canonical_frame(result)

        quarantined = frame.filter(pl.col(QUALITY_STATUS_COLUMN) == QUARANTINED_STATUS)
        assert quarantined.height == 2
        assert quarantined[QUALITY_REASON_COLUMN][0] == "non_positive_or_missing_ohlc"
        eligible = frame.filter(pl.col(QUALITY_STATUS_COLUMN) == ELIGIBLE_STATUS)
        assert eligible.height == 1

    def test_timestamps_are_timezone_aware_and_ordered(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        result = curate_legacy_feature_panel(source, tmp_path / "out", request())
        frame = canonical_frame(result)

        assert frame.schema["observation_time"].time_zone == "UTC"
        assert frame.schema["available_time"].time_zone == "UTC"
        assert (frame["observation_time"] <= frame["available_time"]).all()
        assert frame["session"].to_list() == sorted(frame["session"].to_list())
        assert frame["instrument_id"].to_list() == ["KRX:000050"] * 5

    def test_index_rows_are_excluded_and_malformed_tickers_rejected(self, tmp_path) -> None:
        source = tmp_path / "src"
        kospi = legacy_row(0)
        kospi["ticker"] = "KOSPI"
        kosdaq = legacy_row(1)
        kosdaq["ticker"] = "KOSDAQ"
        write_feature_files(
            source,
            {
                0: pl.DataFrame([kospi]),
                1: pl.DataFrame([kosdaq]),
                2: pl.DataFrame([legacy_row(2)]),
            },
        )
        result = curate_legacy_feature_panel(source, tmp_path / "out", request())
        assert canonical_frame(result)["instrument_id"].to_list() == ["KRX:000050"]

        malformed = tmp_path / "bad"
        bad_row = legacy_row(0)
        bad_row["ticker"] = "ABC"
        write_feature_files(malformed, {0: pl.DataFrame([bad_row])})
        with pytest.raises(ValueError, match="malformed ticker"):
            curate_legacy_feature_panel(malformed, tmp_path / "out2", request())


class TestCurateFailClosed:
    def test_empty_source_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"feat\.parquet"):
            curate_legacy_feature_panel(tmp_path / "src", tmp_path / "out", request())

    def test_duplicate_rows_are_rejected(self, tmp_path) -> None:
        source = tmp_path / "src"
        dup = pl.concat([pl.DataFrame([legacy_row(0)]), pl.DataFrame([legacy_row(0)])])
        write_feature_files(source, {0: dup})
        with pytest.raises(ValueError, match="duplicate"):
            curate_legacy_feature_panel(source, tmp_path / "out", request())

    def test_schema_variant_is_rejected(self, tmp_path) -> None:
        source = tmp_path / "src"
        variant = pl.DataFrame([legacy_row(1)]).with_columns(pl.lit(1.0).alias("extra"))
        write_feature_files(source, {0: pl.DataFrame([legacy_row(0)]), 1: variant})
        with pytest.raises(ValueError, match="schema variant"):
            curate_legacy_feature_panel(source, tmp_path / "out", request())

    def test_missing_required_column_is_rejected(self, tmp_path) -> None:
        source = tmp_path / "src"
        missing = pl.DataFrame([legacy_row(0)]).drop("sector")
        write_feature_files(source, {0: missing})
        with pytest.raises(ValueError, match="required columns"):
            curate_legacy_feature_panel(source, tmp_path / "out", request())

    def test_non_finite_numeric_is_rejected(self, tmp_path) -> None:
        source = tmp_path / "src"
        bad = pl.DataFrame([legacy_row(0)]).with_columns(pl.lit(float("inf")).alias("close"))
        write_feature_files(source, {0: bad})
        with pytest.raises(ValueError, match="non-finite"):
            curate_legacy_feature_panel(source, tmp_path / "out", request())

    def test_nan_is_normalized_to_null_not_rejected(self, tmp_path) -> None:
        source = tmp_path / "src"
        with_nan = pl.DataFrame([legacy_row(0)]).with_columns(
            pl.lit(float("nan")).alias("volatility_20d")
        )
        write_feature_files(source, {0: with_nan})
        result = curate_legacy_feature_panel(source, tmp_path / "out", request())
        frame = canonical_frame(result)
        assert frame["feature__volatility_20d"].to_list() == [None]


class TestCurateCertificationBoundary:
    def test_migration_output_is_provisional_v2(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        result = curate_legacy_feature_panel(source, tmp_path / "out", request())
        assert result.manifest.certification is DatasetCertification.PROVISIONAL
        assert result.manifest.schema_version == "v2"
        assert result.manifest.content_hash
        assert result.manifest.storage_layout == HIVE_PARTITION_LAYOUT
        assert not result.manifest.calendar_hash

    def test_research_without_evidence_is_rejected(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        with pytest.raises(ValueError, match="evidence"):
            curate_legacy_feature_panel(
                source, tmp_path / "out",
                request(certification=DatasetCertification.RESEARCH),
            )

    def test_research_with_evidence_passes(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        result = curate_legacy_feature_panel(
            source,
            tmp_path / "out",
            request(
                certification=DatasetCertification.RESEARCH,
                calendar_hash="c",
                corporate_action_hash="ca",
                cost_source_hash="cost",
            ),
        )
        assert result.manifest.certification is DatasetCertification.RESEARCH
        assert result.manifest.calendar_hash == "c"

    def test_production_without_evidence_is_rejected(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        with pytest.raises(ValueError, match="evidence"):
            curate_legacy_feature_panel(
                source, tmp_path / "out",
                request(certification=DatasetCertification.PRODUCTION),
            )

    def test_production_with_evidence_passes_production_validation(self, tmp_path) -> None:
        source = fixture_source(tmp_path / "src")
        result = curate_legacy_feature_panel(
            source,
            tmp_path / "out",
            request(
                certification=DatasetCertification.PRODUCTION,
                calendar_hash="c",
                corporate_action_hash="ca",
                cost_source_hash="cost",
            ),
        )
        assert validate_production_manifest(result.manifest) is None
        assert result.manifest.certification is DatasetCertification.PRODUCTION
