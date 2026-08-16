"""Integration: quality hardening certification gate across curation and replay."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import DatasetCertification
from src.stocks.data.curation import StockCurationRequest, curate_legacy_feature_panel
from src.stocks.data.quality import (
    CorporateActionInterval,
    CorporateActionSnapshot,
    FeatureAvailabilityRecord,
    InstrumentMasterRecord,
    InstrumentMasterSnapshot,
    KRXSessionCalendar,
)

START = date(2024, 1, 1)
N_SESSIONS = 20
TICKERS = ("000050", "000060", "000070")


def legacy_row(day_index: int, ticker: str) -> dict[str, object]:
    close = 100.0 + float((day_index + int(ticker[-2:])) % 30)
    return {
        "date": START + timedelta(days=day_index),
        "ticker": ticker,
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 1.5,
        "close": close,
        "volume": 1_000_000.0,
        "trading_value": close * 1_000_000.0,
        "market_cap": close * 10_000_000.0,
        "sector": "S1",
        "log_return_5d": 0.01,
        "volatility_20d": 0.2,
        "target_return_5d": 0.05,
    }


def sessions() -> tuple[date, ...]:
    return tuple(START + timedelta(days=i) for i in range(N_SESSIONS))


def write_source(root: Path) -> None:
    for i in range(N_SESSIONS):
        day = START + timedelta(days=i)
        year_dir = root / f"year={day.year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([legacy_row(i, t) for t in TICKERS]).write_parquet(
            year_dir / f"{day.isoformat()}_feat.parquet"
        )


def master_snapshot() -> InstrumentMasterSnapshot:
    dates = sessions()
    records = tuple(
        InstrumentMasterRecord(
            source_identifier=ticker,
            instrument_id=f"KRX:{ticker}",
            asset_type="common_stock",
            is_common_stock=True,
            listed_from=dates[0],
            tradable_from=dates[0],
            tradable_to=dates[-1],
            available_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        for ticker in TICKERS
    )
    return InstrumentMasterSnapshot(
        version="int-master", records=records, generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def calendar_snapshot() -> KRXSessionCalendar:
    return KRXSessionCalendar(
        version="int-calendar", sessions=sessions(), generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def actions_snapshot() -> CorporateActionSnapshot:
    dates = sessions()
    intervals = tuple(
        CorporateActionInterval(
            instrument_id=f"KRX:{ticker}",
            previous_session=dates[i - 1],
            session=dates[i],
            action_code="no_action",
            adjustment_factor=1.0,
        )
        for ticker in TICKERS
        for i in range(1, len(dates))
    )
    return CorporateActionSnapshot(
        version="int-actions", intervals=intervals, generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def base_request(dataset_id: str, **overrides) -> StockCurationRequest:
    params = {
        "dataset_id": dataset_id,
        "start_date": START,
        "end_date": date(2024, 1, 31),
        "generated_time": datetime(2026, 1, 1, tzinfo=UTC),
        "feature_availability": tuple(
            FeatureAvailabilityRecord(
                feature_name=name,
                source_field=name,
                availability_rule="fixture-eod",
                source_version="fixture-v1",
                source_hash="fixture-hash",
                null_rate=0.0,
                use_class="research",
            )
            for name in ("log_return_5d", "volatility_20d")
        ),
    }
    params.update(overrides)
    return StockCurationRequest(**params)


def read_report(result) -> dict[str, object]:
    import json

    return json.loads(result.quality_report_path.read_text())


class TestProvisionalCompletesWithQuarantines:
    def test_provisional_curation_migrates_every_row(self, tmp_path) -> None:
        source = tmp_path / "source"
        write_source(source)
        result = curate_legacy_feature_panel(
            source, tmp_path / "datasets", base_request("krx_provisional_v1")
        )
        assert result.manifest.certification is DatasetCertification.PROVISIONAL
        assert result.row_count == N_SESSIONS * len(TICKERS)
        report = read_report(result)
        assert report["row_counts"]["eligible"] == N_SESSIONS * len(TICKERS)
        assert report["row_counts"]["quarantined"] == 0
        assert report["hashes"]["quality_report"]


class TestResearchRequiresEvidence:
    def test_research_without_master_fails_closed(self, tmp_path) -> None:
        source = tmp_path / "source"
        write_source(source)
        with pytest.raises(ValueError, match="InstrumentMasterSnapshot"):
            curate_legacy_feature_panel(
                source,
                tmp_path / "datasets",
                base_request(
                    "krx_research_fail",
                    certification=DatasetCertification.RESEARCH,
                    calendar_hash="c",
                    corporate_action_hash="ca",
                    cost_source_hash="cost",
                ),
            )

    def test_research_with_full_evidence_passes(self, tmp_path) -> None:
        source = tmp_path / "source"
        write_source(source)
        result = curate_legacy_feature_panel(
            source,
            tmp_path / "datasets",
            base_request(
                "krx_research_v1",
                certification=DatasetCertification.RESEARCH,
                calendar_hash="c",
                corporate_action_hash="ca",
                cost_source_hash="cost",
                instrument_master=master_snapshot(),
                corporate_actions=actions_snapshot(),
                calendar=calendar_snapshot(),
            ),
        )
        assert result.manifest.certification is DatasetCertification.RESEARCH
        assert result.manifest.master_hash
        assert result.manifest.quality_report_hash
        report = read_report(result)
        assert report["certification"] == "research"
        assert report["hashes"]["master"]
        assert report["hashes"]["calendar"]
        assert report["hashes"]["actions"]

    def test_research_public_price_curation_does_not_require_action_hash(self, tmp_path) -> None:
        source = tmp_path / "source"
        write_source(source)

        result = curate_legacy_feature_panel(
            source,
            tmp_path / "datasets",
            base_request(
                "krx_research_public_price_v1",
                certification=DatasetCertification.RESEARCH,
                calendar_hash="c",
                cost_source_hash="cost",
                instrument_master=master_snapshot(),
                calendar=calendar_snapshot(),
            ),
        )

        assert result.manifest.corporate_action_hash == ""
        assert result.manifest.certification is DatasetCertification.RESEARCH

    def test_production_requires_production_gate(self, tmp_path) -> None:
        source = tmp_path / "source"
        write_source(source)
        result = curate_legacy_feature_panel(
            source,
            tmp_path / "datasets",
            base_request(
                "krx_production_v1",
                certification=DatasetCertification.PRODUCTION,
                calendar_hash="c",
                corporate_action_hash="ca",
                cost_source_hash="cost",
                instrument_master=master_snapshot(),
                corporate_actions=actions_snapshot(),
                calendar=calendar_snapshot(),
            ),
        )
        assert result.manifest.certification is DatasetCertification.PRODUCTION
        report = read_report(result)
        assert report["hashes"]["quality_report"]
