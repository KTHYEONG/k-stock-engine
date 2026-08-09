"""Integration: backtest replay provenance and eligible-only PIT consumption."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetCertification
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    StockBacktester,
)
from src.stocks.data.curation import StockCurationRequest, curate_legacy_feature_panel
from src.stocks.data.quality import (
    CorporateActionInterval,
    CorporateActionSnapshot,
    FeatureAvailabilityRecord,
    InstrumentMasterRecord,
    InstrumentMasterSnapshot,
    KRXSessionCalendar,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from tests.fixtures.stocks.helpers import publish_baseline_artifact

START = date(2024, 1, 1)
N_SESSIONS = 40
TICKERS = ("000050", "000060", "000070")
ARTIFACT_ID = "quality_backtest_a001"


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
        "feature_momentum_5d": float((day_index + int(ticker[-2:])) % 7) / 7.0,
        "adtv": close * 1_000_000.0,
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
        version="bt-master", records=records, generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


def calendar_snapshot() -> KRXSessionCalendar:
    return KRXSessionCalendar(
        version="bt-calendar", sessions=sessions(), generated_time=datetime(2026, 1, 1, tzinfo=UTC)
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
        version="bt-actions", intervals=intervals, generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


@pytest.fixture
def curated_dataset(tmp_path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    dataset_root = tmp_path / "datasets"
    artifact_root = tmp_path / "artifacts"
    write_source(source)
    curate_legacy_feature_panel(
        source,
        dataset_root,
        StockCurationRequest(
            dataset_id="krx_quality_backtest_v1",
            start_date=START,
            end_date=date(2024, 2, 10),
            certification=DatasetCertification.RESEARCH,
            calendar_hash="c",
            corporate_action_hash="ca",
            cost_source_hash="cost",
            instrument_master=master_snapshot(),
            corporate_actions=actions_snapshot(),
            calendar=calendar_snapshot(),
            feature_availability=tuple(
                FeatureAvailabilityRecord(
                    feature_name=name,
                    source_field=name,
                    availability_rule="fixture-eod",
                    source_version="fixture-v1",
                    source_hash="fixture-hash",
                    null_rate=0.0,
                    use_class="research",
                )
                for name in ("log_return_5d", "volatility_20d", "feature_momentum_5d", "adtv")
            ),
            generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    return dataset_root, artifact_root


def build_backtester(snapshot, artifact_root: Path) -> StockBacktester:
    registry = ModelArtifactRegistry(artifact_root)
    publish_baseline_artifact(
        registry,
        artifact_id=ARTIFACT_ID,
        feature_schema_hash=snapshot.manifest.schema_hash,
        ranking_feature="rev_5d",
    )
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW")
        for i in sorted(snapshot.frame["instrument_id"].unique().to_list())
    }
    return StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )


def replay_panel(snapshot) -> pl.DataFrame:
    """Eligible canonical rows plus the derived ADTV capacity input.

    The trading-cycle cross-section consumes ``adtv`` as a capacity input; it is
    derived deterministically from the canonical trading value exactly as the
    replay engine's own capacity gate does.
    """
    return snapshot.frame.with_columns(
        pl.col("trading_value")
        .rolling_mean(20, min_samples=1)
        .over("instrument_id")
        .alias("adtv")
    )


def replay_request(snapshot) -> BacktestRequest:
    sessions_dt = sorted(snapshot.frame["session"].unique().to_list())
    start_time = sessions_dt[0]
    end_time = sessions_dt[-1]
    decision_indices = tuple(range(len(sessions_dt) - 1))
    return BacktestRequest(
        strategy_id="quality_backtest",
        start_time=start_time,
        end_time=end_time,
        decision_session_indices=decision_indices,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=StockRiskPolicy(top_k=5),
    )


def test_replay_accepts_only_eligible_pit_rows(curated_dataset) -> None:
    from src.stocks.data.repositories import StockDatasetRepository
    from src.storage.parquet_datasets import ParquetDatasetStore

    dataset_root, artifact_root = curated_dataset
    repo = StockDatasetRepository(ParquetDatasetStore(dataset_root))
    snapshot = repo.read("krx_quality_backtest_v1", "stock_alpha_v1", datetime(2025, 1, 1, tzinfo=UTC))

    assert snapshot.frame["data_quality_status"].unique().to_list() == ["eligible"]
    assert "action_interval_covered" in snapshot.frame.columns
    assert snapshot.frame["action_interval_covered"].null_count() == 0
    assert snapshot.frame["action_interval_covered"].to_list() == [True] * snapshot.frame.height

    backtester = build_backtester(snapshot, artifact_root)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="bt",
        as_of=datetime(2024, 1, 2, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                eligible_to=datetime(2024, 3, 1, tzinfo=UTC),
                artifact_id=ARTIFACT_ID,
            ),
        )
    )
    result = backtester.run(replay_panel(snapshot), artifacts, portfolio, replay_request(snapshot))
    assert result.ledger
    assert result.data_quality["dataset_content_hash"] == snapshot.manifest.content_hash
    assert result.data_quality["quality_report_hash"] == snapshot.manifest.quality_report_hash
    assert result.data_quality["master_hash"] == snapshot.manifest.master_hash
    assert result.data_quality["calendar_hash"] == snapshot.manifest.calendar_hash
    assert result.data_quality["action_hash"] == snapshot.manifest.corporate_action_hash
    assert result.data_quality["cost_hash"] == snapshot.manifest.cost_source_hash


def test_replay_rejects_uncovered_action_intervals(curated_dataset) -> None:
    from src.stocks.data.repositories import StockDatasetRepository
    from src.storage.parquet_datasets import ParquetDatasetStore

    dataset_root, artifact_root = curated_dataset
    repo = StockDatasetRepository(ParquetDatasetStore(dataset_root))
    snapshot = repo.read("krx_quality_backtest_v1", "stock_alpha_v1", datetime(2025, 1, 1, tzinfo=UTC))
    backtester = build_backtester(snapshot, artifact_root)

    tampered = replay_panel(snapshot).with_columns(pl.lit(False).alias("action_interval_covered"))
    portfolio = PortfolioSnapshot(
        account_snapshot_id="bt",
        as_of=datetime(2024, 1, 2, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                eligible_to=datetime(2024, 3, 1, tzinfo=UTC),
                artifact_id=ARTIFACT_ID,
            ),
        )
    )
    with pytest.raises(ValueError, match="uncovered action interval"):
        backtester.run(tampered, artifacts, portfolio, replay_request(snapshot))


def test_quality_report_records_lineage(curated_dataset) -> None:
    dataset_root, _artifact_root = curated_dataset
    report_path = dataset_root / "krx_quality_backtest_v1" / "quality_report.json"
    report = json.loads(report_path.read_text())
    assert report["certification"] == "research"
    assert report["action_coverage"]["uncovered"] == 0
    assert report["action_coverage"]["covered"] == (N_SESSIONS - 1) * len(TICKERS)
    assert report["hashes"]["master"]
    assert report["hashes"]["calendar"]
    assert report["hashes"]["actions"]
