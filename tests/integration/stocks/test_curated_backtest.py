"""Curated dataset integration: repository round-trip, training, and plan replay."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.curation import StockCurationRequest, curate_legacy_feature_panel
from src.stocks.data.repositories import StockDatasetRepository
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import simulate_portfolio
from src.stocks.workflows.train_model import train_model
from src.stocks.workflows.trading_cycle import CycleStatus, TradingCycleRequest, run_trading_cycle
from src.storage.parquet_datasets import ParquetDatasetStore
from tests.fixtures.stocks.helpers import stock_net_alpha_manifest

DATASET_ID = "krx_daily_research_v1_20240101_20240409"
START = date(2024, 1, 1)
N_SESSIONS = 100
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
        "log_return_5d": 0.01 * ((day_index + int(ticker[-2:])) % 7),
        "volatility_20d": 0.2,
        "target_return_5d": 0.05,
    }


def write_source(root: Path) -> None:
    for i in range(N_SESSIONS):
        day = START + timedelta(days=i)
        year_dir = root / f"year={day.year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([legacy_row(i, t) for t in TICKERS]).write_parquet(
            year_dir / f"{day.isoformat()}_feat.parquet"
        )


@pytest.fixture
def curated(tmp_path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    dataset_root = tmp_path / "datasets"
    write_source(source)
    curate_legacy_feature_panel(
        source,
        dataset_root,
        StockCurationRequest(
            dataset_id=DATASET_ID,
            start_date=START,
            end_date=date(2024, 4, 30),
            generated_time=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    return dataset_root, tmp_path / "artifacts"


def instruments_from(snapshot) -> dict[str, Instrument]:
    return {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW")
        for i in sorted(snapshot.frame["instrument_id"].unique().to_list())
    }


def as_net_alpha_snapshot(snapshot) -> DatasetSnapshot:
    """Augment a curated panel with raw net-alpha sources and per-horizon labels."""
    from tests.fixtures.stocks.helpers import stock_net_alpha_composed_df

    frame = stock_net_alpha_composed_df(
        n_sessions=N_SESSIONS, n_tickers=len(TICKERS)
    )
    manifest = stock_net_alpha_manifest(columns=frame.columns)
    return DatasetSnapshot(manifest=manifest, frame=frame)


def test_curated_repository_round_trip_preserves_content_hash(curated) -> None:
    dataset_root, _artifacts = curated
    repo = StockDatasetRepository(ParquetDatasetStore(dataset_root))
    manifest = repo.store.read_manifest(DATASET_ID)
    snapshot = repo.read(DATASET_ID, "stock_alpha_v1", datetime(2025, 1, 1, tzinfo=UTC))

    assert manifest.schema_version == "v2"
    assert snapshot.manifest.content_hash == manifest.content_hash
    assert snapshot.manifest.asset_kind is AssetKind.STOCK
    assert snapshot.frame["instrument_id"].to_list() == sorted(snapshot.frame["instrument_id"].to_list())
    columns = set(snapshot.frame.columns)
    assert not columns & {"target_return_5d", "target_rank"}


def test_curated_dataset_trains_and_simulates_in_plan_mode(curated) -> None:
    _dataset_root, artifact_root = curated
    net_alpha_snapshot = as_net_alpha_snapshot(curated[0])
    registry = ModelArtifactRegistry(artifact_root)

    model_manifest = train_model(
        net_alpha_snapshot,
        registry,
        NetAlphaTrainingRequest(
            artifact_id="stock_net_alpha_curated",
            fold_count=2,
            candidate_horizon_sessions=(5,),
            bootstrap_resamples=50,
        ),
    )
    assert model_manifest.feature_set == "stock_net_alpha_v1"

    result = simulate_portfolio(
        net_alpha_snapshot,
        registry,
        SimulationRequest(
            artifact_id="stock_net_alpha_curated",
            decision_time=datetime(2024, 4, 9, 0, 0, tzinfo=UTC),
        ),
    )
    assert result.ledger
    for row in result.ledger:
        assert (
            abs(
                row.equity
                - (row.settled_cash + row.unsettled_cash + row.positions_value - row.accrued_costs)
            )
            <= 1e-8
        )
    assert result.final_value > 0


def test_curated_replay_uses_only_rows_available_at_decision(curated) -> None:
    _dataset_root, artifact_root = curated
    net_alpha_snapshot = as_net_alpha_snapshot(curated[0])
    registry = ModelArtifactRegistry(artifact_root)
    train_model(
        net_alpha_snapshot,
        registry,
        NetAlphaTrainingRequest(
            artifact_id="stock_net_alpha_curated",
            fold_count=2,
            candidate_horizon_sessions=(5,),
            bootstrap_resamples=50,
        ),
    )

    decision_time = datetime(2024, 4, 9, 0, 0, tzinfo=UTC)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="paper",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    cycle = run_trading_cycle(
        net_alpha_snapshot,
        registry,
        instruments_from(net_alpha_snapshot),
        portfolio,
        TradingCycleRequest(
            strategy_id="stock_net_alpha_v1",
            artifact_id="stock_net_alpha_curated",
            dataset_id=DATASET_ID,
            decision_time=decision_time,
            execution_time=datetime(2024, 4, 10, 0, 0, tzinfo=UTC),
            risk_policy=StockRiskPolicy(),
            mode="plan",
        ),
    )
    latest_available = net_alpha_snapshot.frame.filter(
        net_alpha_snapshot.frame["available_time"] <= decision_time
    )["session"].max()
    assert cycle.status in (CycleStatus.PLANNED, CycleStatus.NO_TRADE)
    if cycle.status is CycleStatus.PLANNED:
        cross_sections = [
            r.split("=", 1)[1] for r in cycle.reasons if r.startswith("cross_section_session=")
        ]
        assert cross_sections
        assert datetime.fromisoformat(cross_sections[0]) == latest_available
