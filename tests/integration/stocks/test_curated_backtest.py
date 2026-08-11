"""Curated dataset integration: repository round-trip, training, and plan replay."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.curation import StockCurationRequest, curate_legacy_feature_panel
from src.stocks.data.repositories import StockDatasetRepository
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.contracts import SimulationRequest, TrainingRequest
from src.stocks.workflows.simulate_portfolio import simulate_portfolio
from src.stocks.workflows.train_model import train_model
from src.stocks.workflows.trading_cycle import CycleStatus, TradingCycleRequest, run_trading_cycle
from src.storage.parquet_datasets import ParquetDatasetStore
from tests.fixtures.stocks.helpers import stock_v2_manifest

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


def as_v2_snapshot(snapshot) -> pl.DataFrame:
    """Augment a curated panel with v2 feature columns and residual labels."""
    from src.stocks.research.features import stock_alpha_v2_allowlist
    from src.stocks.research.labels import residual_open_to_open_label

    allowlist = stock_alpha_v2_allowlist()
    frame = snapshot.frame
    rng = np.random.default_rng(11)
    rows = []
    for row in frame.iter_rows(named=True):
        for index, name in enumerate(allowlist):
            row[f"feature__{name}"] = float(rng.normal(0.0, 1.0)) + (index % 5) * 0.1
        rows.append(row)
    augmented = pl.DataFrame(rows)
    labels = residual_open_to_open_label(
        augmented.select(["instrument_id", "session", "open"])
    )
    return augmented.join(labels, on=["instrument_id", "session"], how="left")


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
    dataset_root, artifact_root = curated
    repo = StockDatasetRepository(ParquetDatasetStore(dataset_root))
    snapshot = repo.read(DATASET_ID, "stock_alpha_v1", datetime(2025, 1, 1, tzinfo=UTC))
    v2_frame = as_v2_snapshot(snapshot)
    v2_manifest = stock_v2_manifest(columns=v2_frame.columns)
    v2_snapshot = DatasetSnapshot(manifest=v2_manifest, frame=v2_frame)
    registry = ModelArtifactRegistry(artifact_root)

    model_manifest = train_model(
        v2_snapshot, registry, TrainingRequest(artifact_id="stock_alpha_v2_curated", n_folds=3)
    )
    assert model_manifest.feature_set == "stock_alpha_v2"

    result = simulate_portfolio(
        v2_snapshot,
        registry,
        SimulationRequest(
            artifact_id="stock_alpha_v2_curated",
            decision_time=datetime(2024, 4, 1, tzinfo=UTC),
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
    dataset_root, artifact_root = curated
    repo = StockDatasetRepository(ParquetDatasetStore(dataset_root))
    snapshot = repo.read(DATASET_ID, "stock_alpha_v1", datetime(2025, 1, 1, tzinfo=UTC))
    v2_frame = as_v2_snapshot(snapshot)
    v2_manifest = stock_v2_manifest(columns=v2_frame.columns)
    v2_snapshot = DatasetSnapshot(manifest=v2_manifest, frame=v2_frame)
    registry = ModelArtifactRegistry(artifact_root)
    train_model(
        v2_snapshot, registry, TrainingRequest(artifact_id="stock_alpha_v2_curated", n_folds=3)
    )

    decision_time = datetime(2024, 3, 15, 8, 50, tzinfo=UTC)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="paper",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    cycle = run_trading_cycle(
        v2_snapshot,
        registry,
        instruments_from(v2_snapshot),
        portfolio,
        TradingCycleRequest(
            strategy_id="stock_alpha_v2",
            artifact_id="stock_alpha_v2_curated",
            dataset_id=DATASET_ID,
            decision_time=decision_time,
            execution_time=datetime(2024, 3, 18, 0, 0, tzinfo=UTC),
            risk_policy=StockRiskPolicy(),
            mode="plan",
        ),
    )
    latest_available = v2_snapshot.frame.filter(
        v2_snapshot.frame["available_time"] <= decision_time
    )["session"].max()
    assert cycle.status in (CycleStatus.PLANNED, CycleStatus.NO_TRADE)
    if cycle.status is CycleStatus.PLANNED:
        cross_sections = [
            r.split("=", 1)[1] for r in cycle.reasons if r.startswith("cross_section_session=")
        ]
        assert cross_sections
        assert datetime.fromisoformat(cross_sections[0]) == latest_available
