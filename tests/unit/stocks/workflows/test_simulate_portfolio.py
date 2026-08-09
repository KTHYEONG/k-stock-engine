"""Portfolio simulation workflow wiring tests."""
from __future__ import annotations

from datetime import UTC, datetime

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.workflows.contracts import SimulationRequest, TrainingRequest
from src.stocks.workflows.simulate_portfolio import simulate_portfolio
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import stock_instrument_df, stock_manifest


def test_simulate_portfolio_reconciles_and_returns_metrics(tmp_path) -> None:
    df = stock_instrument_df(n_sessions=80, n_tickers=3, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="stock_alpha_v1_20240101", n_folds=3),
    )
    decision = datetime(2024, 2, 20, 8, 50, tzinfo=UTC)
    result = simulate_portfolio(
        snapshot,
        registry,
        SimulationRequest(
            artifact_id="stock_alpha_v1_20240101", decision_time=decision
        ),
    )
    assert result.final_value > 0
    assert result.total_return is not None
    assert "cagr" in result.metrics
