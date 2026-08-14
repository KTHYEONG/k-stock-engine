"""Portfolio simulation workflow wiring tests."""
from __future__ import annotations

from datetime import UTC, datetime

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import simulate_portfolio
from src.stocks.workflows.train_model import train_model
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


def test_simulate_portfolio_reconciles_and_returns_metrics(tmp_path) -> None:
    df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    train_model(
        snapshot,
        registry,
        NetAlphaTrainingRequest(
            artifact_id="stock_net_alpha_20240101",
            fold_count=2,
            candidate_horizon_sessions=(5,),
            bootstrap_resamples=50,
        ),
    )
    decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
    result = simulate_portfolio(
        snapshot,
        registry,
        SimulationRequest(
            artifact_id="stock_net_alpha_20240101", decision_time=decision
        ),
    )
    assert result.final_value > 0
    assert result.total_return is not None
    assert "cagr" in result.metrics
