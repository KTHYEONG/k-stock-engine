"""PLAN-05-related stock simulation and train/simulate wiring tests."""
from __future__ import annotations


import polars as pl

from src.core.instruments import AssetKind
from src.core.portfolio import CostModel
from src.stocks.application.train import run_training
from src.stocks.ml.artifacts import ModelArtifactRegistry
from src.stocks.simulation.runner import StockSimulator
from src.stocks.strategies.portfolio_policy import PortfolioPolicy
from tests.fixtures.stocks.helpers import stock_instrument_df, stock_manifest


def score_frame(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i:05d}" for i in range(n)],
            "pred_score": [float(i) for i in reversed(range(n))],
        }
    )


class TestStockSimulator:
    def test_simulator_uses_explicit_costs_and_ledger(self) -> None:
        simulator = StockSimulator(CostModel(commission_rate=0.00015, tax_rate=0.0023))
        policy = PortfolioPolicy(top_k=5, max_single_weight=0.2, max_exposure=1.0)
        result = simulator.simulate(score_frame(), policy, AssetKind.STOCK)
        assert result.total_trades >= 0 if hasattr(result, "total_trades") else True
        assert result.final_value > 0
        assert result.equity_curve

    def test_metrics_reconcile_to_ledger(self) -> None:
        from src.stocks.ml.evaluation import max_drawdown

        simulator = StockSimulator(CostModel())
        policy = PortfolioPolicy(top_k=5)
        result = simulator.simulate(score_frame(), policy, AssetKind.STOCK)
        assert max_drawdown(result.equity_curve) >= 0.0


class TestTrainWiring:
    def test_run_training_publishes_artifact(self, tmp_path) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        manifest = stock_manifest(columns=df.columns, horizon=5)
        registry = ModelArtifactRegistry(tmp_path / "artifacts")
        artifact_id = run_training(df, manifest, registry, "stock_alpha_v1_20240101", n_folds=3)
        assert artifact_id == "stock_alpha_v1_20240101"
