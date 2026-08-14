"""PLAN-05-related stock simulation and train/simulate wiring tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.costs import CostSchedule, CostPoint, default_base_schedule
from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.allocation_policy import AllocationPolicy
from src.stocks.trading.simulator import StockSimulator
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import (
    stock_instrument_df,
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
    stock_v2_composed_df,
    stock_v2_manifest,
)


def scored_panel(n_sessions: int = 30, n_tickers: int = 5) -> pl.DataFrame:
    df = stock_instrument_df(n_sessions=n_sessions, n_tickers=n_tickers)
    return df.with_columns(pl.lit(1.0).alias("pred_score"))


class TestStockSimulator:
    def test_simulator_uses_explicit_costs_and_ledger(self) -> None:
        simulator = StockSimulator(
            cost_schedule=default_base_schedule(),
            stress_schedule=CostSchedule(
                name="stress",
                points=(
                    CostPoint(
                        effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                        commission_rate=0.0005,
                        tax_rate=0.0033,
                        slippage_bps=25.0,
                    ),
                ),
            ),
        )
        policy = AllocationPolicy(top_k=5, max_single_weight=0.2, max_exposure=1.0)
        result = simulator.simulate(scored_panel(), policy, AssetKind.STOCK)

        assert result.ledger
        for row in result.ledger:
            equity = float(row["equity"])
            reconciled = (
                float(row["settled_cash"])
                + float(row["unsettled_cash"])
                + float(row["positions_value"])
                - float(row["accrued_costs"])
            )
            assert abs(equity - reconciled) < 1e-8
            for fill in result.trades:
                if fill.get("quantity") is not None:
                    assert int(fill["quantity"]) == fill["quantity"]

    def test_simulator_never_invents_a_price(self) -> None:
        panel = scored_panel(n_sessions=20)
        bad = panel.with_columns(
            pl.when(pl.col("session_index") == 5)
            .then(None)
            .otherwise(pl.col("open"))
            .alias("open")
        )
        simulator = StockSimulator(cost_schedule=default_base_schedule())
        policy = AllocationPolicy(top_k=5, max_exposure=1.0)
        result = simulator.simulate(bad, policy, AssetKind.STOCK)
        reasons = [t.get("reason") for t in result.trades if t.get("reason")]
        assert reasons, "expected unfilled reasons for a missing open"
        for row in result.ledger:
            assert abs(
                float(row["equity"])
                - (
                    float(row["settled_cash"])
                    + float(row["unsettled_cash"])
                    + float(row["positions_value"])
                    - float(row["accrued_costs"])
                )
            ) < 1e-8

    def test_metrics_reconcile_to_ledger(self) -> None:
        from src.stocks.research.metrics import max_drawdown

        simulator = StockSimulator(cost_schedule=default_base_schedule())
        policy = AllocationPolicy(top_k=5)
        result = simulator.simulate(scored_panel(), policy, AssetKind.STOCK)
        assert result.metrics["max_drawdown"] == pytest.approx(
            max_drawdown(result.equity_curve)
        )
        assert result.metrics["turnover"] >= 0.0
        assert result.metrics["cost_drag"] >= 0.0

    def test_rebalance_sells_positions_missing_from_new_target(self) -> None:
        panel = scored_panel(n_sessions=30, n_tickers=2).with_columns(
            pl.when(
                (pl.col("session_index") < 15)
                & (pl.col("instrument_id") == "KRX:00001")
            )
            .then(2.0)
            .otherwise(0.0)
            .alias("pred_score")
        ).with_columns(
            pl.when(
                (pl.col("session_index") >= 15)
                & (pl.col("instrument_id") == "KRX:00002")
            )
            .then(2.0)
            .otherwise(pl.col("pred_score"))
            .alias("pred_score")
        )
        result = StockSimulator(cost_schedule=default_base_schedule()).simulate(
            panel,
            AllocationPolicy(top_k=1, max_single_weight=1.0, max_exposure=1.0),
            AssetKind.STOCK,
        )
        assert any(
            trade.get("side") == "sell" for trade in result.trades
        )

    def test_buy_cost_is_not_charged_twice(self) -> None:
        panel = pl.DataFrame(
            {
                "instrument_id": ["KRX:00001"] * 12,
                "session": [datetime(2024, 1, 1 + i, tzinfo=UTC) for i in range(12)],
                "open": [100.0] * 12,
                "close": [100.0] * 12,
                "volume": [1_000_000.0] * 12,
                "trading_value": [100_000_000.0] * 12,
                "pred_score": [1.0] * 12,
            }
        )
        schedule = CostSchedule(
            name="commission-only",
            points=(
                CostPoint(
                    effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                    commission_rate=0.001,
                    tax_rate=0.0,
                    slippage_bps=0.0,
                ),
            ),
        )
        result = StockSimulator(
            cost_schedule=schedule,
            initial_cash=100_000.0,
            adtv_participation_limit=0.0,
        ).simulate(
            panel,
            AllocationPolicy(
                top_k=1,
                max_single_weight=1.0,
                max_exposure=1.0,
                volatility_column=None,
            ),
            AssetKind.STOCK,
        )
        buy = next(trade for trade in result.trades if trade.get("side") == "buy")
        assert result.final_value == pytest.approx(100_000.0 - float(buy["cost"]))

    def test_stress_result_is_independent_from_base_result(self) -> None:
        base = default_base_schedule()
        stress = CostSchedule(
            name="stress",
            points=(
                CostPoint(
                    effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                    commission_rate=0.01,
                    tax_rate=0.0,
                    slippage_bps=0.0,
                ),
            ),
        )
        result = StockSimulator(
            cost_schedule=base,
            stress_schedule=stress,
        ).simulate(
            scored_panel(n_sessions=20, n_tickers=2),
            AllocationPolicy(top_k=1),
            AssetKind.STOCK,
        )
        assert result.stress_final_value is not None
        assert result.stress_final_value < result.final_value

    def test_simulator_rejects_missing_columns(self) -> None:
        simulator = StockSimulator(cost_schedule=default_base_schedule())
        policy = AllocationPolicy(top_k=5)
        with pytest.raises(ValueError, match="panel must carry"):
            simulator.simulate(
                scored_panel().drop("volume"), policy, AssetKind.STOCK
            )


class TestTrainWiring:
    def test_run_training_publishes_artifact(self, tmp_path) -> None:
        df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8)
        manifest = stock_net_alpha_manifest(columns=df.columns)
        registry = ModelArtifactRegistry(tmp_path / "artifacts")
        snapshot = DatasetSnapshot(manifest=manifest, frame=df)
        model_manifest = train_model(
            snapshot,
            registry,
            NetAlphaTrainingRequest(
                artifact_id="stock_net_alpha_20240101", fold_count=2,
                candidate_horizon_sessions=(5,), bootstrap_resamples=50,
            ),
        )
        assert model_manifest.artifact_id == "stock_net_alpha_20240101"
        assert model_manifest.model_type in (
            "net_alpha_elastic_net", "net_alpha_lightgbm_l1", "no_trade",
        )
        assert model_manifest.eligible_from == df["session"].min().isoformat()

    def test_run_training_writes_evidence_metrics(self, tmp_path) -> None:
        from src.stocks.research.artifacts import METRICS_FILENAME
        import json

        df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8)
        manifest = stock_net_alpha_manifest(columns=df.columns)
        artifact_root = tmp_path / "artifacts"
        registry = ModelArtifactRegistry(artifact_root)
        snapshot = DatasetSnapshot(manifest=manifest, frame=df)
        train_model(
            snapshot,
            registry,
            NetAlphaTrainingRequest(
                artifact_id="stock_net_alpha_20240101", fold_count=2,
                candidate_horizon_sessions=(5,), bootstrap_resamples=50,
            ),
        )
        metrics_path = artifact_root / "stock_net_alpha_20240101" / METRICS_FILENAME
        assert metrics_path.exists()
        payload = json.loads(metrics_path.read_text())
        assert payload["promoted"] is False
        assert payload["no_trade"] is True
        assert "promotion_reasons" in payload

    def test_training_rejects_non_net_alpha_snapshot(self, tmp_path) -> None:
        df = stock_v2_composed_df(n_sessions=60, n_tickers=3)
        manifest = stock_v2_manifest(columns=df.columns)
        registry = ModelArtifactRegistry(tmp_path / "artifacts")
        with pytest.raises(ValueError, match="net-alpha"):
            train_model(
                DatasetSnapshot(manifest=manifest, frame=df),
                registry,
                NetAlphaTrainingRequest(
                    artifact_id="leak_net_alpha", fold_count=2,
                    candidate_horizon_sessions=(5,), bootstrap_resamples=50,
                ),
            )

    def test_duplicate_version_publish_is_rejected(self, tmp_path) -> None:
        df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8)
        manifest = stock_net_alpha_manifest(columns=df.columns)
        registry = ModelArtifactRegistry(tmp_path / "artifacts")
        snapshot = DatasetSnapshot(manifest=manifest, frame=df)
        train_model(
            snapshot,
            registry,
            NetAlphaTrainingRequest(
                artifact_id="stock_net_alpha_20240101", fold_count=2,
                candidate_horizon_sessions=(5,), bootstrap_resamples=50,
            ),
        )
        with pytest.raises(ValueError, match="already exists"):
            train_model(
                snapshot,
                registry,
                NetAlphaTrainingRequest(
                    artifact_id="stock_net_alpha_20240101", fold_count=2,
                    candidate_horizon_sessions=(5,), bootstrap_resamples=50,
                ),
            )
