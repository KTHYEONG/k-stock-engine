"""Stock application wiring: train, simulate, generate_intents."""
from __future__ import annotations

from datetime import datetime, UTC

from src.core.instruments import AssetKind
from src.core.portfolio import Allocation
from src.execution.domain.order import TradeIntent
from src.stocks.application.generate_intents import generate_intents
from src.stocks.application.simulate import run_simulation, run_simulation_with_policy
from src.stocks.application.train import run_training
from src.stocks.ml.artifacts import ModelArtifactRegistry
from tests.fixtures.stocks.helpers import stock_instrument_df, stock_manifest


class TestGenerateIntents:
    def test_intents_carry_asset_kind_and_idempotency_key(self) -> None:
        from src.core.instruments import Instrument

        instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
        decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
        intents = generate_intents(
            [Allocation(instrument=instrument, target_value=0.5, reason="rank")],
            strategy_id="stock_alpha_v1",
            decision_time=decision,
            execution_time=decision,
        )
        assert len(intents) == 1
        intent = intents[0]
        assert isinstance(intent, TradeIntent)
        assert intent.asset_kind is AssetKind.STOCK
        assert intent.idempotency_key
        assert intent.target_value == 0.5


class TestRunSimulationWiring:
    def test_run_simulation_loads_artifact_and_scores(self, tmp_path) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        manifest = stock_manifest(columns=df.columns, horizon=5)
        registry = ModelArtifactRegistry(tmp_path / "artifacts")
        run_training(df, manifest, registry, "stock_alpha_v1_20240101", n_folds=3)

        decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
        scored = run_simulation(
            registry, "stock_alpha_v1_20240101", df, manifest, decision
        )
        assert not scored.is_empty()
        assert "pred_score" in scored.columns

    def test_run_simulation_with_policy_reconciles(self, tmp_path) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=3, horizon=5)
        manifest = stock_manifest(columns=df.columns, horizon=5)
        registry = ModelArtifactRegistry(tmp_path / "artifacts")
        run_training(df, manifest, registry, "stock_alpha_v1_20240101", n_folds=3)

        decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
        result = run_simulation_with_policy(
            registry, "stock_alpha_v1_20240101", df, manifest, decision
        )
        assert result.final_value > 0
        assert result.total_return is not None
