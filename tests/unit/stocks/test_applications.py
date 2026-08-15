"""Stock workflow wiring: train, simulate, generate_intents."""
from __future__ import annotations

from datetime import datetime, UTC

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation
from src.execution.domain.intents import TradeIntent
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.workflows.contracts import ScoringRequest, SimulationRequest
from src.stocks.workflows.generate_intents import generate_intents
from src.stocks.workflows.score_model import score_model
from src.stocks.workflows.simulate_portfolio import (
    artifact_policy_profile,
    simulate_portfolio,
)
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


class TestGenerateIntents:
    def test_intents_carry_asset_kind_and_idempotency_key(self) -> None:
        instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
        decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
        intents = generate_intents(
            [Allocation(instrument=instrument, target_value=0.5, reason="rank")],
            strategy_id="stock_alpha_v1",
            decision_time=decision,
            execution_time=decision,
            account_snapshot_id="account-a",
        )
        assert len(intents) == 1
        intent = intents[0]
        assert isinstance(intent, TradeIntent)
        assert intent.asset_kind is AssetKind.STOCK
        assert intent.idempotency_key
        assert intent.target_value == 0.5


def build_trained_snapshot(tmp_path) -> tuple[DatasetSnapshot, ModelArtifactRegistry]:
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
    return snapshot, registry


class TestScoreModelWiring:
    def test_score_model_loads_artifact_and_scores(self, tmp_path) -> None:
        snapshot, registry = build_trained_snapshot(tmp_path)
        decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
        scored = score_model(
            snapshot,
            registry,
            ScoringRequest(artifact_id="stock_net_alpha_20240101", decision_time=decision),
        )
        assert not scored.is_empty()
        assert "predicted_net_alpha" in scored.columns

    def test_simulate_portfolio_reconciles(self, tmp_path) -> None:
        snapshot, registry = build_trained_snapshot(tmp_path)
        decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
        policy_profile = artifact_policy_profile(registry, "stock_net_alpha_20240101")
        if policy_profile is not None:
            request = SimulationRequest(
                artifact_id="stock_net_alpha_20240101",
                decision_time=decision,
                top_k=int(policy_profile["top_k"]),
                max_single_weight=float(policy_profile["max_single_weight"]),
                max_exposure=float(policy_profile["max_exposure"]),
                participation_limit=float(policy_profile["participation_limit"]),
                policy_profile_id=str(policy_profile["profile_id"]),
                no_trade_band_bps=float(policy_profile["no_trade_band_bps"]),
            )
        else:
            request = SimulationRequest(
                artifact_id="stock_net_alpha_20240101", decision_time=decision
            )
        result = simulate_portfolio(snapshot, registry, request)
        assert result.final_value > 0
        assert result.total_return is not None
