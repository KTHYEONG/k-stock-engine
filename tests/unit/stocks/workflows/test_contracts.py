"""Stock workflow input contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

from src.stocks.workflows.contracts import SimulationRequest, TrainingRequest


def test_training_request_defaults_are_explicit() -> None:
    request = TrainingRequest(artifact_id="v1")
    assert request.n_folds == 3
    assert request.seed == 42
    assert request.holdout_sessions == 0


def test_simulation_request_carries_policy_inputs() -> None:
    request = SimulationRequest(
        artifact_id="v1",
        decision_time=datetime(2024, 1, 1, tzinfo=UTC),
        top_k=10,
        participation_limit=0.02,
    )
    assert request.top_k == 10
    assert request.participation_limit == 0.02
