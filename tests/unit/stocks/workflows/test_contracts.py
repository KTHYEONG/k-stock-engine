"""Stock workflow input contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.stocks.workflows.contracts import SimulationRequest, TrainingRequest


def test_training_request_defaults_are_explicit() -> None:
    request = TrainingRequest(artifact_id="v1")
    assert request.n_folds == 3
    assert request.seed == 42
    assert request.holdout_sessions == 0
    assert request.optuna_trials == 80
    assert request.max_rss_mib is None


def test_training_request_validates_trial_and_budget_inputs() -> None:
    with pytest.raises(ValueError, match="optuna_trials must be positive"):
        TrainingRequest(artifact_id="v1", optuna_trials=0)
    with pytest.raises(ValueError, match="optuna_trials must be positive"):
        TrainingRequest(artifact_id="v1", optuna_trials=-3)
    with pytest.raises(ValueError, match="max_rss_mib must be positive"):
        TrainingRequest(artifact_id="v1", max_rss_mib=0)
    with pytest.raises(ValueError, match="max_rss_mib must be positive"):
        TrainingRequest(artifact_id="v1", max_rss_mib=-5)
    with pytest.raises(ValueError, match="lgb_threads must be positive"):
        TrainingRequest(artifact_id="v1", lgb_threads=0)
    with pytest.raises(ValueError, match="lgb_threads must be positive"):
        TrainingRequest(artifact_id="v1", lgb_threads=-2)
    request = TrainingRequest(artifact_id="v1", optuna_trials=120, max_rss_mib=4096)
    assert request.optuna_trials == 120
    assert request.max_rss_mib == 4096
    assert request.lgb_threads is None
    assert TrainingRequest(artifact_id="v1", lgb_threads=4).lgb_threads == 4


def test_simulation_request_carries_policy_inputs() -> None:
    request = SimulationRequest(
        artifact_id="v1",
        decision_time=datetime(2024, 1, 1, tzinfo=UTC),
        top_k=10,
        participation_limit=0.02,
    )
    assert request.top_k == 10
    assert request.participation_limit == 0.02
