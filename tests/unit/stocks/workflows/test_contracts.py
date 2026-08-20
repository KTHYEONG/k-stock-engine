"""Stock workflow input contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.workflows.contracts import SimulationRequest


def test_training_request_defaults_are_explicit() -> None:
    request = NetAlphaTrainingRequest(artifact_id="v1")
    assert request.fold_count == 3
    assert request.seed == 42
    assert request.forward_holdout_sessions == 0
    assert request.candidate_horizon_sessions == (10, 20)
    assert request.max_rss_mib is None
    assert request.model_threads == 1


def test_training_request_validates_inputs() -> None:
    with pytest.raises(ValueError, match="model_threads must be positive"):
        NetAlphaTrainingRequest(artifact_id="v1", model_threads=0)
    with pytest.raises(ValueError, match="max_rss_mib must be positive"):
        NetAlphaTrainingRequest(artifact_id="v1", max_rss_mib=0)
    with pytest.raises(ValueError, match="max_rss_mib must be positive"):
        NetAlphaTrainingRequest(artifact_id="v1", max_rss_mib=-5)
    with pytest.raises(ValueError, match="candidate_horizon_sessions must be non-empty"):
        NetAlphaTrainingRequest(artifact_id="v1", candidate_horizon_sessions=())
    with pytest.raises(ValueError, match="candidate_horizon_sessions must be strictly ascending"):
        NetAlphaTrainingRequest(artifact_id="v1", candidate_horizon_sessions=(5, 3))
    request = NetAlphaTrainingRequest(artifact_id="v1", model_threads=2, max_rss_mib=4096)
    assert request.model_threads == 2
    assert request.max_rss_mib == 4096
    assert not hasattr(request, "optuna_trials")
    assert not hasattr(request, "lgb_threads")
    assert not hasattr(request, "run_root")
    assert not hasattr(request, "resume")


def test_simulation_request_carries_policy_inputs() -> None:
    request = SimulationRequest(
        artifact_id="v1",
        decision_time=datetime(2024, 1, 1, tzinfo=UTC),
        top_k=10,
        participation_limit=0.02,
    )
    assert request.top_k == 10
    assert request.participation_limit == 0.02
