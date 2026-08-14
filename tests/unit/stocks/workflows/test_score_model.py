"""Score-model workflow wiring tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.workflows.contracts import ScoringRequest
from src.stocks.workflows.score_model import score_model
from src.stocks.workflows.train_model import train_model
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


def test_score_model_loads_artifact_and_scores(tmp_path) -> None:
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
    scored = score_model(
        snapshot,
        registry,
        ScoringRequest(
            artifact_id="stock_net_alpha_20240101", decision_time=decision
        ),
    )
    assert not scored.is_empty()
    assert "predicted_net_alpha" in scored.columns


def test_score_model_rejects_unavailable_artifact(tmp_path) -> None:
    df = stock_net_alpha_composed_df(n_sessions=60, n_tickers=6)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    with pytest.raises(FileNotFoundError):
        score_model(
            snapshot,
            registry,
            ScoringRequest(
                artifact_id="missing_net_alpha",
                decision_time=datetime(2024, 2, 20, tzinfo=UTC),
            ),
        )
