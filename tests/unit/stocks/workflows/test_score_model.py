"""Score-model workflow wiring tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.workflows.contracts import ScoringRequest, TrainingRequest
from src.stocks.workflows.score_model import score_model
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import stock_instrument_df, stock_manifest


def test_score_model_loads_artifact_and_scores(tmp_path) -> None:
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
    scored = score_model(
        snapshot,
        registry,
        ScoringRequest(artifact_id="stock_alpha_v1_20240101", decision_time=decision),
    )
    assert not scored.is_empty()
    assert "pred_score" in scored.columns


def test_score_model_rejects_unavailable_artifact(tmp_path) -> None:
    df = stock_instrument_df(n_sessions=20, n_tickers=2)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    with pytest.raises(FileNotFoundError):
        score_model(
            snapshot,
            registry,
            ScoringRequest(
                artifact_id="missing_v1", decision_time=datetime(2024, 2, 20, tzinfo=UTC)
            ),
        )
