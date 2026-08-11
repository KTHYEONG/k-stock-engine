"""Model-training workflow: temporal isolation, artifact publication, NO_TRADE gates."""
from __future__ import annotations

import json

import polars as pl
import pytest

from src.core.time import TemporalViolationError
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import (
    stock_v2_composed_df,
    stock_v2_manifest,
)


def _snapshot(n_sessions: int = 80, n_tickers: int = 3) -> tuple[DatasetSnapshot, pl.DataFrame]:
    df = stock_v2_composed_df(n_sessions=n_sessions, n_tickers=n_tickers)
    manifest = stock_v2_manifest(columns=df.columns)
    return DatasetSnapshot(manifest=manifest, frame=df), df


def test_train_model_publishes_v2_artifact_without_hardcoded_dates(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    model_manifest = train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="stock_alpha_v2_20240101", n_folds=3),
    )
    assert model_manifest.artifact_id == "stock_alpha_v2_20240101"
    assert model_manifest.model_type == "lambdarank_blend"
    assert model_manifest.feature_set == "stock_alpha_v2"
    first_session = snapshot.frame["session"].min().isoformat()
    assert model_manifest.eligible_from == first_session


def test_train_model_writes_no_trade_evidence_for_short_history(tmp_path) -> None:
    snapshot, _df = _snapshot()
    artifact_root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry(artifact_root)
    train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="stock_alpha_v2_20240101", n_folds=3),
    )
    payload = json.loads(
        (artifact_root / "stock_alpha_v2_20240101" / METRICS_FILENAME).read_text()
    )
    assert payload["promoted"] is False
    assert payload["no_trade"] is True
    assert payload["promotion_reasons"]
    assert payload["model_type"] == "lambdarank_blend"


def test_train_model_rejects_temporal_leakage(tmp_path) -> None:
    snapshot, _df = _snapshot(n_sessions=30, n_tickers=2)
    bad = snapshot.frame.with_columns(
        (snapshot.frame["available_time"] + pl.duration(hours=2)).alias("observation_time")
    )
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(TemporalViolationError):
        train_model(
            DatasetSnapshot(manifest=snapshot.manifest, frame=bad),
            registry,
            TrainingRequest(artifact_id="leak_v1", n_folds=2),
        )


def test_duplicate_version_publish_is_rejected(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="stock_alpha_v2_20240101", n_folds=3),
    )
    with pytest.raises(ValueError, match="already exists"):
        train_model(
            snapshot,
            registry,
            TrainingRequest(artifact_id="stock_alpha_v2_20240101", n_folds=3),
        )


def test_train_model_rejects_frame_without_v2_features(tmp_path) -> None:
    snapshot, _df = _snapshot()
    stripped = snapshot.frame.drop([c for c in snapshot.frame.columns if c.startswith("feature__")])
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="stock_alpha_v2 feature columns"):
        train_model(
            DatasetSnapshot(manifest=snapshot.manifest, frame=stripped),
            registry,
            TrainingRequest(artifact_id="missing_v2", n_folds=2),
        )
