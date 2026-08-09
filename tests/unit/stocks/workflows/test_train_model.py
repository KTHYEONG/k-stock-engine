"""Model-training workflow: every-fold evaluation and artifact publication tests."""
from __future__ import annotations

import json

import polars as pl
import pytest

from src.core.time import TemporalViolationError
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import stock_instrument_df, stock_manifest


def _snapshot(n_sessions: int = 80, n_tickers: int = 3) -> tuple[DatasetSnapshot, DatasetSnapshot]:
    df = stock_instrument_df(n_sessions=n_sessions, n_tickers=n_tickers, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    return DatasetSnapshot(manifest=manifest, frame=df), df


def test_train_model_publishes_artifact_without_hardcoded_dates(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    model_manifest = train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="stock_alpha_v1_20240101", n_folds=3),
    )
    assert model_manifest.artifact_id == "stock_alpha_v1_20240101"
    assert model_manifest.model_type == "stable_rank_composite"
    assert model_manifest.eligible_from != "2024-01-01T00:00:00+00:00"


def test_train_model_writes_evidence_metrics_for_every_fold(tmp_path) -> None:
    snapshot, _df = _snapshot()
    artifact_root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry(artifact_root)
    train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="stock_alpha_v1_20240101", n_folds=3),
    )
    payload = json.loads(
        (artifact_root / "stock_alpha_v1_20240101" / METRICS_FILENAME).read_text()
    )
    assert payload["n_folds_evaluated"] >= 2
    assert payload["promotion_reasons"]
    assert payload["ledger_metrics"]


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
        TrainingRequest(artifact_id="stock_alpha_v1_20240101", n_folds=3),
    )
    with pytest.raises(ValueError, match="already exists"):
        train_model(
            snapshot,
            registry,
            TrainingRequest(artifact_id="stock_alpha_v1_20240101", n_folds=3),
        )
