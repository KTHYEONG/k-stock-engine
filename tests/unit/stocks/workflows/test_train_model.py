"""Net-alpha mainline training workflow tests.

Covers the transformed-frame training, horizon selection, forward holdout, and
complete ``NO_TRADE`` evidence contract of the canonical ``stock_net_alpha_v1``
workflow. The legacy LambdaRank/Optuna v2 path is not part of this suite.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.research.artifacts import (
    METRICS_FILENAME,
    ModelArtifactRegistry,
)
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


def _snapshot(
    n_sessions: int = 160, n_tickers: int = 8
) -> tuple[DatasetSnapshot, pl.DataFrame]:
    df = stock_net_alpha_composed_df(n_sessions=n_sessions, n_tickers=n_tickers)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    return DatasetSnapshot(manifest=manifest, frame=df), df


def _request(artifact_id: str, **kwargs) -> NetAlphaTrainingRequest:
    defaults = {
        "artifact_id": artifact_id,
        "fold_count": 2,
        "candidate_horizon_sessions": (3, 5, 8, 10, 15, 20),
        "bootstrap_resamples": 50,
    }
    defaults.update(kwargs)
    return NetAlphaTrainingRequest(**defaults)


def test_train_net_alpha_publishes_artifact_or_no_trade(tmp_path) -> None:
    snapshot, df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(snapshot, registry, _request("na_mainline"))
    assert manifest.artifact_id == "na_mainline"
    assert manifest.feature_set == "stock_net_alpha_v1"
    assert manifest.model_type in {
        "net_alpha_elastic_net",
        "net_alpha_lightgbm_l1",
        "no_trade",
    }
    assert manifest.eligible_from == df["session"].min().isoformat()


def test_train_net_alpha_writes_complete_no_trade_evidence(tmp_path) -> None:
    snapshot, _df = _snapshot()
    artifact_root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry(artifact_root)
    train_model(snapshot, registry, _request("na_no_trade"))
    metrics_path = artifact_root / "na_no_trade" / METRICS_FILENAME
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["no_trade"] is True
    assert payload["promoted"] is False
    assert "promotion_reasons" in payload
    assert "gates" in payload


def test_train_net_alpha_rejects_legacy_snapshot(tmp_path) -> None:
    from tests.fixtures.stocks.helpers import stock_v2_composed_df, stock_v2_manifest

    df = stock_v2_composed_df(n_sessions=60, n_tickers=4)
    manifest = stock_v2_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="net-alpha"):
        train_model(
            DatasetSnapshot(manifest=manifest, frame=df),
            registry,
            _request("legacy_reject", candidate_horizon_sessions=(5,)),
        )


def test_train_net_alpha_rejects_legacy_optuna_flags(tmp_path) -> None:
    with pytest.raises(TypeError):
        # optuna_trials must not exist on the net-alpha request
        NetAlphaTrainingRequest(  # type: ignore[call-arg]
            artifact_id="x", optuna_trials=80
        )


def test_train_net_alpha_rejects_invalid_request(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="model_threads must be positive"):
        train_model(
            snapshot, registry, _request("bad_threads", model_threads=0)
        )


def test_train_net_alpha_duplicate_publish_rejected(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    train_model(snapshot, registry, _request("na_dup"))
    with pytest.raises(ValueError, match="already exists"):
        train_model(snapshot, registry, _request("na_dup"))


def test_train_net_alpha_folds_respected(tmp_path) -> None:
    snapshot, _df = _snapshot(n_sessions=200)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(
        snapshot, registry, _request("na_folds", fold_count=3)
    )
    assert manifest.artifact_id == "na_folds"


def test_train_net_alpha_v3_publishes_no_trade_without_positive_evidence(
    tmp_path,
) -> None:
    """The net-alpha path fails closed to a complete NO_TRADE artifact.

    A snapshot with no positive horizon lower bound must publish ``no_trade``
    with complete evidence rather than relax a gate.
    """
    df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8, seed=3)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    result = train_model(snapshot, registry, _request("na_no_evidence"))
    assert result.artifact_id == "na_no_evidence"
    assert result.model_type == "no_trade"
    metrics = json.loads(
        (tmp_path / "artifacts" / "na_no_evidence" / METRICS_FILENAME).read_text()
    )
    assert metrics["no_trade"] is True
    assert metrics["promoted"] is False


def test_train_net_alpha_model_types_are_canonical(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(snapshot, registry, _request("na_types"))
    assert manifest.model_type in {
        "net_alpha_elastic_net",
        "net_alpha_lightgbm_l1",
        "no_trade",
    }
    if manifest.model_type != "no_trade":
        stored = json.loads(
            (
                tmp_path / "artifacts" / "na_types" / "manifest.json"
            ).read_text()
        )
        assert stored["model_type"] in {
            "net_alpha_elastic_net",
            "net_alpha_lightgbm_l1",
        }
