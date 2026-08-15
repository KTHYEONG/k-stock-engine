"""Net-alpha trainer: causal OOF discovery and untouched-holdout contracts."""
from __future__ import annotations

import inspect

from src.stocks.ml import training
from src.stocks.ml.training import _build_horizon_evidence, _evaluate_forward_holdout


def test_horizon_evidence_has_no_proxy_score() -> None:
    assert "_proxy_scores" not in _build_horizon_evidence.__code__.co_names


def test_forward_holdout_contract_signature() -> None:
    parameters = inspect.signature(_evaluate_forward_holdout).parameters
    assert list(parameters) == [
        "model",
        "calibration",
        "holdout_panel",
        "request",
        "horizon_sessions",
    ]


def test_train_net_alpha_model_promotes_champion_or_no_trade(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import NetAlphaTrainingRequest
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from tests.fixtures.stocks.helpers import (
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot,
        datetime(2024, 12, 31, tzinfo=UTC),
        (3, 5, 8, 10, 15, 20),
    )
    registry = ModelArtifactRegistry(Path(tmp_path) / "artifacts")
    request = NetAlphaTrainingRequest(
        artifact_id="na_trainer",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
    )
    manifest = training.train_net_alpha_model(data, registry, request)
    assert manifest.artifact_id == "na_trainer"
    assert manifest.model_type in {
        "net_alpha_elastic_net",
        "net_alpha_lightgbm_l1",
        "no_trade",
    }
