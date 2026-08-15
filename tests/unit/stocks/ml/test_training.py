"""Net-alpha trainer: causal OOF discovery and untouched-holdout contracts."""
from __future__ import annotations

import inspect
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

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


def _training_fixture(n_sessions: int = 120) -> tuple[object, object, object, list[object], tuple[str, ...]]:
    """Composed net-alpha fixture: data, request, pre_holdout, folds, learner columns."""
    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import NetAlphaTrainingRequest
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.ml.features import build_model_features, stock_net_alpha_v1_roles
    from src.stocks.research.folds import PurgedWalkForward
    from tests.fixtures.stocks.helpers import (
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=n_sessions, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC),
        (3, 5, 8, 10, 15, 20),
    )
    roles = dict(stock_net_alpha_v1_roles())
    transformed, learner_columns = build_model_features(data.feature_frame, roles)
    panel = training._index_sessions(transformed)
    request = NetAlphaTrainingRequest(
        artifact_id="na_test",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
    )
    pre_holdout, _holdout, reason = training._locked_holdout(panel, request)
    assert reason == ""
    splitter = PurgedWalkForward(
        n_folds=2,
        label_horizon_sessions=6,
        embargo_sessions=5,
        session_column="session_index",
        validation_window_sessions=20,
        min_train_sessions=40,
    )
    folds = splitter.split(pre_holdout)
    return data, request, pre_holdout, folds, learner_columns


def test_score_is_constant_classifies_degenerate_predictions() -> None:
    assert training._score_is_constant(np.asarray([0.0, 0.0])) is True
    assert training._score_is_constant(np.asarray([0.0, 1.0])) is False
    assert training._score_is_constant(np.asarray([np.nan, np.nan])) is True
    assert training._score_is_constant(np.asarray([1.0, np.nan, 1.0])) is True
    assert training._score_is_constant(np.asarray([1.0, 2.0, np.nan])) is False
    assert training._score_is_constant(np.asarray([])) is True


def test_select_elastic_alpha_recovers_linear_signal() -> None:
    from datetime import timedelta

    from src.stocks.ml.contracts import NetAlphaTrainingRequest, RegularizationGrid
    from src.stocks.research.models import ModelManifest

    rng = np.random.default_rng(7)
    rows: list[dict] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for s in range(80):
        for t in range(12):
            feature = float(rng.normal(0.0, 1.0))
            target = 0.1 * feature + rng.normal(0.0, 0.01)
            rows.append(
                {
                    "session_index": s,
                    "session": start + timedelta(days=s),
                    "instrument_id": f"KRX:{t:05d}",
                    "feature__test_x": feature,
                    "net_alpha_target": target,
                    "realized_net_return": target,
                }
            )
    fold_train = pl.DataFrame(rows)
    request = NetAlphaTrainingRequest(artifact_id="na_alpha")
    manifest = ModelManifest(
        artifact_id="na_alpha",
        asset_kind="stock",
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=3,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
    )
    alpha, fraction, alpha_max = training._select_elastic_alpha(
        fold_train, request, ("feature__test_x",), 3, RegularizationGrid(), manifest
    )
    assert alpha is not None
    assert alpha > 0.0
    assert alpha_max is not None
    assert alpha_max > 0.0
    assert fraction in RegularizationGrid().fractions
    assert alpha == pytest.approx(fraction * alpha_max)


def test_fit_oof_reports_fit_error_instead_of_swallowing() -> None:
    from src.stocks.ml.contracts import FoldScoreDiagnostic

    data, request, pre_holdout, folds, learner_columns = _training_fixture()
    assert folds
    manifest = training._base_manifest(request, data, data.feature_frame, 5)

    class _ExplodingModel:
        def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
            del train, validation
            raise ValueError("boom")

        def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
            del frame
            raise RuntimeError("unreachable")

        def manifest(self) -> object:
            return manifest

    oof, oof_labels, rank_ics, diagnostic = training._fit_oof(
        pre_holdout, folds, data, request, manifest, learner_columns, 5,
        _ExplodingModel, family="net_alpha_lightgbm_l1",
    )
    assert oof.is_empty()
    assert oof_labels.is_empty()
    assert rank_ics == []
    assert any(
        isinstance(diag, FoldScoreDiagnostic)
        and "fit-error:ValueError:boom" in diag.failure_reason
        for diag in diagnostic.fold_diagnostics
    )


def test_horizon_evidence_constant_baseline_triggers_structural_fallback(monkeypatch) -> None:
    from src.stocks.ml.contracts import FoldScoreDiagnostic, HorizonOOFDiagnostic

    data, request, pre_holdout, _folds, learner_columns = _training_fixture()
    real_fit_oof = training._fit_oof
    call_families: list[str] = []

    def spy_fit_oof(*args, **kwargs):
        family = kwargs["family"]
        call_families.append(family)
        if family == "net_alpha_elastic_net":
            return (
                pl.DataFrame(),
                pl.DataFrame(),
                [],
                HorizonOOFDiagnostic(
                    horizon_sessions=args[6],
                    model_family="net_alpha_elastic_net",
                    fold_diagnostics=(
                        FoldScoreDiagnostic(
                            fold_index=0, failure_reason="constant-oof-score"
                        ),
                    ),
                ),
            )
        return real_fit_oof(*args, **kwargs)

    monkeypatch.setattr(training, "_fit_oof", spy_fit_oof)
    discovery = training._build_horizon_evidence(
        pre_holdout, data.feature_frame, data, request, learner_columns
    )
    assert "net_alpha_lightgbm_l1" in call_families
    assert discovery.diagnostics
    assert any(
        d.model_family == "net_alpha_lightgbm_l1" for d in discovery.diagnostics
    )


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
