"""Durable fingerprinted run store: atomic checkpoints, identity, resume."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.economic_selection import SELECTION_POLICY_VERSION
from src.stocks.workflows.training_run_store import (
    TrainingRunStore,
    content_hash,
)
from tests.fixtures.stocks.helpers import (
    stock_v2_composed_df,
    stock_v2_manifest,
)


def _snapshot() -> DatasetSnapshot:
    df = stock_v2_composed_df(n_sessions=60, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    return DatasetSnapshot(manifest=manifest, frame=df)


def _request(run_root: Path, *, resume: bool = False) -> TrainingRequest:
    return TrainingRequest(
        artifact_id="durable_run",
        n_folds=3,
        resume=resume,
        run_root=run_root,
    )


def _resolve(run_root: Path, request: TrainingRequest | None = None) -> TrainingRunStore:
    request = request or _request(run_root)
    store = TrainingRunStore.resolve(
        _snapshot(),
        request,
        ("feature__a", "feature__b"),
        (),
        default_base_schedule(),
        default_stress_schedule(),
    )
    assert store is not None
    return store


def test_atomic_checkpoint_round_trips_and_validates_content(tmp_path) -> None:
    store = _resolve(tmp_path / "durable_run")
    phase = "screen_h5"
    assert store.completed_phase(phase, "anything") is False
    evidence = {"route_horizon": 5, "optuna_trials": 27}
    path = store.checkpoint_phase(phase, evidence)
    assert path.name == "phase_screen_h5.json"
    assert store.completed_phase(phase, content_hash(evidence)) is True
    assert store.completed_phase(phase, "corrupted") is False
    assert store.phase_evidence(phase)["optuna_trials"] == 27
    assert not path.with_suffix(".json.tmp").exists()


def test_identity_mismatch_raises_value_error(tmp_path) -> None:
    run_root = tmp_path / "durable_run"
    _resolve(run_root)

    changed = TrainingRequest(
        artifact_id="durable_run",
        n_folds=5,
        resume=True,
        run_root=run_root,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _resolve(run_root, changed)


def test_non_resume_with_existing_identity_fails_actionably(tmp_path) -> None:
    run_root = tmp_path / "durable_run"
    _resolve(run_root)
    with pytest.raises(ValueError, match="pass --resume"):
        _resolve(run_root, _request(run_root))


def test_resume_reuses_matching_identity(tmp_path) -> None:
    run_root = tmp_path / "durable_run"
    _resolve(run_root)
    resumed = _resolve(run_root, _request(run_root, resume=True))
    assert resumed.resume is True


def test_resolve_returns_none_without_run_root() -> None:
    request = TrainingRequest(artifact_id="plain", n_folds=3)
    assert (
        TrainingRunStore.resolve(
            _snapshot(),
            request,
            (),
            (),
            default_base_schedule(),
            default_stress_schedule(),
        )
        is None
    )


def test_optuna_storage_url_is_per_route_sqlite(tmp_path) -> None:
    store = _resolve(tmp_path / "durable_run")
    url = store.optuna_storage_url(10)
    assert url.startswith("sqlite:///")
    assert url.endswith("study_h10.db")


def test_identity_records_selection_policy_version(tmp_path) -> None:
    store = _resolve(tmp_path / "durable_run")
    persisted = json.loads(
        (store.root / "run_identity.json").read_text()
    )
    assert persisted["selection_policy_version"] == SELECTION_POLICY_VERSION
    assert persisted["artifact_id"] == "durable_run"
    assert persisted["fingerprint"] == store.identity.fingerprint


def test_resumed_tune_skips_completed_screen_and_reuses_champion(
    tmp_path, monkeypatch,
) -> None:
    """A resumed ``_tune_champion`` reuses validated screen units."""

    import src.stocks.workflows.train_model as tm
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from tests.unit.stocks.workflows.test_train_model import (
        _fake_candidate_context,
        _fold_aware_refit,
        _positive_replay,
        _tune_base_manifest,
    )

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)
    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = tm._index_sessions(df)
    label_span = (manifest.label_horizon_sessions or 1) + 1
    folds = tm.PurgedWalkForward(
        n_folds=3,
        label_horizon_sessions=label_span,
        embargo_sessions=5,
        session_column="session_index",
        validation_window_sessions=30,
        min_train_sessions=40,
    ).split(panel)
    monkeypatch.setattr(
        tm,
        "_fit_stable_contexts",
        lambda _p, _tf, *_a, **_kw: (
            _fake_candidate_context(),
            tuple(_fake_candidate_context() for _ in _tf),
            type("_P", (), {"__call__": lambda self, _i: _fake_candidate_context(), "seed": lambda *a, **k: None, "release": lambda self: None})(),
        ),
    )
    monkeypatch.setattr(
        tm,
        "_score_trial_fold",
        lambda *_a, **_kw: tm.ScreenFoldEvidence(
            rank_ic=0.05,
            attempted_orders=10,
            filled_orders=8,
            planned_cycles=1,
            complete_block_count=4,
            rejected_block_count=0,
            block_log_excess_mean=0.01,
            lower_bound=0.01,
            dsr_probability=0.97,
            usable=True,
            failure_reason=None,
            no_trade_reason_counts={},
            block_log_excess=(0.01, 0.01, 0.01, 0.01),
        ),
    )
    monkeypatch.setattr(tm, "_fit_and_score_candidate", _fold_aware_refit())
    monkeypatch.setattr(tm, "_event_ledger_evaluation", lambda *_a, **_kw: _positive_replay())

    def run(resume: bool):
        request = TrainingRequest(
            artifact_id="durable_resume",
            n_folds=3,
            optuna_trials=3,
            resume=resume,
            run_root=tmp_path / "durable_resume",
        )
        store = TrainingRunStore.resolve(
            _snapshot(),
            request,
            tuple(c for c in df.columns if c.startswith("feature__")),
            (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
            default_base_schedule(),
            default_stress_schedule(),
        )
        assert store is not None
        return tm._tune_champion(
            panel[folds[0].train_mask],
            request,
            _tune_base_manifest("durable_resume", manifest, manifest.label_definition),
            tuple(c for c in df.columns if c.startswith("feature__")),
            (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
            dataset_manifest=manifest,
            registry=ModelArtifactRegistry(tmp_path / "artifacts"),
            base_schedule=default_base_schedule(),
            stress_schedule=default_stress_schedule(),
            run_store=store,
        ), store

    (first_config, first_trials, first_route), first_store = run(resume=False)
    assert first_config is not None
    screen_path = first_store.phase_path("screen_h5")
    assert screen_path.exists()

    (resumed_config, resumed_trials, resumed_route), resumed_store = run(resume=True)
    assert resumed_config is not None
    assert resumed_trials == first_trials
    assert resumed_route == first_route
    first_selected = first_config._tuning_telemetry.get("selected_trial_number")
    resumed_selected = resumed_config._tuning_telemetry.get("selected_trial_number")
    assert resumed_selected == first_selected
    assert resumed_store.completed_phase(
        "screen_h5", content_hash(first_store.phase_evidence("screen_h5"))
    )
