"""Model-training workflow: temporal isolation, artifact publication, NO_TRADE gates."""
from __future__ import annotations

import json
from dataclasses import replace as dataclass_replace

import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.time import TemporalViolationError
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import (
    PromotionRiskBudget,
    ReplayResult,
    _evaluate_gates,
    _index_sessions,
    train_model,
)
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


def test_event_replay_executes_intents_for_scored_allocations(tmp_path) -> None:
    from src.stocks.workflows.train_model import (
        _event_ledger_evaluation,
        _prepare_replay_static_context,
    )

    df = stock_v2_composed_df(n_sessions=40, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    oos_scored = df.tail(12).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    panel = _index_sessions(df)
    request = TrainingRequest(artifact_id="candidate", n_folds=3)
    replay = _event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
    )
    assert replay.planned_cycles > 0
    assert replay.attempted_orders > 0
    assert replay.filled_orders > 0
    assert "constraint:insufficient covariance data" not in replay.no_trade_reason_counts
    assert replay.base_total_return != 0.0 or replay.strategy_returns
    assert replay.benchmark_returns

    context = _prepare_replay_static_context(panel, request)
    assert context.cache_bytes > 0
    cached = _event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
        replay_context=context,
    )
    assert cached.planned_cycles == replay.planned_cycles
    assert cached.attempted_orders == replay.attempted_orders
    assert cached.filled_orders == replay.filled_orders
    assert cached.base_total_return == replay.base_total_return
    assert cached.stress_total_return == replay.stress_total_return
    assert cached.benchmark_total_return == replay.benchmark_total_return
    assert cached.metrics == replay.metrics
    assert cached.stress_metrics == replay.stress_metrics
    assert cached.no_trade_reason_counts == replay.no_trade_reason_counts
    assert cached.ledger == replay.ledger
    assert cached.trades == replay.trades
    assert cached.strategy_returns == replay.strategy_returns
    assert cached.excess_returns == replay.excess_returns
    assert cached.benchmark_returns == replay.benchmark_returns
    assert cached.final_value == replay.final_value


def test_training_uses_requested_purged_walk_forward_fold_count(tmp_path, monkeypatch) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)
    monkeypatch.setattr(
        tm,
        "_tune_champion",
        lambda *_args, **_kwargs: (tm.LambdaRankConfig(n_estimators=20, early_stopping_rounds=5), 3),
    )

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="fold_test", n_folds=3),
    )
    payload = json.loads(
        (tmp_path / "artifacts" / "fold_test" / METRICS_FILENAME).read_text()
    )
    assert payload["n_folds_evaluated"] == 3

    label_span = (manifest.label_horizon_sessions or 1) + 1
    folds = tm.PurgedWalkForward(
        n_folds=3,
        label_horizon_sessions=label_span,
        embargo_sessions=5,
        session_column="session_index",
        validation_window_sessions=30,
        min_train_sessions=40,
    ).split(_index_sessions(df))
    assert len(folds) == 3
    assert all(
        folds[k].validation_decision_start < folds[k + 1].validation_decision_start
        for k in range(len(folds) - 1)
    )


def test_stress_gate_requires_actual_stress_excess() -> None:
    budget = PromotionRiskBudget()
    request = TrainingRequest(artifact_id="candidate", n_folds=3)
    shared = {
        "attempted_orders": 5,
        "filled_orders": 3,
        "strategy_returns": [0.001, 0.002, -0.001, 0.001, 0.0, 0.001],
        "benchmark_returns": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    flat = _evaluate_gates(
        ReplayResult(stress_total_return=-0.06, benchmark_total_return=-0.05, **shared),
        [0.05, 0.06, 0.07],
        budget,
        request,
        n_trials=10,
    )
    assert "gate4_stress_cost_excess=False" in flat["reasons"]
    assert flat["passed"] is False

    better_but_negative = _evaluate_gates(
        ReplayResult(stress_total_return=-0.01, benchmark_total_return=-0.05, **shared),
        [0.05, 0.06, 0.07],
        budget,
        request,
        n_trials=10,
    )
    assert "gate4_stress_cost_excess=True" in better_but_negative["reasons"]

    positive = _evaluate_gates(
        ReplayResult(stress_total_return=0.01, benchmark_total_return=-0.05, **shared),
        [0.05, 0.06, 0.07],
        budget,
        request,
        n_trials=10,
    )
    assert "gate4_stress_cost_excess=True" in positive["reasons"]

    missing = _evaluate_gates(
        ReplayResult(stress_total_return=None, benchmark_total_return=-0.05, **shared),
        [0.05, 0.06, 0.07],
        budget,
        request,
        n_trials=10,
    )
    assert "gate4_stress_cost_excess=False" in missing["reasons"]

    empty_benchmark = _evaluate_gates(
        ReplayResult(
            stress_total_return=0.01,
            benchmark_total_return=float("nan"),
            **shared,
        ),
        [0.05, 0.06, 0.07],
        budget,
        request,
        n_trials=10,
    )
    assert "gate4_stress_cost_excess=False" in empty_benchmark["reasons"]


def test_forward_holdout_is_single_use_and_requires_252_label_available_sessions(
    tmp_path,
) -> None:
    import src.stocks.workflows.train_model as tm

    registry = ModelArtifactRegistry(tmp_path / "artifacts")

    ready, reason, evidence = tm._evaluate_forward_holdout(
        registry,
        TrainingRequest(artifact_id="holdout_v1", n_folds=3),
        None,
        None,
        None,
        None,
        (),
        "residual_o2o_5d",
        None,
        None,
        default_base_schedule(),
        default_stress_schedule(),
    )
    assert ready is False
    assert "gate8_forward_holdout_ready=false" in reason
    assert evidence is None

    snapshot, _df = _snapshot()
    train_model(
        snapshot,
        registry,
        TrainingRequest(artifact_id="holdout_v1", n_folds=3),
    )
    payload = json.loads(
        (tmp_path / "artifacts" / "holdout_v1" / METRICS_FILENAME).read_text()
    )
    assert payload["no_trade"] is True

    registry.write_forward_holdout("holdout_v1", "fp-abc", {"ok": True})
    with pytest.raises(ValueError, match="already inspected"):
        registry.write_forward_holdout("holdout_v1", "fp-abc", {"ok": False})


def test_deflated_sharpe_consumes_trials_and_fails_closed() -> None:
    import numpy as np

    from src.stocks.workflows.train_model import _deflated_sharpe_probability

    returns = (0.0005 + 0.01 * np.sin(np.linspace(0, 10 * np.pi, 252))).tolist()
    many_trials = _deflated_sharpe_probability(returns, annualization=252, n_trials=80)
    few_trials = _deflated_sharpe_probability(returns, annualization=252, n_trials=3)
    assert many_trials < few_trials
    assert 0.0 < few_trials < 1.0

    assert _deflated_sharpe_probability([], annualization=252, n_trials=80) == 0.0
    assert _deflated_sharpe_probability([0.001], annualization=252, n_trials=80) == 0.0
    assert _deflated_sharpe_probability(returns, annualization=252, n_trials=0) == 0.0
    assert (
        _deflated_sharpe_probability(
            [float("nan"), 0.01, 0.02], annualization=252, n_trials=80
        )
        == 0.0
    )


def _tune_base_manifest(
    artifact_id: str, manifest, label_definition: str
) -> ModelManifest:
    import src.stocks.workflows.train_model as tm

    return tm.ModelManifest(
        artifact_id=artifact_id,
        asset_kind=__import__("src.core.instruments", fromlist=["AssetKind"]).AssetKind.STOCK,
        feature_set="stock_alpha_v2",
        feature_schema_hash="hash",
        universe_policy_hash="universe",
        label_definition=label_definition,
        label_horizon_sessions=manifest.label_horizon_sessions,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="lambdarank_blend",
    )


def _positive_replay(**overrides) -> ReplayResult:
    """Deterministic economically eligible replay with fixed positive evidence."""
    base = ReplayResult(
        attempted_orders=10,
        filled_orders=8,
        strategy_returns=[0.001] * 24,
        benchmark_returns=[0.0] * 24,
        excess_returns=[0.001] * 24,
        metrics={"max_drawdown": 0.05, "turnover": 1.0},
        base_total_return=0.01,
        benchmark_total_return=0.005,
        stress_total_return=0.01,
        final_value=100_000_001.0,
    )
    return dataclass_replace(base, **overrides)


def test_tuning_never_includes_first_outer_oos(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
    label_span = (manifest.label_horizon_sessions or 1) + 1
    folds = tm.PurgedWalkForward(
        n_folds=3,
        label_horizon_sessions=label_span,
        embargo_sessions=5,
        session_column="session_index",
        validation_window_sessions=30,
        min_train_sessions=40,
    ).split(panel)
    assert len(folds) == 3

    captured: dict[str, int] = {}

    def fake_stable_contexts(tuning_panel, tuning_folds, *_args, **_kwargs):
        captured["tuning_max_session"] = int(tuning_panel["session_index"].max())
        captured["tuning_min_session"] = int(tuning_panel["session_index"].min())
        return [None] * len(tuning_folds)

    monkeypatch.setattr(tm, "_fit_stable_contexts", fake_stable_contexts)
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: 0.01)
    monkeypatch.setattr(
        tm,
        "_fit_and_score_candidate",
        lambda *_a, **_kw: ([0.05, 0.06, 0.07], pl.DataFrame()),
    )
    monkeypatch.setattr(tm, "_event_ledger_evaluation", lambda *_a, **_kw: _positive_replay())

    request = TrainingRequest(artifact_id="tune_oos", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("tune_oos", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert captured["tuning_max_session"] < folds[0].validation_decision_start
    assert n_trials == request.optuna_trials
    assert config is not None


def test_tuning_counts_pruned_trials_for_deflated_sharpe(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=100, n_tickers=10)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
    label_span = (manifest.label_horizon_sessions or 1) + 1
    folds = tm.PurgedWalkForward(
        n_folds=2,
        label_horizon_sessions=label_span,
        embargo_sessions=5,
        session_column="session_index",
        validation_window_sessions=30,
        min_train_sessions=40,
    ).split(panel)
    assert len(folds) == 2

    monkeypatch.setattr(
        tm,
        "_fit_stable_contexts",
        lambda _panel, tuning_folds, *_a, **_kw: [None] * len(tuning_folds),
    )
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: None)

    request = TrainingRequest(artifact_id="tune_prune", n_folds=2, optuna_trials=4)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("tune_prune", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert n_trials == request.optuna_trials
    assert config is None
    assert tm.LambdaRankConfig._tuning_telemetry["selection_status"] == (
        "no_complete_screen_candidate"
    )

def test_tuning_economic_tie_breaks_by_lowest_trial_number(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
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
        lambda _panel, tuning_folds, *_a, **_kw: [None] * len(tuning_folds),
    )
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: 0.01)
    monkeypatch.setattr(
        tm,
        "_fit_and_score_candidate",
        lambda *_a, **_kw: ([0.05, 0.06, 0.07], pl.DataFrame()),
    )
    monkeypatch.setattr(tm, "_event_ledger_evaluation", lambda *_a, **_kw: _positive_replay())

    request = TrainingRequest(artifact_id="tie_break", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("tie_break", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert config is not None
    assert n_trials == request.optuna_trials
    telemetry = config._tuning_telemetry
    assert telemetry["selection_status"] == "selected"
    assert telemetry["selected_trial_number"] == 0
    assert telemetry["screened_trials"] == request.optuna_trials
    assert telemetry["shortlisted_trials"] == request.optuna_trials
    assert telemetry["economically_eligible_trials"] == request.optuna_trials
    assert telemetry["selected_inner_bootstrap_lower_bound"] > 0.0


def test_tuning_rejects_economically_ineligible_candidates(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
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
        lambda _panel, tuning_folds, *_a, **_kw: [None] * len(tuning_folds),
    )
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: 0.01)
    monkeypatch.setattr(
        tm,
        "_fit_and_score_candidate",
        lambda *_a, **_kw: ([0.05, 0.06, 0.07], pl.DataFrame()),
    )
    monkeypatch.setattr(
        tm,
        "_event_ledger_evaluation",
        lambda *_a, **_kw: _positive_replay(attempted_orders=0, filled_orders=0),
    )

    request = TrainingRequest(artifact_id="no_orders", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("no_orders", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert config is None
    assert n_trials == request.optuna_trials
    assert tm.LambdaRankConfig._tuning_telemetry["selection_status"] == (
        "no_economically_eligible_candidate"
    )
    assert tm.LambdaRankConfig._tuning_telemetry["economically_eligible_trials"] == 0
    evidence = tm.LambdaRankConfig._tuning_telemetry["shortlist_candidate_evidence"]
    assert len(evidence) == request.optuna_trials
    for row in evidence:
        assert row["eligible"] is False
        assert set(row["failure_reasons"]) == {
            "no_attempted_orders",
            "no_filled_orders",
        }
        assert row["trial_number"] in (0, 1, 2)


def test_tuning_rejects_non_positive_bootstrap_candidates(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
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
        lambda _panel, tuning_folds, *_a, **_kw: [None] * len(tuning_folds),
    )
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: 0.01)
    monkeypatch.setattr(
        tm,
        "_fit_and_score_candidate",
        lambda *_a, **_kw: ([0.05, 0.06, 0.07], pl.DataFrame()),
    )
    monkeypatch.setattr(
        tm,
        "_event_ledger_evaluation",
        lambda *_a, **_kw: _positive_replay(
            excess_returns=[-0.001] * 24, strategy_returns=[-0.001] * 24
        ),
    )

    request = TrainingRequest(artifact_id="bad_boot", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("bad_boot", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert config is None
    assert n_trials == request.optuna_trials
    assert tm.LambdaRankConfig._tuning_telemetry["selection_status"] == (
        "no_economically_eligible_candidate"
    )
    evidence = tm.LambdaRankConfig._tuning_telemetry["shortlist_candidate_evidence"]
    assert len(evidence) == request.optuna_trials
    for row in evidence:
        assert row["eligible"] is False
        assert row["failure_reasons"] == ["non_positive_bootstrap_lower_bound"]
        assert row["attempted_orders"] > 0
        assert row["filled_orders"] > 0
        assert row["bootstrap_lower_bound"] <= 0.0

def test_economic_candidate_evidence_reason_codes() -> None:
    from src.stocks.workflows.train_model import _evaluate_economic_candidate

    request = TrainingRequest(artifact_id="codes", n_bootstrap=2)
    finite = ReplayResult(
        attempted_orders=2,
        filled_orders=1,
        excess_returns=[0.001, 0.001],
        strategy_returns=[0.001, 0.001],
        benchmark_returns=[0.0, 0.0],
        metrics={"max_drawdown": 0.0, "turnover": 0.0},
        final_value=1.0,
        base_total_return=0.0,
        benchmark_total_return=0.0,
        stress_total_return=0.0,
    )
    assert _evaluate_economic_candidate([0.01], finite, request, 1, 0.02).eligible is True

    bad_ic = _evaluate_economic_candidate([0.0, 0.01], finite, request, 2, 0.02)
    assert bad_ic.eligible is False
    assert bad_ic.failure_reasons == ("non_positive_fold_rank_ic",)

    non_finite = _evaluate_economic_candidate(
        [0.01], dataclass_replace(finite, strategy_returns=[float("nan")]), request, 3, 0.02,
    )
    assert non_finite.eligible is False
    assert non_finite.failure_reasons == ("non_finite_replay",)

    combined = _evaluate_economic_candidate(
        [0.0], dataclass_replace(finite, attempted_orders=0), request, 4, 0.02,
    )
    assert combined.failure_reasons == (
        "non_positive_fold_rank_ic",
        "no_attempted_orders",
    )
    row = combined.to_json_safe()
    assert row["trial_number"] == 4
    assert row["failure_reasons"] == [
        "non_positive_fold_rank_ic",
        "no_attempted_orders",
    ]
    assert row["eligible"] is False


def test_tuning_skips_candidates_that_fail_full_refit(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
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
        lambda _panel, tuning_folds, *_a, **_kw: [None] * len(tuning_folds),
    )
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: 0.01)
    monkeypatch.setattr(tm, "_fit_and_score_candidate", lambda *_a, **_kw: None)
    monkeypatch.setattr(tm, "_event_ledger_evaluation", lambda *_a, **_kw: _positive_replay())

    request = TrainingRequest(artifact_id="refit_fail", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("refit_fail", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert config is None
    assert n_trials == request.optuna_trials
    assert tm.LambdaRankConfig._tuning_telemetry["selection_status"] == (
        "no_economically_eligible_candidate"
    )

def test_full_refit_early_rejects_non_positive_first_fold(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=40, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
    request = TrainingRequest(artifact_id="early_reject", n_folds=3)

    class _FakeContext:
        train_processed = pl.DataFrame({"x": [1.0] * 12})
        prepared = None

    scored = pl.DataFrame(
        {
            "session": ["2024-01-01"] * 2,
            "instrument_id": ["KRX:000001", "KRX:000002"],
            "pred_score": [0.5, 0.4],
        }
    )
    calls = {"count": 0}

    def fake_score(*_a, **_kw):
        calls["count"] += 1
        return (-0.01, scored)

    monkeypatch.setattr(tm, "_score_context_model", fake_score)
    guard = tm.TrialResourceGuard(request, predictor_count=3)
    result = tm._fit_and_score_candidate(
        pl.DataFrame(),
        [],
        [_FakeContext()] * 3,
        request,
        _tune_base_manifest("early_reject", manifest, manifest.label_definition),
        ("feature__x",),
        "residual_o2o_5d",
        "relevance",
        tm.LambdaRankConfig(),
        guard,
        "trial0",
    )
    assert result is not None
    fold_ic, oos = result
    assert fold_ic == [-0.01]
    assert oos is None
    assert calls["count"] == 1


def test_tuning_records_early_rejected_full_refits(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
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
        lambda _panel, tuning_folds, *_a, **_kw: [None] * len(tuning_folds),
    )
    monkeypatch.setattr(tm, "_score_trial_fold", lambda *_a, **_kw: 0.01)

    def fake_refit(*_a, **_kw):
        del _kw
        key = str(_a[10])
        if key == "trial0":
            return ([-0.01], None)
        return ([0.05, 0.06, 0.07], pl.DataFrame())

    monkeypatch.setattr(tm, "_fit_and_score_candidate", fake_refit)
    monkeypatch.setattr(tm, "_event_ledger_evaluation", lambda *_a, **_kw: _positive_replay())

    request = TrainingRequest(artifact_id="early_tele", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        _tune_base_manifest("early_tele", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
        dataset_manifest=manifest,
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert config is not None
    assert n_trials == request.optuna_trials
    telemetry = config._tuning_telemetry
    assert telemetry["selection_status"] == "selected"
    assert telemetry["selected_trial_number"] == 1
    assert telemetry["early_rejected_full_refits"] == 1
    assert telemetry["early_rejected_full_refit_seconds"] >= 0.0
    assert len(telemetry["shortlist_candidate_evidence"]) == 2


def test_static_context_bytes_participate_in_resource_guard(monkeypatch, tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=40, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
    request = TrainingRequest(artifact_id="cap_static", n_folds=3)
    context = tm._prepare_replay_static_context(panel, request)
    assert context.cache_bytes > 0

    admitted: list[int] = []
    original = tm.TrialResourceGuard.admit

    def spy(self, rows, *, extra_bytes=0):
        admitted.append(int(extra_bytes))
        return original(self, rows, extra_bytes=extra_bytes)

    monkeypatch.setattr(tm.TrialResourceGuard, "admit", spy)

    class _FakeContext:
        train_processed = pl.DataFrame({"x": [1.0] * 12})
        prepared = None

    scored = pl.DataFrame(
        {
            "session": ["2024-01-01"] * 2,
            "instrument_id": ["KRX:000001", "KRX:000002"],
            "pred_score": [0.5, 0.4],
        }
    )
    monkeypatch.setattr(tm, "_score_context_model", lambda *_a, **_kw: (0.03, scored))

    guard = tm.TrialResourceGuard(request, predictor_count=3)
    result = tm._fit_and_score_candidate(
        pl.DataFrame(),
        [],
        [_FakeContext()] * 3,
        request,
        _tune_base_manifest("cap_static", manifest, manifest.label_definition),
        ("feature__x",),
        "residual_o2o_5d",
        "relevance",
        tm.LambdaRankConfig(),
        guard,
        "trial0",
        static_cache_bytes=context.cache_bytes,
    )
    assert result is not None
    _, oos = result
    assert oos is not None
    assert admitted
    assert all(value >= context.cache_bytes for value in admitted)

    tight = tm.TrialResourceGuard(
        TrainingRequest(artifact_id="cap_tight", n_folds=3, max_rss_mib=1),
        predictor_count=3,
    )
    with pytest.raises(tm.TrainingCapacityError):
        tm._fit_and_score_candidate(
            pl.DataFrame(),
            [],
            [_FakeContext()] * 3,
            request,
            _tune_base_manifest("cap_tight", manifest, manifest.label_definition),
            ("feature__x",),
            "residual_o2o_5d",
            "relevance",
            tm.LambdaRankConfig(),
            tight,
            "trial0",
            static_cache_bytes=context.cache_bytes,
        )


def test_resource_breach_publishes_no_artifact(tmp_path, monkeypatch) -> None:
    import src.stocks.workflows.train_model as tm

    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)

    snapshot, _df = _snapshot(n_sessions=140, n_tickers=20)
    artifact_root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry(artifact_root)
    with pytest.raises(tm.TrainingCapacityError):
        train_model(
            snapshot,
            registry,
            TrainingRequest(
                artifact_id="breach_v1", n_folds=3, optuna_trials=2, max_rss_mib=1
            ),
        )
    assert not (artifact_root / "breach_v1").exists()


def test_bounded_replay_history_contract(tmp_path) -> None:
    from src.core.portfolio import PortfolioSnapshot
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        construct_target_allocations,
    )
    from src.stocks.workflows.train_model import (
        _bounded_replay_history,
        _instruments_from_frame,
    )

    df = stock_v2_composed_df(n_sessions=100, n_tickers=8)
    panel = _index_sessions(df)
    frame = panel.drop("session_index")
    adtv_lookup = (
        frame.sort("session")
        .with_columns(
            pl.col("trading_value").rolling_mean(20, min_samples=1).over("instrument_id").alias("adtv")
        )
        .select("instrument_id", "session", "adtv")
    )
    last_30_sessions = frame["session"].unique().sort(descending=True).head(30)
    oos_scored = frame.filter(pl.col("session").is_in(last_30_sessions)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    scored_for_replay = (
        frame.join(
            oos_scored.select("instrument_id", "session", "pred_score"),
            on=["instrument_id", "session"],
            how="left",
        ).join(adtv_lookup, on=["instrument_id", "session"], how="left")
    )
    policy = StockRiskPolicy(top_k=8, gross_cap=1.0, single_name_cap=0.2, participation_limit=0.01)
    decision_time = oos_scored["session"].max()
    bounded = _bounded_replay_history(scored_for_replay, decision_time, policy)

    assert bounded["session"].max() <= decision_time
    assert (
        bounded["session"].n_unique()
        <= max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions) + 1
    )
    first_scored = oos_scored["session"].min()
    pre_os = bounded.filter(pl.col("session") < first_scored)
    assert pre_os["pred_score"].null_count() == pre_os.height
    decision_rows = bounded.filter(pl.col("session") == decision_time)
    assert decision_rows["pred_score"].is_not_null().all()

    instruments = _instruments_from_frame(frame)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="promotion",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    full = scored_for_replay.filter(pl.col("session") <= decision_time)
    assert construct_target_allocations(bounded, instruments, portfolio, policy) == (
        construct_target_allocations(full, instruments, portfolio, policy)
    )

    with pytest.raises(ValueError, match="session and pred_score"):
        _bounded_replay_history(frame, decision_time, policy)
    empty = scored_for_replay.filter(pl.col("session") < first_scored)
    with pytest.raises(ValueError, match="no scored cross-section"):
        _bounded_replay_history(empty, decision_time, policy)
