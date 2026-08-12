"""Model-training workflow: temporal isolation, artifact publication, NO_TRADE gates."""
from __future__ import annotations

import json
from dataclasses import replace as dataclass_replace

import polars as pl
import pytest

import src.stocks.workflows.train_model as tm

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.time import TemporalViolationError
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.research.economic_alpha import ALPHA_COLUMN
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
        lambda *_args, **_kwargs: (
            tm.LambdaRankConfig(n_estimators=20, early_stopping_rounds=5),
            3,
            tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),
        ),
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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("tune_oos", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("tune_prune", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("tie_break", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("no_orders", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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


def test_calibrated_replay_records_economic_evidence_and_fails_closed(
    tmp_path,
) -> None:
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=70, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = _index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="calib_replay",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
    )
    ledger = tm._build_calibration_ledger(oos_scored, panel, "residual_o2o_5d")
    assert {"session", "instrument_id", "score", "residual_o2o_5d", "label_available_time"} <= set(
        ledger.columns
    )
    replay = tm._event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
        calibration_ledger=ledger,
    )
    assert replay.calibration_evidence
    assert "history_sessions" in replay.calibration_evidence
    assert "eligible_bucket_count" in replay.calibration_evidence
    assert "calibration_state" in replay.calibration_evidence

    evidence = tm._evaluate_economic_candidate(
        [0.05, 0.06, 0.07], replay, request, 1, 0.05
    )
    assert evidence.calibration_history_sessions >= 0
    assert evidence.eligible_bucket_count >= 0
    assert evidence.cash_cycles >= 0
    row = evidence.to_json_safe()
    assert "calibration_history_sessions" in row
    assert "eligible_bucket_count" in row


def test_calibration_ledger_is_empty_for_empty_oos(tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=40, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    panel = _index_sessions(df)
    empty = pl.DataFrame(
        {
            "session": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "instrument_id": pl.Series([], dtype=pl.Utf8),
            "pred_score": pl.Series([], dtype=pl.Float64),
        }
    )
    ledger = tm._build_calibration_ledger(empty, panel, "residual_o2o_5d")
    assert ledger.is_empty()

    cal = tm.CausalAlphaCalibrator(
        bucket_count=4, min_calibration_sessions=5, seed=42
    )
    del tmp_path, manifest
    decision = df["session"].max()
    scored = df.filter(pl.col("session") == decision).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    out = cal.transform(scored, ledger, decision, default_base_schedule())
    assert out[ALPHA_COLUMN].null_count() == out.height


def test_calibrated_replay_and_plain_replay_share_no_leaked_targets(tmp_path) -> None:
    """Changing a future label never changes an earlier calibrated decision."""
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=70, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = _index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="calib_no_leak",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
    )
    base_ledger = tm._build_calibration_ledger(oos_scored, panel, "residual_o2o_5d")

    future_threshold = panel.select(pl.col("session").max()).to_series()[0]
    flipped = panel.with_columns(
        pl.when(pl.col("label_available_time") > future_threshold)
        .then(pl.lit(0.99))
        .otherwise(pl.col("residual_o2o_5d"))
        .alias("residual_o2o_5d")
    )
    flipped_ledger = tm._build_calibration_ledger(oos_scored, flipped, "residual_o2o_5d")

    a = tm._event_ledger_evaluation(
        panel, oos_scored, request, snapshot.manifest, registry,
        default_base_schedule(), default_stress_schedule(), calibration_ledger=base_ledger,
    )
    b = tm._event_ledger_evaluation(
        panel, oos_scored, request, snapshot.manifest, registry,
        default_base_schedule(), default_stress_schedule(), calibration_ledger=flipped_ledger,
    )
    assert a.ledger == b.ledger
    assert a.trades == b.trades


def test_inner_selection_replay_is_base_only_and_matches_final_base_evidence(
    tmp_path,
) -> None:
    """INNER_SELECTION_BASE_ONLY skips the stress ledger without changing base."""
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=70, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = _index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="base_only",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
    )
    ledger = tm._build_calibration_ledger(oos_scored, panel, "residual_o2o_5d")
    guard = tm.ReplayResourceGuard(
        request, replay_mode=tm.ReplayMode.INNER_SELECTION_BASE_ONLY
    )
    context = tm._prepare_replay_static_context(panel, request, guard=guard)
    base_only = tm._event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
        replay_context=context,
        calibration_ledger=ledger,
        replay_mode=tm.ReplayMode.INNER_SELECTION_BASE_ONLY,
        replay_guard=guard,
    )
    assert base_only.stress_total_return is None
    assert base_only.stress_metrics is None
    assert base_only.replay_mode == "INNER_SELECTION_BASE_ONLY"

    final_guard = tm.ReplayResourceGuard(
        request, replay_mode=tm.ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS
    )
    final_context = tm._prepare_replay_static_context(panel, request, guard=final_guard)
    final = tm._event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
        replay_context=final_context,
        calibration_ledger=ledger,
        replay_mode=tm.ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
        replay_guard=final_guard,
    )
    assert final.stress_total_return is not None
    assert base_only.base_total_return == final.base_total_return
    assert base_only.metrics == final.metrics
    assert base_only.ledger == final.ledger
    assert base_only.trades == final.trades
    assert base_only.attempted_orders == final.attempted_orders
    assert base_only.filled_orders == final.filled_orders
    assert base_only.planned_cycles == final.planned_cycles
    assert base_only.calibration_evidence == final.calibration_evidence


def test_compact_replay_market_index_excludes_feature_and_label_columns() -> None:
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=40, n_tickers=6)
    df = df.with_columns(
        pl.col("residual_o2o_5d").alias("residual_o2o_10d"),
        pl.col("relevance").alias("relevance_10d"),
        (pl.col("label_available_time") + pl.duration(days=5)).alias(
            "label_available_time_10d"
        ),
    )
    panel = _index_sessions(df)
    context = tm._prepare_replay_static_context(
        panel, TrainingRequest(artifact_id="compact_scope", n_folds=3)
    )
    columns = set(context.market_index.columns)
    assert {"instrument_id", "session", "available_time", "open", "close"} <= columns
    assert not any(
        c.startswith("feature__") and c != "feature__volatility_20d"
        for c in columns
    )
    assert not any(c.startswith("residual_o2o_") for c in columns)
    assert not any(c.startswith("relevance") for c in columns)
    assert not any(c.startswith("label_available_time") for c in columns)


def test_replay_capacity_failure_raises_and_records_telemetry(tmp_path) -> None:
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=40, n_tickers=6)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = _index_sessions(df)
    oos_scored = df.tail(12).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(artifact_id="cap_replay", n_folds=3, max_rss_mib=1)
    guard = tm.ReplayResourceGuard(request)
    context = tm._prepare_replay_static_context(panel, request)
    with pytest.raises(tm.TrainingCapacityError, match="replay_capacity_exceeded"):
        tm._event_ledger_evaluation(
            panel,
            oos_scored,
            request,
            snapshot.manifest,
            registry,
            default_base_schedule(),
            default_stress_schedule(),
            replay_context=context,
            replay_guard=guard,
        )
    assert guard.capacity_failure_reason == "replay_capacity_exceeded"
    telemetry = guard.telemetry()
    assert telemetry["capacity_failure_reason"] == "replay_capacity_exceeded"
    assert telemetry["replay_stage_estimated_bytes"]
    assert telemetry["replay_mode"] == "FINAL_PROMOTION_BASE_AND_STRESS"
    assert telemetry["bootstrap_batch_size"] == 0
    assert telemetry["bootstrap_workspace_bytes"] == 0

def test_bootstrap_workspace_cap_selects_bounded_batch_and_records_telemetry(
    monkeypatch,
) -> None:
    """Guard picks a positive bounded batch and exposes it in replay telemetry."""
    import src.stocks.workflows.train_model as tm

    request = TrainingRequest(artifact_id="cap_probe", n_folds=3, max_rss_mib=8000)
    guard = tm.ReplayResourceGuard(request)
    monkeypatch.setattr(tm.ReplayResourceGuard, "_rss_mib", lambda self: 1000.0)
    monkeypatch.setattr(
        tm.ReplayResourceGuard, "_resolve_limit_mib", lambda cls, req: 8000.0
    )

    history_rows = 4_000_000
    per_draw = history_rows * 24
    cap = guard.bootstrap_workspace_cap(
        history_rows=history_rows,
        projected_output_bytes=100_000_000,
        n_bootstrap=200,
    )
    assert cap > 0
    assert cap <= 200 * per_draw
    assert cap % per_draw == 0
    assert guard.bootstrap_batch_size == cap // per_draw
    assert guard.bootstrap_workspace_bytes == cap
    admitted = guard.stage_estimated_bytes["decision_preparation"]
    assert admitted == cap + 100_000_000 + per_draw
    assert admitted <= 8000 * 1024 * 1024
    telemetry = guard.telemetry()
    assert telemetry["bootstrap_batch_size"] == cap // per_draw
    assert telemetry["bootstrap_workspace_bytes"] == cap


def test_bootstrap_workspace_cap_fails_closed_when_one_draw_cannot_fit(
    monkeypatch,
) -> None:
    """One-draw-infeasible memory stays a deterministic decision-preparation failure."""
    import src.stocks.workflows.train_model as tm

    request = TrainingRequest(artifact_id="cap_fail", n_folds=3, max_rss_mib=8000)
    guard = tm.ReplayResourceGuard(request)
    monkeypatch.setattr(tm.ReplayResourceGuard, "_rss_mib", lambda self: 7990.0)
    monkeypatch.setattr(
        tm.ReplayResourceGuard, "_resolve_limit_mib", lambda cls, req: 8000.0
    )

    with pytest.raises(
        tm.TrainingCapacityError, match="replay_capacity_exceeded:decision_preparation"
    ):
        guard.bootstrap_workspace_cap(
            history_rows=4_000_000,
            projected_output_bytes=20_000_000,
            n_bootstrap=200,
        )
    assert guard.capacity_failure_reason == "replay_capacity_exceeded"
    assert guard.telemetry()["capacity_failure_reason"] == "replay_capacity_exceeded"


def test_prepared_replay_guarded_passes_cap_to_calibrator(
    monkeypatch, tmp_path
) -> None:
    """A guarded replay passes the bounded workspace cap into prepare_decision."""
    import src.stocks.workflows.train_model as tm
    from src.stocks.research.economic_alpha import CausalAlphaCalibrator

    df = stock_v2_composed_df(n_sessions=70, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = _index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="guarded_prepared",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
        max_rss_mib=8000,
    )
    ledger = tm._build_calibration_ledger(oos_scored, panel, "residual_o2o_5d")
    guard = tm.ReplayResourceGuard(request)
    monkeypatch.setattr(tm.ReplayResourceGuard, "_rss_mib", lambda self: 1000.0)

    captured: dict[str, object] = {}
    original_prepare = CausalAlphaCalibrator.prepare_decision

    def spy(
        self,
        observations,
        decision_time,
        cost_schedule,
        *,
        max_bootstrap_workspace_bytes=None,
    ):
        captured["max_bootstrap_workspace_bytes"] = max_bootstrap_workspace_bytes
        return original_prepare(
            self,
            observations,
            decision_time,
            cost_schedule,
            max_bootstrap_workspace_bytes=max_bootstrap_workspace_bytes,
        )

    monkeypatch.setattr(CausalAlphaCalibrator, "prepare_decision", spy)
    replay = tm._event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
        calibration_ledger=ledger,
        replay_mode=tm.ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
        replay_guard=guard,
    )
    assert replay.prepared_decision_count > 0
    cap = captured["max_bootstrap_workspace_bytes"]
    assert isinstance(cap, int)
    assert cap > 0
    per_draw = ledger.height * 24
    assert cap % per_draw == 0
    telemetry = guard.telemetry()
    assert telemetry["bootstrap_workspace_bytes"] == cap
    assert telemetry["bootstrap_batch_size"] == cap // per_draw
    assert telemetry["replay_peak_rss_mib"] <= 8000.0


def test_prepared_replay_reuses_calibration_state_and_counts_decisions(tmp_path) -> None:
    """Prepared final replay prepares one calibration per decision timestamp."""
    import src.stocks.workflows.train_model as tm

    df = stock_v2_composed_df(n_sessions=70, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = _index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="prepared_decision",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
    )
    ledger = tm._build_calibration_ledger(oos_scored, panel, "residual_o2o_5d")
    replay = tm._event_ledger_evaluation(
        panel,
        oos_scored,
        request,
        snapshot.manifest,
        registry,
        default_base_schedule(),
        default_stress_schedule(),
        calibration_ledger=ledger,
        replay_mode=tm.ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
    )
    cadence = 5
    decision_count = sum(
        1
        for i in range(len(replay.ledger))
        if i % cadence == 0 and i + 1 < len(replay.ledger)
    )
    assert replay.prepared_decision_count == decision_count
    assert replay.replay_mode == "FINAL_PROMOTION_BASE_AND_STRESS"


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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("bad_boot", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("refit_fail", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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
    config, n_trials, _route = tm._tune_champion(
        panel[folds[0].train_mask],
        request,
        _tune_base_manifest("early_tele", manifest, manifest.label_definition),
        tuple(c for c in df.columns if c.startswith("feature__")),
        (tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),),
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

def test_drop_target_columns_isolates_all_route_label_columns() -> None:
    from src.stocks.workflows.train_model import _drop_target_columns

    df = stock_v2_composed_df(n_sessions=40, n_tickers=6)
    frame = df.with_columns(
        pl.col("residual_o2o_5d").alias("residual_o2o_10d"),
        pl.col("residual_o2o_5d").alias("residual_o2o_15d"),
        pl.col("relevance").alias("relevance_10d"),
        pl.lit("x", dtype=pl.Utf8).alias("not_a_label"),
    )
    dropped = _drop_target_columns(frame, "residual_o2o_5d")
    assert "residual_o2o_5d" not in dropped.columns
    assert "residual_o2o_10d" not in dropped.columns
    assert "residual_o2o_15d" not in dropped.columns
    assert "relevance" not in dropped.columns
    assert "relevance_10d" not in dropped.columns
    assert "label_available_time" not in dropped.columns
    assert "not_a_label" in dropped.columns

def test_resolve_route_specs_fails_closed_for_multi_horizon_missing_columns() -> None:
    from src.stocks.workflows.train_model import _resolve_route_specs

    df = stock_v2_composed_df(n_sessions=40, n_tickers=6)
    multi = df.with_columns(
        pl.col("residual_o2o_5d").alias("residual_o2o_10d"),
        pl.col("relevance").alias("relevance_10d"),
        (pl.col("label_available_time") + pl.duration(days=5)).alias(
            "label_available_time_10d"
        ),
    )
    with pytest.raises(ValueError, match="fail closed"):
        _resolve_route_specs(multi, (5, 10, 15))

    routes = _resolve_route_specs(multi, (5, 10))
    assert [route.horizon for route in routes] == [5, 10]
    assert routes[1].label_column == "residual_o2o_10d"
    assert routes[1].label_available_column == "label_available_time_10d"


def test_resolve_route_specs_keeps_legacy_five_day_columns() -> None:
    from src.stocks.workflows.train_model import _resolve_route_specs

    df = stock_v2_composed_df(n_sessions=40, n_tickers=6)
    routes = _resolve_route_specs(df, (5, 10, 15))
    assert [route.horizon for route in routes] == [5]
    assert routes[0].relevance_column == "relevance"
    assert routes[0].label_available_column == "label_available_time"

    assert _resolve_route_specs(df.drop(["relevance"]), (5,)) == ()


def test_resolve_route_specs_activates_true_5_10_15_routes() -> None:
    from src.stocks.research.labels import residual_open_to_open_label
    from src.stocks.workflows.train_model import _resolve_route_specs

    base = stock_v2_composed_df(n_sessions=60, n_tickers=24)
    label_frames = []
    for h in (5, 10, 15):
        labels = residual_open_to_open_label(
            base.select(["instrument_id", "session", "open"]), horizon_sessions=h
        ).rename(
            {
                "residual_o2o_5d": f"residual_o2o_{h}d",
                "relevance": f"relevance_{h}d",
                "label_available_time": f"label_available_time_{h}d",
            }
        )
        label_frames.append(labels)
    multi = label_frames[0]
    for frame in label_frames[1:]:
        multi = multi.join(frame, on=["instrument_id", "session"], how="inner")
    multi = multi.filter(
        pl.col("residual_o2o_5d").is_not_null()
        & pl.col("residual_o2o_10d").is_not_null()
        & pl.col("residual_o2o_15d").is_not_null()
    )
    assert multi["residual_o2o_10d"].null_count() == 0
    assert multi["residual_o2o_15d"].null_count() == 0
    assert not (multi["residual_o2o_10d"] == multi["residual_o2o_5d"]).all()
    assert not (multi["residual_o2o_15d"] == multi["residual_o2o_5d"]).all()

    routes = _resolve_route_specs(multi, (5, 10, 15))
    assert [route.horizon for route in routes] == [5, 10, 15]
    for route, horizon in zip(routes, (5, 10, 15), strict=True):
        assert route.label_column == f"residual_o2o_{horizon}d"
        assert route.relevance_column == f"relevance_{horizon}d"
        assert route.label_available_column == f"label_available_time_{horizon}d"
        assert route.label_span_sessions == horizon + 1


def test_prepare_replay_static_context_uses_route_rebalance_cadence() -> None:
    from src.stocks.workflows.train_model import _prepare_replay_static_context

    df = stock_v2_composed_df(n_sessions=40, n_tickers=6)
    panel = _index_sessions(df)
    request = TrainingRequest(artifact_id="cadence_v1")
    for horizon in (5, 10, 15):
        context = _prepare_replay_static_context(
            panel, request, holding_horizon_sessions=horizon
        )
        assert context.policy.rebalance_frequency_sessions == horizon


def _route_test_panel() -> pl.DataFrame:
    df = stock_v2_composed_df(n_sessions=140, n_tickers=20)
    df = df.with_columns(
        pl.col("residual_o2o_5d").alias("residual_o2o_10d"),
        pl.col("relevance").alias("relevance_10d"),
        (pl.col("label_available_time") + pl.duration(days=5)).alias(
            "label_available_time_10d"
        ),
    )
    return _index_sessions(df)


def _multi_route_request(artifact_id: str, trials: int = 4) -> TrainingRequest:
    return TrainingRequest(
        artifact_id=artifact_id,
        n_folds=3,
        optuna_trials=trials,
        candidate_horizons=(5, 10),
    )


def _route_tune_mocks(
    monkeypatch,
    *,
    ledger_replay: dict[int, ReplayResult] | None = None,
    screen_ic: float | None = 0.01,
) -> None:
    monkeypatch.setattr(tm, "_MIN_TRAIN_SESSIONS", 40)
    monkeypatch.setattr(tm, "_VALIDATION_BLOCK_SESSIONS", 30)
    monkeypatch.setattr(tm, "_fit_stable_contexts", lambda *_a, **_kw: [None] * 3)
    monkeypatch.setattr(
        tm, "_score_trial_fold", lambda *_a, **_kw: screen_ic
    )
    monkeypatch.setattr(
        tm,
        "_fit_and_score_candidate",
        lambda *_a, **_kw: ([0.05, 0.06, 0.07], pl.DataFrame()),
    )

    def fake_event_ledger(*_a, **_kw):
        horizon = int(_kw.get("holding_horizon_sessions", 5))
        replay = (ledger_replay or {}).get(horizon)
        if replay is not None:
            return replay
        return _positive_replay()

    monkeypatch.setattr(tm, "_event_ledger_evaluation", fake_event_ledger)


def test_tuning_selects_longer_horizon_route_when_bootstrap_is_higher(
    monkeypatch, tmp_path,
) -> None:
    _route_tune_mocks(
        monkeypatch,
        ledger_replay={
            5: _positive_replay(excess_returns=[0.001] * 24),
            10: _positive_replay(excess_returns=[0.002] * 24),
        },
    )
    request = _multi_route_request("route_win_10d")
    panel = _route_test_panel()
    config, n_trials, route = tm._tune_champion(
        panel,
        request,
        _tune_base_manifest("route_win_10d", stock_v2_manifest(columns=panel.columns), "residual_o2o_5d"),
        tuple(c for c in panel.columns if c.startswith("feature__")),
        (
            tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),
            tm.RouteSpec(10, "residual_o2o_10d", "relevance_10d", "label_available_time_10d"),
        ),
        dataset_manifest=stock_v2_manifest(columns=panel.columns),
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert route is not None
    assert route.horizon == 10
    assert n_trials == 2
    assert config is not None
    assert config._tuning_telemetry["selected_horizon"] == 10
    assert config._tuning_telemetry["selection_status"] == "selected"


def test_tuning_tie_breaks_to_shorter_horizon_route(monkeypatch, tmp_path) -> None:
    _route_tune_mocks(monkeypatch)
    request = _multi_route_request("route_tie_5d")
    panel = _route_test_panel()
    config, _n_trials, route = tm._tune_champion(
        panel,
        request,
        _tune_base_manifest("route_tie_5d", stock_v2_manifest(columns=panel.columns), "residual_o2o_5d"),
        tuple(c for c in panel.columns if c.startswith("feature__")),
        (
            tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),
            tm.RouteSpec(10, "residual_o2o_10d", "relevance_10d", "label_available_time_10d"),
        ),
        dataset_manifest=stock_v2_manifest(columns=panel.columns),
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert route is not None
    assert route.horizon == 5
    assert config is not None
    assert config._tuning_telemetry["selected_horizon"] == 5


def test_tuning_records_route_specific_candidate_evidence(monkeypatch, tmp_path) -> None:
    _route_tune_mocks(monkeypatch)
    request = _multi_route_request("route_evidence")
    panel = _route_test_panel()
    config, _n_trials, route = tm._tune_champion(
        panel,
        request,
        _tune_base_manifest("route_evidence", stock_v2_manifest(columns=panel.columns), "residual_o2o_5d"),
        tuple(c for c in panel.columns if c.startswith("feature__")),
        (
            tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),
            tm.RouteSpec(10, "residual_o2o_10d", "relevance_10d", "label_available_time_10d"),
        ),
        dataset_manifest=stock_v2_manifest(columns=panel.columns),
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert route is not None
    telemetry = config._tuning_telemetry
    evidence = telemetry["shortlist_candidate_evidence"]
    assert len(evidence) == 4
    assert {row["holding_horizon_sessions"] for row in evidence} == {5, 10}
    assert any(row["label_column"] == "residual_o2o_5d" for row in evidence)
    assert any(row["label_column"] == "residual_o2o_10d" for row in evidence)
    assert any(row["label_available_column"] == "label_available_time_10d" for row in evidence)
    assert telemetry["economically_eligible_trials"] == 4
    assert set(telemetry["routes"]) == {"5", "10"}
    assert telemetry["per_route_trial_budget"] == 2


def test_tuning_returns_no_trade_when_no_route_survives(monkeypatch, tmp_path) -> None:
    _route_tune_mocks(monkeypatch, screen_ic=None)
    request = _multi_route_request("route_no_trade")
    panel = _route_test_panel()
    config, n_trials, route = tm._tune_champion(
        panel,
        request,
        _tune_base_manifest("route_no_trade", stock_v2_manifest(columns=panel.columns), "residual_o2o_5d"),
        tuple(c for c in panel.columns if c.startswith("feature__")),
        (
            tm.RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time"),
            tm.RouteSpec(10, "residual_o2o_10d", "relevance_10d", "label_available_time_10d"),
        ),
        dataset_manifest=stock_v2_manifest(columns=panel.columns),
        registry=ModelArtifactRegistry(tmp_path / "artifacts"),
        base_schedule=default_base_schedule(),
        stress_schedule=default_stress_schedule(),
    )
    assert config is None
    assert route is None
    assert n_trials == 4
    assert tm.LambdaRankConfig._tuning_telemetry["selection_status"] == "no_complete_screen_candidate"


def test_event_ledger_cadence_matches_holding_horizon(tmp_path) -> None:
    from src.stocks.workflows.train_model import _event_ledger_evaluation

    df = stock_v2_composed_df(n_sessions=60, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    oos_scored = df.tail(20).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    panel = _index_sessions(df)
    request = TrainingRequest(artifact_id="cadence_ledger")
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    five_day = _event_ledger_evaluation(
        panel, oos_scored, request, snapshot.manifest, registry,
        default_base_schedule(), default_stress_schedule(),
        holding_horizon_sessions=5,
    )
    ten_day = _event_ledger_evaluation(
        panel, oos_scored, request, snapshot.manifest, registry,
        default_base_schedule(), default_stress_schedule(),
        holding_horizon_sessions=10,
    )
    assert five_day.planned_cycles > ten_day.planned_cycles
    assert ten_day.planned_cycles > 0


def test_reserve_forward_holdout_uses_route_availability_column() -> None:
    from src.stocks.workflows.train_model import _reserve_forward_holdout

    df = stock_v2_composed_df(n_sessions=100, n_tickers=6)
    df = df.with_columns(
        pl.col("label_available_time").alias("label_available_time_5d"),
        pl.col("label_available_time").alias("label_available_time_10d"),
        pl.col("label_available_time").alias("label_available_time_15d"),
    )
    panel = _index_sessions(df)
    request = TrainingRequest(artifact_id="holdout_route")
    fold, training_panel = _reserve_forward_holdout(
        panel, request, 6, "label_available_time_5d"
    )
    assert fold is None
    assert training_panel is panel

    legacy_fold, legacy_panel = _reserve_forward_holdout(
        panel, request, 6, "label_available_time"
    )
    assert legacy_fold is None
    assert legacy_panel is panel
