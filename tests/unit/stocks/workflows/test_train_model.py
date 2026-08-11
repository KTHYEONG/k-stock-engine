"""Model-training workflow: temporal isolation, artifact publication, NO_TRADE gates."""
from __future__ import annotations

import json

import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.time import TemporalViolationError
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import METRICS_FILENAME, ModelArtifactRegistry
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
    from src.stocks.workflows.train_model import _event_ledger_evaluation

    df = stock_v2_composed_df(n_sessions=40, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    oos_scored = df.tail(12).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    replay = _event_ledger_evaluation(
        _index_sessions(df),
        oos_scored,
        TrainingRequest(artifact_id="candidate", n_folds=3),
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


def test_tuning_never_includes_first_outer_oos(monkeypatch) -> None:
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

    request = TrainingRequest(artifact_id="tune_oos", n_folds=3, optuna_trials=3)
    config, n_trials = tm._tune_champion(
        panel,
        folds,
        request,
        tm.ModelManifest(
            artifact_id="tune_oos",
            asset_kind=__import__("src.core.instruments", fromlist=["AssetKind"]).AssetKind.STOCK,
            feature_set="stock_alpha_v2",
            feature_schema_hash="hash",
            universe_policy_hash="universe",
            label_definition=manifest.label_definition,
            label_horizon_sessions=manifest.label_horizon_sessions,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-12-31T00:00:00+00:00",
            model_type="lambdarank_blend",
        ),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
    )
    assert captured["tuning_max_session"] < folds[0].validation_decision_start
    assert n_trials == request.optuna_trials
    assert config is not None


def test_tuning_counts_pruned_trials_for_deflated_sharpe(monkeypatch) -> None:
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
        tm.ModelManifest(
            artifact_id="tune_prune",
            asset_kind=__import__("src.core.instruments", fromlist=["AssetKind"]).AssetKind.STOCK,
            feature_set="stock_alpha_v2",
            feature_schema_hash="hash",
            universe_policy_hash="universe",
            label_definition=manifest.label_definition,
            label_horizon_sessions=manifest.label_horizon_sessions,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-12-31T00:00:00+00:00",
            model_type="lambdarank_blend",
        ),
        tuple(c for c in df.columns if c.startswith("feature__")),
        "residual_o2o_5d",
        "relevance",
        label_span,
    )
    assert n_trials == request.optuna_trials
    assert config is None


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
