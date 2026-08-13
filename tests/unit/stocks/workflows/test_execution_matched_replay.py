"""Execution-matched proxy kernel: parity with the reference replay and gate behavior."""
from __future__ import annotations

import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.execution_matched_replay import (
    ExecutionMatchedReplayKernel,
    FINAL_PROMOTION_BASE_AND_STRESS,
    INNER_SELECTION_BASE_ONLY,
)
import src.stocks.workflows.train_model as tm
from src.stocks.workflows.train_model import PreparedSelectionRoute, RouteSpec
from tests.fixtures.stocks.helpers import stock_v2_composed_df, stock_v2_manifest


def _route_materialization(tmp_path) -> tuple[dict, dict, object]:

    df = stock_v2_composed_df(n_sessions=70, n_tickers=8)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = tm._index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="kernel_test",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
    )
    base = default_base_schedule()
    stress = default_stress_schedule()
    context = tm._prepare_replay_static_context(panel, request)
    reference = tm._event_ledger_evaluation(
        panel, oos_scored, request, snapshot.manifest, registry, base, stress,
        replay_context=context,
    )
    oos_start = oos_scored["session"].min()
    route = RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time")
    oos_sessions = tuple(
        tm._session_as_datetime(session)
        for session in oos_scored["session"].unique().sort().to_list()
    )
    prepared_route = PreparedSelectionRoute.build(
        panel, oos_sessions, request, route
    )
    prepared = tm._event_ledger_evaluation(
        panel, oos_scored, request, snapshot.manifest, registry, base, stress,
        replay_context=context, prepared_route=prepared_route,
    )
    kernel = ExecutionMatchedReplayKernel(
        panel=panel,
        prepared_route=prepared_route,
        instruments=context.instruments,
        policy=context.policy,
        request=request,
        dataset_manifest=snapshot.manifest,
        registry=registry,
        base_schedule=base,
        stress_schedule=stress,
        holding_horizon_sessions=5,
        label_column="residual_o2o_5d",
        label_available_column="label_available_time",
    )
    return {
        "panel": panel,
        "oos_scored": oos_scored,
        "request": request,
        "snapshot": snapshot,
        "registry": registry,
        "base": base,
        "stress": stress,
        "context": context,
        "prepared_route": prepared_route,
        "kernel": kernel,
        "route": route,
    }, {
        "reference": reference,
        "prepared": prepared,
    }, None


def test_execution_matched_kernel_base_replay_matches_reference(tmp_path) -> None:
    """Proxy kernel and exact base replay share one canonical ledger.

    The kernel ``run_base`` must reproduce the reference base replay exactly:
    identical ledger, metrics, fills, decision count, and no-trade reasons.
    """
    state, expected, _ = _route_materialization(tmp_path)
    reference = expected["reference"]
    kernel = state["kernel"]
    oos_scored = state["oos_scored"]
    panel = state["panel"]
    oos_start = oos_scored["session"].min()
    assert oos_start > panel["session"].min()
    assert kernel.prepared_route.sessions[0] == tm._session_as_datetime(oos_start)
    evidence = kernel.run_base(
        oos_scored,
        None,
        replay_mode=INNER_SELECTION_BASE_ONLY,
    )
    assert evidence.ledger == reference.ledger
    assert evidence.metrics == reference.metrics
    assert evidence.filled_orders == reference.filled_orders
    assert evidence.prepared_decision_count == reference.prepared_decision_count
    assert evidence.no_trade_reason_counts == reference.no_trade_reason_counts
    assert evidence.kernel_parity_version == "execution-matched-v1"
    assert evidence.excess_returns == reference.excess_returns
    # The bounded route replays exactly its own decision schedule; the tail is
    # never part of the proxy interval.
    route = state["route"]
    expected_decision_count = sum(
        1 for index in state["prepared_route"].decision_indices
        if index + 1 < len(state["prepared_route"].sessions)
    )
    assert evidence.prepared_decision_count == expected_decision_count


def test_execution_matched_proxy_rejects_equal_weight_sign_flip(tmp_path) -> None:
    """A positive equal-weight proxy return with negative costed replay rejects.

    The fixture manufactures a candidate whose naive top-k equal-weight label
    mean is positive but whose constrained fill-cost base replay lower bound is
    non-positive, proving the execution-matched proxy rejects before promotion.
    """
    from src.stocks.research.models import ModelManifest

    df = stock_v2_composed_df(n_sessions=90, n_tickers=20)
    manifest = stock_v2_manifest(columns=df.columns)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    panel = tm._index_sessions(df)
    last_30 = df["session"].unique().sort(descending=True).head(30)
    oos_scored = df.filter(pl.col("session").is_in(last_30)).with_columns(
        pl.col("market_cap").rank("dense").over("session").cast(pl.Float64).alias("pred_score")
    )
    request = TrainingRequest(
        artifact_id="sign_flip",
        n_folds=3,
        calibration_bucket_count=4,
        min_calibration_sessions=5,
    )
    base = default_base_schedule()
    stress = default_stress_schedule()
    context = tm._prepare_replay_static_context(panel, request)
    oos_start = oos_scored["session"].min()
    route = RouteSpec(5, "residual_o2o_5d", "relevance", "label_available_time")
    oos_sessions = tuple(
        tm._session_as_datetime(session)
        for session in oos_scored["session"].unique().sort().to_list()
    )
    prepared_route = PreparedSelectionRoute.build(
        panel, oos_sessions, request, route
    )
    kernel = ExecutionMatchedReplayKernel(
        panel=panel,
        prepared_route=prepared_route,
        instruments=context.instruments,
        policy=context.policy,
        request=request,
        dataset_manifest=manifest,
        registry=registry,
        base_schedule=base,
        stress_schedule=stress,
        holding_horizon_sessions=5,
        label_column="residual_o2o_5d",
        label_available_column="label_available_time",
    )
    base_manifest = ModelManifest(
        artifact_id=request.artifact_id,
        asset_kind=manifest.asset_kind,
        feature_set="stock_alpha_v2",
        feature_schema_hash="hash",
        universe_policy_hash="universe",
        label_definition="residual_o2o_5d",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="lambdarank_blend",
    )

    label = panel.select(
        "session_index", "session", "instrument_id", "residual_o2o_5d"
    ).filter(pl.col("session_index") >= 40)
    naive = tm._economic_screen_score(
        label,
        oos_scored,
        label_column="residual_o2o_5d",
        top_k=request.top_k,
        holding_horizon_sessions=5,
        cost_schedule=stress,
        n_bootstrap=request.n_bootstrap,
        bootstrap_alpha=request.bootstrap_alpha,
        seed=request.seed,
    )
    assert naive > 0.0

    evidence = kernel.run_base(
        oos_scored,
        None,
        replay_mode=INNER_SELECTION_BASE_ONLY,
    )
    bound = tm._execution_matched_lower_bound(
        evidence,
        request,
        base_manifest,
        0,
    )
    assert bound <= 0.0


def test_execution_matched_kernel_final_mode_includes_stress(tmp_path) -> None:
    """FINAL_PROMOTION_BASE_AND_STRESS exposes paired stress evidence."""
    state, _, _ = _route_materialization(tmp_path)
    kernel = state["kernel"]
    oos_scored = state["oos_scored"]
    ledger = tm._build_calibration_ledger(oos_scored, state["panel"], "residual_o2o_5d")
    reference = tm._event_ledger_evaluation(
        state["panel"],
        oos_scored,
        state["request"],
        state["snapshot"].manifest,
        state["registry"],
        state["base"],
        state["stress"],
        replay_context=state["context"],
        calibration_ledger=ledger,
        prepared_route=state["prepared_route"],
        replay_mode=tm.ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
    )
    evidence = kernel.run_base(
        oos_scored,
        ledger,
        replay_mode=FINAL_PROMOTION_BASE_AND_STRESS,
    )
    assert evidence.stress_total_return == reference.stress_total_return
    assert evidence.stress_metrics == reference.stress_metrics
    assert evidence.ledger == reference.ledger
    assert evidence.trades == reference.trades


def test_execution_matched_kernel_evidence_is_json_safe(tmp_path) -> None:
    state, _, _ = _route_materialization(tmp_path)
    kernel = state["kernel"]
    oos_scored = state["oos_scored"]
    evidence = kernel.run_base(
        oos_scored,
        None,
        replay_mode=INNER_SELECTION_BASE_ONLY,
    )
    payload = evidence.to_json_safe()
    import json

    json.dumps(payload)
    assert "ledger" not in payload
    assert payload["kernel_parity_version"] == "execution-matched-v1"


def test_inner_selection_kernel_is_base_only_and_stateful(tmp_path) -> None:
    """Base-only proxy folds retain stateful allocation/fill semantics.

    Every proxy fold kernel replays only the base ledger (never stress or
    forward-holdout evidence) and state carry-over is material: a candidate
    whose score overlay favors different names changes the executed ledger,
    never silently resetting the portfolio between decisions.
    """
    state, _, _ = _route_materialization(tmp_path)
    kernel = state["kernel"]
    oos_scored = state["oos_scored"]
    evidence = kernel.run_base(
        oos_scored,
        None,
        replay_mode=INNER_SELECTION_BASE_ONLY,
    )
    assert evidence.stress_metrics is None
    assert evidence.stress_total_return is None
    assert evidence.replay_mode == INNER_SELECTION_BASE_ONLY
    assert evidence.prepared_decision_count >= 1
    assert evidence.filled_orders > 0

    # A score overlay that favors different names must change the executed
    # ledger; a portfolio reset between decisions would not be equivalent.
    low = oos_scored.with_columns(
        pl.when(
            pl.col("instrument_id").is_in(
                [f"KRX:0{t + 1:05d}" for t in range(5)]
            )
        )
        .then(pl.lit(1000.0))
        .otherwise(pl.lit(-1000.0))
        .alias("pred_score")
    )
    low_evidence = kernel.run_base(
        low,
        None,
        replay_mode=INNER_SELECTION_BASE_ONLY,
    )
    assert low_evidence.ledger != evidence.ledger
