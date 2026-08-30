"""Read-only multi-window, multi-family economic study (tail utility first).

``evaluate_economic_family_study`` compares every declared fit window and both
declared model families on one common purged fold calendar. A candidate
reaches the exact base/stress execution replay only after its OOF tail-excess
lower bound and oracle capacity are positive; Rank IC stays observability
only. The winning family must clear every promotion-equivalent economic
predicate on a stitched growth-route certificate. Nothing is published: no
artifact, no ledger record, no production selection state, and the locked
forward holdout is never inspected. Peak memory is one fold model plus one
OOF/replay slice; candidates run sequentially and retain bounded scalars.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import polars as pl
from lightgbm.basic import LightGBMError

from src.stocks.ml.contracts import (
    ELASTIC_NET_FAMILY,
    TAIL_LAMBDARANK_FAMILY,
    EconomicFamilyStudySettings,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    PolicyProfile,
)

# wiring: build_route_tail_relevance(frame, route=request.route_objective, top_k=top_k)
# wiring: _lambda_rank_matrices(train, learner_columns, top_k, route=request.route_objective)
from src.stocks.ml.economic_objective import (
    InvalidOofEconomicUtilityError,
    TailCaptureEvidence,
    build_route_tail_relevance,
    build_tail_relevance,
    measure_tail_capture,
    route_labels_for_capture,
)
from src.stocks.ml.features import (
    apply_model_feature_schema,
    fit_model_feature_schema,
    materialize_model_feature_sources,
    stock_net_alpha_v1_roles,
)
from src.stocks.ml.horizons import (
    GROWTH_ROUTE_VERSION,
    HorizonOOFEvidence,
    stitch_prequential_growth_route,
)
from src.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    GROSS_COLUMN,
    ID_COLUMN,
    REALIZED_RETURN_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
    TARGET_COLUMN,
)
from src.stocks.ml.models import SCORE_COLUMN, NetAlphaModelConfig
from src.stocks.ml.preparation import (
    PreparedTrainingMatrix,
    TrainingPanelView,
    prepare_training_matrix,
)
from src.stocks.ml.replay_resources import (
    MemoryBudgetExceededError as _EnvelopeBudgetError,
)
from src.stocks.ml.replay_resources import (
    plan_training_allocation as _plan_training_allocation,
)
from src.stocks.ml.telemetry import emit_resource_checkpoint as _emit_resource_checkpoint
from src.stocks.ml.training import (
    _base_manifest,
    _build_label_join,
    _causal_oof_calibrate,
    _enforce_memory_budget,
    _evidence_from_execution,
    _fit_oof,
    _growth_route_projection,
    _incremental_growth_gate,
    _index_sessions,
    _locked_holdout,
    _MemoryBudgetExceededError,
    _on_demand_schema,
    _replay_costs_batch,
    _require_caller_registry,
    _seed_policy_or_none,
    build_initial_calibration_seed,
)
from src.stocks.ml.window_research import (
    _certificate_qualifies,
    derive_study_fold_count,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.metrics import certify_growth_route
from src.stocks.research.models import ModelManifest

__all__ = [
    "evaluate_economic_family_study",
    "evaluate_economic_window_candidate",
    "fit_rawnet_lgbm_oof",
    "fit_tail_lambdarank_oof",
]

logger = logging.getLogger("stocks.ml.economic_research")

_ID_COLUMN = ID_COLUMN
_SESSION_IDX = "session_index"
_OOF_SEGMENT = "oof_segment_id"
_MIN_STUDY_FOLDS = 3
_MAX_BLOCKED_VINTAGE_FRACTION = 0.05
_RAWNET_WINSOR_LO = 0.01
_RAWNET_WINSOR_HI = 0.99

_NO_LABEL_CAPACITY = "no-label-capacity"
_TAIL_CAPTURE_INSUFFICIENT = "tail-capture-insufficient"
_EXECUTION_ECONOMICS_INSUFFICIENT = "execution-economics-insufficient"
_RERUN_QUALIFIED_FAMILY = "rerun-qualified-family"
_INVALID_OOF_ECONOMIC_UTILITY = "invalid-oof-economic-utility"
_REPAIR_LABEL_INTEGRITY = "repair-label-integrity"

_STAGE_RANK = {
    _NO_LABEL_CAPACITY: 1,
    _TAIL_CAPTURE_INSUFFICIENT: 2,
    _EXECUTION_ECONOMICS_INSUFFICIENT: 3,
}

_ORACLE_REASON = "non-positive-oracle-excess"
_TAIL_BOUND_REASON = "tail-excess-lower-bound-non-positive"


def evaluate_economic_family_study(
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    settings: EconomicFamilyStudySettings,
    *,
    registry: ModelArtifactRegistry,
) -> dict[str, object]:
    """Compare windows x families on one common causal OOS calendar.

    Sequential per-window evaluation keeps peak memory equal to one candidate;
    selection is causal and family-wise controlled through an alpha split
    across the full declared ``window x family`` grid. Only a candidate with
    positive base/stress/matched-excess lower CAGRs, fills, coverage, and a
    positive tail-excess lower bound can be selected.
    """
    if (
        request.base_cost_schedule is None
        or request.stress_cost_schedule is None
        or request.liquidity_model is None
        or request.stress_liquidity_model is None
    ):
        raise ValueError(
            "the economic family study requires hash-bound base/stress cost "
            "schedules and both liquidity models (cost-evidence-required)"
        )
    annualization = request.compounding.annualization_sessions
    windows = settings.candidate_lookback_sessions
    families = settings.model_families
    finite = [v for v in windows if v is not None]
    if any(v < annualization for v in finite):
        raise ValueError(
            "every finite candidate lookback must be at least "
            f"annualization_sessions={annualization}"
        )
    feasible_cells = request.execution_frontier.require_feasible_horizons(
        request.portfolio.max_exposure,
        request.portfolio.max_single_weight,
    )
    candidate_count = (
        len(windows)
        * len(families)
        * len(feasible_cells)
        * len(request.policy_profiles)
    )
    if candidate_count < 1:
        raise ValueError("economic family study requires at least one candidate")
    alpha_window = request.compounding.bootstrap_alpha / candidate_count
    bootstrap_resamples = max(
        request.compounding.bootstrap_resamples, math.ceil(1.0 / alpha_window)
    )
    total_sessions = int(data.feature_frame[SESSION_COLUMN].n_unique())
    fold_count = derive_study_fold_count(
        total_sessions=total_sessions,
        forward_holdout_sessions=request.forward_holdout_sessions,
        common_min_train_sessions=settings.common_min_train_sessions,
        label_horizon_sessions=max(request.candidate_horizon_sessions) + 1,
        embargo_sessions=request.embargo_sessions,
        annualization_sessions=annualization,
        min_validation_segment_sessions=settings.min_validation_segment_sessions,
    )
    header: dict[str, object] = {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "adjusted_bootstrap_alpha": round(alpha_window, 12),
        "bootstrap_resamples": int(bootstrap_resamples),
        "candidate_count": int(candidate_count),
        "common_fold_count": int(fold_count),
        "selected_family": None,
        "recommended_lookback_sessions": None,
        "recommended_is_expanding": False,
    }
    if fold_count < _MIN_STUDY_FOLDS:
        return {
            **header,
            "study_complete": False,
            "next_action": _NO_LABEL_CAPACITY,
            "rejection_reason_counts": {"insufficient-common-window-calendar": 1},
            "candidates": [],
        }

    candidate_payloads: list[dict[str, object]] = []
    for lookback in windows:
        candidate_request = replace(
            request,
            max_training_lookback_sessions=lookback,
            fold_count=fold_count,
            bootstrap_alpha=alpha_window,
            bootstrap_resamples=bootstrap_resamples,
            compounding=replace(
                request.compounding,
                bootstrap_alpha=alpha_window,
                bootstrap_resamples=bootstrap_resamples,
            ),
        )
        result = evaluate_economic_window_candidate(
            data, candidate_request, settings, registry=registry
        )
        candidate_payloads.append(
            {
                "training_lookback_sessions": lookback,
                "is_expanding": lookback is None,
                **result,
            }
        )

    best_key: tuple[float, float] | None = None
    best_window: dict[str, object] | None = None
    best_family: str | None = None
    worst_stage_rank = 0
    for payload in candidate_payloads:
        families_summary = payload.get("families")
        if isinstance(families_summary, Mapping):
            for family in families:
                summary = families_summary.get(family)
                if not isinstance(summary, Mapping):
                    continue
                stage = summary.get("failure_stage")
                rank = _STAGE_RANK.get(stage) if isinstance(stage, str) else None
                if rank is not None and rank > worst_stage_rank:
                    worst_stage_rank = rank
                certificate = summary.get("certificate")
                if not summary.get("qualified") or not isinstance(certificate, Mapping):
                    continue
                key = _selection_key(certificate)
                if best_key is None or key > best_key:
                    best_key = key
                    best_window = payload
                    best_family = str(family)

    study_complete = all(
        bool(payload.get("study_complete")) for payload in candidate_payloads
    )
    integrity_failures = 0
    for payload in candidate_payloads:
        reasons = payload.get("rejection_reason_counts")
        if isinstance(reasons, Mapping):
            integrity_failures += int(reasons.get(_INVALID_OOF_ECONOMIC_UTILITY, 0))
    next_action = _RERUN_QUALIFIED_FAMILY
    if best_window is None or best_family is None:
        # A data-integrity failure is a repair action, never a capacity verdict.
        next_action = (
            _REPAIR_LABEL_INTEGRITY
            if integrity_failures > 0
            else _worst_stage_token(candidate_payloads, worst_stage_rank)
        )
        best_window = None
        best_family = None
    recommended_lookback = None
    recommended_is_expanding = False
    if best_window is not None:
        raw_lookback = best_window.get("training_lookback_sessions")
        recommended_lookback = raw_lookback if isinstance(raw_lookback, int) else None
        recommended_is_expanding = bool(best_window.get("is_expanding"))
    return {
        **header,
        "study_complete": study_complete,
        "next_action": next_action,
        "selected_family": best_family,
        "recommended_lookback_sessions": recommended_lookback,
        "recommended_is_expanding": recommended_is_expanding,
        "rejection_reason_counts": _aggregate_reasons(candidate_payloads),
        "candidates": candidate_payloads,
    }


def evaluate_economic_window_candidate(
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    settings: EconomicFamilyStudySettings,
    *,
    registry: ModelArtifactRegistry,
) -> dict[str, object]:
    """Run both families for one fit window over the shared fold calendar."""
    missing_horizons = [
        h
        for h in sorted(request.candidate_horizon_sessions)
        if h not in data.labels_by_horizon
    ]
    if missing_horizons:
        raise ValueError(
            f"requested horizon(s) {missing_horizons} have no label data; "
            "a missing requested horizon is a deterministic error"
        )
    if tuple(request.execution_frontier.candidate_horizon_sessions) != tuple(
        request.candidate_horizon_sessions
    ):
        raise ValueError(
            "execution_frontier.candidate_horizon_sessions must equal "
            "candidate_horizon_sessions before fitting"
        )
    request.execution_frontier.require_feasible_horizons(
        request.portfolio.max_exposure, request.portfolio.max_single_weight
    )
    feasible = request.execution_frontier.feasible_cells(
        request.portfolio.max_exposure, request.portfolio.max_single_weight
    )
    cells_by_horizon: dict[int, dict[int, list[int]]] = {}
    for h, c, k in feasible:
        cells_by_horizon.setdefault(h, {}).setdefault(k, []).append(c)

    # Label integrity is validated only at the OOF score/label join boundary
    # (see _process_calibrated_scores), so locked-holdout corruption can never
    # alter a supposedly holdout-blind research result.
    try:
        return _evaluate_window_body(data, request, settings, registry, cells_by_horizon)
    except InvalidOofEconomicUtilityError:
        return _window_rejection({}, _INVALID_OOF_ECONOMIC_UTILITY)
    except (_MemoryBudgetExceededError, _EnvelopeBudgetError) as exc:
        stage = str(getattr(exc, "stage", "") or "fitting_workspace")
        return _window_rejection({}, f"memory-budget-exceeded:{stage}")


def _evaluate_window_body(
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    settings: EconomicFamilyStudySettings,
    registry: ModelArtifactRegistry,
    cells_by_horizon: dict[int, dict[int, list[int]]],
) -> dict[str, object]:
    panel = _index_sessions(data.feature_frame)
    pre_holdout_raw, _holdout_raw, holdout_reason = _locked_holdout(panel, request)
    del _holdout_raw
    if holdout_reason or pre_holdout_raw.is_empty():
        return _window_rejection({}, holdout_reason or "insufficient-pre-holdout-history")
    roles = dict(stock_net_alpha_v1_roles())
    try:
        materialized = materialize_model_feature_sources(pre_holdout_raw, list(roles))
        schema = fit_model_feature_schema(materialized, roles)
        pre_holdout = apply_model_feature_schema(materialized, schema)
    except ValueError as exc:
        return _window_rejection({}, f"feature-schema-failed:{exc}")
    del pre_holdout_raw, materialized
    learner_columns = schema.learner_columns
    if not learner_columns:
        return _window_rejection({}, "no-alpha-learner-columns")

    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=max(request.candidate_horizon_sessions) + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=max(
            request.compounding.annualization_sessions,
            settings.common_min_train_sessions,
        ),
        max_train_sessions=request.max_training_lookback_sessions,
    )
    folds = splitter.split(pre_holdout)
    if not folds:
        return _window_rejection({}, "insufficient-oof-calendar")

    planned_bytes = int(pre_holdout.height) * len(learner_columns) * 4
    envelope = _plan_training_allocation(
        planned_bytes,
        request_limit_bytes=(
            None
            if request.max_rss_mib is None
            else int(request.max_rss_mib) * 1024 * 1024
        ),
        reserve_bytes=int(request.memory_reserve_mib) * 1024 * 1024,
    )
    _emit_resource_checkpoint(
        "economic_matrix_prepare", planned_bytes=planned_bytes, envelope=envelope.to_dict()
    )
    if not envelope.ok:
        raise _MemoryBudgetExceededError("matrix_prepare")
    matrix = prepare_training_matrix(
        TrainingPanelView(pre_holdout),
        _on_demand_schema(learner_columns),
        tuple(folds),
    )

    states = {
        family: _FamilyState(family=family) for family in settings.model_families
    }
    gated_candidates = 0
    for horizon in sorted(cells_by_horizon):
        manifest = _base_manifest(request, data, data.feature_frame, horizon)
        initial_rows = np.flatnonzero(
            matrix.session_code < folds[0].validation_decision_start - 1
        )
        seed_ledger = build_initial_calibration_seed(
            matrix, initial_rows, request, horizon, manifest, data=data
        )
        ks = sorted(cells_by_horizon[horizon])

        if ELASTIC_NET_FAMILY in states:
            elastic_oof, elastic_labels, elastic_ics = _elastic_oof_scores(
                pre_holdout, folds, data, request, manifest, learner_columns,
                horizon, matrix,
            )
            gated_candidates += _process_calibrated_scores(
                states[ELASTIC_NET_FAMILY],
                pre_holdout=pre_holdout,
                data=data,
                request=request,
                registry=registry,
                horizon=horizon,
                segment_count=len(folds),
                ks=ks,
                cadences_by_k={
                    k: sorted(set(cells_by_horizon[horizon][k])) for k in ks
                },
                oof=elastic_oof,
                oof_labels=elastic_labels,
                fold_rank_ics=tuple(elastic_ics),
                seed_ledger=seed_ledger,
            )
            del elastic_oof, elastic_labels

        if TAIL_LAMBDARANK_FAMILY in states:
            state = states[TAIL_LAMBDARANK_FAMILY]
            for k in ks:
                lambdarank_oof, lambdarank_labels = fit_tail_lambdarank_oof(
                    pre_holdout, folds, data, request, learner_columns, horizon, k
                )
                gated_candidates += _process_calibrated_scores(
                    state,
                    pre_holdout=pre_holdout,
                    data=data,
                    request=request,
                    registry=registry,
                    horizon=horizon,
                    segment_count=len(folds),
                    ks=[k],
                    cadences_by_k={k: sorted(set(cells_by_horizon[horizon][k]))},
                    oof=lambdarank_oof,
                    oof_labels=lambdarank_labels,
                    fold_rank_ics=(),
                    seed_ledger=seed_ledger,
                )
                del lambdarank_oof, lambdarank_labels
        del seed_ledger
        _enforce_memory_budget(request, f"economic_horizon_{horizon}")

    families_summary: dict[str, dict[str, object]] = {}
    aggregate_reasons: dict[str, int] = {}
    for family, state in states.items():
        summary = state.summarize(request)
        families_summary[family] = summary
        for token, count in state.reason_counts.items():
            aggregate_reasons[token] = aggregate_reasons.get(token, 0) + count

    selected_family = None
    selected_certificate: dict[str, object] | None = None
    best_key: tuple[float, float] | None = None
    for family in settings.model_families:
        summary = families_summary[family]
        certificate = summary.get("certificate")
        if not summary.get("qualified") or not isinstance(certificate, Mapping):
            continue
        key = _selection_key(certificate)
        if best_key is None or key > best_key:
            best_key = key
            selected_family = family
            selected_certificate = dict(certificate)
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "study_complete": True,
        "fold_count": len(folds),
        "candidates_evaluated": gated_candidates,
        "selected_family": selected_family,
        "certificate": selected_certificate,
        "families": families_summary,
        "rejection_reason_counts": dict(sorted(aggregate_reasons.items())),
    }


def _elastic_oof_scores(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    horizon: int,
    matrix: PreparedTrainingMatrix,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float]]:
    oof, labeled, rank_ics, _diagnostic, _paths = _fit_oof(
        pre_holdout, folds, data, request, manifest, learner_columns,
        horizon, None,
        family="net_alpha_elastic_net",
        matrix=matrix,
    )
    return oof, labeled, rank_ics


def _process_calibrated_scores(
    state: _FamilyState | None,
    *,
    pre_holdout: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    registry: ModelArtifactRegistry,
    horizon: int,
    segment_count: int,
    ks: Sequence[int],
    cadences_by_k: Mapping[int, Sequence[int]],
    oof: pl.DataFrame,
    oof_labels: pl.DataFrame,
    fold_rank_ics: tuple[float, ...],
    seed_ledger: pl.DataFrame,
) -> int:
    """Tail-gate one family's calibrated OOF scores and replay surviving K cells."""
    if state is None:
        return 0
    if oof.is_empty() or oof_labels.is_empty():
        for _k in ks:
            state.reason_counts["no-oof-labels"] = (
                state.reason_counts.get("no-oof-labels", 0) + 1
            )
        return 0
    calibrated = _causal_oof_calibrate(
        oof, oof_labels, request, horizon, seed_ledger=seed_ledger
    )
    gated = 0
    for k in ks:
        gated += 1
        # Wiring: project_route_utility -> measure_tail_capture
        capture_labels = route_labels_for_capture(
            data.labels_by_horizon[horizon], request.route_objective
        )
        capture = measure_tail_capture(
            calibrated,
            capture_labels,
            top_k=int(k),
            bootstrap_alpha=request.bootstrap_alpha,
            bootstrap_resamples=request.bootstrap_resamples,
            seed=request.seed + horizon * 1009 + int(k),
        )
        state.observe_capture(capture)
        if not capture.oracle_capacity_ok:
            state.reason_counts[_ORACLE_REASON] = (
                state.reason_counts.get(_ORACLE_REASON, 0) + 1
            )
            continue
        if not capture.tail_gate_ok:
            state.reason_counts[_TAIL_BOUND_REASON] = (
                state.reason_counts.get(_TAIL_BOUND_REASON, 0) + 1
            )
            continue
        specs = [
            (cadence, int(k), profile)
            for cadence in cadences_by_k[k]
            for profile in request.policy_profiles
        ]
        evidence, dropouts = _replay_family_candidates(
            registry,
            calibrated,
            oof_labels,
            request,
            horizon,
            pre_holdout,
            data,
            specs,
            fold_rank_ics=fold_rank_ics,
            segment_count=segment_count,
            model_family=state.family,
        )
        for token, count in dropouts.items():
            state.reason_counts[token] = state.reason_counts.get(token, 0) + count
        state.admit(evidence)
    return gated


def _replay_family_candidates(
    registry: ModelArtifactRegistry,
    calibrated: pl.DataFrame,
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon: int,
    market_frame: pl.DataFrame,
    data: NetAlphaResearchData,
    specs: Sequence[tuple[int, int, PolicyProfile]],
    *,
    fold_rank_ics: tuple[float, ...],
    segment_count: int,
    model_family: str,
) -> tuple[list[HorizonOOFEvidence], dict[str, int]]:
    """Replay one (family, horizon, K) arm through the exact execution engine."""
    dropout_reasons: dict[str, int] = {}
    evidence: list[HorizonOOFEvidence] = []
    if not specs or calibrated.is_empty():
        dropout_reasons["no-executable-specs"] = 1
        return evidence, dropout_reasons
    batch_results = _replay_costs_batch(
        _require_caller_registry(registry),
        calibrated,
        oof_labels,
        request,
        horizon,
        request.risk,
        market_frame,
        data.manifest,
        list(specs),
        stats_out={},
    )
    for cadence, top_k, profile in specs:
        key = (horizon, cadence, top_k, profile.profile_id)
        pair = batch_results.get(key)
        if pair is None:
            dropout_reasons["replay-batch-error"] = (
                dropout_reasons.get("replay-batch-error", 0) + 1
            )
            continue
        base_evidence = pair.candidate
        shadow_evidence = pair.dense_shadow
        is_v5 = (
            profile.execution_utility_mode == "sparse_hold_replace_v2"
            or profile.sizing_mode == "risk_balanced_waterfill_v2"
        )
        if is_v5:
            if shadow_evidence is None:
                dropout_reasons["missing-shadow-evidence"] = (
                    dropout_reasons.get("missing-shadow-evidence", 0) + 1
                )
                continue
            _incremental_growth_gate(base_evidence, shadow_evidence, request, horizon)
        if not base_evidence.base_log_growth:
            dropout_reasons["no-evaluated-vintages"] = (
                dropout_reasons.get("no-evaluated-vintages", 0) + 1
            )
            continue
        if base_evidence.filled_orders == 0:
            dropout_reasons["no-filled-orders"] = (
                dropout_reasons.get("no-filled-orders", 0) + 1
            )
            continue
        stress_evidence = base_evidence
        paired_stress: tuple[float, ...] = ()
        sparse_turnover = base_evidence.turnover
        shadow_turnover = shadow_evidence.turnover if shadow_evidence is not None else 0.0
        if shadow_evidence is not None:
            paired_stress = tuple(
                float(c - s)
                for c, s in zip(
                    base_evidence.stress_log_growth,
                    shadow_evidence.stress_log_growth,
                    strict=True,
                )
            )
        candidate_evidence = _evidence_from_execution(
            horizon,
            profile.profile_id,
            model_family,
            base_evidence,
            stress_evidence,
            fold_rank_ics,
            segment_count,
            paired_stress_log_growth=paired_stress,
            sparse_turnover=sparse_turnover,
            shadow_turnover=shadow_turnover,
            rebalance_frequency_sessions=cadence,
            top_k=top_k,
        )
        failure_reason = _economic_coverage_failure_reason(candidate_evidence, request)
        if failure_reason:
            dropout_reasons[failure_reason] = (
                dropout_reasons.get(failure_reason, 0) + 1
            )
            continue
        evidence.append(candidate_evidence)
    return evidence, dropout_reasons


def _economic_coverage_failure_reason(
    evidence: HorizonOOFEvidence, request: NetAlphaTrainingRequest
) -> str:
    """Fail-closed coverage/admission reason without any Rank IC clause."""
    distinct_segments = len(set(evidence.cohort_segment_ids))
    if distinct_segments != evidence.segment_count:
        return (
            f"incomplete-segment-coverage:{distinct_segments}/"
            f"{evidence.segment_count}"
        )
    total_vintages = max(1, int(evidence.complete_cohort_count))
    if (
        evidence.blocked_vintage_count / total_vintages
        > _MAX_BLOCKED_VINTAGE_FRACTION
    ):
        return f"selected-exit-unresolved:{evidence.blocked_vintage_count}"
    if evidence.missing_cohort_count > 0:
        return f"missing-realized-vintages:{evidence.missing_cohort_count}"
    observed = int(evidence.complete_cohort_count)
    if observed < request.compounding.min_observed_sessions:
        return (
            f"insufficient-observed-sessions:{observed}/"
            f"{request.compounding.min_observed_sessions}"
        )
    if observed <= 0:
        return "no-complete-cohorts"
    active_fraction = evidence.active_cohort_count / observed
    if active_fraction < request.compounding.min_active_cohort_fraction:
        return f"active-coverage-insufficient:{active_fraction:.4f}"
    return ""


def fit_tail_lambdarank_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon: int,
    top_k: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic LightGBM LambdaRank OOF scores for one (H, K) cell.

    Inner chronological splits pick the boost-round count; every outer model
    refits to that fixed count so outer validation stays target-free. The
    relevance target is ``build_tail_relevance`` exact-K on the decimal
    ``risk_residual - reference_cost``, never MAD-z.
    """
    label_join = _build_label_join(data, horizon)
    config = NetAlphaModelConfig(seed=request.seed)
    inner_iterations: list[int] = []
    for fold in folds:
        train = pre_holdout[fold.train_mask].join(
            label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner"
        )
        if train.is_empty():
            continue
        best = _inner_lambda_rank_iteration(
            train, learner_columns, top_k, config, request.model_threads, route=request.route_objective
        )
        if best is not None:
            inner_iterations.append(best)
    if not inner_iterations:
        return pl.DataFrame(), pl.DataFrame()
    rounds = int(np.median(inner_iterations))

    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    label_subset = label_join.select(
        _ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN, AVAILABLE_COLUMN,
        GROSS_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN,
        REALIZED_RETURN_COLUMN,
    )
    for _fold_index, fold in enumerate(folds):
        train = pre_holdout[fold.train_mask].join(
            label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner"
        )
        validation = pre_holdout[fold.validation_mask]
        if train.is_empty() or validation.is_empty():
            continue
        features, relevance, groups = _lambda_rank_matrices(
            train, learner_columns, top_k, route=request.route_objective
        )
        if features.shape[0] == 0 or groups.size == 0:
            continue
        params = _lambda_rank_params(config, top_k, request.model_threads)
        train_set = lgb.Dataset(
            features, label=relevance, group=groups.tolist(), params={"verbosity": -1}
        )
        booster = lgb.train(params, train_set, num_boost_round=rounds)
        valid_features = _design_matrix(validation, learner_columns)
        scores = np.asarray(
            booster.predict(valid_features), dtype=np.float64
        )
        scored = validation.select(_ID_COLUMN, SESSION_COLUMN, _SESSION_IDX).with_columns(
            pl.Series(SCORE_COLUMN, scores),
            pl.lit(fold.segment_id, dtype=pl.Int64).alias(_OOF_SEGMENT),
        )
        labeled = scored.join(label_subset, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
        if labeled.is_empty():
            continue
        oof_frames.append(scored)
        label_frames.append(labeled)
    if not oof_frames:
        return pl.DataFrame(), pl.DataFrame()
    return pl.concat(oof_frames), pl.concat(label_frames)



def _rawnet_lgbm_params(
    config: NetAlphaModelConfig, num_threads: int, seed: int
) -> dict[str, object]:
    """Deterministic L2 regression parameters for one bagging seed."""
    return {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_samples": config.min_child_samples,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": int(num_threads),
        "seed": int(seed),
        "deterministic": True,
        "force_col_wise": True,
        "data_random_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "verbosity": -1,
    }


def _add_winsorized_utility(train: pl.DataFrame) -> pl.DataFrame:
    """Per-session 1%/99% clip of ``risk_residual - reference_cost``.

    Quantiles come exclusively from the fold's training join; validation and
    holdout rows can never move them.
    """
    utility = pl.col(RISK_RESIDUAL_COLUMN) - pl.col(REFERENCE_COST_COLUMN)
    bounds = train.group_by(SESSION_COLUMN).agg(
        utility.quantile(_RAWNET_WINSOR_LO).alias("__lo"),
        utility.quantile(_RAWNET_WINSOR_HI).alias("__hi"),
    )
    clipped = (
        train.join(bounds, on=SESSION_COLUMN, how="left")
        .with_columns(
            utility.clip(pl.col("__lo"), pl.col("__hi")).alias("__utility")
        )
        .drop(["__lo", "__hi"])
    )
    if clipped["__utility"].null_count() > 0:
        raise InvalidOofEconomicUtilityError(
            "rawnet training rows carry null risk_residual/reference_cost"
        )
    return clipped


def _inner_rawnet_best_round(
    train_sorted: pl.DataFrame,
    learner_columns: tuple[str, ...],
    config: NetAlphaModelConfig,
    num_threads: int,
) -> int | None:
    """Inner chronological-split early-stopped round count on winsorized L2."""
    sessions = train_sorted[SESSION_COLUMN].unique().sort().to_list()
    holdout_n = max(1, len(sessions) // 4)
    if len(sessions) - holdout_n < 1:
        return None
    valid_sessions = sessions[-holdout_n:]
    inner_valid = train_sorted.filter(
        pl.col(SESSION_COLUMN).is_in(list(valid_sessions))
    )
    inner_train = train_sorted.filter(
        ~pl.col(SESSION_COLUMN).is_in(list(valid_sessions))
    )
    if inner_train.is_empty() or inner_valid.is_empty():
        return None
    train_features = _design_matrix(inner_train, learner_columns)
    valid_features = _design_matrix(inner_valid, learner_columns)
    params = _rawnet_lgbm_params(config, num_threads, config.seed)
    train_set = lgb.Dataset(
        train_features,
        label=inner_train["__utility"].to_numpy(),
        params={"verbosity": -1},
    )
    valid_set = lgb.Dataset(
        valid_features,
        label=inner_valid["__utility"].to_numpy(),
        reference=train_set,
        params={"verbosity": -1},
    )
    try:
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=config.n_estimators,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        )
    except LightGBMError:
        return None
    best = int(getattr(booster, "best_iteration", 0) or 0)
    return best or None


def _rawnet_seed_scores(
    train_sorted: pl.DataFrame,
    valid_sorted: pl.DataFrame,
    learner_columns: tuple[str, ...],
    config: NetAlphaModelConfig,
    num_threads: int,
    rounds: int,
    seed: int,
) -> np.ndarray:
    """One seed's validation scores from fixed-round winsorized-L2 fitting."""
    params = _rawnet_lgbm_params(config, num_threads, seed)
    train_set = lgb.Dataset(
        _design_matrix(train_sorted, learner_columns),
        label=train_sorted["__utility"].to_numpy(),
        params={"verbosity": -1},
    )
    booster = lgb.train(params, train_set, num_boost_round=rounds)
    return np.asarray(
        booster.predict(_design_matrix(valid_sorted, learner_columns)),
        dtype=np.float64,
    )


def _rank_average_seed_scores(seed_frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Mean cross-sectional percentile rank across per-seed score frames.

    Every seed frame must carry identical ``(instrument_id, session)`` keys
    and a finite ``score`` column; any coverage mismatch fails closed.
    """
    if len(seed_frames) < 2:
        raise ValueError("seed bagging requires at least two seed score frames")
    key_sets = [
        set(
            zip(
                frame[_ID_COLUMN].to_list(),
                frame[SESSION_COLUMN].to_list(),
                strict=True,
            )
        )
        for frame in seed_frames
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError(
            "seed bagging requires identical (instrument_id, session) coverage "
            "in every seed score frame"
        )
    stacked = pl.concat(
        [
            frame.select(_ID_COLUMN, SESSION_COLUMN, SCORE_COLUMN).with_columns(
                pl.lit(index, dtype=pl.Int64).alias("__seed")
            )
            for index, frame in enumerate(seed_frames)
        ],
        how="vertical",
    )
    ranked = stacked.with_columns(
        pl.col(SCORE_COLUMN)
        .rank("average")
        .over([SESSION_COLUMN, "__seed"])
        .alias("__rk"),
        pl.len().over([SESSION_COLUMN, "__seed"]).alias("__n"),
    ).with_columns(
        pl.when(pl.col("__n") > 1)
        .then((pl.col("__rk") - 1.0) / (pl.col("__n") - 1.0))
        .otherwise(0.5)
        .cast(pl.Float64)
        .alias("__pct")
    )
    averaged = (
        ranked.group_by([_ID_COLUMN, SESSION_COLUMN])
        .agg(pl.col("__pct").mean().alias(SCORE_COLUMN))
        .sort([_ID_COLUMN, SESSION_COLUMN])
    )
    if averaged[SCORE_COLUMN].null_count() > 0 or not bool(
        averaged[SCORE_COLUMN].is_finite().all()
    ):
        raise ValueError("seed-bagged scores must be finite for every row")
    return averaged


def rawnet_fold_rank_ics(labeled: pl.DataFrame) -> list[float]:
    """Mean per-session Rank-IC of bagged scores vs realized net residual.

    One value per ``oof_segment_id`` (ascending), computed only on labeled
    validation rows so a missing realized outcome never fabricates a fold
    statistic.
    """
    if labeled.is_empty() or _OOF_SEGMENT not in labeled.columns:
        return []
    ranked = (
        labeled.select(
            _OOF_SEGMENT,
            _SESSION_IDX,
            SCORE_COLUMN,
            RISK_RESIDUAL_COLUMN,
        )
        .drop_nulls([SCORE_COLUMN, RISK_RESIDUAL_COLUMN])
        .with_columns(
            pl.col(SCORE_COLUMN).rank("average").over(_SESSION_IDX).alias("__rs"),
            pl.col(RISK_RESIDUAL_COLUMN)
            .rank("average")
            .over(_SESSION_IDX)
            .alias("__rr"),
        )
    )
    per_session = ranked.group_by(_OOF_SEGMENT, _SESSION_IDX).agg(
        pl.corr("__rs", "__rr").alias("ic")
    )
    per_fold = (
        per_session.drop_nulls("ic")
        .group_by(_OOF_SEGMENT)
        .agg(pl.col("ic").mean().alias("fold_ic"))
        .sort(_OOF_SEGMENT)
    )
    values = [
        float(value)
        for value in per_fold["fold_ic"].to_list()
        if np.isfinite(value)
    ]
    return values


def fit_rawnet_lgbm_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic winsorized-net LightGBM regression OOF with seed bagging.

    The regression target is the simple-decimal economic utility
    ``risk_residual - reference_cost``, winsorized at per-session 1%/99%
    quantiles computed only on each fold's training join. Three consecutive
    seeds fit at a shared round count (median of inner early-stopped best
    iterations); their validation predictions are combined by mean
    cross-sectional percentile rank so no single seed's ordering dominates.
    Canonical ``[session, instrument_id]`` sorting precedes every matrix build
    and repeated calls on identical inputs are bit-identical.
    """
    label_join = _build_label_join(data, horizon_sessions)
    config = NetAlphaModelConfig(seed=request.seed)
    seeds = (request.seed, request.seed + 1, request.seed + 2)

    inner_iterations: list[int] = []
    prepared_folds: list[tuple[int, pl.DataFrame, pl.DataFrame]] = []
    for fold_index, fold in enumerate(folds):
        raw_train = pre_holdout[fold.train_mask].join(
            label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner"
        )
        validation = pre_holdout[fold.validation_mask]
        if raw_train.is_empty() or validation.is_empty():
            logger.info(
                "[RAWRNET] fold=%d skipped train_rows=%d val_rows=%d "
                "label_join_rows=%d",
                fold_index,
                len(fold.train_mask),
                len(fold.validation_mask),
                label_join.height,
            )
            continue
        train_sorted = _add_winsorized_utility(raw_train).sort(
            [SESSION_COLUMN, _ID_COLUMN], maintain_order=True
        )
        prepared_folds.append((int(fold.segment_id), train_sorted, validation))
        best = _inner_rawnet_best_round(
            train_sorted, learner_columns, config, request.model_threads
        )
        logger.info(
            "[RAWRNET] fold=%d prepared train_rows=%d inner_best_round=%s",
            fold_index,
            train_sorted.height,
            best,
        )
        if best is not None:
            inner_iterations.append(best)
    if not inner_iterations or not prepared_folds:
        logger.info(
            "[RAWRNET] empty result: prepared_folds=%d inner_iterations=%d",
            len(prepared_folds),
            len(inner_iterations),
        )
        return pl.DataFrame(), pl.DataFrame()
    rounds = int(np.median(inner_iterations))

    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    label_subset = label_join.select(
        _ID_COLUMN,
        SESSION_COLUMN,
        TARGET_COLUMN,
        AVAILABLE_COLUMN,
        RISK_RESIDUAL_COLUMN,
        REFERENCE_COST_COLUMN,
        REALIZED_RETURN_COLUMN,
    )
    for segment_id, train_sorted, validation in prepared_folds:
        if validation.is_empty():
            continue
        valid_sorted = validation.sort([SESSION_COLUMN, _ID_COLUMN], maintain_order=True)
        seed_frames: list[pl.DataFrame] = []
        for seed in seeds:
            predictions = _rawnet_seed_scores(
                train_sorted,
                valid_sorted,
                learner_columns,
                config,
                request.model_threads,
                rounds,
                seed,
            )
            seed_frames.append(
                valid_sorted.select(_ID_COLUMN, SESSION_COLUMN).with_columns(
                    pl.Series(SCORE_COLUMN, predictions)
                )
            )
        scored = _rank_average_seed_scores(tuple(seed_frames)).with_columns(
            pl.lit(segment_id, dtype=pl.Int64).alias(_OOF_SEGMENT)
        )
        session_lookup = valid_sorted.select(
            _ID_COLUMN, SESSION_COLUMN, pl.col(_SESSION_IDX)
        )
        scored = scored.join(session_lookup, on=[_ID_COLUMN, SESSION_COLUMN], how="left")
        if scored[_SESSION_IDX].null_count() > 0:
            raise ValueError(
                "rawnet OOF scoring produced a row absent from its fold "
                "validation partition"
            )
        # Canonical column order matches the prepared-array OOF ledger schema
        # so causal calibration can vstack seed ledgers without recasting.
        scored = scored.select(
            _ID_COLUMN, SESSION_COLUMN, _SESSION_IDX, SCORE_COLUMN, _OOF_SEGMENT
        )
        labeled = scored.join(
            label_subset, on=[_ID_COLUMN, SESSION_COLUMN], how="inner"
        )
        if labeled.is_empty():
            continue
        oof_frames.append(scored)
        label_frames.append(labeled)
    if not oof_frames:
        return pl.DataFrame(), pl.DataFrame()
    return pl.concat(oof_frames), pl.concat(label_frames)


def _inner_lambda_rank_iteration(
    train: pl.DataFrame,
    learner_columns: tuple[str, ...],
    top_k: int,
    config: NetAlphaModelConfig,
    num_threads: int,
    *,
    route: object | None = None,
) -> int | None:
    """Median-style inner early-stopped round count on a chronological split."""
    sessions = train[SESSION_COLUMN].unique().sort().to_list()
    holdout_n = max(1, len(sessions) // 4)
    if len(sessions) - holdout_n < 1:
        return None
    valid_sessions = sessions[-holdout_n:]
    inner_valid = train.filter(pl.col(SESSION_COLUMN).is_in(list(valid_sessions)))
    inner_train = train.filter(~pl.col(SESSION_COLUMN).is_in(list(valid_sessions)))
    train_features, train_relevance, train_groups = _lambda_rank_matrices(
        inner_train, learner_columns, top_k, route=route
    )
    valid_features, valid_relevance, valid_groups = _lambda_rank_matrices(
        inner_valid, learner_columns, top_k, route=route
    )
    if train_groups.size == 0 or valid_groups.size == 0:
        return None
    params = _lambda_rank_params(config, top_k, num_threads)
    train_set = lgb.Dataset(
        train_features,
        label=train_relevance,
        group=train_groups.tolist(),
        params={"verbosity": -1},
    )
    valid_set = lgb.Dataset(
        valid_features,
        label=valid_relevance,
        group=valid_groups.tolist(),
        reference=train_set,
        params={"verbosity": -1},
    )
    try:
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=config.n_estimators,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False)
            ],
        )
    except (LightGBMError, ValueError):
        return None
    return int(booster.best_iteration) if booster.best_iteration > 0 else None


def _lambda_rank_matrices(
    frame: pl.DataFrame,
    learner_columns: tuple[str, ...],
    top_k: int,
    *,
    route: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix, exact-K relevance targets, and session group sizes."""
    if route is not None:
        relevance = build_route_tail_relevance(frame, route=route, top_k=top_k).select(_ID_COLUMN, SESSION_COLUMN, "relevance")
    else:
        relevance = build_tail_relevance(
            frame.select(_ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN),
            top_k=top_k,
        ).select(_ID_COLUMN, SESSION_COLUMN, "relevance")
    aligned = frame.join(relevance, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
    aligned = aligned.sort([SESSION_COLUMN, _ID_COLUMN], maintain_order=True)
    features = _design_matrix(aligned, learner_columns)
    labels = aligned["relevance"].to_numpy().astype(np.int32)
    session_values = aligned[SESSION_COLUMN].to_physical().to_numpy()
    boundaries = np.flatnonzero(np.diff(session_values)) + 1
    edges = np.concatenate(([0], boundaries, [aligned.height]))
    groups = np.diff(edges).astype(np.int64)
    return features, labels, groups


def _design_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.ascontiguousarray(
        frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy(),
        dtype=np.float32,
    )


def _lambda_rank_params(
    config: NetAlphaModelConfig, top_k: int, num_threads: int
) -> dict[str, object]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [top_k],
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_samples": config.min_child_samples,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": num_threads,
        "seed": config.seed,
        "deterministic": True,
        "force_col_wise": True,
        "data_random_seed": config.seed,
        "feature_fraction_seed": config.seed,
        "bagging_seed": config.seed,
        "verbosity": -1,
    }


class _FamilyState:
    """Bounded accumulator for one family's study arm (never raw rows)."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.evidence: list[HorizonOOFEvidence] = []
        self.reason_counts: dict[str, int] = {}
        self.any_oracle_ok = False
        self.any_tail_ok = False
        self.best_tail_bound: float | None = None
        self.best_capture_ratio: float | None = None

    def observe_capture(self, capture: TailCaptureEvidence) -> None:
        if capture.oracle_capacity_ok:
            self.any_oracle_ok = True
        if capture.tail_gate_ok:
            self.any_tail_ok = True
        bound = float(capture.tail_excess_lower_bound)
        if self.best_tail_bound is None or bound > self.best_tail_bound:
            self.best_tail_bound = bound
        ratio = capture.tail_capture_ratio
        if ratio is not None and (
            self.best_capture_ratio is None or float(ratio) > self.best_capture_ratio
        ):
            self.best_capture_ratio = float(ratio)

    def admit(self, evidence: list[HorizonOOFEvidence]) -> None:
        self.evidence.extend(evidence)

    def summarize(self, request: NetAlphaTrainingRequest) -> dict[str, object]:
        certificate: dict[str, object] | None = None
        projection: dict[str, object] = {
            "version": GROWTH_ROUTE_VERSION,
            "candidate_count": 0,
            "selected_policy": None,
            "rejection_reason_counts": dict(sorted(self.reason_counts.items())),
        }
        if self.evidence:
            route = stitch_prequential_growth_route(
                tuple(self.evidence),
                request.bootstrap_alpha,
                request.seed,
                request.bootstrap_resamples,
                seed_policy=_seed_policy_or_none(request),
            )
            primary = (
                route.selected_policies[-1][0]
                if route.selected_policies and route.selected_policies[-1] is not None
                else self.evidence[0].horizon_sessions
            )
            _acct = request.account_certification
            # Wiring: certify_growth_route - certified = certify_growth_route(route, primary, request.compounding) - family-study certificate uses the same account minimum_lower_cagr and max_drawdown resolved from the request
            certified = certify_growth_route(
                route,
                primary,
                request.compounding,
                minimum_lower_cagr=float(_acct.minimum_lower_cagr) if _acct is not None else 0.0,
                max_drawdown=float(_acct.max_drawdown) if _acct is not None else None,
            )
            certificate = dict(certified)
            projection = _growth_route_projection(route, certificate, compounding=request.compounding, horizon_sessions=primary, capital_plan_settings=request.capital_plan, account_certification=request.account_certification)  # noqa: E501
        qualified = bool(
            self.any_tail_ok
            and certificate is not None
            and _certificate_qualifies(certificate, request)
        )
        if qualified:
            failure_stage = None
        elif not self.any_oracle_ok:
            failure_stage = _NO_LABEL_CAPACITY
        elif not self.any_tail_ok:
            failure_stage = _TAIL_CAPTURE_INSUFFICIENT
        else:
            failure_stage = _EXECUTION_ECONOMICS_INSUFFICIENT
        return {
            "model_family": self.family,
            "admitted_candidate_count": len(self.evidence),
            "oracle_capacity_observed": self.any_oracle_ok,
            "tail_gate_observed": self.any_tail_ok,
            "best_tail_excess_lower_bound": self.best_tail_bound,
            "best_tail_capture_ratio": self.best_capture_ratio,
            "certificate": certificate,
            "growth_route": projection,
            "qualified": qualified,
            "failure_stage": failure_stage,
            "rejection_reason_counts": dict(sorted(self.reason_counts.items())),
        }


def _selection_key(certificate: Mapping[str, object]) -> tuple[float, float]:
    stress = _float_or_minus_inf(certificate.get("stress_lower_cagr"))
    matched = _float_or_minus_inf(certificate.get("matched_lower_excess_cagr"))
    return (stress, matched)


def _float_or_minus_inf(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return -math.inf


def _worst_stage_token(
    payloads: list[dict[str, object]], worst_stage_rank: int
) -> str:
    ranked = {
        1: _NO_LABEL_CAPACITY,
        2: _TAIL_CAPTURE_INSUFFICIENT,
        3: _EXECUTION_ECONOMICS_INSUFFICIENT,
    }
    return ranked.get(worst_stage_rank, _NO_LABEL_CAPACITY)


def _aggregate_reasons(payloads: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in payloads:
        raw = payload.get("rejection_reason_counts")
        if not isinstance(raw, Mapping):
            continue
        for token, count in raw.items():
            counts[str(token)] = counts.get(str(token), 0) + int(count)
    return dict(sorted(counts.items()))


def _window_rejection(
    reasons: Mapping[str, int], token: str
) -> dict[str, object]:
    counts = {str(key): int(value) for key, value in reasons.items()}
    counts[token] = counts.get(token, 0) + 1
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "study_complete": False,
        "fold_count": 0,
        "candidates_evaluated": 0,
        "selected_family": None,
        "certificate": None,
        "families": {},
        "rejection_reason_counts": dict(sorted(counts.items())),
    }
