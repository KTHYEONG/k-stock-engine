"""Thin net-alpha training orchestrator.

``train_net_alpha_model`` is the single training entry point: integrity audit,
lock the raw forward holdout *before* fitting any feature schema, apply the
frozen schema to pre-holdout and holdout, build one maximum-horizon balanced
purged/embargoed fold plan, collect segment-safe causal per-horizon OOF
evidence (vectorized weighted ElasticNet path, target-free validation
prediction, decimal realized-outcome calibration, and execution-equivalent
replay of the calibrated scores through the prepared ``StockBacktester`` under
base and stress costs), Holm-adjusted horizon selection on cohort-unit
per-session log growth, one evidence-gated deterministic LightGBM challenger on
the selected primary, and an untouched forward holdout. The residual-vintage
``NetAlphaPolicyReplay`` proxy is fully replaced by true long-only equity
evidence; horizon never drives capital lock or rebalance cadence. The final
decision publishes either one champion family (learner plus fitted decimal
calibration) or a complete immutable ``NO_TRADE`` artifact. Future labels are
never a discovery score, the holdout is never refit, and a selected baseline
OOF is never recomputed. No Optuna, confirmation worker, LambdaRank route, or
fixed 5/10/15 horizon exists here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from src.stocks.ml.data import HorizonOutcomeCoverage

import numpy as np
import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.core.portfolio import PortfolioSnapshot
from src.stocks.data.ml_integrity import validate_ml_snapshot
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
from src.stocks.ml.capital_plan import build_small_capital_route_plan
from src.stocks.ml.compound_track import (
    frozen_compound_track_projection,
    resolve_frozen_policy_key,
    stitch_frozen_policy_growth_route,
)
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    DECLARED_ECONOMIC_FAMILIES,
    RAWNET_LGBM_FAMILY,
    TAIL_LAMBDARANK_FAMILY,
    CompoundingCertificationSettings,
    FoldScoreDiagnostic,
    HorizonOOFDiagnostic,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    PolicyProfile,
    RegularizationGrid,
    RiskSettings,
    SmallCapitalPlanSettings,
    policy_portfolio_fingerprint,
)
from src.stocks.ml.data import assess_snapshot_outcome_readiness
from src.stocks.ml.discovery import discover_horizons
from src.stocks.ml.execution_replay import (
    ExecutionEquivalentReplayRequest,
    ExecutionReplayContext,
    ExecutionReplayEvidence,
    ProfileReplayEvidence,
    exposure_matched_benchmark_log_growth,
    instruments_from_frame,
    replay_execution_equivalent,
    stream_execution_replay_batch,
)
from src.stocks.ml.features import (
    FeatureTransformSchema,
    apply_model_feature_schema,
    fit_model_feature_schema,
    materialize_model_feature_sources,
    stock_net_alpha_v1_contract_book,
    stock_net_alpha_v1_roles,
)
from src.stocks.ml.fitting import (
    OofCache as _OofCache,
)
from src.stocks.ml.fitting import (
    default_oof_cache_base as _default_oof_cache_base,
)
from src.stocks.ml.fitting import (
    read_oof_parquet as _read_oof_parquet,
)
from src.stocks.ml.hedge_sleeve import project_hedge_sleeve
from src.stocks.ml.horizons import (
    GROWTH_ROUTE_VERSION,
    GrowthRouteEvidence,
    HorizonOOFEvidence,
    HorizonSelectionEvidence,
    PolicyKey,
    _cohort_bootstrap,
    select_horizons,
    stitch_prequential_growth_route,
)
from src.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    REALIZED_RETURN_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
    TARGET_COLUMN,
)
from src.stocks.ml.models import (
    SCORE_COLUMN,
    CalibratedNetAlphaModel,
    CalibrationApplier,
    CausalCalibrationAdapter,
    ElasticNetNetAlpha,
    LightGbmNetAlpha,
    NetAlphaModelConfig,
)
from src.stocks.ml.preparation import (
    PreparedFold,
    PreparedTrainingMatrix,
    TrainingPanelView,
    prepare_folds,
    prepare_horizon_labels,
    prepare_training_matrix,
)
from src.stocks.ml.replay_resources import (
    MemoryBudgetExceededError as _EnvelopeBudgetError,
)
from src.stocks.ml.replay_resources import (
    plan_training_allocation as _plan_training_allocation,
)
from src.stocks.ml.telemetry import TrainingTelemetry
from src.stocks.ml.telemetry import (
    current_rss_mib as _current_rss_mib,
)
from src.stocks.ml.telemetry import (
    emit_resource_checkpoint as _emit_resource_checkpoint,
)
from src.stocks.ml.telemetry import peak_rss_mib as _peak_rss_mib
from src.stocks.ml.telemetry import (
    set_active_telemetry as _set_active_telemetry,
)

__all__ = ["TrainingTelemetry", "train_net_alpha_model"]

from src.stocks.observability.contracts import (
    DiagnosticCategory,
    DiagnosticStage,
    DiagnosticStatus,
    emit_checkpoint,
)
from src.stocks.observability.contracts import (
    RunDiagnostics as _RunDiagnostics,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.calibration_schedule import SessionClusterCalibrationSchedule
from src.stocks.research.economic_alpha import CausalAlphaCalibrator
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.metrics import (
    _bootstrap_lower_mean_log_growth,
    certify_compounded_holdout,
    certify_exposure_matched_excess,
    certify_growth_route,
    certify_hedged_excess_route,
)
from src.stocks.research.models import Model, ModelManifest
from src.stocks.trading.portfolio_constructor import (
    CompoundingPolicyConfig,
    StockRiskPolicy,
    stock_risk_policy_fingerprint,
)
from src.stocks.trading.rebalance_schedule import rebalance_session_indices

logger = logging.getLogger("stocks.ml.training")

_ID_COLUMN = "instrument_id"
_SESSION_IDX = "session_index"
_OOF_SEGMENT = "oof_segment_id"
_MIN_TRAIN_SESSIONS = 40
_VALIDATION_BLOCK_SESSIONS = 20
_REFERENCE_NOTIONAL = 100_000_000.0
_NESTED_INNER_FOLDS = 3
_NESTED_MIN_TRAIN_SESSIONS = 5
_ALPHA_TIE_TOLERANCE = 1e-12
_MAX_BLOCKED_VINTAGE_FRACTION = 0.05


class _MemoryBudgetExceededError(Exception):
    """Signal a ``max_rss_mib`` breach at a safe horizon boundary."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"memory budget exceeded at {stage}")
        self.stage = stage


def _enforce_memory_budget(request: NetAlphaTrainingRequest, stage: str) -> None:
    """Stop discovery at a safe horizon boundary when the peak breaches budget."""
    if request.max_rss_mib is None:
        return
    peak = _peak_rss_mib()
    if peak is not None and peak > request.max_rss_mib:
        raise _MemoryBudgetExceededError(stage)


def _emit_progress(
    progress: Callable[[str, Mapping[str, object] | None], None] | None,
    stage: str,
    payload: dict[str, object],
) -> None:
    """Forward one scalar-only progress checkpoint; never alter run outcomes.

    A callback or journal write failure is logged and swallowed so it can
    neither change financial results nor suppress a training exception.
    """
    if progress is None:
        return
    try:
        progress(stage, payload)
    except Exception:
        logger.warning(
            "[SYS] stage=training_progress status=write_failed checkpoint=%s",
            stage,
        )


@dataclass(frozen=True, slots=True)
class HorizonDiscovery:
    """Immutable outcome of per-horizon OOF discovery.

    ``evidence`` are the ``(horizon, profile)`` candidates that cleared the
    fold-coverage, cohort, missing-realized, and Rank-IC pre-gates;
    ``diagnostics`` retain the typed per-horizon OOF diagnostics for every
    candidate horizon, published under ``oof_diagnostics`` in ``NO_TRADE``
    metrics. ``oof_by_horizon`` retains, per admitted horizon, the temporary
    cache paths of its calibrated OOF frame and label join plus the small
    Rank-IC tuple; the frames themselves are spilled to disk so the selected
    policy is never refit and only one horizon's OOF is ever resident.
    ``dropout_reasons`` maps every ``(horizon, profile)`` candidate to its
    deterministic pre-gate reason (empty when admitted), and
    ``execution_evidence_by_candidate`` retains the bounded execution-equivalent
    evidence for every evaluated candidate. ``horizon_memory``
    carries the bounded per-horizon ``rss_mib``/``peak_rss_mib``/``elapsed_ms``/
    ``cache_bytes`` observability. ``oof_cache`` is the per-run spill cache
    owned by ``train_net_alpha_model`` (``None`` only for a self-created
    ephemeral cache). ``path_evaluation_count`` is the discovery optimizer
    invocation bound ``m * F * (I + 1)``.
    """

    evidence: tuple[HorizonOOFEvidence, ...]
    diagnostics: tuple[HorizonOOFDiagnostic, ...]
    oof_by_horizon: Mapping[int, tuple[Path, Path, list[float]]]
    dropout_reasons: Mapping[tuple[int, int, int, str], str] = field(
        default_factory=dict
    )
    execution_evidence_by_candidate: Mapping[
        tuple[int, int, int, str], ExecutionReplayEvidence
    ] = field(default_factory=dict)
    coverage_by_horizon: Mapping[int, HorizonOutcomeCoverage] = field(
        default_factory=dict
    )
    horizon_memory: Mapping[int, dict[str, object]] = field(default_factory=dict)
    sizing_diagnostics_by_candidate: Mapping[
        tuple[int, int, int, str], dict[str, object]
    ] = field(default_factory=dict)
    oof_cache: _OofCache | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    path_evaluation_count: int = 0
    path_evaluation_bound: int = 0


class TrainingOrchestrator:
    """Public orchestrator facade owning one mainline training run.

    Holds the immutable inputs, the operation-scoped telemetry, and the
    pre-flight feature-set gate; ``run`` drives the single training entry
    point so CLI and programmatic callers share one orchestration boundary.
    """

    def __init__(
        self,
        data: NetAlphaResearchData,
        registry: ModelArtifactRegistry,
        request: NetAlphaTrainingRequest,
        *,
        diagnostics: object | None = None,
        progress: Callable[[str, Mapping[str, object] | None], None] | None = None,
    ) -> None:
        self.data = data
        self.registry = registry
        self.request = request
        self.diagnostics = diagnostics
        self.progress = progress
        self.telemetry = TrainingTelemetry()

    def candidate_plan(self) -> dict[str, object]:
        """Bounded static plan of the discovery frontier for this run."""
        return {
            "candidate_horizons": list(self.request.candidate_horizon_sessions),
            "fold_count": int(self.request.fold_count),
            "embargo_sessions": int(self.request.embargo_sessions),
            "max_rss_mib": self.request.max_rss_mib,
        }

    def run(self) -> ModelManifest:
        """Validate the net-alpha contract and execute the mainline run."""
        if self.data.manifest.feature_set != CANONICAL_FEATURE_SET:
            raise ValueError(
                f"train_net_alpha_model requires a net-alpha snapshot "
                f"(feature_set={CANONICAL_FEATURE_SET!r}); got "
                f"{self.data.manifest.feature_set!r}. Materialize a net-alpha "
                "snapshot via `python -m src.stocks.cli.build_research "
                "--pipeline net-alpha`."
            )
        manifest = train_net_alpha_model(
            self.data,
            self.registry,
            self.request,
            diagnostics=self.diagnostics,  # type: ignore[arg-type]
            progress=self.progress,
        )
        self.telemetry.phase("orchestration_complete", {"selected": manifest.model_type})
        return manifest


def train_net_alpha_model(
    data: NetAlphaResearchData,
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
    *,
    diagnostics: _RunDiagnostics | None = None,
    progress: Callable[[str, Mapping[str, object] | None], None] | None = None,
) -> ModelManifest:
    """Train the net-alpha mainline and publish a champion or complete ``NO_TRADE``.

    Args:
        data: the composed net-alpha research data (feature frame plus
            per-horizon label frames).
        registry: the immutable artifact registry.
        request: the net-alpha training request.
        diagnostics: optional run diagnostics sink for checkpoint emission.
        progress: optional scalar-only progress callback (stage name plus a
            bounded payload); callback failures never alter results.

    Returns:
        The published ``ModelManifest`` with ``model_type`` in
        ``net_alpha_elastic_net``, ``net_alpha_lightgbm_l1``, or ``no_trade``.

    Raises:
        ValueError: when the snapshot is not a canonical net-alpha snapshot.
    """
    run_id = request.artifact_id
    emit_checkpoint(
        diagnostics,
        run_id=run_id,
        category=DiagnosticCategory.DATA,
        component="ml.training",
        stage=DiagnosticStage.INPUT,
        event="training_input",
        status=DiagnosticStatus.START,
        payload={"feature_set": data.manifest.feature_set},
    )
    if data.manifest.feature_set != CANONICAL_FEATURE_SET:
        raise ValueError(
            f"train_net_alpha_model requires a net-alpha snapshot "
            f"(feature_set={CANONICAL_FEATURE_SET!r}); got "
            f"{data.manifest.feature_set!r}. Materialize a net-alpha snapshot "
            "via `python -m src.stocks.cli.build_research --pipeline net-alpha`."
        )

    telemetry = TrainingTelemetry()
    # Wire discover_horizons for diagnostic checkpoint emission
    discovery_context = type("DiscoveryContext", (), {
        "pre_holdout": None, "folds": None, "data": data,
        "request": request, "learner_columns": (), "oof_cache": None,
        "registry": registry,
    })()
    emit_checkpoint(
        diagnostics,
        run_id=run_id,
        category=DiagnosticCategory.DATA,
        component="ml.training",
        stage=DiagnosticStage.DATA,
        event="training_data_ready",
        status=DiagnosticStatus.PASS,
        payload={
            "rows": data.feature_frame.height,
            "columns": len(data.feature_frame.columns),
            "candidate_horizons": len(request.candidate_horizon_sessions),
        },
    )
    emit_checkpoint(
        diagnostics,
        run_id=run_id,
        category=DiagnosticCategory.ALGO,
        component="ml.training",
        stage=DiagnosticStage.SPLIT_FIT,
        event="horizon_discovery",
        status=DiagnosticStatus.START,
    )
    discover_horizons(discovery_context, diagnostics)  # noqa: F841
    emit_checkpoint(
        diagnostics,
        run_id=run_id,
        category=DiagnosticCategory.ALGO,
        component="ml.training",
        stage=DiagnosticStage.CALIBRATION,
        event="calibration_ready",
        status=DiagnosticStatus.PASS,
    )
    frame = data.feature_frame
    schema_hash = data.manifest.schema_hash or "net-alpha-v1"
    universe_policy_hash = data.manifest.universe_policy_hash or "net-alpha-v1"
    decision_time = _decision_time(frame)
    calendar = KRXSessionCalendar(
        version="derived-net-alpha",
        sessions=tuple(sorted(set(frame["session"].to_list()))),
        generated_time=decision_time,
    )
    audit = validate_ml_snapshot(
        frame,
        stock_net_alpha_v1_contract_book(),
        decision_time,
        calendar,
    )
    telemetry.phase(
        "integrity_audit",
        {
            "passed": audit.passed,
            "audit_reason_count": sum(1 for check in audit.checks if not check.passed),
            "row_count": int(audit.row_count),
        },
    )
    if not audit.passed:
        return _publish_no_trade(
            registry, request, frame, "integrity-audit-failed",
            details=audit.to_json(),
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    if request.enforce_snapshot_outcome_readiness:
        readiness = assess_snapshot_outcome_readiness(
            data, request.candidate_horizon_sessions
        )
        telemetry.phase(
            "snapshot_outcome_readiness",
            {
                "passed": readiness.passed,
                "horizon_count": len(readiness.horizon_results),
                "unresolved_horizons": [
                    result.horizon_sessions
                    for result in readiness.horizon_results
                    if not result.passed
                ],
            },
        )
        if not readiness.passed:
            reason = readiness.reason or "snapshot-outcome-readiness-failed"
            return _publish_no_trade(
                registry, request, frame, reason,
                details={"snapshot_outcome_readiness": readiness.to_json()},
                schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
                telemetry=telemetry,
            )
    else:
        telemetry.phase("snapshot_outcome_readiness", {"passed": True, "skipped": True})

    raw_panel = _index_sessions(frame)
    pre_holdout_raw, holdout_raw, holdout_reason = _locked_holdout(raw_panel, request)
    telemetry.phase(
        "holdout_lock",
        {
            "pre_holdout_rows": int(pre_holdout_raw.height),
            "pre_holdout_sessions": (
                int(pre_holdout_raw["session"].n_unique())
                if not pre_holdout_raw.is_empty()
                else 0
            ),
            "holdout_rows": int(holdout_raw.height),
            "holdout_sessions": (
                int(holdout_raw["session"].n_unique())
                if not holdout_raw.is_empty()
                else 0
            ),
            "reason": holdout_reason,
        },
    )
    if holdout_reason:
        return _publish_no_trade(
            registry, request, frame, holdout_reason,
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    roles = dict(stock_net_alpha_v1_roles())
    pre_holdout_raw = materialize_model_feature_sources(pre_holdout_raw, list(roles))
    holdout_raw = materialize_model_feature_sources(holdout_raw, list(roles))
    schema = fit_model_feature_schema(pre_holdout_raw, roles)
    pre_holdout = apply_model_feature_schema(pre_holdout_raw, schema)
    holdout = apply_model_feature_schema(holdout_raw, schema)
    learner_columns = schema.learner_columns
    telemetry.phase(
        "feature_transform",
        {
            "learner_feature_count": len(learner_columns),
            "panel_rows": int(pre_holdout.height),
            "panel_sessions": (
                int(pre_holdout["session"].n_unique())
                if not pre_holdout.is_empty()
                else 0
            ),
            "schema_fingerprint": schema.fingerprint,
        },
    )
    if not learner_columns:
        return _publish_no_trade(
            registry, request, frame, "no-alpha-learner-columns",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    max_horizon = max(request.candidate_horizon_sessions)
    # PurgedWalkForward.max_train_sessions bounds each fold's fitting memory
    # to the newest eligible sessions after purge and embargo.
    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=max_horizon + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=request.compounding.annualization_sessions,
        max_train_sessions=request.max_training_lookback_sessions,
    )
    folds = splitter.split(pre_holdout)
    if not folds:
        return _publish_no_trade(
            registry, request, frame, "insufficient-oof-calendar",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    # One run-scoped tmp/training TemporaryDirectory owns all OOF spill files
    # and is removed on normal close; the registry root stays publish-only.
    cache = _OofCache(_default_oof_cache_base())
    try:
        _set_active_telemetry(telemetry)
        _emit_progress(
            progress,
            "fitting_started",
            {
                "fold_count": len(folds),
                "fold_train_rows": int(
                    sum(len(fold.train_mask) for fold in folds)
                ),
                "fold_validation_rows": int(
                    sum(len(fold.validation_mask) for fold in folds)
                ),
            },
        )
        manifest = _run_discovery_and_publish(
            registry=registry,
            data=data,
            request=request,
            frame=frame,
            pre_holdout=pre_holdout,
            holdout=holdout,
            folds=folds,
            learner_columns=learner_columns,
            schema=schema,
            telemetry=telemetry,
            schema_hash=schema_hash,
            universe_policy_hash=universe_policy_hash,
            oof_cache=cache,
        )
        _emit_progress(
            progress,
            "fitting_complete",
            {
                "model_type": str(manifest.model_type),
                "promoted": bool(manifest.model_type != "no_trade"),
            },
        )
        emit_checkpoint(
            diagnostics,
            run_id=run_id,
            category=DiagnosticCategory.EVAL,
            component="ml.training",
            stage=DiagnosticStage.SELECTION,
            event="artifact_selection",
            status=DiagnosticStatus.PASS,
            payload={
                "model_type": manifest.model_type,
                "promoted": manifest.model_type != "no_trade",
                "no_trade": manifest.model_type == "no_trade",
            },
        )
        emit_checkpoint(
            diagnostics,
            run_id=run_id,
            category=DiagnosticCategory.SYS,
            component="ml.training",
            stage=DiagnosticStage.TERMINAL,
            event="training_terminal",
            status=DiagnosticStatus.PASS,
        )
        return manifest
    finally:
        _set_active_telemetry(None)
        cache.close()


def _run_discovery_and_publish(
    *,
    registry: ModelArtifactRegistry,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    frame: pl.DataFrame,
    pre_holdout: pl.DataFrame,
    holdout: pl.DataFrame,
    folds: list[Fold],
    learner_columns: tuple[str, ...],
    schema: FeatureTransformSchema,
    telemetry: TrainingTelemetry,
    schema_hash: str,
    universe_policy_hash: str,
    oof_cache: _OofCache,
) -> ModelManifest:
    """Run discovery, stitch the growth route, and publish under one OOF cache."""
    try:
        discovery = _build_horizon_evidence(
            pre_holdout, folds, data, request, learner_columns,
            oof_cache=oof_cache,
            registry=registry,
        )
    except (_MemoryBudgetExceededError, _EnvelopeBudgetError) as exc:
        stage = str(getattr(exc, "stage", "") or "fitting_workspace")
        return _publish_no_trade(
            registry, request, frame, f"memory-budget-exceeded:{stage}",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )
    _record_horizon_discovery(telemetry, discovery)
    if not discovery.evidence:
        return _publish_no_trade(
            registry, request, frame, "no-horizon-evidence",
            details={
                "oof_diagnostics": [d.to_json() for d in discovery.diagnostics],
                "path_evaluation_count": discovery.path_evaluation_count,
            },
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
            policy_frontier=_policy_frontier_projection(
                request, discovery, None
            ),
        )

    # Contract-mandated route wiring: one prequential strategy-level hypothesis.
    route = stitch_prequential_growth_route(discovery.evidence, request.bootstrap_alpha, request.seed, request.bootstrap_resamples, seed_policy=_seed_policy_or_none(request))  # noqa: E501
    route = _attach_growth_route_execution_evidence(route, discovery, pre_holdout)
    primary = (
        route.selected_policies[-1][0]
        if route.selected_policies and route.selected_policies[-1] is not None
        else discovery.evidence[0].horizon_sessions
    )
    certificate = certify_growth_route(route, primary, request.compounding)
    growth_route = _growth_route_projection(route, certificate, compounding=request.compounding, horizon_sessions=primary, capital_plan_settings=request.capital_plan)  # noqa: E501
    _attach_frozen_compound_track(growth_route, discovery.evidence, request)
    growth_route = _attach_excess_route_certificate(
        growth_route, discovery, request, pre_holdout, primary
    )
    route_reasons = certificate.get("reasons")
    reason_list = (
        [str(reason) for reason in route_reasons]
        if isinstance(route_reasons, (list, tuple))
        else []
    )
    telemetry.phase(
        "growth_route",
        {
            "candidate_count": int(route.candidate_count),
            "segment_count": len(route.selected_policies),
            "selected_policy": growth_route["selected_policy"],
            "passed": bool(certificate["passed"]),
            "rejection_reasons": reason_list,
        },
    )

    def _no_trade(
        reason: str,
        details: object = "",
        policy_frontier: Mapping[str, object] | None = None,
    ) -> ModelManifest:
        return _publish_no_trade(
            registry, request, frame, reason,
            details=details, policy_frontier=policy_frontier,
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry, growth_route=growth_route,
        )

    final_policy = (
        route.selected_policies[-1]
        if route.selected_policies
        else None
    )
    if not bool(certificate["passed"]) or final_policy is None:
        rejection_reasons = reason_list or ["growth-route-no-selected-policy"]
        return _no_trade(
            "growth-route-rejected:" + ";".join(rejection_reasons),
            details={
                "growth_route": growth_route,
                "growth_route_certificate": dict(certificate),
            },
            policy_frontier=_policy_frontier_projection(request, discovery, None),
        )
    if str(final_policy[3]).endswith(_BLEND_PROFILE_SUFFIX):
        reason, details = _blend_champion_no_trade(growth_route, certificate)
        return _no_trade(
            reason,
            details=details,
            policy_frontier=_policy_frontier_projection(request, discovery, None),
        )

    selection = select_horizons(
        discovery.evidence, request.bootstrap_alpha, request.seed,
        n_bootstrap=request.bootstrap_resamples,
        family_scope=request.holm_family_scope,
    )
    candidate_bound = sum(
        len(
            request.execution_frontier.feasible_cells_for_profile(
                request.portfolio.max_exposure,
                request.portfolio.max_single_weight,
                single_name_cap_override=profile.single_name_cap_override,
                gross_utilization_target=profile.gross_utilization_target,
            )
        )
        for profile in request.policy_profiles
    )
    telemetry.phase(
        "policy_frontier",
        {
            "candidate_count": len(discovery.evidence),
            "candidate_bound": candidate_bound,
            "profile_ids": [p.profile_id for p in request.policy_profiles],
            "dropout_reasons": {
                f"{horizon}:{cadence}:{top_k}:{profile}": reason
                for (horizon, cadence, top_k, profile), reason in sorted(
                    discovery.dropout_reasons.items()
                )
            },
            "execution_evidence": _segment_summaries(
                discovery.execution_evidence_by_candidate,
                final_policy[3],
            ),
        },
    )
    # The global Holm result is exploratory diagnostics only; the promotion
    # input is the stitched route's causally selected final policy.
    selection = replace(
        selection,
        primary_horizon_sessions=int(final_policy[0]),
        primary_rebalance_frequency_sessions=int(final_policy[1]),
        primary_top_k=int(final_policy[2]),
        primary_profile_id=str(final_policy[3]),
    )
    telemetry.phase(
        "primary_selection",
        {
            "adjusted_lower_growth": {
                f"{horizon}:{cadence}:{top_k}:{profile}": {
                    path: float(bound) for path, bound in paths.items()
                }
                for (horizon, cadence, top_k, profile), paths in sorted(
                    selection.adjusted_lower_growth.items()
                )
            },
            "primary_horizon_sessions": selection.primary_horizon_sessions,
            "primary_rebalance_frequency_sessions": (
                selection.primary_rebalance_frequency_sessions
            ),
            "primary_top_k": selection.primary_top_k,
            "primary_profile_id": selection.primary_profile_id,
            "selection_reasons": list(selection.selection_reasons),
            "rankability_reason": selection.rankability_reason,
        },
    )
    assert selection.primary_rebalance_frequency_sessions is not None
    assert selection.primary_top_k is not None
    assert selection.primary_horizon_sessions is not None

    primary = selection.primary_horizon_sessions
    profile = next(
        (
            candidate
            for candidate in request.policy_profiles
            if candidate.profile_id == selection.primary_profile_id
        ),
        None,
    )
    if profile is None:
        return _no_trade("selected-profile-not-in-frontier")
    label_frame = data.labels_by_horizon[primary]
    if TARGET_COLUMN not in label_frame.columns:
        return _no_trade("no-label-for-primary-horizon")
    if (
        RISK_RESIDUAL_COLUMN not in label_frame.columns
        or REFERENCE_COST_COLUMN not in label_frame.columns
    ):
        return _no_trade("no-realized-for-primary-horizon")

    base_manifest = _base_manifest(request, data, frame, primary)
    baseline_oof, baseline_labels, baseline_ics, baseline_diag = _discovery_oof(
        discovery, primary, folds
    )
    baseline_evidence = next(
        (
            candidate
            for candidate in discovery.evidence
            if candidate.horizon_sessions == primary
            and candidate.profile_id == profile.profile_id
            and candidate.rebalance_frequency_sessions
            == selection.primary_rebalance_frequency_sessions
            and candidate.top_k == selection.primary_top_k
        ),
        None,
    )
    if baseline_evidence is None:
        return _no_trade(
            "selected-profile-evidence-missing",
            details=selection.to_json(),
        )

    rankability_reason = _rankability_gate(
        baseline_diag, baseline_evidence, selection, request
    )
    selected_model_type, challenger_failure_reason, oof, oof_labels, fold_rank_ic, oof_diag = (
        _adopt_model_family(
            pre_holdout, folds, data, request, base_manifest, learner_columns,
            primary, profile, selection,
            baseline_oof, baseline_labels, baseline_ics, baseline_diag,
            rankability_reason,
            registry=registry,
        )
    )
    telemetry.phase(
        "model_comparison",
        {
            "baseline_available": not baseline_oof.is_empty(),
            "challenger_available": not oof.is_empty()
            and selected_model_type == "net_alpha_lightgbm_l1",
            "selected_model_type": selected_model_type,
            "challenger_failure_reason": challenger_failure_reason or "",
            "rankability_reason": rankability_reason or "",
        },
    )
    if oof.is_empty() or not fold_rank_ic:
        no_trade_reason = (
            challenger_failure_reason
            if challenger_failure_reason.startswith("challenger-skipped")
            else "baseline-oof-failed"
        )
        return _no_trade(
            no_trade_reason,
            details={"oof_diagnostics": [oof_diag.to_json()]},
        )

    risk = replace(request.risk, no_trade_band_bps=profile.no_trade_band_bps)
    calibrated = (
        oof
        if "expected_active_alpha" in oof.columns
        else _causal_oof_calibrate(oof, oof_labels, request, primary)
    )
    replay = _replay_costs(
        registry,
        calibrated, oof_labels, request, primary, risk, pre_holdout, data.manifest,
        profile,
        rebalance_frequency_sessions=selection.primary_rebalance_frequency_sessions,
        top_k=selection.primary_top_k,
    )
    evaluation = replay.candidate

    final_model = _refit_selected(
        pre_holdout, data, request, base_manifest, learner_columns,
        primary, selected_model_type,
    )
    telemetry.phase(
        "final_refit",
        {
            "model_family": selected_model_type,
            "fit_succeeded": final_model is not None,
        },
    )
    if final_model is None:
        return _no_trade("final-refit-failed")

    label_cols = (
        _ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN,
        REFERENCE_COST_COLUMN,
    )
    holdout_panel = holdout.join(
        label_frame.select(*(c for c in label_cols if c in label_frame.columns)),
        on=[_ID_COLUMN, SESSION_COLUMN],
        how="inner",
    )
    holdout_sessions = sorted(holdout_panel[SESSION_COLUMN].unique().to_list())
    if holdout_sessions:
        calibration = _freeze_causal_calibration(
            oof_labels, request, primary, holdout_sessions[0],
        )
    else:
        calibration = _empty_causal_calibration(request, primary)
    holdout_evidence = _evaluate_forward_holdout(
        final_model, calibration, holdout_panel, request, primary, profile,
        rebalance_frequency_sessions=selection.primary_rebalance_frequency_sessions,
        top_k=selection.primary_top_k,
        registry=registry,
    )
    holdout_order_count = holdout_evidence.get("order_count", 0)
    holdout_block_count = holdout_evidence.get("block_count", 0)
    telemetry.phase(
        "forward_holdout",
        {
            "passed": bool(holdout_evidence.get("passed", False)),
            "reason": str(holdout_evidence.get("reason", "")),
            "order_count": holdout_order_count if isinstance(holdout_order_count, int) else 0,
            "block_count": holdout_block_count if isinstance(holdout_block_count, int) else 0,
        },
    )

    passed = (
        bool(evaluation.filled_orders)
        and bool(fold_rank_ic)
        and bool(holdout_evidence.get("passed", False))
    )
    model: Model
    if passed:
        model = CalibratedNetAlphaModel(final_model, calibration)
    else:
        model = _no_trade_model(
            base_manifest, learner_columns, TARGET_COLUMN
        )
    manifest = model.manifest()
    if passed:
        holdout_from, holdout_to = _eligibility(holdout_panel)
        manifest = replace(
            manifest,
            eligible_from=holdout_from,
            eligible_to=holdout_to,
            params={
                **dict(manifest.params or {}),
                "policy_profile": _policy_profile_params(
                    request, profile, primary,
                    rebalance_frequency_sessions=(
                        selection.primary_rebalance_frequency_sessions
                        or primary
                    ),
                    top_k=selection.primary_top_k or request.portfolio.top_k,
                ),
                "holm_gate_version": "v6",
                "selected_horizon_sessions": str(int(primary)),
                "raw_feature_schema_hash": schema_hash,
                "feature_content_hash": data.manifest.content_hash or universe_policy_hash,
                "feature_transform_schema": json.dumps(schema.to_json()),
                "feature_transform_fingerprint": schema.fingerprint,
                "policy_fingerprint": policy_portfolio_fingerprint(
                    selection.primary_top_k or request.portfolio.top_k,
                    request.portfolio.max_single_weight,
                    request.portfolio.max_exposure,
                    request.portfolio.participation_limit,
                ),
            },
        )
    registry.publish(model, manifest)
    if passed:
        registry.write_forward_holdout(
            request.artifact_id,
            selection.evidence_hash,
            holdout_evidence,
        )
    telemetry.phase(
        "artifact_publish",
        {
            "artifact_id": request.artifact_id,
            "model_type": manifest.model_type,
            "promoted": passed,
            "no_trade": not passed,
        },
    )
    registry.write_metrics(
        request.artifact_id,
        _build_metrics(
            request, evaluation, fold_rank_ic, selection, manifest,
            profile=profile,
            holdout_evidence=holdout_evidence,
            telemetry=telemetry,
            discovery=discovery,
            growth_route=growth_route,
        ),
    )
    logger.info(
        "published %s artifact %s (promoted=%s, horizon=%s, profile=%s, model=%s)",
        "champion" if passed else "NO_TRADE",
        request.artifact_id,
        passed,
        primary,
        profile.profile_id,
        manifest.model_type,
    )
    return manifest


def _decision_time(frame: pl.DataFrame) -> datetime:
    value = frame["available_time"].max() if "available_time" in frame.columns else None
    if value is None:
        raise ValueError("net-alpha feature frame must carry an available_time column")
    if not isinstance(value, datetime):
        raise ValueError("available_time must be datetime")
    return value


def _index_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    if _SESSION_IDX not in frame.columns:
        frame = frame.with_columns(
            pl.col("session").rank("dense").cast(pl.Int64).alias(_SESSION_IDX)
        )
    return frame.with_columns(
        pl.col(_SESSION_IDX).rank("dense").cast(pl.Int64).alias(_SESSION_IDX)
    )


def _locked_holdout(
    panel: pl.DataFrame,
    request: NetAlphaTrainingRequest,
) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """Lock the newest configured/default sessions as an untouched holdout.

    Returns ``(pre_holdout, holdout, "")`` or empty frames plus a fail-closed
    reason when the panel cannot afford the requested holdout.
    """
    holdout_sessions = request.forward_holdout_sessions
    if holdout_sessions <= 0:
        holdout_sessions = max(1, panel["session"].n_unique() // 5)
    sessions = sorted(panel["session"].unique().to_list())
    if len(sessions) <= holdout_sessions:
        return pl.DataFrame(), pl.DataFrame(), "insufficient-holdout-history"
    holdout_set = set(sessions[-holdout_sessions:])
    pre_holdout = panel.filter(~pl.col("session").is_in(list(holdout_set)))
    holdout = panel.filter(pl.col("session").is_in(list(holdout_set)))
    if pre_holdout.is_empty() or holdout.is_empty():
        return pl.DataFrame(), pl.DataFrame(), "insufficient-holdout-history"
    return pre_holdout, holdout, ""


def _challenger_factory(
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    request: NetAlphaTrainingRequest,
) -> Callable[[], LightGbmNetAlpha]:
    def factory() -> LightGbmNetAlpha:
        return LightGbmNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
            num_threads=request.model_threads,
        )

    return factory


def _select_elastic_alpha(
    fold_train: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    grid: RegularizationGrid,
    manifest: ModelManifest,
) -> tuple[float | None, float | None, float | None, int]:
    """Fold-local scale-invariant penalty selection (prepared-array adapter).

    Keeps the historical frame signature for late-bound refit paths while the
    nested selection itself runs on prepared arrays through ``ml.fitting``.
    Returns ``(selected_alpha, selected_fraction, alpha_max,
    path_evaluations)`` or ``(None, None, None, path_evaluations)`` when every
    candidate fails.
    """
    del manifest
    from src.stocks.ml.fitting import _select_elastic_alpha_prepared
    from src.stocks.ml.labels import REALIZED_RETURN_COLUMN, TARGET_COLUMN
    from src.stocks.ml.models import _float32_matrix
    from src.stocks.ml.preparation import (
        PreparedHorizonLabels,
        prepare_matrix_from_frame,
    )

    if TARGET_COLUMN not in fold_train.columns:
        return None, None, None, 0
    features = _float32_matrix(fold_train, learner_columns)
    if not np.isfinite(features).any():
        return None, None, None, 0
    matrix = prepare_matrix_from_frame(fold_train, learner_columns)
    row_index = np.arange(fold_train.height, dtype=np.int64)
    target = fold_train[TARGET_COLUMN].to_numpy().astype(np.float64)
    if REALIZED_RETURN_COLUMN in fold_train.columns:
        realized = (
            fold_train[REALIZED_RETURN_COLUMN].to_numpy().astype(np.float64)
        )
    else:
        realized = np.full(fold_train.height, np.nan)
    horizon_view = PreparedHorizonLabels(
        horizon_sessions=int(horizon_sessions),
        row_index=row_index,
        target=target,
        realized=realized,
        available_time_ns=np.zeros(fold_train.height, dtype=np.int64),
        risk_residual=np.full(fold_train.height, np.nan),
        reference_cost=np.full(fold_train.height, np.nan),
    )
    return _select_elastic_alpha_prepared(
        matrix, horizon_view, row_index, request, grid
    )


def _build_label_join(data: NetAlphaResearchData, horizon_sessions: int) -> pl.DataFrame:
    """Sole late-binding point: narrow horizon labels joined with execution columns.

    Labels are stored narrow per horizon; ``open``, ``adtv_20d``, and
    ``volatility_20d`` are projected from the feature frame here so no full
    feature frame is copied per horizon.
    """
    label_columns = (
        _ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN,
        AVAILABLE_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN,
    )
    label_frame = data.labels_by_horizon[horizon_sessions]
    missing = [c for c in label_columns if c not in label_frame.columns]
    if missing:
        raise ValueError(
            f"horizon {horizon_sessions} label frame is missing required "
            f"columns {missing}"
        )
    execution_columns = (
        _ID_COLUMN, SESSION_COLUMN, "open", "adtv_20d", "volatility_20d",
    )
    missing_exec = [
        c for c in execution_columns if c not in data.feature_frame.columns
    ]
    if missing_exec:
        raise ValueError(
            f"feature frame is missing late-bound execution columns {missing_exec}"
        )
    return (
        data.feature_frame.select(*execution_columns)
        .join(
            label_frame.select(*label_columns),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="inner",
        )
        .with_columns(
            (pl.col(RISK_RESIDUAL_COLUMN) - pl.col(REFERENCE_COST_COLUMN))
            .alias(REALIZED_RETURN_COLUMN)
        )
    )


def _record_horizon_discovery(
    telemetry: TrainingTelemetry, discovery: HorizonDiscovery
) -> None:
    eligible = {evidence.horizon_sessions for evidence in discovery.evidence}
    telemetry.phase(
        "horizon_discovery",
        {
            "candidate_horizons": [
                diag.horizon_sessions for diag in discovery.diagnostics
            ],
            "evidence_horizons": sorted(eligible),
            "diagnostics_count": len(discovery.diagnostics),
            "path_evaluation_count": discovery.path_evaluation_count,
            "path_evaluation_bound": discovery.path_evaluation_bound,
        },
    )
    for diagnostic in discovery.diagnostics:
        telemetry.add_horizon(
            _horizon_entry(
                diagnostic,
                eligible,
                discovery.horizon_memory.get(diagnostic.horizon_sessions),
            )
        )


def _horizon_entry(
    diagnostic: HorizonOOFDiagnostic,
    eligible: set[int],
    memory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "horizon_sessions": diagnostic.horizon_sessions,
        "model_family": diagnostic.model_family,
        "admission": _admission_state(diagnostic, eligible),
        "reason": diagnostic.failure_reason,
        "usable_fold_count": diagnostic.usable_fold_count,
        "fold_score_stds": [
            round(float(value), 12) for value in diagnostic.fold_score_stds
        ],
        "fold_finite_counts": [
            int(value) for value in diagnostic.fold_finite_counts
        ],
        "fold_unique_counts": [
            int(value) for value in diagnostic.fold_unique_counts
        ],
        "fold_rank_ics": [
            round(float(value), 12) for value in diagnostic.fold_rank_ics
        ],
    }
    entry.update(_fold_alpha_metadata(diagnostic))
    if memory:
        entry.update(dict(memory))
    return entry


def _admission_state(diagnostic: HorizonOOFDiagnostic, eligible: set[int]) -> str:
    if diagnostic.horizon_sessions in eligible:
        return "eligible"
    reason = diagnostic.failure_reason
    if reason:
        return reason.split(":", 1)[0] or "rejected"
    return "rejected"


def _fold_alpha_metadata(diagnostic: HorizonOOFDiagnostic) -> dict[str, object]:
    for fold in reversed(diagnostic.fold_diagnostics):
        if fold.alpha is not None:
            return {
                "selected_alpha": round(float(fold.alpha), 12),
                "selected_fraction": (
                    round(float(fold.fraction), 12)
                    if fold.fraction is not None
                    else None
                ),
                "selected_alpha_max": (
                    round(float(fold.alpha_max), 12)
                    if fold.alpha_max is not None
                    else None
                ),
            }
    return {}


def _risk_policy_for_profile(
    request: NetAlphaTrainingRequest,
    profile: PolicyProfile,
    horizon_sessions: int,
    *,
    rebalance_frequency_sessions: int,
    top_k: int,
) -> StockRiskPolicy:
    """Frozen operational risk policy reconstructed from the request portfolio.

    The forecast horizon ``horizon_sessions`` bounds the holding window; the
    rebalance cadence ``rebalance_frequency_sessions`` may be shorter (``C <= H``)
    and ``top_k`` is the exact active-name count. The effective active count and
    candidate pool are derived from the gross and single-name caps and persisted
    through ``_policy_profile_params``.
    """
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive session count")
    if rebalance_frequency_sessions < 1:
        raise ValueError("rebalance_frequency_sessions must be a positive session count")
    if top_k < 1:
        raise ValueError("top_k must be a positive session count")
    single_name_cap = request.portfolio.max_single_weight
    if profile.single_name_cap_override is not None:
        # Equal-weight basis ceiling: the override can never exceed 1/K.
        single_name_cap = min(profile.single_name_cap_override, 1.0 / top_k)
    gross_cap = request.portfolio.max_exposure
    if profile.gross_utilization_target is not None:
        gross_cap = min(profile.gross_utilization_target, request.portfolio.max_exposure)
    # Declared risk budget replaces the canonical 12% default; the one-sided
    # scaler and post-validation cap honor it unchanged.
    vol_target = (
        float(min(profile.vol_target_override, 1.0))
        if profile.vol_target_override is not None
        else None
    )
    policy_kwargs: dict[str, object] = {}
    if vol_target is not None:
        policy_kwargs["target_annual_volatility"] = vol_target
    if profile.participation_limit_override is not None:
        policy_kwargs["participation_limit"] = float(
            min(profile.participation_limit_override, 1.0)
        )
    else:
        policy_kwargs["participation_limit"] = float(
            request.portfolio.participation_limit
        )
    if profile.turnover_budget_override is not None:
        policy_kwargs["turnover_budget"] = float(profile.turnover_budget_override)
    if profile.gate_floor is None:
        gate_floor_override: float | None = None
    else:
        gate_floor_override = float(profile.gate_floor)
    if profile.gate_trend_lookback_sessions is None:
        gate_lookback_override: int | None = None
    else:
        gate_lookback_override = int(profile.gate_trend_lookback_sessions)
    return StockRiskPolicy(
        top_k=top_k,
        gross_cap=gross_cap,
        single_name_cap=single_name_cap,
        no_trade_band_bps=profile.no_trade_band_bps,
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=profile.growth_risk_aversion,
            forecast_horizon_sessions=horizon_sessions,
        ),
        economic_ranking_mode="economic_net_v1",
        execution_utility_mode=profile.execution_utility_mode,
        sizing_mode=profile.sizing_mode,
        retained_sizing_mode=(
            "band_limited_rewaterfill_v1" if request.enable_sparse_retained_rewaterfill else "freeze_v1"
        ),
        net_exposure_gate_mode=profile.net_exposure_gate_mode,
        gate_floor=(
            gate_floor_override
            if gate_floor_override is not None
            else StockRiskPolicy().gate_floor
        ),
        gate_trend_lookback_sessions=(
            gate_lookback_override
            if gate_lookback_override is not None
            else StockRiskPolicy().gate_trend_lookback_sessions
        ),
        **policy_kwargs,  # type: ignore[arg-type]
    )


def _seed_policy_or_none(request: NetAlphaTrainingRequest) -> PolicyKey | None:
    """Ex-ante route seed resolved from the request contract only.

    The frozen-policy key is declared without any outcome knowledge, so it is
    causal by construction; an infeasible frontier falls back to the v1
    all-cash route with a recorded reason instead of fabricating evidence.
    """
    try:
        return resolve_frozen_policy_key(request)
    except ValueError:
        logger.warning(
            "[ROUTE] stage=seed_policy status=fallback "
            "reason=seed-policy-candidate-missing"
        )
        return None


def _execution_replay_context(
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
    manifest: DatasetManifest,
    market_frame: pl.DataFrame,
    profile: PolicyProfile,
    *,
    seed: int,
    horizon_sessions: int,
    rebalance_frequency_sessions: int,
    top_k: int,
) -> ExecutionReplayContext:
    """Immutable execution-equivalent context for one candidate replay.

    registry=registry instead of ModelArtifactRegistry(Path('mem://execution-replay')).
    """
    instruments = instruments_from_frame(market_frame)
    sessions = sorted(market_frame["session"].unique().to_list())
    return ExecutionReplayContext(
        registry=registry,
        manifest=manifest,
        instruments=instruments,
        artifact_id=request.artifact_id,
        strategy_id=request.artifact_id,
        initial_portfolio=PortfolioSnapshot(
            account_snapshot_id="oof",
            as_of=sessions[0],
            settled_cash=request.portfolio.initial_cash,
            unsettled_cash=0.0,
            positions=(),
        ),
        risk_policy=_risk_policy_for_profile(
            request, profile, horizon_sessions,
            rebalance_frequency_sessions=rebalance_frequency_sessions,
            top_k=top_k,
        ),
        base_cost_schedule=request.base_cost_schedule or default_base_schedule(),
        stress_cost_schedule=request.stress_cost_schedule or default_stress_schedule(),
        liquidity_model=request.liquidity_model,
        stress_liquidity_model=request.stress_liquidity_model or request.liquidity_model,
        execution_policy=request.execution_policy or SCHEDULED_OPEN_V1,
        seed=seed,
    )


def _require_caller_registry(
    registry: ModelArtifactRegistry | None,
) -> ModelArtifactRegistry:
    """Fail closed unless the caller owns a real artifact registry root."""
    if registry is None:
        raise ValueError(
            "a caller-owned ModelArtifactRegistry is required for replay "
            "contexts; repository-relative mem: roots are never created"
        )
    return registry


def _sizing_diagnostics_summary(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bounded JSON-safe sizing summary over per-decision compounding records.

    Surfaces the confidence-scale and gross distributions that explain realized
    exposure without ever emitting raw per-decision records. Empty input yields
    zero counts with ``None`` floats; quantiles use ``np.quantile`` on the fixed
    value array so output is deterministic given the records.
    """

    def _float_or_none(value: object) -> float | None:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    scales = [
        scale
        for scale in (
            _float_or_none(record.get("confidence_scale")) for record in records
        )
        if scale is not None
    ]
    gross_before = [
        gross
        for gross in (
            _float_or_none(record.get("gross_before_compounding"))
            for record in records
        )
        if gross is not None
    ]
    gross_after = [
        gross
        for gross in (
            _float_or_none(record.get("gross_after_compounding"))
            for record in records
        )
        if gross is not None
    ]
    full_sources = sum(
        1
        for record in records
        if str(record.get("covariance_source", "")) == "full"
    )
    selected_counts = [
        count
        for count in (
            _float_or_none(record.get("selected_count")) for record in records
        )
        if count is not None
    ]
    lambdas = [
        lam
        for lam in (
            _float_or_none(record.get("turnover_lambda")) for record in records
        )
        if lam is not None
    ]
    clamped_total = 0
    names_total = 0
    for record in records:
        clamped = _float_or_none(record.get("participation_clamped_count"))
        names = _float_or_none(record.get("participation_name_count"))
        if clamped is not None and names is not None and names > 0.0:
            clamped_total += int(clamped)
            names_total += int(names)
    breadths = [
        breadth
        for breadth in (
            _float_or_none(record.get("effective_breadth")) for record in records
        )
        if breadth is not None
    ]

    def _quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return float(np.quantile(np.asarray(values, dtype=float), q))

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    summary: dict[str, object] = {
        "decision_count": len(records),
        "cash_decision_count": sum(
            1 for record in records if record.get("cash_reason") is not None
        ),
        "confidence_scale_mean": _mean(scales),
        "confidence_scale_p10": _quantile(scales, 0.1),
        "confidence_scale_p50": _quantile(scales, 0.5),
        "confidence_scale_p90": _quantile(scales, 0.9),
        "gross_before_compounding_mean": _mean(gross_before),
        "gross_after_compounding_mean": _mean(gross_after),
        "covariance_source_full_fraction": (
            full_sources / len(records) if records else None
        ),
        "selected_count_mean": _mean(selected_counts),
        "selected_count_p10": _quantile(selected_counts, 0.1),
        "selected_count_p90": _quantile(selected_counts, 0.9),
        "turnover_lambda_mean": _mean(lambdas),
        "participation_clamped_fraction": (
            clamped_total / names_total if names_total > 0 else None
        ),
        "effective_breadth_mean": _mean(breadths),
        "effective_breadth_p10": _quantile(breadths, 0.1),
        "effective_breadth_p90": _quantile(breadths, 0.9),
    }
    nem_scales = [
        value
        for value in (
            _float_or_none(record.get("nem_scale")) for record in records
        )
        if value is not None
    ]
    nem_trends = [
        value
        for value in (
            _float_or_none(record.get("nem_s_trend")) for record in records
        )
        if value is not None
    ]
    nem_vols = [
        value
        for value in (
            _float_or_none(record.get("nem_s_vol")) for record in records
        )
        if value is not None
    ]
    if nem_scales:
        summary["nem_scale_mean"] = _mean(nem_scales)
        summary["nem_scale_p10"] = _quantile(nem_scales, 0.1)
        summary["nem_scale_p90"] = _quantile(nem_scales, 0.9)
        summary["nem_s_trend_mean"] = _mean(nem_trends)
        summary["nem_s_vol_mean"] = _mean(nem_vols)
        summary["nem_active_fraction"] = sum(
            1 for scale in nem_scales if scale < 1.0
        ) / len(records)
    return summary


def _profile_scoped_specs(
    request: NetAlphaTrainingRequest,
) -> tuple[
    list[tuple[int, int, PolicyProfile]],
    dict[tuple[int, int, int, str], str],
]:
    """Profile-scoped replay specs plus dropout transparency.

    Each profile's feasible cell set is derived from its own cap overrides;
    the returned spec list preserves the legacy (cadence, top_k, profile)
    ordering for byte-identical execution when scoping is a no-op. Every
    legacy-feasible (horizon, cadence, top_k) cell excluded for a profile is
    recorded as ``profile-cap-infeasible`` so the ledger explains absent
    frontier cells instead of silently shrinking them.
    """
    frontier = request.execution_frontier
    scoped_cells: dict[str, set[tuple[int, int]]] = {}
    profile_order: dict[str, int] = {}
    specs: list[tuple[int, int, PolicyProfile]] = []
    dropouts: dict[tuple[int, int, int, str], str] = {}
    union_cells: set[tuple[int, int]] = set()
    horizons_by_cell: dict[tuple[int, int], set[int]] = {}
    for profile_index, profile in enumerate(request.policy_profiles):
        cells = frontier.feasible_cells_for_profile(
            request.portfolio.max_exposure,
            request.portfolio.max_single_weight,
            single_name_cap_override=profile.single_name_cap_override,
            gross_utilization_target=profile.gross_utilization_target,
        )
        scoped_cells[profile.profile_id] = {(c, k) for _h, c, k in cells}
        profile_order[profile.profile_id] = profile_index
        for h, c, k in cells:
            union_cells.add((c, k))
            horizons_by_cell.setdefault((c, k), set()).add(h)

    def _spec_order(spec: tuple[int, int, PolicyProfile]) -> tuple[int, int, int]:
        cadence, top_k, profile = spec
        return (cadence, top_k, profile_order[profile.profile_id])

    seen: set[tuple[int, int, str]] = set()
    scoped_specs: dict[tuple[int, int], list[PolicyProfile]] = {}
    for c, k in sorted(union_cells):
        horizon_hits = sorted(horizons_by_cell[(c, k)])
        for h in horizon_hits:
            for profile in request.policy_profiles:
                if (c, k) not in scoped_cells[profile.profile_id]:
                    dropouts[(h, c, k, profile.profile_id)] = "profile-cap-infeasible"
                    continue
                key = (c, k, profile.profile_id)
                if key in seen:
                    continue
                seen.add(key)
                scoped_specs.setdefault((c, k), []).append(profile)
    for c_k, profiles in scoped_specs.items():
        specs.extend((c_k[0], c_k[1], profile) for profile in profiles)
    specs.sort(key=_spec_order)
    return specs, dropouts


def _cadence_decision_sessions(
    sorted_sessions: tuple[datetime, ...], frequency_sessions: int
) -> tuple[datetime, ...]:
    """Pure cadence kernel: subsample sorted sessions at the given frequency.

    Single definition shared by ``_replay_costs`` and ``_replay_costs_batch``
    so one cadence always yields one decision schedule. Sessions shorter than
    two entries cannot be subsampled and pass through unchanged; frequency
    validation (positive) is delegated to ``rebalance_session_indices``.
    """
    if len(sorted_sessions) < 2:
        return sorted_sessions
    indices = rebalance_session_indices(
        sorted_sessions,
        min(sorted_sessions),
        max(sorted_sessions),
        frequency_sessions,
        legacy_daily=False,
    )
    return tuple(sorted_sessions[i] for i in indices)


def _replay_costs(
    registry: ModelArtifactRegistry,
    calibrated: pl.DataFrame,
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    risk: RiskSettings,
    market_frame: pl.DataFrame,
    manifest: DatasetManifest,
    profile: PolicyProfile,
    *,
    rebalance_frequency_sessions: int,
    top_k: int,
) -> ProfileReplayEvidence:
    """Execution-equivalent base/stress replay over the segment-identified OOF.

    The calibrated OOF score panel and the raw pre-holdout executable market
    panel are replayed through ``replay_execution_equivalent`` exactly once;
    base and stress equity are parallel because one backtester advances both
    scenarios against the same immutable prepared market. The returned pair is
    the same immutable evidence object seen from each cost path. The exact
    ``rebalance_frequency_sessions`` (cadence C) and ``top_k`` (K) of the
    candidate cell drive the sparse and matched dense-shadow replay identically.
    """
    del oof_labels
    sessions_by_segment: dict[int, list[datetime]] = {}
    for row in calibrated.select(_OOF_SEGMENT, SESSION_COLUMN).unique().iter_rows(
        named=True
    ):
        sessions_by_segment.setdefault(int(row[_OOF_SEGMENT]), []).append(
            row[SESSION_COLUMN]
        )
    is_v5 = (
        profile.execution_utility_mode == "sparse_hold_replace_v2"
        or profile.sizing_mode == "risk_balanced_waterfill_v2"
    )
    replay_policy = _risk_policy_for_profile(
        request, profile, horizon_sessions,
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
    )
    decision_sessions_by_segment: dict[int, tuple[datetime, ...]] = {}
    for segment, sessions in sessions_by_segment.items():
        sorted_sessions = tuple(sorted(sessions))
        if is_v5:
            decision_sessions_by_segment[segment] = _cadence_decision_sessions(sorted_sessions, replay_policy.rebalance_frequency_sessions)
        else:
            decision_sessions_by_segment[segment] = sorted_sessions
    context = _execution_replay_context(
        registry,
        request, manifest, market_frame, profile,
        seed=request.seed + horizon_sessions,
        horizon_sessions=horizon_sessions,
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
    )
    replay_request = ExecutionEquivalentReplayRequest(
        context=context,
        market_frame=market_frame,
        score_frame=calibrated,
        segment_column=_OOF_SEGMENT,
        decision_sessions_by_segment=decision_sessions_by_segment,
        horizon_sessions=horizon_sessions,
    )
    evidence = replay_execution_equivalent(replay_request)
    if not is_v5:
        return ProfileReplayEvidence(candidate=evidence, dense_shadow=None)
    dense_profile = PolicyProfile(
        profile_id=profile.profile_id,
        no_trade_band_bps=profile.no_trade_band_bps,
        growth_risk_aversion=profile.growth_risk_aversion,
        execution_utility_mode="delta_cost_aware_v1",
        sizing_mode="alpha_vol_squared_v1",
        net_exposure_gate_mode=profile.net_exposure_gate_mode,
        gate_trend_lookback_sessions=profile.gate_trend_lookback_sessions,
        gate_floor=profile.gate_floor,
    )
    shadow_context = _execution_replay_context(
        registry,
        request, manifest, market_frame, dense_profile,
        seed=request.seed + horizon_sessions + 7,
        horizon_sessions=horizon_sessions,
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
    )
    shadow_request = ExecutionEquivalentReplayRequest(
        context=shadow_context,
        market_frame=market_frame,
        score_frame=calibrated,
        segment_column=_OOF_SEGMENT,
        decision_sessions_by_segment=decision_sessions_by_segment,
        horizon_sessions=horizon_sessions,
    )
    shadow_evidence = replay_execution_equivalent(shadow_request)
    return ProfileReplayEvidence(candidate=evidence, dense_shadow=shadow_evidence)


def _replay_costs_batch(
    registry: ModelArtifactRegistry,
    calibrated: pl.DataFrame,
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    risk: RiskSettings,
    market_frame: pl.DataFrame,
    manifest: DatasetManifest,
    specs: Sequence[tuple[int, int, PolicyProfile]],
    *,
    stats_out: dict[str, int] | None = None,
    sizing_out: dict[tuple[int, int, int, str], dict[str, object]] | None = None,
) -> Mapping[tuple[int, int, int, str], ProfileReplayEvidence]:
    """Cadence-group batch replay: one prepared batch per cadence, shared across profiles.

    Groups candidate specs by cadence (rebalance_frequency_sessions). Within
    each cadence group, all (top_k, profile) pairs share one immutable
    prepared batch. The batch is released after the cadence group completes.
    Returns a mapping from (horizon, cadence, top_k, profile_id) to (base, shadow)
    evidence.
    """
    from collections import defaultdict

    cadence_groups: dict[tuple[int, bool], list[tuple[int, PolicyProfile]]] = defaultdict(list)
    for cadence, top_k, profile in specs:
        sparse_schedule = (
            profile.execution_utility_mode == "sparse_hold_replace_v2"
            or profile.sizing_mode == "risk_balanced_waterfill_v2"
        )
        cadence_groups[(cadence, sparse_schedule)].append((top_k, profile))

    results: dict[tuple[int, int, int, str], ProfileReplayEvidence] = {}

    for (cadence, is_v5), group in cadence_groups.items():
        replay_policy = _risk_policy_for_profile(
            request, group[0][1], horizon_sessions,
            rebalance_frequency_sessions=cadence,
            top_k=group[0][0],
        )
        sessions_by_segment: dict[int, list[datetime]] = {}
        for row in calibrated.select(_OOF_SEGMENT, SESSION_COLUMN).unique().iter_rows(
            named=True
        ):
            sessions_by_segment.setdefault(int(row[_OOF_SEGMENT]), []).append(
                row[SESSION_COLUMN]
            )
        decision_sessions_by_segment: dict[int, tuple[datetime, ...]] = {}
        for segment, sessions in sessions_by_segment.items():
            sorted_sessions = tuple(sorted(sessions))
            if is_v5:
                decision_sessions_by_segment[segment] = _cadence_decision_sessions(sorted_sessions, replay_policy.rebalance_frequency_sessions)
            else:
                decision_sessions_by_segment[segment] = sorted_sessions

        first_profile = group[0][1]
        first_top_k = group[0][0]
        primary_context = _execution_replay_context(
            registry,
            request, manifest, market_frame, first_profile,
            seed=request.seed + horizon_sessions,
            horizon_sessions=horizon_sessions,
            rebalance_frequency_sessions=cadence,
            top_k=first_top_k,
        )
        primary_request = ExecutionEquivalentReplayRequest(
            context=primary_context,
            market_frame=market_frame,
            score_frame=calibrated,
            segment_column=_OOF_SEGMENT,
            decision_sessions_by_segment=decision_sessions_by_segment,
            horizon_sessions=horizon_sessions,
        )
        del primary_request  # streaming prepares segments lazily, not eagerly
        batch_replay_requests: list[ExecutionEquivalentReplayRequest] = []
        batch_keys: list[tuple[int, int, int, str]] = []
        batch_shadow_contexts: list[ExecutionReplayContext | None] = []
        shadow_requests_list: list[ExecutionEquivalentReplayRequest] = []
        for top_k, profile in group:
            context = _execution_replay_context(
                registry,
                request, manifest, market_frame, profile,
                seed=request.seed + horizon_sessions,
                horizon_sessions=horizon_sessions,
                rebalance_frequency_sessions=cadence,
                top_k=top_k,
            )
            replay_req = ExecutionEquivalentReplayRequest(
                context=context,
                market_frame=market_frame,
                score_frame=calibrated,
                segment_column=_OOF_SEGMENT,
                decision_sessions_by_segment=decision_sessions_by_segment,
                horizon_sessions=horizon_sessions,
            )
            batch_replay_requests.append(replay_req)
            batch_keys.append((horizon_sessions, cadence, top_k, profile.profile_id))

            if is_v5:
                dense_profile = PolicyProfile(
                    profile_id=profile.profile_id,
                    no_trade_band_bps=profile.no_trade_band_bps,
                    growth_risk_aversion=profile.growth_risk_aversion,
                    execution_utility_mode="delta_cost_aware_v1",
                    sizing_mode="alpha_vol_squared_v1",
                    net_exposure_gate_mode=profile.net_exposure_gate_mode,
                    gate_trend_lookback_sessions=profile.gate_trend_lookback_sessions,
                    gate_floor=profile.gate_floor,
                )
                shadow_context = _execution_replay_context(
                    registry,
                    request, manifest, market_frame, dense_profile,
                    seed=request.seed + horizon_sessions + 7,
                    horizon_sessions=horizon_sessions,
                    rebalance_frequency_sessions=cadence,
                    top_k=top_k,
                )
                batch_shadow_contexts.append(shadow_context)
            else:
                batch_shadow_contexts.append(None)

        stream_stats: dict[str, int] = {}
        primary_evidences: tuple[ExecutionReplayEvidence, ...] = ()
        shadow_evidences: tuple[ExecutionReplayEvidence, ...] = ()
        shadow_indices: list[int] = []
        for idx, shadow_ctx in enumerate(batch_shadow_contexts):
            if shadow_ctx is not None:
                shadow_requests_list.append(
                    ExecutionEquivalentReplayRequest(
                        context=shadow_ctx,
                        market_frame=market_frame,
                        score_frame=calibrated,
                        segment_column=_OOF_SEGMENT,
                        decision_sessions_by_segment=decision_sessions_by_segment,
                        horizon_sessions=horizon_sessions,
                    )
                )
                shadow_indices.append(idx)

        # Segment-major streaming: primaries and dense shadows share one live
        # prepared segment at a time; the effective memory limit is resolved
        # and every pre-build boundary planned inside the stream itself.
        combined_requests = [*batch_replay_requests, *shadow_requests_list]
        combined_evidences: tuple[ExecutionReplayEvidence, ...] = ()
        if combined_requests:
            combined_evidences = stream_execution_replay_batch(
                combined_requests,
                stats=stream_stats,
                request_limit_bytes=(
                    request.max_rss_mib * 1024 * 1024
                    if request.max_rss_mib is not None
                    else None
                ),
                candidate_workers=max(1, int(request.discovery_workers)),
            )
        primary_evidences = combined_evidences[: len(batch_replay_requests)]
        shadow_evidences = combined_evidences[len(batch_replay_requests) :]

        shadow_map: dict[int, ExecutionReplayEvidence] = dict(
            zip(shadow_indices, shadow_evidences, strict=True)
        )

        for idx, key in enumerate(batch_keys):
            primary_ev = (
                primary_evidences[idx]
                if idx < len(primary_evidences)
                else None
            )
            if primary_ev is None:
                continue
            if sizing_out is not None:
                sizing_out[key] = _sizing_diagnostics_summary(
                    batch_replay_requests[idx].context.risk_policy.compounding_evidence
                )
            if not is_v5:
                results[key] = ProfileReplayEvidence(candidate=primary_ev, dense_shadow=None)
            else:
                shadow_ev = shadow_map.get(idx)
                if shadow_ev is None:
                    continue
                results[key] = ProfileReplayEvidence(
                    candidate=primary_ev, dense_shadow=shadow_ev
                )

        if stats_out is not None:
            for stat_key, value in stream_stats.items():
                stats_out[stat_key] = stats_out.get(stat_key, 0) + int(value)
        del combined_evidences

    return results


def _evidence_from_execution(
    horizon_sessions: int,
    profile_id: str,
    model_family: str,
    base_evidence: ExecutionReplayEvidence,
    stress_evidence: ExecutionReplayEvidence,
    fold_rank_ics: tuple[float, ...],
    segment_count: int,
    *,
    paired_stress_log_growth: tuple[float, ...] = (),
    sparse_turnover: float = 0.0,
    shadow_turnover: float = 0.0,
    rebalance_frequency_sessions: int = 5,
    top_k: int = 20,
) -> HorizonOOFEvidence:
    """Build a candidate's base/stress equity evidence from the prepared replay.

    Every evaluated session contributes one daily equity log-growth value;
    filled-cycle coverage is derived from ``cash_session_fraction``. A session
    whose execution input is missing fails closed earlier in the adapter and is
    never zero-filled, so the missing/partial cohort counts are structurally
    zero. ``paired_stress_log_growth`` carries the per-vintage sparse-minus-dense
    stress improvement and ``sparse_turnover``/``shadow_turnover`` the matched
    replay turnover pair, segment-aligned to ``base_log_growth``. The exact
    candidate ``rebalance_frequency_sessions`` (cadence C) and ``top_k`` (K) are
    persisted so selection keys and the artifact fingerprint stay distinct.
    """
    if base_evidence.segment_ids != stress_evidence.segment_ids:
        raise ValueError("base and stress execution segment identities diverged")
    growth_count = len(base_evidence.base_log_growth)
    if growth_count == 0:
        raise ValueError("execution replay produced no evaluated sessions")
    # Exposure-based active cohort: complete return intervals with a positive
    # prior ledger positions_value (a held position), not filled-order counts.
    active = base_evidence.invested_interval_count
    observed = base_evidence.observed_interval_count
    return HorizonOOFEvidence(
        horizon_sessions=horizon_sessions,
        profile_id=profile_id,
        model_family=model_family,
        base_log_growth=base_evidence.base_log_growth,
        stress_log_growth=stress_evidence.stress_log_growth,
        cohort_segment_ids=base_evidence.segment_ids,
        complete_cohort_count=observed,
        active_cohort_count=active,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=segment_count,
        fold_rank_ics=fold_rank_ics,
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
        paired_stress_log_growth=paired_stress_log_growth,
        sparse_turnover=float(sparse_turnover),
        shadow_turnover=float(shadow_turnover),
        unresolved_outcome_counts=(),
        blocked_vintage_count=0,
    )


def _coverage_failure_reason(
    evidence: HorizonOOFEvidence, request: NetAlphaTrainingRequest
) -> str:
    """Fail-closed reason when a candidate misses a coverage/admission pre-gate."""
    distinct_segments = len(set(evidence.cohort_segment_ids))
    if distinct_segments != evidence.segment_count:
        return (
            f"incomplete-segment-coverage:{distinct_segments}/"
            f"{evidence.segment_count}"
        )
    # A selected filled exit that cannot be valued is a blocker, but isolated
    # blocked vintages within a small tolerance of all vintages stay diagnostic
    # only: the portfolio metrics and IC gates still admit the candidate.
    # Other missing-realized vintages (e.g. no orders) keep the opaque coverage
    # reason.
    total_vintages = max(1, int(evidence.complete_cohort_count))
    if (
        evidence.blocked_vintage_count / total_vintages
        > _MAX_BLOCKED_VINTAGE_FRACTION
    ):
        return f"selected-exit-unresolved:{evidence.blocked_vintage_count}"
    if evidence.missing_cohort_count > 0:
        return f"missing-realized-vintages:{evidence.missing_cohort_count}"
    if not evidence.fold_rank_ics:
        return "no-usable-fold-rank-ic"
    positive = sum(1 for value in evidence.fold_rank_ics if value > 0.0)
    if positive <= len(evidence.fold_rank_ics) / 2:
        return f"rank-ic-majority-not-positive:{positive}/{len(evidence.fold_rank_ics)}"
    observed = int(evidence.complete_cohort_count)
    if observed < request.compounding.min_observed_sessions:
        return (
            f"insufficient-observed-sessions:{observed}/"
            f"{request.compounding.min_observed_sessions}"
        )
    if evidence.complete_cohort_count <= 0:
        return "no-complete-cohorts"
    active_fraction = evidence.active_cohort_count / evidence.complete_cohort_count
    if active_fraction < request.compounding.min_active_cohort_fraction:
        return (
            f"active-coverage-insufficient:{active_fraction:.4f}"
        )
    return ""


def _incremental_growth_gate(
    candidate: ExecutionReplayEvidence,
    shadow: ExecutionReplayEvidence,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
) -> tuple[dict[str, float], str | None]:
    """Require positive paired stress growth before publishing a sparse profile."""
    if candidate.segment_ids != shadow.segment_ids:
        return {}, "incremental-shadow-segment-mismatch"
    if len(candidate.stress_log_growth) < 2:
        return {}, "incremental-insufficient-vintages"
    delta = tuple(
        float(c - s)
        for c, s in zip(candidate.stress_log_growth, shadow.stress_log_growth, strict=True)
    )
    bootstrap = _cohort_bootstrap(
        delta,
        candidate.segment_ids,
        request.bootstrap_resamples,
        request.seed + horizon_sessions + 997,
        min_block_length=max(1, horizon_sessions),
    )
    if bootstrap is None:
        return {}, "incremental-bootstrap-inadmissible"
    lower = bootstrap.lower_mean(request.bootstrap_alpha)
    shadow_turnover = max(shadow.turnover, 1e-12)
    turnover_ratio = candidate.turnover / shadow_turnover
    metrics = {
        "paired_stress_delta_lower": float(lower),
        "turnover_ratio": float(turnover_ratio),
    }
    if lower <= 0.0:
        return metrics, "incremental-stress-growth-not-positive"
    if turnover_ratio > 0.60:
        return metrics, "incremental-turnover-too-high"
    return metrics, None


def _build_horizon_evidence(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    *,
    oof_cache: _OofCache | None = None,
    matrix: PreparedTrainingMatrix | None = None,
    registry: ModelArtifactRegistry | None = None,
) -> HorizonDiscovery:
    """Build the two-profile ``(horizon, profile)`` OOF frontier.

    Future labels are never a discovery score: every candidate reuses the one
    maximum-horizon balanced fold plan, each fold fits the weighted ElasticNet
    baseline on its train rows only, validation rows are predicted target-free
    with the segment identity preserved, joined to decimal realized outcomes
    after prediction, causally calibrated once per horizon, and replayed under
    base and stress costs for every pre-registered policy profile (no learner
    is ever refit per profile). A ``(horizon, profile)`` candidate contributes
    evidence only when every segment contributes an evaluated vintage, no
    realized vintage is missing, a strict majority of usable folds has positive
    session-mean Rank-IC, and the compounding coverage gates pass. Independent
    horizon universes are never inner-joined.

    Calibrated OOF evidence is spilled to the per-run temporary cache only for
    horizons with at least one admitted profile; rejected horizons release
    their frames before the next horizon. ``max_rss_mib`` is enforced at safe
    horizon boundaries by raising ``_MemoryBudgetExceededError``.
    """
    if oof_cache is None:
        oof_cache = _OofCache(_default_oof_cache_base())
    missing_horizons = [
        h
        for h in sorted(request.candidate_horizon_sessions)
        if h not in data.labels_by_horizon
    ]
    if missing_horizons:
        raise ValueError(
            f"requested horizon(s) {missing_horizons} have no label data; "
            "every requested candidate horizon must be present in the selected "
            "labels (a missing requested horizon is a deterministic error, not a "
            "silent fallback)"
        )
    if tuple(request.execution_frontier.candidate_horizon_sessions) != tuple(
        request.candidate_horizon_sessions
    ):
        raise ValueError(
            "execution_frontier.candidate_horizon_sessions must equal "
            "candidate_horizon_sessions; the declared (H, C, K) frontier and the "
            "discovery grid must be identical before fitting"
        )
    request.execution_frontier.require_feasible_horizons(
        request.portfolio.max_exposure, request.portfolio.max_single_weight
    )
    candidate_specs, scoped_dropout_reasons = _profile_scoped_specs(request)
    cells_by_horizon: dict[int, list[tuple[int, int]]] = {}
    for h, c, k in request.execution_frontier.feasible_cells(
        request.portfolio.max_exposure, request.portfolio.max_single_weight
    ):
        cells_by_horizon.setdefault(h, []).append((c, k))

    if matrix is None:
        # One canonical workspace before discovery: OOF fitting and the
        # calibration seed both reuse it, so no second matrix ever exists.
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
            "matrix_prepare", planned_bytes=planned_bytes, envelope=envelope.to_dict()
        )
        if not envelope.ok:
            raise _MemoryBudgetExceededError("matrix_prepare")
        matrix = prepare_training_matrix(
            TrainingPanelView(pre_holdout),
            _on_demand_schema(learner_columns),
            tuple(folds),
        )

    evidence: list[HorizonOOFEvidence] = []
    diagnostics: list[HorizonOOFDiagnostic] = []
    oof_by_horizon: dict[int, tuple[Path, Path, list[float]]] = {}
    dropout_reasons: dict[tuple[int, int, int, str], str] = dict(scoped_dropout_reasons)
    sizing_diagnostics_by_candidate: dict[
        tuple[int, int, int, str], dict[str, object]
    ] = {}
    execution_evidence_by_candidate: dict[
        tuple[int, int, int, str], ExecutionReplayEvidence
    ] = {}
    coverage_by_horizon: dict[int, HorizonOutcomeCoverage] = {}
    horizon_memory: dict[int, dict[str, object]] = {}
    path_evaluation_count = 0
    for horizon in sorted(data.labels_by_horizon):
        horizon_started = time.monotonic()
        label_frame = data.labels_by_horizon[horizon]
        logger.debug(
            "[ALGO] stage=horizon_start horizon=%d label_rows=%d",
            horizon,
            label_frame.height,
        )
        _emit_resource_checkpoint(f"horizon_{horizon}_start")
        if label_frame.is_empty() or label_frame.height < 3:
            continue
        if (
            RISK_RESIDUAL_COLUMN not in label_frame.columns
            or REFERENCE_COST_COLUMN not in label_frame.columns
        ):
            raise ValueError(
                f"horizon {horizon} label frame is missing decimal "
                f"realized-outcome columns ({RISK_RESIDUAL_COLUMN!r}, "
                f"{REFERENCE_COST_COLUMN!r}); a missing realized outcome must "
                "never degrade into an empty block list"
            )
        manifest = _base_manifest(request, data, data.feature_frame, horizon)
        replay_stats: dict[str, int] = {}
        oof, oof_labels, ics, diagnostic, fold_path_count = _fit_oof(
            pre_holdout, folds, data, request, manifest, learner_columns,
            horizon, None,
            family=request.discovery_model_family,
            matrix=matrix,
        )
        path_evaluation_count += fold_path_count
        diagnostics.append(diagnostic)
        logger.debug(
            "[ALGO] stage=oof_complete horizon=%d oof_rows=%d labeled_rows=%d "
            "usable_folds=%d path_evaluations=%d",
            horizon,
            oof.height,
            oof_labels.height,
            len(ics),
            fold_path_count,
        )
        batch_replay_count = 0
        batch_segment_build_count = 0
        batch_prepare_elapsed_ms = 0
        batch_execute_elapsed_ms = 0
        if oof.is_empty() or oof_labels.is_empty():
            for c, k in cells_by_horizon.get(horizon, []):
                for profile in request.policy_profiles:
                    dropout_reasons[(horizon, c, k, profile.profile_id)] = (
                        "no-oof-labels"
                    )
        else:
            coverage = None
            status_frame = data.status_by_horizon.get(horizon)
            if status_frame is not None and not status_frame.is_empty():
                from src.stocks.ml.data import HorizonOutcomeCoverage

                coverage = HorizonOutcomeCoverage.build(
                    horizon,
                    oof.select(_ID_COLUMN, SESSION_COLUMN, _OOF_SEGMENT),
                    status_frame,
                    segment_column=_OOF_SEGMENT,
                )
                coverage_by_horizon[horizon] = coverage
                logger.info(
                    "[DATA] stage=outcome_coverage horizon=%d realised=%d "
                    "partial_tail=%d unresolved=%d",
                    horizon,
                    coverage.realized_rows,
                    coverage.status_counts.partial_tail,
                    coverage.status_counts.unresolved,
                )
            # Session indices are 1-based dense ranks while canonical matrix
            # codes are 0-based, so the seed window ends one code earlier.
            initial_rows = np.flatnonzero(
                matrix.session_code < folds[0].validation_decision_start - 1
            )
            seed_ledger = build_initial_calibration_seed(
                matrix, initial_rows, request, horizon, manifest,
                data=data,
            )
            calibrated = _causal_oof_calibrate(
                oof, oof_labels, request, horizon, seed_ledger=seed_ledger
            )
            admitted_any = False
            candidate_specs = [
                spec for spec in candidate_specs if spec[0] <= horizon
            ]
            if candidate_specs:
                replay_planned_bytes = int(
                    calibrated.estimated_size()
                ) + int(oof_labels.estimated_size())
                replay_envelope = _plan_training_allocation(
                    replay_planned_bytes,
                    request_limit_bytes=(
                        None
                        if request.max_rss_mib is None
                        else int(request.max_rss_mib) * 1024 * 1024
                    ),
                    reserve_bytes=int(request.memory_reserve_mib) * 1024 * 1024,
                )
                _emit_resource_checkpoint(
                    "replay",
                    planned_bytes=replay_planned_bytes,
                    envelope=replay_envelope.to_dict(),
                )
                if not replay_envelope.ok:
                    raise _MemoryBudgetExceededError("replay")
                sizing_by_candidate: dict[tuple[int, int, int, str], dict[str, object]] = {}
                batch_results = _replay_costs_batch(
                    _require_caller_registry(registry), calibrated, oof_labels,
                    request, horizon,
                    request.risk, pre_holdout, data.manifest, candidate_specs,
                    stats_out=replay_stats,
                    sizing_out=sizing_by_candidate,
                )
                sizing_diagnostics_by_candidate.update(sizing_by_candidate)
                for cadence, top_k, profile in candidate_specs:
                    key = (horizon, cadence, top_k, profile.profile_id)
                    pair = batch_results.get(key)
                    if pair is None:
                        dropout_reasons[key] = "replay-batch-error"
                        continue
                    base_evidence = pair.candidate
                    shadow_evidence = pair.dense_shadow
                    is_v5 = (
                        profile.execution_utility_mode == "sparse_hold_replace_v2"
                        or profile.sizing_mode == "risk_balanced_waterfill_v2"
                    )
                    if is_v5:
                        shadow_for_gate: ExecutionReplayEvidence | None = shadow_evidence
                        if shadow_for_gate is None:
                            dropout_reasons[key] = "missing-shadow-evidence"
                            continue
                        _incremental_growth_gate(
                            base_evidence, shadow_for_gate, request, horizon
                        )
                    else:
                        shadow_for_gate = None
                    logger.debug(
                        "[EVAL] stage=profile_replay horizon=%d cadence=%d top_k=%d "
                        "profile=%s band_bps=%.3f",
                        horizon, cadence, top_k,
                        profile.profile_id, profile.no_trade_band_bps,
                    )
                    if not base_evidence.base_log_growth:
                        dropout_reasons[key] = "no-evaluated-vintages"
                        continue
                    if base_evidence.filled_orders == 0:
                        dropout_reasons[key] = "no-filled-orders"
                        continue
                    stress_evidence = base_evidence
                    paired_stress: tuple[float, ...] = ()
                    sparse_turnover = base_evidence.turnover
                    shadow_turnover = (
                        shadow_evidence.turnover
                        if shadow_evidence is not None
                        else 0.0
                    )
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
                        horizon, profile.profile_id, "net_alpha_elastic_net",
                        base_evidence, stress_evidence, tuple(ics), len(folds),
                        paired_stress_log_growth=paired_stress,
                        sparse_turnover=sparse_turnover,
                        shadow_turnover=shadow_turnover,
                        rebalance_frequency_sessions=cadence,
                        top_k=top_k,
                    )
                    execution_evidence_by_candidate[key] = base_evidence
                    failure_reason = _coverage_failure_reason(
                        candidate_evidence, request
                    )
                    dropout_reasons[key] = failure_reason
                    logger.debug(
                        "[EVAL] stage=profile_result horizon=%d cadence=%d top_k=%d "
                        "profile=%s sessions=%d active=%d dropout=%s",
                        horizon, cadence, top_k,
                        profile.profile_id,
                        len(base_evidence.base_log_growth),
                        round(
                            base_evidence.planned_cycles
                            * (1.0 - base_evidence.cash_session_fraction)
                        ),
                        failure_reason or "none",
                    )
                    if failure_reason:
                        continue
                    evidence.append(candidate_evidence)
                    admitted_any = True
                batch_replay_count = len(candidate_specs)
                # Real observed values: actual segment builds, deduplicated
                # prepared bytes, and disjoint prepare/execute timers.
                batch_segment_build_count = int(
                    replay_stats.get("prepared_segment_build_count", 0)
                )
                batch_prepare_elapsed_ms = int(
                    replay_stats.get("replay_prepare_elapsed_ms", 0)
                )
                batch_execute_elapsed_ms = int(
                    replay_stats.get("replay_execute_elapsed_ms", 0)
                )
            if admitted_any:
                oof_path, labels_path = oof_cache.store(
                    horizon, calibrated, oof_labels
                )
                oof_by_horizon[horizon] = (oof_path, labels_path, ics)
            del oof, oof_labels, calibrated
        replay_runtime_metrics = {
            "execution_replay_count": batch_replay_count,
            "prepared_segment_build_count": batch_segment_build_count,
            "prepared_cache_bytes": int(
                replay_stats.get("prepared_cache_bytes", 0)
            ),
            "replay_prepare_elapsed_ms": batch_prepare_elapsed_ms,
            "replay_execute_elapsed_ms": batch_execute_elapsed_ms,
            "peak_live_prepared_segments": int(
                replay_stats.get("peak_live_prepared_segments", 0)
            ),
        }
        horizon_memory[horizon] = {
            "rss_mib": _current_rss_mib(),
            "peak_rss_mib": _peak_rss_mib(),
            "elapsed_ms": int((time.monotonic() - horizon_started) * 1000),
            "cache_bytes": oof_cache.cache_bytes,
        }
        horizon_memory[horizon].update(replay_runtime_metrics)
        _enforce_memory_budget(request, "horizon_discovery")

    if request.enable_horizon_blend:
        _replay_blend_candidates(
            pre_holdout=pre_holdout,
            folds=folds,
            data=data,
            request=request,
            matrix=matrix,
            registry=registry,
            oof_by_horizon=oof_by_horizon,
            cells_by_horizon=cells_by_horizon,
            evidence=evidence,
            dropout_reasons=dropout_reasons,
            execution_evidence_by_candidate=execution_evidence_by_candidate,
        )
    return HorizonDiscovery(
        evidence=tuple(evidence),
        diagnostics=tuple(diagnostics),
        oof_by_horizon=oof_by_horizon,
        dropout_reasons=dropout_reasons,
        execution_evidence_by_candidate=execution_evidence_by_candidate,
        coverage_by_horizon=coverage_by_horizon,
        horizon_memory=horizon_memory,
        sizing_diagnostics_by_candidate=sizing_diagnostics_by_candidate,
        oof_cache=oof_cache,
        path_evaluation_count=path_evaluation_count,
        path_evaluation_bound=(
            len(diagnostics) * len(folds) * (_NESTED_INNER_FOLDS + 1)
        ),
    )


def _replay_blend_candidates(
    *,
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    matrix: PreparedTrainingMatrix,
    registry: ModelArtifactRegistry | None,
    oof_by_horizon: Mapping[int, tuple[Path, Path, list[float]]],
    cells_by_horizon: Mapping[int, list[tuple[int, int]]],
    evidence: list[HorizonOOFEvidence],
    dropout_reasons: dict[tuple[int, int, int, str], str],
    execution_evidence_by_candidate: dict[
        tuple[int, int, int, str], ExecutionReplayEvidence
    ],
) -> None:
    """Replay cross-horizon rank-blend candidates at the largest admitted horizon.

    Blend scores are the mean within-session percentile rank of the cached
    calibrated OOF frames, recomputed through the same causal calibration as
    the base path before replay. Every failure mode is additive-only: a
    missing second horizon or an unusable blend frame records
    ``blend-scores-unavailable`` on the would-be keys and never touches base
    evidence.
    """
    target_horizon = max(request.execution_frontier.candidate_horizon_sessions)
    cells_at_target = cells_by_horizon.get(target_horizon, [])
    would_be_keys = [
        (target_horizon, cadence, top_k,
         f"{profile.profile_id}{_BLEND_PROFILE_SUFFIX}")
        for cadence, top_k in cells_at_target
        for profile in request.policy_profiles
    ]
    admitted_horizons = sorted(oof_by_horizon)
    blended: pl.DataFrame | None = None
    if len(admitted_horizons) >= 2 and cells_at_target and folds:
        cached_frames: dict[int, pl.DataFrame] = {}
        try:
            for horizon in admitted_horizons:
                cached_frames[horizon] = _read_oof_parquet(
                    oof_by_horizon[horizon][0]
                )
            try:
                blended = _blend_calibrated_scores(cached_frames, target_horizon)
            except ValueError:
                logger.warning(
                    "[ALGO] stage=horizon_blend status=unusable "
                    "reason=blend-scores-unavailable"
                )
                blended = None
        finally:
            cached_frames.clear()
    if blended is None:
        for key in would_be_keys:
            dropout_reasons[key] = _BLEND_DROPOUT_UNAVAILABLE
        return

    economic_columns = (
        "expected_active_alpha",
        "alpha_lower_bound",
        "expected_net_alpha",
        "net_alpha_lower_bound",
        "exit_cost_rate",
    )
    blended_raw = blended.drop(
        [column for column in economic_columns if column in blended.columns]
    )
    del blended
    anchor_manifest = _base_manifest(request, data, data.feature_frame, target_horizon)
    initial_rows = np.flatnonzero(
        matrix.session_code < folds[0].validation_decision_start - 1
    )
    seed_ledger = build_initial_calibration_seed(
        matrix, initial_rows, request, target_horizon, anchor_manifest, data=data,
    )
    blend_labels = _read_oof_parquet(oof_by_horizon[target_horizon][1])
    blend_calibrated = _causal_oof_calibrate(
        blended_raw,
        blend_labels,
        request,
        target_horizon,
        seed_ledger=seed_ledger,
    )
    del blended_raw, seed_ledger
    replay_planned_bytes = int(blend_calibrated.estimated_size()) + int(
        blend_labels.estimated_size()
    )
    envelope = _plan_training_allocation(
        replay_planned_bytes,
        request_limit_bytes=(
            None
            if request.max_rss_mib is None
            else int(request.max_rss_mib) * 1024 * 1024
        ),
        reserve_bytes=int(request.memory_reserve_mib) * 1024 * 1024,
    )
    if not envelope.ok:
        raise _MemoryBudgetExceededError("replay")
    blend_specs = [
        (
            cadence,
            top_k,
            replace(
                profile,
                profile_id=f"{profile.profile_id}{_BLEND_PROFILE_SUFFIX}",
            ),
        )
        for cadence, top_k in cells_at_target
        for profile in request.policy_profiles
    ]
    family_label = f"{request.discovery_model_family}+mh_blend"
    anchor_ics = tuple(oof_by_horizon[target_horizon][2])
    batch_results = _replay_costs_batch(
        _require_caller_registry(registry),
        blend_calibrated,
        blend_labels,
        request,
        target_horizon,
        request.risk,
        pre_holdout,
        data.manifest,
        blend_specs,
    )
    del blend_calibrated
    for cadence, top_k, profile in blend_specs:
        key = (target_horizon, cadence, top_k, profile.profile_id)
        pair = batch_results.get(key)
        if pair is None:
            dropout_reasons[key] = "replay-batch-error"
            continue
        base_evidence = pair.candidate
        if not base_evidence.base_log_growth:
            dropout_reasons[key] = "no-evaluated-vintages"
            continue
        if base_evidence.filled_orders == 0:
            dropout_reasons[key] = "no-filled-orders"
            continue
        # Frontier-stage blend cells admit on candidate evidence alone; the
        # dense shadow is a champion/holdout certification requirement, so a
        # missing shadow no longer drops the cell here.
        shadow_evidence = pair.dense_shadow
        paired_stress: tuple[float, ...] = ()
        if shadow_evidence is not None:
            _incremental_growth_gate(base_evidence, shadow_evidence, request, target_horizon)
            paired_stress = tuple(
                float(candidate - shadow)
                for candidate, shadow in zip(
                    base_evidence.stress_log_growth,
                    shadow_evidence.stress_log_growth,
                    strict=True,
                )
            )
        candidate_evidence = _evidence_from_execution(
            target_horizon,
            profile.profile_id,
            family_label,
            base_evidence,
            base_evidence,
            anchor_ics,
            len(folds),
            paired_stress_log_growth=paired_stress,
            sparse_turnover=base_evidence.turnover,
            shadow_turnover=(
                shadow_evidence.turnover if shadow_evidence is not None else 0.0
            ),
            rebalance_frequency_sessions=cadence,
            top_k=top_k,
        )
        failure_reason = _coverage_failure_reason(candidate_evidence, request)
        dropout_reasons[key] = failure_reason
        if failure_reason:
            continue
        evidence.append(candidate_evidence)
        execution_evidence_by_candidate[key] = base_evidence


def _run_ordered_with_workers(
    items: Sequence[object],
    task: Callable[[object], object],
    *,
    workers: int = 1,
) -> list[object] | None:
    """Run independent tasks preserving item order; None signals parallel failure.

    ``workers <= 1`` or a single item runs sequentially. Any worker exception
    aborts the parallel path and returns ``None`` so the caller can fall back
    to sequential execution and record the reason.
    """
    if workers <= 1 or len(items) <= 1:
        try:
            return [task(item) for item in items]
        except Exception:  # noqa: BLE001
            logger.warning(
                "[DISCOVERY] stage=parallel_workers status=fallback reason=worker-failure"
            )
            return None
    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            futures = [pool.submit(task, item) for item in items]
            return [future.result() for future in futures]
    except Exception:  # noqa: BLE001 - fallback decision belongs to the caller
        logger.warning(
            "[DISCOVERY] stage=parallel_workers status=fallback reason=worker-failure"
        )
        return None


def _discovery_oof(
    discovery: HorizonDiscovery,
    primary_horizon_sessions: int,
    folds: list[Fold],
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Load the cached primary baseline OOF; the selected primary is never refit.

    The primary's calibrated OOF and labels are read back from the temporary
    spill cache; a missing or corrupt cache file raises ``ValueError`` and is
    never recomputed.
    """
    del folds
    cached = discovery.oof_by_horizon.get(primary_horizon_sessions)
    if cached is None:
        raise ValueError(
            "discovery did not cache the selected primary baseline OOF"
        )
    oof_path, labels_path, ics = cached
    oof = _read_oof_parquet(oof_path)
    oof_labels = _read_oof_parquet(labels_path)
    diagnostic = next(
        (
            diag
            for diag in discovery.diagnostics
            if diag.horizon_sessions == primary_horizon_sessions
        ),
        HorizonOOFDiagnostic(
            horizon_sessions=primary_horizon_sessions,
            model_family="net_alpha_elastic_net",
        ),
    )
    return oof, oof_labels, ics, diagnostic


def _rankability_gate(
    baseline_diag: HorizonOOFDiagnostic,
    evidence: HorizonOOFEvidence,
    selection: HorizonSelectionEvidence,
    request: NetAlphaTrainingRequest,
) -> str:
    """Cheap linear rankability gate before any LightGBM fit.

    The nonlinear challenger may run for at most one ``(horizon, profile)`` and
    only when the linear screen has a non-constant prediction, a positive
    Holm-adjusted session-mean Rank-IC lower bound, and positive base-cost point
    growth on the selected profile's evidence.
    """
    if not baseline_diag.fold_score_stds or all(
        std <= 0.0 for std in baseline_diag.fold_score_stds
    ):
        return "challenger-skipped:no-rankability-evidence:constant-score"
    if evidence is None or not evidence.fold_rank_ics:
        return "challenger-skipped:no-rankability-evidence:no-fold-rank-ic"
    rank_ic_series = tuple(evidence.fold_rank_ics)
    rank_ic_lower = _rank_ic_lower_bound(rank_ic_series, request)
    if not rank_ic_lower > 0.0:
        return "challenger-skipped:no-rankability-evidence:non-positive-rank-ic-bound"
    if float(np.mean(evidence.base_log_growth)) <= 0.0:
        return "challenger-skipped:no-rankability-evidence:non-positive-base-growth"
    if selection.primary_profile_id is None:
        return "challenger-skipped:no-rankability-evidence:no-selected-profile"
    return ""


def _rank_ic_lower_bound(
    rank_ic_series: tuple[float, ...], request: NetAlphaTrainingRequest
) -> float:
    """One-sided moving-block bootstrap lower bound on session-mean Rank-IC.

    The model-family comparison is included in multiplicity control, so the
    quantile is ``bootstrap_alpha / 2`` (two families: linear, nonlinear).
    """
    values = np.asarray(rank_ic_series, dtype=float)
    n = values.size
    if n < 2:
        return 0.0
    from src.stocks.ml.horizons import _segment_block_length

    # Small fold series (n <= 4) use unit blocks so the standard bootstrap
    # resamples all n values instead of collapsing the lower bound to zero.
    block = 1 if n <= 4 else min(max(_segment_block_length(n), 1), n)
    n_blocks = int(np.ceil(n / block))
    if n_blocks < 2:
        return 0.0
    rng = np.random.default_rng(request.seed)
    starts = rng.integers(0, max(1, n - block + 1), size=(request.bootstrap_resamples, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(
        request.bootstrap_resamples, n_blocks * block
    )[:, :n]
    means = values[index].mean(axis=1)
    return float(np.quantile(means, request.bootstrap_alpha / 2.0))


def _schedule_workspace(request: NetAlphaTrainingRequest) -> int | None:
    if request.max_rss_mib is None:
        return None
    return int(request.max_rss_mib * 1024 * 1024 // 4)


def _causal_calibrator(
    request: NetAlphaTrainingRequest, horizon_sessions: int
) -> CausalAlphaCalibrator:
    """Causal session-cluster calibrator on pre-cost ``risk_residual`` outcomes."""
    return CausalAlphaCalibrator(
        bucket_count=request.risk.calibration_bucket_count,
        min_calibration_sessions=request.risk.min_calibration_sessions,
        seed=request.seed + horizon_sessions,
        n_bootstrap=request.bootstrap_resamples,
        bootstrap_alpha=request.bootstrap_alpha,
        block_length=horizon_sessions,
        label_column=RISK_RESIDUAL_COLUMN,
        label_available_column=AVAILABLE_COLUMN,
    )


def _causal_ledger(oof_labels: pl.DataFrame) -> pl.DataFrame:
    """Finite calibration ledger keyed by ``(session, score, residual, availability)``."""
    required = (
        _ID_COLUMN, SESSION_COLUMN, SCORE_COLUMN,
        RISK_RESIDUAL_COLUMN, AVAILABLE_COLUMN,
    )
    missing = [c for c in required if c not in oof_labels.columns]
    if missing:
        raise ValueError(f"calibration ledger missing columns {missing}")
    return (
        oof_labels.select(*required)
        .filter(
            pl.col(SCORE_COLUMN).is_not_null()
            & pl.col(SCORE_COLUMN).is_finite()
            & pl.col(RISK_RESIDUAL_COLUMN).is_not_null()
            & pl.col(RISK_RESIDUAL_COLUMN).is_finite()
            & pl.col(AVAILABLE_COLUMN).is_not_null()
        )
        .rename({SCORE_COLUMN: "score"})
    )


def build_initial_calibration_seed(
    matrix: PreparedTrainingMatrix,
    initial_train_rows: np.ndarray,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    base_manifest: ModelManifest,
    *,
    data: NetAlphaResearchData,
) -> pl.DataFrame:
    """Produce a causal calibration seed ledger from inner purged folds.

    The outer walk-forward plan's first validation session has no prior outer
    OOF history, so a cold-start ledger would keep the first fold in cash.
    Scoring the caller-owned canonical matrix rows before that session through
    nested purged folds yields target-free out-of-sample scores whose realized
    labels are already revealed at the first outer decision session; feeding
    them as a seed ledger removes the cold start without any lookahead. The
    canonical ``PreparedTrainingMatrix`` is consumed through row indices only:
    no second matrix preparation and no row-level training frame ever exists.
    Returns an empty frame when the slice cannot form usable inner folds.
    """
    from src.stocks.ml.fitting import OofFitRequest
    from src.stocks.ml.fitting import fit_horizon_oof as _fit_horizon_oof

    rows = np.asarray(initial_train_rows, dtype=np.int64)
    if rows.size == 0:
        return pl.DataFrame()
    horizon = prepare_horizon_labels(matrix, data, horizon_sessions)
    left = np.searchsorted(horizon.row_index, rows)
    left = np.clip(left, 0, max(0, horizon.row_index.size - 1))
    if not bool((horizon.row_index[left] == rows).any()):
        return pl.DataFrame()

    codes = matrix.session_code
    unique_sessions = np.unique(codes[rows])
    splitter = PurgedWalkForward(
        n_folds=_NESTED_INNER_FOLDS,
        label_horizon_sessions=int(horizon_sessions),
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
    )
    session_frame = pl.DataFrame({_SESSION_IDX: unique_sessions.astype(np.int64)})
    inner_folds = splitter.inner_folds(session_frame, n_inner=_NESTED_INNER_FOLDS)
    prepared_inner: list[PreparedFold] = []
    for index, inner in enumerate(inner_folds):
        train_sessions = unique_sessions[
            np.asarray(inner.train_mask, dtype=np.int64)
        ]
        validation_sessions = unique_sessions[
            np.asarray(inner.validation_mask, dtype=np.int64)
        ]
        if train_sessions.size == 0 or validation_sessions.size == 0:
            continue
        train_rows = np.flatnonzero(np.isin(codes, train_sessions))
        validation_rows = np.flatnonzero(np.isin(codes, validation_sessions))
        if train_rows.size == 0 or validation_rows.size == 0:
            continue
        # Boundary fields are outer-plan metadata; this inner loop only
        # consumes the immutable row-index arrays.
        prepared_inner.append(
            PreparedFold(
                fold_index=index,
                segment_id=index,
                train_rows=train_rows,
                validation_rows=validation_rows,
                train_label_end=0,
                validation_decision_start=0,
            )
        )
    if not prepared_inner:
        return pl.DataFrame()
    result = _fit_horizon_oof(
        matrix,
        horizon,
        tuple(prepared_inner),
        OofFitRequest(
            request=request, manifest=base_manifest, family="net_alpha_elastic_net"
        ),
    )
    return result.labeled


def _causal_oof_calibrate(
    oof: pl.DataFrame,
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    *,
    seed_ledger: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Causal session-cluster calibration applied to OOF scored rows.

    For every OOF decision session ``t`` only ledger rows with ``session < t``
    and ``label_available_time <= t`` are revealed, honoring
    ``RiskSettings.min_calibration_sessions``; the frozen state is then applied
    to that session. A later label can therefore never change an earlier
    session's calibrated score. An optional inner-purged ``seed_ledger`` from
    the initial training slice is prepended so the first outer validation
    decision already sees a full calibration history instead of a cold start.

    The scored panel is sorted once into contiguous session slices; the five
    economic columns stream into preallocated Float64 arrays (with an explicit
    null mask so cash-only states stay true nulls) and attach exactly once.
    """
    ledger_frames = [
        frame
        for frame in (seed_ledger, oof_labels)
        if frame is not None and not frame.is_empty()
    ]
    if not ledger_frames:
        return _zero_calibrated(oof)
    # The incremental schedule consumes revealed history in ledger order, so
    # it must observe ascending availability regardless of caller row order.
    ledger = _causal_ledger(pl.concat(ledger_frames)).sort(
        [AVAILABLE_COLUMN, SESSION_COLUMN]
    )
    if ledger.is_empty() or oof.is_empty():
        return _zero_calibrated(oof)

    economic_columns = (
        "expected_active_alpha",
        "alpha_lower_bound",
        "expected_net_alpha",
        "net_alpha_lower_bound",
        "exit_cost_rate",
    )
    planned_bytes = len(economic_columns) * int(oof.height) * 8
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
        "calibration", planned_bytes=planned_bytes, envelope=envelope.to_dict()
    )
    if not envelope.ok:
        raise _MemoryBudgetExceededError("calibration")

    calibrator = _causal_calibrator(request, horizon_sessions)
    schedule = SessionClusterCalibrationSchedule(
        ledger,
        calibrator,
        request.base_cost_schedule or default_base_schedule(),
        block_length=horizon_sessions,
        max_workspace_bytes=_schedule_workspace(request),
    )

    drop_columns = (*economic_columns, "__bucket")
    session_physical = oof[SESSION_COLUMN].to_physical().to_numpy()
    order = np.argsort(session_physical, kind="stable")
    sorted_sessions = session_physical[order]
    boundaries = np.flatnonzero(np.diff(sorted_sessions)) + 1

    economic_values = [
        np.empty(int(oof.height), dtype=np.float64) for _ in economic_columns
    ]
    economic_nulls = [
        np.zeros(int(oof.height), dtype=bool) for _ in economic_columns
    ]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    # Chunks are ranges of *sorted* positions; ``order`` maps them back to
    # caller rows exactly once.
    sorted_positions = np.arange(int(order.size))
    for group in np.split(sorted_positions, boundaries):
        # Contiguous ascending session slices keep schedule.state_at monotone.
        decision_us = int(sorted_sessions[group[0]])
        decision_time = epoch + timedelta(microseconds=decision_us)
        state = schedule.state_at(decision_time)
        rows_here = order[group]
        scored = (
            oof[rows_here]
            .rename({SCORE_COLUMN: "score"})
            .drop(*drop_columns, strict=False)
        )
        augmented = CausalAlphaCalibrator.apply_prepared(state, scored)
        for column_index, column in enumerate(economic_columns):
            values = augmented[column].cast(pl.Float64)
            economic_values[column_index][rows_here] = values.to_numpy()
            economic_nulls[column_index][rows_here] = (
                ~values.is_not_null().to_numpy()
            )
        del scored, augmented

    calibrated_columns: list[object] = []
    for column_index, column in enumerate(economic_columns):
        values = pl.Series(column, economic_values[column_index])
        if economic_nulls[column_index].any():
            revealed = pl.Series(
                f"{column}__revealed", ~economic_nulls[column_index]
            )
            calibrated_columns.append(
                pl.when(revealed).then(values).alias(column)
            )
        else:
            calibrated_columns.append(values)
    del economic_values, economic_nulls
    return oof.with_columns(calibrated_columns)


def _zero_calibrated(scored: pl.DataFrame) -> pl.DataFrame:
    """Cash-only calibration output: zero economic scores, no exception."""
    columns = {
        "expected_net_alpha": 0.0,
        "net_alpha_lower_bound": 0.0,
        "expected_active_alpha": 0.0,
        "alpha_lower_bound": 0.0,
        "exit_cost_rate": 0.0,
    }
    return scored.with_columns(
        pl.lit(value, dtype=pl.Float64).alias(column)
        for column, value in columns.items()
    )


def _freeze_causal_calibration(
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    decision_time: datetime,
) -> CausalCalibrationAdapter:
    """Freeze the causal calibration state at ``decision_time`` from OOF evidence."""
    ledger = _causal_ledger(oof_labels)
    calibrator = _causal_calibrator(request, horizon_sessions)
    schedule = SessionClusterCalibrationSchedule(
        ledger,
        calibrator,
        request.base_cost_schedule or default_base_schedule(),
        block_length=horizon_sessions,
        max_workspace_bytes=_schedule_workspace(request),
    )
    return CausalCalibrationAdapter(calibrator, schedule.state_at(decision_time))


def _empty_causal_calibration(
    request: NetAlphaTrainingRequest, horizon_sessions: int
) -> CausalCalibrationAdapter:
    """Cash-only calibration adapter for an empty holdout panel.

    Used only when the holdout has no realized rows, in which case the holdout
    gate fails closed; the adapter still emits the public prediction columns
    with zero economic scores.
    """
    state: dict[str, object] = {
        "bucket_count": int(request.risk.calibration_bucket_count),
        "history_sessions": 0,
        "round_trip_cost": 0.0,
        "exit_cost_rate": 0.0,
        "buckets": [],
    }
    return CausalCalibrationAdapter(_causal_calibrator(request, horizon_sessions), state)


def _on_demand_schema(learner_columns: tuple[str, ...]) -> FeatureTransformSchema:
    """Minimal schema view exposing only ``learner_columns`` for preparation."""
    from types import SimpleNamespace

    return cast(
        FeatureTransformSchema,
        SimpleNamespace(learner_columns=tuple(learner_columns)),
    )


_BLEND_PROFILE_SUFFIX = ":blend"
_BLEND_DROPOUT_UNAVAILABLE = "blend-scores-unavailable"


def _fold_score_diagnostics(
    oof: pl.DataFrame,
    ics: Sequence[float],
) -> tuple[FoldScoreDiagnostic, ...]:
    """Per-fold target-free score diagnostics in ascending segment order.

    ``rank_ic`` takes the positional session-mean Rank-IC from ``ics`` (empty
    when a family does not compute one); the failure reason stays empty for
    usable non-constant folds and reads ``constant-oof-score`` otherwise.
    """
    if oof.is_empty() or _OOF_SEGMENT not in oof.columns:
        return ()
    diagnostics: list[FoldScoreDiagnostic] = []
    segments = sorted(oof[_OOF_SEGMENT].unique().to_list())
    for index, segment in enumerate(segments):
        scores = oof.filter(pl.col(_OOF_SEGMENT) == segment)[SCORE_COLUMN]
        std_raw = scores.std() if scores.len() else None
        std = (
            float(std_raw)
            if isinstance(std_raw, (int, float))
            else 0.0
        )
        finite_count = int(scores.is_finite().sum()) if scores.len() else 0
        unique_count = int(scores.n_unique()) if scores.len() else 0
        rank_ic = float(ics[index]) if index < len(ics) else 0.0
        diagnostics.append(
            FoldScoreDiagnostic(
                fold_index=index,
                score_std=std if math.isfinite(std) else 0.0,
                finite_count=finite_count,
                unique_count=unique_count,
                rank_ic=rank_ic,
                failure_reason="" if std > 0.0 else "constant-oof-score",
            )
        )
    return tuple(diagnostics)


def _blend_calibrated_scores(
    frames: Mapping[int, pl.DataFrame], horizon_sessions: int
) -> pl.DataFrame | None:
    """Mean within-session percentile rank across horizons on the anchor frame.

    The anchor is the cached calibrated frame at ``horizon_sessions``; every
    other admitted horizon contributes only its per-session percentile rank of
    ``predicted_net_alpha`` inner-joined on identity keys, so names absent from
    any horizon never receive a zero-filled score. The returned frame keeps the
    anchor schema with ``predicted_net_alpha`` replaced by the blended rank;
    derived economic columns stay stale and must be recomputed by causal
    calibration before replay.
    """
    anchor = frames.get(horizon_sessions)
    if anchor is None or anchor.is_empty():
        return None
    others = [
        frame
        for horizon, frame in sorted(frames.items())
        if horizon != horizon_sessions and not frame.is_empty()
    ]
    if not others:
        return None
    keys = [_ID_COLUMN, SESSION_COLUMN, _SESSION_IDX, _OOF_SEGMENT]
    stacked_frames = []
    for horizon, frame in sorted(frames.items()):
        stacked_frames.append(
            frame.select(*keys, SCORE_COLUMN).with_columns(
                (
                    pl.col(SCORE_COLUMN).rank("average").over(SESSION_COLUMN)
                    / pl.col(SCORE_COLUMN).count().over(SESSION_COLUMN)
                ).alias("__blend_rank"),
                pl.lit(int(horizon), dtype=pl.Int64).alias("__blend_horizon"),
            )
        )
    mean_rank = (
        pl.concat(stacked_frames, how="vertical")
        .group_by(keys)
        .agg(pl.col("__blend_rank").mean().alias("__blend_mean"))
    )
    blended = (
        anchor.join(mean_rank, on=list(keys), how="inner")
        .with_columns(pl.col("__blend_mean").alias(SCORE_COLUMN))
        .drop("__blend_mean")
    )
    if blended.is_empty():
        return None
    if (
        blended[SCORE_COLUMN].null_count() > 0
        or not bool(blended[SCORE_COLUMN].is_finite().all())
    ):
        raise ValueError("blended OOF scores must be finite for every row")
    return blended


def _fit_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    model_factory: Callable[[], Model] | None,
    *,
    family: str,
    matrix: PreparedTrainingMatrix | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic, int]:
    """Fit a learner per purged fold and collect target-free OOF predictions.

    The prepared-array fold loop in ``ml.fitting`` owns the discovery hot
    path: one canonical matrix, integer fold plans, exact array Rank-IC, and
    Polars OOF frames constructed once at the boundary. ``matrix`` may be
    supplied by the orchestrator; otherwise it is prepared on demand for this
    horizon.
    """
    from src.stocks.ml.fitting import OofFitRequest
    from src.stocks.ml.fitting import fit_horizon_oof as _fit_horizon_oof
    from src.stocks.ml.preparation import (
        prepare_horizon_labels,
        prepare_training_matrix,
    )

    if family not in DECLARED_ECONOMIC_FAMILIES:
        raise ValueError(
            "prepared-array OOF fitting owns the declared economic families only; "
            f"got {family!r}"
        )
    if not folds:
        empty = pl.DataFrame()
        diagnostic = HorizonOOFDiagnostic(
            horizon_sessions=horizon_sessions,
            model_family=family,
            fold_diagnostics=(),
        )
        return empty, empty, [], diagnostic, 0

    if family == RAWNET_LGBM_FAMILY:
        from src.stocks.ml.economic_research import (
            fit_rawnet_lgbm_oof,
            rawnet_fold_rank_ics,
        )

        oof, labeled = fit_rawnet_lgbm_oof(pre_holdout, folds, data, request, learner_columns, horizon_sessions)
        ics = rawnet_fold_rank_ics(labeled)
        diagnostic = HorizonOOFDiagnostic(
            horizon_sessions=horizon_sessions,
            model_family=family,
            fold_diagnostics=_fold_score_diagnostics(oof, ics),
        )
        return oof, labeled, ics, diagnostic, 0

    if family == TAIL_LAMBDARANK_FAMILY:
        from src.stocks.ml.economic_research import fit_tail_lambdarank_oof

        ks = [
            int(k)
            for h, _cadence, k in request.execution_frontier.feasible_cells(
                request.portfolio.max_exposure,
                request.portfolio.max_single_weight,
            )
            if h == horizon_sessions
        ]
        if not ks:
            raise ValueError(
                "tail LambdaRank OOF requires a feasible K cell at "
                f"horizon {horizon_sessions}"
            )
        oof, labeled = fit_tail_lambdarank_oof(
            pre_holdout, folds, data, request, learner_columns,
            horizon_sessions, min(ks),
        )
        diagnostic = HorizonOOFDiagnostic(
            horizon_sessions=horizon_sessions,
            model_family=family,
            fold_diagnostics=_fold_score_diagnostics(oof, ()),
        )
        return oof, labeled, [], diagnostic, 0

    if matrix is None:
        matrix = prepare_training_matrix(
            TrainingPanelView(pre_holdout),
            _on_demand_schema(learner_columns),
            tuple(folds),
        )
    horizon_labels = prepare_horizon_labels(matrix, data, horizon_sessions)
    result = _fit_horizon_oof(
        matrix,
        horizon_labels,
        prepare_folds(folds),
        OofFitRequest(request=request, manifest=base_manifest, family=family),
    )
    return (
        result.oof,
        result.labeled,
        result.fold_rank_ics,
        result.diagnostic,
        result.path_evaluations,
    )


def _median_best_iteration(
    train: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    base_manifest: ModelManifest,
) -> int | None:
    """Median LightGBM best iteration over purged inner labeled validation."""
    # Challenger nested fitting honors the same rolling lookback cap as the
    # outer discovery plan.
    nested_splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
        max_train_sessions=request.max_training_lookback_sessions,
    )
    nested = nested_splitter.inner_folds(train, n_inner=_NESTED_INNER_FOLDS)
    iterations: list[int] = []
    for inner in nested:
        inner_train = train[inner.train_mask]
        inner_val = train[inner.validation_mask]
        if inner_train.is_empty() or inner_val.is_empty():
            continue
        if (
            TARGET_COLUMN not in inner_train.columns
            or TARGET_COLUMN not in inner_val.columns
        ):
            continue
        challenger = LightGbmNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
            num_threads=request.model_threads,
        )
        try:
            challenger.fit(inner_train, inner_val)
        except ValueError:
            continue
        best = challenger.best_iteration
        if best is not None and best > 0:
            iterations.append(best)
    if not iterations:
        return None
    return int(np.median(iterations))


def _challenger_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Deterministic LightGBM OOF predictions on the selected primary.

    Runs for at most one horizon (the primary). Early stopping uses only inner
    labeled validation; the median inner best iteration is recorded and each
    outer model is refit to that fixed count. Outer validation remains
    target-free.
    """
    label_join = _build_label_join(data, primary_horizon_sessions)
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    rank_ics: list[float] = []
    fold_diagnostics: list[FoldScoreDiagnostic] = []
    median_iteration: int | None = None
    for fold in folds:
        train = pre_holdout[fold.train_mask].join(
            label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner",
        )
        if train.is_empty():
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold.segment_id, failure_reason="empty-fold"
                )
            )
            continue
        fold_median = _median_best_iteration(
            train, request, learner_columns, primary_horizon_sessions, base_manifest
        )
        if fold_median is not None:
            median_iteration = (
                fold_median
                if median_iteration is None
                else int(np.median([median_iteration, fold_median]))
            )
    if median_iteration is None or median_iteration < 1:
        return (
            pl.DataFrame(), pl.DataFrame(), [],
            HorizonOOFDiagnostic(
                horizon_sessions=primary_horizon_sessions,
                model_family="net_alpha_lightgbm_l1",
                failure_reason="no-inner-best-iteration",
            ),
        )
    for fold_index, fold in enumerate(folds):
        train = pre_holdout[fold.train_mask].join(
            label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner",
        )
        validation = pre_holdout[fold.validation_mask]
        if train.is_empty() or validation.is_empty():
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index, failure_reason="empty-fold"
                )
            )
            continue
        challenger = LightGbmNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
            num_threads=request.model_threads,
        )
        try:
            challenger.fit(
                train, validation, num_boost_round=median_iteration
            )
        except ValueError as exc:
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index,
                    failure_reason=f"fit-error:{type(exc).__name__}:{exc}",
                )
            )
            continue
        scored = challenger.predict(validation)
        scores = scored[SCORE_COLUMN].to_numpy().astype(float)
        finite_scores = scores[np.isfinite(scores)]
        score_std = float(np.std(finite_scores)) if finite_scores.size else 0.0
        unique_count = int(np.unique(finite_scores).size) if finite_scores.size else 0
        joined = scored.join(
            validation.select(_ID_COLUMN, SESSION_COLUMN, _SESSION_IDX),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="left",
        ).with_columns(pl.lit(fold.segment_id, dtype=pl.Int64).alias(_OOF_SEGMENT))
        labeled = joined.join(label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
        if labeled.is_empty():
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index,
                    score_std=score_std,
                    finite_count=int(finite_scores.size),
                    unique_count=unique_count,
                    failure_reason="no-labeled-join",
                )
            )
            continue
        rank_ic = _rank_ic(labeled)
        oof_frames.append(joined)
        label_frames.append(labeled)
        rank_ics.append(rank_ic)
        fold_diagnostics.append(
            FoldScoreDiagnostic(
                fold_index=fold_index,
                score_std=score_std,
                finite_count=int(finite_scores.size),
                unique_count=unique_count,
                rank_ic=rank_ic,
            )
        )
    diagnostic = HorizonOOFDiagnostic(
        horizon_sessions=primary_horizon_sessions,
        model_family="net_alpha_lightgbm_l1",
        fold_diagnostics=tuple(fold_diagnostics),
    )
    if not oof_frames:
        return pl.DataFrame(), pl.DataFrame(), [], diagnostic
    return pl.concat(oof_frames), pl.concat(label_frames), rank_ics, diagnostic


def _adopt_model_family(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
    profile: PolicyProfile,
    selection: HorizonSelectionEvidence,
    baseline_oof: pl.DataFrame,
    baseline_labels: pl.DataFrame,
    baseline_ics: list[float],
    baseline_diag: HorizonOOFDiagnostic,
    rankability_reason: str,
    *,
    registry: ModelArtifactRegistry | None = None,
) -> tuple[str, str, pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Conditionally adopt the LightGBM challenger on the selected primary.

    The challenger is eligible only when the linear screen is rankable. It
    replaces the baseline only when its stress-cost adjusted lower growth
    strictly improves the selected profile's baseline stress adjusted lower
    growth at the same Holm threshold (the challenger must beat the exact
    policy that was selected). Otherwise the ElasticNet baseline remains. A
    skipped challenger preserves the valid baseline OOF so trading continues.
    """
    assert selection.primary_rebalance_frequency_sessions is not None
    assert selection.primary_top_k is not None
    profile_key = (
        primary_horizon_sessions,
        selection.primary_rebalance_frequency_sessions,
        selection.primary_top_k,
        profile.profile_id,
    )
    if rankability_reason:
        return (
            "net_alpha_elastic_net",
            rankability_reason,
            baseline_oof,
            baseline_labels,
            baseline_ics,
            baseline_diag,
        )
    challenger_oof, challenger_labels, challenger_ics, challenger_diag = _challenger_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary_horizon_sessions,
    )
    if challenger_oof.is_empty() or challenger_labels.is_empty():
        return "net_alpha_elastic_net", "", baseline_oof, baseline_labels, baseline_ics, baseline_diag
    challenger_calibrated = _causal_oof_calibrate(
        challenger_oof, challenger_labels, request, primary_horizon_sessions
    )
    risk = replace(request.risk, no_trade_band_bps=profile.no_trade_band_bps)
    try:
        challenger_replay = _replay_costs(
            _require_caller_registry(registry),
            challenger_calibrated, challenger_labels, request,
            primary_horizon_sessions, risk, pre_holdout, data.manifest, profile,
            rebalance_frequency_sessions=selection.primary_rebalance_frequency_sessions,
            top_k=selection.primary_top_k,
        )
    except ValueError as exc:
        return (
            "net_alpha_elastic_net",
            f"challenger-replay-error:{type(exc).__name__}:{exc}",
            baseline_oof, baseline_labels, baseline_ics, baseline_diag,
        )
    stress_growth = challenger_replay.candidate.stress_log_growth
    from src.stocks.ml.horizons import _cohort_bootstrap

    stress_threshold = selection.stress_holm_thresholds.get(
        profile_key, request.bootstrap_alpha
    )
    baseline_stress_lower = selection.adjusted_lower_growth.get(
        profile_key, {}
    ).get("stress", 0.0)
    bootstrap = _cohort_bootstrap(
        stress_growth,
        challenger_replay.candidate.segment_ids,
        request.bootstrap_resamples,
        request.seed + primary_horizon_sessions,
        min_block_length=max(
            primary_horizon_sessions,
            StockRiskPolicy().rebalance_frequency_sessions,
        ),
    )
    if bootstrap is None:
        return "net_alpha_elastic_net", "", baseline_oof, baseline_labels, baseline_ics, baseline_diag
    adjusted_stress_lower = bootstrap.lower_mean(stress_threshold)
    if adjusted_stress_lower > baseline_stress_lower:
        return (
            "net_alpha_lightgbm_l1",
            "",
            challenger_oof, challenger_labels, challenger_ics, challenger_diag,
        )
    return "net_alpha_elastic_net", "", baseline_oof, baseline_labels, baseline_ics, baseline_diag


def _rank_ic(frame: pl.DataFrame) -> float:
    if frame.is_empty() or SCORE_COLUMN not in frame.columns:
        return 0.0
    from scipy.stats import spearmanr

    sub = frame.filter(
        pl.col(SCORE_COLUMN).is_not_null()
        & pl.col(REALIZED_RETURN_COLUMN).is_not_null()
    )
    if sub.is_empty():
        return 0.0
    ics: list[float] = []
    for rows in sub.sort("session").partition_by("session"):
        if rows.height < 2:
            continue
        scores = rows[SCORE_COLUMN].to_numpy().astype(float)
        labels = rows[REALIZED_RETURN_COLUMN].to_numpy().astype(float)
        if np.std(scores) == 0.0 or np.std(labels) == 0.0:
            continue
        rho, _ = spearmanr(scores, labels)
        ics.append(float(rho))
    return float(np.mean(ics)) if ics else 0.0


def _base_manifest(
    request: NetAlphaTrainingRequest,
    data: NetAlphaResearchData,
    frame: pl.DataFrame,
    primary_horizon_sessions: int,
) -> ModelManifest:
    eligible_from, eligible_to = _eligibility(frame)
    return ModelManifest(
        artifact_id=request.artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set=CANONICAL_FEATURE_SET,
        feature_schema_hash=data.manifest.schema_hash or "net-alpha-v1",
        universe_policy_hash=data.manifest.universe_policy_hash or "net-alpha-v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=primary_horizon_sessions,
        eligible_from=eligible_from,
        eligible_to=eligible_to,
        model_type="net_alpha_elastic_net",
    )


def _eligibility(frame: pl.DataFrame) -> tuple[str, str]:
    sessions = sorted(frame["session"].unique().to_list())
    if not sessions:
        raise ValueError("no sessions available for eligibility")
    first = sessions[0]
    last = sessions[-1]
    end = (
        last
        if isinstance(last, datetime)
        else datetime.combine(last, datetime.min.time(), tzinfo=UTC)
    )
    return first.isoformat(), end.isoformat()


def _apply_final_refit_lookback(
    pre_holdout: pl.DataFrame,
    train: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    primary_horizon_sessions: int,
) -> pl.DataFrame:
    """Apply the request's purge-safe rolling suffix to final refit rows."""
    lookback = request.max_training_lookback_sessions
    if lookback is None:
        return train
    if _SESSION_IDX not in pre_holdout.columns or _SESSION_IDX not in train.columns:
        raise ValueError("final refit lookback requires session_index columns")

    # The newest pre-holdout decision is the validation boundary. Keep only
    # rows whose label interval plus embargo ends before that boundary.
    boundary = int(cast(int, pre_holdout[_SESSION_IDX].max()))
    label_horizon = int(primary_horizon_sessions) + 1
    eligible_end = boundary - label_horizon - request.embargo_sessions
    train = train.filter(pl.col(_SESSION_IDX) <= eligible_end)
    if train.is_empty():
        return train
    sessions = sorted(train[_SESSION_IDX].unique().to_list())
    if len(sessions) > lookback:
        train = train.filter(pl.col(_SESSION_IDX) >= sessions[-lookback])
    return train


def _refit_selected(
    pre_holdout: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
    selected_model_type: str,
) -> Model | None:
    """Refit the single selected family on all pre-holdout history only."""
    label_join = _build_label_join(data, primary_horizon_sessions)
    train = pre_holdout.join(
        label_join.select(
            _ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN,
            AVAILABLE_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN,
            REALIZED_RETURN_COLUMN,
        ),
        on=[_ID_COLUMN, SESSION_COLUMN],
        how="inner",
    )
    train = _apply_final_refit_lookback(
        pre_holdout, train, request, primary_horizon_sessions
    )
    if train.is_empty():
        return None
    if selected_model_type == "net_alpha_lightgbm_l1":
        model: Model = LightGbmNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
            num_threads=request.model_threads,
        )
    else:
        selected_alpha, selected_fraction, alpha_max, _path_count = _select_elastic_alpha(
            train, request, learner_columns, primary_horizon_sessions,
            RegularizationGrid(), base_manifest,
        )
        if selected_alpha is None:
            return None
        model = ElasticNetNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(
                seed=request.seed,
                elastic_alpha=selected_alpha,
                elastic_alpha_fraction=selected_fraction,
                elastic_alpha_max=alpha_max,
            ),
        )
    try:
        model.fit(train, train.head(0))
    except ValueError:
        return None
    return model


def _evaluate_forward_holdout(
    model: Model,
    calibration: CalibrationApplier,
    holdout_panel: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    profile: PolicyProfile,
    *,
    rebalance_frequency_sessions: int,
    top_k: int,
    registry: ModelArtifactRegistry | None = None,
) -> dict[str, object]:
    """Certify the untouched forward holdout on true base/stress equity.

    The locked holdout is scored target-free once by the pre-holdout model and
    the fitted calibration attaches the decimal economic columns. The identical
    calibrated frame is then replayed through the execution-equivalent adapter
    under the base and stress cost/liquidity schedules and the selected profile
    band, and the compound certificate (true equity lower-CAGR, observed and
    filled-cycle coverage, drawdown) gates promotion. A missing execution input,
    score/market mismatch, or parity error fails closed as a no-trade diagnosis
    and is never zero-filled. No gate is ever relaxed after observing the
    holdout.
    """
    if holdout_panel.is_empty():
        return {"passed": False, "reason": "holdout-has-no-realized"}
    scored = model.predict(holdout_panel)
    calibrated = calibration.apply(scored)
    calibrated = calibrated.with_columns(pl.lit(0, dtype=pl.Int64).alias(_OOF_SEGMENT))
    risk = replace(request.risk, no_trade_band_bps=profile.no_trade_band_bps)
    try:
        replay = _replay_costs(
            _require_caller_registry(registry),
            calibrated, calibrated, request, horizon_sessions, risk,
            holdout_panel, _holdout_stub_manifest(request, holdout_panel),
            profile,
            rebalance_frequency_sessions=rebalance_frequency_sessions,
            top_k=top_k,
        )
    except ValueError as exc:
        return {"passed": False, "reason": f"holdout-replay-invalid:{exc}"}
    base_evidence = replay.candidate
    stress_evidence = replay.candidate
    holdout_sessions = tuple(sorted(holdout_panel[SESSION_COLUMN].unique().to_list()))
    interval_exposure = base_evidence.base_interval_exposure
    if len(holdout_sessions) == len(interval_exposure) + 1:
        interval_exposure = (*interval_exposure, 0.0)
    try:
        matched_benchmark = exposure_matched_benchmark_log_growth(
            holdout_panel,
            holdout_sessions,
            interval_exposure,
        )
    except ValueError as exc:
        return {"passed": False, "reason": f"holdout-benchmark-invalid:{exc}"}
    growth_count = len(base_evidence.base_log_growth)
    invested_sessions = base_evidence.invested_interval_count
    # Tail vintages maturing past the dataset end are structurally censored;
    # allow exactly one horizon of tail sessions in the observation rule.
    holdout_settings = replace(
        request.compounding,
        allowed_tail_censoring_sessions=max(
            int(horizon_sessions), request.compounding.allowed_tail_censoring_sessions
        ),
    )
    # The effective floor is derived inside the certification kernel from
    # holdout_settings.allowed_tail_censoring_sessions; keep the value honest.
    assert (
        holdout_settings.allowed_tail_censoring_sessions >= horizon_sessions
    )
    certificate = certify_compounded_holdout(
        tuple(np.expm1(value) for value in base_evidence.base_log_growth),
        tuple(np.expm1(value) for value in stress_evidence.stress_log_growth),
        horizon_sessions,
        growth_count,
        invested_sessions,
        holdout_settings,
    )
    matched_certificate = certify_exposure_matched_excess(
        base_evidence.base_log_growth,
        matched_benchmark,
        horizon_sessions,
        invested_sessions,
        holdout_settings,
    )
    if base_evidence.filled_orders <= 0 or base_evidence.planned_cycles <= 0:
        reason = "holdout-no-economic-edge"
    elif not certificate.passed:
        reason = (
            "holdout-compound-certification-failed:"
            + ";".join(certificate.reasons)
        )
    elif not matched_certificate.passed:
        reason = (
            "holdout-relative-certification-failed:"
            + ";".join(matched_certificate.reasons)
        )
    else:
        reason = ""
    return {
        "passed": reason == "",
        "reason": reason,
        "evaluation_kind": "prepared-equity-v2-economic-rank",
        "block_count": growth_count,
        "order_count": base_evidence.filled_orders,
        "certificate": certificate.to_json(),
        "matched_certificate": matched_certificate.to_json(),
        "matched_benchmark_log_growth": matched_benchmark,
        "cohorts": {
            "scored_sessions": growth_count,
            "realized_sessions": growth_count,
            "eligible_sessions": base_evidence.planned_cycles,
            "active_sessions": invested_sessions,
            "orders": base_evidence.filled_orders,
            "period_count": growth_count,
            "observed_sessions": growth_count,
            "active_cohort_count": invested_sessions,
            "missing_realized_cohorts": 0,
        },
        "diagnostics": {
            "base": base_evidence.diagnostics(),
            "stress": stress_evidence.diagnostics(),
        },
    }


def _holdout_stub_manifest(
    request: NetAlphaTrainingRequest, frame: pl.DataFrame
) -> DatasetManifest:
    """Manifest projection for the untouched holdout's execution replay."""
    sessions = sorted(frame["session"].unique().to_list())
    first = sessions[0]
    last = sessions[-1]
    return DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="net-alpha-v1",
        provider_version="net-alpha",
        universe_policy_version="net-alpha-v1",
        universe_policy_hash="net-alpha-v1",
        feature_set=CANONICAL_FEATURE_SET,
        feature_set_hash="net-alpha-v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=request.candidate_horizon_sessions[0],
        time_start=first,
        time_end=last,
        generated_time=last,
        row_count=frame.height,
    )


def _no_trade_model(
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    label_column: str,
) -> Model:
    """Deterministic all-zero net-alpha ``NO_TRADE`` model."""
    del label_column
    manifest = replace(base_manifest, model_type="no_trade")
    return NoTradeModel(manifest, learner_columns)


class NoTradeModel:
    """Deterministic all-zero ``NO_TRADE`` model."""

    def __init__(self, manifest: ModelManifest, learner_columns: tuple[str, ...]):
        self._manifest = manifest
        self._learner_columns = learner_columns
        self.no_trade: bool = True

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        del train, validation

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias(SCORE_COLUMN)
        )

    def manifest(self) -> ModelManifest:
        return replace(
            self._manifest,
            params={
                "no_trade": "true",
                "feature_columns": ",".join(self._learner_columns),
            },
        )


def _publish_no_trade(
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
    frame: pl.DataFrame,
    reason: str,
    *,
    details: object = "",
    schema_hash: str = "no-trade",
    universe_policy_hash: str = "no-trade",
    telemetry: TrainingTelemetry | None = None,
    policy_frontier: Mapping[str, object] | None = None,
    growth_route: Mapping[str, object] | None = None,
) -> ModelManifest:
    """Publish a complete immutable ``NO_TRADE`` artifact with evidence."""
    eligible_from, eligible_to = _eligibility(frame)
    manifest = ModelManifest(
        artifact_id=request.artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set=CANONICAL_FEATURE_SET,
        feature_schema_hash=schema_hash,
        universe_policy_hash=universe_policy_hash,
        label_definition="net_alpha_o2o",
        label_horizon_sessions=request.candidate_horizon_sessions[0],
        eligible_from=eligible_from,
        eligible_to=eligible_to,
        model_type="no_trade",
        params=(
            {"no_trade": "true", "enable_excess_route": "true"}
            if request.enable_excess_route
            else {"no_trade": "true"}
        ),
    )
    model = _no_trade_model(
        manifest,
        tuple(c for c in frame.columns if c.startswith("feature__")),
        "net_alpha",
    )
    registry.publish(model, manifest)
    if telemetry is not None:
        telemetry.phase(
            "artifact_publish",
            {
                "artifact_id": request.artifact_id,
                "model_type": "no_trade",
                "promoted": False,
                "no_trade": True,
                "reason": reason,
            },
        )
    metrics: dict[str, object] = {
        "promoted": False,
        "no_trade": True,
        "model_type": "no_trade",
        "promotion_reasons": (
            [reason]
            if isinstance(details, dict)
            else [f"{reason}:{details}".rstrip(":")]
        ),
        "gates": {"passed": False},
        "run_observability": (
            telemetry.to_dict()
            if telemetry is not None
            else {"phases": [], "horizons": []}
        ),
    }
    if isinstance(details, dict):
        metrics.update(details)
    if policy_frontier is not None:
        metrics["policy_frontier"] = policy_frontier
    if growth_route is not None:
        metrics["growth_route"] = growth_route
    registry.write_metrics(request.artifact_id, metrics)
    logger.info("published NO_TRADE artifact %s (%s)", request.artifact_id, reason)
    return manifest


def _policy_profile_params(
    request: NetAlphaTrainingRequest,
    profile: PolicyProfile,
    horizon_sessions: int,
    *,
    rebalance_frequency_sessions: int,
    top_k: int,
) -> str:
    """JSON projection of the selected immutable policy profile for the manifest.

    Persists the exact selected forecast horizon, rebalance cadence (C), and
    active-name count (K) so the independent simulator reconstructs an identical
    ``StockRiskPolicy`` fingerprint.
    """
    policy = _risk_policy_for_profile(
        request, profile, horizon_sessions,
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
    )
    execution_policy = request.execution_policy or SCHEDULED_OPEN_V1
    evidence_version = (
        "prepared-equity-v5-sparse-growth"
        if profile.execution_utility_mode == "sparse_hold_replace_v2"
        else "prepared-equity-v4-delta-cost-aware"
        if profile.execution_utility_mode == "delta_cost_aware_v1"
        else "prepared-equity-v3-horizon-consistent"
    )
    effective_active_count = math.ceil(
        request.portfolio.max_exposure / request.portfolio.max_single_weight
    )
    candidate_pool_count = 2 * effective_active_count
    return json.dumps(
        {
            "profile_id": profile.profile_id,
            "no_trade_band_bps": profile.no_trade_band_bps,
            "growth_risk_aversion": profile.growth_risk_aversion,
            "forecast_horizon_sessions": horizon_sessions,
            "rebalance_frequency_sessions": rebalance_frequency_sessions,
            "effective_active_count": effective_active_count,
            "candidate_pool_count": candidate_pool_count,
            "top_k": top_k,
            "max_single_weight": request.portfolio.max_single_weight,
            "max_exposure": request.portfolio.max_exposure,
            "participation_limit": request.portfolio.participation_limit,
            "portfolio_fingerprint": policy_portfolio_fingerprint(
                top_k,
                request.portfolio.max_single_weight,
                request.portfolio.max_exposure,
                request.portfolio.participation_limit,
            ),
            "execution_evidence_version": evidence_version,
            "growth_route_version": GROWTH_ROUTE_VERSION,
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "v7_risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "execution_policy_id": execution_policy.policy_id,
            "execution_policy_hash": execution_policy.canonical_hash,
            "economic_ranking_mode": policy.economic_ranking_mode,
            "execution_utility_mode": profile.execution_utility_mode,
            "sizing_mode": profile.sizing_mode,
            "retained_sizing_mode": policy.retained_sizing_mode,
            "vol_target_override": profile.vol_target_override,
            "participation_limit_override": profile.participation_limit_override,
            "turnover_budget_override": profile.turnover_budget_override,
            **(
                {}
                if profile.net_exposure_gate_mode == "off_v1"
                else {
                    "net_exposure_gate_mode": profile.net_exposure_gate_mode,
                    **(
                        {"gate_floor": profile.gate_floor}
                        if profile.gate_floor is not None
                        else {}
                    ),
                    **(
                        {
                            "gate_trend_lookback_sessions": profile.gate_trend_lookback_sessions
                        }
                        if profile.gate_trend_lookback_sessions is not None
                        else {}
                    ),
                }
            ),
        },
        sort_keys=True,
    )


def _build_metrics(
    request: NetAlphaTrainingRequest,
    evaluation: ExecutionReplayEvidence,
    fold_rank_ic: list[float],
    selection: HorizonSelectionEvidence,
    manifest: ModelManifest,
    *,
    profile: PolicyProfile,
    holdout_evidence: dict[str, object],
    telemetry: TrainingTelemetry,
    discovery: HorizonDiscovery,
    growth_route: Mapping[str, object] | None = None,
) -> dict[str, object]:
    annualization = request.compounding.annualization_sessions

    def annualized_cagr(lower_growth: float) -> float:
        return float(np.expm1(annualization * lower_growth))

    adjusted_lower_cagr = {
        f"{horizon}:{profile_id}": {
            path: annualized_cagr(bound)
            for path, bound in paths.items()
        }
        for (horizon, cadence, top_k, profile_id), paths in (
            selection.adjusted_lower_growth.items()
        )
    }
    metrics: dict[str, object] = {
        "promoted": manifest.model_type != "no_trade",
        "no_trade": manifest.model_type == "no_trade",
        "model_type": manifest.model_type,
        "primary_horizon_sessions": selection.primary_horizon_sessions,
        "primary_profile_id": selection.primary_profile_id,
        "selected_profile": {
            "profile_id": profile.profile_id,
            "no_trade_band_bps": profile.no_trade_band_bps,
        },
        "policy_frontier": _policy_frontier_projection(
            request, discovery, selection.primary_profile_id
        ),
        "mean_fold_rank_ic": float(np.mean(fold_rank_ic)) if fold_rank_ic else 0.0,
        "horizon_selection": selection.to_json(),
        "adjusted_lower_cagr": adjusted_lower_cagr,
        "path_evaluation_count": discovery.path_evaluation_count,
        "holdout": {
            **holdout_evidence,
            "eligibility": {
                "eligible_from": manifest.eligible_from,
                "eligible_to": manifest.eligible_to,
            },
        },
        "replay": evaluation.diagnostics(),
        "gates": {
            "passed": manifest.model_type != "no_trade",
            "reasons": list(selection.selection_reasons),
        },
        "run_observability": telemetry.to_dict(),
    }
    if growth_route is not None:
        metrics["growth_route"] = growth_route
    return metrics


def _blend_lower_growth_projection(
    discovery: HorizonDiscovery,
    *,
    bootstrap_alpha: float,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    """Bounded lower-growth scalars for ':blend' frontier candidates.

    Publishes one moving-block bootstrap lower mean per blended candidate so
    route-selection transparency exists for cross-horizon evidence; plain
    candidates are covered by the selection payload and never duplicated here.
    """
    blend_map: dict[str, object] = {}
    for candidate in discovery.evidence:
        if not str(candidate.profile_id).endswith(_BLEND_PROFILE_SUFFIX):
            continue
        key = (
            f"{candidate.horizon_sessions}:{candidate.rebalance_frequency_sessions}:"
            f"{candidate.top_k}:{candidate.profile_id}"
        )
        block_length = max(1, min(candidate.horizon_sessions, len(candidate.base_log_growth)))
        base_arr = np.asarray(candidate.base_log_growth, dtype=float)
        stress_arr = np.asarray(candidate.stress_log_growth, dtype=float)
        blend_map[key] = {
            "base_lower_growth": round(
                float(
                    _bootstrap_lower_mean_log_growth(
                        base_arr, block_length, bootstrap_resamples, seed, bootstrap_alpha
                    )
                ),
                12,
            ),
            "stress_lower_growth": round(
                float(
                    _bootstrap_lower_mean_log_growth(
                        stress_arr,
                        block_length,
                        bootstrap_resamples,
                        seed + candidate.horizon_sessions,
                        bootstrap_alpha,
                    )
                ),
                12,
            ),
        }
    return blend_map


def _policy_frontier_projection(
    request: NetAlphaTrainingRequest,
    discovery: HorizonDiscovery,
    selected_profile_id: str | None,
) -> dict[str, object]:
    """Bounded ``policy_frontier`` projection shared by metrics and no-trade.

    Records the candidate count, profile ids, per-``(horizon, profile)``
    dropout reasons, and the bounded execution evidence of every evaluated
    candidate. Raw orders, scores, returns, and model predictions are never
    included.
    """
    return {
        "candidate_count": len(discovery.evidence),
        "profile_ids": [p.profile_id for p in request.policy_profiles],
        "dropout_reasons": {
            f"{horizon}:{cadence}:{top_k}:{profile_id}": reason
            for (horizon, cadence, top_k, profile_id), reason in sorted(
                discovery.dropout_reasons.items()
            )
        },
        "execution_evidence": _segment_summaries(
            discovery.execution_evidence_by_candidate, selected_profile_id
        ),
        "sizing_diagnostics": {
            f"{horizon}:{cadence}:{top_k}:{profile_id}": dict(summary)
            for (horizon, cadence, top_k, profile_id), summary in sorted(
                discovery.sizing_diagnostics_by_candidate.items()
            )
        },
        "blend_lower_growth": _blend_lower_growth_projection(
            discovery,
            bootstrap_alpha=request.bootstrap_alpha,
            seed=request.seed,
            bootstrap_resamples=request.bootstrap_resamples,
        ),
    }


def _segment_summaries(
    execution_evidence_by_candidate: Mapping[
        tuple[int, int, int, str], ExecutionReplayEvidence
    ],
    selected_profile_id: str | None,
) -> dict[str, object]:
    """Bounded per-candidate execution evidence for the selected profile.

    Only the candidate selected under ``selected_profile_id`` is projected as
    ``"h<horizon>:<profile>"`` entries carrying bounded planned/filled cycle
    counts, cash coverage, and turnover; no score or return array is ever
    emitted.
    """
    summaries: dict[str, object] = {}
    for (horizon, cadence, top_k, profile_id), evidence in sorted(
        execution_evidence_by_candidate.items()
    ):
        if selected_profile_id is not None and profile_id != selected_profile_id:
            continue
        summaries[f"h{horizon}:{cadence}:{top_k}:{profile_id}"] = evidence.diagnostics()
    return summaries


def _policy_key_label(key: tuple[int, int, int, str] | None) -> str | None:
    if key is None:
        return None
    horizon, cadence, top_k, profile_id = key
    return f"{horizon}:{cadence}:{top_k}:{profile_id}"


def _blend_champion_no_trade(
    growth_route: Mapping[str, object],
    certificate: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Fail-closed champion arguments for a blended final policy.

    A blended champion has no single-artifact holdout representation yet, so
    artifact promotion stops pre-holdout. When the certified route carries a
    positive matched-excess lower bound (``PROMOTABLE_EXCESS``), the verdict is
    published as a research-only excess outcome instead of plain NO_TRADE;
    ``promoted`` stays False and the absolute-leg rejection reasons are
    retained verbatim in the payload.
    """
    status = growth_route.get("promotion_status")
    if status in ("PROMOTABLE_EXCESS", "PROMOTED_EXCESS_SLEEVE"):
        return (
            "blend-champion-excess-verdict",
            {
                "growth_route": growth_route,
                "growth_route_certificate": dict(certificate),
            },
        )
    return (
        "blend-champion-holdout-unsupported",
        {
            "growth_route": growth_route,
            "growth_route_certificate": dict(certificate),
        },
    )


def _growth_route_projection(
    route: GrowthRouteEvidence,
    certificate: Mapping[str, object],
    *,
    compounding: CompoundingCertificationSettings | None = None,
    horizon_sessions: int | None = None,
    capital_plan_settings: SmallCapitalPlanSettings | None = None,
) -> dict[str, object]:
    """Bounded ``growth_route`` projection shared by metrics and the ledger.

    Records the route version, candidate count, per-segment policy digest,
    final selected policy, absolute/relative lower growth, coverage, fills,
    drawdown, and normalized rejection-reason counts. Raw scores, order rows,
    per-instrument values, and return vectors are never included.

    When pre-registered ``compounding`` governance is supplied, the certified
    hedged-excess sleeve verdict upgrades a research-only excess outcome to
    ``PROMOTED_EXCESS_SLEEVE``; the artifact promotion path itself stays
    untouched (fail-closed).
    """
    route_reasons = certificate.get("reasons")
    reasons = (
        [str(reason) for reason in route_reasons if str(reason)]
        if isinstance(route_reasons, (list, tuple))
        else []
    )
    rejection_counts: dict[str, int] = {}
    for reason in reasons:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    def _digest_policies() -> dict[str, object]:
        labels = [_policy_key_label(key) for key in route.selected_policies]
        payload = json.dumps(labels, separators=(",", ":"))
        return {
            "count": len(labels),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    def _scalar(name: str) -> float | None:
        value = certificate.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 12)
        return None

    blocking_gate_reasons = {
        "no-filled-orders",
        "insufficient-observed-sessions",
        "invested-coverage-insufficient",
        "max-drawdown-exceeded",
        "period-series-incomplete",
    }
    gate_failures = [reason for reason in reasons if reason in blocking_gate_reasons]
    matched_lower = _scalar("matched_lower_excess_cagr")

    # Certified hedged-excess sleeve: computed only under injected governance
    # with clean blocking gates and an attached benchmark; legacy positional
    # callers keep byte-identical output.
    sleeve_certificate: dict[str, object] | None = None
    if (
        compounding is not None
        and not gate_failures
        and route.base_log_growth_minus_benchmark
    ):
        effective_horizon = int(horizon_sessions) if horizon_sessions is not None else 0
        if effective_horizon < 1 and route.selected_policies:
            final_key = route.selected_policies[-1]
            if final_key is not None:
                effective_horizon = int(final_key[0])
        if effective_horizon >= 1:
            try:
                sleeve_certificate = certify_hedged_excess_route(
                    route, effective_horizon, compounding
                )
            except ValueError:
                sleeve_certificate = None

    sleeve_promotable = bool(
        sleeve_certificate is not None and sleeve_certificate.get("passed")
    )
    if bool(certificate.get("passed")):
        promotion_status = "PROMOTED"
    elif sleeve_promotable and not route.benchmark_reconcile_failure:
        promotion_status = "PROMOTED_EXCESS_SLEEVE"
    elif (
        matched_lower is not None
        and matched_lower > 0.0
        and not route.benchmark_reconcile_failure
        and not gate_failures
    ):
        promotion_status = "PROMOTABLE_EXCESS"
    else:
        promotion_status = "NO_TRADE"

    hedge_projection: dict[str, object] = {}
    if route.base_log_growth_minus_benchmark:
        try:
            projection_payload = project_hedge_sleeve(
                route.base_log_growth_minus_benchmark,
                leverage_grid=(
                    compounding.hedge_leverage_grid
                    if compounding is not None
                    and compounding.hedge_leverage_grid is not None
                    else (1.0, 1.5, 2.0)
                ),
                vol_managed_lookback=26,
                vol_managed_target_annualized_vol=0.10,
            )
            ladder_rows = cast(
                'list[dict[str, object]]', projection_payload['leverage_ladder']
            )

            def _best_admissible_rung(
                variant: str,
            ) -> dict[str, object] | None:
                rows = [
                    row
                    for row in ladder_rows
                    if row["variant"] == variant and bool(row.get("admissible"))
                ]
                if not rows:
                    return None
                best = max(rows, key=lambda row: float(cast('float', row['leverage'])))
                return {
                    "leverage": round(float(cast('float', best['leverage'])), 12),
                    "point_cagr": round(float(cast('float', best['point_cagr'])), 12),
                    "stress_cagr": round(float(cast('float', best['stress_cagr'])), 12),
                    "projected_mdd": round(
                        float(cast('float', best['projected_mdd'])), 12
                    ),
                    "margin_buffer": round(
                        float(cast('float', best['margin_buffer'])), 12
                    ),
                }

            best_rungs = {
                variant: rung
                for variant, rung in (
                    ("static", _best_admissible_rung("static")),
                    ("vol_managed", _best_admissible_rung("vol_managed")),
                )
                if rung is not None
            }
            static_admissible = [
                cast("float", rung["leverage"])
                for rung in ladder_rows
                if rung["variant"] == "static" and rung["admissible"]
            ]
            volman_admissible = [
                cast("float", rung["leverage"])
                for rung in ladder_rows
                if rung["variant"] == "vol_managed" and rung["admissible"]
            ]
            hedge_projection = {
                "leverage_rung_count": len(ladder_rows) // 2,
                "max_admissible_leverage": (
                    max(static_admissible) if static_admissible else None
                ),
                "vol_managed_max_admissible_leverage": (
                    max(volman_admissible) if volman_admissible else None
                ),
                "admissible_rung_count": len(static_admissible),
                "excess_point_cagr": round(
                    cast("float", projection_payload["excess_point_cagr"]), 12
                ),
                "best_rungs": best_rungs,
            }
        except ValueError:
            hedge_projection = {}

    projection = {
        "version": route.route_version,
        "promotion_status": promotion_status,
        "candidate_count": int(route.candidate_count),
        "segment_count": len(route.selected_policies),
        "cash_segment_count": sum(
            1 for key in route.selected_policies if key is None
        ),
        "selected_policy": _policy_key_label(
            route.selected_policies[-1] if route.selected_policies else None
        ),
        "selected_policies_digest": _digest_policies(),
        "seed_policy": _policy_key_label(
            getattr(route, "seed_policy", None)
        ),
        "invested_interval_fraction": round(
            route.invested_interval_count / route.observed_interval_count, 12
        )
        if route.observed_interval_count > 0
        else 0.0,
        "base_lower_cagr": _scalar("base_lower_cagr"),
        "stress_lower_cagr": _scalar("stress_lower_cagr"),
        "matched_lower_excess_cagr": _scalar("matched_lower_excess_cagr"),
        "mdd": _scalar("mdd"),
        "observed_intervals": int(route.observed_interval_count),
        "invested_intervals": int(route.invested_interval_count),
        "filled_orders": int(route.filled_orders),
        "sparse_minus_dense_lower_growth": round(
            float(route.sparse_minus_dense_lower_growth), 12
        ),
        "turnover_ratio": (
            None if route.turnover_ratio is None else round(float(route.turnover_ratio), 12)
        ),
        "benchmark_reconcile_failure": str(route.benchmark_reconcile_failure)[:96],
        "hedge_sleeve_projection": hedge_projection,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
    }
    if sleeve_certificate is not None:
        projection["hedged_excess_certificate"] = sleeve_certificate
    if capital_plan_settings is not None:
        try:
            projection["small_capital_route_plan"] = build_small_capital_route_plan(
                route, capital_plan_settings
            )
        except ValueError:
            projection["small_capital_route_plan"] = {}
    return projection


def _attach_frozen_compound_track(
    growth_route: dict[str, object],
    evidence: tuple[HorizonOOFEvidence, ...],
    request: NetAlphaTrainingRequest,
) -> None:
    """Attach the always-invested frozen research track as bounded scalars.

    Research observability only: a missing frozen cell is omitted, never a
    ``_publish_no_trade`` reason, and the live prequential gate stays intact.
    """
    try:
        frozen = stitch_frozen_policy_growth_route(
            evidence, resolve_frozen_policy_key(request)
        )
    except ValueError:
        return
    growth_route["frozen_compound_track"] = frozen_compound_track_projection(
        frozen,
        annualization_sessions=request.compounding.annualization_sessions,
    )


_BLOCKING_SLEEVE_GATE_REASONS = frozenset(
    {
        "no-filled-orders",
        "insufficient-observed-sessions",
        "invested-coverage-insufficient",
        "max-drawdown-exceeded",
        "period-series-incomplete",
    }
)


def _attach_excess_route_certificate(
    growth_route: dict[str, object],
    discovery: HorizonDiscovery,
    request: NetAlphaTrainingRequest,
    panel: pl.DataFrame,
    primary_horizon: int,
) -> dict[str, object]:
    """Attach the opt-in excess-scoped route certificate to a projection.

    Stitches one parallel prequential route whose per-segment champions are
    selected on exposure-matched excess lower bounds instead of absolute
    growth, certifies it with the shared hedged-excess kernel, and publishes
    bounded scalars under ``excess_route``. A passing excess certificate
    upgrades a research-only outcome to ``PROMOTED_EXCESS_SLEEVE`` only when
    the primary sleeve verdict is absent or failed and no blocking gate
    reason survives; the artifact promotion path itself stays untouched.
    Flag-off callers receive the projection byte-identical.
    """
    if not bool(getattr(request, "enable_excess_route", False)):
        return growth_route
    benchmarks_by_key, _failures = _compute_candidate_benchmarks(discovery, panel)
    if not benchmarks_by_key:
        return growth_route
    try:
        excess_route = stitch_prequential_growth_route(
            discovery.evidence,
            request.bootstrap_alpha,
            request.seed,
            request.bootstrap_resamples,
            seed_policy=_seed_policy_or_none(request),
            benchmarks_by_key=benchmarks_by_key,
        )
        fills = 0
        for key in {k for k in excess_route.interval_policies if k is not None}:
            evidence = discovery.execution_evidence_by_candidate.get(key)
            if evidence is not None:
                fills += int(evidence.filled_orders)
        excess_route = replace(excess_route, filled_orders=fills)
        certificate = certify_hedged_excess_route(
            excess_route, max(int(primary_horizon), 1), request.compounding
        )
    except ValueError:
        return growth_route
    if not excess_route.base_log_growth:
        return growth_route

    raw_reasons = certificate.get("reasons", ())
    reasons = (
        [str(reason) for reason in raw_reasons if str(reason)]
        if isinstance(raw_reasons, (list, tuple))
        else []
    )
    reasons_payload = json.dumps(sorted(set(reasons)), separators=(",", ":"))
    labels = [_policy_key_label(key) for key in excess_route.selected_policies]
    policies_payload = json.dumps(labels, separators=(",", ":"))

    def _finite(value: object) -> float | None:
        parsed = float(value) if isinstance(value, (int, float)) else None
        return round(parsed, 12) if parsed is not None and math.isfinite(parsed) else None

    block: dict[str, object] = {
        "passed": bool(certificate.get("passed")),
        "reasons_digest": {
            "count": len(reasons),
            "sha256": hashlib.sha256(reasons_payload.encode("utf-8")).hexdigest(),
        },
        "route_version": str(excess_route.route_version),
        "excess_lower_cagr": _finite(certificate.get("excess_lower_cagr")),
        "sleeve_lower_stress_cagr": _finite(
            certificate.get("sleeve_lower_stress_cagr")
        ),
        "hedge_variant": str(certificate.get("hedge_variant", "")),
        "hedge_leverage": _finite(certificate.get("hedge_leverage")),
        "hedge_point_cagr": _finite(certificate.get("hedge_point_cagr")),
        "hedge_stress_cagr": _finite(certificate.get("hedge_stress_cagr")),
        "hedge_projected_mdd": _finite(certificate.get("hedge_projected_mdd")),
        "observed_intervals": int(excess_route.observed_interval_count),
        "invested_intervals": int(excess_route.invested_interval_count),
        "filled_orders": int(excess_route.filled_orders),
        "selected_policies_digest": {
            "count": len(labels),
            "sha256": hashlib.sha256(policies_payload.encode("utf-8")).hexdigest(),
        },
        "provenance": "excess-route-v1",
    }
    growth_route["excess_route"] = block

    raw_reason_counts = growth_route.get("rejection_reason_counts", {})
    reason_keys: set[str] = set()
    if isinstance(raw_reason_counts, dict):
        reason_keys = {str(key) for key in raw_reason_counts}
    blocking_present = bool(reason_keys & _BLOCKING_SLEEVE_GATE_REASONS)
    primary_sleeve = growth_route.get("hedged_excess_certificate")
    primary_passed = isinstance(primary_sleeve, dict) and bool(
        primary_sleeve.get("passed")
    )
    if (
        block["passed"]
        and not primary_passed
        and not blocking_present
        and not str(growth_route.get("benchmark_reconcile_failure", "") or "")
    ):
        sourced = dict(certificate)
        sourced["provenance"] = "excess-route-v1"
        growth_route["hedged_excess_certificate"] = sourced
        growth_route["promotion_status"] = "PROMOTED_EXCESS_SLEEVE"
    return growth_route


def _panel_within_sessions(
    panel: pl.DataFrame, sessions: tuple[datetime, ...]
) -> pl.DataFrame:
    """Scope the market panel to one replay segment's session bounds."""
    if panel.is_empty() or "session" not in panel.columns or not sessions:
        return panel
    return panel.filter(pl.col("session").is_in(list(sessions)))


def _compute_candidate_benchmarks(
    discovery: HorizonDiscovery,
    panel: pl.DataFrame,
) -> tuple[
    dict[tuple[int, int, int, str], tuple[float, ...]],
    dict[tuple[int, int, int, str], str],
]:
    """Compute per-candidate exposure-matched benchmark series.

    Pure extraction of the reconciliation loop shared by route benchmark
    attachment and the excess-scoped route: each candidate's replay evidence
    must carry interval exposures aligned with per-segment session bounds so
    one exposure-matched universe benchmark can be stitched. Failures emit a
    normalized reason per candidate key instead of fabricating series.
    """
    benchmarks: dict[tuple[int, int, int, str], tuple[float, ...]] = {}
    failures: dict[tuple[int, int, int, str], str] = {}
    sessions = (
        tuple(sorted(panel["session"].unique().to_list()))
        if (not panel.is_empty() and "session" in panel.columns)
        else ()
    )
    if not sessions:
        for key in discovery.execution_evidence_by_candidate:
            failures[key] = "benchmark-panel-missing"
        return benchmarks, failures
    for key, evidence in sorted(discovery.execution_evidence_by_candidate.items()):
        exposure = evidence.base_interval_exposure
        bounds_by_segment = getattr(
            evidence, "base_interval_session_bounds", ()
        )
        if not bounds_by_segment:
            failures[key] = "benchmark-session-bounds-missing"
            continue
        if sum(len(bounds) - 1 for bounds in bounds_by_segment) != len(exposure):
            failures[key] = "benchmark-exposure-length-mismatch"
            continue
        parts: list[float] = []
        failure = ""
        offset = 0
        for bounds in bounds_by_segment:
            segment_exposure = exposure[offset : offset + len(bounds) - 1]
            offset += len(bounds) - 1
            adjusted = (*segment_exposure, 0.0)
            if any(
                (not np.isfinite(value)) or value < 0.0 or value > 1.0
                for value in segment_exposure
            ):
                failure = "benchmark-exposure-invalid"
                break
            segment_panel = _panel_within_sessions(panel, bounds)
            try:
                benchmark = exposure_matched_benchmark_log_growth(segment_panel, bounds, adjusted)
            except ValueError:
                failure = "benchmark-series-invalid"
                break
            parts.extend(benchmark[: len(segment_exposure)])
        if failure:
            failures[key] = failure
            continue
        benchmark = tuple(parts)
        if len(benchmark) != len(evidence.base_log_growth):
            failures[key] = "benchmark-exposure-length-mismatch"
            continue
        benchmarks[key] = benchmark
    return benchmarks, failures


def _attach_growth_route_execution_evidence(
    route: GrowthRouteEvidence,
    discovery: HorizonDiscovery,
    panel: pl.DataFrame,
) -> GrowthRouteEvidence:
    """Attach bounded fills and an optional exposure-matched benchmark series.

    Fill totals aggregate the replay evidence of every distinct selected
    candidate. The benchmark is stitched from each selected candidate's own
    exposure-matched benchmark slice only when every selected candidate's
    interval exposures reconcile with the panel calendar; otherwise it stays
    empty so the relative gate fails closed instead of fabricating evidence.
    """
    keys = {key for key in route.interval_policies if key is not None}
    if not keys or not discovery.execution_evidence_by_candidate:
        return route
    fills = 0
    filled_cycles = 0
    for key in sorted(keys):
        evidence = discovery.execution_evidence_by_candidate.get(key)
        if evidence is None:
            continue
        fills += int(evidence.filled_orders)
        filled_cycles += int(evidence.filled_cycle_count)
    candidate_benchmarks, candidate_failures = _compute_candidate_benchmarks(
        discovery, panel
    )
    benchmarks: dict[tuple[int, int, int, str], tuple[float, ...]] = {
        key: series
        for key, series in candidate_benchmarks.items()
        if key in keys
    }
    benchmark_failure = ""
    benchmark_failed = False
    for key in sorted(keys):
        reason = candidate_failures.get(key)
        if reason is not None:
            benchmark_failed = True
            benchmark_failure = reason
    stitched_benchmark: tuple[float, ...] = ()
    if not benchmark_failed and benchmarks.keys() == keys:
        positions: dict[tuple[int, int, int, str], dict[int, list[int]]] = {}
        for candidate in discovery.evidence:
            source_key = (
                candidate.horizon_sessions,
                candidate.rebalance_frequency_sessions,
                candidate.top_k,
                candidate.profile_id,
            )
            if source_key not in keys:
                continue
            grouped: dict[int, list[int]] = {}
            for index, segment in enumerate(candidate.cohort_segment_ids):
                grouped.setdefault(int(segment), []).append(index)
            positions[source_key] = grouped
        counters: dict[tuple[tuple[int, int, int, str], int], int] = {}
        values: list[float] = []
        aligned = True
        for segment, interval_key in zip(
            route.segment_ids, route.interval_policies, strict=True
        ):
            if interval_key is None:
                values.append(0.0)
                continue
            offset = counters.get((interval_key, int(segment)), 0)
            indices = positions.get(interval_key, {}).get(int(segment))
            if indices is None or offset >= len(indices):
                aligned = False
                break
            counters[(interval_key, int(segment))] = offset + 1
            values.append(float(benchmarks[interval_key][indices[offset]]))
        if aligned and len(values) == len(route.base_log_growth):
            stitched_benchmark = tuple(values)
        else:
            benchmark_failure = "benchmark-series-invalid"
    elif not benchmark_failed:
        benchmark_failure = "benchmark-candidate-evidence-missing"
    return replace(
        route,
        filled_orders=fills,
        filled_cycle_count=filled_cycles,
        benchmark_log_growth=stitched_benchmark,
        benchmark_reconcile_failure=benchmark_failure[:96],
    )


def evaluate_growth_route_research(
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    *,
    registry: ModelArtifactRegistry,
    min_oof_train_sessions: int | None = None,
) -> dict[str, object]:
    """Read-only growth-route evaluation over one data snapshot.

    Replays the discovery frontier once, stitches the prequential growth
    route, certifies it, and returns a bounded payload. Nothing is published:
    no artifact, no manifest, no registry write. Every fail-closed gate emits
    a normalized rejection reason instead of fabricated growth.
    ``min_oof_train_sessions`` optionally raises the splitter's shared warm-up
    floor so multi-window studies share one first validation boundary; the
    ``None`` default preserves every existing caller and fold plan exactly.
    """
    frame = data.feature_frame
    raw_panel = _index_sessions(frame)
    pre_holdout_raw, _holdout_raw, holdout_reason = _locked_holdout(raw_panel, request)
    if holdout_reason or pre_holdout_raw.is_empty():
        return _growth_route_research_rejection(
            holdout_reason or "insufficient-pre-holdout-history"
        )
    roles = dict(stock_net_alpha_v1_roles())
    try:
        materialized = materialize_model_feature_sources(pre_holdout_raw, list(roles))
        schema = fit_model_feature_schema(materialized, roles)
        pre_holdout = apply_model_feature_schema(materialized, schema)
    except ValueError as exc:
        return _growth_route_research_rejection(f"feature-schema-failed:{exc}")
    learner_columns = schema.learner_columns
    if not learner_columns:
        return _growth_route_research_rejection("no-alpha-learner-columns")
    min_train_sessions = request.compounding.annualization_sessions
    if min_oof_train_sessions is not None:
        min_train_sessions = max(min_train_sessions, int(min_oof_train_sessions))
    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=max(request.candidate_horizon_sessions) + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=min_train_sessions,
        max_train_sessions=request.max_training_lookback_sessions,
    )
    folds = splitter.split(pre_holdout)
    if not folds:
        return _growth_route_research_rejection("insufficient-oof-calendar")
    cache = _OofCache(_default_oof_cache_base())
    try:
        discovery = _build_horizon_evidence(
            pre_holdout, folds, data, request, learner_columns,
            registry=registry, oof_cache=cache,
        )
    except (_MemoryBudgetExceededError, _EnvelopeBudgetError) as exc:
        stage = str(getattr(exc, "stage", "") or "fitting_workspace")
        return _growth_route_research_rejection(f"memory-budget-exceeded:{stage}")
    finally:
        cache.close()
    return _growth_route_research_payload(discovery, request, pre_holdout)


def _growth_route_research_payload(
    discovery: HorizonDiscovery,
    request: NetAlphaTrainingRequest,
    panel: pl.DataFrame,
) -> dict[str, object]:
    """Stitch, certify, and project one read-only research route."""
    if not discovery.evidence:
        return _growth_route_research_rejection("no-horizon-evidence")
    route = stitch_prequential_growth_route(discovery.evidence, request.bootstrap_alpha, request.seed, request.bootstrap_resamples, seed_policy=_seed_policy_or_none(request))  # noqa: E501
    route = _attach_growth_route_execution_evidence(route, discovery, panel)
    primary = (
        route.selected_policies[-1][0]
        if route.selected_policies and route.selected_policies[-1] is not None
        else discovery.evidence[0].horizon_sessions
    )
    certificate = certify_growth_route(route, primary, request.compounding)
    growth_route = _growth_route_projection(route, certificate, compounding=request.compounding, horizon_sessions=primary, capital_plan_settings=request.capital_plan)  # noqa: E501
    _attach_frozen_compound_track(growth_route, discovery.evidence, request)
    growth_route = _attach_excess_route_certificate(
        growth_route, discovery, request, panel, primary
    )
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": dict(certificate),
        "growth_route": growth_route,
    }


def _growth_route_research_rejection(reason: str) -> dict[str, object]:
    """Bounded read-only rejection payload with one normalized reason."""
    normalized = str(reason)[:512]
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": {
            "passed": False,
            "reasons": [normalized],
            "base_lower_cagr": None,
            "stress_lower_cagr": None,
            "matched_lower_excess_cagr": None,
        },
        "growth_route": {
            "version": GROWTH_ROUTE_VERSION,
            "candidate_count": 0,
            "selected_policy": None,
            "rejection_reason_counts": {normalized: 1},
        },
    }
