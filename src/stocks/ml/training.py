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

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    FoldScoreDiagnostic,
    HorizonOOFDiagnostic,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    PolicyProfile,
    RegularizationGrid,
    RiskSettings,
    policy_portfolio_fingerprint,
)
from src.stocks.ml.data import assess_snapshot_outcome_readiness
from src.stocks.ml.execution_replay import (
    ExecutionEquivalentReplayRequest,
    ExecutionReplayContext,
    ExecutionReplayEvidence,
    instruments_from_frame,
    replay_execution_equivalent,
)
from src.stocks.ml.features import (
    apply_model_feature_schema,
    fit_model_feature_schema,
    stock_net_alpha_v1_contract_book,
    stock_net_alpha_v1_roles,
)
from src.stocks.ml.horizons import (
    HorizonOOFEvidence,
    HorizonSelectionEvidence,
    select_horizons,
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
    fit_weighted_elastic_path,
)
from src.stocks.ml.result_ledger import (
    current_rss_mib as _current_rss_mib,
)
from src.stocks.ml.result_ledger import peak_rss_mib as _peak_rss_mib
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.calibration_schedule import SessionClusterCalibrationSchedule
from src.stocks.research.economic_alpha import CausalAlphaCalibrator
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.metrics import certify_compounded_holdout
from src.stocks.research.models import Model, ModelManifest
from src.stocks.trading.portfolio_constructor import (
    StockRiskPolicy,
    stock_risk_policy_fingerprint,
)

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


def _default_oof_cache_base() -> Path:
    from src.core.paths import PROJECT_ROOT

    return PROJECT_ROOT / "tmp" / "training"


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write a Zstandard Parquet file atomically via a same-dir rename."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temp_path, compression="zstd")
    os.replace(temp_path, path)


def _read_oof_parquet(path: Path) -> pl.DataFrame:
    """Load a cached OOF file; missing/corrupt files raise ``ValueError``."""
    if not path.exists():
        raise ValueError(f"missing cached OOF file {path}")
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"corrupt cached OOF file {path}: {exc}") from exc


class _OofCache:
    """Per-run temporary spill cache below ``<registry.root>/.training``.

    Admitted horizons write the calibrated OOF scores and the label join as
    separate Zstandard Parquet files and release the DataFrames; only the file
    paths and the small Rank-IC tuple stay in process memory. Reading a
    missing/corrupt file raises ``ValueError`` and never recomputes an OOF.
    """

    def __init__(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(dir=base_dir, prefix="oof-")
        self._root = Path(self._temporary.name)
        self._cache_bytes = 0
        self._closed = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def store(
        self,
        horizon_sessions: int,
        calibrated: pl.DataFrame,
        labels: pl.DataFrame,
    ) -> tuple[Path, Path]:
        if self._closed:
            raise ValueError("OOF cache is closed")
        oof_path = self._root / f"horizon_{horizon_sessions}_oof.parquet.zst"
        labels_path = self._root / f"horizon_{horizon_sessions}_labels.parquet.zst"
        _atomic_write_parquet(calibrated, oof_path)
        _atomic_write_parquet(labels, labels_path)
        self._cache_bytes += oof_path.stat().st_size + labels_path.stat().st_size
        return oof_path, labels_path

    def load(self, horizon_sessions: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        oof_path = self._root / f"horizon_{horizon_sessions}_oof.parquet.zst"
        labels_path = self._root / f"horizon_{horizon_sessions}_labels.parquet.zst"
        return _read_oof_parquet(oof_path), _read_oof_parquet(labels_path)

    def close(self) -> None:
        if not self._closed:
            self._temporary.cleanup()
            self._closed = True


def _enforce_memory_budget(request: NetAlphaTrainingRequest, stage: str) -> None:
    """Stop discovery at a safe horizon boundary when the peak breaches budget."""
    if request.max_rss_mib is None:
        return
    peak = _peak_rss_mib()
    if peak is not None and peak > request.max_rss_mib:
        raise _MemoryBudgetExceededError(stage)


class TrainingTelemetry:
    """Bounded scalar/dictionary observer for one training run.

    The telemetry observes only already-computed values: it never fits a second
    model, runs a second replay, or rescans the panel. The terminal projection
    is embedded under ``run_observability`` in the artifact ``metrics.json``.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._started_at = self._clock()
        self._last_at = self._started_at
        self._phases: list[dict[str, object]] = []
        self._horizons: list[dict[str, object]] = []

    def phase(self, name: str, evidence: Mapping[str, object] | None = None) -> None:
        now = self._clock()
        elapsed_ms = int((now - self._last_at).total_seconds() * 1000)
        sample: dict[str, object] = {
            "name": name,
            "elapsed_ms": elapsed_ms,
            "peak_rss_mib": _peak_rss_mib(),
            "rss_mib": _current_rss_mib(),
        }
        if evidence:
            sample.update(dict(evidence))
        self._phases.append(sample)
        self._last_at = now

    def add_horizon(self, entry: Mapping[str, object]) -> None:
        self._horizons.append(dict(entry))

    def to_dict(self) -> dict[str, object]:
        return {"phases": list(self._phases), "horizons": list(self._horizons)}


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
    dropout_reasons: Mapping[tuple[int, str], str] = field(default_factory=dict)
    execution_evidence_by_candidate: Mapping[
        tuple[int, str], ExecutionReplayEvidence
    ] = field(default_factory=dict)
    coverage_by_horizon: Mapping[int, HorizonOutcomeCoverage] = field(
        default_factory=dict
    )
    horizon_memory: Mapping[int, dict[str, object]] = field(default_factory=dict)
    oof_cache: _OofCache | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    path_evaluation_count: int = 0
    path_evaluation_bound: int = 0


def train_net_alpha_model(
    data: NetAlphaResearchData,
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
) -> ModelManifest:
    """Train the net-alpha mainline and publish a champion or complete ``NO_TRADE``.

    Args:
        data: the composed net-alpha research data (feature frame plus
            per-horizon label frames).
        registry: the immutable artifact registry.
        request: the net-alpha training request.

    Returns:
        The published ``ModelManifest`` with ``model_type`` in
        ``net_alpha_elastic_net``, ``net_alpha_lightgbm_l1``, or ``no_trade``.

    Raises:
        ValueError: when the snapshot is not a canonical net-alpha snapshot.
    """
    if data.manifest.feature_set != CANONICAL_FEATURE_SET:
        raise ValueError(
            f"train_net_alpha_model requires a net-alpha snapshot "
            f"(feature_set={CANONICAL_FEATURE_SET!r}); got "
            f"{data.manifest.feature_set!r}. Materialize a net-alpha snapshot "
            "via `python -m src.stocks.cli.build_research --pipeline net-alpha`."
        )

    telemetry = TrainingTelemetry()
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
    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=max_horizon + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=request.compounding.annualization_sessions,
    )
    folds = splitter.split(pre_holdout)
    if not folds:
        return _publish_no_trade(
            registry, request, frame, "insufficient-oof-calendar",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    cache = _OofCache(registry.root / ".training")
    try:
        return _select_publish_and_promote(
            registry=registry,
            data=data,
            request=request,
            frame=frame,
            pre_holdout=pre_holdout,
            holdout=holdout,
            folds=folds,
            learner_columns=learner_columns,
            telemetry=telemetry,
            schema_hash=schema_hash,
            universe_policy_hash=universe_policy_hash,
            oof_cache=cache,
        )
    finally:
        cache.close()


def _select_publish_and_promote(
    *,
    registry: ModelArtifactRegistry,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    frame: pl.DataFrame,
    pre_holdout: pl.DataFrame,
    holdout: pl.DataFrame,
    folds: list[Fold],
    learner_columns: tuple[str, ...],
    telemetry: TrainingTelemetry,
    schema_hash: str,
    universe_policy_hash: str,
    oof_cache: _OofCache,
) -> ModelManifest:
    """Run discovery, selection, comparison, and promotion under one OOF cache."""
    try:
        discovery = _build_horizon_evidence(
            pre_holdout, folds, data, request, learner_columns,
            oof_cache=oof_cache,
        )
    except _MemoryBudgetExceededError as exc:
        return _publish_no_trade(
            registry, request, frame, f"memory-budget-exceeded:{exc.stage}",
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

    selection = select_horizons(
        discovery.evidence, request.bootstrap_alpha, request.seed,
        n_bootstrap=request.bootstrap_resamples,
        rebalance_frequency_sessions=(
            _risk_policy_for_profile(request, 0.0).rebalance_frequency_sessions
        ),
    )
    telemetry.phase(
        "policy_frontier",
        {
            "candidate_count": len(discovery.evidence),
            "candidate_bound": 2 * len(request.candidate_horizon_sessions),
            "profile_ids": [p.profile_id for p in request.policy_profiles],
            "dropout_reasons": {
                f"{horizon}:{profile}": reason
                for (horizon, profile), reason in sorted(
                    discovery.dropout_reasons.items()
                )
            },
            "execution_evidence": _segment_summaries(
                discovery.execution_evidence_by_candidate,
                selection.primary_profile_id,
            ),
        },
    )
    telemetry.phase(
        "primary_selection",
        {
            "adjusted_lower_growth": {
                f"{horizon}:{profile}": {
                    path: float(bound) for path, bound in paths.items()
                }
                for (horizon, profile), paths in sorted(
                    selection.adjusted_lower_growth.items()
                )
            },
            "primary_horizon_sessions": selection.primary_horizon_sessions,
            "primary_profile_id": selection.primary_profile_id,
            "selection_reasons": list(selection.selection_reasons),
            "rankability_reason": selection.rankability_reason,
        },
    )
    if selection.primary_horizon_sessions is None:
        return _publish_no_trade(
            registry, request, frame, "no-selected-horizon",
            details=selection.to_json(),
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

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
        return _publish_no_trade(
            registry, request, frame, "selected-profile-not-in-frontier",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )
    label_frame = data.labels_by_horizon[primary]
    if TARGET_COLUMN not in label_frame.columns:
        return _publish_no_trade(
            registry, request, frame, "no-label-for-primary-horizon",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )
    if (
        RISK_RESIDUAL_COLUMN not in label_frame.columns
        or REFERENCE_COST_COLUMN not in label_frame.columns
    ):
        return _publish_no_trade(
            registry, request, frame, "no-realized-for-primary-horizon",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

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
        ),
        None,
    )
    if baseline_evidence is None:
        return _publish_no_trade(
            registry, request, frame, "selected-profile-evidence-missing",
            details=selection.to_json(),
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
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
        return _publish_no_trade(
            registry, request, frame, no_trade_reason,
            details={"oof_diagnostics": [oof_diag.to_json()]},
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    risk = replace(request.risk, no_trade_band_bps=profile.no_trade_band_bps)
    calibrated = _causal_oof_calibrate(oof, oof_labels, request, primary)
    base_evidence, _stress_evidence = _replay_costs(
        calibrated, oof_labels, request, primary, risk, pre_holdout, data.manifest,
    )
    evaluation = base_evidence

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
        return _publish_no_trade(
            registry, request, frame, "final-refit-failed",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    holdout_panel = holdout.join(
        label_frame.select(
            _ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN,
            REFERENCE_COST_COLUMN, "open", "adtv_20d", "volatility_20d",
        ),
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
                "policy_profile": _policy_profile_params(request, profile),
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


def _score_is_constant(values: np.ndarray) -> bool:
    """True when every finite value is equal (a degenerate prediction)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return True
    return bool(np.all(finite == finite[0]))


def _standardized_design(
    frame: pl.DataFrame, learner_columns: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fold-standardized Float32 design and its finite-valid mask.

    Mirrors the weighted preprocessing used by ``ElasticNetNetAlpha.fit`` and
    ``fit_weighted_elastic_path`` so the nested ``alpha_max`` and the actual
    fold fit see the identical standardized design. Returns ``None`` when any
    learner column is missing.
    """
    from src.stocks.ml.models import (
        _float32_matrix,
        normalize_session_weights,
        session_balanced_weights,
        weighted_fold_statistics,
    )

    missing = [c for c in learner_columns if c not in frame.columns]
    if missing:
        return None
    features = _float32_matrix(frame, learner_columns)
    valid = np.isfinite(features).all(axis=1)
    if not valid.any():
        return features, valid
    weights = normalize_session_weights(
        session_balanced_weights(frame), total=int(valid.sum())
    )
    mean, std = weighted_fold_statistics(features, weights, valid)
    return (features - mean) / std, valid


def _compute_alpha_max(
    train_slice: pl.DataFrame,
    learner_columns: tuple[str, ...],
    fraction: float,
    seed: int,
    *,
    standardized: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float] | None:
    """Scale-invariant absolute ElasticNet penalty for one training slice.

    ``alpha_max = max(abs(X.T @ (w * y_centered))) / sum(w)`` on the weighted
    fold-standardized design; the candidate absolute alpha is
    ``fraction * alpha_max``. Returns ``(alpha, alpha_max)`` or ``None`` when
    the slice has no usable rows. The ``standardized`` design may be supplied
    precomputed by the caller so a nested alpha search reuses one design build
    across every penalty fraction.
    """
    del seed
    if TARGET_COLUMN not in train_slice.columns:
        return None
    if standardized is None:
        standardized = _standardized_design(train_slice, learner_columns)
    if standardized is None:
        return None
    from src.stocks.ml.models import (
        normalize_session_weights,
        session_balanced_weights,
    )

    features, valid = standardized
    targets = train_slice[TARGET_COLUMN].cast(pl.Float64).to_numpy()
    finite = valid & np.isfinite(targets)
    if not finite.any():
        return None
    weights = normalize_session_weights(
        session_balanced_weights(train_slice), total=int(finite.sum())
    )
    sub = weights[finite]
    x = features[finite]
    y = targets[finite]
    y_centered = y - float(np.sum(sub * y) / float(np.sum(sub)))
    n = float(np.sum(sub))
    alpha_max = float(np.max(np.abs(x.T @ (sub * y_centered)))) / n
    if not np.isfinite(alpha_max) or alpha_max <= 0.0:
        return None
    return fraction * alpha_max, alpha_max


def _best_fraction(
    candidates: tuple[float, ...], ics: dict[float, list[float]]
) -> float:
    """Largest mean nested rank IC; a tie within 1e-12 picks the stronger penalty."""
    best = candidates[0]
    best_ic = float(np.mean(ics[best]))
    for fraction in candidates[1:]:
        ic = float(np.mean(ics[fraction]))
        if ic > best_ic + _ALPHA_TIE_TOLERANCE or (
            abs(ic - best_ic) <= _ALPHA_TIE_TOLERANCE and fraction > best
        ):
            best, best_ic = fraction, ic
    return best


def _select_elastic_alpha(
    fold_train: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    grid: RegularizationGrid,
    manifest: ModelManifest,
) -> tuple[float | None, float | None, float | None, int]:
    """Fold-local, scale-invariant ElasticNet penalty selection.

    Uses only the outer fold's purged training rows and nested purged expanding
    folds. Every fraction is evaluated on its nested validation rank IC through
    one deterministic weighted coordinate path per inner fold (``alpha_max`` is
    derived once from the shared standardized design); a candidate whose finite
    predictions are constant in any evaluated inner fold is discarded. Returns
    ``(selected_alpha, selected_fraction, alpha_max, path_evaluations)`` or
    ``(None, None, None, path_evaluations)`` when every candidate fails.
    """
    del manifest
    nested_splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
    )
    nested = nested_splitter.inner_folds(fold_train, n_inner=_NESTED_INNER_FOLDS)
    if not nested:
        return None, None, None, 0
    candidate_ics: dict[float, list[float]] = {fraction: [] for fraction in grid.fractions}
    constant: set[float] = set()
    alpha_maxes: list[float] = []
    path_evaluations = 0
    realized_join = fold_train.select(
        _ID_COLUMN, SESSION_COLUMN, REALIZED_RETURN_COLUMN
    )
    for inner in nested:
        inner_train = fold_train[inner.train_mask]
        inner_val = fold_train[inner.validation_mask]
        if inner_train.is_empty() or inner_val.is_empty():
            continue
        if (
            TARGET_COLUMN not in inner_train.columns
            or REALIZED_RETURN_COLUMN not in inner_val.columns
        ):
            continue
        solution = fit_weighted_elastic_path(
            inner_train, learner_columns, grid.fractions, request.seed
        )
        if solution is None:
            continue
        path_evaluations += 1
        alpha_maxes.append(solution.alpha_max)
        scores_by_fraction = solution.predict(inner_val, learner_columns)
        for fraction in grid.fractions:
            scores = scores_by_fraction[fraction]
            if _score_is_constant(scores):
                constant.add(fraction)
                continue
            scored = inner_val.with_columns(
                pl.Series(SCORE_COLUMN, scores.astype(np.float64))
            ).select(_ID_COLUMN, SESSION_COLUMN, SCORE_COLUMN)
            joined = scored.join(realized_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
            if joined.is_empty():
                continue
            candidate_ics[fraction].append(_rank_ic(joined))

    usable = [f for f in grid.fractions if f not in constant and candidate_ics[f]]
    if usable:
        best = _best_fraction(tuple(usable), candidate_ics)
        alpha_max = float(np.mean(alpha_maxes)) if alpha_maxes else 0.0
        if alpha_max <= 0.0:
            return None, None, None, min(path_evaluations, _NESTED_INNER_FOLDS)
        return (
            best * alpha_max,
            best,
            alpha_max,
            min(path_evaluations, _NESTED_INNER_FOLDS),
        )
    non_constant = [f for f in grid.fractions if f not in constant]
    if non_constant:
        # No usable inner fold: pick the stronger (largest) fraction by the
        # same deterministic order and derive its alpha on the full fold slice.
        best = max(non_constant)
        solution = fit_weighted_elastic_path(
            fold_train, learner_columns, grid.fractions, request.seed
        )
        if solution is None:
            return None, None, None, min(path_evaluations, _NESTED_INNER_FOLDS)
        return (
            best * solution.alpha_max,
            best,
            solution.alpha_max,
            min(path_evaluations, _NESTED_INNER_FOLDS) + 1,
        )
    return None, None, None, min(path_evaluations, _NESTED_INNER_FOLDS)


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
    request: NetAlphaTrainingRequest, band_bps: float
) -> StockRiskPolicy:
    """Frozen operational risk policy reconstructed from the request portfolio."""
    return StockRiskPolicy(
        top_k=request.portfolio.top_k,
        gross_cap=request.portfolio.max_exposure,
        single_name_cap=request.portfolio.max_single_weight,
        participation_limit=request.portfolio.participation_limit,
        no_trade_band_bps=band_bps,
    )


def _execution_replay_context(
    request: NetAlphaTrainingRequest,
    manifest: DatasetManifest,
    market_frame: pl.DataFrame,
    band_bps: float,
    *,
    seed: int,
) -> ExecutionReplayContext:
    """Immutable execution-equivalent context for one candidate replay."""
    instruments = instruments_from_frame(market_frame)
    sessions = sorted(market_frame["session"].unique().to_list())
    return ExecutionReplayContext(
        registry=ModelArtifactRegistry(Path("mem://execution-replay")),
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
        risk_policy=_risk_policy_for_profile(request, band_bps),
        base_cost_schedule=request.base_cost_schedule or default_base_schedule(),
        stress_cost_schedule=request.stress_cost_schedule or default_stress_schedule(),
        liquidity_model=request.liquidity_model,
        stress_liquidity_model=request.stress_liquidity_model or request.liquidity_model,
        execution_policy=request.execution_policy or SCHEDULED_OPEN_V1,
        seed=seed,
    )


def _replay_costs(
    calibrated: pl.DataFrame,
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    risk: RiskSettings,
    market_frame: pl.DataFrame,
    manifest: DatasetManifest,
) -> tuple[ExecutionReplayEvidence, ExecutionReplayEvidence]:
    """Execution-equivalent base/stress replay over the segment-identified OOF.

    The calibrated OOF score panel and the raw pre-holdout executable market
    panel are replayed through ``replay_execution_equivalent`` exactly once;
    base and stress equity are parallel because one backtester advances both
    scenarios against the same immutable prepared market. The returned pair is
    the same immutable evidence object seen from each cost path.
    """
    del oof_labels
    sessions_by_segment: dict[int, list[datetime]] = {}
    for row in calibrated.select(_OOF_SEGMENT, SESSION_COLUMN).unique().iter_rows(
        named=True
    ):
        sessions_by_segment.setdefault(int(row[_OOF_SEGMENT]), []).append(
            row[SESSION_COLUMN]
        )
    decision_sessions_by_segment = {
        segment: tuple(sorted(sessions))
        for segment, sessions in sessions_by_segment.items()
    }
    context = _execution_replay_context(
        request, manifest, market_frame, risk.no_trade_band_bps,
        seed=request.seed + horizon_sessions,
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
    return evidence, evidence


def _evidence_from_execution(
    horizon_sessions: int,
    profile_id: str,
    model_family: str,
    base_evidence: ExecutionReplayEvidence,
    stress_evidence: ExecutionReplayEvidence,
    fold_rank_ics: tuple[float, ...],
    segment_count: int,
) -> HorizonOOFEvidence:
    """Build a candidate's base/stress equity evidence from the prepared replay.

    Every evaluated session contributes one daily equity log-growth value;
    filled-cycle coverage is derived from ``cash_session_fraction``. A session
    whose execution input is missing fails closed earlier in the adapter and is
    never zero-filled, so the missing/partial cohort counts are structurally
    zero.
    """
    if base_evidence.segment_ids != stress_evidence.segment_ids:
        raise ValueError("base and stress execution segment identities diverged")
    growth_count = len(base_evidence.base_log_growth)
    if growth_count == 0:
        raise ValueError("execution replay produced no evaluated sessions")
    active = round(base_evidence.planned_cycles * (1.0 - base_evidence.cash_session_fraction))
    return HorizonOOFEvidence(
        horizon_sessions=horizon_sessions,
        profile_id=profile_id,
        model_family=model_family,
        base_log_growth=base_evidence.base_log_growth,
        stress_log_growth=stress_evidence.stress_log_growth,
        cohort_segment_ids=base_evidence.segment_ids,
        complete_cohort_count=growth_count,
        active_cohort_count=active,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=segment_count,
        fold_rank_ics=fold_rank_ics,
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


def _build_horizon_evidence(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    *,
    oof_cache: _OofCache | None = None,
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
    evidence: list[HorizonOOFEvidence] = []
    diagnostics: list[HorizonOOFDiagnostic] = []
    oof_by_horizon: dict[int, tuple[Path, Path, list[float]]] = {}
    dropout_reasons: dict[tuple[int, str], str] = {}
    execution_evidence_by_candidate: dict[
        tuple[int, str], ExecutionReplayEvidence
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
        oof, oof_labels, ics, diagnostic, fold_path_count = _fit_oof(
            pre_holdout, folds, data, request, manifest, learner_columns,
            horizon, None,
            family="net_alpha_elastic_net",
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
        if oof.is_empty() or oof_labels.is_empty():
            for profile in request.policy_profiles:
                dropout_reasons[(horizon, profile.profile_id)] = (
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
            initial_train = pre_holdout.filter(
                pl.col(_SESSION_IDX) < folds[0].validation_decision_start
            )
            seed_ledger = build_initial_calibration_seed(
                initial_train, request, learner_columns, horizon, manifest,
                data=data,
            )
            calibrated = _causal_oof_calibrate(
                oof, oof_labels, request, horizon, seed_ledger=seed_ledger
            )
            admitted_any = False

            def replay_profile(
                profile: PolicyProfile,
                *,
                _calibrated: pl.DataFrame = calibrated,
                _oof_labels: pl.DataFrame = oof_labels,
                _request: NetAlphaTrainingRequest = request,
                _horizon: int = horizon,
                _pre_holdout: pl.DataFrame = pre_holdout,
                _manifest: DatasetManifest = data.manifest,
            ) -> tuple[PolicyProfile, ExecutionReplayEvidence | None, str]:
                risk = replace(
                    _request.risk, no_trade_band_bps=profile.no_trade_band_bps
                )
                try:
                    base_evidence, _ = _replay_costs(
                        _calibrated, _oof_labels, _request, _horizon, risk,
                        _pre_holdout, _manifest,
                    )
                except ValueError as exc:
                    return profile, None, (
                        f"replay-error:{type(exc).__name__}:{exc}"
                    )
                return profile, base_evidence, ""

            with ThreadPoolExecutor(
                max_workers=min(len(request.policy_profiles), request.model_threads)
            ) as executor:
                replay_results = list(
                    executor.map(replay_profile, request.policy_profiles)
                )
            for profile, base_evidence, replay_error in replay_results:
                logger.debug(
                    "[EVAL] stage=profile_replay horizon=%d profile=%s band_bps=%.3f",
                    horizon,
                    profile.profile_id,
                    profile.no_trade_band_bps,
                )
                if base_evidence is None:
                    dropout_reasons[(horizon, profile.profile_id)] = (
                        replay_error or "replay-error"
                    )
                    continue
                if not base_evidence.base_log_growth:
                    dropout_reasons[(horizon, profile.profile_id)] = (
                        "no-evaluated-vintages"
                    )
                    continue
                stress_evidence = base_evidence
                candidate_evidence = _evidence_from_execution(
                    horizon, profile.profile_id, "net_alpha_elastic_net",
                    base_evidence, stress_evidence, tuple(ics), len(folds),
                )
                execution_evidence_by_candidate[(horizon, profile.profile_id)] = (
                    base_evidence
                )
                failure_reason = _coverage_failure_reason(candidate_evidence, request)
                dropout_reasons[(horizon, profile.profile_id)] = failure_reason
                logger.debug(
                    "[EVAL] stage=profile_result horizon=%d profile=%s sessions=%d "
                    "active=%d dropout=%s",
                    horizon,
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
            if admitted_any:
                oof_path, labels_path = oof_cache.store(
                    horizon, calibrated, oof_labels
                )
                oof_by_horizon[horizon] = (oof_path, labels_path, ics)
            del oof, oof_labels, calibrated
        horizon_memory[horizon] = {
            "rss_mib": _current_rss_mib(),
            "peak_rss_mib": _peak_rss_mib(),
            "elapsed_ms": int((time.monotonic() - horizon_started) * 1000),
            "cache_bytes": oof_cache.cache_bytes,
        }
        _enforce_memory_budget(request, "horizon_discovery")
    return HorizonDiscovery(
        evidence=tuple(evidence),
        diagnostics=tuple(diagnostics),
        oof_by_horizon=oof_by_horizon,
        dropout_reasons=dropout_reasons,
        execution_evidence_by_candidate=execution_evidence_by_candidate,
        coverage_by_horizon=coverage_by_horizon,
        horizon_memory=horizon_memory,
        oof_cache=oof_cache,
        path_evaluation_count=path_evaluation_count,
        path_evaluation_bound=(
            len(diagnostics) * len(folds) * (_NESTED_INNER_FOLDS + 1)
        ),
    )


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

    block = min(max(_segment_block_length(n), 1), n)
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
    train_data: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    base_manifest: ModelManifest,
    *,
    data: NetAlphaResearchData,
) -> pl.DataFrame:
    """Produce a causal calibration seed ledger from inner purged folds.

    The outer walk-forward plan's first validation session has no prior outer
    OOF history, so a cold-start ledger would keep the first fold in cash.
    Predicting the initial training slice through nested purged folds yields
    target-free out-of-sample scores whose realized labels are already revealed
    at the first outer decision session; feeding them as a seed ledger removes
    the cold start without any lookahead. Returns an empty frame when the slice
    cannot form usable inner folds.
    """
    if train_data.is_empty() or not learner_columns:
        return pl.DataFrame()
    splitter = PurgedWalkForward(
        n_folds=_NESTED_INNER_FOLDS,
        label_horizon_sessions=horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
    )
    inner = splitter.inner_folds(train_data, n_inner=_NESTED_INNER_FOLDS)
    if not inner:
        return pl.DataFrame()
    _scored, seed_labels, _ics, _diagnostic, _path_evaluations = _fit_oof(
        train_data, inner, data, request, base_manifest, learner_columns,
        horizon_sessions, None,
        family="net_alpha_elastic_net",
    )
    return seed_labels


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
    """
    ledger_frames = [
        frame
        for frame in (seed_ledger, oof_labels)
        if frame is not None and not frame.is_empty()
    ]
    if not ledger_frames:
        return _zero_calibrated(oof)
    ledger = _causal_ledger(pl.concat(ledger_frames))
    if ledger.is_empty() or oof.is_empty():
        return _zero_calibrated(oof)
    calibrator = _causal_calibrator(request, horizon_sessions)
    schedule = SessionClusterCalibrationSchedule(
        ledger,
        calibrator,
        request.base_cost_schedule or default_base_schedule(),
        block_length=horizon_sessions,
        max_workspace_bytes=_schedule_workspace(request),
    )
    frames: list[pl.DataFrame] = []
    by_session = {
        key[0]: frame
        for key, frame in oof.partition_by(
            SESSION_COLUMN, maintain_order=True, as_dict=True
        ).items()
    }
    for decision_time in sorted(by_session):
        state = schedule.state_at(decision_time)
        scored = by_session[decision_time].rename(
            {SCORE_COLUMN: "score"}
        )
        augmented = CausalAlphaCalibrator.apply_prepared(state, scored)
        augmented = augmented.drop("__bucket", strict=False).with_columns(
            pl.col("expected_active_alpha").cast(pl.Float64),
            pl.col("alpha_lower_bound").cast(pl.Float64),
            pl.col("expected_net_alpha").cast(pl.Float64),
            pl.col("net_alpha_lower_bound").cast(pl.Float64),
            pl.col("exit_cost_rate").cast(pl.Float64),
        ).rename({"score": SCORE_COLUMN})
        frames.append(augmented)
    if not frames:
        return _zero_calibrated(oof)
    return pl.concat(frames)


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
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic, int]:
    """Fit a learner per purged fold and collect target-free OOF predictions.

    Each fold trains only on its own train rows (target joined), predicts the
    validation rows with target/availability/realized columns dropped, and the
    resulting OOF predictions are joined to decimal realized outcomes only
    after prediction. The ElasticNet baseline selects a fold-local
    scale-invariant alpha through one vectorized weighted penalty path per
    inner fold. Returns ``(oof_scored, oof_labeled, fold_rank_ics,
    diagnostic, path_evaluations)``; expected invalid inputs are classified in
    the diagnostic instead of being swallowed.
    """
    label_join = _build_label_join(data, horizon_sessions)
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    rank_ics: list[float] = []
    fold_diagnostics: list[FoldScoreDiagnostic] = []
    grid = RegularizationGrid()
    path_evaluations = 0
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
        selected_alpha: float | None = None
        selected_fraction: float | None = None
        alpha_max: float | None = None
        if family == "net_alpha_elastic_net":
            (
                selected_alpha, selected_fraction, alpha_max, fold_path_count
            ) = _select_elastic_alpha(
                train, request, learner_columns, horizon_sessions, grid, base_manifest,
            )
            path_evaluations += fold_path_count
            if selected_alpha is None:
                fold_diagnostics.append(
                    FoldScoreDiagnostic(
                        fold_index=fold_index, failure_reason="constant-oof-score"
                    )
                )
                continue
            model: Model = ElasticNetNetAlpha(
                base_manifest, learner_columns, TARGET_COLUMN,
                config=NetAlphaModelConfig(
                    seed=request.seed,
                    elastic_alpha=selected_alpha,
                    elastic_alpha_fraction=selected_fraction,
                    elastic_alpha_max=alpha_max,
                ),
            )
            path_evaluations += 1
        else:
            if model_factory is None:
                raise ValueError("lightgbm OOF requires a model factory")
            model = model_factory()
        try:
            model.fit(train, validation)
        except ValueError as exc:
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index,
                    alpha=selected_alpha,
                    fraction=selected_fraction,
                    alpha_max=alpha_max,
                    failure_reason=f"fit-error:{type(exc).__name__}:{exc}",
                )
            )
            continue
        scored = model.predict(validation)
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
                    alpha=selected_alpha,
                    fraction=selected_fraction,
                    alpha_max=alpha_max,
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
                alpha=selected_alpha,
                fraction=selected_fraction,
                alpha_max=alpha_max,
            )
        )
    diagnostic = HorizonOOFDiagnostic(
        horizon_sessions=horizon_sessions,
        model_family=family,
        fold_diagnostics=tuple(fold_diagnostics),
    )
    if not oof_frames:
        return pl.DataFrame(), pl.DataFrame(), [], diagnostic, path_evaluations
    return (
        pl.concat(oof_frames),
        pl.concat(label_frames),
        rank_ics,
        diagnostic,
        path_evaluations,
    )


def _median_best_iteration(
    train: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    base_manifest: ModelManifest,
) -> int | None:
    """Median LightGBM best iteration over purged inner labeled validation."""
    nested_splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
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
) -> tuple[str, str, pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Conditionally adopt the LightGBM challenger on the selected primary.

    The challenger is eligible only when the linear screen is rankable. It
    replaces the baseline only when its stress-cost adjusted lower growth
    strictly improves the selected profile's baseline stress adjusted lower
    growth at the same Holm threshold (the challenger must beat the exact
    policy that was selected). Otherwise the ElasticNet baseline remains. A
    skipped challenger on a non-rankable screen is a ``NO_TRADE`` signal.
    """
    profile_key = (primary_horizon_sessions, profile.profile_id)
    if rankability_reason:
        return (
            "net_alpha_elastic_net",
            rankability_reason,
            pl.DataFrame(),
            pl.DataFrame(),
            [],
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
        _base_evidence, stress_evidence = _replay_costs(
            challenger_calibrated, challenger_labels, request,
            primary_horizon_sessions, risk, pre_holdout, data.manifest,
        )
    except ValueError as exc:
        return (
            "net_alpha_elastic_net",
            f"challenger-replay-error:{type(exc).__name__}:{exc}",
            baseline_oof, baseline_labels, baseline_ics, baseline_diag,
        )
    stress_growth = stress_evidence.stress_log_growth
    from src.stocks.ml.horizons import _cohort_bootstrap

    stress_threshold = selection.stress_holm_thresholds.get(
        profile_key, request.bootstrap_alpha
    )
    baseline_stress_lower = selection.adjusted_lower_growth.get(
        profile_key, {}
    ).get("stress", 0.0)
    bootstrap = _cohort_bootstrap(
        stress_growth,
        stress_evidence.segment_ids,
        request.bootstrap_resamples,
        request.seed + primary_horizon_sessions,
        min_block_length=max(
            primary_horizon_sessions,
            _risk_policy_for_profile(request, 0.0).rebalance_frequency_sessions,
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
        base_evidence, stress_evidence = _replay_costs(
            calibrated, calibrated, request, horizon_sessions, risk,
            holdout_panel, _holdout_stub_manifest(request, holdout_panel),
        )
    except ValueError as exc:
        return {"passed": False, "reason": f"holdout-replay-invalid:{exc}"}
    growth_count = len(base_evidence.base_log_growth)
    filled_sessions = round(
        base_evidence.planned_cycles * (1.0 - base_evidence.cash_session_fraction)
    )
    certificate = certify_compounded_holdout(
        tuple(np.expm1(value) for value in base_evidence.base_log_growth),
        tuple(np.expm1(value) for value in stress_evidence.stress_log_growth),
        horizon_sessions,
        growth_count,
        filled_sessions,
        request.compounding,
    )
    if base_evidence.filled_orders <= 0 or base_evidence.planned_cycles <= 0:
        reason = "holdout-no-economic-edge"
    elif not certificate.passed:
        reason = (
            "holdout-compound-certification-failed:"
            + ";".join(certificate.reasons)
        )
    else:
        reason = ""
    return {
        "passed": reason == "",
        "reason": reason,
        "evaluation_kind": "prepared-equity-v1",
        "block_count": growth_count,
        "order_count": base_evidence.filled_orders,
        "certificate": certificate.to_json(),
        "cohorts": {
            "scored_sessions": growth_count,
            "realized_sessions": growth_count,
            "eligible_sessions": base_evidence.planned_cycles,
            "active_sessions": filled_sessions,
            "orders": base_evidence.filled_orders,
            "period_count": growth_count,
            "observed_sessions": growth_count,
            "active_cohort_count": filled_sessions,
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
        params={"no_trade": "true"},
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
    registry.write_metrics(request.artifact_id, metrics)
    logger.info("published NO_TRADE artifact %s (%s)", request.artifact_id, reason)
    return manifest


def _policy_profile_params(
    request: NetAlphaTrainingRequest, profile: PolicyProfile
) -> str:
    """JSON projection of the selected immutable policy profile for the manifest."""
    policy = _risk_policy_for_profile(request, profile.no_trade_band_bps)
    execution_policy = request.execution_policy or SCHEDULED_OPEN_V1
    return json.dumps(
        {
            "profile_id": profile.profile_id,
            "no_trade_band_bps": profile.no_trade_band_bps,
            "top_k": request.portfolio.top_k,
            "max_single_weight": request.portfolio.max_single_weight,
            "max_exposure": request.portfolio.max_exposure,
            "participation_limit": request.portfolio.participation_limit,
            "portfolio_fingerprint": policy_portfolio_fingerprint(
                request.portfolio.top_k,
                request.portfolio.max_single_weight,
                request.portfolio.max_exposure,
                request.portfolio.participation_limit,
            ),
            "execution_evidence_version": "prepared-equity-v1",
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "execution_policy_id": execution_policy.policy_id,
            "execution_policy_hash": execution_policy.canonical_hash,
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
) -> dict[str, object]:
    annualization = request.compounding.annualization_sessions

    def annualized_cagr(lower_growth: float) -> float:
        return float(np.expm1(annualization * lower_growth))

    adjusted_lower_cagr = {
        f"{horizon}:{profile_id}": {
            path: annualized_cagr(bound)
            for path, bound in paths.items()
        }
        for (horizon, profile_id), paths in selection.adjusted_lower_growth.items()
    }
    return {
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
            f"{horizon}:{profile_id}": reason
            for (horizon, profile_id), reason in sorted(
                discovery.dropout_reasons.items()
            )
        },
        "execution_evidence": _segment_summaries(
            discovery.execution_evidence_by_candidate, selected_profile_id
        ),
    }


def _segment_summaries(
    execution_evidence_by_candidate: Mapping[
        tuple[int, str], ExecutionReplayEvidence
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
    for (horizon, profile_id), evidence in sorted(
        execution_evidence_by_candidate.items()
    ):
        if selected_profile_id is not None and profile_id != selected_profile_id:
            continue
        summaries[f"h{horizon}:{profile_id}"] = evidence.diagnostics()
    return summaries
