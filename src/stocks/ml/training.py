"""Thin net-alpha training orchestrator.

``train_net_alpha_model`` is the single training entry point: integrity audit,
``build_model_features`` into the canonical learner frame, a locked forward
holdout, causal per-horizon OOF evidence (purged/embargoed folds, target-free
validation prediction, decimal realized-outcome calibration, common-policy
replay), horizon selection, a conditional deterministic LightGBM challenger on
the selected primary, and an untouched forward holdout. The final decision
publishes either one champion family (learner plus fitted decimal calibration)
or a complete immutable ``NO_TRADE`` artifact. Future labels are never a
discovery score and the holdout is never refit. No Optuna, confirmation
worker, LambdaRank route, or fixed 5/10/15 horizon exists here.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind
from src.stocks.data.ml_integrity import validate_ml_snapshot
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    FoldScoreDiagnostic,
    HorizonOOFDiagnostic,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    RegularizationGrid,
)
from src.stocks.ml.features import (
    build_model_features,
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
)
from src.stocks.ml.replay import NetAlphaPolicyReplay
from src.stocks.ml.result_ledger import peak_rss_mib as _peak_rss_mib
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.calibration_schedule import SessionClusterCalibrationSchedule
from src.stocks.research.economic_alpha import CausalAlphaCalibrator
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.metrics import certify_compounded_holdout
from src.stocks.research.models import Model, ModelManifest

logger = logging.getLogger("stocks.ml.training")

_ID_COLUMN = "instrument_id"
_SESSION_IDX = "session_index"
_MIN_TRAIN_SESSIONS = 40
_VALIDATION_BLOCK_SESSIONS = 20
_REFERENCE_NOTIONAL = 100_000_000.0
_NESTED_INNER_FOLDS = 3
_NESTED_MIN_TRAIN_SESSIONS = 5
_ALPHA_TIE_TOLERANCE = 1e-12


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

    ``evidence`` are the horizons that cleared the three-nonzero-block and
    positive-lower-bound gates; ``diagnostics`` retain the typed per-horizon
    OOF diagnostics for every candidate horizon, published under
    ``oof_diagnostics`` in ``NO_TRADE`` metrics.
    """

    evidence: tuple[HorizonOOFEvidence, ...]
    diagnostics: tuple[HorizonOOFDiagnostic, ...]


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

    roles = dict(stock_net_alpha_v1_roles())
    transformed, learner_columns = build_model_features(frame, roles)
    telemetry.phase(
        "feature_transform",
        {
            "learner_feature_count": len(learner_columns),
            "panel_rows": int(transformed.height),
            "panel_sessions": (
                int(transformed["session"].n_unique())
                if not transformed.is_empty()
                else 0
            ),
        },
    )
    if not learner_columns:
        return _publish_no_trade(
            registry, request, frame, "no-alpha-learner-columns",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    panel = _index_sessions(transformed)
    pre_holdout, holdout, holdout_reason = _locked_holdout(panel, request)
    telemetry.phase(
        "holdout_lock",
        {
            "pre_holdout_rows": int(pre_holdout.height),
            "pre_holdout_sessions": (
                int(pre_holdout["session"].n_unique())
                if not pre_holdout.is_empty()
                else 0
            ),
            "holdout_rows": int(holdout.height),
            "holdout_sessions": (
                int(holdout["session"].n_unique())
                if not holdout.is_empty()
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

    discovery = _build_horizon_evidence(
        pre_holdout, frame, data, request, learner_columns
    )
    _record_horizon_discovery(telemetry, discovery)
    if not discovery.evidence:
        return _publish_no_trade(
            registry, request, frame, "no-horizon-evidence",
            details={"oof_diagnostics": [d.to_json() for d in discovery.diagnostics]},
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    selection = select_horizons(
        discovery.evidence, request.bootstrap_alpha, request.seed,
        n_bootstrap=request.bootstrap_resamples,
    )
    telemetry.phase(
        "primary_selection",
        {
            "lower_bounds": {
                int(horizon): float(bound)
                for horizon, bound in selection.lower_bounds.items()
            },
            "primary_horizon_sessions": selection.primary_horizon_sessions,
            "secondary_horizon_sessions": selection.secondary_horizon_sessions,
            "effective_horizon_count": float(selection.effective_horizon_count),
            "selection_reasons": list(selection.selection_reasons),
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
    primary_family = next(
        (
            candidate.model_family
            for candidate in discovery.evidence
            if candidate.horizon_sessions == primary
        ),
        "net_alpha_elastic_net",
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

    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=primary + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        validation_window_sessions=_VALIDATION_BLOCK_SESSIONS,
        min_train_sessions=_MIN_TRAIN_SESSIONS,
    )
    folds = splitter.split(pre_holdout)
    if not folds:
        return _publish_no_trade(
            registry, request, frame, "no-eligible-folds",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
            telemetry=telemetry,
        )

    base_manifest = _base_manifest(
        request, data, frame, primary
    )
    if primary_family == "net_alpha_lightgbm_l1":
        # The primary's discovery evidence came from the baseline structural
        # fallback, so the challenger is the producing family: refit it and
        # skip the ordinary baseline-vs-challenger comparison.
        selected_model_type = "net_alpha_lightgbm_l1"
        oof, oof_labels, fold_rank_ic, oof_diag = _challenger_oof(
            pre_holdout, folds, data, request, base_manifest, learner_columns,
            primary,
        )
        telemetry.phase(
            "model_comparison",
            {
                "baseline_available": False,
                "challenger_available": not oof.is_empty(),
                "selected_model_type": selected_model_type,
                "challenger_failure_reason": oof_diag.failure_reason or "",
            },
        )
        if oof.is_empty() or not fold_rank_ic:
            return _publish_no_trade(
                registry, request, frame, "baseline-oof-failed",
                details={"oof_diagnostics": [oof_diag.to_json()]},
                schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
                telemetry=telemetry,
            )
    else:
        baseline_oof, baseline_labels, baseline_ics, baseline_diag = _baseline_oof(
            pre_holdout, folds, data, request, base_manifest, learner_columns,
            primary,
        )
        challenger_oof, challenger_labels, challenger_ics, _ = _challenger_oof(
            pre_holdout, folds, data, request, base_manifest, learner_columns,
            primary,
        )
        selected_model_type, challenger_failure_reason = _challenger_if_better(
            baseline_oof, baseline_labels, challenger_oof, challenger_labels,
            request, primary,
        )
        if selected_model_type == "net_alpha_lightgbm_l1":
            oof, oof_labels, fold_rank_ic = (
                challenger_oof, challenger_labels, challenger_ics
            )
            oof_diag = HorizonOOFDiagnostic(
                horizon_sessions=primary,
                model_family="net_alpha_lightgbm_l1",
                failure_reason="",
            )
        else:
            oof, oof_labels, fold_rank_ic = baseline_oof, baseline_labels, baseline_ics
            oof_diag = baseline_diag
        if challenger_failure_reason:
            oof_diag = replace(
                oof_diag,
                failure_reason=challenger_failure_reason,
            )
        telemetry.phase(
            "model_comparison",
            {
                "baseline_available": not baseline_oof.is_empty(),
                "challenger_available": not challenger_oof.is_empty(),
                "selected_model_type": selected_model_type,
                "challenger_failure_reason": challenger_failure_reason or "",
            },
        )
        if oof.is_empty() or not fold_rank_ic:
            return _publish_no_trade(
                registry, request, frame, "baseline-oof-failed",
                details={"oof_diagnostics": [oof_diag.to_json()]},
                schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
                telemetry=telemetry,
            )

    calibrated = _causal_oof_calibrate(oof, oof_labels, request, primary)

    replay = NetAlphaPolicyReplay(
        horizon_sessions=primary,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        liquidity_model=request.liquidity_model,
        seed=request.seed,
    )
    evaluation = replay.evaluate(calibrated, oof_labels)

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
        final_model, calibration, holdout_panel, request, primary,
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
        bool(evaluation.blocks)
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
            manifest, eligible_from=holdout_from, eligible_to=holdout_to
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
            holdout_evidence=holdout_evidence,
            telemetry=telemetry,
        ),
    )
    logger.info(
        "published %s artifact %s (promoted=%s, horizon=%s, model=%s)",
        "champion" if passed else "NO_TRADE",
        request.artifact_id,
        passed,
        primary,
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

    Mirrors ``ElasticNetNetAlpha.fit`` preprocessing so the nested alpha_max
    and the actual fold fit see the identical standardized design. Returns
    ``None`` when any learner column is missing.
    """
    missing = [c for c in learner_columns if c not in frame.columns]
    if missing:
        return None
    from src.stocks.ml.models import _float32_matrix

    features = _float32_matrix(frame, learner_columns)
    valid = np.isfinite(features).all(axis=1)
    if not valid.any():
        return features, valid
    sub = features[valid]
    mean = sub.mean(axis=0)
    std = sub.std(axis=0)
    std[std == 0.0] = 1.0
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

    ``alpha_max = max(abs(X.T @ y_centered)) / n`` on the fold-standardized
    design; the candidate absolute alpha is ``fraction * alpha_max``. Returns
    ``(alpha, alpha_max)`` or ``None`` when the slice has no usable rows. The
    ``standardized`` design may be supplied precomputed by the caller so a
    nested alpha search reuses one design build across every penalty fraction.
    """
    del seed
    if TARGET_COLUMN not in train_slice.columns:
        return None
    if standardized is None:
        standardized = _standardized_design(train_slice, learner_columns)
    if standardized is None:
        return None
    features, valid = standardized
    targets = train_slice[TARGET_COLUMN].cast(pl.Float64).to_numpy()
    centered = targets - float(np.mean(targets))
    keep = valid & np.isfinite(centered)
    if not keep.any():
        return None
    x = features[keep]
    y = centered[keep]
    n = x.shape[0]
    alpha_max = float(np.max(np.abs(x.T @ y))) / n
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
) -> tuple[float | None, float | None, float | None]:
    """Fold-local, scale-invariant ElasticNet penalty selection.

    Uses only the outer fold's purged training rows and nested purged expanding
    folds. Every fraction is evaluated on its nested validation rank IC; a
    candidate whose finite predictions are constant in any evaluated inner fold
    is discarded. Returns ``(selected_alpha, selected_fraction, alpha_max)`` or
    ``(None, None, None)`` when every candidate is constant or fails.
    """
    nested_splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
    )
    nested = nested_splitter.inner_folds(fold_train, n_inner=_NESTED_INNER_FOLDS)
    if not nested:
        return None, None, None
    candidate_ics: dict[float, list[float]] = {fraction: [] for fraction in grid.fractions}
    constant: set[float] = set()
    alpha_max_by_fraction: dict[float, list[float]] = {
        fraction: [] for fraction in grid.fractions
    }
    for inner in nested:
        inner_train = fold_train[inner.train_mask]
        inner_val = fold_train[inner.validation_mask]
        if inner_train.is_empty() or inner_val.is_empty():
            continue
        if TARGET_COLUMN not in inner_train.columns or TARGET_COLUMN not in inner_val.columns:
            continue
        standardized = _standardized_design(inner_train, learner_columns)
        for fraction in grid.fractions:
            computed = _compute_alpha_max(
                inner_train, learner_columns, fraction, request.seed,
                standardized=standardized,
            )
            if computed is None:
                continue
            alpha, alpha_max = computed
            alpha_max_by_fraction[fraction].append(alpha_max)
            model = ElasticNetNetAlpha(
                manifest, learner_columns, TARGET_COLUMN,
                config=NetAlphaModelConfig(seed=request.seed, elastic_alpha=alpha),
            )
            try:
                model.fit(inner_train, inner_val)
                predicted = model.predict(inner_val)
            except ValueError:
                continue
            scores = predicted[SCORE_COLUMN].to_numpy().astype(float)
            if _score_is_constant(scores):
                constant.add(fraction)
                continue
            joined = predicted.join(
                inner_val.select(
                    _ID_COLUMN, SESSION_COLUMN, REALIZED_RETURN_COLUMN
                ),
                on=[_ID_COLUMN, SESSION_COLUMN],
                how="inner",
            )
            if joined.is_empty():
                continue
            candidate_ics[fraction].append(_rank_ic(joined))

    usable = [f for f in grid.fractions if f not in constant and candidate_ics[f]]
    if usable:
        best = _best_fraction(tuple(usable), candidate_ics)
        alpha_max = float(np.mean(alpha_max_by_fraction[best]))
        return best * alpha_max, best, alpha_max
    non_constant = [f for f in grid.fractions if f not in constant]
    if non_constant:
        # No usable inner fold: pick the stronger (largest) fraction by the
        # same deterministic order and derive its alpha on the full fold slice.
        best = max(non_constant)
        computed = _compute_alpha_max(fold_train, learner_columns, best, request.seed)
        if computed is None:
            return None, None, None
        alpha, alpha_max = computed
        return alpha, best, alpha_max
    return None, None, None


def _build_label_join(data: NetAlphaResearchData, horizon_sessions: int) -> pl.DataFrame:
    """Canonical per-horizon label join frame with realized and availability columns."""
    label_frame = data.labels_by_horizon[horizon_sessions]
    return label_frame.select(
        _ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN,
        AVAILABLE_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN,
        "open", "adtv_20d", "volatility_20d",
    ).with_columns(
        (pl.col(RISK_RESIDUAL_COLUMN) - pl.col(REFERENCE_COST_COLUMN))
        .alias(REALIZED_RETURN_COLUMN)
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
        },
    )
    for diagnostic in discovery.diagnostics:
        telemetry.add_horizon(_horizon_entry(diagnostic, eligible))


def _horizon_entry(
    diagnostic: HorizonOOFDiagnostic, eligible: set[int]
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
    return entry


def _admission_state(diagnostic: HorizonOOFDiagnostic, eligible: set[int]) -> str:
    if diagnostic.horizon_sessions in eligible:
        return "eligible"
    reason = diagnostic.failure_reason
    if reason:
        return reason.split(":", 1)[0] or "rejected"
    return "no-nonzero-blocks"


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


def _build_horizon_evidence(
    pre_holdout: pl.DataFrame,
    frame: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
) -> HorizonDiscovery:
    """Per-horizon OOF block evidence from purged model predictions only.

    Future labels are never a discovery score: for every horizon the
    pre-holdout history is split into purged/embargoed folds, each fold fits
    on its train rows only, validation rows are predicted target-free, the OOF
    predictions are joined to decimal realized outcomes after prediction,
    causally calibrated, and replayed through the common policy. When the
    baseline score diagnostics show a constant/invalid OOF prediction, the
    LightGBM challenger runs as a structural fallback for that horizon and
    carries its family with the evidence. A horizon contributes only when it
    has at least three realized nonzero-order OOF blocks. Independent horizon
    universes are never inner-joined.
    """
    evidence: list[HorizonOOFEvidence] = []
    diagnostics: list[HorizonOOFDiagnostic] = []
    for horizon in sorted(data.labels_by_horizon):
        label_frame = data.labels_by_horizon[horizon]
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
        manifest = _base_manifest(request, data, frame, horizon)
        splitter = PurgedWalkForward(
            n_folds=request.fold_count,
            label_horizon_sessions=horizon + 1,
            embargo_sessions=request.embargo_sessions,
            session_column=_SESSION_IDX,
            validation_window_sessions=_VALIDATION_BLOCK_SESSIONS,
            min_train_sessions=_MIN_TRAIN_SESSIONS,
        )
        folds = splitter.split(pre_holdout)
        if not folds:
            continue
        oof, oof_labels, _ics, diagnostic = _fit_oof(
            pre_holdout, folds, data, request, manifest, learner_columns,
            horizon, None,
            family="net_alpha_elastic_net",
        )
        model_family = "net_alpha_elastic_net"
        if _score_diagnostics_constant(diagnostic) or oof.is_empty():
            oof, oof_labels, _ics, diagnostic = _fit_oof(
                pre_holdout, folds, data, request, manifest, learner_columns,
                horizon, _challenger_factory(manifest, learner_columns, request),
                family="net_alpha_lightgbm_l1",
            )
            model_family = "net_alpha_lightgbm_l1"
        diagnostics.append(diagnostic)
        if oof.is_empty() or oof_labels.is_empty():
            continue
        calibrated = _causal_oof_calibrate(oof, oof_labels, request, horizon)
        replay = NetAlphaPolicyReplay(
            horizon_sessions=horizon,
            portfolio=request.portfolio,
            risk=request.risk,
            cost_schedule=request.base_cost_schedule or default_base_schedule(),
            liquidity_model=request.liquidity_model,
            seed=request.seed + horizon,
        )
        try:
            evaluation = replay.evaluate(calibrated, oof_labels)
        except ValueError as exc:
            diagnostics[-1] = replace(
                diagnostic,
                failure_reason=(
                    f"replay-error:{type(exc).__name__}:{exc}"
                ),
            )
            continue
        nonzero_blocks = [b for b in evaluation.blocks if b.order_count > 0]
        if len(nonzero_blocks) < 3:
            continue
        evidence.append(
            HorizonOOFEvidence(
                horizon_sessions=horizon,
                block_log_excess=tuple(evaluation.block_log_excess),
                model_family=model_family,
            )
        )
    return HorizonDiscovery(evidence=tuple(evidence), diagnostics=tuple(diagnostics))


def _score_diagnostics_constant(diagnostic: HorizonOOFDiagnostic) -> bool:
    """True when every fold produced a constant/failed score prediction."""
    stds = diagnostic.fold_score_stds
    if not stds:
        return True
    return all(std <= 0.0 for std in stds)


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


def _causal_oof_calibrate(
    oof: pl.DataFrame,
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
) -> pl.DataFrame:
    """Causal session-cluster calibration applied to OOF scored rows.

    For every OOF decision session ``t`` only ledger rows with ``session < t``
    and ``label_available_time <= t`` are revealed, honoring
    ``RiskSettings.min_calibration_sessions``; the frozen state is then applied
    to that session. A later label can therefore never change an earlier
    session's calibrated score.
    """
    ledger = _causal_ledger(oof_labels)
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
        augmented = augmented.drop(
            "expected_active_alpha", "alpha_lower_bound", "exit_cost_rate"
        ).with_columns(
            pl.col("expected_net_alpha").cast(pl.Float64),
            pl.col("net_alpha_lower_bound").cast(pl.Float64),
        ).rename({"score": SCORE_COLUMN})
        frames.append(augmented)
    if not frames:
        return _zero_calibrated(oof)
    return pl.concat(frames)


def _zero_calibrated(scored: pl.DataFrame) -> pl.DataFrame:
    """Cash-only calibration output: zero economic scores, no exception."""
    return scored.with_columns(
        pl.lit(0.0, dtype=pl.Float64).alias("expected_net_alpha"),
        pl.lit(0.0, dtype=pl.Float64).alias("net_alpha_lower_bound"),
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
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Fit a learner per purged fold and collect target-free OOF predictions.

    Each fold trains only on its own train rows (target joined), predicts the
    validation rows with target/availability/realized columns dropped, and the
    resulting OOF predictions are joined to decimal realized outcomes only
    after prediction. The ElasticNet baseline selects a fold-local
    scale-invariant alpha via nested purged folds. Returns
    ``(oof_scored, oof_labeled, fold_rank_ics, diagnostic)``; expected invalid
    inputs are classified in the diagnostic instead of being swallowed.
    """
    label_join = _build_label_join(data, horizon_sessions)
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    rank_ics: list[float] = []
    fold_diagnostics: list[FoldScoreDiagnostic] = []
    grid = RegularizationGrid()
    for fold_index, fold in enumerate(folds):
        train = pre_holdout[fold.train_mask].join(
            label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner",
        )
        validation = pre_holdout[fold.validation_mask]
        if train.is_empty() or validation.is_empty():
            fold_diagnostics.append(
                FoldScoreDiagnostic(fold_index=fold_index, failure_reason="empty-fold")
            )
            continue
        selected_alpha: float | None = None
        selected_fraction: float | None = None
        alpha_max: float | None = None
        if family == "net_alpha_elastic_net":
            selected_alpha, selected_fraction, alpha_max = _select_elastic_alpha(
                train, request, learner_columns, horizon_sessions, grid, base_manifest,
            )
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
        )
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
        return pl.DataFrame(), pl.DataFrame(), [], diagnostic
    return pl.concat(oof_frames), pl.concat(label_frames), rank_ics, diagnostic


def _baseline_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Deterministic fold-local ElasticNet OOF predictions on the selected primary."""
    return _fit_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary_horizon_sessions, None,
        family="net_alpha_elastic_net",
    )


def _challenger_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float], HorizonOOFDiagnostic]:
    """Deterministic LightGBM OOF predictions on the selected primary."""
    return _fit_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary_horizon_sessions,
        _challenger_factory(base_manifest, learner_columns, request),
        family="net_alpha_lightgbm_l1",
    )


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


def _challenger_if_better(
    baseline_oof: pl.DataFrame,
    baseline_labels: pl.DataFrame,
    challenger_oof: pl.DataFrame,
    challenger_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    primary_horizon_sessions: int,
) -> tuple[str, str]:
    """Conditionally adopt the LightGBM challenger on the selected primary.

    Both families are OOF-calibrated to decimal lower bounds and replayed;
    the challenger is adopted only when its paired incremental policy-utility
    lower bound over the baseline is strictly positive. Otherwise the
    ElasticNet baseline remains.
    """
    if baseline_oof.is_empty() or challenger_oof.is_empty():
        return "net_alpha_elastic_net", ""
    baseline_calibrated = _causal_oof_calibrate(
        baseline_oof, baseline_labels, request, primary_horizon_sessions
    )
    challenger_calibrated = _causal_oof_calibrate(
        challenger_oof, challenger_labels, request, primary_horizon_sessions
    )

    replay = NetAlphaPolicyReplay(
        horizon_sessions=primary_horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        liquidity_model=request.liquidity_model,
        seed=request.seed,
    )
    try:
        baseline_eval = replay.evaluate(baseline_calibrated, baseline_labels)
        challenger_eval = replay.evaluate(challenger_calibrated, challenger_labels)
    except ValueError as exc:
        return (
            "net_alpha_elastic_net",
            f"challenger-replay-error:{type(exc).__name__}:{exc}",
        )

    baseline_blocks = np.asarray(baseline_eval.block_log_excess, dtype=float)
    challenger_blocks = np.asarray(challenger_eval.block_log_excess, dtype=float)
    if baseline_blocks.size == 0 or challenger_blocks.size == 0:
        return "net_alpha_elastic_net", ""
    length = min(baseline_blocks.size, challenger_blocks.size)
    incremental = challenger_blocks[:length] - baseline_blocks[:length]
    lower_bound = float(np.quantile(incremental, request.bootstrap_alpha))
    if lower_bound > 0.0:
        return "net_alpha_lightgbm_l1", ""
    return "net_alpha_elastic_net", ""


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
        selected_alpha, selected_fraction, alpha_max = _select_elastic_alpha(
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
) -> dict[str, object]:
    """Evaluate the untouched forward holdout under base and stress costs.

    The locked holdout is scored target-free once by the pre-holdout model and
    the fitted calibration attaches the decimal lower bound. The identical
    calibrated frame is then replayed under the base and stress cost schedules
    (with their matching liquidity models) and the compound certificate gates
    promotion. No-trade diagnosis is kept separate from missing realized
    evidence, and no gate is ever relaxed after observing the holdout.
    """
    if holdout_panel.is_empty():
        return {"passed": False, "reason": "holdout-has-no-realized"}
    scored = model.predict(holdout_panel)
    calibrated = calibration.apply(scored)
    base_replay = NetAlphaPolicyReplay(
        horizon_sessions=horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        liquidity_model=request.liquidity_model,
        seed=request.seed,
    )
    stress_replay = NetAlphaPolicyReplay(
        horizon_sessions=horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.stress_cost_schedule or default_stress_schedule(),
        liquidity_model=request.stress_liquidity_model or request.liquidity_model,
        seed=request.seed,
    )
    try:
        base_evaluation = base_replay.evaluate(calibrated, holdout_panel)
        stress_evaluation = stress_replay.evaluate(calibrated, holdout_panel)
    except ValueError as exc:
        return {"passed": False, "reason": f"holdout-replay-invalid:{exc}"}
    certificate = certify_compounded_holdout(
        base_evaluation.period_net_returns,
        stress_evaluation.period_net_returns,
        horizon_sessions,
        base_evaluation.observed_sessions,
        base_evaluation.active_cohort_count,
        request.compounding,
    )
    missing_realized = (
        base_evaluation.missing_realized_cohort_count
        or stress_evaluation.missing_realized_cohort_count
    )
    if missing_realized > 0:
        reason = "holdout-incomplete-realized-cohorts"
    elif base_evaluation.eligible_sessions == 0:
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
        "block_count": len(base_evaluation.blocks),
        "order_count": len(base_evaluation.orders),
        "certificate": certificate.to_json(),
        "cohorts": {
            "scored_sessions": base_evaluation.scored_sessions,
            "realized_sessions": base_evaluation.realized_sessions,
            "eligible_sessions": base_evaluation.eligible_sessions,
            "active_sessions": base_evaluation.active_sessions,
            "orders": len(base_evaluation.orders),
            "period_count": base_evaluation.period_count,
            "observed_sessions": base_evaluation.observed_sessions,
            "active_cohort_count": base_evaluation.active_cohort_count,
            "missing_realized_cohorts": base_evaluation.missing_realized_cohort_count,
        },
        "diagnostics": {
            "base": base_evaluation.replay_diagnostics(),
            "stress": stress_evaluation.replay_diagnostics(),
        },
    }


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
    registry.write_metrics(request.artifact_id, metrics)
    logger.info("published NO_TRADE artifact %s (%s)", request.artifact_id, reason)
    return manifest


def _build_metrics(
    request: NetAlphaTrainingRequest,
    evaluation: object,
    fold_rank_ic: list[float],
    selection: HorizonSelectionEvidence,
    manifest: ModelManifest,
    *,
    holdout_evidence: dict[str, object],
    telemetry: TrainingTelemetry,
) -> dict[str, object]:
    del request
    return {
        "promoted": manifest.model_type != "no_trade",
        "no_trade": manifest.model_type == "no_trade",
        "model_type": manifest.model_type,
        "primary_horizon_sessions": selection.primary_horizon_sessions,
        "secondary_horizon_sessions": selection.secondary_horizon_sessions,
        "mean_fold_rank_ic": float(np.mean(fold_rank_ic)) if fold_rank_ic else 0.0,
        "horizon_selection": selection.to_json(),
        "holdout": {
            **holdout_evidence,
            "eligibility": {
                "eligible_from": manifest.eligible_from,
                "eligible_to": manifest.eligible_to,
            },
        },
        "replay": getattr(evaluation, "to_json", lambda: {})() if evaluation else {},
        "gates": {
            "passed": manifest.model_type != "no_trade",
            "reasons": list(selection.selection_reasons),
        },
        "run_observability": telemetry.to_dict(),
    }
