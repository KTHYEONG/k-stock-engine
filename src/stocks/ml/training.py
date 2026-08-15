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
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import polars as pl

from src.core.costs import default_base_schedule
from src.core.instruments import AssetKind
from src.stocks.data.ml_integrity import validate_ml_snapshot
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
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
    REALIZED_RETURN_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
    TARGET_COLUMN,
)
from src.stocks.ml.models import (
    SCORE_COLUMN,
    CalibratedNetAlphaModel,
    ElasticNetNetAlpha,
    LightGbmNetAlpha,
    NetAlphaCalibrator,
    NetAlphaModelConfig,
)
from src.stocks.ml.replay import NetAlphaPolicyReplay
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.models import Model, ModelManifest

logger = logging.getLogger("stocks.ml.training")

_ID_COLUMN = "instrument_id"
_SESSION_IDX = "session_index"
_MIN_TRAIN_SESSIONS = 40
_VALIDATION_BLOCK_SESSIONS = 20
_REFERENCE_NOTIONAL = 100_000_000.0


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
    if not audit.passed:
        return _publish_no_trade(
            registry, request, frame, "integrity-audit-failed",
            details=audit.to_json(),
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    roles = dict(stock_net_alpha_v1_roles())
    transformed, learner_columns = build_model_features(frame, roles)
    if not learner_columns:
        return _publish_no_trade(
            registry, request, frame, "no-alpha-learner-columns",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    panel = _index_sessions(transformed)
    pre_holdout, holdout, holdout_reason = _locked_holdout(panel, request)
    if holdout_reason:
        return _publish_no_trade(
            registry, request, frame, holdout_reason,
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    horizon_evidence = _build_horizon_evidence(
        pre_holdout, frame, data, request, learner_columns
    )
    if not horizon_evidence:
        return _publish_no_trade(
            registry, request, frame, "no-horizon-evidence",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    selection = select_horizons(
        tuple(horizon_evidence), request.bootstrap_alpha, request.seed,
        n_bootstrap=request.bootstrap_resamples,
    )
    if selection.primary_horizon_sessions is None:
        return _publish_no_trade(
            registry, request, frame, "no-selected-horizon",
            details=selection.to_json(),
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    primary = selection.primary_horizon_sessions
    label_frame = data.labels_by_horizon[primary]
    if TARGET_COLUMN not in label_frame.columns:
        return _publish_no_trade(
            registry, request, frame, "no-label-for-primary-horizon",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )
    if (
        RISK_RESIDUAL_COLUMN not in label_frame.columns
        or REFERENCE_COST_COLUMN not in label_frame.columns
    ):
        return _publish_no_trade(
            registry, request, frame, "no-realized-for-primary-horizon",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
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
        )

    base_manifest = _base_manifest(
        request, data, frame, primary
    )
    baseline_oof, baseline_labels, baseline_ics = _baseline_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary,
    )
    challenger_oof, challenger_labels, challenger_ics = _challenger_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary,
    )
    selected_model_type = _challenger_if_better(
        baseline_oof, baseline_labels, challenger_oof, challenger_labels,
        request, primary,
    )
    if selected_model_type == "net_alpha_lightgbm_l1":
        oof, oof_labels, fold_rank_ic = (
            challenger_oof, challenger_labels, challenger_ics
        )
    else:
        oof, oof_labels, fold_rank_ic = baseline_oof, baseline_labels, baseline_ics
    if oof.is_empty() or not fold_rank_ic:
        return _publish_no_trade(
            registry, request, frame, "baseline-oof-failed",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    calibrator = _fit_calibrator(oof_labels, request, primary, seed=request.seed)
    if calibrator is None:
        return _publish_no_trade(
            registry, request, frame, "calibration-failed",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )
    calibrated = calibrator.apply(oof)

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
    if final_model is None:
        return _publish_no_trade(
            registry, request, frame, "final-refit-failed",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    holdout_panel = holdout.join(
        label_frame.select(
            _ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN,
            REFERENCE_COST_COLUMN, "open", "adtv_20d", "volatility_20d",
        ),
        on=[_ID_COLUMN, SESSION_COLUMN],
        how="inner",
    )
    holdout_evidence = _evaluate_forward_holdout(
        final_model, calibrator, holdout_panel, request, primary,
    )

    passed = (
        bool(evaluation.blocks)
        and bool(fold_rank_ic)
        and bool(holdout_evidence.get("passed", False))
    )
    model: Model
    if passed:
        model = CalibratedNetAlphaModel(final_model, calibrator)
    else:
        model = _no_trade_model(
            base_manifest, learner_columns, TARGET_COLUMN
        )
    manifest = model.manifest()
    registry.publish(model, manifest)
    if passed:
        registry.write_forward_holdout(
            request.artifact_id,
            selection.evidence_hash,
            holdout_evidence,
        )
    registry.write_metrics(
        request.artifact_id,
        _build_metrics(
            request, evaluation, fold_rank_ic, selection, manifest,
            holdout_evidence=holdout_evidence,
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


def _baseline_factory(
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    request: NetAlphaTrainingRequest,
) -> Callable[[], ElasticNetNetAlpha]:
    def factory() -> ElasticNetNetAlpha:
        return ElasticNetNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
        )

    return factory


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


def _build_horizon_evidence(
    pre_holdout: pl.DataFrame,
    frame: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
) -> list[HorizonOOFEvidence]:
    """Per-horizon OOF block evidence from purged model predictions only.

    Future labels are never a discovery score: for every horizon the
    pre-holdout history is split into purged/embargoed folds, each fold fits
    on its train rows only, validation rows are predicted target-free, the OOF
    predictions are joined to decimal realized outcomes after prediction,
    cross-calibrated, and replayed through the common policy. A horizon
    contributes only when it has at least three realized nonzero-order OOF
    blocks. Independent horizon universes are never inner-joined.
    """
    evidence: list[HorizonOOFEvidence] = []
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
        oof, oof_labels, _ics = _fit_oof(
            pre_holdout, folds, data, request, manifest, learner_columns,
            horizon, _baseline_factory(manifest, learner_columns, request),
        )
        if oof.is_empty() or oof_labels.is_empty():
            continue
        calibrator = _fit_calibrator(oof_labels, request, horizon, seed=request.seed + horizon)
        if calibrator is None:
            continue
        calibrated = calibrator.apply(oof)
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
        except ValueError:
            continue
        nonzero_blocks = [b for b in evaluation.blocks if b.order_count > 0]
        if len(nonzero_blocks) < 3:
            continue
        evidence.append(
            HorizonOOFEvidence(
                horizon_sessions=horizon,
                block_log_excess=tuple(evaluation.block_log_excess),
            )
        )
    return evidence


def _fit_calibrator(
    oof_labels: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
    *,
    seed: int,
) -> NetAlphaCalibrator | None:
    """Fit the monotone decimal calibration on OOF predictions."""
    calibrator = NetAlphaCalibrator(
        bucket_count=request.risk.calibration_bucket_count,
        seed=seed,
        n_bootstrap=request.bootstrap_resamples,
        bootstrap_alpha=request.bootstrap_alpha,
        block_length=horizon_sessions,
        label_column=REALIZED_RETURN_COLUMN,
    )
    try:
        calibrator.fit(oof_labels)
    except ValueError:
        return None
    return calibrator


def _fit_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    horizon_sessions: int,
    model_factory: Callable[[], Model],
) -> tuple[pl.DataFrame, pl.DataFrame, list[float]]:
    """Fit a learner per purged fold and collect target-free OOF predictions.

    Each fold trains only on its own train rows (target joined), predicts the
    validation rows with target/availability/realized columns dropped, and the
    resulting OOF predictions are joined to decimal realized outcomes only
    after prediction. Returns ``(oof_scored, oof_labeled, fold_rank_ics)``.
    """
    label_frame = data.labels_by_horizon[horizon_sessions]
    label_join = label_frame.select(
        _ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN,
        RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN,
        "open", "adtv_20d", "volatility_20d",
    ).with_columns(
        (pl.col(RISK_RESIDUAL_COLUMN) - pl.col(REFERENCE_COST_COLUMN))
        .alias(REALIZED_RETURN_COLUMN)
    )
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    rank_ics: list[float] = []
    for fold in folds:
        train = pre_holdout[fold.train_mask].join(
            label_join.select(_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="inner",
        )
        validation = pre_holdout[fold.validation_mask]
        if train.is_empty() or validation.is_empty():
            continue
        model = model_factory()
        try:
            model.fit(train, validation)
        except ValueError:
            continue
        scored = model.predict(validation)
        joined = scored.join(
            validation.select(_ID_COLUMN, SESSION_COLUMN, _SESSION_IDX),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="left",
        )
        labeled = joined.join(label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
        if labeled.is_empty():
            continue
        oof_frames.append(joined)
        label_frames.append(labeled)
        rank_ics.append(_rank_ic(labeled))
    if not oof_frames:
        return pl.DataFrame(), pl.DataFrame(), []
    return pl.concat(oof_frames), pl.concat(label_frames), rank_ics


def _baseline_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float]]:
    """Deterministic ElasticNet OOF predictions on the selected primary."""
    return _fit_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary_horizon_sessions,
        _baseline_factory(base_manifest, learner_columns, request),
    )


def _challenger_oof(
    pre_holdout: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float]]:
    """Deterministic LightGBM OOF predictions on the selected primary."""
    return _fit_oof(
        pre_holdout, folds, data, request, base_manifest, learner_columns,
        primary_horizon_sessions,
        _challenger_factory(base_manifest, learner_columns, request),
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
) -> str:
    """Conditionally adopt the LightGBM challenger on the selected primary.

    Both families are OOF-calibrated to decimal lower bounds and replayed;
    the challenger is adopted only when its paired incremental policy-utility
    lower bound over the baseline is strictly positive. Otherwise the
    ElasticNet baseline remains.
    """
    if baseline_oof.is_empty() or challenger_oof.is_empty():
        return "net_alpha_elastic_net"
    baseline_cal = _fit_calibrator(
        baseline_labels, request, primary_horizon_sessions, seed=request.seed + 17
    )
    challenger_cal = _fit_calibrator(
        challenger_labels, request, primary_horizon_sessions, seed=request.seed + 23
    )
    if baseline_cal is None or challenger_cal is None:
        return "net_alpha_elastic_net"
    baseline_calibrated = baseline_cal.apply(baseline_oof)
    challenger_calibrated = challenger_cal.apply(challenger_oof)

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
    except ValueError:
        return "net_alpha_elastic_net"

    baseline_blocks = np.asarray(baseline_eval.block_log_excess, dtype=float)
    challenger_blocks = np.asarray(challenger_eval.block_log_excess, dtype=float)
    if baseline_blocks.size == 0 or challenger_blocks.size == 0:
        return "net_alpha_elastic_net"
    length = min(baseline_blocks.size, challenger_blocks.size)
    incremental = challenger_blocks[:length] - baseline_blocks[:length]
    lower_bound = float(np.quantile(incremental, request.bootstrap_alpha))
    if lower_bound > 0.0:
        return "net_alpha_lightgbm_l1"
    return "net_alpha_elastic_net"


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
    label_frame = data.labels_by_horizon[primary_horizon_sessions]
    train = pre_holdout.join(
        label_frame.select(_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN),
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
        model = ElasticNetNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
        )
    try:
        model.fit(train, train.head(0))
    except ValueError:
        return None
    return model


def _evaluate_forward_holdout(
    model: Model,
    calibration: NetAlphaCalibrator,
    holdout_panel: pl.DataFrame,
    request: NetAlphaTrainingRequest,
    horizon_sessions: int,
) -> dict[str, object]:
    """Evaluate the untouched forward holdout with the same policy replay.

    The holdout is scored target-free by the pre-holdout model, the fitted
    calibration attaches the decimal lower bound, and realized outcomes are
    replayed through the same cost/risk gate. A failed or absent holdout never
    relaxes a gate.
    """
    if holdout_panel.is_empty():
        return {"passed": False, "reason": "holdout-has-no-realized"}
    scored = model.predict(holdout_panel)
    calibrated = calibration.apply(scored)
    replay = NetAlphaPolicyReplay(
        horizon_sessions=horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        liquidity_model=request.liquidity_model,
        seed=request.seed,
    )
    try:
        evaluation = replay.evaluate(calibrated, holdout_panel)
    except ValueError as exc:
        return {"passed": False, "reason": f"holdout-replay-invalid:{exc}"}
    passed = bool(evaluation.blocks)
    return {
        "passed": passed,
        "block_count": len(evaluation.blocks),
        "order_count": len(evaluation.orders),
        "reason": "" if passed else "holdout-replay-no-trade",
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
    registry.write_metrics(
        request.artifact_id,
        {
            "promoted": False,
            "no_trade": True,
            "model_type": "no_trade",
            "promotion_reasons": [f"{reason}:{details}".rstrip(":")],
            "gates": {"passed": False},
        },
    )
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
        "holdout": holdout_evidence,
        "replay": getattr(evaluation, "to_json", lambda: {})() if evaluation else {},
        "gates": {
            "passed": manifest.model_type != "no_trade",
            "reasons": list(selection.selection_reasons),
        },
    }
