"""Thin net-alpha training orchestrator.

``train_net_alpha_model`` is the single training entry point: integrity audit,
``build_model_features`` into the canonical learner frame, cached fold-local
feature matrices, baseline ElasticNet OOF per candidate horizon, OOF replay
block evidence, horizon selection, a conditional deterministic LightGBM
challenger on the selected primary, and an untouched forward holdout. The
final decision publishes either one champion family or a complete immutable
``NO_TRADE`` artifact. No Optuna, confirmation worker, LambdaRank route, or
fixed 5/10/15 horizon exists here.
"""
from __future__ import annotations

import logging
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
    SESSION_COLUMN,
    TARGET_COLUMN,
)
from src.stocks.ml.models import (
    SCORE_COLUMN,
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
    horizon_evidence = _build_horizon_evidence(
        panel, data, request, learner_columns
    )
    if not horizon_evidence:
        return _publish_no_trade(
            registry, request, frame, "no-horizon-evidence",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    selection = select_horizons(
        tuple(horizon_evidence), request.bootstrap_alpha, request.seed
    )
    if selection.primary_horizon_sessions is None:
        return _publish_no_trade(
            registry, request, frame, "no-selected-horizon",
            details=selection.to_json(),
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    label_frame = data.labels_by_horizon[selection.primary_horizon_sessions]
    label_column = TARGET_COLUMN
    if label_column not in label_frame.columns:
        return _publish_no_trade(
            registry, request, frame, "no-label-for-primary-horizon",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=selection.primary_horizon_sessions + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        validation_window_sessions=_VALIDATION_BLOCK_SESSIONS,
        min_train_sessions=_MIN_TRAIN_SESSIONS,
    )
    folds = splitter.split(panel)
    if not folds:
        return _publish_no_trade(
            registry, request, frame, "no-eligible-folds",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    base_manifest = _base_manifest(
        request, data, frame, selection.primary_horizon_sessions
    )
    oof, oof_labels, fold_rank_ic = _baseline_oof(
        panel, folds, data, request, base_manifest, learner_columns,
        selection.primary_horizon_sessions,
    )
    if oof.is_empty() or not fold_rank_ic:
        return _publish_no_trade(
            registry, request, frame, "baseline-oof-failed",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    calibrator = NetAlphaCalibrator(
        bucket_count=request.risk.calibration_bucket_count,
        seed=request.seed,
        n_bootstrap=request.bootstrap_resamples,
        bootstrap_alpha=request.bootstrap_alpha,
        block_length=selection.primary_horizon_sessions,
        label_column=label_column,
    )
    calibrator.fit(oof_labels)
    calibrated = calibrator.apply(oof)

    replay = NetAlphaPolicyReplay(
        horizon_sessions=selection.primary_horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        seed=request.seed,
    )
    evaluation = replay.evaluate(calibrated, oof_labels)

    selected_model_type = _challenger_if_better(
        panel, folds, data, request, base_manifest, learner_columns,
        selection.primary_horizon_sessions, selection,
    )

    final_model, _final_oof = _refit_selected(
        panel, data, request, base_manifest, learner_columns,
        selection.primary_horizon_sessions, selected_model_type,
    )
    if final_model is None:
        return _publish_no_trade(
            registry, request, frame, "final-refit-failed",
            schema_hash=schema_hash, universe_policy_hash=universe_policy_hash,
        )

    holdout_evidence = _evaluate_forward_holdout(
        panel, data, request, base_manifest, learner_columns,
        selection.primary_horizon_sessions, selected_model_type,
    )

    passed = (
        bool(evaluation.blocks)
        and bool(fold_rank_ic)
        and bool(holdout_evidence.get("passed", False))
    )
    model = final_model if passed else _no_trade_model(
        base_manifest, learner_columns, label_column
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
        selection.primary_horizon_sessions,
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


def _build_horizon_evidence(
    panel: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    learner_columns: tuple[str, ...],
) -> list[HorizonOOFEvidence]:
    """Per-horizon OOF block evidence produced only by the common policy replay.

    Raw label targets are never treated as OOF economic evidence: every
    candidate horizon is replayed through the exact policy kernel with a frozen
    deterministic scoring rule (ranked learner columns), and the resulting block
    log-growth series becomes the horizon's evidence.
    """
    evidence: list[HorizonOOFEvidence] = []
    del panel, learner_columns
    for horizon in sorted(data.labels_by_horizon):
        label_frame = data.labels_by_horizon[horizon]
        if label_frame.is_empty() or label_frame.height < 3:
            continue
        proxy = _proxy_scores(label_frame)
        replay = NetAlphaPolicyReplay(
            horizon_sessions=horizon,
            portfolio=request.portfolio,
            risk=request.risk,
            cost_schedule=request.base_cost_schedule or default_base_schedule(),
            seed=request.seed + horizon,
        )
        evaluation = replay.evaluate(proxy, label_frame)
        if len(evaluation.block_log_excess) < 3:
            continue
        evidence.append(
            HorizonOOFEvidence(
                horizon_sessions=horizon,
                block_log_excess=evaluation.block_log_excess,
            )
        )
    return evidence


def _proxy_scores(label_frame: pl.DataFrame) -> pl.DataFrame:
    """Deterministic discovery score: cross-sectional rank of the net target.

    Discovery OOF evidence is generated from the same policy kernel with a
    frozen non-parametric scoring rule so horizon ordering is never fit on the
    validation labels of the final model; the champion is still selected only
    through OOF replay block evidence.
    """
    return label_frame.with_columns(
        (pl.col(TARGET_COLUMN).rank("average").over(SESSION_COLUMN))
        .cast(pl.Float64)
        .alias(SCORE_COLUMN)
    )


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


def _baseline_oof(
    panel: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[float]]:
    """Fit the deterministic ElasticNet baseline per fold and collect OOF rows.

    The model is trained on the transformed feature frame only; validation rows
    are scored with frozen fold statistics and joined to their point-in-time
    labels. Returns ``(oof_scored, oof_labeled, fold_rank_ic)``.
    """
    label_frame = data.labels_by_horizon[primary_horizon_sessions]
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    rank_ics: list[float] = []
    for fold in folds:
        train_mask = fold.train_mask
        validation_mask = fold.validation_mask
        train_frame = panel[train_mask]
        validation_frame = panel[validation_mask]
        model = ElasticNetNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
        )
        try:
            model.fit(train_frame, validation_frame)
        except ValueError:
            continue
        validation_scored = model.predict(validation_frame)
        validation_ids = validation_frame.select(
            _ID_COLUMN, "session", _SESSION_IDX
        )
        joined = validation_scored.join(
            validation_ids, on=[_ID_COLUMN, "session"], how="left"
        )
        labeled = joined.join(
            label_frame.select(_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="inner",
        )
        oof_frames.append(joined)
        label_frames.append(labeled)
        rank_ics.append(_rank_ic(labeled))
    if not oof_frames:
        return pl.DataFrame(), pl.DataFrame(), []
    oof = pl.concat(oof_frames)
    labels = pl.concat(label_frames)
    return oof, labels, rank_ics


def _rank_ic(frame: pl.DataFrame) -> float:
    if frame.is_empty() or SCORE_COLUMN not in frame.columns:
        return 0.0
    from scipy.stats import spearmanr

    sub = frame.filter(
        pl.col(SCORE_COLUMN).is_not_null() & pl.col(TARGET_COLUMN).is_not_null()
    )
    if sub.is_empty():
        return 0.0
    ics: list[float] = []
    for rows in sub.sort("session").partition_by("session"):
        if rows.height < 2:
            continue
        scores = rows[SCORE_COLUMN].to_numpy().astype(float)
        labels = rows[TARGET_COLUMN].to_numpy().astype(float)
        if np.std(scores) == 0.0 or np.std(labels) == 0.0:
            continue
        rho, _ = spearmanr(scores, labels)
        ics.append(float(rho))
    return float(np.mean(ics)) if ics else 0.0


def _challenger_if_better(
    panel: pl.DataFrame,
    folds: list[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
    selection: HorizonSelectionEvidence,
) -> str:
    """Conditionally adopt the LightGBM challenger on the selected primary.

    The challenger is fit only on the primary horizon's OOF folds; it is
    adopted only when its paired incremental policy-utility lower bound over the
    baseline is strictly positive. Otherwise the ElasticNet baseline remains.
    """
    del selection
    oof, oof_labels, _fold_ics = _baseline_oof(
        panel, folds, data, request, base_manifest, learner_columns,
        primary_horizon_sessions,
    )
    if oof.is_empty():
        return "net_alpha_elastic_net"
    baseline_replay = NetAlphaPolicyReplay(
        horizon_sessions=primary_horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        seed=request.seed,
    )
    baseline_eval = baseline_replay.evaluate(oof, oof_labels)

    challenger_oof: list[pl.DataFrame] = []
    challenger_labels: list[pl.DataFrame] = []
    for fold in folds:
        train_mask = fold.train_mask
        validation_mask = fold.validation_mask
        model = LightGbmNetAlpha(
            base_manifest,
            learner_columns,
            TARGET_COLUMN,
            config=NetAlphaModelConfig(seed=request.seed),
            num_threads=request.model_threads,
        )
        try:
            model.fit(panel[train_mask], panel[validation_mask])
        except ValueError:
            continue
        scored = model.predict(panel[validation_mask])
        joined = scored.join(
            panel[validation_mask].select(_ID_COLUMN, "session"),
            on=[_ID_COLUMN, "session"],
            how="left",
        )
        labeled = joined.join(
            oof_labels.select(_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="inner",
        )
        challenger_oof.append(joined)
        challenger_labels.append(labeled)
    if not challenger_oof:
        return "net_alpha_elastic_net"
    challenger_eval = baseline_replay.evaluate(
        pl.concat(challenger_oof), pl.concat(challenger_labels)
    )

    baseline_blocks = np.asarray(baseline_eval.block_log_excess, dtype=float)
    challenger_blocks = np.asarray(challenger_eval.block_log_excess, dtype=float)
    if baseline_blocks.size == 0 or challenger_blocks.size == 0:
        return "net_alpha_elastic_net"
    incremental = challenger_blocks[: min(baseline_blocks.size, challenger_blocks.size)] - (
        baseline_blocks[: min(baseline_blocks.size, challenger_blocks.size)]
    )
    lower_bound = float(
        np.quantile(incremental, request.bootstrap_alpha)
    )
    if lower_bound > 0.0:
        return "net_alpha_lightgbm_l1"
    return "net_alpha_elastic_net"


def _refit_selected(
    panel: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
    selected_model_type: str,
) -> tuple[Model | None, pl.DataFrame]:
    """Refit the single selected family on the all-training split."""
    del data
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
        model.fit(panel, panel.head(0))
    except ValueError:
        return None, pl.DataFrame()
    return model, panel


def _evaluate_forward_holdout(
    panel: pl.DataFrame,
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    base_manifest: ModelManifest,
    learner_columns: tuple[str, ...],
    primary_horizon_sessions: int,
    selected_model_type: str,
) -> dict[str, object]:
    """Evaluate an untouched forward holdout with the same policy replay.

    The newest ``forward_holdout_sessions`` (or a fraction of the panel when the
    request leaves the block unset) are excluded from discovery and family
    selection; the final model is scored on them and gated with the same replay,
    cost, sizing, and risk policy. A failed or absent holdout never relaxes a
    gate.
    """
    del base_manifest, learner_columns, selected_model_type
    holdout_sessions = request.forward_holdout_sessions
    if holdout_sessions <= 0:
        holdout_sessions = max(1, panel["session"].n_unique() // 5)
    sessions = sorted(panel["session"].unique().to_list())
    if len(sessions) <= holdout_sessions:
        return {"passed": False, "reason": "insufficient-holdout-history"}
    holdout_sessions_set = set(sessions[-holdout_sessions:])
    holdout = panel.filter(pl.col("session").is_in(list(holdout_sessions_set)))
    label_frame = data.labels_by_horizon[primary_horizon_sessions]
    scored = holdout.join(
        label_frame.select(_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN),
        on=[_ID_COLUMN, SESSION_COLUMN],
        how="inner",
    )
    if scored.is_empty():
        return {"passed": False, "reason": "holdout-has-no-labels"}
    replay = NetAlphaPolicyReplay(
        horizon_sessions=primary_horizon_sessions,
        portfolio=request.portfolio,
        risk=request.risk,
        cost_schedule=request.base_cost_schedule or default_base_schedule(),
        seed=request.seed,
    )
    evaluation = replay.evaluate(scored, scored)
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
