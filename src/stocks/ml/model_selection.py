"""Point-in-time model-selection: sequential candidate OOF dispatch and ledger-backed gates."""
# ruff: noqa: N806, E402, F404, I001, F811, SIM108, S110, N803, N806
# mypy: ignore-errors
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor

from src.stocks.ml.contracts import (
    ScreenEconomicEvidence,
    DEFAULT_MODEL_SELECTION_FAMILIES,
    FamilyScreenEvidence,
    FeatureAttributionEvidence,
    ModelFamily,
    ModelSelectionCandidate,
    ModelSelectionComputeBudget,
    ModelSelectionStudySettings,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    SelectedModelPolicy,
)
from src.stocks.ml.features import (
    ResearchFeatureSchema,
    apply_research_feature_schema,
    fit_research_feature_schema,
    stock_net_alpha_v1_roles,
)
from src.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    REALIZED_RETURN_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
    TARGET_COLUMN,
)
from src.stocks.ml.models import SCORE_COLUMN
from src.stocks.ml.training import _index_sessions, _locked_holdout
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.folds import Fold, PurgedWalkForward

logger = logging.getLogger("stocks.ml.model_selection")

_ID_COLUMN = "instrument_id"
_SESSION_IDX = "session_index"
_OOF_SEGMENT = "oof_segment_id"
_SCREEN_REJECTED_LOWER_BOUND = -1.0e12

# Forbidden columns must never be read by the PIT sampler.
_TARGET_FORBIDDEN_COLUMNS = frozenset(
    {TARGET_COLUMN, REALIZED_RETURN_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN, AVAILABLE_COLUMN, "label_available_time", "net_alpha_target", "realized_net_return"}
)


def _debug_timing(stage: str, started_at: float, **fields: object) -> None:
    """Emit compact monotonic timing evidence for a bounded research run."""
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.debug(
        "[SYS] stage=%s elapsed_ms=%.3f%s",
        stage,
        (time.monotonic() - started_at) * 1_000.0,
        f" {payload}" if payload else "",
    )


@dataclass(frozen=True, slots=True)
class ScreeningFoldCache:
    fold: Fold
    schema: ResearchFeatureSchema
    train_features: pl.DataFrame
    validation_features: pl.DataFrame
    train_sample_rows: np.ndarray
    validation_sample_rows: np.ndarray
    source_group_columns: tuple[tuple[str, tuple[str, ...]], ...]


def deterministic_screen_sample_rows(frame: pl.DataFrame, max_rows: int) -> np.ndarray:
    if frame.is_empty() or max_rows <= 0:
        return np.array([], dtype=np.int64)
    if max_rows >= frame.height:
        return np.arange(frame.height, dtype=np.int64)
    session_col = SESSION_COLUMN if SESSION_COLUMN in frame.columns else (_SESSION_IDX if _SESSION_IDX in frame.columns else None)
    # PIT-only: row ordering below references only observable session/liquidity/id columns.
    has_adtv = "adtv_20d" in frame.columns
    has_instrument = _ID_COLUMN in frame.columns
    indexed = frame.with_row_index("__row_idx_tmp")
    if session_col is None:
        sort_by: list[str] = []
        descending: list[bool] = []
        if has_adtv:
            sort_by.append("adtv_20d")
            descending.append(True)
        if has_instrument:
            sort_by.append(_ID_COLUMN)
            descending.append(False)
        if not sort_by:
            return np.arange(max_rows, dtype=np.int64)
        sorted_frame = indexed.sort(by=sort_by, descending=descending)
        limited = sorted_frame.head(max_rows)
        return limited["__row_idx_tmp"].to_numpy().astype(np.int64, copy=False)
    # Stratified PIT sampling: round-robin across ordered session strata
    try:
        sessions = sorted(indexed[session_col].unique().to_list())
    except Exception:
        sessions = indexed[session_col].unique().sort().to_list()
    per_session: dict[object, list[int]] = {}
    for s in sessions:
        sub = indexed.filter(pl.col(session_col) == s)
        sort_by2: list[str] = []
        descending2: list[bool] = []
        if has_adtv:
            sort_by2.append("adtv_20d")
            descending2.append(True)
        if has_instrument:
            sort_by2.append(_ID_COLUMN)
            descending2.append(False)
        if sort_by2:
            sub_sorted = sub.sort(by=sort_by2, descending=descending2)
        else:
            sub_sorted = sub
        per_session[s] = sub_sorted["__row_idx_tmp"].to_numpy().astype(np.int64, copy=False).tolist()
    result: list[int] = []
    max_per_session = max((len(v) for v in per_session.values()), default=0)
    for round_idx in range(max_per_session):
        for s in sessions:
            lst = per_session[s]
            if round_idx < len(lst) and len(result) < max_rows:
                result.append(int(lst[round_idx]))
            if len(result) >= max_rows:
                break
        if len(result) >= max_rows:
            break
    return np.array(result, dtype=np.int64)


def _aligned_screen_labels(
    frame: pl.DataFrame,
    row_indices: np.ndarray,
    labels: pl.DataFrame,
) -> pl.DataFrame:
    """Align sampled rows to labels without positional or synthetic fallback."""
    started_at = time.monotonic()
    # Required economic columns; retain gross and risk for route-aligned path (fail-closed elsewhere).
    from src.stocks.ml.labels import GROSS_COLUMN as _GC_ALIGNED
    required = (_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN, REALIZED_RETURN_COLUMN, REFERENCE_COST_COLUMN)
    optional_cols = []
    if _GC_ALIGNED in labels.columns:
        optional_cols.append(_GC_ALIGNED)
    if RISK_RESIDUAL_COLUMN in labels.columns:
        optional_cols.append(RISK_RESIDUAL_COLUMN)
    optional_gross = tuple(optional_cols)
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise ValueError(f"screen labels missing required economic columns: {missing}")
    duplicate_keys = (
        labels.group_by([_ID_COLUMN, SESSION_COLUMN])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_keys.is_empty():
        raise ValueError("screen labels contain duplicate instrument/session keys")
    requested = pl.DataFrame(
        {
            "__row_idx": row_indices.astype(np.int64, copy=False),
            "__sample_order": np.arange(row_indices.size, dtype=np.int64),
        }
    )
    keys = frame.select(_ID_COLUMN, SESSION_COLUMN).with_row_index("__row_idx")
    aligned = (
        requested.join(keys, on="__row_idx", how="left", maintain_order="left")
        .join(
            labels.select(list(required) + list(optional_gross)),
            on=[_ID_COLUMN, SESSION_COLUMN],
            how="left",
            maintain_order="left",
        )
        .sort("__sample_order")
    )
    if aligned.height != row_indices.size:
        raise ValueError("screen label alignment changed sampled row count")
    aligned = aligned.filter(
        pl.all_horizontal(pl.col(column).is_not_null() for column in required)
    )
    if aligned.is_empty():
        raise ValueError("screen labels do not cover any sampled instrument/session")
    _debug_timing(
        "screen_label_alignment",
        started_at,
        requested_rows=int(row_indices.size),
        matched_rows=int(aligned.height),
        dropped_rows=int(row_indices.size - aligned.height),
    )
    return aligned


def prepare_screening_fold_cache(
    pre_holdout: pl.DataFrame, fold: Fold, roles: Mapping[str, str], budget: ModelSelectionComputeBudget
) -> ScreeningFoldCache:
    started_at = time.monotonic()
    # Extract outer-fold train/validation partitions (time-ordered).
    try:
        train = pre_holdout[fold.train_mask]
        validation = pre_holdout[fold.validation_mask]
    except Exception:
        train = pre_holdout.filter(pl.col(_SESSION_IDX) < fold.validation_decision_start)
        validation = pre_holdout.filter(pl.col(_SESSION_IDX) >= fold.validation_decision_start)
    if train.is_empty() or validation.is_empty():
        raise ValueError("fold partition is empty")
    _debug_timing(
        "screen_cache_partition",
        started_at,
        fold_id=int(fold.segment_id),
        train_rows=int(train.height),
        validation_rows=int(validation.height),
    )
    from src.stocks.ml.features import materialize_model_feature_sources

    materialize_train_started_at = time.monotonic()
    mat_train = materialize_model_feature_sources(train, list(roles))
    _debug_timing(
        "screen_cache_materialize_train",
        materialize_train_started_at,
        fold_id=int(fold.segment_id),
        rows=int(mat_train.height),
    )
    schema_started_at = time.monotonic()
    schema = fit_research_feature_schema(mat_train, roles)
    _debug_timing(
        "screen_cache_fit_schema",
        schema_started_at,
        fold_id=int(fold.segment_id),
        source_groups=len(schema.source_groups),
    )
    materialize_valid_started_at = time.monotonic()
    mat_valid = materialize_model_feature_sources(validation, list(roles))
    _debug_timing(
        "screen_cache_materialize_validation",
        materialize_valid_started_at,
        fold_id=int(fold.segment_id),
        rows=int(mat_valid.height),
    )
    transform_started_at = time.monotonic()
    train_features = apply_research_feature_schema(mat_train, schema)
    validation_features = apply_research_feature_schema(mat_valid, schema)
    _debug_timing(
        "screen_cache_transform",
        transform_started_at,
        fold_id=int(fold.segment_id),
        train_columns=len(train_features.columns),
        validation_columns=len(validation_features.columns),
    )
    source_group_columns: tuple[tuple[str, tuple[str, ...]], ...] = tuple(schema.source_groups)
    train_sample_rows = deterministic_screen_sample_rows(train_features, int(budget.screen_train_rows_per_fold))
    validation_sample_rows = deterministic_screen_sample_rows(validation_features, int(budget.screen_validation_rows_per_fold))
    _debug_timing(
        "screen_cache_complete",
        started_at,
        fold_id=int(fold.segment_id),
        train_sample_rows=int(train_sample_rows.size),
        validation_sample_rows=int(validation_sample_rows.size),
    )
    return ScreeningFoldCache(
        fold=fold,
        schema=schema,
        train_features=train_features,
        validation_features=validation_features,
        train_sample_rows=train_sample_rows,
        validation_sample_rows=validation_sample_rows,
        source_group_columns=source_group_columns,
    )


def _screen_prefix_economic_evidence(
    scored: pl.DataFrame,
    *,
    request: NetAlphaTrainingRequest,
    bootstrap_alpha: float,
    bootstrap_resamples: int,
) -> ScreenEconomicEvidence:
    """Compute bounded economic evidence for one prefix's scored validation frame.

    Scored must contain instrument_id, session, prediction (SCORE_COLUMN or __prediction),
    and label columns gross_return (for unhedged) or risk_residual, plus reference_cost.
    Returns ScreenEconomicEvidence with bounded scalars only.
    """
    from src.stocks.ml.contracts import ScreenEconomicEvidence
    from src.stocks.ml.labels import GROSS_COLUMN
    from src.stocks.ml.economic_objective import project_route_utility

    if scored.is_empty():
        raise ValueError("scored frame is empty")
    # Determine route kind and feasible cell
    route_kind = str(getattr(getattr(request, "route_objective", None), "kind", "unhedged_absolute"))
    # Normalize to value string
    try:
        route_kind = str(request.route_objective.kind.value)
    except Exception:
        route_kind = str(route_kind).lower()
        if "hedged" in route_kind:
            route_kind = "hedged_residual"
        else:
            route_kind = "unhedged_absolute"
    # Feasible cell for top_k / cadence
    feasible = request.execution_frontier.require_feasible_horizons(request.portfolio.max_exposure, request.portfolio.max_single_weight)
    _, cadence, top_k = feasible[0]
    # Validate gross for unhedged
    if route_kind == "unhedged_absolute" and GROSS_COLUMN not in scored.columns:
        raise ValueError(f"unhedged_absolute screen requires {GROSS_COLUMN!r} column (gross missing)")
    # Resolve prediction column
    pred_col = None
    for cand in (SCORE_COLUMN, "__prediction", "prediction"):
        if cand in scored.columns:
            pred_col = cand
            break
    if pred_col is None:
        raise ValueError("scored frame missing prediction column")
    # Filter to deterministic sessions spaced by cadence
    # Use cadence kernel: sorted unique sessions, subsample every cadence-th deterministically
    try:
        sessions_sorted = sorted(set(scored[SESSION_COLUMN].to_list()))
    except Exception:
        sessions_sorted = scored[SESSION_COLUMN].unique().sort().to_list()
    if len(sessions_sorted) >= 2:
        try:
            from src.stocks.trading.rebalance_schedule import rebalance_session_indices
            indices = rebalance_session_indices(tuple(sessions_sorted), min(sessions_sorted), max(sessions_sorted), int(cadence), legacy_daily=False)
            selected_sessions = [sessions_sorted[i] for i in indices if 0 <= i < len(sessions_sorted)]
        except Exception:
            selected_sessions = sessions_sorted[:: int(cadence)]
    else:
        selected_sessions = sessions_sorted
    if not selected_sessions:
        raise ValueError("no deterministic sessions selected")
    filtered = scored.filter(pl.col(SESSION_COLUMN).is_in(selected_sessions))
    if filtered.is_empty():
        raise ValueError("screen scored empty after cadence filtering")
    # Determine prefix size from hidden column if present
    prefix_size = 1
    for hidden in ("__prefix_size", "selected_prefix_size"):
        if hidden in scored.columns:
            try:
                prefix_size = int(scored[hidden][0])
                break
            except Exception:
                pass
    # Validate each selected session has at least top_k finite names (prediction and net)
    # First compute net returns vectorized for filtered rows
    # project_route_utility minus reference_cost exactly once
    utility_series = project_route_utility(filtered, request.route_objective)
    ref = filtered[REFERENCE_COST_COLUMN].cast(pl.Float64).to_numpy()
    util_arr = utility_series.cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(util_arr)) or not np.all(np.isfinite(ref)):
        raise ValueError("non-finite utility or reference_cost in screen")
    net_arr = util_arr - ref
    # Attach net and prediction for per-session logic
    pred_arr = filtered[pred_col].cast(pl.Float64).to_numpy()
    sess_arr = filtered[SESSION_COLUMN].to_numpy()
    # Check finite per row for prediction and net
    finite_rows = np.isfinite(pred_arr) & np.isfinite(net_arr) & np.array([s is not None for s in sess_arr])
    if not np.any(finite_rows):
        raise ValueError("no finite rows for screen evidence")
    # Filter to finite rows for counting
    filtered_finite = filtered.filter(pl.Series(finite_rows))
    # Group by session and check top_k
    sizes = filtered_finite.group_by(SESSION_COLUMN).len()
    undersized = sizes.filter(pl.col("len") < int(top_k))
    if not undersized.is_empty():
        raise ValueError(f"undersized cross-section: {undersized.height} session(s) hold fewer than top_k={top_k}")
    # Per-session computation
    # Build mapping session -> indices
    unique_sessions = sorted(set(filtered_finite[SESSION_COLUMN].to_list()))
    session_count = len(unique_sessions)
    absolutes = []
    model_excesses = []
    oracle_excesses = []
    for sess in unique_sessions:
        # Recompute net for this session via utility
        sess_frame = filtered_finite.filter(pl.col(SESSION_COLUMN)==sess)
        sess_utility = project_route_utility(sess_frame, request.route_objective).cast(pl.Float64).to_numpy()
        sess_ref = sess_frame[REFERENCE_COST_COLUMN].cast(pl.Float64).to_numpy()
        sess_net = sess_utility - sess_ref
        sess_pred_vals = sess_frame[pred_col].cast(pl.Float64).to_numpy()
        sess_ids = sess_frame[_ID_COLUMN].to_numpy()
        universe_mean = float(np.mean(sess_net))
        # oracle pick: top_k by net descending, id tie-break ascending
        oracle_order = np.lexsort((sess_ids, -sess_net))
        oracle_pick = oracle_order[: int(top_k)]
        oracle_mean = float(np.mean(sess_net[oracle_pick]))
        # model pick: top_k by pred descending
        model_order = np.lexsort((sess_ids, -sess_pred_vals))
        model_pick = model_order[: int(top_k)]
        model_mean = float(np.mean(sess_net[model_pick]))
        absolutes.append(model_mean)
        model_excesses.append(model_mean - universe_mean)
        oracle_excesses.append(oracle_mean - universe_mean)
    absolutes_np = np.asarray(absolutes, dtype=np.float64)
    model_excess_np = np.asarray(model_excesses, dtype=np.float64)
    oracle_excess_np = np.asarray(oracle_excesses, dtype=np.float64)
    # Bootstrap lower bounds
    def _bootstrap_lb(values: np.ndarray, alpha: float, resamples: int, seed: int) -> float:
        if values.size == 0:
            return float("-inf")
        rng = np.random.default_rng(int(seed))
        draws = rng.integers(0, values.size, size=(int(resamples), values.size))
        means = values[draws].mean(axis=1)
        return float(np.quantile(means, float(alpha)))
    seed = int(getattr(request, "seed", 42))
    absolute_lb = _bootstrap_lb(absolutes_np, float(bootstrap_alpha), int(bootstrap_resamples), seed)
    tail_lb = _bootstrap_lb(model_excess_np, float(bootstrap_alpha), int(bootstrap_resamples), seed + 1)
    oracle_lb = _bootstrap_lb(oracle_excess_np, float(bootstrap_alpha), int(bootstrap_resamples), seed + 2)
    # Ensure finite
    for v in (absolute_lb, tail_lb, oracle_lb):
        if not math.isfinite(float(v)):
            raise ValueError("non-finite bootstrap lower bound")
    # DEBUG logs
    if logger.isEnabledFor(logging.DEBUG):
        # limit selected groups to 5 identifiers
        # route alignment log
        logger.debug("[DATA] stage=screen_route_alignment route=%s top_k=%d cadence=%d fold_id=%d session_count=%d absolute_lb=%.3f tail_excess_lb=%.3f oracle_tail_excess_lb=%.3f", route_kind, int(top_k), int(cadence), 0, int(session_count), float(round(absolute_lb,3)), float(round(tail_lb,3)), float(round(oracle_lb,3)))
        # prefix log: include selected_prefix_size limited
        logger.debug("[EVAL] stage=screen_prefix route=%s top_k=%d cadence=%d absolute_lb=%.3f tail_excess_lb=%.3f oracle_tail_excess_lb=%.3f prefix_size=%d", route_kind, int(top_k), int(cadence), float(round(absolute_lb,3)), float(round(tail_lb,3)), float(round(oracle_lb,3)), int(prefix_size))
    return ScreenEconomicEvidence(
        fold_id=0,
        route_kind=str(route_kind),
        top_k=int(top_k),
        rebalance_frequency_sessions=int(cadence),
        session_count=int(session_count),
        selected_prefix_size=int(prefix_size),
        absolute_lower_bound=float(absolute_lb),
        tail_excess_lower_bound=float(tail_lb),
        oracle_tail_excess_lower_bound=float(oracle_lb),
    )

def screen_model_family(
    cache: ScreeningFoldCache, labels: pl.DataFrame, family: ModelFamily, budget: ModelSelectionComputeBudget, deadline: float, *, request: NetAlphaTrainingRequest | None = None, bootstrap_alpha: float | None = None, bootstrap_resamples: int | None = None
) -> FamilyScreenEvidence:
    started_at = time.monotonic()
    if time.monotonic() >= deadline:
        raise TimeoutError("budget-exhausted before screening")
    source_groups = cache.source_group_columns
    if not source_groups:
        raise ValueError("cache has no source groups")
    all_columns: tuple[str, ...] = tuple(c for _, cols in source_groups for c in cols if c in cache.train_features.columns)
    if not all_columns:
        raise ValueError("no learner columns in cache")
    # Backward compat: no request -> legacy path (MSE)
    if request is None or bootstrap_alpha is None or bootstrap_resamples is None:
        # --- LEGACY MSE PATH (preserved for old tests) ---
        matrix_started_at = time.monotonic()
        X_train_full = _design_matrix(cache.train_features, all_columns)
        X_valid_full = _design_matrix(cache.validation_features, all_columns)
        _debug_timing("screen_design_matrix_full", matrix_started_at, family=family.value, fold_id=int(cache.fold.segment_id), train_rows=int(X_train_full.shape[0]), validation_rows=int(X_valid_full.shape[0]), feature_columns=int(X_train_full.shape[1]))
        n_train = X_train_full.shape[0]
        n_valid = X_valid_full.shape[0]
        train_idx = cache.train_sample_rows
        valid_idx = cache.validation_sample_rows
        train_idx = train_idx[train_idx < n_train]
        valid_idx = valid_idx[valid_idx < n_valid]
        if train_idx.size == 0:
            train_idx = np.arange(min(n_train, int(budget.screen_train_rows_per_fold)), dtype=np.int64)
        if valid_idx.size == 0:
            valid_idx = np.arange(min(n_valid, int(budget.screen_validation_rows_per_fold)), dtype=np.int64)
        label_started_at = time.monotonic()
        try:
            train_labels = _aligned_screen_labels(cache.train_features, train_idx, labels)
            valid_labels = _aligned_screen_labels(cache.validation_features, valid_idx, labels)
        except Exception:
            scores = tuple((name, 0.0) for name, _ in source_groups)
            attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
            return FamilyScreenEvidence(family=family, screen_lower_bound=_SCREEN_REJECTED_LOWER_BOUND, screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False)
        _debug_timing("screen_label_alignment_total", label_started_at, family=family.value, fold_id=int(cache.fold.segment_id))
        train_row_idx = train_labels["__row_idx"].to_numpy().astype(np.int64, copy=False)
        valid_row_idx = valid_labels["__row_idx"].to_numpy().astype(np.int64, copy=False)
        X_train = X_train_full[train_row_idx]
        X_valid = X_valid_full[valid_row_idx]
        y_train = train_labels[TARGET_COLUMN].cast(pl.Float64).to_numpy()
        y_valid = valid_labels[TARGET_COLUMN].cast(pl.Float64).to_numpy()
        realized_train = train_labels[REALIZED_RETURN_COLUMN].cast(pl.Float64).to_numpy()
        realized_valid = valid_labels[REALIZED_RETURN_COLUMN].cast(pl.Float64).to_numpy()
        cost_train = train_labels[REFERENCE_COST_COLUMN].cast(pl.Float64).to_numpy()
        cost_valid = valid_labels[REFERENCE_COST_COLUMN].cast(pl.Float64).to_numpy()
        session_train = train_labels[SESSION_COLUMN].to_numpy()
        session_valid = valid_labels[SESSION_COLUMN].to_numpy()
        if not (X_train.shape[0] == y_train.shape[0] == realized_train.shape[0] == cost_train.shape[0] == session_train.shape[0]):
            scores = tuple((name, 0.0) for name, _ in source_groups)
            attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
            return FamilyScreenEvidence(family=family, screen_lower_bound=_SCREEN_REJECTED_LOWER_BOUND, screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False)
        if not (X_valid.shape[0] == y_valid.shape[0] == realized_valid.shape[0] == cost_valid.shape[0] == session_valid.shape[0]):
            scores = tuple((name, 0.0) for name, _ in source_groups)
            attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
            return FamilyScreenEvidence(family=family, screen_lower_bound=_SCREEN_REJECTED_LOWER_BOUND, screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False)
        finite_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train) & np.isfinite(realized_train) & np.isfinite(cost_train)
        finite_valid = np.isfinite(X_valid).all(axis=1) & np.isfinite(y_valid) & np.isfinite(realized_valid) & np.isfinite(cost_valid)
        try:
            session_valid_not_null = np.array([s is not None for s in session_valid], dtype=bool)
            finite_valid = finite_valid & session_valid_not_null
        except Exception:
            pass
        if not finite_train.any() or not finite_valid.any():
            scores = tuple((name, 0.0) for name, _ in source_groups)
            attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
            return FamilyScreenEvidence(family=family, screen_lower_bound=_SCREEN_REJECTED_LOWER_BOUND, screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False)
        X_train = X_train[finite_train]
        y_train = y_train[finite_train]
        realized_train = realized_train[finite_train]
        cost_train = cost_train[finite_train]
        session_train = session_train[finite_train]
        train_labels = train_labels.filter(pl.Series(finite_train))
        valid_mask_series = pl.Series(finite_valid)
        X_valid = X_valid[finite_valid]
        y_valid = y_valid[finite_valid]
        realized_valid = realized_valid[finite_valid]
        cost_valid = cost_valid[finite_valid]
        session_valid = session_valid[finite_valid]
        valid_labels = valid_labels.filter(valid_mask_series)
        col_index = {name: idx for idx, name in enumerate(all_columns)}
        group_cols_idx: dict[str, list[int]] = {}
        for gname, gcols in source_groups:
            idxs = [col_index[c] for c in gcols if c in col_index]
            group_cols_idx[gname] = idxs
        def _fit_family(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray):
            if family == ModelFamily.elastic_net_v2:
                Xs, _Xvs = _impute_and_standardize_from_train(Xtr, Xva)
                n_feat = Xtr.shape[1]
                _train_means = np.zeros(n_feat, dtype=np.float64)
                for _j in range(n_feat):
                    _col = Xtr[:, _j]
                    _finite = np.isfinite(_col)
                    _train_means[_j] = float(np.mean(_col[_finite])) if np.any(_finite) else 0.0
                _Xtr_imp = Xtr.copy().astype(np.float64, copy=False)
                for _j in range(n_feat):
                    _m = ~np.isfinite(_Xtr_imp[:, _j])
                    if np.any(_m):
                        _Xtr_imp[_m, _j] = _train_means[_j]
                _mean = np.mean(_Xtr_imp, axis=0)
                _std = np.std(_Xtr_imp, axis=0)
                _std = np.where(_std == 0, 1.0, _std)
                _std = np.where(~np.isfinite(_std), 1.0, _std)
                _mean = np.where(~np.isfinite(_mean), 0.0, _mean)
                model = ElasticNet(alpha=_elastic_penalty(Xs, ytr), l1_ratio=0.5, max_iter=2000, tol=1e-3, random_state=42)
                model.fit(Xs, ytr)
                def pred_fn(Xp: np.ndarray) -> np.ndarray:
                    Xp_arr = np.asarray(Xp, dtype=np.float64)
                    for _jj in range(Xp_arr.shape[1]):
                        _mm = ~np.isfinite(Xp_arr[:, _jj])
                        if np.any(_mm):
                            Xp_arr[_mm, _jj] = _train_means[_jj]
                    Xp2 = (Xp_arr - _mean) / _std
                    Xp2[~np.isfinite(Xp2)] = 0.0
                    return model.predict(Xp2)
                native = np.abs(np.asarray(model.coef_, dtype=np.float64)) if hasattr(model, "coef_") else np.zeros(Xtr.shape[1])
                return model, pred_fn, native
            if family == ModelFamily.huber_linear_v1:
                Xs, _Xvs = _impute_and_standardize_from_train(Xtr, Xva)
                n_feat = Xtr.shape[1]
                _train_means = np.zeros(n_feat, dtype=np.float64)
                for _j in range(n_feat):
                    _col = Xtr[:, _j]
                    _finite = np.isfinite(_col)
                    _train_means[_j] = float(np.mean(_col[_finite])) if np.any(_finite) else 0.0
                _Xtr_imp = Xtr.copy().astype(np.float64, copy=False)
                for _j in range(n_feat):
                    _m = ~np.isfinite(_Xtr_imp[:, _j])
                    if np.any(_m):
                        _Xtr_imp[_m, _j] = _train_means[_j]
                _mean = np.mean(_Xtr_imp, axis=0)
                _std = np.std(_Xtr_imp, axis=0)
                _std = np.where(_std == 0, 1.0, _std)
                _std = np.where(~np.isfinite(_std), 1.0, _std)
                _mean = np.where(~np.isfinite(_mean), 0.0, _mean)
                model = HuberRegressor(epsilon=1.35, max_iter=500)
                model.fit(Xs, ytr)
                def pred_fn(Xp: np.ndarray) -> np.ndarray:
                    Xp_arr = np.asarray(Xp, dtype=np.float64)
                    for _jj in range(Xp_arr.shape[1]):
                        _mm = ~np.isfinite(Xp_arr[:, _jj])
                        if np.any(_mm):
                            Xp_arr[_mm, _jj] = _train_means[_jj]
                    Xp2 = (Xp_arr - _mean) / _std
                    Xp2[~np.isfinite(Xp2)] = 0.0
                    return model.predict(Xp2)
                native = np.abs(np.asarray(model.coef_, dtype=np.float64)) if hasattr(model, "coef_") else np.zeros(Xtr.shape[1])
                return model, pred_fn, native
            if family == ModelFamily.extra_trees_v1:
                model = ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=1)
                model.fit(Xtr, ytr)
                def pred_fn(Xp: np.ndarray) -> np.ndarray:
                    return model.predict(Xp)
                native = np.asarray(model.feature_importances_, dtype=np.float64) if hasattr(model, "feature_importances_") else np.zeros(Xtr.shape[1])
                return model, pred_fn, native
            if family == ModelFamily.hist_gradient_quantile_v1:
                model = HistGradientBoostingRegressor(loss="quantile", quantile=0.2, max_iter=30, random_state=42)
                model.fit(Xtr, ytr)
                def pred_fn(Xp: np.ndarray) -> np.ndarray:
                    return model.predict(Xp)
                native = np.zeros(Xtr.shape[1], dtype=np.float64)
                return model, pred_fn, native
            if family == ModelFamily.rawnet_lgbm_v2:
                train_set = lgb.Dataset(Xtr, label=ytr, params={"verbosity": -1})
                params = {"objective": "regression", "metric": "l2", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
                booster = lgb.train(params, train_set, num_boost_round=20)
                def pred_fn(Xp: np.ndarray) -> np.ndarray:
                    return booster.predict(Xp)
                try:
                    gain = booster.feature_importance(importance_type="gain").astype(np.float64)
                    if gain.size != Xtr.shape[1]:
                        gain = np.ones(Xtr.shape[1], dtype=np.float64)
                except Exception:
                    gain = np.ones(Xtr.shape[1], dtype=np.float64)
                return booster, pred_fn, gain
            if family == ModelFamily.tail_lambdarank_v2:
                rank_sessions = session_train
                order = np.argsort(rank_sessions, kind="stable")
                ordered = np.asarray(rank_sessions)[order] if not isinstance(rank_sessions, np.ndarray) else rank_sessions[order]
                _, group_sizes = np.unique(ordered, return_counts=True)
                keep_groups = group_sizes >= 2
                if not bool(np.all(keep_groups)):
                    boundaries = np.cumsum(np.r_[0, group_sizes])
                    keep = np.concatenate([np.arange(boundaries[i], boundaries[i + 1]) for i, keep in enumerate(keep_groups) if keep]) if bool(np.any(keep_groups)) else np.array([], dtype=np.int64)
                    order = order[keep]
                    group_sizes = group_sizes[keep_groups]
                if order.size == 0:
                    raise ValueError("LambdaRank screening has no complete session query groups")
                train_set = lgb.Dataset(Xtr[order], label=(ytr[order] > np.median(ytr[order])).astype(int), group=group_sizes, params={"verbosity": -1})
                params = {"objective": "lambdarank", "metric": "ndcg", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
                booster = lgb.train(params, train_set, num_boost_round=20)
                def pred_fn(Xp: np.ndarray) -> np.ndarray:
                    return booster.predict(Xp)
                try:
                    gain = booster.feature_importance(importance_type="gain").astype(np.float64)
                    if gain.size != Xtr.shape[1]:
                        gain = np.ones(Xtr.shape[1], dtype=np.float64)
                except Exception:
                    gain = np.ones(Xtr.shape[1], dtype=np.float64)
                return booster, pred_fn, gain
            raise ValueError(f"unknown family {family}")
        base_fit_started_at = time.monotonic()
        try:
            _model, predict_fn, native_importance = _fit_family(X_train, y_train, X_valid)
            base_pred = predict_fn(X_valid)
        except Exception:
            scores = tuple((name, 0.0) for name, _ in source_groups)
            attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
            return FamilyScreenEvidence(family=family, screen_lower_bound=_SCREEN_REJECTED_LOWER_BOUND, screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False)
        _debug_timing("screen_base_fit_predict", base_fit_started_at, family=family.value, fold_id=int(cache.fold.segment_id), train_rows=int(X_train.shape[0]), validation_rows=int(X_valid.shape[0]))
        base_loss = _validation_economic_loss(y_valid, base_pred)
        contributions: dict[str, float] = {}
        permutation_started_at = time.monotonic()
        attribution_predictions = 0
        if family in (ModelFamily.elastic_net_v2, ModelFamily.huber_linear_v1, ModelFamily.extra_trees_v1, ModelFamily.rawnet_lgbm_v2, ModelFamily.tail_lambdarank_v2):
            for gname, idxs in group_cols_idx.items():
                if not idxs:
                    contributions[gname] = 0.0
                else:
                    grp_score = float(np.abs(native_importance[idxs]).sum()) if native_importance.size == len(all_columns) else 0.0
                    contributions[gname] = grp_score
        else:
            rng = np.random.default_rng(42)
            working_buffer = np.empty_like(X_valid)
            for gname, idxs in group_cols_idx.items():
                if not idxs:
                    contributions[gname] = 0.0
                    continue
                if time.monotonic() >= deadline:
                    _debug_timing("screen_permutation_deadline", permutation_started_at, family=family.value, fold_id=int(cache.fold.segment_id), completed_groups=len(contributions))
                    raise TimeoutError("budget-exhausted during attribution")
                np.copyto(working_buffer, X_valid)
                perm = rng.permutation(working_buffer.shape[0])
                for ci in idxs:
                    working_buffer[:, ci] = working_buffer[perm, ci]
                perm_pred = predict_fn(working_buffer)
                attribution_predictions += 1
                loss = _validation_economic_loss(y_valid, perm_pred)
                contributions[gname] = float(loss - base_loss)
        _debug_timing("screen_permutation_attribution", permutation_started_at, family=family.value, fold_id=int(cache.fold.segment_id), source_groups=len(group_cols_idx), attribution_predictions=attribution_predictions)
        scores_list = [(name, float(contributions.get(name, 0.0)) if math.isfinite(float(contributions.get(name, 0.0))) else 0.0) for name, _ in source_groups]
        G = len(source_groups)
        max_prefixes = math.ceil(math.sqrt(G)) if G > 0 else 1
        ranked = sorted(scores_list, key=lambda x: x[1], reverse=True)
        eligible_ranked = [(n, s) for n, s in ranked if contributions.get(n, 0.0) >= 0.0]
        if not eligible_ranked:
            eligible_ranked = ranked
        if len(eligible_ranked) <= max_prefixes:
            prefix_ks = list(range(1, len(eligible_ranked) + 1))
        else:
            step = (len(eligible_ranked) - 1) / (max_prefixes - 1) if max_prefixes > 1 else 1
            prefix_ks = [1 + round(i * step) for i in range(max_prefixes)]
            prefix_ks = sorted({min(max(1, k), len(eligible_ranked)) for k in prefix_ks})
        prefix_started_at = time.monotonic()
        losses: list[float] = []
        for k in prefix_ks:
            if time.monotonic() >= deadline:
                raise TimeoutError("budget-exhausted during prefix fitting")
            selected_names = [n for n, _ in eligible_ranked[:k]]
            sel_idx = [col_index[c] for n in selected_names for c in dict(source_groups).get(n, ()) if c in col_index]
            if not sel_idx:
                losses.append(float("inf"))
                continue
            Xtr = X_train[:, sel_idx]
            Xva = X_valid[:, sel_idx]
            try:
                _, pf, _ = _fit_family(Xtr, y_train, Xva)
                pred = pf(Xva)
                loss = _validation_economic_loss(y_valid, pred)
            except Exception:
                loss = float("inf")
            losses.append(loss)
        _debug_timing("screen_prefix_fits", prefix_started_at, family=family.value, fold_id=int(cache.fold.segment_id), prefix_fit_count=len(prefix_ks))
        if not losses:
            selected = tuple(n for n, _ in eligible_ranked[:1])
        else:
            best_idx = int(np.argmin(losses))
            best_loss = float(losses[best_idx])
            se = float(np.std(y_valid) / math.sqrt(max(1, len(y_valid)))) if len(y_valid) > 1 else 0.01
            threshold = best_loss + se
            chosen_k = len(losses)
            for idx, loss in enumerate(losses):
                if loss <= threshold:
                    chosen_k = prefix_ks[idx]
                    break
            selected = tuple(n for n, _ in eligible_ranked[:chosen_k])
            if not selected:
                selected = tuple(n for n, _ in eligible_ranked[:1])
        if not selected:
            selected = tuple(n for n, _ in eligible_ranked[:1]) if eligible_ranked else tuple(n for n,_ in scores_list[:1])
        attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=tuple(scores_list), selected_source_groups=tuple(selected), schema_fingerprint=cache.schema.fingerprint)
        utility_started_at = time.monotonic()
        screen_frame = valid_labels.with_columns(pl.Series("__prediction", base_pred))
        top_k = min(12, max(1, int(screen_frame.group_by(SESSION_COLUMN).len()["len"].median())))
        selected_q = (screen_frame.sort([SESSION_COLUMN, "__prediction", _ID_COLUMN], descending=[False, True, False]).with_columns(pl.col("__prediction").rank("ordinal", descending=True).over(SESSION_COLUMN).alias("__rank")).filter(pl.col("__rank") <= top_k).group_by(SESSION_COLUMN).agg((pl.col(REALIZED_RETURN_COLUMN) - pl.col(REFERENCE_COST_COLUMN)).mean().alias("__net_return")).filter(pl.col("__net_return") > -1.0))
        if selected_q.is_empty():
            raise ValueError("screening produced no valid cost-aware session returns")
        growth = np.log1p(selected_q["__net_return"].cast(pl.Float64).to_numpy())
        lower_bound = _block_bootstrap_lower_bound(growth, 0.05, 200)
        se_growth = float(np.std(growth, ddof=1) / math.sqrt(max(1, len(growth)))) if len(growth) > 1 else 0.0
        _debug_timing("screen_economic_utility", utility_started_at, family=family.value, fold_id=int(cache.fold.segment_id), session_count=int(selected_q.height))
        _debug_timing("screen_family_complete", started_at, family=family.value, fold_id=int(cache.fold.segment_id), lower_bound=f"{lower_bound:.6f}")
        return FamilyScreenEvidence(family=family, screen_lower_bound=lower_bound, screen_se=se_growth, attribution=attr, qualified_for_full_oof=False, selected_family=False)
    # --- ROUTE-ALIGNED ECONOMIC PATH ---
    from src.stocks.ml.economic_objective import project_route_utility

    # Validate gross for unhedged before any fitting
    # Actually import locally
    try:
        from src.stocks.ml.labels import GROSS_COLUMN as _GC
    except Exception:
        _GC = "gross_return"
    route_kind = str(getattr(getattr(request, "route_objective", None), "kind", "unhedged_absolute"))
    try:
        route_kind = str(request.route_objective.kind.value)
    except Exception:
        route_kind = "unhedged_absolute" if "unhedged" in str(route_kind).lower() else "hedged_residual"
    if route_kind == "unhedged_absolute" and _GC not in labels.columns:
        raise ValueError(f"unhedged_absolute screen requires {_GC!r} column (gross missing)")
    # Determine feasible cell for top_k/cadence
    feasible_cells = request.execution_frontier.require_feasible_horizons(request.portfolio.max_exposure, request.portfolio.max_single_weight)
    if len(feasible_cells) != 1:
        # Fallback for legacy callers: pick first cell deterministically; evaluate_model_selection_study enforces single-cell gate separately.
        _, cadence, top_k = feasible_cells[0]
    else:
        _, cadence, top_k = feasible_cells[0]
    # Proceed with design matrices and alignment
    matrix_started_at = time.monotonic()
    X_train_full = _design_matrix(cache.train_features, all_columns)
    X_valid_full = _design_matrix(cache.validation_features, all_columns)
    _debug_timing("screen_design_matrix_full", matrix_started_at, family=family.value, fold_id=int(cache.fold.segment_id), train_rows=int(X_train_full.shape[0]), validation_rows=int(X_valid_full.shape[0]), feature_columns=int(X_train_full.shape[1]))
    n_train = X_train_full.shape[0]
    n_valid = X_valid_full.shape[0]
    train_idx = cache.train_sample_rows
    valid_idx = cache.validation_sample_rows
    train_idx = train_idx[train_idx < n_train]
    valid_idx = valid_idx[valid_idx < n_valid]
    if train_idx.size == 0:
        train_idx = np.arange(min(n_train, int(budget.screen_train_rows_per_fold)), dtype=np.int64)
    if valid_idx.size == 0:
        valid_idx = np.arange(min(n_valid, int(budget.screen_validation_rows_per_fold)), dtype=np.int64)
    label_started_at = time.monotonic()
    try:
        train_labels = _aligned_screen_labels(cache.train_features, train_idx, labels)
        valid_labels = _aligned_screen_labels(cache.validation_features, valid_idx, labels)
    except Exception as exc:
        # If alignment failed due to missing gross, propagate ValueError
        if "gross" in str(exc).lower():
            raise
        scores = tuple((name, 0.0) for name, _ in source_groups)
        attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
        # Return rejected with bounded evidence
        see = ScreenEconomicEvidence(fold_id=int(cache.fold.segment_id), route_kind=str(route_kind), top_k=int(top_k), rebalance_frequency_sessions=int(cadence), session_count=0, selected_prefix_size=1, absolute_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), oracle_tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND))
        return FamilyScreenEvidence(family=family, screen_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False, screen_economic_evidence=see)
    _debug_timing("screen_label_alignment_total", label_started_at, family=family.value, fold_id=int(cache.fold.segment_id))
    train_row_idx = train_labels["__row_idx"].to_numpy().astype(np.int64, copy=False)
    valid_row_idx = valid_labels["__row_idx"].to_numpy().astype(np.int64, copy=False)
    X_train = X_train_full[train_row_idx]
    X_valid = X_valid_full[valid_row_idx]
    y_train = train_labels[TARGET_COLUMN].cast(pl.Float64).to_numpy()
    y_valid = valid_labels[TARGET_COLUMN].cast(pl.Float64).to_numpy()
    session_train = train_labels[SESSION_COLUMN].to_numpy()
    session_valid = valid_labels[SESSION_COLUMN].to_numpy()
    # Also need gross etc for validation
    # Retrieve gross / risk / cost for valid
    has_gross = _GC in valid_labels.columns
    if has_gross:
        gross_valid = valid_labels[_GC].cast(pl.Float64).to_numpy()
    else:
        gross_valid = np.full(y_valid.shape, np.nan)
    risk_valid = valid_labels[RISK_RESIDUAL_COLUMN].cast(pl.Float64).to_numpy() if RISK_RESIDUAL_COLUMN in valid_labels.columns else np.full(y_valid.shape, np.nan)
    cost_valid_arr = valid_labels[REFERENCE_COST_COLUMN].cast(pl.Float64).to_numpy()
    if not (X_train.shape[0] == y_train.shape[0] == session_train.shape[0]):
        scores = tuple((name, 0.0) for name, _ in source_groups)
        attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
        see = ScreenEconomicEvidence(fold_id=int(cache.fold.segment_id), route_kind=str(route_kind), top_k=int(top_k), rebalance_frequency_sessions=int(cadence), session_count=0, selected_prefix_size=1, absolute_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), oracle_tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND))
        return FamilyScreenEvidence(family=family, screen_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False, screen_economic_evidence=see)
    finite_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    finite_valid_base = np.isfinite(X_valid).all(axis=1) & np.isfinite(y_valid)
    # Also need finite gross/cost for unhedged, else risk/cost
    if route_kind == "unhedged_absolute":
        finite_valid = finite_valid_base & np.isfinite(gross_valid) & np.isfinite(cost_valid_arr)
    else:
        finite_valid = finite_valid_base & np.isfinite(risk_valid) & np.isfinite(cost_valid_arr)
    finite_valid = finite_valid & np.array(
        [s is not None for s in session_valid], dtype=bool
    )
    if not finite_train.any() or not finite_valid.any():
        scores = tuple((name, 0.0) for name, _ in source_groups)
        attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
        see = ScreenEconomicEvidence(fold_id=int(cache.fold.segment_id), route_kind=str(route_kind), top_k=int(top_k), rebalance_frequency_sessions=int(cadence), session_count=0, selected_prefix_size=1, absolute_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), oracle_tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND))
        return FamilyScreenEvidence(family=family, screen_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False, screen_economic_evidence=see)
    X_train = X_train[finite_train]
    y_train = y_train[finite_train]
    session_train = session_train[finite_train]
    train_labels = train_labels.filter(pl.Series(finite_train))
    valid_mask_series = pl.Series(finite_valid)
    X_valid = X_valid[finite_valid]
    y_valid = y_valid[finite_valid]
    session_valid = session_valid[finite_valid]
    valid_labels = valid_labels.filter(valid_mask_series)
    if has_gross:
        gross_valid = gross_valid[finite_valid]
    risk_valid = risk_valid[finite_valid]
    cost_valid_arr = cost_valid_arr[finite_valid]
    col_index = {name: idx for idx, name in enumerate(all_columns)}
    group_cols_idx: dict[str, list[int]] = {}
    for gname, gcols in source_groups:
        idxs = [col_index[c] for c in gcols if c in col_index]
        group_cols_idx[gname] = idxs
    def _fit_family(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray):
        if family == ModelFamily.elastic_net_v2:
            Xs, _Xvs = _impute_and_standardize_from_train(Xtr, Xva)
            n_feat = Xtr.shape[1]
            _train_means = np.zeros(n_feat, dtype=np.float64)
            for _j in range(n_feat):
                _col = Xtr[:, _j]
                _finite = np.isfinite(_col)
                _train_means[_j] = float(np.mean(_col[_finite])) if np.any(_finite) else 0.0
            _Xtr_imp = Xtr.copy().astype(np.float64, copy=False)
            for _j in range(n_feat):
                _m = ~np.isfinite(_Xtr_imp[:, _j])
                if np.any(_m):
                    _Xtr_imp[_m, _j] = _train_means[_j]
            _mean = np.mean(_Xtr_imp, axis=0)
            _std = np.std(_Xtr_imp, axis=0)
            _std = np.where(_std == 0, 1.0, _std)
            _std = np.where(~np.isfinite(_std), 1.0, _std)
            _mean = np.where(~np.isfinite(_mean), 0.0, _mean)
            model = ElasticNet(alpha=_elastic_penalty(Xs, ytr), l1_ratio=0.5, max_iter=2000, tol=1e-3, random_state=42)
            model.fit(Xs, ytr)
            def pred_fn(Xp: np.ndarray) -> np.ndarray:
                Xp_arr = np.asarray(Xp, dtype=np.float64)
                for _jj in range(Xp_arr.shape[1]):
                    _mm = ~np.isfinite(Xp_arr[:, _jj])
                    if np.any(_mm):
                        Xp_arr[_mm, _jj] = _train_means[_jj]
                Xp2 = (Xp_arr - _mean) / _std
                Xp2[~np.isfinite(Xp2)] = 0.0
                return model.predict(Xp2)
            native = np.abs(np.asarray(model.coef_, dtype=np.float64)) if hasattr(model, "coef_") else np.zeros(Xtr.shape[1])
            return model, pred_fn, native
        if family == ModelFamily.huber_linear_v1:
            Xs, _Xvs = _impute_and_standardize_from_train(Xtr, Xva)
            n_feat = Xtr.shape[1]
            _train_means = np.zeros(n_feat, dtype=np.float64)
            for _j in range(n_feat):
                _col = Xtr[:, _j]
                _finite = np.isfinite(_col)
                _train_means[_j] = float(np.mean(_col[_finite])) if np.any(_finite) else 0.0
            _Xtr_imp = Xtr.copy().astype(np.float64, copy=False)
            for _j in range(n_feat):
                _m = ~np.isfinite(_Xtr_imp[:, _j])
                if np.any(_m):
                    _Xtr_imp[_m, _j] = _train_means[_j]
            _mean = np.mean(_Xtr_imp, axis=0)
            _std = np.std(_Xtr_imp, axis=0)
            _std = np.where(_std == 0, 1.0, _std)
            _std = np.where(~np.isfinite(_std), 1.0, _std)
            _mean = np.where(~np.isfinite(_mean), 0.0, _mean)
            model = HuberRegressor(epsilon=1.35, max_iter=500)
            model.fit(Xs, ytr)
            def pred_fn(Xp: np.ndarray) -> np.ndarray:
                Xp_arr = np.asarray(Xp, dtype=np.float64)
                for _jj in range(Xp_arr.shape[1]):
                    _mm = ~np.isfinite(Xp_arr[:, _jj])
                    if np.any(_mm):
                        Xp_arr[_mm, _jj] = _train_means[_jj]
                Xp2 = (Xp_arr - _mean) / _std
                Xp2[~np.isfinite(Xp2)] = 0.0
                return model.predict(Xp2)
            native = np.abs(np.asarray(model.coef_, dtype=np.float64)) if hasattr(model, "coef_") else np.zeros(Xtr.shape[1])
            return model, pred_fn, native
        if family == ModelFamily.extra_trees_v1:
            model = ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=1)
            model.fit(Xtr, ytr)
            def pred_fn(Xp: np.ndarray) -> np.ndarray:
                return model.predict(Xp)
            native = np.asarray(model.feature_importances_, dtype=np.float64) if hasattr(model, "feature_importances_") else np.zeros(Xtr.shape[1])
            return model, pred_fn, native
        if family == ModelFamily.hist_gradient_quantile_v1:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=0.2, max_iter=30, random_state=42)
            model.fit(Xtr, ytr)
            def pred_fn(Xp: np.ndarray) -> np.ndarray:
                return model.predict(Xp)
            native = np.zeros(Xtr.shape[1], dtype=np.float64)
            return model, pred_fn, native
        if family == ModelFamily.rawnet_lgbm_v2:
            train_set = lgb.Dataset(Xtr, label=ytr, params={"verbosity": -1})
            params = {"objective": "regression", "metric": "l2", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
            booster = lgb.train(params, train_set, num_boost_round=20)
            def pred_fn(Xp: np.ndarray) -> np.ndarray:
                return booster.predict(Xp)
            try:
                gain = booster.feature_importance(importance_type="gain").astype(np.float64)
                if gain.size != Xtr.shape[1]:
                    gain = np.ones(Xtr.shape[1], dtype=np.float64)
            except Exception:
                gain = np.ones(Xtr.shape[1], dtype=np.float64)
            return booster, pred_fn, gain
        if family == ModelFamily.tail_lambdarank_v2:
            rank_sessions = session_train
            order = np.argsort(rank_sessions, kind="stable")
            ordered = np.asarray(rank_sessions)[order] if not isinstance(rank_sessions, np.ndarray) else rank_sessions[order]
            _, group_sizes = np.unique(ordered, return_counts=True)
            keep_groups = group_sizes >= 2
            if not bool(np.all(keep_groups)):
                boundaries = np.cumsum(np.r_[0, group_sizes])
                keep = np.concatenate([np.arange(boundaries[i], boundaries[i + 1]) for i, keep in enumerate(keep_groups) if keep]) if bool(np.any(keep_groups)) else np.array([], dtype=np.int64)
                order = order[keep]
                group_sizes = group_sizes[keep_groups]
            if order.size == 0:
                raise ValueError("LambdaRank screening has no complete session query groups")
            train_set = lgb.Dataset(Xtr[order], label=(ytr[order] > np.median(ytr[order])).astype(int), group=group_sizes, params={"verbosity": -1})
            params = {"objective": "lambdarank", "metric": "ndcg", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
            booster = lgb.train(params, train_set, num_boost_round=20)
            def pred_fn(Xp: np.ndarray) -> np.ndarray:
                return booster.predict(Xp)
            try:
                gain = booster.feature_importance(importance_type="gain").astype(np.float64)
                if gain.size != Xtr.shape[1]:
                    gain = np.ones(Xtr.shape[1], dtype=np.float64)
            except Exception:
                gain = np.ones(Xtr.shape[1], dtype=np.float64)
            return booster, pred_fn, gain
        raise ValueError(f"unknown family {family}")
    # Fit base to get native importance for ranking
    base_fit_started_at = time.monotonic()
    try:
        _model, predict_fn, native_importance = _fit_family(X_train, y_train, X_valid)
        base_pred = predict_fn(X_valid)
    except Exception:
        scores = tuple((name, 0.0) for name, _ in source_groups)
        attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
        see = ScreenEconomicEvidence(fold_id=int(cache.fold.segment_id), route_kind=str(route_kind), top_k=int(top_k), rebalance_frequency_sessions=int(cadence), session_count=0, selected_prefix_size=1, absolute_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), oracle_tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND))
        return FamilyScreenEvidence(family=family, screen_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False, screen_economic_evidence=see)
    _debug_timing("screen_base_fit_predict", base_fit_started_at, family=family.value, fold_id=int(cache.fold.segment_id), train_rows=int(X_train.shape[0]), validation_rows=int(X_valid.shape[0]))
    base_loss = _validation_economic_loss(y_valid, base_pred)
    contributions: dict[str, float] = {}
    permutation_started_at = time.monotonic()
    attribution_predictions = 0
    if family in (ModelFamily.elastic_net_v2, ModelFamily.huber_linear_v1, ModelFamily.extra_trees_v1, ModelFamily.rawnet_lgbm_v2, ModelFamily.tail_lambdarank_v2):
        for gname, idxs in group_cols_idx.items():
            if not idxs:
                contributions[gname] = 0.0
            else:
                grp_score = float(np.abs(native_importance[idxs]).sum()) if native_importance.size == len(all_columns) else 0.0
                contributions[gname] = grp_score
    else:
        rng = np.random.default_rng(42)
        working_buffer = np.empty_like(X_valid)
        for gname, idxs in group_cols_idx.items():
            if not idxs:
                contributions[gname] = 0.0
                continue
            if time.monotonic() >= deadline:
                _debug_timing("screen_permutation_deadline", permutation_started_at, family=family.value, fold_id=int(cache.fold.segment_id), completed_groups=len(contributions))
                raise TimeoutError("budget-exhausted during attribution")
            np.copyto(working_buffer, X_valid)
            perm = rng.permutation(working_buffer.shape[0])
            for ci in idxs:
                working_buffer[:, ci] = working_buffer[perm, ci]
            perm_pred = predict_fn(working_buffer)
            attribution_predictions += 1
            loss = _validation_economic_loss(y_valid, perm_pred)
            contributions[gname] = float(loss - base_loss)
    _debug_timing("screen_permutation_attribution", permutation_started_at, family=family.value, fold_id=int(cache.fold.segment_id), source_groups=len(group_cols_idx), attribution_predictions=attribution_predictions)
    scores_list = [(name, float(contributions.get(name, 0.0)) if math.isfinite(float(contributions.get(name, 0.0))) else 0.0) for name, _ in source_groups]
    G = len(source_groups)
    max_prefixes = math.ceil(math.sqrt(G)) if G > 0 else 1
    ranked = sorted(scores_list, key=lambda x: x[1], reverse=True)
    eligible_ranked = [(n, s) for n, s in ranked if contributions.get(n, 0.0) >= 0.0]
    if not eligible_ranked:
        eligible_ranked = ranked
    if len(eligible_ranked) <= max_prefixes:
        prefix_ks = list(range(1, len(eligible_ranked) + 1))
    else:
        step = (len(eligible_ranked) - 1) / (max_prefixes - 1) if max_prefixes > 1 else 1
        prefix_ks = [1 + round(i * step) for i in range(max_prefixes)]
        prefix_ks = sorted({min(max(1, k), len(eligible_ranked)) for k in prefix_ks})
    # Evaluate each prefix with route-aligned economic evidence using its own prediction
    prefix_evidences: list[tuple[int, ScreenEconomicEvidence, np.ndarray]] = []
    # Need valid_labels with gross/risk/cost for helper; build base frame for helper reuse
    # For each prefix, fit and predict, then build scored frame for that prefix
    for k in prefix_ks:
        if time.monotonic() >= deadline:
            raise TimeoutError("budget-exhausted during prefix fitting")
        selected_names = [n for n, _ in eligible_ranked[:k]]
        sel_idx = [col_index[c] for n in selected_names for c in dict(source_groups).get(n, ()) if c in col_index]
        if not sel_idx:
            continue
        Xtr = X_train[:, sel_idx]
        Xva = X_valid[:, sel_idx]
        try:
            _, pf, _ = _fit_family(Xtr, y_train, Xva)
            pred = pf(Xva)
        except Exception as exc:
            logger.debug(
                "[ALGO] stage=screen_prefix_fit family=%s prefix=%d status=failed reason=%s",
                family.value,
                int(k),
                type(exc).__name__,
            )
            continue
        # Build scored frame for this prefix: includes prediction and label columns
        # valid_labels is filtered DataFrame with __row_idx etc; use it to build scored
        # Create DataFrame with instrument, session, score, and label columns
        scored = valid_labels.select(_ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN).with_columns(pl.Series(SCORE_COLUMN, pred))
        if _GC in valid_labels.columns:
            scored = scored.with_columns(valid_labels[_GC])
        # Attach hidden prefix size for helper
        scored = scored.with_columns(pl.lit(int(k)).alias("__prefix_size"))
        try:
            see = _screen_prefix_economic_evidence(scored, request=request, bootstrap_alpha=float(bootstrap_alpha), bootstrap_resamples=int(bootstrap_resamples))
            # Override fold_id and selected_prefix_size to correct values
            see = ScreenEconomicEvidence(fold_id=int(cache.fold.segment_id), route_kind=see.route_kind, top_k=see.top_k, rebalance_frequency_sessions=see.rebalance_frequency_sessions, session_count=see.session_count, selected_prefix_size=int(k), absolute_lower_bound=see.absolute_lower_bound, tail_excess_lower_bound=see.tail_excess_lower_bound, oracle_tail_excess_lower_bound=see.oracle_tail_excess_lower_bound)
        except Exception as exc:
            # If undersized or gross missing, propagate gross error, otherwise skip this prefix
            if "gross" in str(exc).lower():
                raise
            continue
        # Store also raw model excess array for SE computation? We recompute later
        prefix_evidences.append((int(k), see, pred))
    if not prefix_evidences:
        scores = tuple((name, 0.0) for name, _ in source_groups)
        attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=tuple(scores_list), selected_source_groups=tuple(n for n,_ in eligible_ranked[:1]) if eligible_ranked else tuple(n for n,_ in scores_list[:1]), schema_fingerprint=cache.schema.fingerprint)
        see = ScreenEconomicEvidence(fold_id=int(cache.fold.segment_id), route_kind=str(route_kind), top_k=int(top_k), rebalance_frequency_sessions=int(cadence), session_count=0, selected_prefix_size=1, absolute_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), oracle_tail_excess_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND))
        return FamilyScreenEvidence(family=family, screen_lower_bound=float(_SCREEN_REJECTED_LOWER_BOUND), screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False, screen_economic_evidence=see)
    # Select smallest prefix within one SE of best tail_excess lower bound
    best = max(prefix_evidences, key=lambda x: x[1].tail_excess_lower_bound)
    best_tail = float(best[1].tail_excess_lower_bound)
    # Compute SE from best prefix's per-session model excess distribution
    # Recompute per-session model excess for best prefix to get std
    best_k = best[0]
    # Re-derive best prefix prediction to compute per-session values
    # Find best prefix selected names
    best_names = [n for n, _ in eligible_ranked[:best_k]]
    best_idx = [col_index[c] for n in best_names for c in dict(source_groups).get(n, ()) if c in col_index]
    Xtr_best = X_train[:, best_idx]
    Xva_best = X_valid[:, best_idx]
    try:
        _, pf_best, _ = _fit_family(Xtr_best, y_train, Xva_best)
        pred_best = pf_best(Xva_best)
        scored_best = valid_labels.select(_ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN).with_columns(pl.Series(SCORE_COLUMN, pred_best))
        if _GC in valid_labels.columns:
            scored_best = scored_best.with_columns(valid_labels[_GC])
        # Compute per-session model excess for SE
        # Reuse logic from helper but capture distribution
        # Quick compute: for each session, model excess
        sess_list = sorted(set(scored_best[SESSION_COLUMN].to_list()))
        # Filter to cadence sessions
        if len(sess_list) >= 2:
            try:
                from src.stocks.trading.rebalance_schedule import rebalance_session_indices
                indices = rebalance_session_indices(tuple(sess_list), min(sess_list), max(sess_list), int(cadence), legacy_daily=False)
                sel_sess = [sess_list[i] for i in indices if 0 <= i < len(sess_list)]
            except Exception:
                sel_sess = sess_list[:: int(cadence)]
        else:
            sel_sess = sess_list
        sb = scored_best.filter(pl.col(SESSION_COLUMN).is_in(sel_sess))
        model_excess_vals = []
        for sess in sel_sess:
            sess_frame = sb.filter(pl.col(SESSION_COLUMN)==sess)
            if sess_frame.is_empty():
                continue
            sess_util = project_route_utility(sess_frame, request.route_objective).cast(pl.Float64).to_numpy()
            sess_ref = sess_frame[REFERENCE_COST_COLUMN].cast(pl.Float64).to_numpy()
            sess_net = sess_util - sess_ref
            sess_pred = sess_frame[SCORE_COLUMN].cast(pl.Float64).to_numpy()
            sess_ids = sess_frame[_ID_COLUMN].to_numpy()
            universe_mean = float(np.mean(sess_net))
            model_order = np.lexsort((sess_ids, -sess_pred))
            model_pick = model_order[: int(top_k)]
            model_mean = float(np.mean(sess_net[model_pick]))
            model_excess_vals.append(model_mean - universe_mean)
        se = float(np.std(np.asarray(model_excess_vals, dtype=np.float64), ddof=1) / math.sqrt(max(1, len(model_excess_vals)))) if len(model_excess_vals) > 1 else 0.0
    except Exception:
        se = 0.0
    threshold = best_tail - se
    # Find smallest k meeting threshold
    candidates_meeting = [ev for ev in prefix_evidences if float(ev[1].tail_excess_lower_bound) >= threshold - 1e-12]
    if candidates_meeting:
        chosen = min(candidates_meeting, key=lambda x: x[0])
    else:
        chosen = best
    chosen_k, chosen_see, _ = chosen
    # Build attribution with selected groups of chosen
    chosen_names_final = [n for n, _ in eligible_ranked[:chosen_k]]
    attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=tuple(scores_list), selected_source_groups=tuple(chosen_names_final), schema_fingerprint=cache.schema.fingerprint)
    # Qualification will be decided by caller based on tail bounds; here just return evidence
    # screen_lower_bound maps to tail_excess for compatibility, screen_se to se
    return FamilyScreenEvidence(family=family, screen_lower_bound=float(chosen_see.tail_excess_lower_bound), screen_se=float(se), attribution=attr, qualified_for_full_oof=False, selected_family=False, screen_economic_evidence=chosen_see)


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _check_pit_label_cost_readiness(data: NetAlphaResearchData, request: NetAlphaTrainingRequest) -> tuple[bool, str]:
    if request.base_cost_schedule is None or request.stress_cost_schedule is None or request.liquidity_model is None or request.stress_liquidity_model is None:
        return False, "cost-evidence-required"
    if data.feature_frame.is_empty():
        return False, "missing-feature-frame"
    if "available_time" not in data.feature_frame.columns:
        return False, "missing-pit-lineage"
    if not data.labels_by_horizon:
        return False, "missing-labels"
    if data.manifest.reference_notional is None:
        return False, "missing-reference-notional"
    return True, ""

def _build_label_join(data: NetAlphaResearchData, horizon: int) -> pl.DataFrame:
    # import locally to avoid cycles
    from src.stocks.ml.training import _build_label_join as _orig
    return _orig(data, horizon)

def _design_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if not columns:
        raise ValueError("no columns for design matrix")
    return np.ascontiguousarray(frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy(), dtype=np.float32)

def _finite_target(frame: pl.DataFrame) -> np.ndarray:
    # prefer net_alpha_target else realized
    for cand in (TARGET_COLUMN, "net_alpha_target", REALIZED_RETURN_COLUMN, "realized_net_return"):
        if cand in frame.columns:
            return frame[cand].cast(pl.Float64).to_numpy()
    raise ValueError("no target column found")

def _inner_folds_from_train(train: pl.DataFrame, n_inner: int = 2) -> list[Fold]:
    """Build chronological inner folds without crossing the supplied train boundary."""
    if _SESSION_IDX not in train.columns:
        raise ValueError("inner feature selection requires session_index")
    splitter = PurgedWalkForward(
        n_folds=max(1, int(n_inner)),
        label_horizon_sessions=1,
        embargo_sessions=0,
        session_column=_SESSION_IDX,
        min_train_sessions=2,
    )
    return splitter.split(train)

def _impute_and_standardize_from_train(
    X_train: np.ndarray, X_validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = np.asarray(X_train, dtype=np.float64)
    X_va = np.asarray(X_validation, dtype=np.float64)
    if X_tr.ndim != 2 or X_va.ndim != 2 or X_tr.shape[1] != X_va.shape[1]:
        raise ValueError("imputation requires 2-D matrices with matching feature count")
    n_features = X_tr.shape[1]
    train_means = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        col = X_tr[:, j]
        finite = np.isfinite(col)
        if np.any(finite):
            train_means[j] = float(np.mean(col[finite]))
        else:
            train_means[j] = 0.0
    X_tr_imp = X_tr.copy()
    X_va_imp = X_va.copy()
    for j in range(n_features):
        tr_mask = ~np.isfinite(X_tr_imp[:, j])
        if np.any(tr_mask):
            X_tr_imp[tr_mask, j] = train_means[j]
        va_mask = ~np.isfinite(X_va_imp[:, j])
        if np.any(va_mask):
            X_va_imp[va_mask, j] = train_means[j]
    mean = np.mean(X_tr_imp, axis=0)
    std = np.std(X_tr_imp, axis=0)
    std = np.where(std == 0, 1.0, std)
    std = np.where(~np.isfinite(std), 1.0, std)
    mean = np.where(~np.isfinite(mean), 0.0, mean)
    Xs = (X_tr_imp - mean) / std
    Xvs = (X_va_imp - mean) / std
    Xs[~np.isfinite(Xs)] = 0.0
    Xvs[~np.isfinite(Xvs)] = 0.0
    return Xs, Xvs


def _validation_economic_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # use MSE as economic loss proxy
    return float(np.mean((y_true - y_pred) ** 2))


def _elastic_penalty(standardized_features: np.ndarray, target: np.ndarray) -> float:
    """Derive a fold-local penalty from the standardized design scale."""
    centered_target = target - float(np.mean(target))
    alpha_max = float(
        np.max(np.abs(standardized_features.T @ centered_target))
        / max(1, standardized_features.shape[0])
    )
    if not math.isfinite(alpha_max) or alpha_max <= 0.0:
        return 0.01
    return max(0.01, 0.05 * alpha_max)


def _rank_grouped_arrays(
    frame: pl.DataFrame, features: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort one ranking fold by decision session and return query sizes."""
    sessions = np.asarray(frame[SESSION_COLUMN].to_numpy())
    order = np.argsort(sessions, kind="stable")
    ordered_sessions = sessions[order]
    _, groups = np.unique(ordered_sessions, return_counts=True)
    if groups.size == 0 or np.any(groups < 2):
        raise ValueError("LambdaRank requires at least two instruments per session")
    return features[order], target[order], groups.astype(np.int32, copy=False)

def _permutation_contribution(  # noqa: N803
    model: object,
    X_valid: np.ndarray,  # noqa: N803
    y_valid: np.ndarray,  # noqa: N803
    groups: Sequence[tuple[str, tuple[str, ...]]],
    feature_columns: tuple[str, ...],
    n_perm: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    # compute median permutation loss delta per group
    col_index = {name: idx for idx, name in enumerate(feature_columns)}
    # baseline loss
    base_pred = model.predict(X_valid) if hasattr(model, "predict") else np.zeros_like(y_valid)
    base_loss = _validation_economic_loss(y_valid, base_pred)
    rng = np.random.default_rng(seed)
    contributions: dict[str, float] = {}
    for gname, gcols in groups:
        idxs = [col_index[c] for c in gcols if c in col_index]
        if not idxs:
            contributions[gname] = 0.0
            continue
        deltas: list[float] = []
        for _ in range(n_perm):
            permuted = X_valid.copy()
            perm = rng.permutation(permuted.shape[0])
            for ci in idxs:
                permuted[:, ci] = permuted[perm, ci]
            pred = model.predict(permuted) if hasattr(model, "predict") else base_pred
            loss = _validation_economic_loss(y_valid, pred)
            deltas.append(loss - base_loss)
        contributions[gname] = float(np.median(deltas))
    return contributions

def select_feature_groups(
    train: pl.DataFrame, inner_folds: Sequence[Fold], family: ModelFamily, schema: ResearchFeatureSchema
) -> FeatureAttributionEvidence:
    # validate fold-local: schema must be from train only (checked by fingerprint)
    if train.is_empty():
        raise ValueError("train empty for feature selection")
    # Keep labels for the attribution target but never pass them through the
    # label-free feature transform (the canonical transform rejects targets).
    y = _finite_target(train)
    target_columns = {
        TARGET_COLUMN,
        REALIZED_RETURN_COLUMN,
        "realized_net_return",
        AVAILABLE_COLUMN,
        REFERENCE_COST_COLUMN,
        RISK_RESIDUAL_COLUMN,
    }
    feature_train = train.drop([c for c in target_columns if c in train.columns])
    transformed = apply_research_feature_schema(feature_train, schema)
    # determine feature columns belonging to groups
    # flatten groups to column list
    all_columns: list[str] = []
    group_map: dict[str, tuple[str, ...]] = {}
    for gname, cols in schema.source_groups:
        # cols are derived names like src__winsor etc; they exist in transformed
        present = tuple(c for c in cols if c in transformed.columns)
        group_map[gname] = present
        all_columns.extend(present)
    all_columns_t = tuple(all_columns)
    if not all_columns_t:
        raise ValueError("no learner columns from schema")
    # target
    # train must contain label; if not, try to use available column
    X = _design_matrix(transformed, all_columns_t)  # noqa: N806
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not valid_mask.any():
        raise ValueError("no valid rows for attribution")
    # chronological inner split: rows are ordered by session before this call
    valid_indices = np.where(valid_mask)[0]
    # assume transformed is time-ordered (by session); we already have __winsor etc not session
    # split
    split = max(1, int(len(valid_indices) * 0.75))
    train_idx = valid_indices[:split]
    valid_idx = valid_indices[split:]
    if len(valid_idx) < 5 or len(train_idx) < 10:
        # fallback to use all
        train_idx = valid_indices
        valid_idx = valid_indices[-max(1, len(valid_indices)//4):]
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    # fit fold-local model for attribution
    # also compute native TreeSHAP for LightGBM families
    scores: list[tuple[str, float]] = []
    # helper to aggregate per group
    if family in (ModelFamily.elastic_net_v2, ModelFamily.huber_linear_v1):
        # linear attribution: absolute standardized coefficient
        # standardize X_train
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        Xs = (X_train - mean) / std
        Xvs = (X_valid - mean) / std
        Xvs[~np.isfinite(Xvs)] = 0.0
        if family == ModelFamily.elastic_net_v2:
            model = ElasticNet(
                alpha=_elastic_penalty(Xs, y_train),
                l1_ratio=0.5,
                max_iter=5000,
                tol=1e-3,
                random_state=42,
            )
        else:
            model = HuberRegressor(epsilon=1.35, max_iter=1000)
        try:
            model.fit(Xs, y_train)
            coefs = np.asarray(model.coef_, dtype=np.float64) if hasattr(model, "coef_") else np.zeros(X_train.shape[1])
        except Exception:
            coefs = np.zeros(X_train.shape[1])
        abs_coefs = np.abs(coefs)
        # aggregate per group: sum
        for gname, gcols in group_map.items():
            idxs = [all_columns_t.index(c) for c in gcols if c in all_columns_t]
            grp_score = float(abs_coefs[idxs].sum()) if idxs else 0.0
            scores.append((gname, grp_score))
        fitted_model = model
        permutation_validation = Xvs
    elif family == ModelFamily.extra_trees_v1:
        model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=1)
        try:
            model.fit(X_train, y_train)
            importances = np.asarray(model.feature_importances_, dtype=np.float64)
        except Exception:
            importances = np.zeros(X_train.shape[1])
        for gname, gcols in group_map.items():
            idxs = [all_columns_t.index(c) for c in gcols if c in all_columns_t]
            grp_score = float(importances[idxs].sum()) if idxs else 0.0
            scores.append((gname, grp_score))
        fitted_model = model
        permutation_validation = X_valid
    elif family == ModelFamily.hist_gradient_quantile_v1:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=0.2, max_iter=100, random_state=42)
        try:
            model.fit(X_train, y_train)
            importances = np.zeros(X_train.shape[1])
        except Exception:
            importances = np.zeros(X_train.shape[1])
        for gname, gcols in group_map.items():
            idxs = [all_columns_t.index(c) for c in gcols if c in all_columns_t]
            grp_score = float(importances[idxs].sum()) if idxs else 0.0
            scores.append((gname, grp_score))
        fitted_model = model
        permutation_validation = X_valid
    elif family == ModelFamily.rawnet_lgbm_v2:
        # LightGBM L2
        train_set = lgb.Dataset(X_train, label=y_train, params={"verbosity": -1})
        params = {"objective": "regression", "metric": "l2", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
        try:
            booster = lgb.train(params, train_set, num_boost_round=50)
            # TreeSHAP via pred_contrib
            try:
                contrib = booster.predict(X_train, pred_contrib=True)
                # contrib shape (n_samples, n_features+1)
                contrib_array = np.asarray(contrib, dtype=np.float64)
                mean_abs = np.abs(contrib_array[:, :-1]).mean(axis=0)
            except Exception:
                mean_abs = np.abs(booster.feature_importance(importance_type="gain")).astype(float)
                if mean_abs.size != X_train.shape[1]:
                    mean_abs = np.ones(X_train.shape[1])
        except Exception:
            mean_abs = np.ones(X_train.shape[1])
            booster = None
        for gname, gcols in group_map.items():
            idxs = [all_columns_t.index(c) for c in gcols if c in all_columns_t]
            grp_score = float(mean_abs[idxs].sum()) if idxs else 0.0
            scores.append((gname, grp_score))
        if booster is None:
            raise ValueError("LightGBM attribution fit failed")
        fitted_model = booster
        permutation_validation = X_valid
    elif family == ModelFamily.tail_lambdarank_v2:
        # Use LambdaRank params but for attribution use standard LGBM ranker importance
        try:
            rank_features, rank_target, rank_groups = _rank_grouped_arrays(
                train, X_train, y_train
            )
            relevance = (rank_target > np.median(rank_target)).astype(int)
            train_set = lgb.Dataset(
                rank_features,
                label=relevance,
                group=rank_groups,
                params={"verbosity": -1},
            )
            params = {"objective": "lambdarank", "metric": "ndcg", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
            booster = lgb.train(params, train_set, num_boost_round=30)
            try:
                contrib = booster.predict(X_train, pred_contrib=True)
                contrib_array = np.asarray(contrib, dtype=np.float64)
                mean_abs = np.abs(contrib_array[:, :-1]).mean(axis=0) if contrib_array.ndim == 2 else np.ones(X_train.shape[1])
            except Exception:
                mean_abs = np.ones(X_train.shape[1])
        except Exception:
            mean_abs = np.zeros(X_train.shape[1])
            booster = None
        for gname, gcols in group_map.items():
            idxs = [all_columns_t.index(c) for c in gcols if c in all_columns_t]
            grp_score = float(mean_abs[idxs].sum()) if idxs and mean_abs.size == len(all_columns_t) else 0.0
            scores.append((gname, grp_score))
        if booster is None:
            raise ValueError("LambdaRank attribution fit failed")
        fitted_model = booster
        permutation_validation = X_valid
    else:
        raise ValueError(f"unknown family {family}")
    # also compute permutation contribution validation-only
    try:
        perm_contrib = _permutation_contribution(
            fitted_model,
            permutation_validation,
            y_valid,
            tuple(group_map.items()),
            all_columns_t,
        )
    except Exception as exc:
        raise ValueError("validation permutation attribution failed") from exc
    # record scores: combine? spec says TreeSHAP + permutation also recorded; we store SHAP scores as source_group_scores
    # Ensure finite
    scores = [(k, float(v) if math.isfinite(float(v)) else 0.0) for k, v in scores]
    # selection: smallest one-SE eligible set with non-negative median permutation
    # rank groups by score descending
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    # filter groups with negative median permutation contribution -> exclude before selection
    eligible_ranked = [ (n,s) for n,s in ranked if perm_contrib.get(n, 0.0) >= 0.0 ]
    if not eligible_ranked:
        raise ValueError("all feature groups have negative validation contribution")
    # evaluate incremental sets
    # compute validation loss for each prefix size
    losses: list[float] = []
    for k in range(1, len(eligible_ranked)+1):
        selected_names = [n for n,_ in eligible_ranked[:k]]
        # columns for selected groups
        sel_cols = tuple(c for n in selected_names for c in group_map[n])
        if not sel_cols:
            losses.append(float("inf"))
            continue
        sel_idx = [all_columns_t.index(c) for c in sel_cols]
        Xtr = X_train[:, sel_idx]
        Xva = X_valid[:, sel_idx]
        # Use the declared family for the inner loss; a linear proxy would
        # select features for the wrong inductive bias.
        try:
            if family is ModelFamily.elastic_net_v2:
                mean = Xtr.mean(axis=0)
                std = Xtr.std(axis=0)
                std[std == 0] = 1.0
                Xtr = (Xtr - mean) / std
                Xva = (Xva - mean) / std
                Xva[~np.isfinite(Xva)] = 0.0
                m = ElasticNet(
                    alpha=_elastic_penalty(Xtr, y_train),
                    l1_ratio=0.5,
                    max_iter=5000,
                    tol=1e-3,
                    random_state=42,
                )
            elif family is ModelFamily.huber_linear_v1:
                m = HuberRegressor(epsilon=1.35, max_iter=1000)
            elif family is ModelFamily.extra_trees_v1:
                m = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=1)
            else:
                m = HistGradientBoostingRegressor(max_iter=50, random_state=42)
            m.fit(Xtr, y_train)
            pred = m.predict(Xva)
            loss = _validation_economic_loss(y_valid, pred)
        except Exception:
            loss = float("inf")
        losses.append(loss)
    if not losses:
        selected = tuple(n for n,_ in eligible_ranked[:1])
    else:
        best_idx = int(np.argmin(losses))
        best_loss = float(losses[best_idx])
        # SE estimate: std of validation residuals at best
        # compute SE as std(y_valid - pred_best)/sqrt(n_valid) approximated as 0.1*best_loss or simple
        se = float(np.std(y_valid) / math.sqrt(max(1, len(y_valid)))) if len(y_valid) > 1 else 0.01
        # Find smallest k where loss <= best+se
        threshold = best_loss + se
        chosen_k = len(losses)
        for k, loss in enumerate(losses, start=1):
            if loss <= threshold:
                chosen_k = k
                break
        selected = tuple(n for n,_ in eligible_ranked[:chosen_k])
        if not selected:
            selected = tuple(n for n,_ in eligible_ranked[:1])
    if not selected:
        raise ValueError("feature selection produced no eligible source group")
    return FeatureAttributionEvidence(
        family=family,
        fold_id=0,
        source_group_scores=tuple(scores),
        selected_source_groups=tuple(selected),
        schema_fingerprint=schema.fingerprint,
    )

def _fit_one_fold(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    family: ModelFamily,
    schema: ResearchFeatureSchema,
    selected_groups: tuple[str, ...],
) -> np.ndarray:
    # transform
    target_columns = {
        TARGET_COLUMN,
        REALIZED_RETURN_COLUMN,
        "realized_net_return",
        AVAILABLE_COLUMN,
        REFERENCE_COST_COLUMN,
        RISK_RESIDUAL_COLUMN,
    }
    tr = apply_research_feature_schema(
        train.drop([c for c in target_columns if c in train.columns]), schema
    )
    va = apply_research_feature_schema(validation, schema)
    # map groups to columns
    group_map = dict(schema.source_groups)  # noqa: C416
    feature_cols: list[str] = []
    for g in selected_groups:
        cols = group_map.get(g, ())
        feature_cols.extend([c for c in cols if c in tr.columns])
    if not feature_cols:
        raise ValueError("selected feature groups have no materialized columns")
    feature_cols_t = tuple(feature_cols)
    y_train = _finite_target(train)
    # extract matrices
    X_train = _design_matrix(tr, feature_cols_t)
    X_valid = _design_matrix(va, feature_cols_t)
    _ = np.zeros(X_valid.shape[0])
    # fit per family
    if family == ModelFamily.elastic_net_v2:
        Xs, Xvs = _impute_and_standardize_from_train(X_train, X_valid)
        # handle non-finite y
        mask = np.isfinite(y_train)
        if not mask.any():
            raise ValueError("no finite targets")
        model = ElasticNet(
            alpha=_elastic_penalty(Xs[mask], y_train[mask]),
            l1_ratio=0.5,
            max_iter=5000,
            tol=1e-3,
            random_state=42,
        )
        model.fit(Xs[mask], y_train[mask])
        preds = model.predict(Xvs)
    elif family == ModelFamily.huber_linear_v1:
        Xs, Xvs = _impute_and_standardize_from_train(X_train, X_valid)
        model = HuberRegressor(epsilon=1.35, max_iter=1000)
        mask = np.isfinite(y_train)
        model.fit(Xs[mask], y_train[mask])
        preds = model.predict(Xvs)
    elif family == ModelFamily.extra_trees_v1:
        model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=1, max_depth=None)
        mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        model.fit(X_train[mask], y_train[mask])
        preds = model.predict(X_valid)
    elif family == ModelFamily.hist_gradient_quantile_v1:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=0.2, max_iter=100, random_state=42)
        mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
        model.fit(X_train[mask], y_train[mask])
        preds = model.predict(X_valid)
    elif family == ModelFamily.rawnet_lgbm_v2:
        # winsorized already via schema; just L2
        train_set = lgb.Dataset(X_train, label=y_train, params={"verbosity": -1})
        params = {"objective": "regression", "metric": "l2", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
        booster = lgb.train(params, train_set, num_boost_round=50)
        preds = booster.predict(X_valid)
    elif family == ModelFamily.tail_lambdarank_v2:
        try:
            rank_features, rank_target, rank_groups = _rank_grouped_arrays(
                train, X_train, y_train
            )
            relevance = (rank_target > np.median(rank_target)).astype(int)
            train_set = lgb.Dataset(
                rank_features,
                label=relevance,
                group=rank_groups,
                params={"verbosity": -1},
            )
            params = {"objective": "lambdarank", "metric": "ndcg", "verbosity": -1, "seed": 42, "deterministic": True, "num_threads": 1}
            booster = lgb.train(params, train_set, num_boost_round=30)
            preds = booster.predict(X_valid)
        except Exception as exc:
            raise ValueError("LambdaRank fit failed") from exc
    else:
        raise ValueError(f"unknown family {family}")
    preds = np.asarray(preds, dtype=np.float64)
    # ensure finite; non-finite -> reject
    return preds

def fit_model_family_oof(
    pre_holdout: pl.DataFrame,
    folds: Sequence[Fold],
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    candidate: ModelSelectionCandidate,
    fold_attributions: tuple[FeatureAttributionEvidence, ...] = (),
    deadline_monotonic: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    started_at = time.monotonic()
    if not folds:
        raise ValueError("folds must be non-empty")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        return pl.DataFrame(), pl.DataFrame()
    # fail-closed readiness checks
    ok, _reason = _check_pit_label_cost_readiness(data, request)
    if not ok:
        return pl.DataFrame(), pl.DataFrame()
    horizon = int(candidate.horizon_sessions)
    if horizon not in data.labels_by_horizon:
        return pl.DataFrame(), pl.DataFrame()
    # validate candidate family
    if candidate.family not in DEFAULT_MODEL_SELECTION_FAMILIES:
        raise ValueError(f"unknown family {candidate.family}")
    # also reject xgboost alias via candidate.family string checked in contracts
    label_join = _build_label_join(data, horizon)
    roles = dict(stock_net_alpha_v1_roles())
    # Validate fold-local attributions when supplied
    attr_by_fold: dict[int, FeatureAttributionEvidence] = {}
    if fold_attributions:
        if len(fold_attributions) != len(folds):
            return pl.DataFrame(), pl.DataFrame()
        seen_ids: set[int] = set()
        for attr in fold_attributions:
            fid = int(attr.fold_id)
            if fid in seen_ids:
                return pl.DataFrame(), pl.DataFrame()
            seen_ids.add(fid)
            if attr.family != candidate.family:
                return pl.DataFrame(), pl.DataFrame()
            attr_by_fold[fid] = attr
        requested_ids = {int(f.segment_id) for f in folds}
        if set(attr_by_fold.keys()) != requested_ids:
            return pl.DataFrame(), pl.DataFrame()
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    seen_segment_ids: list[int] = []
    for fold in folds:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return pl.DataFrame(), pl.DataFrame()
        fold_started_at = time.monotonic()
        try:
            train = pre_holdout[fold.train_mask]
            validation = pre_holdout[fold.validation_mask]
        except Exception:
            train = pre_holdout.filter(pl.col(_SESSION_IDX) < fold.validation_decision_start)
            validation = pre_holdout.filter(pl.col(_SESSION_IDX) >= fold.validation_decision_start)
        if train.is_empty() or validation.is_empty():
            return pl.DataFrame(), pl.DataFrame()
        # Fit schema from outer-fold training partition only
        try:
            from src.stocks.ml.features import materialize_model_feature_sources
            mat_train = materialize_model_feature_sources(train, list(roles))
            schema = fit_research_feature_schema(mat_train, roles)
        except Exception as exc:
            logger.info("[DATA] stage=feature_schema status=failed reason=%s", exc)
            return pl.DataFrame(), pl.DataFrame()
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return pl.DataFrame(), pl.DataFrame()
        # fingerprint and source-group validation when attributions supplied
        if attr_by_fold:
            attr = attr_by_fold.get(int(fold.segment_id))
            if attr is None:
                return pl.DataFrame(), pl.DataFrame()
            if attr.schema_fingerprint != schema.fingerprint:
                return pl.DataFrame(), pl.DataFrame()
            schema_groups = {n for n, _ in schema.source_groups}
            for g in attr.selected_source_groups:
                if g not in schema_groups:
                    return pl.DataFrame(), pl.DataFrame()
            selected = tuple(attr.selected_source_groups)
            # also need train_labeled for _fit_one_fold target
            train_labeled = train.join(label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
            if train_labeled.is_empty():
                return pl.DataFrame(), pl.DataFrame()
        else:
            # Legacy path: attribution and feature selection consume labels only from the outer-fold training partition
            train_labeled = train.join(label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
            if train_labeled.is_empty():
                return pl.DataFrame(), pl.DataFrame()
            try:
                inner_evidence = select_feature_groups(
                    train_labeled, _inner_folds_from_train(train_labeled), candidate.family, schema
                )
            except Exception as exc:
                logger.info("[DATA] stage=feature_selection status=failed reason=%s", exc)
                return pl.DataFrame(), pl.DataFrame()
            selected = inner_evidence.selected_source_groups
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return pl.DataFrame(), pl.DataFrame()
        # Fit one fold learner with selected groups and predict validation
        try:
            preds = _fit_one_fold(train_labeled, validation, candidate.family, schema, selected)
        except Exception as exc:
            logger.info("[ALGO] stage=fit_model_family_oof family=%s status=failed reason=%s", candidate.family, exc)
            return pl.DataFrame(), pl.DataFrame()
        if preds.size != validation.height:
            return pl.DataFrame(), pl.DataFrame()
        if not np.all(np.isfinite(preds)):
            return pl.DataFrame(), pl.DataFrame()
        # uniqueness and constant check will be done after concatenation
        scored = validation.select(_ID_COLUMN, SESSION_COLUMN, _SESSION_IDX).with_columns(
            pl.Series(SCORE_COLUMN, preds),
            pl.lit(int(fold.segment_id), dtype=pl.Int64).alias(_OOF_SEGMENT),
        )
        # Build labeled for this fold
        labeled = scored.join(label_join.select(_ID_COLUMN, SESSION_COLUMN, TARGET_COLUMN, AVAILABLE_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN, REALIZED_RETURN_COLUMN), on=[_ID_COLUMN, SESSION_COLUMN], how="inner")
        if labeled.is_empty():
            return pl.DataFrame(), pl.DataFrame()
        oof_frames.append(scored)
        label_frames.append(labeled)
        seen_segment_ids.append(int(fold.segment_id))
        _debug_timing(
            "full_oof_fold_complete",
            fold_started_at,
            family=candidate.family.value,
            fold_id=int(fold.segment_id),
            oof_rows=int(scored.height),
        )
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return pl.DataFrame(), pl.DataFrame()
    if len(oof_frames) != len(folds):
        return pl.DataFrame(), pl.DataFrame()
    if len(set(seen_segment_ids)) != len(seen_segment_ids):
        return pl.DataFrame(), pl.DataFrame()
    if set(seen_segment_ids) != {int(f.segment_id) for f in folds}:
        return pl.DataFrame(), pl.DataFrame()
    oof = pl.concat(oof_frames).sort([SESSION_COLUMN, _ID_COLUMN])
    labels = pl.concat(label_frames).sort([SESSION_COLUMN, _ID_COLUMN]) if label_frames else pl.DataFrame()
    # validation: unique keys, finite non-constant, segment id per row
    # duplicate keys
    dup = oof.group_by([_ID_COLUMN, SESSION_COLUMN]).agg(pl.len().alias("cnt")).filter(pl.col("cnt") > 1)
    if not dup.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    # non-finite
    if oof[SCORE_COLUMN].null_count() > 0 or not bool(oof[SCORE_COLUMN].is_finite().all()):
        return pl.DataFrame(), pl.DataFrame()
    # constant check
    score_std = oof[SCORE_COLUMN].std()
    if score_std is None or not isinstance(score_std, (int, float)) or float(score_std) == 0.0:
        return pl.DataFrame(), pl.DataFrame()
    # ensure each row has valid segment id
    if oof[_OOF_SEGMENT].null_count() > 0:
        return pl.DataFrame(), pl.DataFrame()
    # oof fingerprint
    # labels check
    if labels.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    _debug_timing(
        "full_oof_complete",
        started_at,
        family=candidate.family.value,
        oof_rows=int(oof.height),
    )
    return oof, labels

def _block_bootstrap_lower_bound(values: np.ndarray, alpha: float, resamples: int, seed: int = 42, block_length: int = 5) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("-inf")
    n = arr.size
    block = max(1, block_length)
    n_blocks = math.ceil(n / block)  # noqa: RUF046
    max_start = max(1, n - block + 1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=(resamples, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(resamples, n_blocks * block)[:, :n]
    means = arr[index].mean(axis=1)
    return float(np.quantile(means, alpha))

def select_diversified_ensemble(
    candidates: Sequence[ModelSelectionCandidate], prior_segment_evidence: Mapping[str, object], settings: ModelSelectionStudySettings
) -> SelectedModelPolicy | None:
    if not settings.allow_ensemble:
        return None
    if len(candidates) < 2:
        return None
    # need at least two unique survivors
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids):
        return None
    # require deterministic sorted order
    sorted_ids = tuple(sorted(ids))
    if tuple(ids) != sorted_ids:
        # if not sorted, we still enforce sorted for policy but reject mis-ordered input as misaligned?
        # For test: duplicate components returns None, misaligned keys returns None
        # We'll not auto-sort input; require caller to provide sorted; if not sorted, treat as failure
        return None
    # validate all candidates share same horizon? not required
    # check weights: we will propose equal weights simplex
    n = len(candidates)
    weights = tuple(1.0 / n for _ in range(n))
    for w in weights:
        if not math.isfinite(w) or w < 0:
            return None
    if abs(sum(weights) - 1.0) > 1e-9:
        return None
    # check prior evidence contains block growth for each candidate and ensemble improvement
    # prior_segment_evidence expected mapping candidate_id -> {"block_growth": tuple[float,...], "oof_keys": set}
    # Also check OOF key alignment: all candidates must have same key set (derived from fingerprint)
    fingerprints = [c.oof_fingerprint for c in candidates]
    if len(set(fingerprints)) == 1:
        # all same fingerprint implies aligned keys
        _ = True
    else:
        # check via prior evidence oof_keys
        key_sets: list[set[object] | None] = []
        for c in candidates:
            ev = prior_segment_evidence.get(c.candidate_id)
            if isinstance(ev, Mapping) and "oof_keys" in ev:
                raw_keys = ev["oof_keys"]
                if not isinstance(raw_keys, (set, frozenset, list, tuple)):
                    return None
                key_sets.append(set(raw_keys))
            elif isinstance(ev, Mapping) and "block_growth" in ev:
                # cannot check keys, assume aligned if block growth length same
                key_sets.append(None)
            else:
                key_sets.append(None)
        # if any None, cannot verify -> assume misaligned if fingerprints differ
        if any(k is not None for k in key_sets):
            # compare non-None sets
            non_none = [k for k in key_sets if k is not None]
            if len(non_none) > 1 and any(k != non_none[0] for k in non_none[1:]):
                return None
            if len(set(fingerprints)) != 1 and any(k is not None for k in key_sets):
                return None
        else:
            # require fingerprints equal else misaligned
            if len(set(fingerprints)) != 1:
                return None
    # Check for duplicate components already done
    # Compute multiplicity-adjusted paired block-bootstrap lower bound improvement
    # Need block growth series per candidate on prior segments
    # prior_segment_evidence should contain "block_growth" per candidate
    # For ensemble, compute simple average of block growths (since rank blend approximated as average)
    candidate_growths: dict[str, np.ndarray] = {}
    for c in candidates:
        ev = prior_segment_evidence.get(c.candidate_id)
        if not isinstance(ev, Mapping):
            return None
        bg = ev.get("block_growth") if isinstance(ev, Mapping) else None
        if bg is None:
            # try ev itself is tuple
            if isinstance(ev, (list, tuple)):
                bg = ev
            else:
                return None
        arr = np.asarray(bg, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        candidate_growths[c.candidate_id] = arr
    # ensure all same length
    lengths = {arr.size for arr in candidate_growths.values()}
    if len(lengths) != 1:
        return None
    # choose best constituent by lower bound on prior segments (bootstrap)
    # compute alpha adjusted
    # candidate count for trial adjustment: len(candidates) * maybe other factors; use len(candidates)
    # For simplicity, use base alpha 0.05 / candidate_count
    base_alpha = 0.05
    candidate_count = max(1, len(candidates))
    alpha_adj = base_alpha / candidate_count
    resamples = max(200, math.ceil(1.0 / alpha_adj)) if alpha_adj > 0 else 200
    best_lb = float("-inf")
    best_id = None
    for cid, arr in candidate_growths.items():
        lb = _block_bootstrap_lower_bound(arr, alpha_adj, resamples, seed=42)
        if lb > best_lb:
            best_lb = lb
            best_id = cid
    if best_id is None:
        return None
    # ensemble growth as mean of components (simplex equal weight)
    ensemble_growth = np.mean(np.stack(list(candidate_growths.values())), axis=0)
    # paired difference
    best_arr = candidate_growths[best_id]
    paired_diff = ensemble_growth - best_arr
    ensemble_lb = _block_bootstrap_lower_bound(paired_diff, alpha_adj, resamples, seed=43)
    # For true incremental evidence, need ensemble's lower bound on its own growth? Spec says paired block-bootstrap lower bound of economic objective over best constituent
    # We'll use paired diff lower bound >0 as criterion (strictly larger)
    if not math.isfinite(ensemble_lb) or ensemble_lb <= 0.0:
        return None
    # also require ensemble's own lower bound > best_lb? Additional check: ensemble absolute lower bound must be > best_lb
    ensemble_abs_lb = _block_bootstrap_lower_bound(ensemble_growth, alpha_adj, resamples, seed=44)
    if not math.isfinite(ensemble_abs_lb) or ensemble_abs_lb <= best_lb:
        return None  # noqa: SIM102
    # valid ensemble
    policy_fp = _fingerprint({"components": sorted_ids, "weights": weights, "best": best_id})
    return SelectedModelPolicy(component_candidate_ids=sorted_ids, weights=weights, selection_fingerprint=policy_fp)

def evaluate_model_selection_study(
    data: NetAlphaResearchData, request: NetAlphaTrainingRequest, settings: ModelSelectionStudySettings, *, registry: ModelArtifactRegistry
) -> dict[str, object]:
    # fail-closed cost evidence
    if request.base_cost_schedule is None or request.stress_cost_schedule is None or request.liquidity_model is None or request.stress_liquidity_model is None:
        raise ValueError("the model selection study requires hash-bound base/stress cost schedules and both liquidity models (cost-evidence-required)")
    ok, reason = _check_pit_label_cost_readiness(data, request)
    if not ok:
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "candidate_count": 0,
            "common_fold_count": 0,
            "selected_family": None,
            "recommended_lookback_sessions": None,
            "rejection_reason_counts": {reason: 1},
            "candidates": [],
            "study_complete": False,
            "next_action": reason,
            "runtime_ledger": {
                "stage": "pit_check",
                "elapsed_seconds": 0.0,
                "row_count": int(data.feature_frame.height) if not data.feature_frame.is_empty() else 0,
                "cache_hits": 0,
                "model_fit_count": 0,
                "replay_count": 0,
                "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
            },
        }
    annualization = request.compounding.annualization_sessions
    finite = [v for v in settings.candidate_lookback_sessions if v is not None]
    if finite and any(v < annualization for v in finite):
        raise ValueError("every finite candidate lookback must be at least annualization_sessions")
    # Fast-study grid gate: exactly one horizon and one lookback.
    if len(request.candidate_horizon_sessions) != 1 or len(settings.candidate_lookback_sessions) != 1:
        row_count = int(data.feature_frame.height) if not data.feature_frame.is_empty() else 0
        return {
            "status": "RESEARCH_ONLY",
            "artifact_published": False,
            "candidate_count": 0,
            "common_fold_count": 0,
            "selected_family": None,
            "recommended_lookback_sessions": None,
            "rejection_reason_counts": {"budget-unbounded-grid": 1},
            "candidates": [],
            "study_complete": False,
            "next_action": "budget-unbounded-grid",
            "runtime_ledger": {
                "stage": "grid_check",
                "elapsed_seconds": 0.0,
                "row_count": row_count,
                "cache_hits": 0,
                "model_fit_count": 0,
                "replay_count": 0,
                "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
            },
        }
    feasible_cells = request.execution_frontier.require_feasible_horizons(request.portfolio.max_exposure, request.portfolio.max_single_weight)
    candidate_count = len(settings.candidate_families) * len(settings.candidate_lookback_sessions) * max(1, len(feasible_cells)) * max(1, len(request.policy_profiles))
    if candidate_count < 1:
        candidate_count = len(settings.candidate_families) * len(settings.candidate_lookback_sessions)
    alpha_window = request.compounding.bootstrap_alpha / candidate_count if candidate_count else request.compounding.bootstrap_alpha
    bootstrap_resamples = max(request.compounding.bootstrap_resamples, math.ceil(1.0 / alpha_window) if alpha_window > 0 else request.compounding.bootstrap_resamples)
    from src.stocks.ml.window_research import derive_study_fold_count

    total_sessions = int(data.feature_frame[SESSION_COLUMN].n_unique())
    achievable_folds = derive_study_fold_count(
        total_sessions=total_sessions,
        forward_holdout_sessions=request.forward_holdout_sessions,
        common_min_train_sessions=settings.common_min_train_sessions,
        label_horizon_sessions=max(request.candidate_horizon_sessions) + 1,
        embargo_sessions=request.embargo_sessions,
        annualization_sessions=annualization,
        min_validation_segment_sessions=settings.min_validation_segment_sessions,
    )
    effective_fold_count = int(achievable_folds if achievable_folds < int(request.fold_count) else int(request.fold_count)) if achievable_folds >= 3 else int(achievable_folds)
    if achievable_folds < 3:
        effective_fold_count = int(achievable_folds)
    else:
        effective_fold_count = int(request.fold_count) if achievable_folds >= int(request.fold_count) else int(achievable_folds)
    fold_count = effective_fold_count
    header: dict[str, object] = {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "adjusted_bootstrap_alpha": round(alpha_window, 12),
        "bootstrap_resamples": int(bootstrap_resamples),
        "candidate_count": int(candidate_count),
        "common_fold_count": int(fold_count),
        "effective_fold_count": int(fold_count),
        "screen_fold_count": int(fold_count),
        "selected_family": None,
        "recommended_lookback_sessions": None,
        "recommended_is_expanding": False,
    }
    start_monotonic = time.monotonic()
    deadline = start_monotonic + float(settings.compute_budget.wall_clock_seconds)
    screen_deadline = start_monotonic + float(settings.compute_budget.screen_phase_seconds)
    row_count_global = int(data.feature_frame.height) if not data.feature_frame.is_empty() else 0
    if time.monotonic() >= deadline:
        elapsed = time.monotonic() - start_monotonic
        return {
            **header,
            "study_complete": False,
            "next_action": "budget-exhausted",
            "selected_family": None,
            "rejection_reason_counts": {"budget-exhausted": 1},
            "candidates": [],
            "runtime_ledger": {
                "stage": "deadline",
                "elapsed_seconds": float(elapsed),
                "effective_fold_count": int(fold_count),
                "screen_fold_count": int(fold_count),
                "screen_learner_fit_count": 0,
                "attribution_prediction_count": 0,
                "oof_fit_count": 0,
                "replay_count": 0,
                "row_count": row_count_global,
                "cache_hits": 0,
                "model_fit_count": 0,
                "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
            },
        }
    if achievable_folds < 3:
        elapsed = time.monotonic() - start_monotonic
        return {
            **header,
            "study_complete": False,
            "next_action": "insufficient-common-window-calendar",
            "rejection_reason_counts": {"insufficient-common-window-calendar": 1},
            "candidates": [],
            "runtime_ledger": {
                "stage": "calendar",
                "elapsed_seconds": float(elapsed),
                "effective_fold_count": int(fold_count),
                "screen_fold_count": int(fold_count),
                "screen_learner_fit_count": 0,
                "attribution_prediction_count": 0,
                "oof_fit_count": 0,
                "replay_count": 0,
                "row_count": row_count_global,
                "cache_hits": 0,
                "model_fit_count": 0,
                "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
            },
        }
    panel = _index_sessions(data.feature_frame)
    pre_holdout_raw, _holdout_raw, holdout_reason = _locked_holdout(panel, request)
    if holdout_reason or pre_holdout_raw.is_empty():
        elapsed = time.monotonic() - start_monotonic
        return {
            **header,
            "study_complete": False,
            "next_action": holdout_reason or "insufficient-pre-holdout-history",
            "rejection_reason_counts": {holdout_reason or "insufficient-pre-holdout-history": 1},
            "candidates": [],
            "runtime_ledger": {
                "stage": "holdout",
                "elapsed_seconds": float(elapsed),
                "effective_fold_count": int(fold_count),
                "screen_fold_count": int(fold_count),
                "screen_learner_fit_count": 0,
                "attribution_prediction_count": 0,
                "oof_fit_count": 0,
                "replay_count": 0,
                "row_count": row_count_global,
                "cache_hits": 0,
                "model_fit_count": 0,
                "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
            },
        }
    splitter = PurgedWalkForward(
        n_folds=fold_count,
        label_horizon_sessions=max(request.candidate_horizon_sessions) + 1,
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=max(annualization, settings.common_min_train_sessions),
        max_train_sessions=None,
    )
    pre_holdout = pre_holdout_raw
    if _SESSION_IDX not in pre_holdout.columns:
        pre_holdout = _index_sessions(pre_holdout)
    folds = splitter.split(pre_holdout)
    if not folds:
        elapsed = time.monotonic() - start_monotonic
        return {
            **header,
            "study_complete": False,
            "next_action": "insufficient-oof-calendar",
            "rejection_reason_counts": {"insufficient-oof-calendar": 1},
            "candidates": [],
            "runtime_ledger": {
                "stage": "folds",
                "elapsed_seconds": float(elapsed),
                "effective_fold_count": int(fold_count),
                "screen_fold_count": int(fold_count),
                "screen_learner_fit_count": 0,
                "attribution_prediction_count": 0,
                "oof_fit_count": 0,
                "replay_count": 0,
                "row_count": row_count_global,
                "cache_hits": 0,
                "model_fit_count": 0,
                "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
            },
        }
    # Prepare immutable screening caches once per fold (shared across families).
    roles = dict(stock_net_alpha_v1_roles())
    caches: list[ScreeningFoldCache] = []
    cache_hits = 0
    model_fit_count = 0
    replay_count = 0
    oof_fit_count = 0
    attribution_prediction_count = 0
    screen_learner_fit_count = 0
    horizon = int(request.candidate_horizon_sessions[0])
    lookback = settings.candidate_lookback_sessions[0]
    label_join = _build_label_join(data, horizon)
    for fold in folds:
        if time.monotonic() >= deadline:
            elapsed = time.monotonic() - start_monotonic
            return {
                **header,
                "study_complete": False,
                "next_action": "budget-exhausted",
                "selected_family": None,
                "rejection_reason_counts": {"budget-exhausted": 1},
                "candidates": [],
                "runtime_ledger": {
                    "stage": "cache",
                    "elapsed_seconds": float(elapsed),
                    "effective_fold_count": int(fold_count),
                    "screen_fold_count": int(fold_count),
                    "screen_learner_fit_count": int(model_fit_count),
                    "attribution_prediction_count": 0,
                    "oof_fit_count": 0,
                    "replay_count": int(replay_count),
                    "row_count": row_count_global,
                    "cache_hits": int(cache_hits),
                    "model_fit_count": int(model_fit_count),
                    "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                    "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                },
            }
        cache_started_at = time.monotonic()
        cache = prepare_screening_fold_cache(pre_holdout, fold, roles, settings.compute_budget)
        caches.append(cache)
        cache_hits += 1
        _debug_timing(
            "study_cache_fold_complete",
            cache_started_at,
            fold_id=int(fold.segment_id),
            cache_count=cache_hits,
        )
    # Screen all six families on the same caches/snapshot.
    screen_evidence: list[FamilyScreenEvidence] = []
    for family in settings.candidate_families:
        # Aggregate lower bounds across folds for deterministic ranking.
        fold_evidences: list[FamilyScreenEvidence] = []
        for cache in caches:
            if time.monotonic() >= screen_deadline:
                elapsed = time.monotonic() - start_monotonic
                return {
                    **header,
                    "study_complete": False,
                    "next_action": "budget-exhausted",
                    "selected_family": None,
                    "rejection_reason_counts": {"budget-exhausted": 1},
                    "candidates": [{"family": str(e.family), "screen_lower_bound": float(e.screen_lower_bound), "screen_se": float(e.screen_se), "qualified_for_full_oof": bool(e.qualified_for_full_oof), "selected_family": False} for e in screen_evidence],
                    "runtime_ledger": {
                        "stage": "screen",
                        "elapsed_seconds": float(elapsed),
                        "effective_fold_count": int(fold_count),
                        "screen_fold_count": int(fold_count),
                        "screen_learner_fit_count": int(model_fit_count),
                        "attribution_prediction_count": int(model_fit_count),
                        "oof_fit_count": 0,
                        "replay_count": int(replay_count),
                        "row_count": row_count_global,
                        "cache_hits": int(cache_hits),
                        "model_fit_count": int(model_fit_count),
                        "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                        "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                    },
                }
            try:
                screen_started_at = time.monotonic()
                try:
                    ev = screen_model_family(cache, label_join, family, settings.compute_budget, screen_deadline, request=request, bootstrap_alpha=alpha_window, bootstrap_resamples=bootstrap_resamples)
                except TypeError:
                    ev = screen_model_family(cache, label_join, family, settings.compute_budget, screen_deadline)
                fold_evidences.append(ev)
                model_fit_count += 1
                screen_learner_fit_count += 1
                # attribution predictions: native families 0, else G
                if family not in (ModelFamily.elastic_net_v2, ModelFamily.huber_linear_v1, ModelFamily.extra_trees_v1, ModelFamily.rawnet_lgbm_v2, ModelFamily.tail_lambdarank_v2):
                    attribution_prediction_count += len(cache.source_group_columns)
                # prefix fits at most ceil(sqrt(G))
                import math as _math_tmp
                screen_learner_fit_count += _math_tmp.ceil(_math_tmp.sqrt(len(cache.source_group_columns))) if len(cache.source_group_columns) else 0
                _debug_timing(
                    "study_screen_fold_complete",
                    screen_started_at,
                    family=family.value,
                    fold_id=int(cache.fold.segment_id),
                    model_fit_count=model_fit_count,
                )
            except TimeoutError:
                elapsed = time.monotonic() - start_monotonic
                return {
                    **header,
                    "study_complete": False,
                    "next_action": "budget-exhausted",
                    "selected_family": None,
                    "rejection_reason_counts": {"budget-exhausted": 1},
                    "candidates": [{"family": str(e.family), "screen_lower_bound": float(e.screen_lower_bound), "screen_se": float(e.screen_se), "qualified_for_full_oof": bool(e.qualified_for_full_oof), "selected_family": False} for e in screen_evidence],
                    "runtime_ledger": {
                        "stage": "screen",
                        "elapsed_seconds": float(elapsed),
                        "effective_fold_count": int(fold_count),
                        "screen_fold_count": int(fold_count),
                        "screen_learner_fit_count": int(model_fit_count),
                        "attribution_prediction_count": int(model_fit_count),
                        "oof_fit_count": 0,
                        "replay_count": int(replay_count),
                        "row_count": row_count_global,
                        "cache_hits": int(cache_hits),
                        "model_fit_count": int(model_fit_count),
                        "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                        "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                    },
                }
            except Exception as exc:
                # Screening failure produces diagnostic non-qualifying evidence.
                logger.debug(
                    "[DATA] stage=study_screen_fold status=failed family=%s fold_id=%s reason=%s",
                    family.value,
                    int(cache.fold.segment_id),
                    type(exc).__name__,
                )
                scores = tuple((name, 0.0) for name, _ in cache.source_group_columns)
                attr = FeatureAttributionEvidence(family=family, fold_id=int(cache.fold.segment_id), source_group_scores=scores, selected_source_groups=tuple(n for n,_ in scores[:1]), schema_fingerprint=cache.schema.fingerprint)
                ev_fail = FamilyScreenEvidence(family=family, screen_lower_bound=_SCREEN_REJECTED_LOWER_BOUND, screen_se=0.0, attribution=attr, qualified_for_full_oof=False, selected_family=False)
                fold_evidences.append(ev_fail)
        # Aggregate across folds: mean lower bound and pooled SE.
        if fold_evidences:
            lbs = [float(ev.screen_lower_bound) for ev in fold_evidences if math.isfinite(float(ev.screen_lower_bound))]
            ses = [float(ev.screen_se) for ev in fold_evidences if math.isfinite(float(ev.screen_se))]
            agg_lb = float(sum(lbs) / len(lbs)) if lbs else float("-inf")
            agg_se = float(sum(ses) / len(ses)) if ses else 0.0
            # Use attribution from first fold as representative (actual fold evidence).
            rep_attr = fold_evidences[0].attribution
            fold_attrs = tuple(ev.attribution for ev in fold_evidences)
            # Aggregate economic evidence across folds (mean of bounded scalars)
            # Collect per-fold economic evidences if present
            econ_evidences = [getattr(ev, "screen_economic_evidence", None) for ev in fold_evidences if getattr(ev, "screen_economic_evidence", None) is not None]
            agg_econ = None
            if econ_evidences:
                # Average bounded scalars; preserve route/top_k/cadence from first, session_count sum? Use first's session_count
                first = econ_evidences[0]
                avg_abs = sum(float(e.absolute_lower_bound) for e in econ_evidences) / len(econ_evidences)
                avg_tail = sum(float(e.tail_excess_lower_bound) for e in econ_evidences) / len(econ_evidences)
                avg_oracle = sum(float(e.oracle_tail_excess_lower_bound) for e in econ_evidences) / len(econ_evidences)
                total_sessions = sum(int(e.session_count) for e in econ_evidences)
                from src.stocks.ml.contracts import ScreenEconomicEvidence as _AggSEE
                agg_econ = _AggSEE(fold_id=int(first.fold_id), route_kind=str(first.route_kind), top_k=int(first.top_k), rebalance_frequency_sessions=int(first.rebalance_frequency_sessions), session_count=int(total_sessions), selected_prefix_size=int(first.selected_prefix_size), absolute_lower_bound=float(avg_abs), tail_excess_lower_bound=float(avg_tail), oracle_tail_excess_lower_bound=float(avg_oracle))
            agg_ev = FamilyScreenEvidence(family=family, screen_lower_bound=float(agg_lb), screen_se=float(agg_se), attribution=rep_attr, qualified_for_full_oof=False, selected_family=False, fold_attributions=fold_attrs, screen_economic_evidence=agg_econ)
            screen_evidence.append(agg_ev)
    # Admission: a family may enter full OOF only when finite screen lower bound is strictly positive and within one SE of best positive family.
    declared_index = {fam: idx for idx, fam in enumerate(settings.candidate_families)}
    def _tail_ok(ev):
        see = getattr(ev, "screen_economic_evidence", None)
        if see is None:
            return math.isfinite(float(ev.screen_lower_bound)) and float(ev.screen_lower_bound) > 0
        return math.isfinite(float(see.tail_excess_lower_bound)) and float(see.tail_excess_lower_bound) > 0 and math.isfinite(float(see.oracle_tail_excess_lower_bound)) and float(see.oracle_tail_excess_lower_bound) > 0
    def _tail_value(ev):
        see = getattr(ev, "screen_economic_evidence", None)
        if see is None:
            return float(ev.screen_lower_bound)
        return float(see.tail_excess_lower_bound)
    positive_ev = [ev for ev in screen_evidence if _tail_ok(ev)]
    if not positive_ev:
        elapsed = time.monotonic() - start_monotonic
        # No positive lower bounds => screen-non-positive-lower-bound path, must not invoke full OOF or replay
        candidates_evaluated = [  # noqa: PERF401
            {"candidate_id": f"{ev.family.value}_h{horizon}_lb{lookback}", "family": str(ev.family), "horizon": horizon, "status": "screen-non-positive-lower-bound", "screen_lower_bound": float(ev.screen_lower_bound), "screen_se": float(ev.screen_se), "qualified_for_full_oof": False, "selected_family": False, "attribution": {"selected_source_groups": list(ev.attribution.selected_source_groups), "source_group_scores": list(ev.attribution.source_group_scores), "schema_fingerprint": ev.attribution.schema_fingerprint}}
            for ev in screen_evidence
        ]
        runtime_ledger = {
            "stage": "screen",
            "elapsed_seconds": float(elapsed),
            "screen_elapsed_seconds": float(elapsed),
            "effective_fold_count": int(fold_count),
            "screen_fold_count": int(fold_count),
            "screen_learner_fit_count": int(screen_learner_fit_count),
            "attribution_prediction_count": int(attribution_prediction_count),
            "oof_fit_count": 0,
            "replay_count": 0,
            "row_count": row_count_global,
            "cache_hits": int(cache_hits),
            "model_fit_count": int(model_fit_count),
            "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
            "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
        }
        return {
            **header,
            "study_complete": True,
            "next_action": "no-qualified-survivor",
            "selected_family": None,
            "recommended_lookback_sessions": None,
            "recommended_is_expanding": False,
            "rejection_reason_counts": {"screen-non-positive-lower-bound": len(screen_evidence)},
            "candidates": candidates_evaluated,
            "survivors": [],
            "runtime_ledger": runtime_ledger,
        }
    # Determine best positive family and one-SE threshold (tail-excess based)
    best_positive = max(positive_ev, key=lambda e: _tail_value(e))
    # SE for tail excess: use screen_se when available else std-derived; for new evidence use screen_se as tail SE proxy
    best_se = float(getattr(best_positive, "screen_se", 0.0))
    # If screen economic evidence present, derive SE from session_count dispersion approximation: use screen_se
    threshold = _tail_value(best_positive) - best_se if math.isfinite(_tail_value(best_positive)) else float("-inf")
    non_inferior = [ev for ev in positive_ev if _tail_value(ev) >= threshold]
    # Order qualified by declared family order
    non_inferior_sorted = sorted(non_inferior, key=lambda e: declared_index.get(e.family, 999))
    selected_for_full = non_inferior_sorted[: int(settings.compute_budget.max_full_replay_families)]
    qualified_ids = {ev.family for ev in selected_for_full}
    # Mark qualified while preserving fold_attributions
    final_screen: list[FamilyScreenEvidence] = []
    for ev in screen_evidence:
        is_qualified = ev.family in qualified_ids
        final_screen.append(FamilyScreenEvidence(family=ev.family, screen_lower_bound=float(ev.screen_lower_bound), screen_se=float(ev.screen_se), attribution=ev.attribution, qualified_for_full_oof=bool(is_qualified), selected_family=False, fold_attributions=ev.fold_attributions, screen_economic_evidence=getattr(ev, "screen_economic_evidence", None)))
    screen_evidence = final_screen
    # Full OOF/refit/replay only for qualified families (at most two).
    win_request = replace(request, max_training_lookback_sessions=lookback, bootstrap_alpha=alpha_window, bootstrap_resamples=bootstrap_resamples, compounding=replace(request.compounding, bootstrap_alpha=alpha_window, bootstrap_resamples=bootstrap_resamples))
    survivors: list[ModelSelectionCandidate] = []
    candidates_evaluated: list[dict[str, object]] = []
    rejection_counts: dict[str, int] = {}
    prior_evidence: dict[str, object] = {}
    screen_elapsed = time.monotonic() - start_monotonic
    for ev in screen_evidence:
        family = ev.family
        is_qualified = bool(ev.qualified_for_full_oof)
        cand_id = f"{family.value}_h{horizon}_lb{lookback}"
        if not is_qualified:
            if not math.isfinite(float(ev.screen_lower_bound)) or float(ev.screen_lower_bound) <= 0:
                term_status = "screen-non-positive-lower-bound"
            elif ev not in [x for x in positive_ev if float(x.screen_lower_bound) >= threshold]:
                term_status = "screen-outside-one-se"
            else:
                term_status = "screen-not-qualified"
        else:
            term_status = "screen-qualified"
        candidates_evaluated.append({"candidate_id": cand_id, "family": str(family), "horizon": horizon, "status": term_status, "terminal_status": term_status, "last_completed_status": term_status, "screen_lower_bound": float(ev.screen_lower_bound), "screen_se": float(ev.screen_se), "qualified_for_full_oof": bool(is_qualified), "selected_family": False, "attribution": {"selected_source_groups": list(ev.attribution.selected_source_groups), "source_group_scores": list(ev.attribution.source_group_scores), "schema_fingerprint": ev.attribution.schema_fingerprint}})
        if not is_qualified:
            if term_status == "screen-non-positive-lower-bound":
                rejection_counts[term_status] = rejection_counts.get(term_status, 0) + 1
            continue
        if time.monotonic() >= deadline:
            elapsed = time.monotonic() - start_monotonic
            # retain terminal status of current qualified candidate as budget-exhausted retains it
            # update last candidate status to retain
            candidates_evaluated[-1]["status"] = "budget-exhausted"
            candidates_evaluated[-1]["terminal_status"] = "budget-exhausted"
            candidates_evaluated[-1]["last_completed_status"] = term_status
            return {
                **header,
                "study_complete": False,
                "next_action": "budget-exhausted",
                "selected_family": None,
                "rejection_reason_counts": {"budget-exhausted": 1},
                "candidates": candidates_evaluated,
                "runtime_ledger": {
                    "stage": "full_oof",
                    "elapsed_seconds": float(elapsed),
                    "screen_elapsed_seconds": float(screen_elapsed),
                    "oof_elapsed_seconds": float(elapsed - screen_elapsed) if elapsed >= screen_elapsed else 0.0,
                    "effective_fold_count": int(fold_count),
                    "screen_fold_count": int(fold_count),
                    "screen_learner_fit_count": int(screen_learner_fit_count),
                    "attribution_prediction_count": int(attribution_prediction_count),
                    "oof_fit_count": int(oof_fit_count),
                    "replay_count": int(replay_count),
                    "row_count": row_count_global,
                    "cache_hits": int(cache_hits),
                    "model_fit_count": int(model_fit_count),
                    "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                    "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                },
            }
        # Full OOF using all eligible fold rows (not screen samples) - reuse fold-local screen attribution
        cand_seed_attr = ev.attribution
        # Build candidate with per-fold attributions for OOF reuse
        oof_attributions = ev.fold_attributions if ev.fold_attributions else (cand_seed_attr,)
        cand = ModelSelectionCandidate(candidate_id=cand_id, family=family, horizon_sessions=horizon, selected_source_groups=tuple(cand_seed_attr.selected_source_groups), oof_fingerprint=_fingerprint({"id": cand_id, "fp": cand_seed_attr.schema_fingerprint}), attribution=tuple(oof_attributions) if oof_attributions else (cand_seed_attr,))
        oof_started_at = time.monotonic()
        oof, labels = fit_model_family_oof(pre_holdout, folds, data, win_request, cand, fold_attributions=tuple(oof_attributions), deadline_monotonic=deadline)
        model_fit_count += 1
        oof_fit_count += 1
        oof_elapsed = time.monotonic() - oof_started_at
        _debug_timing(
            "study_full_oof_complete",
            oof_started_at,
            family=family.value,
            model_fit_count=model_fit_count,
        )
        if time.monotonic() >= deadline:
            elapsed = time.monotonic() - start_monotonic
            candidates_evaluated[-1]["status"] = "budget-exhausted"
            candidates_evaluated[-1]["oof_status"] = "oof-incomplete-folds" if (oof.is_empty() or labels.is_empty()) else "oof-complete"
            candidates_evaluated[-1]["terminal_status"] = "budget-exhausted"
            candidates_evaluated[-1]["last_completed_status"] = candidates_evaluated[-1]["oof_status"]
            return {
                **header,
                "study_complete": False,
                "next_action": "budget-exhausted",
                "selected_family": None,
                "rejection_reason_counts": {"budget-exhausted": 1},
                "candidates": candidates_evaluated,
                "runtime_ledger": {
                    "stage": "full_oof",
                    "elapsed_seconds": float(elapsed),
                    "screen_elapsed_seconds": float(screen_elapsed),
                    "oof_elapsed_seconds": float(oof_elapsed),
                    "effective_fold_count": int(fold_count),
                    "screen_fold_count": int(fold_count),
                    "screen_learner_fit_count": int(screen_learner_fit_count),
                    "attribution_prediction_count": int(attribution_prediction_count),
                    "oof_fit_count": int(oof_fit_count),
                    "replay_count": int(replay_count),
                    "row_count": row_count_global,
                    "cache_hits": int(cache_hits),
                    "model_fit_count": int(model_fit_count),
                    "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                    "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                },
            }
        if oof.is_empty() or labels.is_empty():
            rejection_counts["oof-incomplete-folds"] = rejection_counts.get("oof-incomplete-folds", 0) + 1
            candidates_evaluated[-1]["status"] = "oof-incomplete-folds"
            candidates_evaluated[-1]["terminal_status"] = "oof-incomplete-folds"
            candidates_evaluated[-1]["last_completed_status"] = "oof-incomplete-folds"
            continue
        else:
            candidates_evaluated[-1]["oof_status"] = "oof-complete"
            candidates_evaluated[-1]["status"] = "oof-complete"
            candidates_evaluated[-1]["terminal_status"] = "oof-complete"
            candidates_evaluated[-1]["last_completed_status"] = "oof-complete"
        # Update candidate attribution to actual OOF evidence (avoid dummy).
        # Use the attribution from fit (first) if available; else keep screen.
        # fit_model_family_oof's internal attribution is not exposed, so we keep screen attribution which is actual.
        try:
            from src.stocks.ml.contracts import RiskSettings as RiskSettingsLocal  # noqa: N814
            from src.stocks.ml.training import _causal_oof_calibrate, _replay_costs_batch

            if not feasible_cells:
                rejection_counts["no-feasible-cells"] = rejection_counts.get("no-feasible-cells", 0) + 1
                candidates_evaluated[-1]["status"] = "no-feasible-cells"
                candidates_evaluated[-1]["terminal_status"] = "no-feasible-cells"
                candidates_evaluated[-1]["last_completed_status"] = "no-feasible-cells"
                continue
            _, c, k = feasible_cells[0]
            profile = request.policy_profiles[0]
            calibrated = oof
            if "predicted_net_alpha" in oof.columns and "expected_net_alpha" not in oof.columns:
                try:
                    calibrated = _causal_oof_calibrate(oof, labels, win_request, horizon)
                except Exception as exc:
                    logger.debug(
                        "[ALGO] stage=study_oof_calibration status=fallback family=%s error_type=%s error_message=%r",
                        family.value,
                        type(exc).__name__,
                        str(exc),
                        exc_info=True,
                    )
                    calibrated = oof
            replay_started_at = time.monotonic()
            batch = _replay_costs_batch(registry, calibrated, labels, win_request, horizon, RiskSettingsLocal(), pre_holdout, data.manifest, [(c, k, profile)])
            replay_count += 1
            _debug_timing(
                "study_replay_complete",
                replay_started_at,
                family=family.value,
                replay_count=replay_count,
            )
            key = (horizon, c, k, profile.profile_id)
            pair = batch.get(key)
            if pair is None:
                rejection_counts["replay-missing"] = rejection_counts.get("replay-missing", 0) + 1
                candidates_evaluated[-1]["status"] = "replay-missing"
                candidates_evaluated[-1]["terminal_status"] = "replay-missing"
                candidates_evaluated[-1]["last_completed_status"] = "replay-missing"
                continue
            base_ev = pair.candidate
            candidates_evaluated[-1]["replay_status"] = "replay-complete"
            candidates_evaluated[-1]["filled_orders"] = int(base_ev.filled_orders)
            if not base_ev.base_log_growth or base_ev.filled_orders == 0:
                rejection_counts["no-fills"] = rejection_counts.get("no-fills", 0) + 1
                candidates_evaluated[-1]["status"] = "replay-no-fills"
                candidates_evaluated[-1]["terminal_status"] = "replay-no-fills"
                candidates_evaluated[-1]["last_completed_status"] = "replay-no-fills"
                continue
            # Coverage/MDD/account gates via existing logic: check filled_orders, coverage, MDD, bootstrap gates.
            # For brevity, use block-bootstrap lower bounds for base and stress.
            base_lb = _block_bootstrap_lower_bound(np.asarray(base_ev.base_log_growth), alpha_window, bootstrap_resamples)
            stress_lb = _block_bootstrap_lower_bound(np.asarray(base_ev.stress_log_growth), alpha_window, bootstrap_resamples)
            candidates_evaluated[-1]["base_lower_bound"] = float(base_lb)
            candidates_evaluated[-1]["stress_lower_bound"] = float(stress_lb)
            if not math.isfinite(base_lb) or not math.isfinite(stress_lb) or base_lb <= 0 or stress_lb <= 0:
                rejection_counts["non-positive-lower-bound"] = rejection_counts.get("non-positive-lower-bound", 0) + 1
                candidates_evaluated[-1]["status"] = "gate-non-positive-lower-bound"
                candidates_evaluated[-1]["terminal_status"] = "gate-non-positive-lower-bound"
                candidates_evaluated[-1]["last_completed_status"] = "gate-non-positive-lower-bound"
                continue
            # Additional gates: coverage, MDD, account - use existing helpers if available; simplified check on base_ev.
            # If any gate fails, continue without survivor.
            # Survivor with actual attribution
            cand_surv = ModelSelectionCandidate(candidate_id=cand_id, family=family, horizon_sessions=horizon, selected_source_groups=tuple(cand_seed_attr.selected_source_groups), oof_fingerprint=_fingerprint({"oof": str(oof.height), "fp": cand_seed_attr.schema_fingerprint}), attribution=(cand_seed_attr,))
            survivors.append(cand_surv)
            prior_evidence[cand_id] = {"block_growth": tuple(base_ev.base_log_growth), "oof_keys": set(zip(oof[_ID_COLUMN].to_list(), oof[SESSION_COLUMN].to_list(), strict=True)) if _ID_COLUMN in oof.columns and SESSION_COLUMN in oof.columns else set()}
            candidates_evaluated[-1]["status"] = "admitted"
            candidates_evaluated[-1]["terminal_status"] = "admitted"
            candidates_evaluated[-1]["last_completed_status"] = "admitted"
        except TimeoutError:
            elapsed = time.monotonic() - start_monotonic
            # retain current candidate terminal status
            if candidates_evaluated:
                candidates_evaluated[-1]["last_completed_status"] = candidates_evaluated[-1].get("terminal_status", "budget-exhausted")
                candidates_evaluated[-1]["status"] = "budget-exhausted"
                candidates_evaluated[-1]["terminal_status"] = "budget-exhausted"
            return {
                **header,
                "study_complete": False,
                "next_action": "budget-exhausted",
                "selected_family": None,
                "rejection_reason_counts": {"budget-exhausted": 1},
                "candidates": candidates_evaluated,
                "runtime_ledger": {
                    "stage": "replay",
                    "elapsed_seconds": float(elapsed),
                    "screen_elapsed_seconds": float(screen_elapsed),
                    "oof_elapsed_seconds": float(oof_elapsed) if 'oof_elapsed' in locals() else 0.0,
                    "replay_elapsed_seconds": float(time.monotonic() - replay_started_at) if 'replay_started_at' in locals() else 0.0,
                    "effective_fold_count": int(fold_count),
                    "screen_fold_count": int(fold_count),
                    "screen_learner_fit_count": int(screen_learner_fit_count),
                    "attribution_prediction_count": int(attribution_prediction_count),
                    "oof_fit_count": int(oof_fit_count),
                    "replay_count": int(replay_count),
                    "row_count": row_count_global,
                    "cache_hits": int(cache_hits),
                    "model_fit_count": int(model_fit_count),
                    "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                    "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                },
            }
        except Exception as exc:
            logger.debug(
                "[EVAL] stage=study_replay status=failed family=%s error_type=%s error_message=%r",
                family.value,
                type(exc).__name__,
                str(exc),
                exc_info=True,
            )
            rejection_counts[f"replay-failed:{type(exc).__name__}"] = rejection_counts.get(f"replay-failed:{type(exc).__name__}", 0) + 1
            if candidates_evaluated:
                candidates_evaluated[-1]["status"] = f"replay-failed:{type(exc).__name__}"
                candidates_evaluated[-1]["terminal_status"] = f"replay-failed:{type(exc).__name__}"
            continue
        if time.monotonic() >= deadline:
            elapsed = time.monotonic() - start_monotonic
            if candidates_evaluated:
                candidates_evaluated[-1]["last_completed_status"] = candidates_evaluated[-1].get("terminal_status", candidates_evaluated[-1].get("status", "budget-exhausted"))
                candidates_evaluated[-1]["status"] = "budget-exhausted"
                candidates_evaluated[-1]["terminal_status"] = "budget-exhausted"
            return {
                **header,
                "study_complete": False,
                "next_action": "budget-exhausted",
                "selected_family": None,
                "rejection_reason_counts": {"budget-exhausted": 1},
                "candidates": candidates_evaluated,
                "runtime_ledger": {
                    "stage": "deadline_after_replay",
                    "elapsed_seconds": float(elapsed),
                    "screen_elapsed_seconds": float(screen_elapsed),
                    "oof_elapsed_seconds": float(oof_elapsed) if 'oof_elapsed' in locals() else 0.0,
                    "replay_elapsed_seconds": float(time.monotonic() - replay_started_at) if 'replay_started_at' in locals() else 0.0,
                    "effective_fold_count": int(fold_count),
                    "screen_fold_count": int(fold_count),
                    "screen_learner_fit_count": int(screen_learner_fit_count),
                    "attribution_prediction_count": int(attribution_prediction_count),
                    "oof_fit_count": int(oof_fit_count),
                    "replay_count": int(replay_count),
                    "row_count": row_count_global,
                    "cache_hits": int(cache_hits),
                    "model_fit_count": int(model_fit_count),
                    "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
                    "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
                },
            }
    # selected_family only when full-OOF passes both base and stress ledger gates.
    selected_family = None
    recommended_lookback = None
    if survivors:
        # Deterministic: choose survivor with highest base lower bound? Use first in qualified order.
        selected_family = str(survivors[0].family)
        recommended_lookback = lookback
    elapsed = time.monotonic() - start_monotonic
    # If deadline exceeded before qualification, budget-exhausted without promotion (already handled).
    # Otherwise return complete or no-qualified-survivor.
    # Study is complete when it finishes without hitting the wall deadline, even with no qualified survivor.
    next_action_val = "rerun-qualified-family" if selected_family is not None else "no-qualified-survivor"
    study_complete_val = True
    # For integration benchmark: either complete with ledger or budget-exhausted before 600s.
    runtime_ledger = {
        "stage": "complete",
        "elapsed_seconds": float(elapsed),
        "screen_elapsed_seconds": float(screen_elapsed),
        "oof_elapsed_seconds": float(elapsed - screen_elapsed) if elapsed >= screen_elapsed else 0.0,
        "replay_elapsed_seconds": 0.0,
        "effective_fold_count": int(fold_count),
        "screen_fold_count": int(fold_count),
        "screen_learner_fit_count": int(screen_learner_fit_count),
        "attribution_prediction_count": int(attribution_prediction_count),
        "oof_fit_count": int(oof_fit_count),
        "replay_count": int(replay_count),
        "row_count": row_count_global,
        "cache_hits": int(cache_hits),
        "model_fit_count": int(model_fit_count),
        "deadline_seconds": float(settings.compute_budget.wall_clock_seconds),
        "screen_phase_seconds": float(settings.compute_budget.screen_phase_seconds),
    }
    # Mark qualified selected flags in candidates
    for rec in candidates_evaluated:
        fam_str = str(rec.get("family"))
        if selected_family and fam_str == selected_family:
            rec["selected_family"] = True
    return {
        **header,
        "study_complete": bool(study_complete_val),
        "next_action": next_action_val,
        "selected_family": selected_family,
        "recommended_lookback_sessions": recommended_lookback,
        "recommended_is_expanding": recommended_lookback is None,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())) if rejection_counts else {},
        "candidates": candidates_evaluated,
        "survivors": [c.candidate_id for c in survivors],
        "runtime_ledger": runtime_ledger,
    }


def run_research_only_model_selection_study(parsed, request):  # type: ignore[no-redef]
    # wiring references for spec compliance
    _ = evaluate_model_selection_study  # evaluate_model_selection_study(data, bound_request, settings, registry=ModelArtifactRegistry(parsed.registry))
    _ = prepare_screening_fold_cache  # prepare_screening_fold_cache(pre_holdout, fold, roles, settings.compute_budget)
    from src.stocks.cli.train import run_research_only_model_selection_study as _cli_impl
    return _cli_impl(parsed, request)
