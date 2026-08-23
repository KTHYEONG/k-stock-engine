"""OOF fitting infrastructure: prepared-array fold fitting plus spill cache.

``OofCache`` manages the per-run temporary spill cache for OOF Parquet files;
``atomic_write_parquet``/``read_oof_parquet`` are shared I/O helpers.
``fit_horizon_oof`` owns the prepared-array fold loop and alpha selection
moved out of ``training.py``: one canonical matrix, integer fold plans, exact
array Rank-IC, and Polars OOF frames constructed only once at the boundary.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from src.core.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.stats import spearmanr as _spearmanr  # noqa: F401

    from src.stocks.ml.contracts import (
        FoldScoreDiagnostic,
        NetAlphaTrainingRequest,
        RegularizationGrid,
    )
    from src.stocks.ml.models import ElasticPathResult
    from src.stocks.ml.preparation import (
        PreparedFold,
        PreparedHorizonLabels,
        PreparedTrainingMatrix,
    )


def default_oof_cache_base() -> Path:
    return PROJECT_ROOT / "tmp" / "training"


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write a Zstandard Parquet file atomically via a same-dir rename."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temp_path, compression="zstd")
    os.replace(temp_path, path)


def read_oof_parquet(path: Path) -> pl.DataFrame:
    """Load a cached OOF file; missing/corrupt files raise ``ValueError``."""
    if not path.exists():
        raise ValueError(f"missing cached OOF file {path}")
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"corrupt cached OOF file {path}: {exc}") from exc


class OofCache:
    """Per-run temporary spill cache for OOF Parquet files.

    Admitted horizons write the calibrated OOF scores and the label join as
    separate Zstandard Parquet files and release the DataFrames; only the file
    paths and the small Rank-IC tuple stay in process memory.
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
        atomic_write_parquet(calibrated, oof_path)
        atomic_write_parquet(labels, labels_path)
        self._cache_bytes += oof_path.stat().st_size + labels_path.stat().st_size
        return oof_path, labels_path

    def load(self, horizon_sessions: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        oof_path = self._root / f"horizon_{horizon_sessions}_oof.parquet.zst"
        labels_path = self._root / f"horizon_{horizon_sessions}_labels.parquet.zst"
        return read_oof_parquet(oof_path), read_oof_parquet(labels_path)

    def close(self) -> None:
        if not self._closed:
            self._temporary.cleanup()
            self._closed = True


_ALPHA_TIE_TOLERANCE = 1e-12
_NESTED_INNER_FOLDS = 3
_NESTED_MIN_TRAIN_SESSIONS = 5
_ID_COLUMN = "instrument_id"
_SESSION_COLUMN = "session"
_SESSION_IDX = "session_index"
_SCORE_COLUMN = "predicted_net_alpha"
_OOF_SEGMENT = "oof_segment_id"


def _matrix_aligned_target(
    horizon: PreparedHorizonLabels, num_rows: int
) -> np.ndarray:
    """Narrow NaN-padded float64 target vector aligned to canonical rows."""
    target = np.full(num_rows, np.nan, dtype=np.float64)
    target[horizon.row_index] = horizon.target
    return target


def _plan_fitting_workspace(
    request: NetAlphaTrainingRequest,
    planned_bytes: int,
    stage: str,
) -> None:
    """Fail closed before one fitting workspace allocation.

    The typed envelope checks every finite headroom (request RSS budget minus
    current process RSS, cgroup limit minus current minus reserve, and
    ``MemAvailable`` minus reserve) before any array is materialized.
    """
    from src.stocks.ml.replay_resources import (
        MemoryBudgetExceededError,
        plan_training_allocation,
    )

    envelope = plan_training_allocation(
        planned_bytes,
        request_limit_bytes=(
            None
            if request.max_rss_mib is None
            else int(request.max_rss_mib) * 1024 * 1024
        ),
        reserve_bytes=int(request.memory_reserve_mib) * 1024 * 1024,
    )
    if not envelope.ok:
        finite_headrooms = [
            headroom
            for headroom in (
                envelope.process_headroom_bytes,
                envelope.cgroup_headroom_bytes,
                envelope.system_headroom_bytes,
            )
            if headroom is not None
        ]
        raise MemoryBudgetExceededError(
            f"{stage} workspace of {planned_bytes} bytes exceeds memory "
            f"envelope: {envelope.reason}",
            planned_bytes=planned_bytes,
            limit_bytes=min(finite_headrooms) if finite_headrooms else 0,
        )


@dataclass(frozen=True, slots=True)
class OofFitRequest:
    """Typed context for one prepared-array horizon OOF fit.

    ``request`` is the immutable training request; ``manifest`` carries the
    base model manifest for provenance; ``family`` must be the elastic
    baseline, which owns the discovery hot path.
    """

    request: Any
    manifest: object = None
    family: str = "net_alpha_elastic_net"
    model_factory: Callable[[], object] | None = None


@dataclass(frozen=True, slots=True)
class OofFitResult:
    """Immutable outcome of one horizon's prepared-array OOF fit."""

    oof: pl.DataFrame
    labeled: pl.DataFrame
    fold_rank_ics: list[float]
    diagnostic: Any
    path_evaluations: int


def session_rank_ic_from_arrays(
    scores: np.ndarray,
    realized: np.ndarray,
    session_codes: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Exact session-mean Spearman Rank-IC over finite valid rows.

    Sessions iterate in ascending code order (chronological) and rows keep
    their original matrix order inside a session; a session with fewer than
    two rows or zero score/realized variance is skipped, matching the
    frame-based reference.
    """
    from scipy.stats import spearmanr

    scores_arr = np.asarray(scores, dtype=np.float64)
    realized_arr = np.asarray(realized, dtype=np.float64)
    codes_arr = np.asarray(session_codes)
    finite_valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(scores_arr)
        & np.isfinite(realized_arr)
    )
    if not finite_valid.any():
        return 0.0
    selected_scores = scores_arr[finite_valid]
    selected_realized = realized_arr[finite_valid]
    selected_codes = codes_arr[finite_valid]
    order = np.argsort(selected_codes, kind="stable")
    sorted_codes = selected_codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    ics: list[float] = []
    for group in np.split(order, boundaries):
        if group.size < 2:
            continue
        session_scores = selected_scores[group]
        session_realized = selected_realized[group]
        if np.std(session_scores) == 0.0 or np.std(session_realized) == 0.0:
            continue
        rho, _ = spearmanr(session_scores, session_realized)
        ics.append(float(rho))
    return float(np.mean(ics)) if ics else 0.0


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


def _score_is_constant(values: np.ndarray) -> bool:
    """True when every finite value is equal (a degenerate prediction)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return True
    return bool(np.all(finite == finite[0]))


def _select_elastic_alpha_prepared(
    matrix: PreparedTrainingMatrix,
    horizon: PreparedHorizonLabels,
    train_labeled: np.ndarray,
    request: NetAlphaTrainingRequest,
    grid: RegularizationGrid,
) -> tuple[float | None, float | None, float | None, int]:
    """Fold-local scale-invariant penalty selection on indexed workspaces.

    Mirrors the historical nested selection: purged expanding inner folds over
    the labeled training slice, one weighted coordinate path per inner fold on
    a bounded-chunk Fortran-order design, per-fraction session Rank-IC through
    exact array ranks, the shared alpha_max mean, and the deterministic
    stronger-penalty fallback. No full-size ``matrix.X[rows]`` copy exists on
    this path.
    """
    from src.stocks.ml.models import (
        ELASTIC_DESIGN_CHUNK_ROWS,
        fit_prepared_elastic_path,
        prepare_indexed_elastic_design,
    )
    from src.stocks.research.folds import PurgedWalkForward

    x_matrix = matrix.X
    codes = matrix.session_code
    target_full = _matrix_aligned_target(horizon, matrix.num_rows)
    # PurgedWalkForward.max_train_sessions applies the request's rolling
    # lookback cap to nested selection exactly as the outer discovery plan.
    splitter = PurgedWalkForward(
        n_folds=request.fold_count,
        label_horizon_sessions=int(horizon.horizon_sessions),
        embargo_sessions=request.embargo_sessions,
        session_column=_SESSION_IDX,
        min_train_sessions=_NESTED_MIN_TRAIN_SESSIONS,
        max_train_sessions=request.max_training_lookback_sessions,
    )
    slice_frame = pl.DataFrame({_SESSION_IDX: codes[train_labeled]})
    nested = splitter.inner_folds(slice_frame, n_inner=_NESTED_INNER_FOLDS)
    if not nested:
        return None, None, None, 0
    candidate_ics: dict[float, list[float]] = {
        fraction: [] for fraction in grid.fractions
    }
    constant: set[float] = set()
    alpha_maxes: list[float] = []
    path_evaluations = 0

    def positions_of(rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(horizon.row_index, rows)

    for inner in nested:
        inner_train = train_labeled[np.asarray(inner.train_mask, dtype=np.int64)]
        inner_val = train_labeled[np.asarray(inner.validation_mask, dtype=np.int64)]
        if inner_train.size == 0 or inner_val.size == 0:
            continue
        _plan_fitting_workspace(
            request,
            int(inner_train.size) * matrix.num_features * 8,
            "elastic_alpha_selection",
        )
        design = prepare_indexed_elastic_design(
            x_matrix,
            inner_train,
            target_full,
            codes,
            chunk_rows=ELASTIC_DESIGN_CHUNK_ROWS,
        )
        if design is None:
            continue
        solution = fit_prepared_elastic_path(design, grid.fractions, seed=request.seed)
        if solution is None:
            continue
        path_evaluations += 1
        alpha_maxes.append(solution.alpha_max)
        scores_by_fraction = _predict_rows_by_fraction(x_matrix, inner_val, solution)
        realized_va = horizon.realized[positions_of(inner_val)]
        for fraction in grid.fractions:
            scores = scores_by_fraction[fraction]
            if _score_is_constant(scores):
                constant.add(fraction)
                continue
            valid = np.isfinite(scores) & np.isfinite(realized_va)
            if not valid.any():
                continue
            candidate_ics[fraction].append(
                session_rank_ic_from_arrays(
                    scores, realized_va, codes[inner_val], valid
                )
            )

    bounded = min(path_evaluations, _NESTED_INNER_FOLDS)
    usable = [f for f in grid.fractions if f not in constant and candidate_ics[f]]
    if usable:
        best = _best_fraction(tuple(usable), candidate_ics)
        alpha_max = float(np.mean(alpha_maxes)) if alpha_maxes else 0.0
        if alpha_max <= 0.0:
            return None, None, None, bounded
        return best * alpha_max, best, alpha_max, bounded
    non_constant = [f for f in grid.fractions if f not in constant]
    if non_constant:
        best = max(non_constant)
        _plan_fitting_workspace(
            request,
            int(train_labeled.size) * matrix.num_features * 8,
            "elastic_alpha_selection",
        )
        design = prepare_indexed_elastic_design(
            x_matrix,
            train_labeled,
            target_full,
            codes,
            chunk_rows=ELASTIC_DESIGN_CHUNK_ROWS,
        )
        if design is None:
            return None, None, None, bounded
        solution = fit_prepared_elastic_path(design, grid.fractions, seed=request.seed)
        if solution is None:
            return None, None, None, bounded
        return (
            best * solution.alpha_max,
            best,
            solution.alpha_max,
            min(path_evaluations, _NESTED_INNER_FOLDS) + 1,
        )
    return None, None, None, bounded


def _fit_single_weighted_elastic(
    matrix: PreparedTrainingMatrix,
    target: np.ndarray,
    train_rows: np.ndarray,
    alpha: float,
    *,
    l1_ratio: float,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | None:
    """One weighted ElasticNet at a fixed penalty on an indexed workspace.

    ``target`` is the full NaN-padded vector aligned to canonical rows.
    Returns ``(coefficients, intercept, mean, std)`` or ``None`` when the
    slice has no finite rows. Weights use valid-only session counts so mixed
    invalid rows keep aligned shapes; the only full-size allocation is the
    single Fortran-order standardized design.
    """
    from sklearn.linear_model import ElasticNet

    from src.stocks.ml.models import ELASTIC_DESIGN_CHUNK_ROWS, prepare_indexed_elastic_design

    design = prepare_indexed_elastic_design(
        matrix.X,
        train_rows,
        target,
        matrix.session_code,
        chunk_rows=ELASTIC_DESIGN_CHUNK_ROWS,
    )
    if design is None:
        return None
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=2000,
        random_state=seed,
    )
    model.fit(design.standardized, design.target, sample_weight=design.weights)
    return (
        np.asarray(model.coef_, dtype=np.float64),
        float(model.intercept_),
        design.mean,
        design.std,
    )


def _predict_prepared(
    x_block: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    standardized = (np.asarray(x_block, dtype=np.float64) - mean) / std
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    return np.asarray(standardized @ coefficients + intercept, dtype=np.float64)


def _predict_rows(
    x_matrix: np.ndarray,
    rows: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Chunked standardized prediction over selected rows; no full copy."""
    from src.stocks.ml.models import ELASTIC_DESIGN_CHUNK_ROWS

    out = np.empty(rows.size, dtype=np.float64)
    chunk = max(1, int(ELASTIC_DESIGN_CHUNK_ROWS))
    for start in range(0, rows.size, chunk):
        block = x_matrix[rows[start : start + chunk]]
        stop = start + block.shape[0]
        out[start:stop] = _predict_prepared(
            block, coefficients, intercept, mean, std
        )
    return out


def _predict_rows_by_fraction(
    x_matrix: np.ndarray,
    rows: np.ndarray,
    solution: ElasticPathResult,
) -> dict[float, np.ndarray]:
    """Chunked target-free scores for every penalty fraction over ``rows``."""
    from src.stocks.ml.models import ELASTIC_DESIGN_CHUNK_ROWS

    scores = {
        fraction: np.empty(rows.size, dtype=np.float64)
        for fraction in solution.fractions
    }
    chunk = max(1, int(ELASTIC_DESIGN_CHUNK_ROWS))
    for start in range(0, rows.size, chunk):
        block = x_matrix[rows[start : start + chunk]]
        stop = start + block.shape[0]
        for index, fraction in enumerate(solution.fractions):
            scores[fraction][start:stop] = _predict_prepared(
                block,
                solution.coefficients[index],
                float(solution.intercepts[index]),
                solution.mean,
                solution.std,
            )
    return scores


def _datetime_us_series(ns_values: np.ndarray, name: str) -> pl.Series:
    """Timezone-aware microsecond datetime series from int64 epoch nanos."""
    raw = np.asarray(ns_values, dtype="datetime64[ns]")
    series = pl.Series(name, raw).dt.replace_time_zone("UTC")
    return series.dt.cast_time_unit("us")


def fit_horizon_oof(
    matrix: PreparedTrainingMatrix,
    horizon: PreparedHorizonLabels,
    folds: tuple[PreparedFold, ...],
    request: NetAlphaTrainingRequest | OofFitRequest,
) -> OofFitResult:
    """Fit the prepared-array fold loop for one horizon.

    Each fold selects its scale-invariant penalty through nested purged folds
    computed purely on arrays, fits once on its labeled training rows, predicts
    the full validation block target-free, and joins decimal realized outcomes
    only after prediction. Polars OOF frames are constructed exactly once here
    at the calibration/replay boundary; expected invalid inputs are classified
    in the diagnostic instead of being swallowed.
    """
    from src.stocks.ml.contracts import (
        FoldScoreDiagnostic,
        HorizonOOFDiagnostic,
        RegularizationGrid,
    )
    from src.stocks.ml.labels import (
        AVAILABLE_COLUMN,
        REALIZED_RETURN_COLUMN,
        REFERENCE_COST_COLUMN,
        RISK_RESIDUAL_COLUMN,
        TARGET_COLUMN,
    )

    context = (
        request if isinstance(request, OofFitRequest) else OofFitRequest(request=request)
    )
    if context.family != "net_alpha_elastic_net":
        raise ValueError(
            "prepared-array OOF fitting owns the elastic baseline family only; "
            f"got {context.family!r}"
        )
    inner_request = context.request
    grid = RegularizationGrid()
    oof_frames: list[pl.DataFrame] = []
    label_frames: list[pl.DataFrame] = []
    rank_ics: list[float] = []
    fold_diagnostics: list[FoldScoreDiagnostic] = []
    path_evaluations = 0
    x_matrix = matrix.X
    codes = matrix.session_code
    target_full = _matrix_aligned_target(horizon, matrix.num_rows)

    for pfold in folds:
        fold_index = pfold.fold_index
        left = np.searchsorted(pfold.train_rows, horizon.row_index)
        left = np.clip(left, 0, max(0, pfold.train_rows.size - 1))
        hit = pfold.train_rows[left] == horizon.row_index
        train_labeled = horizon.row_index[hit]
        validation_rows = pfold.validation_rows
        if train_labeled.size == 0 or validation_rows.size == 0:
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index, failure_reason="empty-fold"
                )
            )
            continue

        selected_alpha, selected_fraction, alpha_max, fold_path_count = (
            _select_elastic_alpha_prepared(
                matrix, horizon, train_labeled, inner_request, grid
            )
        )
        path_evaluations += fold_path_count
        if selected_alpha is None:
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index, failure_reason="constant-oof-score"
                )
            )
            continue
        _plan_fitting_workspace(
            inner_request,
            int(train_labeled.size) * matrix.num_features * 8,
            "outer_fold_fit",
        )
        fitted = _fit_single_weighted_elastic(
            matrix,
            target_full,
            train_labeled,
            selected_alpha,
            l1_ratio=0.5,
            seed=inner_request.seed,
        )
        if fitted is None:
            fold_diagnostics.append(
                FoldScoreDiagnostic(
                    fold_index=fold_index, failure_reason="empty-fold"
                )
            )
            continue
        coefficients, intercept, mean, std = fitted
        scores = _predict_rows(
            x_matrix, validation_rows, coefficients, intercept, mean, std
        )
        finite_scores = scores[np.isfinite(scores)]
        score_std = float(np.std(finite_scores)) if finite_scores.size else 0.0
        unique_count = int(np.unique(finite_scores).size) if finite_scores.size else 0

        left_val = np.searchsorted(horizon.row_index, validation_rows)
        left_val = np.clip(left_val, 0, max(0, horizon.row_index.size - 1))
        val_hit = horizon.row_index[left_val] == validation_rows
        val_labeled_rows = validation_rows[val_hit]
        val_positions = left_val[val_hit]
        realized_va = horizon.realized[val_positions]
        scores_va = scores[val_hit]
        valid_ic = np.isfinite(scores_va) & np.isfinite(realized_va)
        rank_ic = (
            session_rank_ic_from_arrays(
                scores_va, realized_va, codes[val_labeled_rows], valid_ic
            )
            if valid_ic.any()
            else 0.0
        )
        if val_labeled_rows.size == 0:
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

        segment_series = pl.Series(
            _OOF_SEGMENT, [pfold.segment_id] * validation_rows.size, dtype=pl.Int64
        )
        oof_frames.append(
            pl.DataFrame(
                {
                    _ID_COLUMN: pl.Series(
                        matrix.instrument_ids_at(validation_rows).tolist(),
                        dtype=pl.String,
                    ),
                    _SESSION_COLUMN: _datetime_us_series(
                        matrix.session_timestamps_ns[
                            codes[validation_rows].astype(np.int64)
                        ],
                        _SESSION_COLUMN,
                    ),
                    _SESSION_IDX: pl.Series(codes[validation_rows].astype(np.int64)),
                    _SCORE_COLUMN: pl.Series(scores.astype(np.float64)),
                    segment_series.name: segment_series,
                }
            )
        )
        label_frames.append(
            pl.DataFrame(
                {
                    _ID_COLUMN: pl.Series(
                        matrix.instrument_ids_at(val_labeled_rows).tolist(),
                        dtype=pl.String,
                    ),
                    _SESSION_COLUMN: _datetime_us_series(
                        matrix.session_timestamps_ns[
                            codes[val_labeled_rows].astype(np.int64)
                        ],
                        _SESSION_COLUMN,
                    ),
                    _SESSION_IDX: pl.Series(codes[val_labeled_rows].astype(np.int64)),
                    _SCORE_COLUMN: pl.Series(scores_va.astype(np.float64)),
                    _OOF_SEGMENT: pl.Series(
                        [pfold.segment_id] * val_labeled_rows.size, dtype=pl.Int64
                    ),
                    TARGET_COLUMN: pl.Series(
                        horizon.target[val_positions].astype(np.float64)
                    ),
                    AVAILABLE_COLUMN: _datetime_us_series(
                        horizon.available_time_ns[val_positions], AVAILABLE_COLUMN
                    ),
                    RISK_RESIDUAL_COLUMN: pl.Series(
                        horizon.risk_residual[val_positions].astype(np.float64)
                    ),
                    REFERENCE_COST_COLUMN: pl.Series(
                        horizon.reference_cost[val_positions].astype(np.float64)
                    ),
                    REALIZED_RETURN_COLUMN: pl.Series(realized_va.astype(np.float64)),
                }
            )
        )
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
        horizon_sessions=int(horizon.horizon_sessions),
        model_family=context.family,
        fold_diagnostics=tuple(fold_diagnostics),
    )
    if not oof_frames:
        empty = pl.DataFrame()
        return OofFitResult(empty, empty, [], diagnostic, path_evaluations)
    return OofFitResult(
        pl.concat(oof_frames),
        pl.concat(label_frames),
        rank_ics,
        diagnostic,
        path_evaluations,
    )
