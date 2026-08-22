"""Deterministic continuous net-alpha models and out-of-fold calibration.

The learner stack is an explicit baseline/challenger pair:

- ``ElasticNetNetAlpha`` fits a deterministic ``ElasticNet`` on fold-local
  standardized rank features — the stable linear benchmark.
- ``LightGbmNetAlpha`` is the L1-regression challenger with pinned seed and
  deterministic thread/leaf settings.
- ``NetAlphaCalibrator`` fits a monotone calibration on OOF predictions only,
  converting a score into an expected net alpha and a decimal lower bound,
  without ever reading an in-fold target. Calibration targets are decimal
  realized returns (``risk_residual - reference_cost``), never the
  MAD-standardized ``net_alpha_target``.
- ``CalibratedNetAlphaModel`` wraps a fitted learner plus the fitted
  calibration so the persisted artifact applies the decimal lower bound to
  every production prediction.

A challenger is accepted only when its paired OOF incremental policy-utility
block-bootstrap lower bound over the baseline is strictly positive; otherwise
the baseline is retained or both are ``NO_TRADE``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet

from src.stocks.research.economic_alpha import CausalAlphaCalibrator
from src.stocks.research.models import Model, ModelManifest

TARGET_PREFIXES = ("target_", "label_")
SCORE_COLUMN = "predicted_net_alpha"
MIN_BUCKET_OBSERVATIONS = 5


@dataclass(frozen=True, slots=True)
class NetAlphaModelConfig:
    """Deterministic training configuration shared by baseline and challenger.

    ``elastic_alpha_fraction`` and ``elastic_alpha_max`` record the fold-local
    scale-invariant penalty selection used by the net-alpha trainer; they are
    optional metadata so an explicit absolute ``elastic_alpha`` remains valid
    for tests and backward-compatible deserialization.
    """

    seed: int = 42
    num_leaves: int = 31
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_samples: int = 200
    feature_fraction: float = 0.8
    n_estimators: int = 500
    early_stopping_rounds: int = 50
    elastic_alpha: float = 0.5
    elastic_l1_ratio: float = 0.5
    elastic_alpha_fraction: float | None = None
    elastic_alpha_max: float | None = None

    def __post_init__(self) -> None:
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")
        if not 0.0 < self.feature_fraction <= 1.0:
            raise ValueError("feature_fraction must be in (0, 1]")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.elastic_alpha < 0:
            raise ValueError("elastic_alpha must be non-negative")
        if not 0.0 <= self.elastic_l1_ratio <= 1.0:
            raise ValueError("elastic_l1_ratio must be in [0, 1]")
        if self.elastic_alpha_fraction is not None and (
            not np.isfinite(self.elastic_alpha_fraction) or self.elastic_alpha_fraction <= 0.0
        ):
            raise ValueError("elastic_alpha_fraction must be finite and positive when supplied")
        if self.elastic_alpha_max is not None and (
            not np.isfinite(self.elastic_alpha_max) or self.elastic_alpha_max <= 0.0
        ):
            raise ValueError("elastic_alpha_max must be finite and positive when supplied")


def _float32_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Contiguous Float32 design matrix from declared columns."""
    cast_exprs = [
        pl.col(column).cast(pl.Float32)
        if frame.schema[column] != pl.Float32
        else pl.col(column)
        for column in columns
    ]
    array = frame.select(cast_exprs).to_numpy()
    if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    return array


def _reject_target_columns(frame: pl.DataFrame, label_column: str) -> None:
    offending = [
        c
        for c in frame.columns
        if c.startswith(TARGET_PREFIXES) or c == label_column
    ]
    if offending:
        raise ValueError(f"predict rejects target/label columns: {offending}")


def session_balanced_weights(
    frame: pl.DataFrame, *, session_column: str = "session"
) -> np.ndarray:
    """Per-row sample weights giving every training session equal total weight.

    Each row in session ``s`` receives weight ``1 / n_s`` (the reciprocal of its
    session's finite row count), so a session with more instruments never
    dominates the training objective. Rows are matched back to ``frame`` by row
    order. The raw weights are returned; callers normalize them so the total
    equals the number of finite training rows.
    """
    if session_column not in frame.columns:
        raise ValueError(f"sample-weight frame missing session column {session_column!r}")
    counts = (
        frame.group_by(session_column)
        .agg(pl.len().alias("__session_n"))
        .sort(session_column)
    )
    count_map = {
        row[session_column]: int(row["__session_n"])
        for row in counts.iter_rows(named=True)
    }
    weights = np.asarray(
        [1.0 / count_map[value] for value in frame[session_column].to_list()],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
        raise ValueError("session-balanced weights must be finite and positive")
    return weights


def normalize_session_weights(
    weights: np.ndarray, *, total: int | None = None
) -> np.ndarray:
    """Scale ``weights`` so they sum to ``total`` (default: their row count)."""
    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)) or float(np.sum(arr)) <= 0.0:
        raise ValueError("malformed sample weights cannot be normalized")
    target = float(total if total is not None else arr.size)
    return arr * (target / float(np.sum(arr)))


def session_balanced_weights_from_codes(
    session_codes: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Session-balanced normalized weights over finite rows only.

    Counts are taken over the *valid* rows of each session only, so a session
    whose rows carry invalid feature/target values still receives an equal raw
    total among its surviving rows. The valid mask is applied exactly once:
    the returned full-length array carries zero weight on invalid rows and
    sums to one across the selected rows (within float rounding), so fold
    statistics indexed by the same mask see matching lengths and totals.
    """
    codes = np.asarray(session_codes)
    valid_mask = np.asarray(valid, dtype=bool)
    if codes.ndim != 1 or valid_mask.shape != codes.shape:
        raise ValueError("session_codes and valid must be aligned 1-D arrays")
    if codes.size == 0:
        return np.zeros(0, dtype=np.float64)
    selected = np.asarray(codes[valid_mask], dtype=np.int64)
    counts = np.bincount(selected) if selected.size else np.zeros(0, dtype=np.int64)
    per_row_counts = (
        counts[selected].astype(np.float64) if selected.size else np.zeros(0)
    )
    raw = np.zeros(codes.size, dtype=np.float64)
    if selected.size:
        positive = per_row_counts > 0
        raw_valid = np.where(positive, 1.0 / np.where(positive, per_row_counts, 1.0), 0.0)
        total = float(raw_valid.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("malformed sample weights cannot be normalized")
        raw[valid_mask] = raw_valid / total
    return raw


def weighted_fold_statistics(
    features: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted fold-local mean and variance on the finite-valid design rows.

    The weights are pre-normalized (sum equals the finite row count), so the
    weighted mean/std reproduce the ordinary fold statistics when every session
    has equal size. Zero-variance columns are set to unit standard deviation so
    standardization is always well defined.
    """
    sub = features[valid]
    w = weights[valid]
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("fold has no positive-weight rows")
    mean = np.sum(sub * w[:, None], axis=0) / total
    centered = sub - mean
    variance = np.sum(w[:, None] * centered * centered, axis=0) / total
    return mean, _finalize_weighted_std(variance, 1.0, mean)


def _finalize_weighted_std(
    variance: np.ndarray, total: float, mean: np.ndarray
) -> np.ndarray:
    """Unit std for zero-variance columns across every design route.

    Exact float equality cannot detect a constant column once the weighted
    mean carries rounding noise, so variance below a scale-relative epsilon
    is treated as zero too; every route (frame oracle, extracted arrays,
    indexed workspace) shares this one rule so parity stays deterministic.
    """
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64) / total, 0.0))
    scale = np.maximum(np.abs(mean), 1.0)
    std[std <= scale * 1e-12] = 1.0
    return std


@dataclass(frozen=True, slots=True)
class ElasticPathResult:
    """Frozen weighted ElasticNet coefficient path over one penalty grid.

    ``fractions`` is ascending and ``coefficients[:, i]``/``intercepts[i]``
    solve penalty ``fractions[i] * alpha_max`` in one descending warm-start
    coordinate path. ``mean``/``std`` are the weighted fold-local
    standardization frozen from the training slice, so :meth:`predict` is
    target-free.
    """

    fractions: tuple[float, ...]
    coefficients: np.ndarray
    intercepts: np.ndarray
    alpha_max: float
    mean: np.ndarray
    std: np.ndarray

    def predict_array(self, features: np.ndarray) -> dict[float, np.ndarray]:
        """Target-free score per penalty fraction on a standardized design."""
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.std
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        return {
            fraction: standardized @ self.coefficients[index] + self.intercepts[index]
            for index, fraction in enumerate(self.fractions)
        }

    def predict(
        self, frame: pl.DataFrame, feature_columns: tuple[str, ...]
    ) -> dict[float, np.ndarray]:
        """Frame-based convenience wrapper over :meth:`predict_array`."""
        return self.predict_array(_float32_matrix(frame, feature_columns))


#: Backward-compatible name for the array-native path result.
ElasticPathSolution = ElasticPathResult


@dataclass(frozen=True, slots=True)
class PreparedElasticDesign:
    """Bounded indexed-fit workspace over selected canonical-matrix rows.

    ``standardized`` is the single full-size allocation: one Fortran-contiguous
    float64 design over the valid (all-finite feature/target) rows only.
    ``target``/``weights`` are aligned to those rows, ``weights`` are the
    session-balanced values summing to one, and ``row_indices`` records which
    canonical rows survived. No float32 indexed copy, centered matrix, or
    second standardized matrix is retained.
    """

    standardized: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    row_indices: np.ndarray


def fit_weighted_elastic_path(
    features: np.ndarray,
    target: np.ndarray,
    session_codes: np.ndarray,
    alpha_fractions: tuple[float, ...],
    *,
    seed: int = 42,
    l1_ratio: float = 0.5,
) -> ElasticPathResult | None:
    """One deterministic weighted coordinate path over every penalty fraction.

    Builds the weighted standardized design once on the finite rows only,
    derives ``alpha_max`` once (``max(abs(X.T @ (w * y_centered))) / sum(w)``
    on the weighted fold-standardized design), and solves all penalty
    fractions in one descending warm-start coordinate descent. Weights come
    from :func:`session_balanced_weights_from_codes`, so mixed non-finite rows
    keep equal selected row/weight lengths. Returns ``None`` when the slice
    has no finite feature/target rows or ``alpha_max`` is degenerate.
    """
    if not alpha_fractions:
        raise ValueError("penalty fractions must be non-empty")
    features_arr = np.asarray(features)
    targets = np.asarray(target, dtype=np.float64)
    codes = np.asarray(session_codes)
    if features_arr.shape[0] != targets.shape[0] or codes.shape[0] != targets.shape[0]:
        raise ValueError("features, target, and session_codes must be row-aligned")
    valid = np.isfinite(features_arr).all(axis=1) & np.isfinite(targets)
    if not valid.any():
        return None
    sub = np.asarray(features_arr[valid], dtype=np.float64)
    y = targets[valid]
    weights_full = session_balanced_weights_from_codes(codes, valid)
    weights = weights_full[valid]
    if not np.all(weights > 0.0):
        raise ValueError("malformed sample weights cannot be used in elastic path")
    total = float(np.sum(weights))
    mean = np.sum(sub * weights[:, None], axis=0) / total
    centered = sub - mean
    variance = np.sum(weights[:, None] * centered * centered, axis=0) / total
    std = _finalize_weighted_std(variance, 1.0, mean)
    standardized = (sub - mean) / std
    return _solve_elastic_path_arrays(
        standardized,
        y,
        weights,
        mean,
        std,
        tuple(alpha_fractions),
        seed=seed,
        l1_ratio=l1_ratio,
    )


def _solve_elastic_path_arrays(
    standardized: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    alpha_fractions: tuple[float, ...],
    *,
    seed: int,
    l1_ratio: float,
) -> ElasticPathResult | None:
    """Shared ``alpha_max`` plus one descending warm-start coordinate path.

    ``standardized`` must be the weighted fold-standardized design over valid
    rows; ``weights`` sum to one within float rounding. Both the extracted-
    array oracle route and the indexed design workspace converge here so the
    penalty selection semantics stay identical.
    """
    if not alpha_fractions:
        raise ValueError("penalty fractions must be non-empty")
    total = float(np.sum(weights))
    y_centered = y - float(np.sum(weights * y) / total)
    alpha_max = float(
        np.max(np.abs(standardized.T @ (weights * y_centered))) / total
    )
    if not np.isfinite(alpha_max) or alpha_max <= 0.0:
        return None

    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    model = ElasticNet(
        alpha=0.0,
        l1_ratio=l1_ratio,
        max_iter=2000,
        random_state=seed,
        warm_start=True,
    )
    for fraction in sorted(alpha_fractions, reverse=True):
        model.set_params(alpha=fraction * alpha_max)
        model.fit(standardized, y, sample_weight=weights)
        coefficients.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(float(model.intercept_))
    ordered = tuple(sorted(alpha_fractions))
    return ElasticPathResult(
        fractions=ordered,
        coefficients=np.asarray(list(reversed(coefficients)), dtype=np.float64),
        intercepts=np.asarray(list(reversed(intercepts)), dtype=np.float64),
        alpha_max=alpha_max,
        mean=mean,
        std=std,
    )


#: Bounded row-chunk size for indexed design scans and fills.
ELASTIC_DESIGN_CHUNK_ROWS = 65536


def prepare_indexed_elastic_design(
    features: np.ndarray,
    row_indices: np.ndarray,
    target: np.ndarray,
    session_codes: np.ndarray,
    *,
    chunk_rows: int,
) -> PreparedElasticDesign | None:
    """Build the single-allocation indexed fit workspace over selected rows.

    ``features``, ``target``, and ``session_codes`` are full canonical-matrix
    arrays; ``row_indices`` selects the candidate rows. Bounded chunks scan
    for validity (all-finite feature vector and target), per-session valid-row
    counts, weighted mean, and centered weighted variance before exactly one
    Fortran-order float64 standardized design is allocated and filled
    chunkwise — so no full-size float32 indexed copy, centered matrix, or
    second standardized matrix ever exists. Weights keep every valid session's
    equal share (``w_i = 1 / (n_valid_sessions * valid_rows_in_session_i)``,
    normalized). Returns ``None`` when no selected row is valid.
    """
    chunk = max(1, int(chunk_rows))
    features_arr = np.asarray(features)
    rows = np.asarray(row_indices, dtype=np.int64)
    targets = np.asarray(target, dtype=np.float64)
    codes = np.asarray(session_codes)
    n_all = features_arr.shape[0]
    if targets.shape[0] != n_all or codes.shape[0] != n_all:
        raise ValueError("features, target, and session_codes must be row-aligned")
    if rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if rows.size == 0:
        return None
    n_features = features_arr.shape[1]

    # Pass 1: bounded-chunk finite-mask discovery plus per-session counts.
    valid = np.zeros(rows.size, dtype=bool)
    max_code = int(codes.max()) + 1
    counts = np.zeros(max_code, dtype=np.int64)
    for start in range(0, rows.size, chunk):
        block_rows = rows[start : start + chunk]
        block = np.asarray(features_arr[block_rows], dtype=np.float64)
        block_valid = (
            np.isfinite(block).all(axis=1) & np.isfinite(targets[block_rows])
        )
        valid[start : start + block_rows.size] = block_valid
        if block_valid.any():
            counts += np.bincount(
                codes[block_rows[block_valid]].astype(np.int64),
                minlength=max_code,
            )
    if not valid.any():
        return None
    valid_rows = rows[valid]
    selected_codes = codes[valid_rows].astype(np.int64)
    raw = 1.0 / counts[selected_codes]
    weight_total = float(raw.sum())
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("malformed sample weights cannot be normalized")
    weights_selected = raw / weight_total

    # Pass 2: bounded-chunk weighted mean on valid rows only.
    mean = np.zeros(n_features, dtype=np.float64)
    for start in range(0, valid_rows.size, chunk):
        block_rows = valid_rows[start : start + chunk]
        block = np.asarray(features_arr[block_rows], dtype=np.float64)
        w = weights_selected[start : start + block_rows.size]
        mean += np.sum(block * w[:, None], axis=0)

    # Pass 3: bounded-chunk centered weighted variance.
    variance = np.zeros(n_features, dtype=np.float64)
    for start in range(0, valid_rows.size, chunk):
        block_rows = valid_rows[start : start + chunk]
        block = np.asarray(features_arr[block_rows], dtype=np.float64)
        centered = block - mean
        w = weights_selected[start : start + block_rows.size]
        variance += np.sum(w[:, None] * centered * centered, axis=0)

    # Pass 4: allocate once and fill chunkwise with the frozen standardization.
    # Variance was accumulated with the already-normalized weights, so no
    # second division by weight_total happens here.
    std = _finalize_weighted_std(variance, 1.0, mean)
    standardized = np.empty(
        (int(valid_rows.size), n_features), dtype=np.float64, order="F"
    )
    for start in range(0, valid_rows.size, chunk):
        block_rows = valid_rows[start : start + chunk]
        block = np.asarray(features_arr[block_rows], dtype=np.float64)
        standardized[start : start + block_rows.size] = (block - mean) / std

    return PreparedElasticDesign(
        standardized=standardized,
        target=np.ascontiguousarray(targets[valid_rows]),
        weights=weights_selected,
        mean=mean,
        std=std,
        row_indices=np.asarray(valid_rows, dtype=np.int64),
    )


def fit_prepared_elastic_path(
    design: PreparedElasticDesign,
    alpha_fractions: tuple[float, ...],
    *,
    seed: int = 42,
    l1_ratio: float = 0.5,
) -> ElasticPathResult | None:
    """Solve the deterministic weighted penalty path on an indexed workspace.

    Shares the exact ``alpha_max``/coordinate-descent semantics of
    :func:`fit_weighted_elastic_path`, but consumes the already-standardized
    Fortran-order design so no second full-size matrix is materialized.
    """
    return _solve_elastic_path_arrays(
        design.standardized,
        design.target,
        design.weights,
        design.mean,
        design.std,
        tuple(alpha_fractions),
        seed=seed,
        l1_ratio=l1_ratio,
    )


def _fit_weighted_elastic_path_reference(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...],
    fractions: tuple[float, ...],
    seed: int,
    *,
    l1_ratio: float = 0.5,
) -> ElasticPathResult | None:
    """Legacy frame-based reference path kept as the parity oracle.

    Mirrors the historical implementation exactly — full-slice session counts,
    normalized-then-masked weights, and Polars feature extraction — so parity
    tests can prove the prepared-array path reproduces it within tolerance.
    """
    if not fractions:
        raise ValueError("penalty fractions must be non-empty")
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns {missing} in path fit")
    features = _float32_matrix(frame, feature_columns)
    targets = frame["net_alpha_target"].cast(pl.Float64).to_numpy()
    valid = np.isfinite(features).all(axis=1) & np.isfinite(targets)
    if not valid.any():
        return None
    raw_weights = session_balanced_weights(frame)
    weights = normalize_session_weights(raw_weights, total=int(valid.sum()))
    mean, std = weighted_fold_statistics(features, weights, valid)
    sub = features[valid]
    y = targets[valid]
    w = weights[valid]
    standardized = (sub - mean) / std
    y_centered = y - float(np.sum(w * y) / float(np.sum(w)))
    alpha_max = float(
        np.max(np.abs(standardized.T @ (w * y_centered)))
        / float(np.sum(w))
    )
    if not np.isfinite(alpha_max) or alpha_max <= 0.0:
        return None

    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    model = ElasticNet(
        alpha=0.0,
        l1_ratio=l1_ratio,
        max_iter=2000,
        random_state=seed,
        warm_start=True,
    )
    for fraction in sorted(fractions, reverse=True):
        model.set_params(alpha=fraction * alpha_max)
        model.fit(standardized, y, sample_weight=w)
        coefficients.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(float(model.intercept_))
    ordered = tuple(sorted(fractions))
    return ElasticPathResult(
        fractions=ordered,
        coefficients=np.asarray(list(reversed(coefficients)), dtype=np.float64),
        intercepts=np.asarray(list(reversed(intercepts)), dtype=np.float64),
        alpha_max=alpha_max,
        mean=mean,
        std=std,
    )


class ElasticNetNetAlpha:
    """Deterministic ElasticNet baseline on fold-standardized rank features."""

    def __init__(
        self,
        manifest: ModelManifest,
        feature_columns: tuple[str, ...],
        label_column: str,
        config: NetAlphaModelConfig | None = None,
    ):
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not label_column:
            raise ValueError("label_column must not be empty")
        self._manifest = manifest
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._config = config or NetAlphaModelConfig()
        self._model: ElasticNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        del validation
        if self._label_column not in train.columns:
            raise ValueError(
                f"missing label column {self._label_column!r} in training fold"
            )
        missing = [c for c in self._feature_columns if c not in train.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in training fold")
        features = _float32_matrix(train, self._feature_columns)
        targets = train[self._label_column].cast(pl.Float64).to_numpy()
        finite = np.isfinite(targets)
        valid = np.isfinite(features).all(axis=1) & finite
        if not valid.any():
            raise ValueError("training fold has no finite feature/target rows")
        raw_weights = session_balanced_weights(train)
        weights = normalize_session_weights(raw_weights, total=int(valid.sum()))
        if not np.all(weights[valid] > 0.0):
            raise ValueError("malformed sample weights cannot be used in ElasticNet")
        self._mean, self._std = weighted_fold_statistics(features, weights, valid)
        standardized = (features - self._mean) / self._std
        self._model = ElasticNet(
            alpha=self._config.elastic_alpha,
            l1_ratio=self._config.elastic_l1_ratio,
            max_iter=2000,
            random_state=self._config.seed,
        )
        self._model.fit(standardized[valid], targets[valid], sample_weight=weights[valid])

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._model is None:
            raise ValueError("model is not fitted")
        _reject_target_columns(frame, self._label_column)
        missing = [c for c in self._feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in predict frame")
        features = _float32_matrix(frame, self._feature_columns)
        if self._mean is None or self._std is None:
            raise ValueError("model has no frozen fold statistics")
        standardized = (features - self._mean) / self._std
        standardized[~np.isfinite(standardized)] = 0.0
        scores = self._model.predict(standardized)
        return frame.with_columns(
            pl.Series(SCORE_COLUMN, scores.astype(np.float64))
        )

    def manifest(self) -> ModelManifest:
        params: dict[str, str] = {
            "family": "elastic_net",
            "alpha": f"{self._config.elastic_alpha:.6g}",
            "l1_ratio": f"{self._config.elastic_l1_ratio:.6g}",
            "seed": str(self._config.seed),
            "feature_columns": ",".join(self._feature_columns),
        }
        if self._config.elastic_alpha_fraction is not None:
            params["alpha_fraction"] = f"{self._config.elastic_alpha_fraction:.6g}"
        if self._config.elastic_alpha_max is not None:
            params["alpha_max"] = f"{self._config.elastic_alpha_max:.6g}"
        return ModelManifest(
            artifact_id=self._manifest.artifact_id,
            asset_kind=self._manifest.asset_kind,
            feature_set=self._manifest.feature_set,
            feature_schema_hash=self._manifest.feature_schema_hash,
            universe_policy_hash=self._manifest.universe_policy_hash,
            label_definition=self._manifest.label_definition,
            label_horizon_sessions=self._manifest.label_horizon_sessions,
            eligible_from=self._manifest.eligible_from,
            eligible_to=self._manifest.eligible_to,
            model_type="net_alpha_elastic_net",
            params=params,
        )


class LightGbmNetAlpha:
    """Deterministic L1 LightGBM regressor challenger on rank features."""

    def __init__(
        self,
        manifest: ModelManifest,
        feature_columns: tuple[str, ...],
        label_column: str,
        config: NetAlphaModelConfig | None = None,
        num_threads: int = 1,
    ):
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not label_column:
            raise ValueError("label_column must not be empty")
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        self._manifest = manifest
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._config = config or NetAlphaModelConfig()
        self._num_threads = num_threads
        self._booster: lgb.Booster | None = None
        self._best_iteration: int | None = None

    def _params(self) -> dict[str, object]:
        config = self._config
        return {
            "objective": "regression_l1",
            "metric": "mae",
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
            "num_threads": self._num_threads,
            "seed": config.seed,
            "deterministic": True,
            "force_col_wise": True,
            "data_random_seed": config.seed,
            "feature_fraction_seed": config.seed,
            "bagging_seed": config.seed,
            "verbosity": -1,
        }

    def fit(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        *,
        num_boost_round: int | None = None,
    ) -> None:
        if self._label_column not in train.columns:
            raise ValueError(
                f"missing label column {self._label_column!r} in training fold"
            )
        missing = [c for c in self._feature_columns if c not in train.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in training fold")
        train_features = _float32_matrix(train, self._feature_columns)
        train_targets = train[self._label_column].cast(pl.Float64).to_numpy()
        finite = np.isfinite(train_targets)
        if not finite.any():
            raise ValueError("training fold has no finite target rows")
        train_features = train_features[finite]
        train_targets = train_targets[finite]
        raw_weights = session_balanced_weights(train)
        train_weights = normalize_session_weights(
            raw_weights, total=int(finite.sum())
        )[finite]
        if not np.all(train_weights > 0.0):
            raise ValueError("malformed sample weights cannot be used in LightGBM")
        train_set = lgb.Dataset(
            train_features,
            label=train_targets,
            weight=train_weights,
            params={"verbosity": -1},
        )
        valid_set = None
        if (
            not validation.is_empty()
            and self._label_column in validation.columns
            and all(c in validation.columns for c in self._feature_columns)
        ):
            valid_features = _float32_matrix(validation, self._feature_columns)
            valid_targets = validation[self._label_column].cast(pl.Float64).to_numpy()
            vfinite = np.isfinite(valid_targets)
            if vfinite.any():
                valid_weights = normalize_session_weights(
                    session_balanced_weights(validation),
                    total=int(vfinite.sum()),
                )[vfinite]
                valid_set = lgb.Dataset(
                    valid_features[vfinite],
                    label=valid_targets[vfinite],
                    weight=valid_weights,
                    reference=train_set,
                    params={"verbosity": -1},
                )
        # Early stopping needs a finite labeled validation Dataset.  The
        # target-free outer validation frame and the empty final-refit frame
        # train the deterministic fixed estimator budget instead and never
        # request early stopping (installing it without a valid set raises).
        callbacks: list[Any] = []
        if valid_set is not None and num_boost_round is None:
            callbacks = [
                lgb.early_stopping(
                    self._config.early_stopping_rounds, verbose=False, min_delta=0.0
                )
            ]
        num_round = num_boost_round or self._config.n_estimators
        self._booster = lgb.train(
            self._params(),
            train_set,
            num_boost_round=num_round,
            valid_sets=[valid_set] if valid_set is not None else None,
            callbacks=callbacks,
        )
        if valid_set is not None and self._booster.best_iteration > 0:
            self._best_iteration = self._booster.best_iteration
        else:
            self._best_iteration = None

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._booster is None:
            raise ValueError("model is not fitted")
        _reject_target_columns(frame, self._label_column)
        missing = [c for c in self._feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in predict frame")
        features = _float32_matrix(frame, self._feature_columns)
        if not features.flags.writeable:
            features = features.copy()
        features[~np.isfinite(features)] = 0.0
        scores = np.asarray(
            self._booster.predict(features, num_iteration=self._best_iteration),
            dtype=np.float64,
        )
        return frame.with_columns(pl.Series(SCORE_COLUMN, scores))

    @property
    def best_iteration(self) -> int | None:
        """Deterministic iteration count selected by inner labeled validation."""
        return self._best_iteration

    def manifest(self) -> ModelManifest:
        params: dict[str, str] = {
            "family": "lightgbm_l1",
            "objective": "regression_l1",
            "num_leaves": str(self._config.num_leaves),
            "learning_rate": f"{self._config.learning_rate:.6g}",
            "min_child_samples": str(self._config.min_child_samples),
            "seed": str(self._config.seed),
            "num_threads": str(self._num_threads),
            "feature_columns": ",".join(self._feature_columns),
        }
        return ModelManifest(
            artifact_id=self._manifest.artifact_id,
            asset_kind=self._manifest.asset_kind,
            feature_set=self._manifest.feature_set,
            feature_schema_hash=self._manifest.feature_schema_hash,
            universe_policy_hash=self._manifest.universe_policy_hash,
            label_definition=self._manifest.label_definition,
            label_horizon_sessions=self._manifest.label_horizon_sessions,
            eligible_from=self._manifest.eligible_from,
            eligible_to=self._manifest.eligible_to,
            model_type="net_alpha_lightgbm_l1",
            params=params,
        )


@dataclass(frozen=True, slots=True)
class BucketEvidence:
    """One calibration bucket's OOF evidence."""

    bucket: int
    sample_size: int
    expected_alpha: float
    alpha_lower_bound: float


@dataclass(frozen=True, slots=True)
class CalibrationState:
    """Frozen calibration table for prediction-time application."""

    buckets: tuple[BucketEvidence, ...]
    bucket_count: int
    boundaries: tuple[float, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "bucket_count": self.bucket_count,
            "boundaries": list(self.boundaries),
            "buckets": [
                {
                    "bucket": b.bucket,
                    "sample_size": b.sample_size,
                    "expected_alpha": b.expected_alpha,
                    "alpha_lower_bound": b.alpha_lower_bound,
                }
                for b in self.buckets
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> CalibrationState:
        """Rehydrate a frozen calibration table from its JSON serialization."""
        raw_buckets = payload.get("buckets") or []
        if not isinstance(raw_buckets, list):
            raise ValueError("calibration state 'buckets' must be a list")
        buckets = tuple(
            BucketEvidence(
                bucket=int(item["bucket"]),
                sample_size=int(item["sample_size"]),
                expected_alpha=float(item["expected_alpha"]),
                alpha_lower_bound=float(item["alpha_lower_bound"]),
            )
            for item in raw_buckets
        )
        raw_boundaries = payload.get("boundaries") or ()
        boundaries = tuple(
            float(value)
            for value in (raw_boundaries if isinstance(raw_boundaries, list) else ())
        )
        raw_count = payload.get("bucket_count", 0)
        bucket_count = int(raw_count) if isinstance(raw_count, (int, float)) else 0
        return cls(
            buckets=buckets,
            bucket_count=bucket_count,
            boundaries=boundaries,
        )


class NetAlphaCalibrator:
    """Monotone OOF-only calibration of scores to expected net alpha.

    Bucket boundaries and statistics are fit on out-of-fold predictions only;
    the in-fold targets are never read at calibration time. Each bucket is
    usable only when it holds at least ``MIN_BUCKET_OBSERVATIONS`` observations
    and its seeded block-bootstrap lower bound is strictly positive; otherwise
    the bucket is treated as a non-trade signal.

    The calibration label is a decimal realized return (``risk_residual -
    reference_cost``), never the MAD-standardized ``net_alpha_target``, so the
    ``net_alpha_lower_bound`` it emits is an economic return in compatible
    units with replay transaction costs.
    """

    def __init__(
        self,
        bucket_count: int,
        seed: int = 42,
        n_bootstrap: int = 200,
        bootstrap_alpha: float = 0.05,
        block_length: int = 5,
        label_column: str = "realized_net_return",
    ):
        if bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")
        self._bucket_count = bucket_count
        self._seed = seed
        self._n_bootstrap = n_bootstrap
        self._bootstrap_alpha = bootstrap_alpha
        self._block_length = block_length
        self._label_column = label_column
        self._state: CalibrationState | None = None

    @property
    def state(self) -> CalibrationState:
        """The frozen calibration table; raises until ``fit`` has run."""
        if self._state is None:
            raise ValueError("calibrator has no frozen state; call fit first")
        return self._state

    def state_json(self) -> dict[str, object]:
        """JSON serialization of the frozen calibration table."""
        return self.state.to_json()

    def load_state(self, payload: dict[str, object]) -> NetAlphaCalibrator:
        """Restore a frozen calibration table from its JSON serialization."""
        self._state = CalibrationState.from_json(payload)
        return self

    def fit(self, oof: pl.DataFrame) -> CalibrationState:
        """Fit monotone bucket statistics on OOF predictions only."""
        if oof.is_empty():
            raise ValueError("cannot calibrate an empty OOF panel")
        required = (SCORE_COLUMN, self._label_column, "session")
        missing = [c for c in required if c not in oof.columns]
        if missing:
            raise ValueError(f"OOF panel missing columns {missing}")
        scored = oof.filter(
            pl.col(SCORE_COLUMN).is_not_null()
            & pl.col(self._label_column).is_not_null()
            & pl.col(SCORE_COLUMN).is_finite()
            & pl.col(self._label_column).is_finite()
        )
        if scored.is_empty():
            raise ValueError("OOF panel has no finite score/label rows")

        score_values = scored[SCORE_COLUMN].to_numpy().astype(float)
        quantiles = np.linspace(0.0, 1.0, self._bucket_count + 1)[1:-1]
        boundaries = tuple(float(v) for v in np.quantile(score_values, quantiles))
        boundaries = tuple(sorted(set(boundaries)))
        bucket_expr = (
            pl.col(SCORE_COLUMN)
            .cut(boundaries, left_closed=False)
            .to_physical()
            .cast(pl.Int32)
        )
        joined = scored.with_columns(bucket_expr.alias("__bucket"))
        buckets: list[BucketEvidence] = []
        for bucket in range(self._bucket_count):
            rows = joined.filter(pl.col("__bucket") == bucket)
            if rows.height < MIN_BUCKET_OBSERVATIONS:
                continue
            labels = rows[self._label_column].to_numpy().astype(float)
            if not np.all(np.isfinite(labels)):
                continue
            bound = self._lower_bound(labels)
            if bound <= 0.0:
                continue
            buckets.append(
                BucketEvidence(
                    bucket=bucket,
                    sample_size=rows.height,
                    expected_alpha=float(np.mean(labels)),
                    alpha_lower_bound=bound,
                )
            )
        if not buckets:
            state = CalibrationState(
                buckets=(), bucket_count=self._bucket_count, boundaries=boundaries
            )
        else:
            buckets.sort(key=lambda b: b.expected_alpha)
            state = CalibrationState(
                buckets=tuple(buckets),
                bucket_count=self._bucket_count,
                boundaries=boundaries,
            )
        self._state = state
        return state

    def _lower_bound(self, labels: np.ndarray) -> float:
        n = labels.size
        block = min(max(self._block_length, 1), n)
        n_blocks = int(np.ceil(n / block))
        max_start = max(1, n - block + 1)
        rng = np.random.default_rng(self._seed)
        starts = rng.integers(0, max_start, size=(self._n_bootstrap, n_blocks))
        offsets = np.arange(block)
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(
            self._n_bootstrap, n_blocks * block
        )[:, :n]
        means = labels[index].mean(axis=1)
        return float(np.quantile(means, self._bootstrap_alpha))

    def apply(self, scored: pl.DataFrame) -> pl.DataFrame:
        """Apply the frozen calibration to a scored panel."""
        if self._state is None:
            raise ValueError("calibrator has no frozen state; call fit first")
        return _apply_calibration(scored, self._state)


def _apply_calibration(
    scored: pl.DataFrame, state: CalibrationState
) -> pl.DataFrame:
    """Attach expected net alpha and lower bound columns from the bucket table."""
    if not state.buckets:
        return scored.with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias("expected_net_alpha"),
            pl.lit(0.0, dtype=pl.Float64).alias("net_alpha_lower_bound"),
        )
    boundaries = list(state.boundaries) or sorted(
        {b.expected_alpha for b in state.buckets}
    )
    bucket_expr = (
        pl.col(SCORE_COLUMN)
        .cut(boundaries, left_closed=False)
        .to_physical()
        .cast(pl.Int32)
    )
    table = pl.DataFrame(
        {
            "bucket": [b.bucket for b in state.buckets],
            "expected_net_alpha": [b.expected_alpha for b in state.buckets],
            "net_alpha_lower_bound": [b.alpha_lower_bound for b in state.buckets],
        }
    )
    augmented = scored.with_columns(bucket_expr.alias("bucket_idx"))
    merged = augmented.join(
        table, left_on="bucket_idx", right_on="bucket", how="left"
    ).with_columns(
        pl.col("expected_net_alpha").fill_null(0.0),
        pl.col("net_alpha_lower_bound").fill_null(0.0),
    )
    return merged.drop("bucket_idx")


class CalibratedNetAlphaModel:
    """Learner plus fitted calibration persisted as one production artifact.

    ``predict`` delegates to the wrapped learner and then attaches the frozen
    decimal ``net_alpha_lower_bound``, so production planning consumes the same
    economic score the replay gated on. The wrapper is joblib-serialized with
    the calibration table; ``manifest`` also records the serialized calibration
    for auditability. ``calibrator`` is any applier exposing ``apply(scored)``
    and a ``state`` with ``to_json()`` — the legacy ``NetAlphaCalibrator`` or
    the causal session-cluster adapter.
    """

    def __init__(self, model: Model, calibrator: CalibrationApplier):
        if model is None:
            raise ValueError("CalibratedNetAlphaModel requires a base model")
        if calibrator is None:
            raise ValueError("CalibratedNetAlphaModel requires a fitted calibrator")
        self._model = model
        self._calibrator = calibrator

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        self._model.fit(train, validation)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        scored = self._model.predict(frame)
        return self._calibrator.apply(scored)

    def manifest(self) -> ModelManifest:
        base = self._model.manifest()
        params = dict(base.params or {})
        params["calibration_state"] = json.dumps(
            self._calibrator.state.to_json(), sort_keys=True
        )
        return replace(base, params=params)


@runtime_checkable
class CalibrationStateView(Protocol):
    """Serialization view of a frozen calibration state."""

    def to_json(self) -> dict[str, object]: ...


@runtime_checkable
class CalibrationApplier(Protocol):
    """Prediction-time calibration contract: apply a frozen state to a frame."""

    @property
    def state(self) -> CalibrationStateView: ...

    def apply(self, scored: pl.DataFrame) -> pl.DataFrame: ...


@dataclass(frozen=True, slots=True)
class CausalCalibrationState:
    """JSON-safe frozen causal calibration snapshot for artifact serialization."""

    payload: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return dict(self.payload)


class CausalCalibrationAdapter:
    """Applies a frozen causal calibration state to any scored frame.

    ``state`` is the immutable dict produced by
    ``CausalAlphaCalibrator.prepare_decision`` /
    ``SessionClusterCalibrationSchedule.state_at``. ``apply`` preserves
    ``predicted_net_alpha`` and appends decimal ``expected_net_alpha`` and
    ``net_alpha_lower_bound``; a frozen state with no eligible positive bucket
    yields zero/absent economic scores and cash, never an exception or a buy.
    """

    def __init__(self, calibrator: object, state: dict[str, object]):
        self._calibrator = calibrator
        self._state = state

    def apply(self, scored: pl.DataFrame) -> pl.DataFrame:
        if SCORE_COLUMN not in scored.columns:
            raise ValueError(f"apply requires a {SCORE_COLUMN!r} score column")
        prepared = scored.rename({SCORE_COLUMN: "score"})
        augmented = CausalAlphaCalibrator.apply_prepared(self._state, prepared)
        return augmented.drop("__bucket", strict=False).with_columns(
            pl.col("expected_net_alpha").cast(pl.Float64),
            pl.col("net_alpha_lower_bound").cast(pl.Float64),
        ).rename({"score": SCORE_COLUMN})

    @property
    def state(self) -> CausalCalibrationState:
        payload = (
            self._calibrator.calibration_state()
            if isinstance(self._calibrator, CausalAlphaCalibrator)
            else self._state
        )
        return CausalCalibrationState(payload)
