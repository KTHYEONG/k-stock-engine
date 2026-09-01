"""Prepared training representation: one canonical matrix plus narrow labels.

``PreparedTrainingMatrix`` owns a single C-contiguous ``float32`` learner
matrix ``X`` and integer instrument/session codes derived once from the
composed feature frame. Target, realized-outcome, availability, market, and
execution columns never enter ``X``. Per-horizon targets stay independent in
:class:`PreparedHorizonLabels`, aligned to matrix rows through a one-to-one
integer key mapping; duplicate or unmatched keys fail closed.

Fold row lists are converted to immutable integer index arrays at this
boundary without changing :class:`PurgedWalkForward`'s public contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from legacy.stocks.ml.contracts import NetAlphaResearchData, NetAlphaTrainingData
from legacy.stocks.ml.features import FeatureTransformSchema
from legacy.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    GROSS_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
    TARGET_COLUMN,
)
from legacy.stocks.ml.models import _float32_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from legacy.stocks.research.folds import Fold

#: Training-side schema alias for the transformed feature schema.
FeatureSchema = FeatureTransformSchema

__all__ = [
    "FeatureSchema",
    "NetAlphaTrainingData",
    "PreparedFold",
    "PreparedHorizonLabels",
    "PreparedTrainingMatrix",
    "TrainingPanelView",
    "prepare_folds",
    "prepare_horizon_labels",
    "prepare_matrix_from_frame",
    "prepare_training_matrix",
]


@dataclass(frozen=True, slots=True)
class PreparedFold:
    """One fold as immutable integer row-index arrays into the matrix."""

    fold_index: int
    segment_id: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    train_label_end: int
    validation_decision_start: int


def prepare_folds(folds: Sequence[Fold]) -> tuple[PreparedFold, ...]:
    """Convert frame-based fold masks to immutable integer index arrays."""
    prepared: list[PreparedFold] = []
    for index, fold in enumerate(folds):
        train_rows = np.asarray(sorted(int(r) for r in fold.train_mask), dtype=np.int64)
        validation_rows = np.asarray(
            sorted(int(r) for r in fold.validation_mask), dtype=np.int64
        )
        prepared.append(
            PreparedFold(
                fold_index=index,
                segment_id=int(fold.segment_id),
                train_rows=train_rows,
                validation_rows=validation_rows,
                train_label_end=int(fold.train_label_end),
                validation_decision_start=int(fold.validation_decision_start),
            )
        )
    return tuple(prepared)


@dataclass(frozen=True, slots=True)
class PreparedTrainingMatrix:
    """Immutable canonical learner matrix with encoded identities.

    ``X`` is C-contiguous ``float32`` of shape ``(N, F)``; codes are dense
    integers (sessions chronological). ``_sorted_keys``/``_sorted_rows``
    support vectorized one-to-one key alignment for label joins.
    """

    X: np.ndarray
    feature_columns: tuple[str, ...]
    instrument_code: np.ndarray
    session_code: np.ndarray
    session_timestamps_ns: np.ndarray
    instrument_vocabulary: tuple[str, ...]
    sorted_keys: np.ndarray
    sorted_rows: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_sessions(self) -> int:
        return int(self.session_timestamps_ns.size)

    def key_of(self, instrument_codes: np.ndarray, session_codes: np.ndarray) -> np.ndarray:
        """Combined integer key per row: ``instrument * S + session``."""
        return (
            np.asarray(instrument_codes, dtype=np.int64) * self.num_sessions
            + np.asarray(session_codes, dtype=np.int64)
        )

    def rows_for_keys(self, keys: np.ndarray) -> np.ndarray:
        """Resolve combined keys to matrix rows; unmatched become ``-1``."""
        position = np.searchsorted(self.sorted_keys, keys)
        position = np.clip(position, 0, max(0, self.sorted_keys.size - 1))
        matched = self.sorted_keys[position] == keys
        rows = np.where(matched, self.sorted_rows[position], -1)
        return np.asarray(rows, dtype=np.int64)

    def instrument_ids_at(self, rows: np.ndarray) -> np.ndarray:
        vocabulary = np.asarray(self.instrument_vocabulary, dtype=object)
        return np.asarray(
            vocabulary[self.instrument_code[np.asarray(rows, dtype=np.int64)]],
            dtype=object,
        )

    def session_datetimes_at(self, rows: np.ndarray) -> np.ndarray:
        from datetime import datetime, timedelta

        ns = self.session_timestamps_ns[
            self.session_code[np.asarray(rows, dtype=np.int64)]
        ]
        epoch = datetime(1970, 1, 1, tzinfo=None)
        return np.asarray(
            [epoch + timedelta(microseconds=int(value) // 1000) for value in ns],
            dtype=object,
        )


@dataclass(frozen=True, slots=True)
class TrainingPanelView:
    """Structural view exposing only ``feature_frame`` for preparation.

    Discovery prepares the canonical matrix from the transformed pre-holdout
    panel rather than the composed research data, so callers wrap the panel in
    this view instead of materializing a full research-data copy.
    """

    feature_frame: pl.DataFrame


def prepare_training_matrix(
    data: NetAlphaTrainingData | TrainingPanelView,
    schema: FeatureSchema,
    folds: tuple[Fold, ...],
) -> PreparedTrainingMatrix:
    """Build the canonical prepared matrix from composed training data.

    The learner matrix contains exactly the schema's learner columns cast to
    C-contiguous ``float32``; every other column (targets, realized outcomes,
    availability, market/execution fields) is excluded by construction.
    Duplicate ``(instrument_id, session)`` keys fail closed, sessions are
    encoded chronologically, and fold row lists are frozen into integer
    arrays via :func:`prepare_folds` at the caller boundary.
    """
    del folds  # Fold conversion happens through prepare_folds at the caller.
    return prepare_matrix_from_frame(data.feature_frame, tuple(schema.learner_columns))


def prepare_matrix_from_frame(
    feature_frame: pl.DataFrame,
    learner_columns: tuple[str, ...],
) -> PreparedTrainingMatrix:
    """Encode one transformed panel into the canonical prepared matrix."""
    missing = [c for c in learner_columns if c not in feature_frame.columns]
    if missing:
        raise ValueError(f"feature frame missing learner columns {missing}")
    x_matrix = _float32_matrix(feature_frame, learner_columns)  # noqa: N806 - canonical learner matrix name
    if not x_matrix.flags["C_CONTIGUOUS"]:  # pragma: no cover - defensive
        x_matrix = np.ascontiguousarray(x_matrix, dtype=np.float32)

    instrument_values = feature_frame["instrument_id"].to_list()
    vocabulary = tuple(sorted({str(v) for v in instrument_values}))
    vocabulary_index = {name: code for code, name in enumerate(vocabulary)}
    instrument_code = np.asarray(
        [vocabulary_index[str(v)] for v in instrument_values], dtype=np.int32
    )

    session_physical = (
        feature_frame[SESSION_COLUMN].to_physical().to_numpy().astype(np.int64)
    ) * 1_000  # Polars datetime unit → nanoseconds
    unique_sessions = np.unique(session_physical)
    session_code = np.searchsorted(unique_sessions, session_physical).astype(np.int32)

    keys = (
        instrument_code.astype(np.int64) * unique_sessions.size
        + session_code.astype(np.int64)
    )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise ValueError(
            "duplicate (instrument_id, session) keys fail closed in preparation"
        )

    return PreparedTrainingMatrix(
        X=x_matrix,
        feature_columns=learner_columns,
        instrument_code=instrument_code,
        session_code=session_code,
        session_timestamps_ns=unique_sessions,
        instrument_vocabulary=vocabulary,
        sorted_keys=sorted_keys,
        sorted_rows=order.astype(np.int64),
    )


@dataclass(frozen=True, slots=True)
class PreparedHorizonLabels:
    """Narrow per-horizon target/outcome/availability aligned to matrix rows."""

    horizon_sessions: int
    row_index: np.ndarray
    target: np.ndarray
    realized: np.ndarray
    available_time_ns: np.ndarray
    risk_residual: np.ndarray
    reference_cost: np.ndarray
    gross_return: np.ndarray

    def train_positions(self, train_rows: np.ndarray) -> np.ndarray:
        """Positions of labeled rows inside a sorted candidate row array."""
        return np.searchsorted(np.asarray(train_rows, dtype=np.int64), self.row_index)


def prepare_horizon_labels(
    matrix: PreparedTrainingMatrix,
    data: NetAlphaResearchData,
    horizon_sessions: int,
    *,
    route_objective: object | None = None,
) -> PreparedHorizonLabels:
    """Align one horizon's narrow labels onto matrix rows fail-closed.

    Duplicate label keys are rejected outright and any label key without a
    matching decision row raises instead of being silently dropped.
    """
    label_frame = data.labels_by_horizon.get(horizon_sessions)
    if label_frame is None or label_frame.is_empty():
        raise ValueError(f"horizon {horizon_sessions} has no label frame")
    required = ("instrument_id", SESSION_COLUMN, TARGET_COLUMN, AVAILABLE_COLUMN)
    missing = [c for c in required if c not in label_frame.columns]
    if missing:
        raise ValueError(
            f"horizon {horizon_sessions} label frame missing columns {missing}"
        )
    duplicates = (
        label_frame.group_by(["instrument_id", SESSION_COLUMN])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ValueError(
            f"duplicate label keys at horizon {horizon_sessions} fail closed"
        )

    vocabulary_index = {
        name: code for code, name in enumerate(matrix.instrument_vocabulary)
    }
    label_session_ns = (
        label_frame[SESSION_COLUMN].to_physical().to_numpy().astype(np.int64)
    ) * 1_000
    timestamps = matrix.session_timestamps_ns
    first_ts = int(timestamps[0])
    last_ts = int(timestamps[-1])
    session_code = np.searchsorted(timestamps, label_session_ns).astype(np.int32)
    clipped = np.clip(session_code, 0, matrix.num_sessions - 1)
    exact_match = timestamps[clipped] == label_session_ns
    in_window = (label_session_ns >= first_ts) & (label_session_ns <= last_ts)
    if bool(np.any(in_window & ~exact_match)):
        raise ValueError(
            f"unmatched label sessions fail closed at horizon {horizon_sessions}"
        )
    # Sessions outside the prepared window sit beyond the locked holdout or
    # warm-up boundary; legacy composition dropped them via the same key
    # inner join, so they are excluded here rather than failing the run.
    keep = exact_match
    if not bool(np.all(keep)):
        label_frame = label_frame.filter(pl.Series(keep))
        label_session_ns = label_session_ns[keep]
        session_code = session_code[keep]
    instrument_values = label_frame["instrument_id"].to_list()
    unknown = {str(v) for v in instrument_values} - set(vocabulary_index)
    if unknown:
        raise ValueError(
            f"unmatched label instruments fail closed: {sorted(unknown)[:5]}"
        )
    instrument_code = np.asarray(
        [vocabulary_index[str(v)] for v in instrument_values], dtype=np.int64
    )
    keys = instrument_code * matrix.num_sessions + session_code.astype(np.int64)
    rows = matrix.rows_for_keys(keys)
    if np.any(rows < 0):
        raise ValueError(
            f"{int((rows < 0).sum())} unmatched label keys fail closed "
            f"at horizon {horizon_sessions}"
        )

    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order].astype(np.int64)
    available_ns = (
        label_frame[AVAILABLE_COLUMN].to_physical().to_numpy().astype(np.int64)
    )[order] * 1_000

    # Determine route kind if supplied
    route_kind = None
    if route_objective is not None:
        kind_val = getattr(route_objective, "kind", route_objective)
        route_kind = str(getattr(kind_val, "value", kind_val)).lower()

    # Gross column handling
    if GROSS_COLUMN in label_frame.columns:
        gross_raw = label_frame[GROSS_COLUMN].to_numpy().astype(np.float64)[order]
    else:
        gross_raw = np.full(rows_sorted.size, np.nan, dtype=np.float64)

    if route_kind is not None and "unhedged" in route_kind:
        if GROSS_COLUMN not in label_frame.columns:
            raise ValueError(f"unhedged_absolute route requires {GROSS_COLUMN!r} column (gross missing)")
        # validate gross and reference_cost present and finite
        if REFERENCE_COST_COLUMN not in label_frame.columns:
            raise ValueError(f"unhedged route missing {REFERENCE_COST_COLUMN!r}")
        gross_series = label_frame[GROSS_COLUMN]
        cost_series = label_frame[REFERENCE_COST_COLUMN]
        if gross_series.null_count() > 0 or cost_series.null_count() > 0:
            raise ValueError("gross_return/reference_cost has null rows")
        gross_arr = gross_series.to_numpy().astype(np.float64)
        cost_arr = cost_series.to_numpy().astype(np.float64)
        if not np.all(np.isfinite(gross_arr)) or not np.all(np.isfinite(cost_arr)):
            raise ValueError("gross_return/reference_cost must be finite")
        if np.any(cost_arr < 0):
            raise ValueError("reference_cost must be non-negative")
        # project target/realized to gross - cost
        gross = label_frame[GROSS_COLUMN].to_numpy().astype(np.float64)
        cost = label_frame[REFERENCE_COST_COLUMN].to_numpy().astype(np.float64)
        target = (gross - cost)[order]
        realized = (gross - cost)[order]
        risk_residual = label_frame[RISK_RESIDUAL_COLUMN].to_numpy().astype(np.float64)[order] if RISK_RESIDUAL_COLUMN in label_frame.columns else np.full(rows_sorted.size, np.nan)
        reference_cost = cost[order]
        gross_return = gross[order]
    elif route_kind is not None and "hedged" in route_kind:
        if RISK_RESIDUAL_COLUMN not in label_frame.columns or REFERENCE_COST_COLUMN not in label_frame.columns:
            raise ValueError("hedged route missing risk_residual/reference_cost")
        resid = label_frame[RISK_RESIDUAL_COLUMN].to_numpy().astype(np.float64)
        cost = label_frame[REFERENCE_COST_COLUMN].to_numpy().astype(np.float64)
        if not np.all(np.isfinite(resid)) or not np.all(np.isfinite(cost)):
            raise ValueError("hedged route columns must be finite")
        target = (resid - cost)[order]
        realized = (resid - cost)[order]
        risk_residual = resid[order]
        reference_cost = cost[order]
        gross_return = gross_raw
    else:
        # legacy path
        target = label_frame[TARGET_COLUMN].to_numpy().astype(np.float64)[order]
        if RISK_RESIDUAL_COLUMN in label_frame.columns and REFERENCE_COST_COLUMN in label_frame.columns:
            residual = label_frame[RISK_RESIDUAL_COLUMN].to_numpy().astype(np.float64)
            cost = label_frame[REFERENCE_COST_COLUMN].to_numpy().astype(np.float64)
            realized = (residual - cost)[order]
            risk_residual = residual[order]
            reference_cost = cost[order]
        else:
            realized = np.full(rows_sorted.size, np.nan, dtype=np.float64)
            risk_residual = np.full(rows_sorted.size, np.nan, dtype=np.float64)
            reference_cost = np.full(rows_sorted.size, np.nan, dtype=np.float64)
        gross_return = gross_raw

    return PreparedHorizonLabels(
        horizon_sessions=int(horizon_sessions),
        row_index=rows_sorted,
        target=target,
        realized=realized,
        available_time_ns=available_ns.astype(np.int64),
        risk_residual=risk_residual,
        reference_cost=reference_cost,
        gross_return=gross_return,
    )
