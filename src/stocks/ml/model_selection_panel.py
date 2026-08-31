# mypy: ignore-errors
"""Extracted fold learning panel boundary."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from src.stocks.ml.labels import AVAILABLE_COLUMN, SESSION_COLUMN, TARGET_COLUMN
from src.stocks.research.folds import Fold

logger = logging.getLogger("stocks.ml.model_selection_panel")

_ID_COLUMN = "instrument_id"
_SESSION_IDX = "session_index"


@dataclass(frozen=True, slots=True)
class FoldLearningPanel:
    train: pl.DataFrame
    validation: pl.DataFrame
    dropped_unlabeled_train_rows: int
    dropped_unlabeled_validation_rows: int
    training_cutoff: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.train, pl.DataFrame):
            raise ValueError("train must be DataFrame")
        if not isinstance(self.validation, pl.DataFrame):
            raise ValueError("validation must be DataFrame")
        if not isinstance(self.dropped_unlabeled_train_rows, int) or self.dropped_unlabeled_train_rows < 0:
            raise ValueError("dropped_unlabeled_train_rows must be non-negative int")
        if not isinstance(self.dropped_unlabeled_validation_rows, int) or self.dropped_unlabeled_validation_rows < 0:
            raise ValueError("dropped_unlabeled_validation_rows must be non-negative int")
        if not isinstance(self.training_cutoff, datetime):
            raise ValueError("training_cutoff must be datetime")


def build_fold_learning_panel(*, feature_frame: pl.DataFrame, label_join: pl.DataFrame, fold: Fold) -> FoldLearningPanel:
    for frame, name in ((feature_frame, "feature"), (label_join, "label")):
        if _ID_COLUMN in frame.columns and SESSION_COLUMN in frame.columns:
            dup = frame.group_by([_ID_COLUMN, SESSION_COLUMN]).len().filter(pl.col("len") > 1)
            if not dup.is_empty():
                raise ValueError(f"duplicate {name} keys")
    training_cutoff: datetime
    if _SESSION_IDX in feature_frame.columns:
        try:
            val_sessions = feature_frame.filter(pl.col(_SESSION_IDX) >= int(fold.validation_decision_start))
            if not val_sessions.is_empty() and SESSION_COLUMN in val_sessions.columns:
                training_cutoff = val_sessions[SESSION_COLUMN].min()
                if training_cutoff is None:
                    training_cutoff = feature_frame[SESSION_COLUMN].max()
            elif SESSION_COLUMN in feature_frame.columns:
                training_cutoff = feature_frame[SESSION_COLUMN].sort().to_list()[int(fold.validation_decision_start)] if int(fold.validation_decision_start) < feature_frame.height else feature_frame[SESSION_COLUMN].max()
            else:
                training_cutoff = datetime.now(tz=None)
        except Exception:
            training_cutoff = feature_frame[SESSION_COLUMN].min() if SESSION_COLUMN in feature_frame.columns else datetime.min
    else:
        if SESSION_COLUMN in feature_frame.columns:
            try:
                sessions_sorted = sorted(feature_frame[SESSION_COLUMN].unique().to_list())
                idx = int(fold.validation_decision_start)
                if 0 <= idx < len(sessions_sorted):
                    training_cutoff = sessions_sorted[idx]
                else:
                    training_cutoff = sessions_sorted[-1] if sessions_sorted else datetime.min
            except Exception:
                training_cutoff = feature_frame[SESSION_COLUMN].min()
        else:
            training_cutoff = datetime.min
    if not isinstance(training_cutoff, datetime):
        try:
            training_cutoff = feature_frame[SESSION_COLUMN].min()
        except Exception:
            training_cutoff = datetime(2024, 1, 1, tzinfo=UTC)
    try:
        train_features = feature_frame[fold.train_mask]
        validation_features = feature_frame[fold.validation_mask]
    except Exception:
        train_features = feature_frame.filter(pl.col(_SESSION_IDX) < int(fold.validation_decision_start)) if _SESSION_IDX in feature_frame.columns else feature_frame.head(int(fold.train_label_end) + 1)
        validation_features = feature_frame.filter(pl.col(_SESSION_IDX) >= int(fold.validation_decision_start)) if _SESSION_IDX in feature_frame.columns else feature_frame.slice(int(fold.train_label_end) + 1)
    def _prepare_panel(part_features: pl.DataFrame, is_train: bool) -> tuple[pl.DataFrame, int]:
        if part_features.is_empty():
            return part_features.head(0), 0
        joined = part_features.join(label_join, on=[_ID_COLUMN, SESSION_COLUMN], how="left")
        total = int(part_features.height)
        if TARGET_COLUMN not in joined.columns:
            return part_features.head(0), total
        target_series = joined[TARGET_COLUMN].cast(pl.Float64, strict=False)
        avail_col = "label_available_time" if "label_available_time" in joined.columns else AVAILABLE_COLUMN if AVAILABLE_COLUMN in joined.columns else None
        usable_mask = target_series.is_not_null() & target_series.is_finite()
        if avail_col is not None and avail_col in joined.columns:
            avail_series = joined[avail_col]
            if is_train:
                usable_mask = usable_mask & avail_series.is_not_null() & (avail_series <= training_cutoff)
            else:
                usable_mask = usable_mask & avail_series.is_not_null()
        try:
            filtered = joined.filter(usable_mask)
        except Exception:
            filtered = joined.filter(pl.col(TARGET_COLUMN).is_not_null())
        dropped = total - int(filtered.height)
        return filtered, dropped
    train_panel, dropped_train = _prepare_panel(train_features, True)
    validation_panel, dropped_validation = _prepare_panel(validation_features, False)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[DATA] stage=learning_panel fold_id=%d rows=%d dropped_unlabeled=%d sessions=%d",
            int(fold.segment_id),
            int(train_panel.height + validation_panel.height),
            int(dropped_train + dropped_validation),
            int(train_panel[SESSION_COLUMN].n_unique() if SESSION_COLUMN in train_panel.columns else 0) + int(validation_panel[SESSION_COLUMN].n_unique() if SESSION_COLUMN in validation_panel.columns else 0),
        )
    return FoldLearningPanel(
        train=train_panel,
        validation=validation_panel,
        dropped_unlabeled_train_rows=int(dropped_train),
        dropped_unlabeled_validation_rows=int(dropped_validation),
        training_cutoff=training_cutoff,
    )
