"""Purged, embargoed walk-forward folds over trading-session indices."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/validation pair expressed in session-index space.

    ``train_label_end`` is the session index at which the last training label
    interval terminates; ``validation_decision_start`` is the session index at
    which the first validation decision is made. Purged + embargoed folds always
    satisfy ``train_label_end < validation_decision_start``.
    """

    train_mask: list[int]
    validation_mask: list[int]
    train_label_end: int
    validation_decision_start: int

    def num_train(self) -> int:
        return len(self.train_mask)

    def num_validation(self) -> int:
        return len(self.validation_mask)


class PurgedWalkForward:
    """Deterministic walk-forward splitter with purging and embargo.

    Samples are expected to carry a ``session_index`` column of integer trading
    sessions. A training label interval spans ``label_horizon_sessions`` sessions
    from its decision session, so any training sample whose label window could
    touch the validation window is purged, then an additional embargo of
    ``embargo_sessions`` is applied.
    """

    def __init__(
        self,
        n_folds: int,
        label_horizon_sessions: int,
        embargo_sessions: int = 0,
        session_column: str = "session_index",
    ):
        if n_folds < 1:
            raise ValueError("n_folds must be positive")
        if label_horizon_sessions < 1:
            raise ValueError("label_horizon_sessions must be positive")
        if embargo_sessions < 0:
            raise ValueError("embargo_sessions must be non-negative")
        self.n_folds = n_folds
        self.label_horizon_sessions = label_horizon_sessions
        self.embargo_sessions = embargo_sessions
        self.session_column = session_column

    def _validate_no_duplicate_sessions(self, samples: pl.DataFrame) -> None:
        """Fail closed on a temporal violation: the same instrument observed at
        the same session twice is not a monotonic timeline."""
        id_col = "instrument_id"
        if id_col not in samples.columns:
            return
        dup = samples.group_by([id_col, self.session_column]).len().filter(pl.col("len") > 1)
        if not dup.is_empty():
            row = dup.head(1).to_dicts()[0]
            raise ValueError(
                f"duplicate session rows for {id_col}={row[id_col]!r} "
                f"at {self.session_column}={row[self.session_column]}"
            )

    def split(self, samples: pl.DataFrame) -> list[Fold]:
        if self.session_column not in samples.columns:
            raise ValueError(f"missing session column {self.session_column!r}")
        sessions = sorted(samples[self.session_column].unique().to_list())
        if any(sessions[i] <= sessions[i - 1] for i in range(1, len(sessions))):
            raise ValueError("session indices must be strictly increasing")
        if len(sessions) < self.n_folds:
            raise ValueError(
                f"cannot create {self.n_folds} folds from {len(sessions)} sessions"
            )
        self._validate_no_duplicate_sessions(samples)

        rows = samples.with_row_index("__row_idx").to_dicts()
        by_session: dict[int, list[int]] = {}
        for row in rows:
            by_session.setdefault(int(row[self.session_column]), []).append(int(row["__row_idx"]))

        folds: list[Fold] = []
        validation_window = max(1, len(sessions) // self.n_folds)
        for k in range(self.n_folds):
            v_start = k * validation_window
            v_end = min(len(sessions), v_start + validation_window) if k < self.n_folds - 1 else len(sessions)
            validation_sessions = set(sessions[v_start:v_end])
            validation_decision_start = sessions[v_start]

            train_sessions = [
                s
                for s in sessions[:v_start]
                if s + self.label_horizon_sessions + self.embargo_sessions
                < validation_decision_start
            ]

            if not train_sessions:
                continue

            train_mask: list[int] = []
            validation_mask: list[int] = []
            for s in train_sessions:
                train_mask.extend(by_session[s])
            for s in validation_sessions:
                validation_mask.extend(by_session[s])

            train_label_end = train_sessions[-1] + self.label_horizon_sessions
            folds.append(
                Fold(
                    train_mask=sorted(train_mask),
                    validation_mask=sorted(validation_mask),
                    train_label_end=train_label_end,
                    validation_decision_start=validation_decision_start,
                )
            )
        return folds
