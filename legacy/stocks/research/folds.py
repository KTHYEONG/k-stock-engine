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

    ``segment_id`` is the stable identity of this validation segment in the
    common fold plan (equal to the fold index), and ``validation_sessions``
    records the exact session-index boundaries of the contiguous validation
    block so every candidate horizon reuses the same segment.
    """

    train_mask: list[int]
    validation_mask: list[int]
    train_label_end: int
    validation_decision_start: int
    segment_id: int = 0
    validation_sessions: tuple[int, ...] = ()

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
    ``embargo_sessions`` is applied. Training windows expand across folds, and
    the newest block can be pinned as a manifest-guarded holdout.
    ``max_train_sessions`` optionally bounds each training window to the
    newest eligible sessions retained after purge and embargo; ``None``
    preserves the expanding behavior exactly.
    """

    def __init__(
        self,
        n_folds: int,
        label_horizon_sessions: int,
        embargo_sessions: int = 0,
        session_column: str = "session_index",
        validation_window_sessions: int | None = None,
        min_train_sessions: int = 0,
        balanced: bool = True,
        max_train_sessions: int | None = None,
    ):
        if n_folds < 1:
            raise ValueError("n_folds must be positive")
        if label_horizon_sessions < 1:
            raise ValueError("label_horizon_sessions must be positive")
        if embargo_sessions < 0:
            raise ValueError("embargo_sessions must be non-negative")
        if validation_window_sessions is not None and validation_window_sessions < 1:
            raise ValueError("validation_window_sessions must be positive")
        if min_train_sessions < 0:
            raise ValueError("min_train_sessions must be non-negative")
        if max_train_sessions is not None and max_train_sessions <= 0:
            raise ValueError("max_train_sessions must be positive")
        self.n_folds = n_folds
        self.label_horizon_sessions = label_horizon_sessions
        self.embargo_sessions = embargo_sessions
        self.session_column = session_column
        self.validation_window_sessions = validation_window_sessions
        self.min_train_sessions = min_train_sessions
        self.balanced = balanced
        # Optional rolling fit window: a finite cap retains only the newest
        # eligible sessions after purge and embargo; None keeps expanding.
        self.max_train_sessions = max_train_sessions
        self._inspected_holdout_fingerprints: set[str] = set()

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

    def _row_index(self, samples: pl.DataFrame) -> tuple[list[int], dict[int, list[int]]]:
        self._validate_no_duplicate_sessions(samples)
        indexed = samples.with_row_index("__row_idx").select(
            self.session_column, "__row_idx"
        )
        grouped = indexed.group_by(self.session_column, maintain_order=True).agg(
            pl.col("__row_idx")
        )
        by_session: dict[int, list[int]] = {}
        for session_value, row_indices in grouped.iter_rows():
            by_session[int(session_value)] = [int(index) for index in row_indices]
        return [int(value) for value in indexed["__row_idx"].to_list()], by_session

    def split(self, samples: pl.DataFrame) -> list[Fold]:
        """Outer expanding walk-forward folds over all samples.

        With ``balanced=True`` (default) and no explicit
        ``validation_window_sessions``, every session after the first
        validation decision is split into ``n_folds`` contiguous validation
        segments whose session counts differ by at most one, so no fold
        inherits a disproportionate remainder. An explicit
        ``validation_window_sessions`` preserves the legacy fixed-window
        calendar.
        """
        if self.session_column not in samples.columns:
            raise ValueError(f"missing session column {self.session_column!r}")
        sessions = sorted(samples[self.session_column].unique().to_list())
        if len(sessions) < self.n_folds:
            raise ValueError(
                f"cannot create {self.n_folds} folds from {len(sessions)} sessions"
            )
        _all_rows, by_session = self._row_index(samples)
        first_validation_start = (
            self.min_train_sessions + self.label_horizon_sessions + self.embargo_sessions
        )
        if self.balanced and self.validation_window_sessions is None:
            return self._balanced_contiguous_splits(
                sessions, by_session, first_validation_start
            )

        folds: list[Fold] = []
        validation_window = self.validation_window_sessions or max(1, len(sessions) // self.n_folds)
        for k in range(self.n_folds):
            v_start = first_validation_start + k * validation_window
            v_end = (
                min(len(sessions), v_start + validation_window)
                if k < self.n_folds - 1
                else len(sessions)
            )
            if v_start >= len(sessions):
                break
            fold = self._purged_fold(sessions, by_session, sessions[v_start:v_end], segment_id=k)
            if fold is None:
                continue
            folds.append(fold)
        return folds

    def _balanced_contiguous_splits(
        self,
        sessions: list[int],
        by_session: dict[int, list[int]],
        first_validation_start: int,
    ) -> list[Fold]:
        """Split every remaining session into balanced contiguous segments.

        Each segment's session count differs from every other by at most one.
        If ``n_folds`` non-empty contiguous segments cannot be formed (the
        remaining session count is smaller than ``n_folds``), no fold is
        returned and the caller fails closed.
        """
        remaining = len(sessions) - first_validation_start
        if remaining < self.n_folds:
            return []
        base, extra = divmod(remaining, self.n_folds)
        folds: list[Fold] = []
        cursor = first_validation_start
        for k in range(self.n_folds):
            width = base + (1 if k < extra else 0)
            segment = sessions[cursor : cursor + width]
            cursor += width
            fold = self._purged_fold(sessions, by_session, segment, segment_id=k)
            if fold is None:
                continue
            folds.append(fold)
        return folds

    def inner_folds(self, samples: pl.DataFrame, n_inner: int | None = None) -> list[Fold]:
        """Nested expanding folds restricted to ``samples``.

        Inner folds never see outer validation rows: they are generated only
        from the caller-supplied training slice.
        """
        if self.session_column not in samples.columns:
            raise ValueError(f"missing session column {self.session_column!r}")
        sessions = sorted(samples[self.session_column].unique().to_list())
        n_inner = n_inner or max(1, self.n_folds)
        if len(sessions) < n_inner:
            n_inner = len(sessions)
        if n_inner < 1:
            return []
        _, by_session = self._row_index(samples)
        folds: list[Fold] = []
        validation_window = self.validation_window_sessions or max(1, len(sessions) // n_inner)
        for k in range(n_inner):
            v_start = k * validation_window
            v_end = (
                min(len(sessions), v_start + validation_window)
                if k < n_inner - 1
                else len(sessions)
            )
            if v_start >= len(sessions):
                break
            fold = self._purged_fold(sessions, by_session, sessions[v_start:v_end])
            if fold is not None:
                folds.append(fold)
        return folds

    def holdout(self, samples: pl.DataFrame, holdout_sessions: int) -> Fold:
        """Pin the newest ``holdout_sessions`` sessions as a locked final holdout."""
        if self.session_column not in samples.columns:
            raise ValueError(f"missing session column {self.session_column!r}")
        if holdout_sessions < 1:
            raise ValueError("holdout_sessions must be positive")
        sessions = sorted(samples[self.session_column].unique().to_list())
        if len(sessions) <= holdout_sessions:
            raise ValueError("holdout_sessions must be less than the session count")
        _, by_session = self._row_index(samples)
        fold = self._purged_fold(sessions, by_session, sessions[-holdout_sessions:])
        if fold is None:
            raise ValueError("holdout has no eligible training rows")
        return fold

    def pin_holdout(self, fingerprint: str) -> None:
        """Lock a holdout fingerprint for one candidate version."""
        if fingerprint in self._inspected_holdout_fingerprints:
            raise ValueError(
                f"holdout for candidate version {fingerprint!r} already inspected; "
                "reuse of the same version after holdout inspection is rejected"
            )
        self._inspected_holdout_fingerprints.add(fingerprint)

    def mark_holdout_inspected(self, fingerprint: str) -> None:
        """Record that a pinned holdout was evaluated once for this version."""
        self._inspected_holdout_fingerprints.add(fingerprint)

    def _purged_fold(
        self,
        sessions: list[int],
        by_session: dict[int, list[int]],
        validation_sessions: list[int],
        segment_id: int = 0,
    ) -> Fold | None:
        validation_decision_start = validation_sessions[0]
        train_sessions = [
            s
            for s in sessions
            if s < validation_decision_start
            and s + self.label_horizon_sessions + self.embargo_sessions
            < validation_decision_start
        ]
        if not train_sessions:
            return None
        if len(train_sessions) < self.min_train_sessions:
            return None
        if (
            self.max_train_sessions is not None
            and len(train_sessions) > self.max_train_sessions
        ):
            # Retain only the newest eligible sessions; purge and embargo
            # invariants are preserved by the truncated suffix.
            train_sessions = train_sessions[-self.max_train_sessions :]
        train_mask: list[int] = []
        validation_mask: list[int] = []
        for s in train_sessions:
            train_mask.extend(by_session[s])
        for s in validation_sessions:
            validation_mask.extend(by_session[s])
        return Fold(
            train_mask=sorted(train_mask),
            validation_mask=sorted(validation_mask),
            train_label_end=train_sessions[-1] + self.label_horizon_sessions,
            validation_decision_start=validation_decision_start,
            segment_id=segment_id,
            validation_sessions=tuple(validation_sessions),
        )
