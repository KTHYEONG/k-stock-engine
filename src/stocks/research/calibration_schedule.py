"""Monotonic causal calibration schedule for economic replay decisions.

``CausalCalibrationSchedule`` advances a single monotonic cursor over the
canonical (session, instrument) sorted calibration ledger once, so the repeated
per-decision filter, sort, and score-bucket recomputation of the reference
``CausalAlphaCalibrator.prepare_decision`` collapse to per-bucket residual
appends. The moving-block bootstrap then runs on incremental cumulative sums
through the deterministic block-prefix-sum kernel, reducing workspace from
``O(draws * rows)`` to ``O(draws * rows / block_length)`` while generating the
identical seeded block starts.

Numerical parity is guaranteed by construction:

- The schedule only follows the fast path when every session shares a single
  ``label_available_time`` and that timestamp is non-decreasing with ``session``
  (the normal multi-horizon label invariant). Any partial-availability session
  or non-monotone availability silently selects the reference
  ``prepare_decision`` path for every affected decision.
- A bucket whose prefix-sum lower bound lands within the conservative float64
  error bound of the zero economic gate is recomputed with the exact reference
  kernel; gate outcomes never depend on prefix-sum rounding.
- ``apply_prepared`` remains the scoring path, so future labels can never enter
  a returned state.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from src.core.costs import CostSchedule
from src.stocks.research.economic_alpha import (
    _MIN_BUCKET_OBSERVATIONS,
    _SHRINKAGE_THRESHOLD,
    INSTRUMENT_COLUMN,
    SCORE_COLUMN,
    SESSION_COLUMN,
    BucketEvidence,
    CausalAlphaCalibrator,
    _block_bootstrap_lower_bound,
    _bootstrap_error_bound_from_scale,
    _bucket_expression,
    _exit_cost_rate,
    _prefix_sum_means_from_csum,
    _round_trip_cost_rate,
    _validate_observations,
)


@dataclass(slots=True)
class _SessionRecord:
    """One canonical-sorted session with precomputed buckets and residuals."""

    session: datetime
    label_available_time: datetime
    bucket: np.ndarray
    residual: np.ndarray


@dataclass(slots=True)
class _BucketState:
    """Incrementally grown cumulative-sum residuals for one score bucket."""

    csum: np.ndarray
    size: int
    max_abs: float

    @classmethod
    def empty(cls) -> _BucketState:
        return cls(csum=np.zeros(1, dtype=np.float64), size=0, max_abs=0.0)


class CausalCalibrationSchedule:
    """Incrementally revealed, state-identical causal calibration schedule."""

    def __init__(
        self,
        observations: pl.DataFrame,
        calibrator: CausalAlphaCalibrator,
        cost_schedule: CostSchedule,
        *,
        max_workspace_bytes: int | None,
    ) -> None:
        self._observations = observations
        self._calibrator = calibrator
        self._cost_schedule = cost_schedule
        self._max_workspace_bytes = max_workspace_bytes
        self._records: list[_SessionRecord] = []
        self._buckets: dict[int, _BucketState] = {}
        self._total_sum = 0.0
        self._total_count = 0
        self._history_sessions = 0
        self._pass_idx = 0
        self._reveal_idx = 0
        self._use_reference = False
        self._prepare(observations)

    @classmethod
    def build(
        cls,
        observations: pl.DataFrame,
        decision_times: Sequence[datetime],
        calibrator: CausalAlphaCalibrator,
        cost_schedule: CostSchedule,
        *,
        max_workspace_bytes: int,
    ) -> CausalCalibrationSchedule:
        """Build a schedule for the given canonical ledger and decision times.

        ``decision_times`` seeds the expected route decision schedule used for
        monotonicity pre-checks; ``state_at`` still accepts any non-decreasing
        decision time. Raises ``ValueError`` for an invalid ledger, an empty
        decision schedule, or a non-positive workspace cap.
        """
        if not decision_times:
            raise ValueError("decision_times must be non-empty")
        if max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive")
        return cls(
            observations,
            calibrator,
            cost_schedule,
            max_workspace_bytes=max_workspace_bytes,
        )

    def _prepare(self, observations: pl.DataFrame) -> None:
        label_column = self._calibrator.label_column
        label_available_column = self._calibrator.label_available_column
        _validate_observations(observations, label_column, label_available_column)
        canonical = observations.sort([SESSION_COLUMN, INSTRUMENT_COLUMN])
        if canonical.is_empty():
            self._use_reference = True
            return
        bucketed = canonical.with_columns(
            _bucket_expression(SCORE_COLUMN, self._calibrator.bucket_count).alias("__bucket")
        )
        prev_lat: datetime | None = None
        for rows in bucketed.partition_by(SESSION_COLUMN, maintain_order=True):
            lat_values = rows[label_available_column].unique().to_list()
            if len(lat_values) != 1:
                self._use_reference = True
                return
            if lat_values[0] is None:
                continue
            lat = cast_datetime(lat_values[0])
            session = cast_datetime(rows[SESSION_COLUMN][0])
            if prev_lat is not None and lat < prev_lat:
                self._use_reference = True
                return
            prev_lat = lat
            self._records.append(
                _SessionRecord(
                    session=session,
                    label_available_time=lat,
                    bucket=rows["__bucket"].to_numpy().astype(np.int64),
                    residual=rows[label_column].to_numpy().astype(np.float64),
                )
            )

    def eligible_prefix_rows(self, decision_time: datetime) -> int:
        """Rows revealed at ``decision_time`` (monotonic, idempotent)."""
        self._advance(decision_time)
        return self._total_count

    def _advance(self, decision_time: datetime) -> None:
        records = self._records
        while self._pass_idx < len(records) and records[self._pass_idx].session < decision_time:
            self._pass_idx += 1
        while (
            self._reveal_idx < self._pass_idx
            and records[self._reveal_idx].label_available_time <= decision_time
        ):
            self._append_session(records[self._reveal_idx])
            self._reveal_idx += 1

    def _append_session(self, record: _SessionRecord) -> None:
        residual = record.residual
        for bucket in np.unique(record.bucket):
            mask = (record.bucket == bucket) & ~np.isnan(residual)
            residuals = residual[mask]
            if residuals.size == 0:
                continue
            state = self._buckets.get(int(bucket))
            if state is None:
                state = _BucketState.empty()
                self._buckets[int(bucket)] = state
            start = state.csum[-1]
            state.csum = np.concatenate([state.csum, start + np.cumsum(residuals)])
            state.size += residuals.size
            state.max_abs = max(state.max_abs, float(np.max(np.abs(residuals))))
            self._total_sum += float(np.sum(residuals))
            self._total_count += residuals.size
        self._history_sessions += 1

    def state_at(
        self,
        decision_time: datetime,
        *,
        max_bootstrap_workspace_bytes: int | None = None,
    ) -> dict[str, object]:
        """Frozen calibration state at a non-decreasing decision time.

        Returns the same state shape as
        :meth:`CausalAlphaCalibrator.prepare_decision` for the same inputs, so
        ``apply_prepared`` consumes it unchanged. Partial-availability or
        non-monotone ledgers and near-gate bootstrap buckets delegate to the
        exact reference computation.
        """
        workspace_cap = (
            max_bootstrap_workspace_bytes
            if max_bootstrap_workspace_bytes is not None
            else self._max_workspace_bytes
        )
        if self._use_reference:
            return self._reference_state(decision_time, workspace_cap)
        self._advance(decision_time)
        point = self._cost_schedule.cost_for(decision_time)
        round_trip_cost = _round_trip_cost_rate(point)
        exit_cost = _exit_cost_rate(point)
        if self._total_count == 0 or self._history_sessions < self._calibrator.min_calibration_sessions:
            empty: dict[str, object] = {
                "bucket_count": int(self._calibrator.bucket_count),
                "history_sessions": int(self._history_sessions),
                "round_trip_cost": float(round_trip_cost),
                "exit_cost_rate": float(exit_cost),
                "buckets": [],
            }
            self._sync_calibrator(empty)
            return empty
        global_mean = self._total_sum / self._total_count
        bucket_rows: list[dict[str, object]] = []
        for bucket in sorted(self._buckets):
            bucket_state = self._buckets[bucket]
            if bucket_state.size < _MIN_BUCKET_OBSERVATIONS:
                bucket_rows.append(
                    {
                        "bucket": int(bucket),
                        "sample_size": int(bucket_state.size),
                        "expected_active_alpha": None,
                        "alpha_lower_bound": None,
                    }
                )
                continue
            lower_bound = self._bucket_lower_bound(
                bucket, bucket_state, workspace_cap
            )
            if lower_bound <= 0.0:
                bucket_rows.append(
                    {
                        "bucket": int(bucket),
                        "sample_size": int(bucket_state.size),
                        "expected_active_alpha": None,
                        "alpha_lower_bound": None,
                    }
                )
                continue
            shrink = min(1.0, _SHRINKAGE_THRESHOLD / bucket_state.size)
            bucket_mean = float(bucket_state.csum[-1]) / bucket_state.size
            shrunk = global_mean + (1.0 - shrink) * (bucket_mean - global_mean)
            bucket_rows.append(
                {
                    "bucket": int(bucket),
                    "sample_size": int(bucket_state.size),
                    "expected_active_alpha": float(shrunk),
                    "alpha_lower_bound": float(lower_bound),
                }
            )
        result: dict[str, object] = {
            "bucket_count": int(self._calibrator.bucket_count),
            "history_sessions": int(self._history_sessions),
            "round_trip_cost": float(round_trip_cost),
            "exit_cost_rate": float(exit_cost),
            "buckets": bucket_rows,
        }
        self._sync_calibrator(result)
        return result

    def _sync_calibrator(self, state: dict[str, object]) -> None:
        """Mirror the computed state onto the calibrator's frozen evidence.

        ``calibrator.calibration_state()`` (the artifact-serialized snapshot)
        is fed by the calibrator's private fields, so the schedule refreshes
        them from the identical state it returned.
        """
        calibrator = self._calibrator
        calibrator._last_history_sessions = int(cast(int, state["history_sessions"]))
        calibrator._last_round_trip_cost = float(cast(float, state["round_trip_cost"]))
        calibrator._last_exit_cost = float(cast(float, state["exit_cost_rate"]))
        calibrator._last_evidence = tuple(
            BucketEvidence(
                bucket=cast(int, row["bucket"]),
                sample_size=cast(int, row["sample_size"]),
                expected_active_alpha=(
                    None
                    if row["expected_active_alpha"] is None
                    else float(cast(float, row["expected_active_alpha"]))
                ),
                alpha_lower_bound=(
                    None
                    if row["alpha_lower_bound"] is None
                    else float(cast(float, row["alpha_lower_bound"]))
                ),
            )
            for row in cast(list[dict[str, object]], state["buckets"])
        )

    def _bucket_lower_bound(
        self,
        bucket: int,
        state: _BucketState,
        workspace_cap: int | None,
    ) -> float:
        n = state.size
        block = max(self._calibrator.block_length, 1)
        n_blocks = int(np.ceil(n / block))
        max_start = max(1, n - block + 1)
        means = _prefix_sum_means_from_csum(
            state.csum,
            n=n,
            block=block,
            n_blocks=n_blocks,
            max_start=max_start,
            n_bootstrap=self._calibrator.n_bootstrap,
            seed=self._calibrator.seed + bucket,
            max_workspace_bytes=workspace_cap,
        )
        estimate = float(np.quantile(means, self._calibrator.bootstrap_alpha))
        bound = _bootstrap_error_bound_from_scale(state.max_abs, n)
        if abs(estimate) <= bound:
            residuals = np.diff(state.csum)
            return _block_bootstrap_lower_bound(
                residuals,
                block,
                self._calibrator.n_bootstrap,
                self._calibrator.seed + bucket,
                self._calibrator.bootstrap_alpha,
                max_bootstrap_workspace_bytes=workspace_cap,
            )
        return estimate

    def _reference_state(
        self,
        decision_time: datetime,
        workspace_cap: int | None,
    ) -> dict[str, object]:
        return self._calibrator.prepare_decision(
            self._observations,
            decision_time,
            self._cost_schedule,
            max_bootstrap_workspace_bytes=workspace_cap,
        )


def cast_datetime(value: object) -> datetime:
    """Normalize a ledger timestamp to a tz-aware ``datetime``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"non-datetime calibration timestamp: {value!r}")


@dataclass(slots=True)
class _ClusterSessionRecord:
    """One canonical-sorted session with per-bucket residual sums/counts."""

    session: datetime
    label_available_time: datetime
    bucket: np.ndarray
    residual_sum: np.ndarray
    residual_count: np.ndarray


@dataclass(slots=True)
class _ClusterBucketState:
    """Chronologically appended per-session cluster aggregates for one bucket."""

    session_sums: np.ndarray
    session_counts: np.ndarray
    row_sum: float
    row_count: int
    max_abs: float

    @classmethod
    def empty(cls) -> _ClusterBucketState:
        return cls(
            session_sums=np.zeros(0, dtype=np.float64),
            session_counts=np.zeros(0, dtype=np.float64),
            row_sum=0.0,
            row_count=0,
            max_abs=0.0,
        )


class SessionClusterCalibrationSchedule:
    """Causal calibration schedule with a session-cluster bootstrap unit.

    Identical causal reveal and cost/shrinkage contract to
    :class:`CausalCalibrationSchedule`, but the bootstrap lower bound is
    computed over chronological **sessions** instead of individual instrument
    rows: residuals are aggregated per ``(session, bucket)`` into a ``float64``
    sum and count, the deterministic moving-block bootstrap draws blocks of
    consecutive sessions, and each draw computes ``draw_sum / draw_count``
    before the ``alpha`` quantile. Instruments exposed to the same market
    session are therefore not treated as independent adjacent observations,
    and the bootstrap state shrinks from roughly one aggregate per instrument
    row to one aggregate per session. The route holding horizon is the block
    length in sessions.

    A partial-availability or non-monotone-availability ledger silently selects
    the exact row-level reference ``prepare_decision`` path for every affected
    decision (tracked as a reference-fallback count). Missing or non-finite
    group sums/counts raise ``ValueError``.
    """

    def __init__(
        self,
        observations: pl.DataFrame,
        calibrator: CausalAlphaCalibrator,
        cost_schedule: CostSchedule,
        *,
        block_length: int,
        max_workspace_bytes: int | None,
    ) -> None:
        if block_length < 1:
            raise ValueError("session-cluster block length must be positive")
        self._observations = observations
        self._calibrator = calibrator
        self._cost_schedule = cost_schedule
        self._block_length = block_length
        self._max_workspace_bytes = max_workspace_bytes
        self._records: list[_ClusterSessionRecord] = []
        self._buckets: dict[int, _ClusterBucketState] = {}
        self._total_sum = 0.0
        self._total_count = 0
        self._history_sessions = 0
        self._pass_idx = 0
        self._reveal_idx = 0
        self._use_reference = False
        self._reference_fallback_count = 0
        self._prepare(observations)

    @classmethod
    def build(
        cls,
        observations: pl.DataFrame,
        decision_times: Sequence[datetime],
        calibrator: CausalAlphaCalibrator,
        cost_schedule: CostSchedule,
        *,
        block_length: int,
        max_workspace_bytes: int,
    ) -> SessionClusterCalibrationSchedule:
        """Build a session-cluster schedule for the canonical ledger."""
        if not decision_times:
            raise ValueError("decision_times must be non-empty")
        if max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive")
        return cls(
            observations,
            calibrator,
            cost_schedule,
            block_length=block_length,
            max_workspace_bytes=max_workspace_bytes,
        )

    def _prepare(self, observations: pl.DataFrame) -> None:
        label_column = self._calibrator.label_column
        label_available_column = self._calibrator.label_available_column
        _validate_observations(observations, label_column, label_available_column)
        canonical = observations.sort([SESSION_COLUMN, INSTRUMENT_COLUMN])
        if canonical.is_empty():
            self._use_reference = True
            return
        bucketed = canonical.with_columns(
            _bucket_expression(SCORE_COLUMN, self._calibrator.bucket_count).alias("__bucket")
        )
        prev_lat: datetime | None = None
        for rows in bucketed.partition_by(SESSION_COLUMN, maintain_order=True):
            lat_values = rows[label_available_column].unique().to_list()
            if len(lat_values) != 1:
                self._use_reference = True
                return
            if lat_values[0] is None:
                continue
            lat = cast_datetime(lat_values[0])
            session = cast_datetime(rows[SESSION_COLUMN][0])
            if prev_lat is not None and lat < prev_lat:
                self._use_reference = True
                return
            prev_lat = lat
            bucket = rows["__bucket"].to_numpy().astype(np.int64)
            residual = rows[label_column].to_numpy().astype(np.float64)
            non_null = ~np.isnan(residual)
            if np.any(non_null & ~np.isfinite(residual)):
                raise ValueError("session-cluster calibration residual must be finite")
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            for unique_bucket in np.unique(bucket):
                mask = (bucket == unique_bucket) & non_null
                counts[int(unique_bucket)] = int(np.count_nonzero(mask))
                sums[int(unique_bucket)] = float(np.sum(residual[mask]))
            self._records.append(
                _ClusterSessionRecord(
                    session=session,
                    label_available_time=lat,
                    bucket=np.asarray(list(sums), dtype=np.int64),
                    residual_sum=np.asarray(
                        [sums[b] for b in sums], dtype=np.float64
                    ),
                    residual_count=np.asarray(
                        [counts[b] for b in sums], dtype=np.float64
                    ),
                )
            )

    def eligible_prefix_rows(self, decision_time: datetime) -> int:
        """Rows revealed at ``decision_time`` (monotonic, idempotent)."""
        self._advance(decision_time)
        return self._total_count

    def _advance(self, decision_time: datetime) -> None:
        records = self._records
        while self._pass_idx < len(records) and records[self._pass_idx].session < decision_time:
            self._pass_idx += 1
        while (
            self._reveal_idx < self._pass_idx
            and records[self._reveal_idx].label_available_time <= decision_time
        ):
            self._append_session(records[self._reveal_idx])
            self._reveal_idx += 1

    def _append_session(self, record: _ClusterSessionRecord) -> None:
        for bucket, session_sum, session_count in zip(
            record.bucket, record.residual_sum, record.residual_count, strict=True
        ):
            if session_count <= 0:
                continue
            state = self._buckets.get(int(bucket))
            if state is None:
                state = _ClusterBucketState.empty()
                self._buckets[int(bucket)] = state
            state.session_sums = np.append(state.session_sums, session_sum)
            state.session_counts = np.append(state.session_counts, session_count)
            state.row_sum += float(session_sum)
            state.row_count += int(session_count)
            state.max_abs = max(
                state.max_abs,
                float(np.abs(session_sum)) / max(int(session_count), 1),
            )
            self._total_sum += float(session_sum)
            self._total_count += int(session_count)
        self._history_sessions += 1

    def state_at(
        self,
        decision_time: datetime,
        *,
        max_bootstrap_workspace_bytes: int | None = None,
    ) -> dict[str, object]:
        """Frozen calibration state at a non-decreasing decision time.

        Returns the same state shape as :class:`CausalCalibrationSchedule` so
        ``apply_prepared`` consumes it unchanged. Partial-availability or
        non-monotone ledgers delegate to the exact row-level reference
        computation and increment the reference-fallback count.
        """
        if self._use_reference:
            self._reference_fallback_count += 1
            return self._reference_state(
                decision_time,
                max_bootstrap_workspace_bytes
                if max_bootstrap_workspace_bytes is not None
                else self._max_workspace_bytes,
            )
        self._advance(decision_time)
        point = self._cost_schedule.cost_for(decision_time)
        round_trip_cost = _round_trip_cost_rate(point)
        exit_cost = _exit_cost_rate(point)
        if self._total_count == 0 or self._history_sessions < self._calibrator.min_calibration_sessions:
            empty: dict[str, object] = {
                "bucket_count": int(self._calibrator.bucket_count),
                "history_sessions": int(self._history_sessions),
                "round_trip_cost": float(round_trip_cost),
                "exit_cost_rate": float(exit_cost),
                "buckets": [],
            }
            self._sync_calibrator(empty)
            return empty
        global_mean = self._total_sum / self._total_count
        bucket_rows: list[dict[str, object]] = []
        for bucket in sorted(self._buckets):
            bucket_state = self._buckets[bucket]
            if bucket_state.row_count < _MIN_BUCKET_OBSERVATIONS:
                bucket_rows.append(
                    {
                        "bucket": int(bucket),
                        "sample_size": int(bucket_state.row_count),
                        "expected_active_alpha": None,
                        "alpha_lower_bound": None,
                    }
                )
                continue
            lower_bound = self._bucket_lower_bound(bucket, bucket_state)
            if lower_bound <= 0.0:
                bucket_rows.append(
                    {
                        "bucket": int(bucket),
                        "sample_size": int(bucket_state.row_count),
                        "expected_active_alpha": None,
                        "alpha_lower_bound": None,
                    }
                )
                continue
            shrink = min(1.0, _SHRINKAGE_THRESHOLD / bucket_state.row_count)
            bucket_mean = bucket_state.row_sum / bucket_state.row_count
            shrunk = global_mean + (1.0 - shrink) * (bucket_mean - global_mean)
            bucket_rows.append(
                {
                    "bucket": int(bucket),
                    "sample_size": int(bucket_state.row_count),
                    "expected_active_alpha": float(shrunk),
                    "alpha_lower_bound": float(lower_bound),
                }
            )
        result: dict[str, object] = {
            "bucket_count": int(self._calibrator.bucket_count),
            "history_sessions": int(self._history_sessions),
            "round_trip_cost": float(round_trip_cost),
            "exit_cost_rate": float(exit_cost),
            "buckets": bucket_rows,
        }
        self._sync_calibrator(result)
        return result

    def _bucket_lower_bound(self, bucket: int, state: _ClusterBucketState) -> float:
        n = state.session_sums.size
        block = max(self._block_length, 1)
        n_blocks = int(np.ceil(n / block))
        max_start = max(1, n - block + 1)
        means = _session_cluster_bootstrap_means(
            state.session_sums,
            state.session_counts,
            block=block,
            n_blocks=n_blocks,
            max_start=max_start,
            n_bootstrap=self._calibrator.n_bootstrap,
            seed=self._calibrator.seed + bucket,
        )
        return float(np.quantile(means, self._calibrator.bootstrap_alpha))

    def _sync_calibrator(self, state: dict[str, object]) -> None:
        calibrator = self._calibrator
        calibrator._last_history_sessions = int(cast(int, state["history_sessions"]))
        calibrator._last_round_trip_cost = float(cast(float, state["round_trip_cost"]))
        calibrator._last_exit_cost = float(cast(float, state["exit_cost_rate"]))
        calibrator._last_evidence = tuple(
            BucketEvidence(
                bucket=cast(int, row["bucket"]),
                sample_size=cast(int, row["sample_size"]),
                expected_active_alpha=(
                    None
                    if row["expected_active_alpha"] is None
                    else float(cast(float, row["expected_active_alpha"]))
                ),
                alpha_lower_bound=(
                    None
                    if row["alpha_lower_bound"] is None
                    else float(cast(float, row["alpha_lower_bound"]))
                ),
            )
            for row in cast(list[dict[str, object]], state["buckets"])
        )

    def _reference_state(
        self,
        decision_time: datetime,
        workspace_cap: int | None,
    ) -> dict[str, object]:
        return self._calibrator.prepare_decision(
            self._observations,
            decision_time,
            self._cost_schedule,
            max_bootstrap_workspace_bytes=workspace_cap,
        )

    def telemetry(self) -> dict[str, object]:
        return {
            "bootstrap_unit": "session_cluster",
            "block_length": int(self._block_length),
            "rows": int(self._total_count),
            "sessions": int(self._history_sessions),
            "draws": int(self._calibrator.n_bootstrap),
            "reference_fallback_count": int(self._reference_fallback_count),
        }


def _session_cluster_bootstrap_means(
    session_sums: np.ndarray,
    session_counts: np.ndarray,
    *,
    block: int,
    n_blocks: int,
    max_start: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Deterministic moving-block bootstrap means over chronological sessions.

    Each draw samples ``n_blocks`` blocks of ``block`` consecutive sessions
    (the final block may be truncated) and computes ``draw_sum / draw_count``
    where the sum/count accumulate the sampled block residual sums/counts. The
    prefix-sum reduction keeps the workspace ``O(draws * n_blocks)`` and the
    seeded RNG stream is consumed in a fixed row-major order.
    """
    if session_sums.size == 0:
        return np.zeros(n_bootstrap, dtype=np.float64)
    if not np.all(np.isfinite(session_sums)) or not np.all(
        np.isfinite(session_counts)
    ):
        raise ValueError("session-cluster bootstrap requires finite sums and counts")
    n = session_sums.size
    block = max(block, 1)
    n_blocks = max(1, n_blocks)
    max_start = max(1, min(max_start, n - block + 1))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
    csum_sums = np.zeros(n + 1, dtype=np.float64)
    csum_counts = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(session_sums, out=csum_sums[1:])
    np.cumsum(session_counts, out=csum_counts[1:])
    full_blocks = np.arange(max(0, n_blocks - 1))
    sum_total = (
        csum_sums[starts[:, full_blocks] + block] - csum_sums[starts[:, full_blocks]]
    ).sum(axis=1)
    count_total = (
        csum_counts[starts[:, full_blocks] + block]
        - csum_counts[starts[:, full_blocks]]
    ).sum(axis=1)
    last_end = np.minimum(starts[:, -1] + block, n)
    sum_total = sum_total + (csum_sums[last_end] - csum_sums[starts[:, -1]])
    count_total = count_total + (
        csum_counts[last_end] - csum_counts[starts[:, -1]]
    )
    return np.asarray(sum_total / np.where(count_total > 0, count_total, 1.0))
