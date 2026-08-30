"""Causal expected-active-alpha calibration for economic portfolio construction.

``CausalAlphaCalibrator`` maps each current cross-sectional score percentile
bucket to an expected residual active return using only *prior* label-available
out-of-sample observations (the route's ``label_available_column <= decision_time``
and ``session < decision_time``), shrinks the bucket mean toward the history mean,
and requires a positive moving-block bootstrap lower bound before the bucket is
usable. The expected alpha is then netted against the effective round-trip and
one-way exit costs resolved from the decision's ``CostSchedule`` (plus optional
liquidity impact), so ``expected_net_alpha`` is in the same session unit as the
route's residual label, and ``net_alpha_lower_bound`` is the bootstrap lower
bound net of the full round-trip cost. The label and availability columns are
constructor parameters so a longer-horizon route is calibrated against its own
label without scaling a five-day estimate.

A bucket whose evidence is insufficient or whose bootstrap lower bound is not
positive yields ``null`` expected alpha: ``null`` is not a buy signal and drives
the allocation path to cash or sell-only. All frame transforms are vectorized
Polars/NumPy; no ``pd.apply`` or per-row Python callback is used on this path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, cast

import numpy as np
import polars as pl

from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel

INSTRUMENT_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
SCORE_COLUMN = "score"
LABEL_COLUMN = "residual_o2o_5d"
LABEL_AVAILABLE_COLUMN = "label_available_time"

ALPHA_COLUMN = "expected_active_alpha"
NET_ALPHA_COLUMN = "expected_net_alpha"
LOWER_BOUND_COLUMN = "alpha_lower_bound"
NET_LOWER_BOUND_COLUMN = "net_alpha_lower_bound"
EXIT_COST_COLUMN = "exit_cost_rate"
ALPHA_STANDARD_ERROR_COLUMN: Final[str] = "alpha_standard_error"

_MIN_BUCKET_OBSERVATIONS = 5
_SHRINKAGE_THRESHOLD = 20.0
_ADAPTIVE_BUCKET_MIN_SESSIONS = 252
_ADAPTIVE_BUCKET_COLD_COUNT = 5
_DEFAULT_BLOCK_LENGTH = 5
_VOLATILITY_WINDOW_SESSIONS = 10
_DEFAULT_PARTICIPATION_LIMIT = 0.01


@dataclass(frozen=True, slots=True)
class BucketEvidence:
    """Immutable per-bucket economic evidence; JSON-safe via ``to_json_safe``."""

    bucket: int
    sample_size: int
    expected_active_alpha: float | None
    alpha_lower_bound: float | None
    alpha_standard_error: float | None = None

    def to_json_safe(self) -> dict[str, object]:
        return {
            "bucket": int(self.bucket),
            "sample_size": int(self.sample_size),
            "expected_active_alpha": (
                None
                if self.expected_active_alpha is None
                else round(float(self.expected_active_alpha), 10)
            ),
            "alpha_lower_bound": (
                None
                if self.alpha_lower_bound is None
                else round(float(self.alpha_lower_bound), 10)
            ),
            "alpha_standard_error": (
                None
                if self.alpha_standard_error is None
                else round(float(self.alpha_standard_error), 10)
            ),
        }


class CausalAlphaCalibrator:
    """Deterministic time-separated score-to-net-alpha calibration.

    The same input and seed always yield the same buckets, expected alpha, and
    confidence lower bound: the bootstrap resampling uses
    ``np.random.default_rng(seed + bucket)`` so the result is reproducible.
    """

    def __init__(
        self,
        bucket_count: int,
        min_calibration_sessions: int,
        seed: int = 42,
        n_bootstrap: int = 200,
        bootstrap_alpha: float = 0.05,
        block_length: int = _DEFAULT_BLOCK_LENGTH,
        participation_limit: float = _DEFAULT_PARTICIPATION_LIMIT,
        label_column: str = LABEL_COLUMN,
        label_available_column: str = LABEL_AVAILABLE_COLUMN,
        preserve_negative_bound_null: bool = True,
    ) -> None:
        if bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")
        if min_calibration_sessions < 1:
            raise ValueError("min_calibration_sessions must be positive")
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        if not 0.0 < bootstrap_alpha < 1.0:
            raise ValueError("bootstrap_alpha must be in (0, 1)")
        if block_length < 1:
            raise ValueError("block_length must be positive")
        if not 0.0 <= participation_limit <= 1.0:
            raise ValueError("participation_limit must be in [0, 1]")
        if not label_column:
            raise ValueError("label_column must be non-empty")
        if not label_available_column:
            raise ValueError("label_available_column must be non-empty")
        self.bucket_count = bucket_count
        self.min_calibration_sessions = min_calibration_sessions
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        self.bootstrap_alpha = bootstrap_alpha
        self.block_length = block_length
        self.participation_limit = participation_limit
        self.label_column = label_column
        self.label_available_column = label_available_column
        self.preserve_negative_bound_null = bool(preserve_negative_bound_null)
        self._last_evidence: tuple[BucketEvidence, ...] = ()
        self._last_history_sessions = 0
        self._last_round_trip_cost = 0.0
        self._last_exit_cost = 0.0
        self._last_effective_bucket_count = bucket_count

    @property
    def bucket_evidence(self) -> tuple[BucketEvidence, ...]:
        return self._last_evidence

    @property
    def history_sessions(self) -> int:
        return self._last_history_sessions

    @property
    def round_trip_cost(self) -> float:
        return self._last_round_trip_cost

    @property
    def exit_cost_rate(self) -> float:
        return self._last_exit_cost

    def transform(
        self,
        scored: pl.DataFrame,
        observations: pl.DataFrame,
        decision_time: datetime,
        cost_schedule: CostSchedule,
        liquidity_model: LiquiditySlippageModel | None = None,
    ) -> pl.DataFrame:
        """Return ``scored`` augmented with expected-active/net-alpha evidence.

        ``observations`` must carry ``instrument_id``, ``session``, ``score``,
        the route's residual label (``label_column``), and the route's
        ``label_available_column``; ``scored`` must carry ``instrument_id``,
        ``session``, and a score column (``pred_score`` or ``score``). Missing
        or non-finite inputs raise ``ValueError``. Only observations with
        ``label_available_column <= decision_time`` and
        ``session < decision_time`` are used, so no current or future label can
        leak into the calibration. ``scored`` rows at or after the decision time
        are excluded from the returned frame.
        """
        _validate_scored(scored)
        _validate_observations(observations, self.label_column, self.label_available_column)
        score_column = _resolve_score_column(scored)
        visible = scored.filter(pl.col(SESSION_COLUMN) <= decision_time)

        point = cost_schedule.cost_for(decision_time)
        self._last_round_trip_cost = _round_trip_cost_rate(point)
        self._last_exit_cost = _exit_cost_rate(point)
        liquidity_rate = (
            _liquidity_slippage_rates(
                visible, liquidity_model, self.participation_limit, decision_time
            )
            if liquidity_model is not None
            else None
        )

        eligible = observations.filter(
            (pl.col(self.label_available_column) <= decision_time)
            & (pl.col(SESSION_COLUMN) < decision_time)
        )
        self._last_history_sessions = int(
            eligible.select(pl.col(SESSION_COLUMN).n_unique()).to_series()[0]
        )
        if (
            eligible.is_empty()
            or self._last_history_sessions < self.min_calibration_sessions
        ):
            self._last_evidence = ()
            return _augment(
                visible, score_column, self.bucket_count,
                self._last_round_trip_cost, self._last_exit_cost, liquidity_rate,
            )
        bucket_count = adaptive_bucket_count(
            self.bucket_count, self._last_history_sessions
        )
        self._last_effective_bucket_count = bucket_count
        stats = _bucket_statistics(
            eligible,
            bucket_count,
            label_column=self.label_column,
            seed=self.seed,
            n_bootstrap=self.n_bootstrap,
            bootstrap_alpha=self.bootstrap_alpha,
            block_length=self.block_length,
            preserve_negative_bound_null=self.preserve_negative_bound_null,
        )
        self._last_evidence = tuple(
            sorted(
                (
                    BucketEvidence(
                        bucket=int(row["__bucket"]),
                        sample_size=int(row["sample_size"]),
                        expected_active_alpha=(
                            None
                            if row[ALPHA_COLUMN] is None
                            else float(row[ALPHA_COLUMN])
                        ),
                        alpha_lower_bound=(
                            None
                            if row[LOWER_BOUND_COLUMN] is None
                            else float(row[LOWER_BOUND_COLUMN])
                        ),
                        alpha_standard_error=(
                            None
                            if row.get(ALPHA_STANDARD_ERROR_COLUMN) is None
                            else float(row[ALPHA_STANDARD_ERROR_COLUMN])
                        ),
                    )
                    for row in stats.to_dicts()
                ),
                key=lambda evidence: evidence.bucket,
            )
        )
        return _augment(
            visible, score_column, bucket_count,
            self._last_round_trip_cost, self._last_exit_cost, liquidity_rate,
            bucket_stats=stats,
        )

    def prepare_decision(
        self,
        observations: pl.DataFrame,
        decision_time: datetime,
        cost_schedule: CostSchedule,
        *,
        max_bootstrap_workspace_bytes: int | None = None,
    ) -> dict[str, object]:
        """Compute the route-scoped calibration evidence once for a decision.

        Performs the ``session < decision_time`` and
        ``label_available_column <= decision_time`` filter, score-bucket
        calculation, shrinkage, and deterministic bootstrap exactly once,
        returning an immutable state consumed by ``apply_prepared``. The
        observations are ordered by ``(session, instrument_id)`` before
        bootstrap sampling so the artifact state is canonical. The reference
        ``transform`` and this prepared schedule share the same input ordering
        and therefore produce identical bucket IDs, sample sizes, alpha, and
        lower bounds for the same decision. ``max_bootstrap_workspace_bytes``
        bounds the bootstrap workspace so a memory-constrained replay processes
        the draws in deterministic batches without changing the evidence.
        """
        if max_bootstrap_workspace_bytes is not None and max_bootstrap_workspace_bytes <= 0:
            raise ValueError("max_bootstrap_workspace_bytes must be positive when supplied")
        _validate_observations(observations, self.label_column, self.label_available_column)
        point = cost_schedule.cost_for(decision_time)
        round_trip_cost = _round_trip_cost_rate(point)
        exit_cost = _exit_cost_rate(point)
        eligible = observations.filter(
            (pl.col(self.label_available_column) <= decision_time)
            & (pl.col(SESSION_COLUMN) < decision_time)
        ).sort([SESSION_COLUMN, INSTRUMENT_COLUMN])
        self._last_history_sessions = int(
            eligible.select(pl.col(SESSION_COLUMN).n_unique()).to_series()[0]
        ) if not eligible.is_empty() else 0
        self._last_round_trip_cost = round_trip_cost
        self._last_exit_cost = exit_cost
        if (
            eligible.is_empty()
            or self._last_history_sessions < self.min_calibration_sessions
        ):
            self._last_evidence = ()
            return {
                "bucket_count": int(self.bucket_count),
                "history_sessions": int(self._last_history_sessions),
                "round_trip_cost": float(round_trip_cost),
                "exit_cost_rate": float(exit_cost),
                "buckets": [],
            }
        bucket_count = adaptive_bucket_count(
            self.bucket_count, self._last_history_sessions
        )
        self._last_effective_bucket_count = bucket_count
        stats = _bucket_statistics(
            eligible,
            bucket_count,
            label_column=self.label_column,
            seed=self.seed,
            n_bootstrap=self.n_bootstrap,
            bootstrap_alpha=self.bootstrap_alpha,
            block_length=self.block_length,
            max_bootstrap_workspace_bytes=max_bootstrap_workspace_bytes,
            preserve_negative_bound_null=self.preserve_negative_bound_null,
        )
        self._last_evidence = tuple(
            sorted(
                (
                    BucketEvidence(
                        bucket=int(row["__bucket"]),
                        sample_size=int(row["sample_size"]),
                        expected_active_alpha=(
                            None
                            if row[ALPHA_COLUMN] is None
                            else float(row[ALPHA_COLUMN])
                        ),
                        alpha_lower_bound=(
                            None
                            if row[LOWER_BOUND_COLUMN] is None
                            else float(row[LOWER_BOUND_COLUMN])
                        ),
                        alpha_standard_error=(
                            None
                            if row.get(ALPHA_STANDARD_ERROR_COLUMN) is None
                            else float(row[ALPHA_STANDARD_ERROR_COLUMN])
                        ),
                    )
                    for row in stats.to_dicts()
                ),
                key=lambda evidence: evidence.bucket,
            )
        )
        return {
            "bucket_count": int(bucket_count),
            "history_sessions": int(self._last_history_sessions),
            "round_trip_cost": float(round_trip_cost),
            "exit_cost_rate": float(exit_cost),
            "buckets": [
                {
                    "bucket": int(evidence.bucket),
                    "sample_size": int(evidence.sample_size),
                    "expected_active_alpha": evidence.expected_active_alpha,
                    "alpha_lower_bound": evidence.alpha_lower_bound,
                    "alpha_standard_error": evidence.alpha_standard_error,
                }
                for evidence in self._last_evidence
            ],
        }

    @staticmethod
    def apply_prepared(
        prepared: dict[str, object],
        scored: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join the frozen prepared evidence onto a compact allocation history.

        ``prepared`` is the immutable state returned by ``prepare_decision``;
        this operation only joins that fixed bucket table and cost evidence to
        ``scored``, so it is numerically identical to the reference
        ``transform`` for the same decision timestamp. ``scored`` is expected
        to be the already-bounded allocation history (``session <=
        decision_time``) carrying ``instrument_id``, ``session``, and a score
        column. Stateless: the prepared state fully determines the output.
        """
        _validate_scored(scored)
        score_column = _resolve_score_column(scored)
        buckets = cast(list[dict[str, object]], prepared["buckets"])
        bucket_stats: pl.DataFrame | None = None
        if buckets:
            bucket_stats = pl.DataFrame(
                {
                    "__bucket": [
                        cast(int, b["bucket"]) for b in buckets
                    ],
                    ALPHA_COLUMN: [
                        b.get("expected_active_alpha") for b in buckets
                    ],
                    LOWER_BOUND_COLUMN: [
                        b.get("alpha_lower_bound") for b in buckets
                    ],
                    ALPHA_STANDARD_ERROR_COLUMN: [
                        b.get("alpha_standard_error") for b in buckets
                    ],
                }
            )
        return _augment(
            scored,
            score_column,
            cast(int, prepared["bucket_count"]),
            cast(float, prepared["round_trip_cost"]),
            cast(float, prepared["exit_cost_rate"]),
            None,
            bucket_stats=bucket_stats,
        )

    def calibration_state(self) -> dict[str, object]:
        """JSON-safe frozen calibration snapshot for artifact serialization."""
        return {
            "bucket_count": int(self._last_effective_bucket_count),
            "min_calibration_sessions": int(self.min_calibration_sessions),
            "seed": int(self.seed),
            "n_bootstrap": int(self.n_bootstrap),
            "bootstrap_alpha": float(self.bootstrap_alpha),
            "block_length": int(self.block_length),
            "participation_limit": float(self.participation_limit),
            "label_column": str(self.label_column),
            "label_available_column": str(self.label_available_column),
            "preserve_negative_bound_null": bool(self.preserve_negative_bound_null),
            "history_sessions": int(self._last_history_sessions),
            "round_trip_cost": round(float(self._last_round_trip_cost), 12),
            "exit_cost_rate": round(float(self._last_exit_cost), 12),
            "buckets": [evidence.to_json_safe() for evidence in self._last_evidence],
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> CausalAlphaCalibrator:
        """Reconstruct a calibrator that applies a frozen bucket table."""
        calibrator = cls(
            bucket_count=cast(int, state["bucket_count"]),
            min_calibration_sessions=cast(int, state["min_calibration_sessions"]),
            seed=cast(int, state["seed"]),
            n_bootstrap=cast(int, state["n_bootstrap"]),
            bootstrap_alpha=cast(float, state["bootstrap_alpha"]),
            block_length=cast(int, state["block_length"]),
            participation_limit=cast(float, state["participation_limit"]),
            label_column=cast(str, state.get("label_column", LABEL_COLUMN)),
            label_available_column=cast(
                str, state.get("label_available_column", LABEL_AVAILABLE_COLUMN)
            ),
            preserve_negative_bound_null=bool(state.get("preserve_negative_bound_null", True)),
        )
        calibrator._last_history_sessions = cast(int, state["history_sessions"])
        calibrator._last_round_trip_cost = cast(float, state["round_trip_cost"])
        calibrator._last_exit_cost = cast(float, state["exit_cost_rate"])
        calibrator._last_evidence = tuple(
            BucketEvidence(
                bucket=cast(int, row["bucket"]),
                sample_size=cast(int, row["sample_size"]),
                expected_active_alpha=(
                    None
                    if row["expected_active_alpha"] is None
                    else cast(float, row["expected_active_alpha"])
                ),
                alpha_lower_bound=(
                    None
                    if row["alpha_lower_bound"] is None
                    else cast(float, row["alpha_lower_bound"])
                ),
                alpha_standard_error=(
                    None
                    if row.get("alpha_standard_error") is None
                    else cast(float, row["alpha_standard_error"])
                ),
            )
            for row in cast(list[dict[str, object]], state["buckets"])
        )
        return calibrator

    def apply_frozen(self, scored: pl.DataFrame) -> pl.DataFrame:
        """Apply the frozen bucket table to a scored frame (prediction path)."""
        _validate_scored(scored)
        score_column = _resolve_score_column(scored)
        stats = pl.DataFrame(
            {
                "__bucket": [e.bucket for e in self._last_evidence],
                ALPHA_COLUMN: [
                    e.expected_active_alpha for e in self._last_evidence
                ],
                LOWER_BOUND_COLUMN: [
                    e.alpha_lower_bound for e in self._last_evidence
                ],
                ALPHA_STANDARD_ERROR_COLUMN: [
                    e.alpha_standard_error for e in self._last_evidence
                ],
            }
        )
        return _augment(
            scored, score_column, self._last_effective_bucket_count,
            self._last_round_trip_cost, self._last_exit_cost, None,
            bucket_stats=stats,
        )


def _validate_scored(scored: pl.DataFrame) -> None:
    required = (INSTRUMENT_COLUMN, SESSION_COLUMN)
    missing = [c for c in required if c not in scored.columns]
    if missing:
        raise ValueError(f"scored frame must carry {', '.join(missing)}")
    score_column = _resolve_score_column(scored)
    non_finite = scored.filter(
        pl.col(score_column).is_not_null() & ~pl.col(score_column).is_finite()
    )
    if not non_finite.is_empty():
        raise ValueError("non-finite score in scored frame")


def _validate_observations(
    observations: pl.DataFrame,
    label_column: str = LABEL_COLUMN,
    label_available_column: str = LABEL_AVAILABLE_COLUMN,
) -> None:
    required = (
        INSTRUMENT_COLUMN,
        SESSION_COLUMN,
        SCORE_COLUMN,
        label_column,
        label_available_column,
    )
    missing = [c for c in required if c not in observations.columns]
    if missing:
        raise ValueError(
            f"calibration observations must carry {', '.join(missing)}"
        )
    non_finite = observations.filter(
        (pl.col(SCORE_COLUMN).is_not_null() & ~pl.col(SCORE_COLUMN).is_finite())
        | (
            pl.col(label_column).is_not_null()
            & ~pl.col(label_column).is_finite()
        )
    )
    if not non_finite.is_empty():
        raise ValueError("non-finite calibration input")


def _resolve_score_column(frame: pl.DataFrame) -> str:
    if "pred_score" in frame.columns:
        return "pred_score"
    if SCORE_COLUMN in frame.columns:
        return SCORE_COLUMN
    raise ValueError("scored frame must carry pred_score or score")


def adaptive_bucket_count(nominal: int, history_sessions: int) -> int:
    """Fewer score buckets while calibration history is thin.

    Cold-start calibration splits the accumulated sessions into score buckets;
    splitting a small sample into the full nominal deciles leaves each bucket
    too thin for a stable block bootstrap. Below one year of sessions the
    effective count drops to quintiles; at or above it the nominal count is
    used.
    """
    if history_sessions < _ADAPTIVE_BUCKET_MIN_SESSIONS:
        return min(nominal, _ADAPTIVE_BUCKET_COLD_COUNT)
    return nominal


def _shrink_lower_bound(lower: float, location: float, sample_size: int) -> float:
    """Pull a thin-sample bootstrap lower bound toward its series location.

    The bootstrap ``alpha`` quantile is a pessimistic estimate whose spread
    inflates with sample variance; on thin samples it can collapse below zero
    purely from sampling noise. The deviation from the series location is
    down-weighted by ``sample_size / (sample_size + one_year_of_sessions)`` so
    thin samples stay anchored to the observed location while thick samples
    keep the full pessimistic quantile.
    """
    weight = sample_size / (sample_size + _ADAPTIVE_BUCKET_MIN_SESSIONS)
    return location + (lower - location) * weight


def _block_bootstrap_statistics(
    residuals: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    bootstrap_alpha: float,
    *,
    max_bootstrap_workspace_bytes: int | None,
) -> tuple[float, float]:
    arr = np.asarray(residuals, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    # use helper to get means via exact block bootstrap
    # reuse _block_bootstrap_lower_bound helpers but also compute SE
    n = arr.size
    block = max(int(block_length), 1)
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    rng = np.random.default_rng(int(seed))
    offsets = np.arange(block)
    if max_bootstrap_workspace_bytes is None:
        starts = rng.integers(0, max_start, size=(int(n_bootstrap), n_blocks))
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(int(n_bootstrap), n_blocks * block)[:, :n]
        means = arr[index].mean(axis=1)
    else:
        batch_draws = max_bootstrap_workspace_bytes // (n * 24) if n > 0 else int(n_bootstrap)
        batch_draws = max(1, min(int(batch_draws), int(n_bootstrap)))
        means = np.empty(int(n_bootstrap), dtype=float)
        for offset in range(0, int(n_bootstrap), batch_draws):
            stop = min(offset + batch_draws, int(n_bootstrap))
            count = stop - offset
            starts = rng.integers(0, max_start, size=(count, n_blocks))
            index = (starts[:, :, None] + offsets[None, None, :]).reshape(count, n_blocks * block)[:, :n]
            means[offset:stop] = arr[index].mean(axis=1)
    lower = float(np.quantile(means, float(bootstrap_alpha)))
    se = float(np.std(means, ddof=1)) if means.size > 1 else 0.0
    if not np.isfinite(se):
        se = 0.0
    return lower, max(0.0, se)


def _bucket_expression(score_column: str, bucket_count: int) -> pl.Expr:
    within = pl.col(score_column).count().over(SESSION_COLUMN)
    pct_rank = pl.when(within > 1).then(
        (pl.col(score_column).rank("average").over(SESSION_COLUMN) - 1.0)
        / (within - 1.0)
    ).otherwise(0.5)
    return (
        (pct_rank * bucket_count).floor().cast(pl.Int64).clip(0, bucket_count - 1)
    ).alias("__bucket")


def _bucket_statistics(
    eligible: pl.DataFrame,
    bucket_count: int,
    label_column: str = LABEL_COLUMN,
    seed: int = 42,
    n_bootstrap: int = 200,
    bootstrap_alpha: float = 0.05,
    block_length: int = _DEFAULT_BLOCK_LENGTH,
    *,
    max_bootstrap_workspace_bytes: int | None = None,
    preserve_negative_bound_null: bool = False,
) -> pl.DataFrame:
    """Per-bucket shrunk expected alpha and bootstrap lower bound.

    The whole function is vectorized except a bounded loop over the fixed bucket
    count, which is not a per-row callback. The global mean anchors the
    shrinkage so low-sample buckets pull toward the history mean rather than
    trusting a noisy bucket average. Calibration observations are ordered by
    ``(session, instrument_id)`` before bootstrap sampling so the reference
    transform and the prepared-decision schedule share one canonical order.
    A supplied ``max_bootstrap_workspace_bytes`` is propagated unchanged to the
    bootstrap helper so memory-bounded replay keeps a bounded peak workspace.
    """
    if max_bootstrap_workspace_bytes is not None and max_bootstrap_workspace_bytes <= 0:
        raise ValueError("max_bootstrap_workspace_bytes must be positive when supplied")
    eligible = eligible.sort([SESSION_COLUMN, INSTRUMENT_COLUMN])
    bucketed = eligible.with_columns(
        _bucket_expression(SCORE_COLUMN, bucket_count)
    )
    grouped = bucketed.group_by("__bucket").agg(
        pl.col(label_column).count().alias("sample_size"),
        pl.col(label_column).mean().alias("bucket_mean"),
        pl.col(label_column).implode().alias("__residuals"),
    )
    global_mean_series = bucketed.select(
        pl.col(label_column).mean()
    ).to_series()
    global_mean_value = global_mean_series[0] if not global_mean_series.is_empty() else None
    if global_mean_value is None:
        return pl.DataFrame(
            {
                "__bucket": [],
                "sample_size": [],
                ALPHA_COLUMN: [],
                LOWER_BOUND_COLUMN: [],
                ALPHA_STANDARD_ERROR_COLUMN: [],
            }
        )
    global_mean = float(global_mean_value)
    rows: list[dict[str, object]] = []
    for row in grouped.to_dicts():
        bucket = int(row["__bucket"])
        sample_size = int(row["sample_size"])
        residuals = np.asarray(row["__residuals"], dtype=float)
        if sample_size < _MIN_BUCKET_OBSERVATIONS:
            rows.append(
                {
                    "__bucket": bucket,
                    "sample_size": sample_size,
                    ALPHA_COLUMN: None,
                    LOWER_BOUND_COLUMN: None,
                    ALPHA_STANDARD_ERROR_COLUMN: None,
                }
            )
            continue
        lower_raw, se_raw = _block_bootstrap_statistics(
            residuals,
            block_length,
            n_bootstrap,
            seed + bucket,
            bootstrap_alpha,
            max_bootstrap_workspace_bytes=max_bootstrap_workspace_bytes,
        )
        lower_bound = _shrink_lower_bound(lower_raw, float(np.mean(residuals)), sample_size)
        se = float(se_raw)
        if not np.isfinite(se) or se < 0:
            se = 0.0
        shrink = min(1.0, _SHRINKAGE_THRESHOLD / sample_size)
        bucket_mean = float(row["bucket_mean"])
        shrunk = global_mean + (1.0 - shrink) * (bucket_mean - global_mean)
        if not np.isfinite(shrunk) or not np.isfinite(lower_bound):
            rows.append(
                {
                    "__bucket": bucket,
                    "sample_size": sample_size,
                    ALPHA_COLUMN: None,
                    LOWER_BOUND_COLUMN: None,
                    ALPHA_STANDARD_ERROR_COLUMN: None,
                }
            )
            continue
        if preserve_negative_bound_null and lower_bound <= 0.0:
            rows.append(
                {
                    "__bucket": bucket,
                    "sample_size": sample_size,
                    ALPHA_COLUMN: None,
                    LOWER_BOUND_COLUMN: None,
                    ALPHA_STANDARD_ERROR_COLUMN: None,
                }
            )
            continue
        rows.append(
            {
                "__bucket": bucket,
                "sample_size": sample_size,
                ALPHA_COLUMN: float(shrunk),
                LOWER_BOUND_COLUMN: float(lower_bound),
                ALPHA_STANDARD_ERROR_COLUMN: float(se),
            }
        )
    return pl.DataFrame(rows)


def _augment(
    scored: pl.DataFrame,
    score_column: str,
    bucket_count: int,
    round_trip_cost: float,
    exit_cost_rate: float,
    liquidity_rate: pl.Series | None,
    *,
    bucket_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join bucket statistics onto ``scored`` and derive net alpha columns."""
    if bucket_stats is None or bucket_stats.is_empty():
        stats = pl.DataFrame(
            {
                "__bucket": [0],
                ALPHA_COLUMN: [None],
                LOWER_BOUND_COLUMN: [None],
                ALPHA_STANDARD_ERROR_COLUMN: [None],
            }
        )
    else:
        cols = ["__bucket", ALPHA_COLUMN, LOWER_BOUND_COLUMN]
        if ALPHA_STANDARD_ERROR_COLUMN in bucket_stats.columns:
            cols.append(ALPHA_STANDARD_ERROR_COLUMN)
        stats = bucket_stats.select(*cols)
    out = scored.with_columns(_bucket_expression(score_column, bucket_count)).join(
        stats, on="__bucket", how="left"
    )
    active = pl.col(ALPHA_COLUMN)
    lower = pl.col(LOWER_BOUND_COLUMN)
    cost_expr: pl.Expr = pl.lit(round_trip_cost, dtype=pl.Float64)
    drops = ["__bucket"]
    if liquidity_rate is not None:
        cost_expr = cost_expr + pl.col("__liquidity_rate")
        out = out.with_columns(pl.Series("__liquidity_rate", liquidity_rate))
        drops.append("__liquidity_rate")
    return out.with_columns(
        (active - cost_expr).alias(NET_ALPHA_COLUMN),
        (lower - cost_expr).alias(NET_LOWER_BOUND_COLUMN),
        pl.lit(exit_cost_rate, dtype=pl.Float64).alias(EXIT_COST_COLUMN),
    ).drop(*drops)


def _round_trip_cost_rate(point: CostPoint) -> float:
    return (
        2.0 * point.commission_rate
        + point.tax_rate
        + 2.0 * point.slippage_bps / 10_000.0
    )


def _exit_cost_rate(point: CostPoint) -> float:
    return (
        point.commission_rate
        + point.tax_rate
        + point.slippage_bps / 10_000.0
    )


def _block_bootstrap_lower_bound(
    values: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    alpha: float,
    *,
    max_bootstrap_workspace_bytes: int | None = None,
    use_prefix_sum: bool = False,
) -> float:
    """Vectorized moving-block bootstrap ``alpha`` quantile of block means.

    When ``max_bootstrap_workspace_bytes`` is supplied, the draws are processed
    in deterministic contiguous batches whose conservative workspace (three
    row-sized ``int64``/``float64`` work arrays, 24 bytes per row per draw) stays
    at or below the cap. Consecutive batch shapes partition the first dimension,
    so the identical RNG stream is consumed in the identical row-major order and
    the batched means are bit-identical to the legacy one-shot means.

    ``use_prefix_sum`` selects the block-prefix-sum kernel: the identical seeded
    block starts are generated and the sampled block means are summed from a
    precomputed cumulative sum, reducing workspace from ``O(draws * rows)`` to
    ``O(draws * rows / block_length)``. The result stays within the conservative
    :func:`_bootstrap_error_bound`; when a lower bound lands inside that bound of
    the zero economic gate, the bucket is recomputed with the exact reference
    kernel so a near-gate decision is never made on prefix-sum rounding.
    """
    if max_bootstrap_workspace_bytes is not None and max_bootstrap_workspace_bytes <= 0:
        raise ValueError("max_bootstrap_workspace_bytes must be positive when supplied")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    n = arr.size
    block = max(block_length, 1)
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    if use_prefix_sum:
        estimate = float(
            np.quantile(
                _prefix_sum_block_means(
                    arr, block, n_blocks, max_start, n_bootstrap, seed,
                    max_workspace_bytes=max_bootstrap_workspace_bytes,
                ),
                alpha,
            )
        )
        if abs(estimate) <= _bootstrap_error_bound(arr):
            return _block_bootstrap_lower_bound(
                arr, block_length, n_bootstrap, seed, alpha,
                max_bootstrap_workspace_bytes=max_bootstrap_workspace_bytes,
            )
        return estimate
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    if max_bootstrap_workspace_bytes is None:
        starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(
            n_bootstrap, n_blocks * block
        )[:, :n]
        means = arr[index].mean(axis=1)
        return float(np.quantile(means, alpha))
    batch_draws = max_bootstrap_workspace_bytes // (n * 24)
    if batch_draws < 1:
        raise ValueError("bootstrap workspace cannot fit one draw")
    batch_draws = min(batch_draws, n_bootstrap)
    means = np.empty(n_bootstrap, dtype=float)
    for offset in range(0, n_bootstrap, batch_draws):
        stop = min(offset + batch_draws, n_bootstrap)
        count = stop - offset
        starts = rng.integers(0, max_start, size=(count, n_blocks))
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(
            count, n_blocks * block
        )[:, :n]
        means[offset:stop] = arr[index].mean(axis=1)
    return float(np.quantile(means, alpha))


def _prefix_sum_block_means(
    arr: np.ndarray,
    block: int,
    n_blocks: int,
    max_start: int,
    n_bootstrap: int,
    seed: int | None,
    *,
    max_workspace_bytes: int | None = None,
    starts: np.ndarray | None = None,
) -> np.ndarray:
    """Moving-block bootstrap means via a deterministic block-prefix sum.

    Each draw's sample is the concatenation of ``n_blocks`` length-``block``
    blocks starting at seeded starts; only the final block may be truncated at
    ``arr.size``. Complete block sums come from the cumulative-sum array and the
    final partial block from the truncated cumulative difference, so the mean of
    every draw equals the reference materialized mean within float rounding.
    Draws are generated in the identical seeded order and, when a workspace cap
    is supplied, processed in contiguous batches that consume the same RNG
    stream in the same row-major order. A pre-generated ``starts`` matrix may
    be supplied to share one externally seeded draw order across kernels.
    """
    csum = np.empty(arr.size + 1, dtype=np.float64)
    np.cumsum(arr, out=csum[1:])
    csum[0] = 0.0
    return _prefix_sum_means_from_csum(
        csum,
        n=arr.size,
        block=block,
        n_blocks=n_blocks,
        max_start=max_start,
        n_bootstrap=n_bootstrap,
        seed=seed,
        max_workspace_bytes=max_workspace_bytes,
        starts=starts,
    )


def _prefix_sum_means_from_csum(
    csum: np.ndarray,
    *,
    n: int,
    block: int,
    n_blocks: int,
    max_start: int,
    n_bootstrap: int,
    seed: int | None,
    max_workspace_bytes: int | None = None,
    starts: np.ndarray | None = None,
) -> np.ndarray:
    """Bootstrap means from a precomputed ``float64`` cumulative-sum array.

    ``csum[i]`` is the sum of the first ``i`` residuals, so the schedule can
    extend cumulative sums incrementally as eligible sessions arrive instead of
    re-materializing residuals per decision. The generated starts and the
    reduction order are identical to :func:`_prefix_sum_block_means`; a
    supplied ``starts`` matrix replaces the internal draw entirely.
    """
    rng = None if starts is not None else np.random.default_rng(seed)
    per_draw_bytes = n_blocks * (8 + 8)
    batch_draws = (
        n_bootstrap if max_workspace_bytes is None else max(1, max_workspace_bytes // per_draw_bytes)
    )
    batch_draws = min(batch_draws, n_bootstrap)
    # The sample is the blocks laid out contiguously and truncated at exactly
    # ``n`` elements, so the final block contributes only its surviving head:
    # ``n - (n_blocks - 1) * block`` elements from its start.
    full_blocks = np.arange(max(0, n_blocks - 1))
    last_keep = n - max(0, n_blocks - 1) * block
    means = np.empty(n_bootstrap, dtype=np.float64)
    for offset in range(0, n_bootstrap, batch_draws):
        stop = min(offset + batch_draws, n_bootstrap)
        count = stop - offset
        if starts is None:
            assert rng is not None
            block_starts = rng.integers(0, max_start, size=(count, n_blocks))
        else:
            block_starts = starts[offset:stop]
        if full_blocks.size:
            leading = block_starts[:, full_blocks]
            full_sums = csum[leading + block] - csum[leading]
            full_total = full_sums.sum(axis=1)
        else:
            full_total = 0.0
        last_sums = (
            csum[block_starts[:, -1] + last_keep] - csum[block_starts[:, -1]]
        )
        means[offset:stop] = (full_total + last_sums) / n
    return means


def _bootstrap_error_bound(arr: np.ndarray) -> float:
    """Conservative float64 rounding bound for the prefix-sum reduction.

    A prefix-sum value is the difference of two cumulative sums of at most
    ``arr.size`` elements; the block sums then combine ``ceil(n / block)`` such
    differences. The bound is intentionally conservative: ``float64`` epsilon
    times the data scale times the full reduction depth, so a near-gate value
    always falls back to the exact reference kernel.
    """
    if arr.size == 0:
        return 0.0
    return _bootstrap_error_bound_from_scale(
        float(np.max(np.abs(arr))), arr.size
    )


def _bootstrap_error_bound_from_scale(scale: float, n: int) -> float:
    """Conservative float64 rounding bound from a tracked residual scale."""
    if n <= 0:
        return 0.0
    return float(np.finfo(float).eps * max(1.0, float(scale)) * (n + 2) * 4.0)


def _liquidity_slippage_rates(
    scored: pl.DataFrame,
    liquidity_model: LiquiditySlippageModel,
    participation_limit: float,
    decision_time: datetime,
) -> pl.Series:
    """Vectorized one-way liquidity slippage (fraction) per scored row.

    Daily volatility is estimated from the scored frame's close returns; the
    impact term follows the same ADTV/participation assumption as the fills.
    """
    price_column = "close" if "close" in scored.columns else None
    if price_column is None:
        return pl.Series(
            "__liquidity_rate",
            [0.0] * scored.height,
            dtype=pl.Float64,
        )
    prices = scored[price_column].cast(pl.Float64).to_numpy()
    adtvs = (
        scored["adtv"].cast(pl.Float64).to_numpy()
        if "adtv" in scored.columns
        else np.full(scored.height, 1.0)
    )
    notional = np.maximum(adtvs * participation_limit, 1.0)
    vol_bps = _daily_vol_bps(scored).to_numpy()
    half_spread = _half_spread_bps(liquidity_model, prices, decision_time)
    impact = (
        liquidity_model.impact_coefficient
        * liquidity_model.stress_multiplier
        * vol_bps
        * np.sqrt(notional / np.maximum(adtvs, 1.0))
    )
    return pl.Series(
        "__liquidity_rate",
        (half_spread + impact) / 10_000.0,
        dtype=pl.Float64,
    )


def _daily_vol_bps(scored: pl.DataFrame) -> pl.Series:
    if "close" not in scored.columns:
        return pl.Series("__vol_bps", [0.0] * scored.height, dtype=pl.Float64)
    return (
        scored.with_columns(
            pl.col("close")
            .log()
            .diff()
            .over(INSTRUMENT_COLUMN)
            .rolling_std(window_size=_VOLATILITY_WINDOW_SESSIONS, min_samples=2)
            .over(INSTRUMENT_COLUMN)
            .fill_null(0.0)
            .alias("__vol_bps")
        )["__vol_bps"].cast(pl.Float64)
    ) * 10_000.0


def _half_spread_bps(
    liquidity_model: LiquiditySlippageModel,
    prices: np.ndarray,
    decision_time: datetime,
) -> np.ndarray:
    """Vectorized tick-based half-spread in bps for an array of prices."""
    rules = liquidity_model.tick_schedule.rules
    effective = [r for r in rules if r.effective_from <= decision_time]
    if not effective:
        raise ValueError("no liquidity tick coverage at the decision time")
    latest = max(r.effective_from for r in effective)
    bands = sorted(
        (r for r in effective if r.effective_from == latest),
        key=lambda rule: rule.lower_inclusive,
    )
    lowers = np.asarray([r.lower_inclusive for r in bands], dtype=float)
    ticks = np.asarray([r.tick for r in bands], dtype=float)
    index = np.clip(
        np.searchsorted(lowers, prices, side="right") - 1, 0, len(bands) - 1
    )
    return np.asarray(
        0.5 * ticks[index] / np.maximum(prices, 1.0) * 10_000.0,
        dtype=float,
    )
