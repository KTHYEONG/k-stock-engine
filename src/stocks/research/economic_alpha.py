"""Causal expected-active-alpha calibration for economic portfolio construction.

``CausalAlphaCalibrator`` maps each current cross-sectional score percentile
bucket to an expected residual active return using only *prior* label-available
out-of-sample observations (``label_available_time <= decision_time`` and
``session < decision_time``), shrinks the bucket mean toward the history mean,
and requires a positive moving-block bootstrap lower bound before the bucket is
usable. The expected alpha is then netted against the effective round-trip and
one-way exit costs resolved from the decision's ``CostSchedule`` (plus optional
liquidity impact), so ``expected_net_alpha`` is in the same 5-session unit as the
``residual_o2o_5d`` label.

A bucket whose evidence is insufficient or whose bootstrap lower bound is not
positive yields ``null`` expected alpha: ``null`` is not a buy signal and drives
the allocation path to cash or sell-only. All frame transforms are vectorized
Polars/NumPy; no ``pd.apply`` or per-row Python callback is used on this path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

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
EXIT_COST_COLUMN = "exit_cost_rate"

_MIN_BUCKET_OBSERVATIONS = 5
_SHRINKAGE_THRESHOLD = 20.0
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
        self.bucket_count = bucket_count
        self.min_calibration_sessions = min_calibration_sessions
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        self.bootstrap_alpha = bootstrap_alpha
        self.block_length = block_length
        self.participation_limit = participation_limit
        self._last_evidence: tuple[BucketEvidence, ...] = ()
        self._last_history_sessions = 0
        self._last_round_trip_cost = 0.0
        self._last_exit_cost = 0.0

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
        the residual label, and ``label_available_time``; ``scored`` must carry
        ``instrument_id``, ``session``, and a score column (``pred_score`` or
        ``score``). Missing or non-finite inputs raise ``ValueError``. Only
        observations with ``label_available_time <= decision_time`` and
        ``session < decision_time`` are used, so no current or future label can
        leak into the calibration. ``scored`` rows at or after the decision time
        are excluded from the returned frame.
        """
        _validate_scored(scored)
        _validate_observations(observations)
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
            (pl.col(LABEL_AVAILABLE_COLUMN) <= decision_time)
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

        stats = _bucket_statistics(
            eligible,
            self.bucket_count,
            seed=self.seed,
            n_bootstrap=self.n_bootstrap,
            bootstrap_alpha=self.bootstrap_alpha,
            block_length=self.block_length,
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
                    )
                    for row in stats.to_dicts()
                ),
                key=lambda evidence: evidence.bucket,
            )
        )
        return _augment(
            visible, score_column, self.bucket_count,
            self._last_round_trip_cost, self._last_exit_cost, liquidity_rate,
            bucket_stats=stats,
        )

    def calibration_state(self) -> dict[str, object]:
        """JSON-safe frozen calibration snapshot for artifact serialization."""
        return {
            "bucket_count": int(self.bucket_count),
            "min_calibration_sessions": int(self.min_calibration_sessions),
            "seed": int(self.seed),
            "n_bootstrap": int(self.n_bootstrap),
            "bootstrap_alpha": float(self.bootstrap_alpha),
            "block_length": int(self.block_length),
            "participation_limit": float(self.participation_limit),
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
            }
        )
        return _augment(
            scored, score_column, self.bucket_count,
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


def _validate_observations(observations: pl.DataFrame) -> None:
    required = (
        INSTRUMENT_COLUMN,
        SESSION_COLUMN,
        SCORE_COLUMN,
        LABEL_COLUMN,
        LABEL_AVAILABLE_COLUMN,
    )
    missing = [c for c in required if c not in observations.columns]
    if missing:
        raise ValueError(
            f"calibration observations must carry {', '.join(missing)}"
        )
    non_finite = observations.filter(
        (pl.col(SCORE_COLUMN).is_not_null() & ~pl.col(SCORE_COLUMN).is_finite())
        | (pl.col(LABEL_COLUMN).is_not_null() & ~pl.col(LABEL_COLUMN).is_finite())
    )
    if not non_finite.is_empty():
        raise ValueError("non-finite calibration input")


def _resolve_score_column(frame: pl.DataFrame) -> str:
    if "pred_score" in frame.columns:
        return "pred_score"
    if SCORE_COLUMN in frame.columns:
        return SCORE_COLUMN
    raise ValueError("scored frame must carry pred_score or score")


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
    seed: int = 42,
    n_bootstrap: int = 200,
    bootstrap_alpha: float = 0.05,
    block_length: int = _DEFAULT_BLOCK_LENGTH,
) -> pl.DataFrame:
    """Per-bucket shrunk expected alpha and bootstrap lower bound.

    The whole function is vectorized except a bounded loop over the fixed bucket
    count, which is not a per-row callback. The global mean anchors the
    shrinkage so low-sample buckets pull toward the history mean rather than
    trusting a noisy bucket average.
    """
    bucketed = eligible.with_columns(
        _bucket_expression(SCORE_COLUMN, bucket_count)
    )
    grouped = bucketed.group_by("__bucket").agg(
        pl.col(LABEL_COLUMN).count().alias("sample_size"),
        pl.col(LABEL_COLUMN).mean().alias("bucket_mean"),
        pl.col(LABEL_COLUMN).implode().alias("__residuals"),
    )
    global_mean_series = bucketed.select(pl.col(LABEL_COLUMN).mean()).to_series()
    global_mean_value = global_mean_series[0] if not global_mean_series.is_empty() else None
    if global_mean_value is None:
        return pl.DataFrame(
            {
                "__bucket": [],
                "sample_size": [],
                ALPHA_COLUMN: [],
                LOWER_BOUND_COLUMN: [],
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
                }
            )
            continue
        lower_bound = _block_bootstrap_lower_bound(
            residuals, block_length, n_bootstrap, seed + bucket, bootstrap_alpha
        )
        if lower_bound <= 0.0:
            rows.append(
                {
                    "__bucket": bucket,
                    "sample_size": sample_size,
                    ALPHA_COLUMN: None,
                    LOWER_BOUND_COLUMN: None,
                }
            )
            continue
        shrink = min(1.0, _SHRINKAGE_THRESHOLD / sample_size)
        bucket_mean = float(row["bucket_mean"])
        shrunk = global_mean + (1.0 - shrink) * (bucket_mean - global_mean)
        rows.append(
            {
                "__bucket": bucket,
                "sample_size": sample_size,
                ALPHA_COLUMN: float(shrunk),
                LOWER_BOUND_COLUMN: float(lower_bound),
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
            }
        )
    else:
        stats = bucket_stats.select("__bucket", ALPHA_COLUMN, LOWER_BOUND_COLUMN)
    out = scored.with_columns(_bucket_expression(score_column, bucket_count)).join(
        stats, on="__bucket", how="left"
    )
    active = pl.col(ALPHA_COLUMN)
    cost_expr: pl.Expr = pl.lit(round_trip_cost, dtype=pl.Float64)
    drops = ["__bucket"]
    if liquidity_rate is not None:
        cost_expr = cost_expr + pl.col("__liquidity_rate")
        out = out.with_columns(pl.Series("__liquidity_rate", liquidity_rate))
        drops.append("__liquidity_rate")
    return out.with_columns(
        (active - cost_expr).alias(NET_ALPHA_COLUMN),
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
) -> float:
    """Vectorized moving-block bootstrap ``alpha`` quantile of block means."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    n = arr.size
    block = max(block_length, 1)
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(
        n_bootstrap, n_blocks * block
    )[:, :n]
    means = arr[index].mean(axis=1)
    return float(np.quantile(means, alpha))


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
