"""Segment metadata and one-segment-at-a-time replay preparation.

Preparation ownership extracted from ``execution_replay.py``: this module
validates a same-cadence batch once, derives cheap per-segment
:class:`ReplaySegmentMetadata` before any material allocation, and builds at
most one live :class:`PreparedReplaySegment` at a time through
:func:`iter_prepared_replay_segments`.

Each segment window includes the exact causal ADTV/volatility lookback rows
that precede its first decision session, so first-session rolling statistics
and every fill/cost decision equal the full-history reference while emitted
evidence still begins at the declared segment start.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from src.stocks.backtesting.contracts import ArtifactSchedule, ArtifactSlot
from src.stocks.backtesting.market import PreparedReplayMarket
from src.stocks.ml.replay_resources import (
    EffectiveMemoryLimit,
    MemoryBudgetExceededError,
    estimate_replay_allocation,
)

if TYPE_CHECKING:
    from src.stocks.trading.portfolio_constructor import StockRiskPolicy

SCORE_CANDIDATES = ("predicted_net_alpha", "pred_score")

ECONOMIC_COLUMNS = (
    "expected_active_alpha",
    "alpha_lower_bound",
    "expected_net_alpha",
    "net_alpha_lower_bound",
    "exit_cost_rate",
)
ADTV_WINDOW = 20

# Conservative bytes per prepared market row beyond the aligned float64
# arrays: object keys, row instances, index mappings, and shared strings.
_PER_ROW_OVERHEAD_BYTES = 176

_VOLATILITY_GATE_COLUMNS = frozenset(
    {"data_quality_status", "is_universe", "tradable"}
)

_REQUIRED_BACKTEST_COLUMNS = (
    "instrument_id",
    "session",
    "open",
    "close",
    "volume",
    "trading_value",
)


@dataclass(frozen=True, slots=True)
class ExecutionReplayBatchRequest:
    """One same-cadence candidate group plus its resource constraints.

    ``requests`` are compatible :class:`ExecutionEquivalentReplayRequest`
    candidates sharing one market/score/segment structure; the first request's
    context and frames own preparation. ``prepared_batch`` optionally supplies
    an already-prepared legacy batch whose segments are reused without new
    builds.
    """

    requests: tuple[Any, ...]
    resource_plan: Any | None = None
    prepared_batch: Any | None = None
    request_limit_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ReplaySegmentMetadata:
    """Cheap per-segment allocation descriptor (no heavy state retained).

    ``window_sessions`` includes the causal lookback rows before the first
    decision and one trailing session for final settlement; ``decision_indices``
    address positions inside that window so they match the prepared market's
    session order exactly.
    """

    segment_id: int
    decision_sessions: tuple[datetime, ...]
    window_sessions: tuple[datetime, ...]
    lookback_session_count: int
    row_estimate: int
    estimated_prepared_bytes: int


@dataclass(frozen=True, slots=True)
class _ScoredSessionIndex:
    """Pre-indexed session boundaries for zero-scan decision slicing."""

    stops: tuple[int, ...]
    available_times: tuple[datetime, ...]
    uniform: bool

    def stop_for(self, decision_time: datetime) -> int | None:
        """Row-prefix stop for every session available at or before decision_time."""
        if not self.uniform:
            return None
        position = bisect_right(self.available_times, decision_time)
        if position == 0:
            return 0
        return int(self.stops[position - 1])


@dataclass(frozen=True, slots=True)
class PreparedReplaySegment:
    """One live prepared segment; raw ledgers are never retained here."""

    metadata: ReplaySegmentMetadata
    prepared_market: PreparedReplayMarket
    scored_market: pl.DataFrame
    session_index: _ScoredSessionIndex
    decision_times: tuple[datetime, ...]
    decision_indices: tuple[int, ...]
    dataset_hash: str
    score_column: str
    score_overlay: np.ndarray

    def release(self) -> None:
        """Drop references to the heavy prepared state immediately."""
        import contextlib

        for name in (
            "prepared_market",
            "scored_market",
            "session_index",
            "score_overlay",
        ):
            with contextlib.suppress(AttributeError):
                object.__delattr__(self, name)


def validate_market_frame(market: pl.DataFrame) -> None:
    missing = [
        column for column in _REQUIRED_BACKTEST_COLUMNS if column not in market.columns
    ]
    if missing:
        raise ValueError(f"market frame must carry {', '.join(missing)}")
    if "available_time" not in market.columns:
        raise ValueError("market frame must carry an available_time column")
    if market.is_empty():
        raise ValueError("market frame has no rows")
    available_type = market.schema.get("available_time")
    if not isinstance(available_type, pl.Datetime) or available_type.time_zone is None:
        raise ValueError("market available_time must be a timezone-aware datetime column")
    if market["session"].null_count() or market["instrument_id"].null_count():
        raise ValueError("market frame carries null session or instrument_id values")


def resolve_score_column(scores: pl.DataFrame) -> str:
    for candidate in SCORE_CANDIDATES:
        if candidate in scores.columns:
            return candidate
    raise ValueError(
        f"score frame must carry one of {', '.join(SCORE_CANDIDATES)}"
    )


def validate_scores(
    scores: pl.DataFrame, segment_column: str, score_column: str
) -> None:
    required = ("instrument_id", "session", segment_column, score_column)
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"score frame must carry {', '.join(missing)}")
    if scores.is_empty():
        raise ValueError("score frame has no rows")
    duplicate = (
        scores.group_by(["instrument_id", "session"]).len().filter(pl.col("len") > 1)
    )
    if not duplicate.is_empty():
        raise ValueError("score frame contains duplicate instrument/session keys")
    invalid = scores.filter(
        pl.col(score_column).is_null() | ~pl.col(score_column).is_finite()
    )
    if not invalid.is_empty():
        raise ValueError("score frame contains null or non-finite scores")
    if scores[segment_column].null_count():
        raise ValueError("score frame segment column carries null values")


def validate_score_market_alignment(
    market: pl.DataFrame, scores: pl.DataFrame
) -> None:
    unmatched = (
        scores.select("instrument_id", "session")
        .join(
            market.select("instrument_id", "session").unique(),
            on=["instrument_id", "session"],
            how="anti",
        )
        .limit(1)
    )
    if not unmatched.is_empty():
        row = unmatched.to_dicts()[0]
        raise ValueError(
            "score key without a market row: "
            f"{row['instrument_id']!r} at {row['session']!r}"
        )


def assert_decision_scores(
    seg_scores: pl.DataFrame,
    decisions: tuple[datetime, ...],
    segment_id: int,
) -> None:
    for decision in decisions:
        if seg_scores.filter(pl.col("session") == decision).is_empty():
            raise ValueError(
                f"absent score at declared decision {decision.isoformat()} "
                f"(segment {segment_id})"
            )


def decision_times_of(
    segment_ordered: pl.DataFrame,
    decisions: tuple[datetime, ...],
) -> tuple[datetime, ...]:
    decision_rows = segment_ordered.filter(pl.col("session").is_in(decisions))
    times: dict[datetime, datetime] = {}
    for session, available in zip(
        decision_rows["session"].to_list(),
        decision_rows["available_time"].to_list(),
        strict=True,
    ):
        if available is None:
            continue
        if session not in times or available > times[session]:
            times[session] = available
    result: list[datetime] = []
    for decision in decisions:
        value = times.get(decision)
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(
                f"missing timezone-aware available_time at decision {decision.isoformat()}"
            )
        result.append(value)
    return tuple(result)


def frame_hash(frame: pl.DataFrame) -> str:
    ordered = frame.select(sorted(frame.columns)).sort(["instrument_id", "session"])
    return hashlib.sha256(ordered.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()


def scored_session_index(scored_market: pl.DataFrame) -> _ScoredSessionIndex:
    """Build contiguous session row stops from the session-sorted scored panel."""
    agg = (
        scored_market.group_by("session")
        .agg(
            pl.len().alias("__rows"),
            pl.col("available_time").max().alias("__available"),
            pl.col("available_time").n_unique().alias("__variants"),
        )
        .sort("session")
    )
    stops: list[int] = []
    running = 0
    for rows in agg["__rows"].to_list():
        running += int(rows)
        stops.append(running)
    available = tuple(agg["__available"].to_list())
    uniform = bool((agg["__variants"] == 1).all())
    monotonic = all(
        previous <= current for previous, current in pairwise(available)
    )
    return _ScoredSessionIndex(
        stops=tuple(stops),
        available_times=available,
        uniform=uniform and monotonic,
    )


def can_precompute_volatility(scored_market: pl.DataFrame) -> bool:
    """Slicing/rolling precompute is exact only when no row-gate filter applies."""
    return not _VOLATILITY_GATE_COLUMNS.intersection(scored_market.columns)


def precompute_volatility(frame: pl.DataFrame, window: int) -> pl.DataFrame:
    """Causally precompute the constructor's ``__vol`` column once per segment."""
    if "log_return" in frame.columns:
        returns = pl.col("log_return")
    elif "ret" in frame.columns:
        returns = pl.col("ret").log1p()
    else:
        returns = (
            pl.col("close").log() - pl.col("close").log().shift(1).over("instrument_id")
        )
    return frame.with_columns(
        returns.rolling_std(window_size=window, min_samples=2)
        .over("instrument_id")
        .alias("__vol")
    )


def volatility_lookback_sessions(risk_policy: StockRiskPolicy) -> int:
    lookback = getattr(risk_policy, "volatility_lookback_sessions", ADTV_WINDOW)
    try:
        value = int(lookback)
    except (TypeError, ValueError):
        return ADTV_WINDOW
    return max(1, value)


def _estimate_segment_bytes(row_estimate: int, width: int) -> int:
    per_row = width * 8 + _PER_ROW_OVERHEAD_BYTES + 8
    return max(int(row_estimate) * per_row, 64)


def iter_replay_segment_metadata(
    primary_request: Any,
    *,
    risk_policy: StockRiskPolicy | None = None,
) -> list[ReplaySegmentMetadata]:
    """Derive cheap per-segment descriptors without preparing any segment.

    Validates the shared batch inputs first so an invalid request fails closed
    before any segment window is computed.
    """
    market = primary_request.market_frame
    scores = primary_request.score_frame
    if primary_request.horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive session count")
    validate_market_frame(market)
    score_column = resolve_score_column(scores)
    validate_scores(scores, primary_request.segment_column, score_column)
    validate_score_market_alignment(market, scores)

    ordered_market = market.sort(["session", "instrument_id"])
    session_to_index = {
        session: index
        for index, session in enumerate(
            ordered_market["session"].unique().sort().to_list()
        )
    }
    full_sessions = sorted(session_to_index)
    lookback_window = max(
        ADTV_WINDOW,
        volatility_lookback_sessions(
            risk_policy or primary_request.context.risk_policy
        ),
    )

    metadata: list[ReplaySegmentMetadata] = []
    for segment_id in sorted(primary_request.decision_sessions_by_segment):
        decisions = primary_request.decision_sessions_by_segment[segment_id]
        if not decisions:
            raise ValueError(f"segment {segment_id} has no declared decision sessions")
        seg_scores = scores.filter(pl.col(primary_request.segment_column) == segment_id)
        if seg_scores.is_empty():
            raise ValueError(f"segment {segment_id} has no score rows")
        assert_decision_scores(seg_scores, decisions, segment_id)

        first_index = session_to_index.get(min(decisions))
        last_index = session_to_index.get(max(decisions))
        if first_index is None or last_index is None:
            raise ValueError(
                f"segment {segment_id} declares a decision outside the market window"
            )
        start_index = max(0, first_index - lookback_window)
        stop_index = min(last_index + 2, len(full_sessions))
        window_sessions = tuple(full_sessions[start_index:stop_index])
        row_estimate = int(
            ordered_market.filter(pl.col("session").is_in(window_sessions)).height
        )
        metadata.append(
            ReplaySegmentMetadata(
                segment_id=int(segment_id),
                decision_sessions=tuple(decisions),
                window_sessions=window_sessions,
                lookback_session_count=first_index - start_index,
                row_estimate=row_estimate,
                estimated_prepared_bytes=_estimate_segment_bytes(
                    row_estimate, len(market.columns)
                ),
            )
        )
    return metadata


def build_prepared_replay_segment(
    primary_request: Any,
    metadata: ReplaySegmentMetadata,
) -> PreparedReplaySegment:
    """Prepare exactly one segment from its metadata window.

    The prepared market includes the causal lookback rows preceding the first
    declared decision; evidence consumers trim ledger rows before the segment
    start so emitted series begin at the declared boundary.
    """
    context = primary_request.context
    market = primary_request.market_frame
    scores = primary_request.score_frame
    score_column = resolve_score_column(scores)
    ordered_market = market.sort(["session", "instrument_id"])
    seg_scores = scores.filter(
        pl.col(primary_request.segment_column) == metadata.segment_id
    )

    decision_indices = tuple(
        index
        for index, session in enumerate(metadata.window_sessions)
        if session in set(metadata.decision_sessions)
    )
    segment_ordered = ordered_market.filter(
        pl.col("session").is_in(metadata.window_sessions)
    )
    decision_times = decision_times_of(segment_ordered, metadata.decision_sessions)
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=min(decision_times),
                eligible_to=max(decision_times),
                artifact_id=context.artifact_id,
            ),
        )
    )
    prepared_market = PreparedReplayMarket.build(
        segment_ordered,
        ADTV_WINDOW,
        instruments=context.instruments,
        artifacts=artifacts,
        initial_portfolio=context.initial_portfolio,
    )

    score_cols = [score_column] + [
        column for column in ECONOMIC_COLUMNS if column in seg_scores.columns
    ]
    scored_market = segment_ordered.join(
        seg_scores.select(["instrument_id", "session", *score_cols]),
        on=["instrument_id", "session"],
        how="left",
    ).with_columns(pl.Series("adtv", prepared_market.adtv))
    if can_precompute_volatility(scored_market):
        scored_market = precompute_volatility(
            scored_market, context.risk_policy.volatility_lookback_sessions
        )
    session_index = scored_session_index(scored_market)
    dataset_hash = frame_hash(segment_ordered)

    score_overlay = scored_market[score_column].to_numpy().astype(np.float64)
    score_overlay.setflags(write=False)
    return PreparedReplaySegment(
        metadata=metadata,
        prepared_market=prepared_market,
        scored_market=scored_market,
        session_index=session_index,
        decision_times=decision_times,
        decision_indices=decision_indices,
        dataset_hash=dataset_hash,
        score_column=score_column,
        score_overlay=score_overlay,
    )


def _current_live_bytes() -> int:
    from src.stocks.ml.telemetry import current_rss_mib

    rss = current_rss_mib()
    return int(rss * 1024 * 1024) if rss is not None else 0


def iter_prepared_replay_segments(
    request: Any,
    limit: EffectiveMemoryLimit | None,
) -> Iterator[PreparedReplaySegment]:
    """Yield one prepared segment at a time under the effective memory limit.

    Before each segment build the planner verifies
    ``current_live + planned + largest_next <= effective_limit`` and raises
    :class:`MemoryBudgetExceededError` fail-closed when the invariant cannot
    hold, so no material allocation begins on a breached budget.
    """
    primary = request.requests[0] if request.requests else request.prepared_batch
    if primary is None:
        raise ValueError("batch requires at least one request")
    metadata_list = iter_replay_segment_metadata(primary)
    effective_limit_bytes = limit.effective_limit_bytes if limit is not None else None
    workers = getattr(getattr(request, "resource_plan", None), "max_workers", 1) or 1
    for position, metadata in enumerate(metadata_list):
        next_metadata = (
            metadata_list[position + 1]
            if position + 1 < len(metadata_list)
            else None
        )
        plan = estimate_replay_allocation(
            metadata,
            candidate_count=len(request.requests),
            worker_count=max(1, int(workers)),
            current_live_bytes=_current_live_bytes(),
            next_metadata=next_metadata,
            effective_limit_bytes=effective_limit_bytes,
        )
        if not plan.ok:
            raise MemoryBudgetExceededError(
                plan.reason, planned_bytes=plan.projected_total_bytes(),
                limit_bytes=int(effective_limit_bytes or 0),
            )
        yield build_prepared_replay_segment(primary, metadata)


def stream_plan_from(
    requests: Sequence[Any],
    resource_plan: Any | None,
) -> ExecutionReplayBatchRequest:
    """Bundle legacy positional streaming arguments into the batch contract."""
    return ExecutionReplayBatchRequest(
        requests=tuple(requests),
        resource_plan=resource_plan,
    )
