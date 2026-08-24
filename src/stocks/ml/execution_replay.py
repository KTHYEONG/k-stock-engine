"""Execution-equivalent ML evidence: OOF scores replayed through StockBacktester.

The adapter decouples horizon/profile economics and forward-holdout
certification from the residual-vintage proxy. It injects already
causal-calibrated OOF (or frozen-holdout) scores into the same prepared
``StockBacktester`` the operational path uses, so selection evidence is the
actual long-only equity under next-open fills, partial fills, capacity, T+2
settlement, and the base/stress cost schedules.

One immutable ``PreparedReplayMarket`` is built per OOF segment; every segment
starts from the context's initial portfolio and cannot carry positions, pending
orders, settlement, or backtester state across its boundary. ``horizon_sessions``
only bounds the segment decision window; it never drives portfolio capital lock,
target holding period, or rebalance cadence.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import cast

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    BacktestLedgerRow,
    BacktestRequest,
    BacktestResult,
    PreparedReplayDecision,
    PreparedReplayMarket,
    ReplayDecisionProvider,
    ReplayScenarioPlanner,
    StockBacktester,
)
from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
from src.stocks.ml.replay_aggregation import (
    CandidateReplayAccumulator,
    SegmentExecutionSummary,
)
from src.stocks.ml.replay_preparation import (
    ExecutionReplayBatchRequest,
    PreparedReplaySegment,
    iter_prepared_replay_segments,
    iter_replay_segment_metadata,
)
from src.stocks.ml.replay_preparation import (
    _ScoredSessionIndex as _ScoredSessionIndex,
)
from src.stocks.ml.replay_preparation import (
    build_prepared_replay_segment as _build_prepared_replay_segment,
)
from src.stocks.ml.replay_preparation import (
    resolve_score_column as _resolve_score_column,
)
from src.stocks.ml.replay_resources import (
    resolve_effective_memory_limit,
)
from src.stocks.ml.telemetry import current_rss_mib as _telemetry_current_rss_mib
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import (
    PreparedAllocationMarket,
    StockRiskPolicy,
)
from src.stocks.workflows.trading_cycle import (
    TradingCycleRequest,
    TradingCycleResult,
    plan_prepared_scored_cycle,
)


@dataclass(frozen=True, slots=True)
class ReplayResourcePlan:
    """Bounded resource plan for streaming execution replay.

    ``max_workers`` is in [1, requested_workers] when projected peak fits
    the budget; 0 means resource-budget-unavailable and no replay starts.
    ``max_prepared_segments`` is at most 1.  ``projected_peak_bytes`` is
    the estimated peak RSS for the planned worker count.
    """

    max_workers: int
    max_prepared_segments: int
    projected_peak_bytes: int


def plan_execution_replay_resources(
    available_bytes: int,
    prepared_segment_bytes: int,
    requested_workers: int,
) -> ReplayResourcePlan:
    """Select a safe worker count from measured segment bytes and headroom.

    If a safe worker cannot be scheduled (available < one segment + fixed
    reserve), returns a plan with max_workers=0 so the caller publishes
    resource-budget-unavailable before any replay starts.
    """
    fixed_reserve = 50 * 1024 * 1024  # 50 MiB for Python runtime overhead
    if available_bytes <= 0 or prepared_segment_bytes <= 0:
        return ReplayResourcePlan(
            max_workers=0, max_prepared_segments=0, projected_peak_bytes=0
        )
    per_worker_budget = prepared_segment_bytes + fixed_reserve
    if available_bytes < per_worker_budget:
        return ReplayResourcePlan(
            max_workers=0, max_prepared_segments=0, projected_peak_bytes=0
        )
    max_safe = max(1, available_bytes // per_worker_budget)
    workers = min(max(1, requested_workers), max_safe)
    return ReplayResourcePlan(
        max_workers=workers,
        max_prepared_segments=1,
        projected_peak_bytes=workers * prepared_segment_bytes,
    )


@dataclass(frozen=True, slots=True)
class ProfileReplayEvidence:
    """Named immutable replay evidence: candidate owns base/stress; dense_shadow is optional.

    ``candidate`` contains both base and stress log-growth paths from one
    policy. ``dense_shadow`` is an optional paired control used only for
    incremental-growth and turnover-reduction evidence; it is never read as a
    stress path for certification.
    """

    candidate: ExecutionReplayEvidence
    dense_shadow: ExecutionReplayEvidence | None = None

    def __iter__(self) -> Iterator[ExecutionReplayEvidence]:
        """Preserve the historical two-value unpacking API for callers."""
        yield self.candidate
        yield self.dense_shadow or self.candidate


@dataclass(frozen=True, slots=True)
class PreparedExecutionReplaySegment:
    """Immutable prepared segment: one PreparedReplayMarket per OOF segment.

    Built once per cadence group and reused across all compatible replay
    requests within that group. Contains the ordered market, score join, ADTV
    array, session index, decision times, and the PreparedReplayMarket.
    """

    segment_id: int
    prepared_market: PreparedReplayMarket
    scored_market: pl.DataFrame
    session_index: _ScoredSessionIndex
    decision_times: tuple[datetime, ...]
    decision_indices: tuple[int, ...]
    dataset_hash: str
    score_column: str
    score_overlay: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedExecutionReplayBatch:
    """Immutable prepared batch: one per same-cadence request group.

    Materializes each segment's ordered market, score join, ADTV/volatility
    arrays, session index, decision times, and PreparedReplayMarket exactly
    once. Compatible requests share this read-only batch for their backtest
    execution; each request retains its own StockBacktester, policy, seed,
    and mutable state.
    """

    market_frame: pl.DataFrame
    score_frame: pl.DataFrame
    segment_column: str
    decision_sessions_by_segment: Mapping[int, tuple[datetime, ...]]
    segment_data: Mapping[int, PreparedExecutionReplaySegment]
    context_static_signature: tuple[object, ...]
    score_column: str
    execution_replay_count: int = 0
    prepared_segment_build_count: int = 0
    prepared_cache_bytes: int = 0
    replay_prepare_elapsed_ms: int = 0
    replay_execute_elapsed_ms: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionReplayContext:
    """Immutable full configuration of one execution-equivalent replay."""

    registry: ModelArtifactRegistry
    manifest: DatasetManifest
    instruments: Mapping[str, Instrument]
    artifact_id: str
    strategy_id: str
    initial_portfolio: PortfolioSnapshot
    risk_policy: StockRiskPolicy
    base_cost_schedule: CostSchedule
    stress_cost_schedule: CostSchedule
    liquidity_model: LiquiditySlippageModel | None
    stress_liquidity_model: LiquiditySlippageModel | None
    execution_policy: ExecutionOutcomePolicy
    seed: int


@dataclass(frozen=True, slots=True)
class ExecutionEquivalentReplayRequest:
    """One OOF (or frozen-holdout) execution-equivalent replay request.

    ``market_frame`` is the raw executable panel carrying
    ``REQUIRED_BACKTEST_COLUMNS`` and a timezone-aware ``available_time``.
    ``score_frame`` is the causal-calibrated score panel with a unique
    ``(instrument_id, session)`` key, the declared ``segment_column``, and the
    pre-calibrated production score/economic columns. ``horizon_sessions`` only
    bounds the bootstrap block floor; it is never fed into allocation math.
    """

    context: ExecutionReplayContext
    market_frame: pl.DataFrame
    score_frame: pl.DataFrame
    segment_column: str
    decision_sessions_by_segment: Mapping[int, tuple[datetime, ...]]
    horizon_sessions: int


@dataclass(frozen=True, slots=True)
class ExecutionReplayEvidence:
    """Parallel base/stress daily equity log-growth plus bounded execution sums.

    Coverage is exposure-based: ``invested_interval_count`` counts the complete
    ledger return intervals whose prior ledger row carried a positive
    ``positions_value`` (a held position), regardless of whether any order was
    placed in that interval. ``observed_interval_count`` is the total number of
    complete return intervals. ``invested_interval_fraction`` is their ratio and
    is the canonical admission coverage metric; ``cash_session_fraction`` remains
    a separate, backward-compatible telemetry value derived from filled
    decision cycles. ``filled_cycle_count`` counts the planned decision cycles
    that executed at least one fill. Every per-segment aggregate must sum to the
    published totals (segment-local integrity is verified by the caller).
    """

    base_log_growth: tuple[float, ...]
    stress_log_growth: tuple[float, ...]
    segment_ids: tuple[int, ...]
    planned_cycles: int
    filled_orders: int
    cash_session_fraction: float
    turnover: float
    observed_interval_count: int = 0
    invested_interval_count: int = 0
    invested_interval_fraction: float = 0.0
    filled_cycle_count: int = 0
    unfilled_order_reason_counts: tuple[tuple[str, int], ...] = ()
    utility_transition_diagnostics: tuple[tuple[str, float | int], ...] = ()
    action_diagnostics: tuple[tuple[str, float | int], ...] = ()
    base_cost_drag: float = 0.0
    stress_cost_drag: float = 0.0
    base_exposure: float = 0.0
    stress_exposure: float = 0.0
    base_interval_exposure: tuple[float, ...] = ()
    stress_interval_exposure: tuple[float, ...] = ()
    base_interval_session_bounds: tuple[tuple[datetime, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.base_log_growth) != len(self.stress_log_growth):
            raise ValueError("base and stress log growth series must be parallel")
        if len(self.base_log_growth) != len(self.segment_ids):
            raise ValueError("log growth series and segment ids must be parallel")
        if not np.all(np.isfinite(self.base_log_growth)) or not np.all(
            np.isfinite(self.stress_log_growth)
        ):
            raise ValueError("execution log growth must be finite")
        if not 0.0 <= self.cash_session_fraction <= 1.0:
            raise ValueError("cash_session_fraction must be in [0, 1]")
        if not np.isfinite(self.turnover) or self.turnover < 0.0:
            raise ValueError("turnover must be a finite non-negative value")
        if self.observed_interval_count < 0:
            raise ValueError("observed_interval_count must be non-negative")
        if self.invested_interval_count < 0:
            raise ValueError("invested_interval_count must be non-negative")
        if self.invested_interval_count > self.observed_interval_count:
            raise ValueError(
                "invested_interval_count cannot exceed observed_interval_count"
            )
        if not 0.0 <= self.invested_interval_fraction <= 1.0:
            raise ValueError("invested_interval_fraction must be in [0, 1]")
        if self.filled_cycle_count < 0:
            raise ValueError("filled_cycle_count must be non-negative")
        if not np.isfinite(self.base_cost_drag) or self.base_cost_drag < 0.0:
            raise ValueError("base_cost_drag must be a finite non-negative value")
        if not np.isfinite(self.stress_cost_drag) or self.stress_cost_drag < 0.0:
            raise ValueError("stress_cost_drag must be a finite non-negative value")
        if not np.isfinite(self.base_exposure) or self.base_exposure < 0.0:
            raise ValueError("base_exposure must be a finite non-negative value")
        if not np.isfinite(self.stress_exposure) or self.stress_exposure < 0.0:
            raise ValueError("stress_exposure must be a finite non-negative value")
        n = len(self.base_log_growth)
        if self.base_interval_exposure:
            if len(self.base_interval_exposure) != n:
                raise ValueError("base_interval_exposure must be parallel to base_log_growth")
            for exp in self.base_interval_exposure:
                if not np.isfinite(exp) or not 0.0 <= exp <= 1.0:
                    raise ValueError("base_interval_exposure values must be finite in [0,1]")
        if self.stress_interval_exposure:
            if len(self.stress_interval_exposure) != n:
                raise ValueError("stress_interval_exposure must be parallel to stress_log_growth")
            for exp in self.stress_interval_exposure:
                if not np.isfinite(exp) or not 0.0 <= exp <= 1.0:
                    raise ValueError("stress_interval_exposure values must be finite in [0,1]")
        if self.base_interval_session_bounds:
            bounded = sum(max(0, len(bounds) - 1) for bounds in self.base_interval_session_bounds)
            if bounded != n:
                raise ValueError(
                    "base_interval_session_bounds must partition base_log_growth intervals"
                )

    def diagnostics(self) -> dict[str, object]:
        """Bounded execution evidence projection; never raw score/price vectors."""
        return {
            "planned_cycles": int(self.planned_cycles),
            "filled_orders": int(self.filled_orders),
            "filled_cycle_count": int(self.filled_cycle_count),
            "cash_session_fraction": round(float(self.cash_session_fraction), 12),
            "turnover": round(float(self.turnover), 12),
            "observed_interval_count": int(self.observed_interval_count),
            "invested_interval_count": int(self.invested_interval_count),
            "invested_interval_fraction": round(float(self.invested_interval_fraction), 12),
            "unfilled_order_reason_counts": {
                str(reason): int(count) for reason, count in self.unfilled_order_reason_counts
            },
            "action_diagnostics": {
                str(key): value for key, value in self.action_diagnostics
            },
            "base_cost_drag": round(float(self.base_cost_drag), 12),
            "stress_cost_drag": round(float(self.stress_cost_drag), 12),
            "base_exposure": round(float(self.base_exposure), 12),
            "stress_exposure": round(float(self.stress_exposure), 12),
        }


def replay_execution_equivalent(
    request: ExecutionEquivalentReplayRequest,
) -> ExecutionReplayEvidence:
    """Replay calibrated scores through the prepared engine, segment by segment.

    Each segment builds exactly one ``PreparedReplayMarket`` and starts from the
    context's initial portfolio with fresh pending orders, settlement, and
    backtester state. Score keys are aligned vectorized to the market rows
    (``NaN`` only on unscored market rows); duplicate/unmatched/non-finite score
    keys, missing executable columns, naive timestamps, or an absent score at a
    declared decision fail closed with ``ValueError``.
    """
    return replay_execution_equivalent_batch((request,))[0]


def stream_execution_replay_batch(
    requests: ExecutionReplayBatchRequest
    | Sequence[ExecutionEquivalentReplayRequest],
    resource_plan: ReplayResourcePlan | None = None,
    *,
    prepared_batch: PreparedExecutionReplayBatch | None = None,
    request_limit_bytes: int | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[ExecutionReplayEvidence, ...]:
    """Segment-major streaming replay under the effective memory limit.

    Prepares exactly one segment at a time (including its causal lookback
    rows), executes every candidate against that live segment, appends compact
    evidence into per-candidate accumulators, and releases raw ledgers and the
    prepared segment before the next boundary. Evidence is finalized in the
    declared candidate order regardless of worker count. ``stats`` optionally
    receives disjoint replay telemetry: actual build counts, deduplicated
    prepared cache bytes, peak live prepared segments, and prepare/execute
    elapsed milliseconds measured by independent timers.
    """
    import time as _time

    batch_request = (
        requests
        if isinstance(requests, ExecutionReplayBatchRequest)
        else ExecutionReplayBatchRequest(
            requests=tuple(requests),
            resource_plan=resource_plan,
            prepared_batch=prepared_batch,
            request_limit_bytes=request_limit_bytes,
        )
    )
    candidate_requests = tuple(batch_request.requests)
    stats_out = stats if stats is not None else {}
    if not candidate_requests:
        return ()
    plan = batch_request.resource_plan or resource_plan
    if plan is not None and plan.max_workers == 0:
        return ()

    accumulators = [CandidateReplayAccumulator() for _ in candidate_requests]
    build_count = 0
    cache_bytes_total = 0
    peak_live_prepared_segments = 0
    prepare_elapsed_ms = 0
    execute_elapsed_ms = 0

    supplied_batch = batch_request.prepared_batch or prepared_batch
    if supplied_batch is not None:
        for candidate_request in candidate_requests:
            _validate_batch_request_compatibility(candidate_request, supplied_batch)
        segments: list[PreparedExecutionReplaySegment | PreparedReplaySegment] = [
            supplied_batch.segment_data[segment_id]
            for segment_id in sorted(supplied_batch.segment_data)
        ]
        build_count = int(supplied_batch.prepared_segment_build_count)
        cache_bytes_total = sum(
            segment.prepared_market.cache_bytes for segment in segments
        )
        for segment in segments:
            segment_start_ms = _time.monotonic()
            _run_candidates_on_segment(
                candidate_requests, segment, accumulators,
                decisions_by_segment=candidate_requests[0].decision_sessions_by_segment,
            )
            execute_elapsed_ms += int((_time.monotonic() - segment_start_ms) * 1000)
            peak_live_prepared_segments = max(peak_live_prepared_segments, 1)
    else:
        limit = resolve_effective_memory_limit(batch_request.request_limit_bytes)
        workers = max(1, int(getattr(plan, "max_workers", 1) or 1))
        del workers  # Phase 7 keeps replay pinned to one worker until benchmarked.
        segments_iter = iter_prepared_replay_segments(batch_request, limit)
        while True:
            build_started_ms = _time.monotonic()
            try:
                # The iterator plans each boundary against the effective limit
                # and raises fail-closed before any breached allocation.
                segment = next(segments_iter)
            except StopIteration:
                break
            prepare_elapsed_ms += int((_time.monotonic() - build_started_ms) * 1000)
            build_count += 1
            cache_bytes_total += segment.prepared_market.cache_bytes
            peak_live_prepared_segments = max(peak_live_prepared_segments, 1)
            execute_started_ms = _time.monotonic()
            try:
                _run_candidates_on_segment(
                    candidate_requests, segment, accumulators,
                    decisions_by_segment={
                        segment.metadata.segment_id: segment.metadata.decision_sessions,
                    },
                )
            finally:
                execute_elapsed_ms += int((_time.monotonic() - execute_started_ms) * 1000)
                segment.release()
                del segment

    stats_out.update(
        {
            "execution_replay_count": len(candidate_requests),
            "prepared_segment_build_count": build_count,
            "prepared_cache_bytes": cache_bytes_total,
            "peak_live_prepared_segments": peak_live_prepared_segments,
            "replay_prepare_elapsed_ms": prepare_elapsed_ms,
            "replay_execute_elapsed_ms": execute_elapsed_ms,
        }
    )
    return tuple(
        cast(ExecutionReplayEvidence, accumulator.finalize())
        for accumulator in accumulators
    )


def _streaming_current_live_bytes() -> int:
    rss_mib = _telemetry_current_rss_mib()
    return int(rss_mib * 1024 * 1024) if rss_mib is not None else 0


def _run_candidates_on_segment(
    candidate_requests: Sequence[ExecutionEquivalentReplayRequest],
    segment: PreparedExecutionReplaySegment | PreparedReplaySegment,
    accumulators: list[CandidateReplayAccumulator],
    *,
    decisions_by_segment: Mapping[int, tuple[datetime, ...]],
) -> None:
    """Execute every candidate against one live prepared segment."""
    segment_id = (
        segment.metadata.segment_id
        if isinstance(segment, PreparedReplaySegment)
        else segment.segment_id
    )
    decisions = decisions_by_segment[int(segment_id)]
    for candidate_request, accumulator in zip(candidate_requests, accumulators, strict=True):
        summary = _execute_candidate_segment(
            candidate_request.context, segment, int(segment_id), decisions
        )
        accumulator.add_segment(summary)


def _execute_candidate_segment(
    context: ExecutionReplayContext,
    segment: PreparedExecutionReplaySegment | PreparedReplaySegment,
    segment_id: int,
    decisions: tuple[datetime, ...],
) -> SegmentExecutionSummary:
    """Run one candidate's paired base/stress backtest over one live segment.

    Returns bounded per-segment scalars only: raw ``BacktestResult`` ledgers
    and trades are released before returning so no candidate retains segment
    state.
    """
    decision_times = segment.decision_times
    bt_request = BacktestRequest(
        strategy_id=context.strategy_id,
        start_time=min(decision_times),
        end_time=_window_end(min(decision_times), max(decision_times)),
        decision_session_indices=tuple(segment.decision_indices),
        cost_schedule=context.base_cost_schedule,
        stress_cost_schedule=context.stress_cost_schedule,
        risk_policy=context.risk_policy,
        seed=context.seed,
    )
    backtester = StockBacktester(
        registry=context.registry,
        instruments=context.instruments,
        manifest=context.manifest,
        cost_schedule=context.base_cost_schedule,
        stress_cost_schedule=context.stress_cost_schedule,
        seed=context.seed,
        policy=context.execution_policy,
        base_liquidity_model=context.liquidity_model,
        stress_liquidity_model=context.stress_liquidity_model,
        decision_provider=_decision_provider(
            segment.scored_market,
            segment.session_index,
            PreparedAllocationMarket.build(segment.scored_market),
            segment.score_overlay,
        ),
        scenario_planner=_scenario_planner(context, segment.dataset_hash),
    )
    result = backtester.run_prepared(bt_request, segment.prepared_market, segment.score_overlay)

    segment_start = min(decisions)
    ledger = tuple(row for row in result.ledger if row.session >= segment_start)
    stress_ledger = tuple(row for row in result.stress_ledger if row.session >= segment_start)

    segment_growth, segment_invested = _ledger_growth_and_exposure(ledger)
    stress_segment_growth, _ = _ledger_growth_and_exposure(stress_ledger)
    _, seg_base_exp = _ledger_growth_and_interval_exposure(ledger)
    _, seg_stress_exp = _ledger_growth_and_interval_exposure(stress_ledger)
    interval_session_bounds = tuple(row.session for row in ledger)
    segment_filled_cycles = _filled_sessions(result)
    stress_metrics = result.stress_metrics or {}
    unfilled = {
        str(reason): int(count)
        for reason, count in result.unfilled_order_reason_counts.items()
    }
    return SegmentExecutionSummary(
        segment_id=int(segment_id),
        base_growth=segment_growth,
        stress_growth=stress_segment_growth,
        base_interval_exposure=seg_base_exp,
        stress_interval_exposure=seg_stress_exp,
        interval_session_bounds=interval_session_bounds,
        planned_cycles=int(result.planned_cycles),
        filled_orders=int(result.filled_orders),
        filled_sessions=segment_filled_cycles,
        invested_intervals=int(segment_invested),
        filled_cycles=segment_filled_cycles,
        turnover=float(result.metrics.get("turnover", 0.0)),
        base_cost_drag=float(result.metrics.get("cost_drag", 0.0)),
        stress_cost_drag=float(stress_metrics.get("cost_drag", 0.0)),
        base_exposure=float(result.metrics.get("exposure", 0.0)),
        stress_exposure=float(stress_metrics.get("exposure", 0.0)),
        unfilled_reason_counts=unfilled,
    )


def prepare_execution_replay_batch(
    request: ExecutionEquivalentReplayRequest,
) -> PreparedExecutionReplayBatch:
    """Build immutable prepared-segment batch for one same-cadence request group.

    Performs market/score validation once and materializes each segment's
    ordered market (including its causal lookback rows), score join,
    ADTV/volatility arrays, session index, decision times, and
    PreparedReplayMarket. The resulting batch is read-only and reusable across
    all compatible replay requests; streaming callers should prefer
    :func:`stream_execution_replay_batch`, which keeps at most one of these
    segments live.
    """
    context = request.context
    market = request.market_frame
    scores = request.score_frame
    if request.horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive session count")
    metadata_list = iter_replay_segment_metadata(request)

    segment_data: dict[int, PreparedExecutionReplaySegment] = {}
    score_column = _resolve_score_column(scores)
    for metadata in metadata_list:
        segment = _build_prepared_replay_segment(request, metadata)
        segment_data[metadata.segment_id] = PreparedExecutionReplaySegment(
            segment_id=metadata.segment_id,
            prepared_market=segment.prepared_market,
            scored_market=segment.scored_market,
            session_index=segment.session_index,
            decision_times=segment.decision_times,
            decision_indices=segment.decision_indices,
            dataset_hash=segment.dataset_hash,
            score_column=score_column,
            score_overlay=segment.score_overlay,
        )
        del segment

    return PreparedExecutionReplayBatch(
        market_frame=market,
        score_frame=scores,
        segment_column=request.segment_column,
        decision_sessions_by_segment={
            int(segment_id): tuple(decisions)
            for segment_id, decisions in request.decision_sessions_by_segment.items()
        },
        segment_data=segment_data,
        context_static_signature=_context_static_signature(context),
        score_column=score_column,
    )


def replay_execution_equivalent_batch(
    requests: Sequence[ExecutionEquivalentReplayRequest],
    *,
    prepared_batch: PreparedExecutionReplayBatch | None = None,
    max_workers: int = 1,
) -> tuple[ExecutionReplayEvidence, ...]:
    """Batch replay of compatible same-cadence requests sharing one prepared batch.

    Validates that all requests share the same market/score/segment structure,
    builds one prepared batch, then executes each request's backtest against
    the shared read-only segment inputs. Each request retains its own
    StockBacktester, policy, seed, and mutable state.

    ``max_workers`` bounds the parallel worker count; each worker owns an
    independent ``StockBacktester`` and mutable scenario state while shared
    batch arrays/DataFrames remain immutable. Evidence order equals request order.
    """
    import concurrent.futures

    if not requests:
        raise ValueError("batch requires at least one request")
    batch = prepared_batch or prepare_execution_replay_batch(requests[0])
    start_ms = int(time.monotonic() * 1000)
    effective_workers = min(max(1, max_workers), len(requests))

    for request in requests:
        _validate_batch_request_compatibility(request, batch)

    if effective_workers <= 1:
        results: list[ExecutionReplayEvidence] = []
        for request in requests:
            evidence = _execute_batch_request(request, batch)
            results.append(evidence)
    else:
        ordered_results: list[ExecutionReplayEvidence | None] = [None] * len(requests)

        def _worker(index: int, req: ExecutionEquivalentReplayRequest) -> tuple[int, ExecutionReplayEvidence]:
            return index, _execute_batch_request(req, batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_worker, i, req): i
                for i, req in enumerate(requests)
            }
            for future in concurrent.futures.as_completed(futures):
                idx, evidence = future.result()
                ordered_results[idx] = evidence

        results = [ev for ev in ordered_results if ev is not None]

    end_ms = int(time.monotonic() * 1000)
    total_replays = len(results)
    total_builds = len(batch.segment_data)
    object.__setattr__(
        batch,
        "execution_replay_count",
        total_replays,
    )
    object.__setattr__(batch, "prepared_segment_build_count", total_builds)
    object.__setattr__(
        batch,
        "replay_prepare_elapsed_ms",
        batch.replay_prepare_elapsed_ms,
    )
    object.__setattr__(batch, "replay_execute_elapsed_ms", end_ms - start_ms)
    return tuple(results)


def _validate_batch_request_compatibility(
    request: ExecutionEquivalentReplayRequest,
    batch: PreparedExecutionReplayBatch,
) -> None:
    """Validate a request is compatible with the prepared batch before reuse."""
    if request.horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive session count")
    if request.market_frame is not batch.market_frame:
        raise ValueError("market_frame identity mismatch in batch request")
    if request.score_frame is not batch.score_frame:
        raise ValueError("score_frame identity mismatch in batch request")
    if request.segment_column != batch.segment_column:
        raise ValueError("segment_column mismatch in batch request")
    request_decisions = {
        int(segment_id): tuple(decisions)
        for segment_id, decisions in request.decision_sessions_by_segment.items()
    }
    if request_decisions != batch.decision_sessions_by_segment:
        raise ValueError("decision_sessions_by_segment mismatch in batch request")
    if _context_static_signature(request.context) != batch.context_static_signature:
        raise ValueError("execution context mismatch in batch request")


def _context_static_signature(context: ExecutionReplayContext) -> tuple[object, ...]:
    """Return execution inputs that must match before prepared reuse."""
    return (
        context.manifest.content_hash,
        tuple(sorted(context.instruments)),
        context.artifact_id,
        context.strategy_id,
        context.initial_portfolio,
        context.base_cost_schedule,
        context.stress_cost_schedule,
        context.liquidity_model,
        context.stress_liquidity_model,
        context.execution_policy,
        context.risk_policy.volatility_lookback_sessions,
    )


def _execute_batch_request(
    request: ExecutionEquivalentReplayRequest,
    batch: PreparedExecutionReplayBatch,
) -> ExecutionReplayEvidence:
    """Execute one request's backtest against the shared prepared batch."""
    accumulator = CandidateReplayAccumulator()
    for segment_id in sorted(request.decision_sessions_by_segment):
        decisions = request.decision_sessions_by_segment[segment_id]
        if not decisions:
            raise ValueError(f"segment {segment_id} has no declared decision sessions")
        segment = batch.segment_data[segment_id]
        summary = _execute_candidate_segment(request.context, segment, segment_id, decisions)
        accumulator.add_segment(summary)
    evidence = accumulator.finalize()
    assert isinstance(evidence, ExecutionReplayEvidence)
    return evidence


def _window_end(start: datetime, end: datetime) -> datetime:
    if end > start:
        return end
    return start + timedelta(seconds=1)


def _ledger_log_growth(ledger: tuple[BacktestLedgerRow, ...]) -> tuple[float, ...]:
    growth, _ = _ledger_growth_and_exposure(ledger)
    return growth


def _ledger_growth_and_exposure(
    ledger: tuple[BacktestLedgerRow, ...],
) -> tuple[tuple[float, ...], int]:
    """Single O(n) pass over a ledger producing log growth and invested intervals.

    ``growth`` is the per-interval log return ``log(equity_t / equity_{t-1})``;
    a non-positive or non-finite equity fails closed immediately. ``invested``
    is the count of complete intervals whose prior ledger row carried a positive
    ``positions_value`` (a held position), so economic exposure is measured from
    the ledger rather than from fills. The function never allocates raw
    score/price vectors.
    """
    equities = [row.equity for row in ledger]
    growth: list[float] = []
    invested = 0
    for previous_index, current_index in pairwise(
        range(len(equities))
    ):
        previous = equities[previous_index]
        current = equities[current_index]
        if previous <= 0.0 or current <= 0.0:
            raise ValueError("non-positive equity in replay ledger")
        if not np.isfinite(previous) or not np.isfinite(current):
            raise ValueError("non-finite equity in replay ledger")
        growth.append(float(np.log(current / previous)))
        if ledger[previous_index].positions_value > 0.0:
            invested += 1
    return tuple(growth), invested


def _ledger_growth_and_interval_exposure(
    ledger: tuple[BacktestLedgerRow, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Single O(n) pass producing log growth and per-interval prior-row exposure.

    ``growth`` is ``log(equity_t / equity_{t-1})``. ``exposure`` is
    ``min(positions_value_{t-1} / equity_{t-1}, 1.0)`` clamped to [0, 1] and
    finite.  Both series have length ``len(ledger) - 1``.
    """
    equities = [row.equity for row in ledger]
    growth: list[float] = []
    exposure: list[float] = []
    for previous_index, current_index in pairwise(range(len(equities))):
        previous = equities[previous_index]
        current = equities[current_index]
        if previous <= 0.0 or current <= 0.0:
            raise ValueError("non-positive equity in replay ledger")
        if not np.isfinite(previous) or not np.isfinite(current):
            raise ValueError("non-finite equity in replay ledger")
        growth.append(float(np.log(current / previous)))
        prior_pos = ledger[previous_index].positions_value
        exp = float(np.clip(prior_pos / previous, 0.0, 1.0)) if previous > 0.0 else 0.0
        if not np.isfinite(exp):
            raise ValueError("non-finite interval exposure in replay ledger")
        exposure.append(exp)
    return tuple(growth), tuple(exposure)


def exposure_matched_benchmark_log_growth(
    market_frame: pl.DataFrame,
    sessions: Sequence[datetime],
    interval_exposure: Sequence[float],
) -> tuple[float, ...]:
    """Deterministic gross equal-weight benchmark scaled by prior-interval exposure.

    Uses the same eligible instruments and executable open-to-open intervals as
    the replay.  Each interval's benchmark growth is
    ``exposure_t * mean(log(open_{t+1} / open_t))`` over eligible instruments.
    Zero exposure yields zero growth.  ``interval_exposure`` must have the same
    length as ``sessions``.
    """
    if len(interval_exposure) != len(sessions):
        raise ValueError(
            f"interval_exposure length {len(interval_exposure)} "
            f"does not match sessions length {len(sessions)}"
        )
    if len(sessions) < 1:
        return ()
    sorted_sessions = sorted(sessions)
    opens_by_session: dict[datetime, dict[str, float]] = {}
    for row in (
        market_frame.select("instrument_id", "session", "open")
        .filter(pl.col("session").is_in(sorted_sessions))
        .iter_rows(named=True)
    ):
        session = row["session"]
        if session not in opens_by_session:
            opens_by_session[session] = {}
        opens_by_session[session][str(row["instrument_id"])] = float(row["open"])
    growth: list[float] = []
    for idx in range(len(sorted_sessions) - 1):
        current_session = sorted_sessions[idx]
        next_session = sorted_sessions[idx + 1]
        exp = float(interval_exposure[idx])
        if exp == 0.0:
            growth.append(0.0)
            continue
        curr_prices = opens_by_session.get(current_session, {})
        next_prices = opens_by_session.get(next_session, {})
        common = set(curr_prices) & set(next_prices)
        if not common:
            growth.append(0.0)
            continue
        instrument_growth = [
            math.log(next_prices[instrument] / curr_prices[instrument])
            for instrument in sorted(common)
            if curr_prices[instrument] > 0
        ]
        if not instrument_growth:
            growth.append(0.0)
            continue
        mean_growth = sum(instrument_growth) / len(instrument_growth)
        growth.append(max(mean_growth * exp, -1.0))
    growth.append(0.0)
    return tuple(growth)


def _ledger_decision_equity(
    ledger: Sequence[object], decision_times: Sequence[datetime]
) -> dict[datetime, float]:
    """Map each declared decision time to its ledger equity (last occurrence)."""
    decision_set = set(decision_times)
    equities: dict[datetime, float] = {}
    for row in ledger:
        when = getattr(row, "session", None)
        equity = getattr(row, "equity", None)
        if when is None or equity is None:
            continue
        if when in decision_set:
            equities[when] = float(equity)
    return equities


def _decision_interval_log_growth(
    ledger: Sequence[object],
    decision_times: Sequence[datetime],
) -> tuple[float, ...]:
    """Per-completed-interval log growth between consecutive decision times.

    Returns one finite log-growth observation per *complete* decision interval
    ``(t_k, t_{k+1})`` for which both endpoints carry a finite positive ledger
    equity. Incomplete terminal (or otherwise unobserved) intervals are excluded
    and reported via the returned series length rather than treated as a zero
    return, so a horizon-locked replay exposes exactly its completed intervals.

    Raises ``ValueError`` only on a non-positive or non-finite equity at an
    observed decision endpoint (the ledger itself is malformed).
    """
    if len(decision_times) < 2:
        return ()
    equities = _ledger_decision_equity(ledger, decision_times)
    growth: list[float] = []
    for previous, current in pairwise(decision_times):
        prev_equity = equities.get(previous)
        curr_equity = equities.get(current)
        if prev_equity is None or curr_equity is None:
            continue
        if not (math.isfinite(prev_equity) and math.isfinite(curr_equity)):
            raise ValueError("non-finite equity at a decision interval endpoint")
        if prev_equity <= 0.0 or curr_equity <= 0.0:
            raise ValueError("non-positive equity at a decision interval endpoint")
        growth.append(float(math.log(curr_equity / prev_equity)))
    return tuple(growth)


def _filled_sessions(result: BacktestResult) -> int:
    return len({trade.session for trade in result.trades if trade.quantity > 0})


def _decision_provider(
    scored_market: pl.DataFrame,
    session_index: _ScoredSessionIndex,
    allocation_market: PreparedAllocationMarket,
    score_overlay: np.ndarray,
) -> ReplayDecisionProvider:
    def provider(decision_time: datetime, execution_time: datetime) -> PreparedReplayDecision:
        stop = session_index.stop_for(decision_time)
        if stop is None:
            visible = scored_market.filter(pl.col("available_time") <= decision_time)
        else:
            visible = scored_market.slice(0, stop)
        try:
            allocation_decision_index = next(
                index
                for index, session in enumerate(allocation_market.sessions)
                if session.date() == decision_time.date()
            )
        except StopIteration:
            allocation_decision_index = None
        return PreparedReplayDecision(
            decision_time,
            execution_time,
            visible,
            allocation_market=allocation_market,
            allocation_decision_index=allocation_decision_index,
            score_overlay=score_overlay,
        )

    return provider


def _scenario_planner(
    context: ExecutionReplayContext,
    dataset_hash: str,
) -> ReplayScenarioPlanner:
    def planner(
        prepared: PreparedReplayDecision,
        portfolio: PortfolioSnapshot,
        cycle_request: TradingCycleRequest,
    ) -> TradingCycleResult:
        return plan_prepared_scored_cycle(
            prepared, portfolio, cycle_request, context.instruments, dataset_hash
        )

    return planner


def instruments_from_frame(frame: pl.DataFrame) -> dict[str, Instrument]:
    """Canonical instrument mapping for every instrument in an executable panel."""
    instruments: dict[str, Instrument] = {}
    for row in frame.select("instrument_id").unique().iter_rows(named=True):
        instrument_id = str(row["instrument_id"])
        instruments[instrument_id] = Instrument(
            instrument_id=instrument_id,
            asset_kind=AssetKind.STOCK,
            exchange="KRX",
            symbol=instrument_id.split(":")[-1],
            currency="KRW",
        )
    return instruments
