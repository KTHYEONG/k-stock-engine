"""Compact ordered accumulators for segment-major candidate replay.

Each replay candidate owns one :class:`CandidateReplayAccumulator`. While a
prepared segment is live, every candidate appends only bounded scalars and
per-interval exposure series; raw ``BacktestResult`` ledgers/trades are
discarded by the executor before the next candidate/segment. ``finalize``
assembles the immutable :class:`ExecutionReplayEvidence` in declared candidate
order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class _SegmentTotals:
    planned_cycles: int = 0
    filled_orders: int = 0
    filled_sessions: int = 0
    observed_intervals: int = 0
    invested_intervals: int = 0
    filled_cycles: int = 0
    turnover_weighted: float = 0.0
    base_cost_drag_weighted: float = 0.0
    stress_cost_drag_weighted: float = 0.0
    base_exposure_weighted: float = 0.0
    stress_exposure_weighted: float = 0.0


@dataclass(slots=True)
class CandidateReplayAccumulator:
    """One candidate's ordered compact evidence across all replay segments."""

    unfilled_reasons: dict[str, int] = field(default_factory=dict)
    base_growth: list[float] = field(default_factory=list)
    stress_growth: list[float] = field(default_factory=list)
    segment_ids: list[int] = field(default_factory=list)
    base_interval_exposure: list[float] = field(default_factory=list)
    stress_interval_exposure: list[float] = field(default_factory=list)
    totals: _SegmentTotals = field(default_factory=_SegmentTotals)

    def add_segment(self, summary: SegmentExecutionSummary) -> None:
        """Append one executed segment's bounded evidence."""
        segment_id = summary.segment_id
        base_growth = summary.base_growth
        stress_growth = summary.stress_growth
        planned_cycles = summary.planned_cycles
        filled_orders = summary.filled_orders
        filled_cycles = summary.filled_cycles
        turnover = summary.turnover
        base_cost_drag = summary.base_cost_drag
        stress_cost_drag = summary.stress_cost_drag
        base_exposure = summary.base_exposure
        stress_exposure = summary.stress_exposure
        unfilled_reason_counts = summary.unfilled_reason_counts
        invested_intervals = summary.invested_intervals
        base_interval_exposure = summary.base_interval_exposure
        stress_interval_exposure = summary.stress_interval_exposure
        filled_sessions = summary.filled_sessions
        if len(base_growth) != len(stress_growth):
            raise ValueError(
                f"base and stress ledgers diverged in session count "
                f"for segment {segment_id}"
            )
        self.base_growth.extend(base_growth)
        self.stress_growth.extend(stress_growth)
        self.segment_ids.extend([segment_id] * len(base_growth))
        self.base_interval_exposure.extend(base_interval_exposure)
        self.stress_interval_exposure.extend(stress_interval_exposure)
        weight = max(1, int(planned_cycles))
        totals = self.totals
        totals.planned_cycles += max(0, int(planned_cycles))
        totals.filled_orders += int(filled_orders)
        totals.filled_sessions += int(filled_sessions)
        totals.observed_intervals += len(base_growth)
        totals.invested_intervals += int(invested_intervals)
        totals.filled_cycles += int(filled_cycles)
        totals.turnover_weighted += float(turnover) * weight
        totals.base_cost_drag_weighted += float(base_cost_drag) * weight
        totals.stress_cost_drag_weighted += float(stress_cost_drag) * weight
        totals.base_exposure_weighted += float(base_exposure) * weight
        totals.stress_exposure_weighted += float(stress_exposure) * weight
        for reason, count in unfilled_reason_counts.items():
            key = str(reason)
            self.unfilled_reasons[key] = self.unfilled_reasons.get(key, 0) + int(count)

    def finalize(self) -> object:
        """Assemble the immutable evidence in declared segment order."""
        from src.stocks.ml.execution_replay import ExecutionReplayEvidence

        totals = self.totals
        total_planned = totals.planned_cycles
        cash_fraction = (
            1.0
            if total_planned <= 0
            else float(
                np.clip(1.0 - totals.filled_sessions / total_planned, 0.0, 1.0)
            )
        )
        invested_fraction = (
            float(totals.invested_intervals / totals.observed_intervals)
            if totals.observed_intervals > 0
            else 0.0
        )

        def weighted(value: float) -> float:
            return value / total_planned if total_planned > 0 else 0.0

        return ExecutionReplayEvidence(
            base_log_growth=tuple(self.base_growth),
            stress_log_growth=tuple(self.stress_growth),
            segment_ids=tuple(self.segment_ids),
            planned_cycles=total_planned,
            filled_orders=totals.filled_orders,
            cash_session_fraction=cash_fraction,
            turnover=weighted(totals.turnover_weighted),
            observed_interval_count=int(totals.observed_intervals),
            invested_interval_count=int(totals.invested_intervals),
            invested_interval_fraction=invested_fraction,
            filled_cycle_count=int(totals.filled_cycles),
            unfilled_order_reason_counts=tuple(sorted(self.unfilled_reasons.items())),
            utility_transition_diagnostics=(),
            base_cost_drag=weighted(totals.base_cost_drag_weighted),
            stress_cost_drag=weighted(totals.stress_cost_drag_weighted),
            base_exposure=weighted(totals.base_exposure_weighted),
            stress_exposure=weighted(totals.stress_exposure_weighted),
            base_interval_exposure=tuple(self.base_interval_exposure),
            stress_interval_exposure=tuple(self.stress_interval_exposure),
        )


@dataclass(frozen=True, slots=True)
class SegmentExecutionSummary:
    """Typed bounded result of one candidate's execution over one segment."""

    segment_id: int
    base_growth: tuple[float, ...]
    stress_growth: tuple[float, ...]
    base_interval_exposure: tuple[float, ...]
    stress_interval_exposure: tuple[float, ...]
    planned_cycles: int
    filled_orders: int
    filled_sessions: int
    invested_intervals: int
    filled_cycles: int
    turnover: float
    base_cost_drag: float
    stress_cost_drag: float
    base_exposure: float
    stress_exposure: float
    unfilled_reason_counts: dict[str, int]
