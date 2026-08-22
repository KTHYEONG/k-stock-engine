"""Compact candidate accumulator tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.stocks.ml.replay_aggregation import (
    CandidateReplayAccumulator,
    SegmentExecutionSummary,
)


def _summary(segment_id: int) -> SegmentExecutionSummary:
    return SegmentExecutionSummary(
        segment_id=segment_id,
        base_growth=(0.001, 0.002),
        stress_growth=(0.0005, 0.0015),
        base_interval_exposure=(0.9, 0.8),
        stress_interval_exposure=(0.85, 0.75),
        planned_cycles=2,
        filled_orders=3,
        filled_sessions=1,
        invested_intervals=2,
        filled_cycles=1,
        turnover=0.1,
        base_cost_drag=0.0004,
        stress_cost_drag=0.0006,
        base_exposure=0.9,
        stress_exposure=0.85,
        unfilled_reason_counts={"missing-open": 1},
    )


def _add(accumulator: CandidateReplayAccumulator, segment_id: int) -> None:
    accumulator.add_segment(_summary(segment_id))


def test_finalize_orders_segments_and_weights_by_planned_cycles() -> None:
    accumulator = CandidateReplayAccumulator()
    _add(accumulator, 0)
    _add(accumulator, 1)
    evidence = accumulator.finalize()
    assert evidence.segment_ids == (0, 0, 1, 1)
    assert len(evidence.base_log_growth) == len(evidence.stress_log_growth) == 4
    assert evidence.planned_cycles == 4
    assert evidence.filled_orders == 6
    assert evidence.observed_interval_count == 4
    assert evidence.invested_interval_count == 4
    assert evidence.turnover == pytest.approx(0.1)
    assert evidence.unfilled_order_reason_counts == (("missing-open", 2),)
    assert np.all(np.isfinite(evidence.base_interval_exposure))


def test_diverging_base_stress_lengths_fail_closed() -> None:
    accumulator = CandidateReplayAccumulator()
    with pytest.raises(ValueError, match="diverged"):
        accumulator.add_segment(
            SegmentExecutionSummary(
            segment_id=0,
            base_growth=(0.1,),
            stress_growth=(0.1, 0.2),
            base_interval_exposure=(1.0,),
            stress_interval_exposure=(1.0, 1.0),
            planned_cycles=1,
            filled_orders=0,
            filled_sessions=0,
            invested_intervals=0,
            filled_cycles=0,
            turnover=0.0,
            base_cost_drag=0.0,
            stress_cost_drag=0.0,
            base_exposure=0.0,
            stress_exposure=0.0,
                unfilled_reason_counts={},
            )
        )


def test_zero_planned_cycles_yield_cash_defaults() -> None:
    accumulator = CandidateReplayAccumulator()
    evidence = accumulator.finalize()
    assert evidence.cash_session_fraction == 1.0
    assert evidence.turnover == 0.0
