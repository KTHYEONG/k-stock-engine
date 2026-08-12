"""Monotonic causal calibration schedule parity, isolation, and fallback."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from src.core.costs import default_base_schedule
from src.stocks.research.calibration_schedule import CausalCalibrationSchedule
from src.stocks.research.economic_alpha import CausalAlphaCalibrator


def _floats_close(a: object, b: object, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def _assert_states_equivalent(reference: dict, scheduled: dict) -> None:
    assert reference["history_sessions"] == scheduled["history_sessions"]
    ref_buckets = {b["bucket"]: b for b in reference["buckets"]}
    sched_buckets = {b["bucket"]: b for b in scheduled["buckets"]}
    assert set(ref_buckets) == set(sched_buckets)
    for bucket, ref in ref_buckets.items():
        got = sched_buckets[bucket]
        assert ref["sample_size"] == got["sample_size"]
        assert _floats_close(ref["expected_active_alpha"], got["expected_active_alpha"])
        assert _floats_close(ref["alpha_lower_bound"], got["alpha_lower_bound"])
        if got["expected_active_alpha"] is not None:
            assert got["alpha_lower_bound"] > 0.0


def _positive_observations(n_sessions: int = 60, n_tickers: int = 25) -> pl.DataFrame:
    import numpy as np

    start = datetime(2024, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for s in range(n_sessions):
        for t in range(n_tickers):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t:06d}",
                    "session": start + timedelta(days=s),
                    "score": float(t) + (s % 3),
                    "residual_o2o_5d": float(0.004 * t + rng.normal(0.0, 0.0002)),
                    "label_available_time": start + timedelta(days=s + 6),
                }
            )
    return pl.DataFrame(rows)


def test_schedule_states_equal_reference_at_every_decision() -> None:
    cal = CausalAlphaCalibrator(
        bucket_count=5, min_calibration_sessions=10, n_bootstrap=200,
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    observations = _positive_observations()
    decision_times = [start + timedelta(days=d) for d in range(12, 60, 5)]
    schedule = CausalCalibrationSchedule.build(
        observations, decision_times, cal, default_base_schedule(),
        max_workspace_bytes=10_000_000,
    )
    for decision_time in decision_times:
        reference = cal.prepare_decision(observations, decision_time, default_base_schedule())
        _assert_states_equivalent(reference, schedule.state_at(decision_time))


def test_schedule_reveals_future_labels_only_after_decision() -> None:
    cal = CausalAlphaCalibrator(
        bucket_count=5, min_calibration_sessions=10, n_bootstrap=200,
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    observations = _positive_observations(n_sessions=40, n_tickers=20)
    decision_times = [start + timedelta(days=d) for d in range(12, 40, 5)]
    schedule = CausalCalibrationSchedule.build(
        observations, decision_times, cal, default_base_schedule(),
        max_workspace_bytes=10_000_000,
    )
    early = schedule.state_at(decision_times[0])

    flipped = observations.with_columns(
        pl.when(pl.col("session") >= start + timedelta(days=30))
        .then(pl.lit(0.99))
        .otherwise(pl.col("residual_o2o_5d"))
        .alias("residual_o2o_5d")
    )
    flipped_schedule = CausalCalibrationSchedule.build(
        flipped, decision_times, cal, default_base_schedule(),
        max_workspace_bytes=10_000_000,
    )
    _assert_states_equivalent(early, flipped_schedule.state_at(decision_times[0]))


def test_schedule_partial_availability_uses_reference_path() -> None:
    cal = CausalAlphaCalibrator(
        bucket_count=5, min_calibration_sessions=10, n_bootstrap=50,
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for s in range(40):
        for t in range(20):
            lat = start + timedelta(days=s + 6)
            if s == 10 and t >= 10:
                lat = start + timedelta(days=s + 12)
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t:06d}",
                    "session": start + timedelta(days=s),
                    "score": float(t) + (s % 3),
                    "residual_o2o_5d": float(0.004 * t + 0.001 * (s % 4)),
                    "label_available_time": lat,
                }
            )
    observations = pl.DataFrame(rows)
    decision_times = [start + timedelta(days=d) for d in range(12, 40, 5)]
    schedule = CausalCalibrationSchedule.build(
        observations, decision_times, cal, default_base_schedule(),
        max_workspace_bytes=10_000_000,
    )
    assert schedule._use_reference is True
    for decision_time in decision_times:
        reference = cal.prepare_decision(observations, decision_time, default_base_schedule())
        _assert_states_equivalent(reference, schedule.state_at(decision_time))


def test_schedule_eligible_prefix_rows_is_monotonic() -> None:
    cal = CausalAlphaCalibrator(
        bucket_count=5, min_calibration_sessions=10, n_bootstrap=50,
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    observations = _positive_observations(n_sessions=40, n_tickers=20)
    decision_times = [start + timedelta(days=d) for d in range(12, 40, 5)]
    schedule = CausalCalibrationSchedule.build(
        observations, decision_times, cal, default_base_schedule(),
        max_workspace_bytes=10_000_000,
    )
    counts = [schedule.eligible_prefix_rows(dt) for dt in decision_times]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]
