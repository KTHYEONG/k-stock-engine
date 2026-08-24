"""Growth-route benchmark stitching scenarios: SCENARIO_BENCH_STITCH_*."""
from __future__ import annotations

from datetime import datetime, UTC
from types import SimpleNamespace

import polars as pl

from src.stocks.ml.execution_replay import ExecutionReplayEvidence
from src.stocks.ml.horizons import GrowthRouteEvidence
from src.stocks.ml.training import _attach_growth_route_execution_evidence


def _session(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _panel() -> pl.DataFrame:
    rows = []
    for day in range(1, 7):
        session = _session(day)
        for instrument_id, base in (("KRX:A", 1000.0), ("KRX:B", 2000.0)):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "session": session,
                    "open": base * (1.0 + 0.01 * day),
                }
            )
    return pl.DataFrame(rows)


def _evidence(
    exposures: tuple[float, ...],
    bounds: tuple[tuple[datetime, ...], ...],
) -> ExecutionReplayEvidence:
    return ExecutionReplayEvidence(
        base_log_growth=tuple(0.01 for _ in exposures),
        stress_log_growth=tuple(0.01 for _ in exposures),
        segment_ids=tuple(0 if i < 3 else 1 for i in range(len(exposures))),
        planned_cycles=2,
        filled_orders=10,
        cash_session_fraction=0.0,
        turnover=1.0,
        observed_interval_count=len(exposures),
        invested_interval_count=len(exposures),
        invested_interval_fraction=1.0,
        filled_cycle_count=2,
        base_interval_exposure=exposures,
        stress_interval_exposure=exposures,
        base_interval_session_bounds=bounds,
    )


def _route_and_discovery(exposures, bounds):
    key = (10, 10, 8, "lower_bound_only")
    route = GrowthRouteEvidence(
        base_log_growth=tuple(0.01 for _ in exposures),
        stress_log_growth=tuple(0.01 for _ in exposures),
        segment_ids=(0, 0, 0, 1, 1),
        selected_policies=(key, key),
        interval_policies=(key,) * len(exposures),
        observed_interval_count=len(exposures),
        invested_interval_count=len(exposures),
        filled_orders=10,
    )
    candidate = SimpleNamespace(
        horizon_sessions=key[0],
        rebalance_frequency_sessions=key[1],
        top_k=key[2],
        profile_id=key[3],
        cohort_segment_ids=(0, 0, 0, 1, 1),
    )
    discovery = SimpleNamespace(
        execution_evidence_by_candidate={key: _evidence(exposures, bounds)},
        evidence=(candidate,),
    )
    return route, discovery


def test_segment_scoped_benchmark_matches_route_length() -> None:
    """SCENARIO_BENCH_STITCH_SEGMENT_ALIGNMENT."""
    route, discovery = _route_and_discovery(
        (0.5, 0.5, 0.5, 0.25, 0.25),
        (
            (_session(1), _session(2), _session(3), _session(4)),
            (_session(4), _session(5), _session(6)),
        ),
    )
    attached = _attach_growth_route_execution_evidence(route, discovery, _panel())
    assert len(attached.benchmark_log_growth) == len(route.base_log_growth)
    assert attached.benchmark_log_growth
    assert attached.benchmark_reconcile_failure == ""
    # exposure scaling keeps benchmark growth strictly below gross index growth
    assert all(value <= 0.05 for value in attached.benchmark_log_growth)


def test_benchmark_fail_closed_on_nonfinite_exposure() -> None:
    """SCENARIO_BENCH_STITCH_FAIL_CLOSED_ON_MISMATCH."""
    route, discovery = _route_and_discovery(
        (0.5, 0.5, 0.5, 0.25, 0.25),
        (
            (_session(1), _session(2), _session(3), _session(4)),
            (_session(4), _session(5), _session(6)),
        ),
    )
    # Simulate an upstream corrupted payload bypassing constructor guards.
    evidence = discovery.execution_evidence_by_candidate[(10, 10, 8, "lower_bound_only")]
    object.__setattr__(
        evidence,
        "base_interval_exposure",
        (0.5, float("nan"), 0.5, 0.25, 0.25),
    )
    attached = _attach_growth_route_execution_evidence(route, discovery, _panel())
    assert attached.benchmark_log_growth == ()
    assert attached.benchmark_reconcile_failure.startswith("benchmark-")

    route_high, discovery_high = _route_and_discovery(
        (0.5, 0.5, 0.5, 0.25, 0.25),
        (
            (_session(1), _session(2), _session(3), _session(4)),
            (_session(4), _session(5), _session(6)),
        ),
    )
    evidence_high = discovery_high.execution_evidence_by_candidate[
        (10, 10, 8, "lower_bound_only")
    ]
    object.__setattr__(
        evidence_high,
        "base_interval_exposure",
        (0.5, 1.5, 0.5, 0.25, 0.25),
    )
    attached_high = _attach_growth_route_execution_evidence(
        route_high, discovery_high, _panel()
    )
    assert attached_high.benchmark_log_growth == ()
    assert attached_high.benchmark_reconcile_failure.startswith("benchmark-")
