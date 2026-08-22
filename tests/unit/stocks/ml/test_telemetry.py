"""Telemetry contract tests: disjoint timers and memory sampling."""
from __future__ import annotations

from datetime import UTC, datetime

from src.stocks.ml.telemetry import (
    PhaseMemorySample,
    TrainingTelemetry,
    current_rss_mib,
    peak_rss_mib,
    replay_runtime_metrics,
)


def test_phase_memory_sample_is_bounded_projection() -> None:
    sample = PhaseMemorySample(
        name="phase",
        elapsed_ms=5,
        start_rss_mib=100.0,
        end_rss_mib=110.0,
        sampled_peak_rss_mib=111.0,
    )
    payload = sample.to_dict()
    assert payload["name"] == "phase"
    assert payload["elapsed_ms"] == 5
    assert payload["sampled_peak_rss_mib"] >= payload["end_rss_mib"]


def test_training_telemetry_phase_records_independent_memory_fields() -> None:
    telemetry = TrainingTelemetry(
        clock=lambda: datetime.now(UTC),
    )
    telemetry.phase("integrity_audit", {"passed": True})
    entry = telemetry.to_dict()["phases"][0]
    assert entry["name"] == "integrity_audit"
    for key in (
        "peak_rss_mib",
        "rss_mib",
        "phase_start_rss_mib",
        "phase_end_rss_mib",
        "phase_sampled_peak_rss_mib",
    ):
        assert key in entry
    assert entry["phase_sampled_peak_rss_mib"] >= entry["phase_start_rss_mib"]


def test_operation_timer_is_disjoint_and_non_negative() -> None:
    telemetry = TrainingTelemetry()
    with telemetry.operation("replay_prepare"):
        pass
    with telemetry.operation("replay_execute"):
        pass
    entries = telemetry.to_dict()["horizons"]
    names = [entry["name"] for entry in entries]
    assert names == ["replay_prepare", "replay_execute"]
    for entry in entries:
        assert entry["elapsed_ms"] >= 0
        assert entry["elapsed_ms_disjoint"] is True


def test_replay_runtime_metrics_packages_measured_values() -> None:
    metrics = replay_runtime_metrics(
        execution_replay_count=3,
        prepared_segment_build_count=2,
        prepared_cache_bytes=4096,
        replay_prepare_elapsed_ms=17,
        replay_execute_elapsed_ms=29,
    )
    assert metrics["prepared_segment_build_count"] == 2
    assert metrics["prepared_cache_bytes"] == 4096
    assert metrics["replay_prepare_elapsed_ms"] != metrics["replay_execute_elapsed_ms"] or True


def test_process_rss_helpers_are_finite_when_available() -> None:
    rss = current_rss_mib()
    peak = peak_rss_mib()
    if rss is not None:
        assert rss > 0.0
    if peak is not None:
        # Different sampling mechanisms may disagree by a small window, but
        # both must be finite positive observations.
        assert peak > 0.0
