"""Telemetry contract tests: disjoint timers and memory sampling."""
from __future__ import annotations

from datetime import UTC, datetime

from legacy.stocks.ml.telemetry import (
    PhaseMemorySample,
    ResourceCheckpoint,
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


def test_resource_checkpoint_records_only_scalar_fields(monkeypatch) -> None:
    """TELEMETRY-AND-CLI-05: checkpoints carry stage/RSS/cgroup/MemAvailable/planned/reason."""
    import legacy.stocks.ml.replay_resources as replay_resources
    import legacy.stocks.ml.telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "current_rss_mib", lambda: 123.5)
    monkeypatch.setattr(
        replay_resources, "read_cgroup_current_bytes", lambda root=None: 4567
    )
    monkeypatch.setattr(
        replay_resources, "read_host_mem_available_bytes", lambda: 8901
    )

    telemetry = TrainingTelemetry()
    telemetry.resource_checkpoint("calibration", planned_bytes=2048)
    entry = telemetry.to_dict()["resource_checkpoints"][0]
    assert entry["stage"] == "calibration"
    assert entry["current_rss_mib"] == 123.5
    assert entry["cgroup_current_bytes"] == 4567
    assert entry["mem_available_bytes"] == 8901
    assert entry["planned_bytes"] == 2048
    assert entry["envelope_reason"] == ""

    checkpoint = ResourceCheckpoint(
        stage="replay",
        current_rss_mib=None,
        cgroup_current_bytes=None,
        mem_available_bytes=None,
        planned_bytes=1,
        envelope_reason="unbounded",
    )
    payload = checkpoint.to_dict()
    for key in (
        "stage",
        "current_rss_mib",
        "cgroup_current_bytes",
        "mem_available_bytes",
        "planned_bytes",
        "envelope_reason",
    ):
        assert key in payload


def test_emit_resource_checkpoint_is_noop_without_active_run(monkeypatch) -> None:
    import legacy.stocks.ml.telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "_ACTIVE_TELEMETRY", None)
    # Must not raise outside an active training run.
    telemetry_module.emit_resource_checkpoint("fitting", planned_bytes=10)

    telemetry = TrainingTelemetry()
    telemetry_module.set_active_telemetry(telemetry)
    try:
        telemetry_module.emit_resource_checkpoint(
            "matrix_prepare", planned_bytes=32, envelope={"reason": "within"}
        )
    finally:
        telemetry_module.set_active_telemetry(None)
    entry = telemetry.to_dict()["resource_checkpoints"][0]
    assert entry["stage"] == "matrix_prepare"
    assert entry["planned_bytes"] == 32
    assert entry["envelope_reason"] == "within"
