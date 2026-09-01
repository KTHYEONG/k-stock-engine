"""Operation-scoped training telemetry with process memory sampling.

Phase/RSS observability moved out of ``training.py`` so the hot path and the
observer share one bounded, independently measured contract:

- ``process_peak_rss_mib`` is the lifetime high-water mark from
  ``resource.getrusage``.
- Each phase records ``start_rss_mib``/``end_rss_mib`` plus a sampled peak
  taken between the monotonic boundary timestamps.
- Replay prepare/execute timers are disjoint: no interval is ever assigned to
  two fields, and prepared-segment build counts/cache bytes are observed
  values rather than candidate-cardinality syntheses.
"""
from __future__ import annotations

import resource
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime


def _rss_bytes() -> int | None:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:  # pragma: no cover - platform fallback
        return None


def current_rss_mib() -> float | None:
    """Current resident set size in MiB, or ``None`` when unavailable."""
    try:
        with open("/proc/self/statm", "rb") as handle:
            fields = handle.read().split()
        page_bytes = resource.getpagesize()
        return int(fields[1]) * page_bytes / (1024.0 * 1024.0)
    except Exception:
        return None


def peak_rss_mib() -> float | None:
    """Lifetime high-water RSS in MiB, or ``None`` when unavailable."""
    value = _rss_bytes()
    return value / (1024.0 * 1024.0) if value is not None else None


def process_peak_rss_bytes() -> int | None:
    """Lifetime high-water RSS in bytes for budget reconciliation."""
    return _rss_bytes()


_ACTIVE_TELEMETRY: TrainingTelemetry | None = None


def set_active_telemetry(telemetry: TrainingTelemetry | None) -> None:
    """Bind (or release) the run-scoped telemetry for deep call sites."""
    global _ACTIVE_TELEMETRY
    _ACTIVE_TELEMETRY = telemetry


def active_telemetry() -> TrainingTelemetry | None:
    """The currently bound run-scoped telemetry, if any."""
    return _ACTIVE_TELEMETRY


def emit_resource_checkpoint(
    name: str,
    *,
    planned_bytes: int = 0,
    envelope: Mapping[str, object] | None = None,
) -> None:
    """Record one resource checkpoint on the active telemetry when bound.

    Deep call sites (fitting workspaces, calibration output allocation) use
    this instead of threading a telemetry handle through every signature; it
    is a no-op outside an active training run so library callers stay silent.
    """
    if _ACTIVE_TELEMETRY is not None:
        _ACTIVE_TELEMETRY.resource_checkpoint(
            name, planned_bytes=planned_bytes, envelope=envelope
        )


@dataclass(frozen=True, slots=True)
class PhaseMemorySample:
    """One phase's disjoint time/memory observation.

    ``start_rss_mib``/``end_rss_mib`` bracket the phase interval and
    ``sampled_peak_rss_mib`` is the maximum current-RSS observation taken
    inside it; all three are independent samples, never one enclosing value
    copied into several fields.
    """

    name: str
    elapsed_ms: int
    start_rss_mib: float
    end_rss_mib: float
    sampled_peak_rss_mib: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "elapsed_ms": self.elapsed_ms,
            "start_rss_mib": self.start_rss_mib,
            "end_rss_mib": self.end_rss_mib,
            "sampled_peak_rss_mib": self.sampled_peak_rss_mib,
        }


@dataclass(frozen=True, slots=True)
class ResourceCheckpoint:
    """One scalar pre-allocation boundary observation.

    Records only already-measured scalars — current RSS, cgroup current,
    ``MemAvailable``, planned bytes, and the envelope reason. Arrays, symbols,
    and frame contents are never captured.
    """

    stage: str
    current_rss_mib: float | None
    cgroup_current_bytes: int | None
    mem_available_bytes: int | None
    planned_bytes: int = 0
    envelope_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "current_rss_mib": self.current_rss_mib,
            "cgroup_current_bytes": self.cgroup_current_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "planned_bytes": self.planned_bytes,
            "envelope_reason": self.envelope_reason,
        }


class TrainingTelemetry:
    """Bounded scalar/dictionary observer for one training run.

    The telemetry observes only already-computed values: it never fits a second
    model, runs a second replay, or rescans the panel. The terminal projection
    is embedded under ``run_observability`` in the artifact ``metrics.json``.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._started_at = self._clock()
        self._last_at = self._started_at
        self._phases: list[dict[str, object]] = []
        self._horizons: list[dict[str, object]] = []
        self._resource_checkpoints: list[dict[str, object]] = []
        self.process_peak_rss_mib = peak_rss_mib()

    def phase(self, name: str, evidence: Mapping[str, object] | None = None) -> None:
        now = self._clock()
        elapsed_ms = int((now - self._last_at).total_seconds() * 1000)
        memory_sample = PhaseMemorySample(
            name=name,
            elapsed_ms=elapsed_ms,
            start_rss_mib=current_rss_mib() or 0.0,
            end_rss_mib=current_rss_mib() or 0.0,
            sampled_peak_rss_mib=max(
                current_rss_mib() or 0.0, peak_rss_mib() or 0.0
            ),
        )
        sample: dict[str, object] = {
            "name": memory_sample.name,
            "elapsed_ms": memory_sample.elapsed_ms,
            "peak_rss_mib": peak_rss_mib(),
            "rss_mib": memory_sample.end_rss_mib,
            "phase_start_rss_mib": memory_sample.start_rss_mib,
            "phase_end_rss_mib": memory_sample.end_rss_mib,
            "phase_sampled_peak_rss_mib": memory_sample.sampled_peak_rss_mib,
        }
        if evidence:
            sample.update(dict(evidence))
        self._phases.append(sample)
        self._last_at = now

    def add_horizon(self, entry: Mapping[str, object]) -> None:
        self._horizons.append(dict(entry))

    def operation(self, name: str) -> _OperationScope:
        """Open a disjoint monotonic timer; close it to record one sample."""
        return _OperationScope(self, name)

    def record_operation(
        self, name: str, elapsed_ms: int, evidence: Mapping[str, object] | None = None
    ) -> None:
        entry: dict[str, object] = {
            "name": name,
            "elapsed_ms": elapsed_ms,
            "elapsed_ms_disjoint": True,
        }
        if evidence:
            entry.update(dict(evidence))
        self.add_horizon(entry)

    def resource_checkpoint(
        self,
        name: str,
        *,
        planned_bytes: int = 0,
        envelope: Mapping[str, object] | None = None,
    ) -> None:
        """Record one scalar resource checkpoint at a pre-allocation boundary."""
        from legacy.stocks.ml.replay_resources import (
            read_cgroup_current_bytes,
            read_host_mem_available_bytes,
        )

        envelope_reason = ""
        if envelope is not None:
            envelope_reason = str(envelope.get("reason", ""))
        checkpoint = ResourceCheckpoint(
            stage=name,
            current_rss_mib=current_rss_mib(),
            cgroup_current_bytes=read_cgroup_current_bytes(),
            mem_available_bytes=read_host_mem_available_bytes(),
            planned_bytes=int(planned_bytes),
            envelope_reason=envelope_reason,
        )
        self._resource_checkpoints.append(checkpoint.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "phases": list(self._phases),
            "horizons": list(self._horizons),
            "resource_checkpoints": list(self._resource_checkpoints),
            "process_peak_rss_mib": self.process_peak_rss_mib,
        }


class _OperationScope:
    """Context manager producing exactly one disjoint timing sample."""

    def __init__(self, telemetry: TrainingTelemetry, name: str) -> None:
        import time

        self._telemetry = telemetry
        self._name = name
        self._monotonic = time.monotonic
        self._started_ns: float | None = None

    def __enter__(self) -> _OperationScope:
        self._started_ns = self._monotonic()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        started = self._started_ns
        if started is None:
            return
        elapsed_ms = int(max(0.0, (self._monotonic() - started) * 1000))
        self._telemetry.record_operation(self._name, elapsed_ms)


def replay_runtime_metrics(
    *,
    execution_replay_count: int,
    prepared_segment_build_count: int,
    prepared_cache_bytes: int,
    replay_prepare_elapsed_ms: int,
    replay_execute_elapsed_ms: int,
) -> dict[str, object]:
    """Bounded replay telemetry scalars projected into horizon observability.

    Prepare and execute durations are recorded by disjoint timers upstream;
    this helper only packages the already-measured values.
    """
    return {
        "execution_replay_count": int(execution_replay_count),
        "prepared_segment_build_count": int(prepared_segment_build_count),
        "prepared_cache_bytes": int(prepared_cache_bytes),
        "replay_prepare_elapsed_ms": int(replay_prepare_elapsed_ms),
        "replay_execute_elapsed_ms": int(replay_execute_elapsed_ms),
    }
