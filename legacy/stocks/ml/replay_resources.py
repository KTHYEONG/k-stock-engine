"""Cgroup-aware effective memory limits and pre-allocation planning.

The replay memory guard resolves ``effective_limit`` as the minimum of the
finite request budget, the finite cgroup (v2 then v1) limit, and the finite
process address-space limit. Host physical RAM is advisory only: it is
reported for observability and never raises the effective limit. Before any
material allocation, the planner verifies

    current_live + planned_allocation + largest_next_allocation <= effective_limit

and fails closed before ``PreparedReplayMarket.build`` when the invariant
cannot hold.
"""
from __future__ import annotations

import resource
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_UNLIMITED = "max"

# Conservative per-worker Python runtime overhead beyond prepared-segment bytes.
_WORKER_STATE_BYTES = 50 * 1024 * 1024


class MemoryBudgetExceededError(ValueError):
    """Raised fail-closed when a planned allocation cannot fit the limit."""

    def __init__(self, message: str, *, planned_bytes: int, limit_bytes: int) -> None:
        super().__init__(message)
        self.planned_bytes = planned_bytes
        self.limit_bytes = limit_bytes


def read_cgroup_limit_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Read the cgroup v2/v1 memory limit; ``None`` when absent or unlimited."""
    v2 = root / "memory.max"
    if v2.exists():
        try:
            raw = v2.read_text().strip()
        except OSError:
            return None
        if raw != _UNLIMITED:
            try:
                value = int(raw)
            except ValueError:
                return None
            return value if value > 0 else None
        return None
    v1 = root / "memory" / "memory.limit_in_bytes"
    if v1.exists():
        try:
            raw = v1.read_text().strip()
        except OSError:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        # v1 reports a huge sentinel instead of "max" for unlimited.
        if 0 < value < (1 << 60):
            return value
    return None


def read_cgroup_current_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Read the current cgroup memory usage; ``None`` when unavailable."""
    for name in ("memory.current", "memory.usage_in_bytes"):
        candidate = root / name
        if not candidate.exists():
            continue
        try:
            return int(candidate.read_text().strip())
        except (OSError, ValueError):
            return None
    return None


def read_address_space_limit_bytes() -> int | None:
    """Return the finite ``RLIMIT_AS`` ceiling; ``None`` when unlimited."""
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    except (OSError, ValueError):  # pragma: no cover - platform fallback
        return None
    if soft in (resource.RLIM_INFINITY, -1):
        return None
    return int(soft)


def read_host_ram_bytes() -> int | None:
    """Host physical RAM (advisory only), or ``None`` when unavailable."""
    try:
        with open("/proc/meminfo", "rb") as handle:
            for line in handle:
                if line.startswith(b"MemTotal:"):
                    kib = int(line.split()[1])
                    return kib * 1024
        return None
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True, slots=True)
class EffectiveMemoryLimit:
    """Resolved effective memory budget with full provenance.

    ``effective_limit_bytes`` is the minimum of every finite contributor;
    unlimited sentinels are ignored. ``host_total_bytes`` never participates.
    """

    request_limit_bytes: int | None
    cgroup_limit_bytes: int | None
    address_space_limit_bytes: int | None
    host_total_bytes: int | None
    effective_limit_bytes: int | None

    @property
    def bounded(self) -> bool:
        return self.effective_limit_bytes is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_limit_bytes": self.request_limit_bytes,
            "cgroup_limit_bytes": self.cgroup_limit_bytes,
            "address_space_limit_bytes": self.address_space_limit_bytes,
            "host_total_bytes": self.host_total_bytes,
            "effective_limit_bytes": self.effective_limit_bytes,
        }


def resolve_effective_memory_limit(
    request_limit_bytes: int | None,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    cgroup_reader: Callable[[Path], int | None] = read_cgroup_limit_bytes,
    address_space_reader: Callable[[], int | None] = read_address_space_limit_bytes,
    host_reader: Callable[[], int | None] = read_host_ram_bytes,
) -> EffectiveMemoryLimit:
    """Resolve ``min(finite request, finite cgroup, finite address space)``.

    Host physical RAM is recorded but advisory: it can never raise or become
    the effective limit on its own.
    """
    request_value = (
        int(request_limit_bytes)
        if request_limit_bytes is not None and int(request_limit_bytes) > 0
        else None
    )
    cgroup_value = cgroup_reader(cgroup_root)
    address_value = address_space_reader()
    host_value = host_reader()
    candidates = [
        value
        for value in (request_value, cgroup_value, address_value)
        if value is not None and value > 0
    ]
    effective = min(candidates) if candidates else None
    return EffectiveMemoryLimit(
        request_limit_bytes=request_value,
        cgroup_limit_bytes=cgroup_value,
        address_space_limit_bytes=address_value,
        host_total_bytes=host_value,
        effective_limit_bytes=effective,
    )


@dataclass(frozen=True, slots=True)
class ReplayAllocationPlan:
    """Fail-closed verdict for one pre-allocation boundary."""

    ok: bool
    current_live_bytes: int
    planned_allocation_bytes: int
    largest_next_allocation_bytes: int
    effective_limit_bytes: int | None
    reason: str = ""

    def projected_total_bytes(self) -> int:
        return self.current_live_bytes + self.planned_allocation_bytes + self.largest_next_allocation_bytes


def estimate_replay_allocation(
    metadata: Any,
    candidate_count: int,
    worker_count: int,
    *,
    current_live_bytes: int = 0,
    next_metadata: Any | None = None,
    effective_limit_bytes: int | None = None,
) -> ReplayAllocationPlan:
    """Plan one segment allocation against the effective limit.

    ``metadata``/``next_metadata`` are :class:`ReplaySegmentMetadata`-shaped
    objects exposing ``estimated_prepared_bytes``. The plan covers the live
    bytes plus this segment's prepared estimate plus one worker reserve per
    concurrent worker, and the largest next-segment allocation so the boundary
    after this one can still fit.
    """
    estimated = int(metadata.estimated_prepared_bytes)
    workers = max(1, int(worker_count))
    planned = estimated + max(0, workers - 1) * _WORKER_STATE_BYTES
    largest_next = (
        int(next_metadata.estimated_prepared_bytes) + _WORKER_STATE_BYTES
        if next_metadata is not None
        else 0
    )
    del candidate_count
    if effective_limit_bytes is None:
        return ReplayAllocationPlan(
            ok=True,
            current_live_bytes=int(current_live_bytes),
            planned_allocation_bytes=planned,
            largest_next_allocation_bytes=largest_next,
            effective_limit_bytes=None,
            reason="unbounded",
        )
    total = int(current_live_bytes) + planned + largest_next
    if total <= int(effective_limit_bytes):
        return ReplayAllocationPlan(
            ok=True,
            current_live_bytes=int(current_live_bytes),
            planned_allocation_bytes=planned,
            largest_next_allocation_bytes=largest_next,
            effective_limit_bytes=int(effective_limit_bytes),
        )
    return ReplayAllocationPlan(
        ok=False,
        current_live_bytes=int(current_live_bytes),
        planned_allocation_bytes=planned,
        largest_next_allocation_bytes=largest_next,
        effective_limit_bytes=int(effective_limit_bytes),
        reason=(
            f"replay allocation {total} bytes exceeds effective limit "
            f"{int(effective_limit_bytes)} bytes "
            f"(live={current_live_bytes}, planned={planned}, "
            f"largest_next={largest_next})"
        ),
    )


def read_host_mem_available_bytes() -> int | None:
    """Host ``MemAvailable`` (system-wide allocatable headroom) or ``None``."""
    try:
        with open("/proc/meminfo", "rb") as handle:
            for line in handle:
                if line.startswith(b"MemAvailable:"):
                    kib = int(line.split()[1])
                    return kib * 1024
        return None
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """Typed fail-closed verdict for one training pre-allocation boundary.

    Every finite headroom must still fit ``planned_bytes``: request max RSS
    minus current process RSS, cgroup limit minus cgroup current minus the
    concurrent-workload reserve, and ``MemAvailable`` minus that reserve. The
    reserve is subtracted only from cgroup/system terms; host total RAM never
    becomes allocatable headroom. ``limiting_source`` names the smallest
    finite contributor on success and the breached source on failure.
    """

    ok: bool
    planned_bytes: int
    limiting_source: str | None
    process_headroom_bytes: int | None
    cgroup_headroom_bytes: int | None
    system_headroom_bytes: int | None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "planned_bytes": self.planned_bytes,
            "limiting_source": self.limiting_source,
            "process_headroom_bytes": self.process_headroom_bytes,
            "cgroup_headroom_bytes": self.cgroup_headroom_bytes,
            "system_headroom_bytes": self.system_headroom_bytes,
            "reason": self.reason,
        }


def _current_process_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"VmRSS:"):
                    kib = int(line.split()[1])
                    return kib * 1024
        return None
    except (OSError, ValueError, IndexError):
        return None


def plan_training_allocation(
    planned_bytes: int,
    *,
    request_limit_bytes: int | None,
    reserve_bytes: int,
    current_rss_bytes: int | None = None,
) -> ResourceEnvelope:
    """Plan one fitting/calibration allocation against every finite headroom.

    Fails closed before allocation when any finite term cannot fit
    ``planned_bytes``. Unlimited or unavailable terms are omitted; with no
    finite term at all the plan is unbounded and succeeds.
    """
    planned = int(planned_bytes)
    if planned < 0:
        raise ValueError("planned_bytes must be non-negative")
    reserve = max(0, int(reserve_bytes))
    rss = (
        _current_process_rss_bytes()
        if current_rss_bytes is None
        else int(current_rss_bytes)
    )
    process_limit = (
        int(request_limit_bytes)
        if request_limit_bytes is not None and int(request_limit_bytes) > 0
        else None
    )
    process_headroom = (
        process_limit - rss
        if process_limit is not None and rss is not None
        else None
    )
    cgroup_limit = read_cgroup_limit_bytes()
    cgroup_current = read_cgroup_current_bytes()
    cgroup_headroom = (
        cgroup_limit - cgroup_current - reserve
        if cgroup_limit is not None and cgroup_current is not None
        else None
    )
    mem_available = read_host_mem_available_bytes()
    system_headroom = (
        mem_available - reserve if mem_available is not None else None
    )
    finite = {
        "request_max_rss": process_headroom,
        "cgroup": cgroup_headroom,
        "mem_available": system_headroom,
    }
    breaches = [
        (source, headroom)
        for source, headroom in finite.items()
        if headroom is not None and planned > headroom
    ]
    if breaches:
        source, headroom = min(breaches, key=lambda item: item[1])
        detail = ", ".join(
            f"{name}={headroom_value}" for name, headroom_value in finite.items() if headroom_value is not None
        )
        return ResourceEnvelope(
            ok=False,
            planned_bytes=planned,
            limiting_source=source,
            process_headroom_bytes=process_headroom,
            cgroup_headroom_bytes=cgroup_headroom,
            system_headroom_bytes=system_headroom,
            reason=(
                f"planned {planned} bytes exceeds {source} headroom "
                f"{headroom} bytes ({detail}, reserve={reserve})"
            ),
        )
    bounded = {
        source: headroom for source, headroom in finite.items() if headroom is not None
    }
    if not bounded:
        return ResourceEnvelope(
            ok=True,
            planned_bytes=planned,
            limiting_source=None,
            process_headroom_bytes=process_headroom,
            cgroup_headroom_bytes=cgroup_headroom,
            system_headroom_bytes=system_headroom,
            reason="unbounded",
        )
    limiting_source, _ = min(bounded.items(), key=lambda item: item[1])
    return ResourceEnvelope(
        ok=True,
        planned_bytes=planned,
        limiting_source=limiting_source,
        process_headroom_bytes=process_headroom,
        cgroup_headroom_bytes=cgroup_headroom,
        system_headroom_bytes=system_headroom,
        reason=f"within {limiting_source} headroom",
    )


class TrainingRunDeniedError(MemoryBudgetExceededError):
    """Typed terminal denial for one guarded training-run boundary."""

    def __init__(
        self, message: str, *, planned_bytes: int, limit_bytes: int, stage: str
    ) -> None:
        super().__init__(message, planned_bytes=planned_bytes, limit_bytes=limit_bytes)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class GuardBoundaryRecord:
    """Sampled scalars for one guarded training-run boundary."""

    stage: str
    current_rss_bytes: int | None
    cgroup_current_bytes: int | None
    mem_available_bytes: int | None
    live_owners: tuple[str, ...]
    planned_bytes: int
    verdict: str
    reason: str = ""

    def payload(self) -> dict[str, object]:
        """Bounded scalar diagnostic payload for one boundary."""
        return {
            "stage": self.stage,
            "current_rss_bytes": self.current_rss_bytes,
            "cgroup_current_bytes": self.cgroup_current_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "live_owners": ",".join(self.live_owners)[:512],
            "live_owner_count": len(self.live_owners),
            "planned_bytes": self.planned_bytes,
            "verdict": self.verdict,
            "reason": self.reason[:512],
        }


_BOUNDARY_HEADROOM_FIELDS = {
    "request_max_rss": "process_headroom_bytes",
    "cgroup": "cgroup_headroom_bytes",
    "mem_available": "system_headroom_bytes",
}


def _emit_boundary_checkpoint(
    sink: Any, run_id: str, record: GuardBoundaryRecord
) -> None:
    """Emit one bounded scalar boundary checkpoint (best-effort telemetry)."""
    from legacy.stocks.observability.contracts import (
        DiagnosticCategory,
        DiagnosticEvent,
        DiagnosticStage,
        DiagnosticStatus,
    )

    sequence = int(getattr(sink, "_diagnostic_sequence", 0)) + 1
    sink._diagnostic_sequence = sequence
    try:
        sink.emit(
            DiagnosticEvent(
                run_id=run_id,
                sequence=sequence,
                category=DiagnosticCategory.SYS,
                component="ml.replay_resources",
                stage=DiagnosticStage.ALLOCATION,
                event="training_run_boundary",
                status=(
                    DiagnosticStatus.PASS
                    if record.verdict == "ok"
                    else DiagnosticStatus.FAIL
                ),
                payload=dict(record.payload()),
            )
        )
    except (OSError, ValueError):
        return


class TrainingRunGuard:
    """Samples RSS/cgroup/MemAvailable at each training materialization boundary.

    Every boundary records the sampled current RSS, cgroup current,
    ``MemAvailable``, live owner names, planned bytes, and the verdict as one
    scalar diagnostic event. A predicted breach raises
    :class:`TrainingRunDeniedError` before any allocation happens so the outer
    CLI can publish a durable terminal failure instead of dying mid-run.
    """

    def __init__(
        self,
        *,
        request_limit_bytes: int | None = None,
        reserve_bytes: int = 0,
        run_id: str = "training-run",
        diagnostics: Any | None = None,
    ) -> None:
        self._request_limit_bytes = (
            int(request_limit_bytes)
            if request_limit_bytes is not None and int(request_limit_bytes) > 0
            else None
        )
        self._reserve_bytes = max(0, int(reserve_bytes))
        self._run_id = run_id or "training-run"
        self._diagnostics = diagnostics
        self.records: list[GuardBoundaryRecord] = []

    @property
    def denied(self) -> bool:
        """Whether any recorded boundary breached its finite headroom."""
        return any(record.verdict == "denied" for record in self.records)

    def boundary(
        self,
        stage: str,
        *,
        planned_bytes: int,
        live_owners: Sequence[str] = (),
    ) -> ResourceEnvelope:
        """Sample one boundary, record it, and fail closed on a breach."""
        rss = _current_process_rss_bytes()
        cgroup_current = read_cgroup_current_bytes()
        mem_available = read_host_mem_available_bytes()
        envelope = plan_training_allocation(
            planned_bytes,
            request_limit_bytes=self._request_limit_bytes,
            reserve_bytes=self._reserve_bytes,
            current_rss_bytes=rss,
        )
        record = GuardBoundaryRecord(
            stage=stage,
            current_rss_bytes=rss,
            cgroup_current_bytes=cgroup_current,
            mem_available_bytes=mem_available,
            live_owners=tuple(live_owners),
            planned_bytes=int(envelope.planned_bytes),
            verdict="ok" if envelope.ok else "denied",
            reason=envelope.reason,
        )
        self.records.append(record)
        if self._diagnostics is not None:
            _emit_boundary_checkpoint(self._diagnostics, self._run_id, record)
        if not envelope.ok:
            headroom_field = _BOUNDARY_HEADROOM_FIELDS.get(
                envelope.limiting_source or "", ""
            )
            breached = getattr(envelope, headroom_field, None) if headroom_field else None
            raise TrainingRunDeniedError(
                f"training run guard denied {stage}: {envelope.reason}",
                planned_bytes=envelope.planned_bytes,
                limit_bytes=int(breached) if breached is not None else 0,
                stage=stage,
            )
        return envelope
