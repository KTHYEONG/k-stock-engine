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
from collections.abc import Callable
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
