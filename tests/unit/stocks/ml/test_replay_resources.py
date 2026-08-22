"""Effective memory limit resolution and pre-allocation planning tests."""
from __future__ import annotations

from pathlib import Path


from src.stocks.ml.replay_resources import (
    EffectiveMemoryLimit,
    MemoryBudgetExceededError,
    estimate_replay_allocation,
    read_cgroup_limit_bytes,
    resolve_effective_memory_limit,
)


class _Meta:
    def __init__(self, bytes_: int) -> None:
        self.estimated_prepared_bytes = bytes_


def test_effective_limit_is_minimum_of_finite_contributors() -> None:
    limit = resolve_effective_memory_limit(
        4096 * 1024 * 1024,
        cgroup_reader=lambda root: 2048 * 1024 * 1024,
        address_space_reader=lambda: 8192 * 1024 * 1024,
        host_reader=lambda: 32 * 1024 * 1024 * 1024,
    )
    assert limit.effective_limit_bytes == 2048 * 1024 * 1024
    assert limit.host_total_bytes == 32 * 1024 * 1024 * 1024


def test_unlimited_sentinels_are_ignored_and_host_ram_never_raises() -> None:
    limit = resolve_effective_memory_limit(
        None,
        cgroup_reader=lambda root: None,
        address_space_reader=lambda: None,
        host_reader=lambda: 1024,
    )
    assert limit.effective_limit_bytes is None
    assert limit.host_total_bytes == 1024

    bounded = resolve_effective_memory_limit(
        512 * 1024 * 1024,
        cgroup_reader=lambda root: None,
        address_space_reader=lambda: None,
        host_reader=lambda: 128 * 1024 * 1024,
    )
    assert bounded.effective_limit_bytes == 512 * 1024 * 1024


def test_cgroup_v1_huge_sentinel_is_unlimited(tmp_path: Path) -> None:
    v1_root = tmp_path / "memory"
    v1_root.mkdir()
    (v1_root / "memory.limit_in_bytes").write_text(str(1 << 62))
    assert read_cgroup_limit_bytes(tmp_path) is None
    (v1_root / "memory.limit_in_bytes").write_text("268435456")
    assert read_cgroup_limit_bytes(tmp_path) == 256 * 1024 * 1024


def test_allocation_plan_fails_closed_on_breach() -> None:
    plan = estimate_replay_allocation(
        _Meta(600 * 1024 * 1024),
        candidate_count=2,
        worker_count=1,
        current_live_bytes=100 * 1024 * 1024,
        next_metadata=_Meta(500 * 1024 * 1024),
        effective_limit_bytes=1024 * 1024 * 1024,
    )
    assert not plan.ok
    assert "exceeds effective limit" in plan.reason


def test_allocation_plan_passes_within_budget() -> None:
    plan = estimate_replay_allocation(
        _Meta(300 * 1024 * 1024),
        candidate_count=2,
        worker_count=1,
        current_live_bytes=100 * 1024 * 1024,
        next_metadata=_Meta(300 * 1024 * 1024),
        effective_limit_bytes=1024 * 1024 * 1024,
    )
    assert plan.ok


def test_memory_budget_error_carries_planning_context() -> None:
    error = MemoryBudgetExceededError(
        "breach", planned_bytes=900, limit_bytes=800
    )
    assert error.planned_bytes == 900
    assert error.limit_bytes == 800


def test_effective_memory_limit_dataclass_roundtrip() -> None:
    limit = EffectiveMemoryLimit(
        request_limit_bytes=10,
        cgroup_limit_bytes=20,
        address_space_limit_bytes=None,
        host_total_bytes=30,
        effective_limit_bytes=10,
    )
    assert limit.bounded
    assert limit.to_dict()["effective_limit_bytes"] == 10
