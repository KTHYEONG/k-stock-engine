"""Effective memory limit resolution and pre-allocation planning tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.stocks.ml.replay_resources import (
    EffectiveMemoryLimit,
    MemoryBudgetExceededError,
    ResourceEnvelope,
    estimate_replay_allocation,
    plan_training_allocation,
    read_cgroup_limit_bytes,
    read_host_mem_available_bytes,
    resolve_effective_memory_limit,
)

MIB = 1024 * 1024


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


ALLOCATION_GUARD_02 = "ML_FULL_EXECUTION_P0_ALLOCATION_GUARD_02"


def _guard_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cgroup_limit: int | None,
    cgroup_current: int | None,
    mem_available: int | None,
) -> None:
    from src.stocks.ml import replay_resources as rr

    monkeypatch.setattr(
        rr, "read_cgroup_limit_bytes", lambda root=None: cgroup_limit
    )
    monkeypatch.setattr(
        rr, "read_cgroup_current_bytes", lambda root=None: cgroup_current
    )
    monkeypatch.setattr(rr, "read_host_mem_available_bytes", lambda: mem_available)


def test_allocation_guard_02_within_every_headroom_names_limiting_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_env(
        monkeypatch,
        cgroup_limit=8192 * MIB,
        cgroup_current=1024 * MIB,
        mem_available=16 * 1024 * MIB,
    )
    envelope = plan_training_allocation(
        3000 * MIB,
        request_limit_bytes=4096 * MIB,
        reserve_bytes=512 * MIB,
        current_rss_bytes=1024 * MIB,
    )
    assert isinstance(envelope, ResourceEnvelope)
    assert envelope.ok
    # Process headroom (3072 MiB) is the smallest finite term.
    assert envelope.limiting_source == "request_max_rss"
    assert envelope.process_headroom_bytes == 3072 * MIB
    assert envelope.cgroup_headroom_bytes == 6656 * MIB
    assert envelope.system_headroom_bytes == (16 * 1024 - 512) * MIB


def test_allocation_guard_02_exceeding_process_headroom_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_env(
        monkeypatch,
        cgroup_limit=8192 * MIB,
        cgroup_current=1024 * MIB,
        mem_available=16 * 1024 * MIB,
    )
    envelope = plan_training_allocation(
        4000 * MIB,
        request_limit_bytes=4096 * MIB,
        reserve_bytes=0,
        current_rss_bytes=1024 * MIB,
    )
    assert not envelope.ok
    assert envelope.limiting_source == "request_max_rss"
    assert "request_max_rss" in envelope.reason


def test_allocation_guard_02_reserve_subtracted_from_cgroup_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_env(
        monkeypatch,
        cgroup_limit=2048 * MIB,
        cgroup_current=256 * MIB,
        mem_available=64 * 1024 * MIB,
    )
    envelope = plan_training_allocation(
        1600 * MIB,
        request_limit_bytes=4096 * MIB,
        reserve_bytes=512 * MIB,
        current_rss_bytes=100 * MIB,
    )
    # Process headroom is untouched by the reserve (3996 MiB); the cgroup
    # term loses the full reserve (2048 - 256 - 512 = 1280 MiB) and breaches.
    assert envelope.cgroup_headroom_bytes == 1280 * MIB
    assert envelope.process_headroom_bytes == 3996 * MIB
    assert not envelope.ok
    assert envelope.limiting_source == "cgroup"


def test_allocation_guard_02_exceeding_mem_available_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_env(
        monkeypatch,
        cgroup_limit=None,
        cgroup_current=None,
        mem_available=1024 * MIB,
    )
    envelope = plan_training_allocation(
        600 * MIB,
        request_limit_bytes=None,
        reserve_bytes=512 * MIB,
        current_rss_bytes=100 * MIB,
    )
    assert not envelope.ok
    assert envelope.limiting_source == "mem_available"
    assert envelope.system_headroom_bytes == 512 * MIB


def test_allocation_guard_02_unbounded_when_no_finite_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_env(
        monkeypatch, cgroup_limit=None, cgroup_current=None, mem_available=None
    )
    envelope = plan_training_allocation(
        600 * MIB, request_limit_bytes=None, reserve_bytes=512 * MIB
    )
    assert envelope.ok
    assert envelope.limiting_source is None
    assert envelope.reason == "unbounded"


def test_read_host_mem_available_bytes_is_positive_or_none() -> None:
    value = read_host_mem_available_bytes()
    if value is not None:
        assert value > 0
