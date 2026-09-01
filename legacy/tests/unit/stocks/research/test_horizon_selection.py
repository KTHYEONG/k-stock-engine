"""Horizon discovery evidence-driven selection tests."""
from __future__ import annotations

import pytest

from legacy.stocks.research.horizon_selection import (
    HorizonOOFEvidence,
    select_horizons,
)


def _evidence(horizon: int, blocks: tuple[float, ...]) -> HorizonOOFEvidence:
    return HorizonOOFEvidence(horizon=horizon, block_log_excess=blocks)


def test_selects_primary_with_positive_lower_bound() -> None:
    candidates = (
        _evidence(5, (0.001, 0.002, 0.0015, 0.003, 0.002)),
        _evidence(10, (0.0, 0.001, 0.0005, 0.001, 0.001)),
        _evidence(15, (-0.001, -0.002, -0.001, -0.0015, -0.001)),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon == 5
    assert result.lower_bounds[5] > 0.0
    assert result.lower_bounds[10] > 0.0
    assert result.lower_bounds[15] <= 0.0


def test_no_trade_when_all_lower_bounds_non_positive() -> None:
    candidates = (
        _evidence(5, (-0.01, -0.02, -0.015, -0.01, -0.012)),
        _evidence(10, (0.0, -0.001, -0.002, 0.0, -0.001)),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon is None
    assert result.secondary_horizon is None
    assert result.selected_horizons == ()


def test_secondary_only_when_paired_incremental_bound_positive() -> None:
    primary = _evidence(
        5,
        (0.00288, 0.00643, 0.00698, 0.00541, 0.00559, 0.00790, 0.00648, 0.00392, 0.00498),
    )
    secondary = _evidence(
        10,
        (0.00647, 0.00469, 0.00309, 0.00575, 0.00482, 0.00415, 0.00468, 0.00656, 0.00533),
    )
    low = _evidence(15, (-0.01, -0.02, -0.01, -0.015, -0.01, -0.02, -0.01, -0.015, -0.01))
    result = select_horizons((primary, secondary, low), 0.05, 42)
    assert result.primary_horizon == 5
    assert result.secondary_horizon == 10
    assert result.selected_horizons == (5, 10)


def test_no_secondary_when_incremental_bound_not_positive() -> None:
    primary = _evidence(5, (0.002, 0.003, 0.0025, 0.0035, 0.003))
    correlated = _evidence(10, (0.0019, 0.0029, 0.0024, 0.0034, 0.0029))
    result = select_horizons((primary, correlated), 0.05, 42)
    assert result.primary_horizon == 5
    assert result.secondary_horizon is None


def test_tiebreak_prefers_shorter_horizon() -> None:
    short = _evidence(3, (0.001, 0.0011, 0.0012, 0.0011, 0.001))
    long = _evidence(8, (0.001, 0.0011, 0.0012, 0.0011, 0.001))
    result = select_horizons((short, long), 0.05, 42)
    assert result.primary_horizon == 3


def test_effective_horizon_count_tracks_correlation() -> None:
    independent = (
        _evidence(5, (0.001, 0.002, 0.001, 0.003, 0.002)),
        _evidence(10, (0.002, 0.001, 0.003, 0.001, 0.002)),
    )
    result = select_horizons(independent, 0.05, 42)
    assert result.effective_horizon_count > 1.0

    duplicated = (
        _evidence(5, (0.001, 0.002, 0.0015, 0.003, 0.002)),
        _evidence(10, (0.001, 0.002, 0.0015, 0.003, 0.002)),
    )
    result_dup = select_horizons(duplicated, 0.05, 42)
    assert result_dup.effective_horizon_count < result.effective_horizon_count


def test_evidence_hash_is_deterministic() -> None:
    candidates = (
        _evidence(5, (0.001, 0.002, 0.0015, 0.003, 0.002)),
        _evidence(10, (0.0, 0.001, 0.0005, 0.001, 0.001)),
    )
    first = select_horizons(candidates, 0.05, 42)
    second = select_horizons(candidates, 0.05, 42)
    assert first.evidence_hash == second.evidence_hash
    assert first.to_json()["primary_horizon"] == second.to_json()["primary_horizon"]


def test_rejects_empty_candidates_and_bad_alpha() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_horizons((), 0.05, 42)
    with pytest.raises(ValueError, match="alpha"):
        select_horizons((_evidence(5, (0.001, 0.002)),), 0.0, 42)
    with pytest.raises(ValueError, match="alpha"):
        select_horizons((_evidence(5, (0.001, 0.002)),), 1.5, 42)
