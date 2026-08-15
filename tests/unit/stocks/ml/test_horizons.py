"""ML horizon discovery: request-controlled bootstrap resample count and strict validation."""
from __future__ import annotations

import pytest

from src.stocks.ml.horizons import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    HorizonOOFEvidence,
    select_horizons,
)


def _evidence(horizon: int, blocks: tuple[float, ...]) -> HorizonOOFEvidence:
    return HorizonOOFEvidence(horizon_sessions=horizon, block_log_excess=blocks)


def test_selects_primary_with_positive_lower_bound() -> None:
    candidates = (
        _evidence(5, (0.001, 0.002, 0.0015, 0.003, 0.002)),
        _evidence(10, (0.0, 0.001, 0.0005, 0.001, 0.001)),
        _evidence(15, (-0.001, -0.002, -0.001, -0.0015, -0.001)),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert result.lower_bounds[5] > 0.0
    assert result.lower_bounds[10] > 0.0
    assert result.lower_bounds[15] <= 0.0


def test_uses_request_controlled_resamples() -> None:
    """The selection honours the request-controlled bootstrap resample count."""
    candidates = (
        _evidence(5, (0.001, 0.002, 0.0015, 0.003, 0.002)),
        _evidence(10, (0.0, 0.001, 0.0005, 0.001, 0.001)),
        _evidence(15, (-0.001, -0.002, -0.001, -0.0015, -0.001)),
    )
    default = select_horizons(candidates, 0.05, 42)
    assert default.primary_horizon_sessions == 5
    small = select_horizons(candidates, 0.05, 42, n_bootstrap=50)
    large = select_horizons(candidates, 0.05, 42, n_bootstrap=200)
    assert small.primary_horizon_sessions == large.primary_horizon_sessions == 5
    assert large.lower_bounds[5] > 0.0


def test_rejects_resamples_below_two() -> None:
    candidates = (_evidence(5, (0.001, 0.002, 0.0015, 0.003, 0.002)),)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=1)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=0)
    assert DEFAULT_BOOTSTRAP_RESAMPLES >= 2


def test_no_trade_when_all_lower_bounds_non_positive() -> None:
    candidates = (
        _evidence(5, (-0.01, -0.02, -0.015, -0.01, -0.012)),
        _evidence(10, (0.0, -0.001, -0.002, 0.0, -0.001)),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions is None
    assert result.secondary_horizon_sessions is None
    assert result.selected_horizons == ()
