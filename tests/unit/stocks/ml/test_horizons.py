"""ML horizon discovery: cohort-unit bootstrap, Holm admission, one primary."""
from __future__ import annotations

import numpy as np
import pytest

from src.stocks.ml.horizons import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    HorizonOOFEvidence,
    select_horizons,
)


def _evidence(
    horizon: int,
    base: tuple[float, ...],
    stress: tuple[float, ...] | None = None,
    *,
    segments: tuple[int, ...] | None = None,
    rank_ics: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> HorizonOOFEvidence:
    stress = base if stress is None else stress
    segment_ids = segments or tuple(0 for _ in base)
    return HorizonOOFEvidence(
        horizon_sessions=horizon,
        model_family="net_alpha_elastic_net",
        base_log_growth=base,
        stress_log_growth=stress,
        cohort_segment_ids=segment_ids,
        complete_cohort_count=len(base),
        active_cohort_count=len(base),
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=max(set(segment_ids), default=0) + 1,
        fold_rank_ics=rank_ics,
    )


def _positive(horizon: int, scale: float = 0.01) -> tuple[float, ...]:
    return tuple(float(scale) for _ in range(10))


def test_selects_primary_with_positive_lower_bound() -> None:
    candidates = (
        _evidence(5, _positive(5, 0.01)),
        _evidence(10, _positive(10, 0.005)),
        _evidence(15, tuple(-0.001 for _ in range(10))),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert result.selected_horizons == (5,)
    assert result.adjusted_lower_growth[5]["base"] > 0.0
    assert result.adjusted_lower_growth[5]["stress"] > 0.0
    assert result.adjusted_lower_growth[10]["stress"] > 0.0
    assert result.adjusted_lower_growth[15]["stress"] <= 0.0


def test_longer_horizon_raw_return_cannot_beat_annualized_growth() -> None:
    # h20 has a larger raw per-cohort return scale than h5, but its per-session
    # log growth is smaller; primary must still be the shorter, stronger horizon.
    candidates = (
        _evidence(5, _positive(5, 0.0010)),
        _evidence(20, _positive(20, 0.0006)),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5


def test_uses_request_controlled_resamples() -> None:
    """The selection honours the request-controlled bootstrap resample count."""
    candidates = (
        _evidence(5, _positive(5)),
        _evidence(10, _positive(10, 0.005)),
        _evidence(15, tuple(-0.001 for _ in range(10))),
    )
    default = select_horizons(candidates, 0.05, 42)
    assert default.primary_horizon_sessions == 5
    small = select_horizons(candidates, 0.05, 42, n_bootstrap=50)
    large = select_horizons(candidates, 0.05, 42, n_bootstrap=200)
    assert small.primary_horizon_sessions == large.primary_horizon_sessions == 5
    assert large.adjusted_lower_growth[5]["stress"] > 0.0


def test_rejects_resamples_below_two() -> None:
    candidates = (_evidence(5, _positive(5)),)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=1)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=0)
    assert DEFAULT_BOOTSTRAP_RESAMPLES >= 2


def test_no_trade_when_all_lower_bounds_non_positive() -> None:
    # The prior seven-value h20 series degenerated its bootstrap lower bound to
    # the sample mean (block length clamped to the full series). The corrected
    # cohort-unit bootstrap (block = ceil(7 ** (1/3)) = 2) must not reproduce a
    # positive lower bound equal to the mean, so a mean-positive-but-volatile
    # series cannot be spuriously admitted.
    seven = (0.05, -0.06, 0.05, -0.06, 0.05, -0.06, 0.05)
    candidates = (_evidence(20, seven),)
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions is None
    assert result.selected_horizons == ()
    bound = result.adjusted_lower_growth[20]["stress"]
    mean = float(np.mean(seven))
    assert mean > 0.0
    assert bound != pytest.approx(mean, abs=1e-12)
    assert bound <= 0.0


def test_rejects_all_negative_candidates() -> None:
    candidates = (
        _evidence(5, tuple(-0.01 for _ in range(10))),
        _evidence(10, tuple(-0.001 for _ in range(10))),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions is None
    assert result.selected_horizons == ()


def test_segment_with_fewer_than_two_blocks_is_inadmissible() -> None:
    # Segment 0 holds four cohorts (block 2, two blocks); segment 1 holds two
    # cohorts (block 2, one block). The candidate is inadmissible because one
    # segment has fewer than two resampling blocks.
    base = (0.01,) * 4 + (-0.01,) * 2
    segments = (0, 0, 0, 0, 1, 1)
    candidates = (_evidence(3, base, segments=segments),)
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions is None
    assert result.selected_horizons == ()


def test_segment_resampling_stays_positive_within_segments() -> None:
    # All positive cohorts split across two segments are pooled segment-locally
    # and admit the candidate; resampling never mixes cohort ids across gaps.
    base = (0.01,) * 8
    segments = (0, 0, 0, 0, 1, 1, 1, 1)
    candidates = (_evidence(3, base, segments=segments),)
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 3
    assert result.adjusted_lower_growth[3]["base"] > 0.0


def test_candidate_order_does_not_change_primary() -> None:
    candidates = (
        _evidence(5, _positive(5, 0.01)),
        _evidence(10, _positive(10, 0.005)),
    )
    forward = select_horizons(candidates, 0.05, 42)
    reversed_candidates = (candidates[1], candidates[0])
    backward = select_horizons(reversed_candidates, 0.05, 42)
    assert forward.primary_horizon_sessions == backward.primary_horizon_sessions == 5
    assert forward.evidence_hash == backward.evidence_hash


def test_selection_deterministic_for_identical_inputs() -> None:
    candidates = (
        _evidence(5, _positive(5, 0.01)),
        _evidence(10, _positive(10, 0.005)),
        _evidence(15, tuple(-0.001 for _ in range(10))),
    )
    first = select_horizons(candidates, 0.05, 42)
    second = select_horizons(candidates, 0.05, 42)
    assert first.evidence_hash == second.evidence_hash
    assert first.to_json() == second.to_json()
    assert first.primary_horizon_sessions == second.primary_horizon_sessions
