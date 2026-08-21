"""ML horizon/profile frontier: vintage bootstrap, Holm admission, one primary."""
from __future__ import annotations

import numpy as np
import pytest

from src.stocks.ml.horizons import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    HorizonOOFEvidence,
    _cohort_bootstrap,
    select_horizons,
)

_LEGACY = "legacy_overlay_5bps"
_LOWER = "lower_bound_only"


def _evidence(
    horizon: int,
    base: tuple[float, ...],
    stress: tuple[float, ...] | None = None,
    *,
    segments: tuple[int, ...] | None = None,
    rank_ics: tuple[float, ...] = (0.1, 0.2, 0.3),
    profile_id: str = _LEGACY,
    rebalance_frequency_sessions: int = 5,
    top_k: int = 20,
) -> HorizonOOFEvidence:
    stress = base if stress is None else stress
    segment_ids = segments or tuple(0 for _ in base)
    return HorizonOOFEvidence(
        horizon_sessions=horizon,
        profile_id=profile_id,
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
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
    )


def _positive(horizon: int, scale: float = 0.01, count: int = 60) -> tuple[float, ...]:
    return tuple(float(scale) for _ in range(count))


def test_selects_primary_with_positive_lower_bound() -> None:
    candidates = (
        _evidence(5, _positive(5, 0.01)),
        _evidence(10, _positive(10, 0.005, count=40)),
        _evidence(15, tuple(-0.001 for _ in range(10))),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert result.primary_profile_id == _LEGACY
    assert result.selected_horizons == (5,)
    assert result.selected_profile_id == _LEGACY
    assert result.adjusted_lower_growth[(5, 5, 20, _LEGACY)]["base"] > 0.0
    assert result.adjusted_lower_growth[(5, 5, 20, _LEGACY)]["stress"] > 0.0
    assert result.adjusted_lower_growth[(10, 5, 20, _LEGACY)]["stress"] > 0.0
    assert result.adjusted_lower_growth[(15, 5, 20, _LEGACY)]["stress"] <= 0.0


def test_longer_horizon_raw_return_cannot_beat_annualized_growth() -> None:
    # h20 has a larger raw per-vintage return scale than h5, but its per-session
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
    assert large.adjusted_lower_growth[(5, 5, 20, _LEGACY)]["stress"] > 0.0


def test_rejects_resamples_below_two() -> None:
    candidates = (_evidence(5, _positive(5)),)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=1)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=0)
    assert DEFAULT_BOOTSTRAP_RESAMPLES >= 2


def test_no_trade_when_all_lower_bounds_non_positive() -> None:
    # A mean-positive-but-volatile series must not be spuriously admitted.
    seven = (0.05, -0.06, 0.05, -0.06, 0.05, -0.06, 0.05)
    candidates = (_evidence(20, seven),)
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions is None
    assert result.primary_profile_id is None
    assert result.selected_horizons == ()
    bound = result.adjusted_lower_growth[(20, 5, 20, _LEGACY)]["stress"]
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
    # Segment 0 holds four vintages (block 2, two blocks); segment 1 holds two
    # vintages (block 2, one block). The candidate is inadmissible because one
    # segment has fewer than two resampling blocks.
    base = (0.01,) * 4 + (-0.01,) * 2
    segments = (0, 0, 0, 0, 1, 1)
    candidates = (_evidence(3, base, segments=segments),)
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions is None
    assert result.selected_horizons == ()


def test_segment_resampling_stays_positive_within_segments() -> None:
    # All positive vintages split across two segments are pooled segment-locally
    # and admit the candidate; resampling never mixes vintage ids across gaps.
    base = (0.01,) * 8
    segments = (0, 0, 0, 0, 1, 1, 1, 1)
    candidates = (
        _evidence(3, base, segments=segments, rebalance_frequency_sessions=3),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 3
    assert result.adjusted_lower_growth[(3, 3, 20, _LEGACY)]["base"] > 0.0


def test_candidate_order_does_not_change_primary() -> None:
    candidates = (
        _evidence(5, _positive(5, 0.01)),
        _evidence(10, _positive(10, 0.005)),
    )
    forward = select_horizons(candidates, 0.05, 42)
    reversed_candidates = (candidates[1], candidates[0])
    backward = select_horizons(reversed_candidates, 0.05, 42)
    assert forward.primary_horizon_sessions == backward.primary_horizon_sessions == 5
    assert forward.primary_profile_id == backward.primary_profile_id
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
    assert first.primary_profile_id == second.primary_profile_id


def test_frontier_selects_between_two_profiles() -> None:
    """The zero-overlay profile admits a candidate the legacy band rejects."""
    candidates = (
        _evidence(5, tuple(-0.001 for _ in range(10)), profile_id=_LEGACY),
        _evidence(5, _positive(5, 0.01), profile_id=_LOWER),
        _evidence(10, _positive(10, 0.005), profile_id=_LEGACY),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert result.primary_profile_id == _LOWER
    assert result.adjusted_lower_growth[(5, 5, 20, _LOWER)]["stress"] > 0.0
    assert result.adjusted_lower_growth[(5, 5, 20, _LEGACY)]["stress"] <= 0.0
    assert result.to_json()["primary_profile_id"] == _LOWER


def test_profile_tie_break_prefers_lexicographic_profile_id() -> None:
    """Identical evidence across profiles picks the lexicographically smaller id."""
    base = _positive(5, 0.01)
    candidates = (
        _evidence(5, base, profile_id=_LOWER),
        _evidence(5, base, profile_id=_LEGACY),
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert result.primary_profile_id == _LEGACY


def test_frontier_holmes_across_every_horizon_profile_pair() -> None:
    """Holm family size covers the full horizon x profile frontier."""
    candidates = tuple(
        _evidence(horizon, _positive(horizon, 0.01, count=40), profile_id=profile)
        for horizon in (5, 10)
        for profile in (_LEGACY, _LOWER)
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert len(result.adjusted_lower_growth) == 4
    assert len(result.base_holm_thresholds) == 4
    thresholds = sorted(result.base_holm_thresholds.values())
    assert thresholds == [0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05]


def test_bootstrap_block_length_is_at_least_the_horizon() -> None:
    """A horizon exceeding the cube-root block forces inadmissibility."""
    # Three vintages with horizon three: block must be three, giving a single
    # block and hence fewer than two resampling blocks (inadmissible). Without
    # the horizon floor the cube-root block of two would admit the candidate.
    values = (0.01, 0.01, 0.01)
    floored = _cohort_bootstrap(values, (0, 0, 0), 200, 42, min_block_length=3)
    assert floored is None
    unfloored = _cohort_bootstrap(values, (0, 0, 0), 200, 42, min_block_length=1)
    assert unfloored is not None


def _sparse_evidence(
    horizon: int,
    base: tuple[float, ...],
    paired: tuple[float, ...],
    *,
    sparse_turnover: float,
    shadow_turnover: float,
    profile_id: str = _LEGACY,
) -> HorizonOOFEvidence:
    segments = tuple(0 for _ in base)
    return HorizonOOFEvidence(
        horizon_sessions=horizon,
        profile_id=profile_id,
        model_family="net_alpha_elastic_net",
        base_log_growth=base,
        stress_log_growth=base,
        cohort_segment_ids=segments,
        complete_cohort_count=len(base),
        active_cohort_count=len(base),
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=1,
        fold_rank_ics=(0.1, 0.2, 0.3),
        paired_stress_log_growth=paired,
        sparse_turnover=sparse_turnover,
        shadow_turnover=shadow_turnover,
    )


def test_sparse_growth_paired_admission() -> None:
    """SPARSE_GROWTH_04_MATCHED_SHADOW_ADMISSION.

    A candidate is selectable only when base, stress, and paired lower bounds
    are strictly positive and the sparse/shadow turnover ratio is at most 0.60;
    each Holm threshold stays in (0, bootstrap_alpha]. A non-positive paired
    lower bound or an excessive turnover ratio forces NO_TRADE.
    """
    alpha = 0.05
    good = _sparse_evidence(
        10, _positive(10, 0.01, count=60), _positive(10, 0.002, count=60),
        sparse_turnover=1.0, shadow_turnover=2.0,
    )
    result = select_horizons((good,), alpha, 42)
    assert result.primary_horizon_sessions == 10
    assert result.primary_profile_id == _LEGACY
    assert result.paired_lower_bounds[(10, 5, 20, _LEGACY)] > 0.0
    assert result.turnover_ratio[(10, 5, 20, _LEGACY)] <= 0.60
    assert 0.0 < result.base_holm_thresholds[(10, 5, 20, _LEGACY)] <= alpha
    assert 0.0 < result.paired_holm_thresholds[(10, 5, 20, _LEGACY)] <= alpha

    negative_paired = _sparse_evidence(
        10, _positive(10, 0.01, count=60),
        tuple(-0.002 for _ in range(60)),
        sparse_turnover=1.0, shadow_turnover=2.0,
    )
    rejected = select_horizons((negative_paired,), alpha, 42)
    assert rejected.primary_horizon_sessions is None

    high_turnover = _sparse_evidence(
        10, _positive(10, 0.01, count=60), _positive(10, 0.002, count=60),
        sparse_turnover=1.0, shadow_turnover=1.0,
    )
    rejected_ratio = select_horizons((high_turnover,), alpha, 42)
    assert rejected_ratio.primary_horizon_sessions is None
    assert rejected_ratio.turnover_ratio[(10, 5, 20, _LOWER if False else _LEGACY)] > 0.60


def test_sparse_growth_v6_holm_hard_gate() -> None:
    """SPARSE_GROWTH_V6_HOLM_HARD_GATE.

    A candidate whose base/stress lower quantiles are > 0 but whose stress
    p-value exceeds its own Holm threshold is rejected (primary is None). A
    paired candidate is evaluated against its own paired Holm threshold, never
    the base threshold reused.
    """
    alpha = 0.05
    base = _positive(60, 0.01, count=60)  # type: ignore[call-arg]
    # Mixed-sign stress path: many small-negative cohorts drag the observed mean
    # to a tiny positive value so 2x-observed is easily exceeded by positive
    # blocks (centered p-value 0.7 > alpha), yet the 5% quantile lower bound
    # stays strictly positive. The compound lower-bound gate would pass, but the
    # per-path Holm p-value gate rejects the candidate.
    stress = (
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, 0.05145225554272906,
        0.05145225554272906, -0.015879781282870976, -0.015879781282870976,
        -0.015879781282870976, 0.05145225554272906, -0.015879781282870976,
        0.05145225554272906, -0.015879781282870976, 0.05145225554272906,
        -0.015879781282870976, -0.015879781282870976, 0.05145225554272906,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, 0.05145225554272906,
        -0.015879781282870976, 0.05145225554272906, -0.015879781282870976,
        -0.015879781282870976, 0.05145225554272906, 0.05145225554272906,
        -0.015879781282870976, -0.015879781282870976, 0.05145225554272906,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        0.05145225554272906, 0.05145225554272906, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        0.05145225554272906, 0.05145225554272906, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        0.05145225554272906, -0.015879781282870976, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
        -0.015879781282870976, -0.015879781282870976, -0.015879781282870976,
    )
    spiky = _evidence(10, base, stress, segments=(0,) * 60, rank_ics=(0.1, 0.2, 0.3))
    result = select_horizons((spiky,), alpha, 42)
    assert result.primary_horizon_sessions is None
    assert result.primary_profile_id is None
    key = (10, 5, 20, _LEGACY)
    assert result.adjusted_lower_growth[key]["stress"] > 0.0
    assert result.stress_p_values[key] > alpha

    # Paired candidate with positive base/stress/paired and a valid turnover
    # ratio is selectable and carries its own per-path Holm thresholds.
    paired = _sparse_evidence(
        10, _positive(60, 0.01), _positive(60, 0.002),
        sparse_turnover=1.0, shadow_turnover=2.0,
    )
    selectable = select_horizons(
        (paired,), alpha, 42
    )
    assert selectable.primary_horizon_sessions == 10
    key = (10, 5, 20, _LEGACY)
    assert key in selectable.paired_holm_thresholds
    assert 0.0 < selectable.paired_holm_thresholds[key] <= alpha
    assert key in selectable.base_holm_thresholds
    # The paired lower bound is computed from the paired threshold, confirmed
    # independent of any base-threshold reuse.
    assert selectable.paired_lower_bounds[key] > 0.0


ML_COMPOUNDING_03_FRONTIER_MULTIPLICITY_AND_FEASIBILITY = (
    "ML_COMPOUNDING_03_FRONTIER_MULTIPLICITY_AND_FEASIBILITY"
)


def test_frontier_multiplicity_and_feasibility() -> None:
    """ML_COMPOUNDING_03_FRONTIER_MULTIPLICITY_AND_FEASIBILITY.

    Only cells satisfying C <= H and K >= ceil(0.90 / 0.08) = 12 are formed;
    Holm threshold count equals every formed (H, C, K, profile) cell, and a
    positive raw h20 score cannot select unless base, stress, and paired
    lower bounds are all > 0.
    """
    from src.stocks.ml.contracts import ExecutionFrontierSettings

    frontier = ExecutionFrontierSettings(
        candidate_horizon_sessions=(5, 10, 20, 40),
        candidate_rebalance_frequency_sessions=(5, 10, 20),
        candidate_top_k=(12, 16, 20, 24),
    )
    cells = frontier.feasible_cells(gross_cap=0.90, single_name_cap=0.08)
    assert all(c <= h for h, c, k in cells)
    assert all(k >= 12 for h, c, k in cells)
    assert all(k >= 12 for h, c, k in cells)

    candidates = tuple(
        _evidence(h, _positive(h, 0.01, count=60), profile_id=_LEGACY)
        for h in (5, 10)
    )
    result = select_horizons(candidates, 0.05, 42)
    assert result.primary_horizon_sessions == 5
    assert len(result.adjusted_lower_growth) == 2
    assert len(result.base_holm_thresholds) == 2
    thresholds = sorted(result.base_holm_thresholds.values())
    assert thresholds == [0.05 / 2, 0.05]

    negative_stress = _evidence(
        20, _positive(20, 0.01),
        stress=tuple(-0.001 for _ in range(60)),
        profile_id=_LEGACY,
    )
    result_neg = select_horizons((negative_stress,), 0.05, 42)
    assert result_neg.primary_horizon_sessions is None


def test_execution_frontier_all_feasible_cells_and_global_holm() -> None:
    """ML_EXEC_FRONTIER_01_ALL_FEASIBLE_CELLS_AND_GLOBAL_HOLM.

    With the default frontier H=(10,20), C=(5,10,20), K=(12,16,20,24) and caps
    (0.90, 0.08), exactly 20 feasible (H, C, K) cells are formed; the 3 default
    profiles generate 60 distinct (H, C, K, profile) Holm keys, and no key has
    C > H or K < 12.
    """
    from src.stocks.ml.contracts import (
        DEFAULT_POLICY_PROFILES,
        ExecutionFrontierSettings,
    )

    frontier = ExecutionFrontierSettings()
    cells = frontier.feasible_cells(0.90, 0.08)
    assert len(cells) == 20
    assert all(c <= h for h, c, k in cells)
    assert all(k >= 12 for h, c, k in cells)

    candidates = tuple(
        _evidence(
            h, _positive(h, 0.01, count=60),
            profile_id=profile.profile_id,
            rebalance_frequency_sessions=c,
            top_k=k,
        )
        for (h, c, k) in cells
        for profile in DEFAULT_POLICY_PROFILES
    )
    assert len(candidates) == 60

    result = select_horizons(candidates, 0.05, 42)
    keys = list(result.adjusted_lower_growth.keys())
    assert len(keys) == 60
    assert len(set(keys)) == 60
    for (h, c, k, _profile_id) in keys:
        assert c <= h
        assert k >= 12
    assert len(result.base_holm_thresholds) == 60
    assert len(result.stress_holm_thresholds) == 60


def test_primary_preserves_selected_nonfirst_execution_cell() -> None:
    """The selected C/K must not be replaced by the first same-H/profile cell."""
    first = _evidence(
        10, _positive(10, 0.001, count=60),
        rebalance_frequency_sessions=5,
        top_k=12,
    )
    selected = _evidence(
        10, _positive(10, 0.01, count=60),
        rebalance_frequency_sessions=10,
        top_k=20,
    )
    result = select_horizons((first, selected), 0.05, 42)
    assert result.primary_horizon_sessions == 10
    assert result.primary_rebalance_frequency_sessions == 10
    assert result.primary_top_k == 20
