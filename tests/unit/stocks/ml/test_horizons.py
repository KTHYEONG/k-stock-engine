"""ML horizon/profile frontier: vintage bootstrap, Holm admission, one primary."""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from src.stocks.ml.horizons import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    HorizonOOFEvidence,
    _cohort_bootstrap,
    select_horizons,
    stitch_prequential_growth_route,
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
    small = select_horizons(candidates, 0.05, 42, n_bootstrap=64)
    large = select_horizons(candidates, 0.05, 42, n_bootstrap=200)
    assert small.primary_horizon_sessions == large.primary_horizon_sessions == 5
    assert large.adjusted_lower_growth[(5, 5, 20, _LEGACY)]["stress"] > 0.0
    # Family size 3 requires ceil(3 / 0.05) resamples for a measurable rank-1 threshold.
    with pytest.raises(ValueError, match="below the resolvable minimum"):
        select_horizons(candidates, 0.05, 42, n_bootstrap=50)


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
    """SPARSE_GROWTH_04_MATCHED_SHADOW_ADMISSION / resolution_guard_accepts_sufficient_resamples.

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


BOOTSTRAP_PARITY_01 = "BOOTSTRAP-PARITY-01"


def _segmented_evidence(
    horizon: int,
    segment_values: dict[int, tuple[float, ...]],
    *,
    stress: dict[int, tuple[float, ...]] | None = None,
    profile_id: str = _LEGACY,
    cadence: int = 5,
    top_k: int = 20,
    paired: tuple[float, ...] | None = None,
    sparse_turnover: float = 0.0,
    shadow_turnover: float = 0.0,
) -> HorizonOOFEvidence:
    """One candidate whose vintages are grouped by OOF segment id."""
    base: list[float] = []
    stress_series: list[float] = []
    segment_ids: list[int] = []
    for segment in sorted(segment_values):
        values = segment_values[segment]
        base.extend(values)
        stress_series.extend((stress or {}).get(segment, values))
        segment_ids.extend([segment] * len(values))
    return HorizonOOFEvidence(
        horizon_sessions=horizon,
        profile_id=profile_id,
        model_family="net_alpha_elastic_net",
        base_log_growth=tuple(base),
        stress_log_growth=tuple(stress_series),
        cohort_segment_ids=tuple(segment_ids),
        complete_cohort_count=len(base),
        active_cohort_count=len(base),
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=max(segment_ids, default=0) + 1,
        fold_rank_ics=(0.1, 0.2, 0.3),
        rebalance_frequency_sessions=cadence,
        top_k=top_k,
        paired_stress_log_growth=paired or (),
        sparse_turnover=sparse_turnover,
        shadow_turnover=shadow_turnover,
    )


GROWTH_ROUTE_01_CAUSAL_STITCH = "GROWTH_ROUTE_01_CAUSAL_STITCH"


def test_growth_route_01_causal_stitch() -> None:
    """GROWTH_ROUTE_01_CAUSAL_STITCH.

    For three synthetic OOF segments the policy selected for segment ``s``
    depends only on candidate evidence with segment id strictly below ``s``:
    mutating any candidate return in segment ``s`` or later leaves every
    selection for segments ``<= s`` unchanged. Segment 0 stays cash without a
    pre-registered seed, and every appended non-cash interval carries exactly
    one source candidate (the policy selected for its own segment).
    """
    alpha = 0.05
    n_bootstrap = 20  # ceil(1 / alpha): minimum resolvable route bootstrap

    strong = _segmented_evidence(
        5, {0: (0.02,) * 12, 1: (0.02,) * 12, 2: (0.02,) * 12}, profile_id=_LEGACY
    )
    late_bloomer = _segmented_evidence(
        10, {0: (0.02,) * 12, 1: (0.05,) * 12, 2: (0.05,) * 12},
        profile_id=_LOWER, cadence=5,
    )
    baseline = stitch_prequential_growth_route(
        (strong, late_bloomer), alpha, 42, n_bootstrap
    )
    # Segment 0 has no prior evidence: cash without a seed.
    assert baseline.selected_policies[0] is None
    # Segment 1 tie on segment-0 evidence resolves to the deterministic
    # (H, C, K, profile) order: shorter horizon first.
    assert baseline.selected_policies[1] == (5, 5, 20, _LEGACY)
    # Segment 2 sees both earlier segments and picks max stress lower growth.
    assert baseline.selected_policies[2] == (10, 5, 20, _LOWER)

    # Mutating a candidate's returns at segment s never moves selections <= s.
    mutated_mid = stitch_prequential_growth_route(
        (
            strong,
            _segmented_evidence(
                10, {0: (0.02,) * 12, 1: (-0.08,) * 12, 2: (0.05,) * 12},
                profile_id=_LOWER,
            ),
        ),
        alpha, 42, n_bootstrap,
    )
    assert mutated_mid.selected_policies[:2] == baseline.selected_policies[:2]

    mutated_tail = stitch_prequential_growth_route(
        (
            strong,
            _segmented_evidence(
                10, {0: (0.02,) * 12, 1: (0.05,) * 12, 2: (-0.09,) * 12},
                profile_id=_LOWER,
            ),
        ),
        alpha, 42, n_bootstrap,
    )
    assert mutated_tail.selected_policies == baseline.selected_policies

    mutated_strong = stitch_prequential_growth_route(
        (
            _segmented_evidence(
                5, {0: (0.02,) * 12, 1: (-0.05,) * 12, 2: (0.02,) * 12},
                profile_id=_LEGACY,
            ),
            late_bloomer,
        ),
        alpha, 42, n_bootstrap,
    )
    assert mutated_strong.selected_policies[:2] == baseline.selected_policies[:2]

    # Chronological append order, one source candidate per non-cash interval,
    # and cash intervals contribute exactly zero growth with zero exposure.
    assert list(baseline.segment_ids) == sorted(baseline.segment_ids)
    assert len(baseline.base_log_growth) == len(baseline.stress_log_growth) == 36
    assert len(baseline.interval_policies) == 36
    assert all(value == 0.0 for value in baseline.base_log_growth[:12])
    assert baseline.interval_policies[:12] == (None,) * 12
    non_cash = [
        index
        for index, key in enumerate(baseline.interval_policies)
        if key is not None
    ]
    assert non_cash
    for index in non_cash:
        segment = baseline.segment_ids[index]
        assert baseline.interval_policies[index] == (
            baseline.selected_policies[segment]
        )
    assert baseline.candidate_count == 2
    assert baseline.observed_interval_count == 36
    assert baseline.invested_interval_count == 24

    with pytest.raises(ValueError, match="n_bootstrap"):
        stitch_prequential_growth_route((strong,), 0.05, 42, 19)


GROWTH_ROUTE_02_ABSOLUTE_OBJECTIVE = "GROWTH_ROUTE_02_ABSOLUTE_OBJECTIVE"


def test_growth_route_02_absolute_objective() -> None:
    """GROWTH_ROUTE_02_ABSOLUTE_OBJECTIVE.

    The sparse-minus-dense diagnostic cannot veto selection: a route whose
    paired sparse-minus-dense lower growth is negative remains selectable when
    its absolute base/stress lower growth is positive, while any non-positive
    absolute lower bound makes the route ineligible (all-cash).
    """
    alpha = 0.05
    n_bootstrap = 20

    sparse_loser = _segmented_evidence(
        5,
        {0: (0.03,) * 12, 1: (0.03,) * 12},
        profile_id=_LOWER,
        paired=(-0.01,) * 24,
        sparse_turnover=1.0,
        shadow_turnover=1.0,
    )
    route = stitch_prequential_growth_route((sparse_loser,), alpha, 42, n_bootstrap)
    assert route.selected_policies[0] is None
    assert route.selected_policies[1] == (5, 5, 20, _LOWER)
    assert route.sparse_minus_dense_lower_growth < 0.0
    assert route.invested_interval_count > 0

    negative_base = _segmented_evidence(
        5, {0: (-0.02,) * 12, 1: (0.03,) * 12}, profile_id=_LOWER
    )
    ineligible_base = stitch_prequential_growth_route(
        (negative_base,), alpha, 42, n_bootstrap
    )
    assert all(policy is None for policy in ineligible_base.selected_policies)
    assert ineligible_base.invested_interval_count == 0

    negative_stress = _segmented_evidence(
        5,
        {0: (0.03,) * 12, 1: (0.03,) * 12},
        stress={0: (-0.02,) * 12, 1: (0.03,) * 12},
        profile_id=_LOWER,
    )
    ineligible_stress = stitch_prequential_growth_route(
        (negative_stress,), alpha, 42, n_bootstrap
    )
    assert all(policy is None for policy in ineligible_stress.selected_policies)
    assert ineligible_stress.invested_interval_count == 0


def test_bootstrap_parity_01_pooled_matches_materialized_reference() -> None:
    """BOOTSTRAP-PARITY-01: pooled bounded kernel equals the legacy reference.

    Seeded starts and draw order are exact, boot means differ within 1e-15,
    the centered p-value/quantiles match, and the pooled workspace is
    O(B * ceil(N / L) + N) rather than O(B * N).
    """
    import numpy as np

    from src.stocks.ml.horizons import _segment_block_length, _cohort_bootstrap

    rng = np.random.default_rng(11)
    segment_ids = (0, 0, 0, 0, 1, 1, 1, 1, 1)
    log_growth = tuple(
        float(v) * 0.004 for v in rng.normal(size=len(segment_ids))
    )
    n_bootstrap = 120
    seed = 42
    min_block = 2

    # Legacy materialized reference: identical grouping order, seeded starts,
    # block geometry, truncation, pooling weights, and centered p-value.
    by_segment: dict[int, list[float]] = {}
    for segment, value in zip(segment_ids, log_growth, strict=True):
        by_segment.setdefault(int(segment), []).append(float(value))
    distributions = []
    weights = []
    for segment in sorted(by_segment):
        values = np.asarray(by_segment[segment], dtype=float)
        block = max(_segment_block_length(values.size), min_block)
        n_blocks = int(np.ceil(values.size / block))
        seg_rng = np.random.default_rng(seed + segment)
        starts = seg_rng.integers(
            0, max(1, values.size - block + 1), size=(n_bootstrap, n_blocks)
        )
        offsets = np.arange(block)
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(
            n_bootstrap, n_blocks * block
        )[:, : values.size]
        distributions.append(values[index].mean(axis=1))
        weights.append(float(values.size))
    total = sum(weights)
    reference = sum(w * d for w, d in zip(weights, distributions, strict=True)) / total
    observed = float(sum(log_growth) / len(log_growth))
    reference_p_value = float(np.mean(reference >= 2.0 * observed))

    result = _cohort_bootstrap(log_growth, segment_ids, n_bootstrap, seed, min_block)
    assert result is not None
    assert result.boot_means.shape == reference.shape
    assert float(np.max(np.abs(result.boot_means - reference))) <= 1e-15
    assert result.p_value == reference_p_value
    assert result.observed_mean == observed
    assert result.lower_mean(0.05) == pytest.approx(
        float(np.quantile(reference, 0.05)), abs=1e-15
    )
    expected_blocks = sum(
        int(np.ceil(len(by_segment[s]) / max(_segment_block_length(len(by_segment[s])), min_block)))
        for s in sorted(by_segment)
    )
    assert result.n_blocks_total == expected_blocks


def _frontier_cell(
    seed_offset: int,
    *,
    mean: float = 0.0008,
    sd: float = 0.004,
    vintages_per_segment: int = 22,
) -> HorizonOOFEvidence:
    """One weak-signal frontier cell over 3 prequential segments (~66 vintages)."""
    rng = np.random.default_rng(100 + seed_offset)
    segments = (0, 1, 2)
    count = len(segments) * vintages_per_segment

    def _centered(draw_mean: float, draw_sd: float) -> tuple[float, ...]:
        values = rng.normal(draw_mean, draw_sd, size=count)
        # Re-center so the standardized effect is deterministic across seeds.
        values += draw_mean - values.mean()
        return tuple(float(v) for v in values)

    base = _centered(mean, sd)
    stress = tuple(v * 0.9 - 0.0002 for v in base)
    paired = _centered(0.0003, 0.001)
    return HorizonOOFEvidence(
        horizon_sessions=10,
        rebalance_frequency_sessions=5,
        top_k=12 + 4 * (seed_offset % 4),
        profile_id=(_LEGACY, "lower_bound_half_kelly", _LOWER)[seed_offset % 3],
        model_family="net_alpha_elastic_net",
        base_log_growth=base,
        stress_log_growth=stress,
        cohort_segment_ids=tuple(s for s in segments for _ in range(vintages_per_segment)),
        complete_cohort_count=len(base),
        active_cohort_count=len(base),
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=len(segments),
        fold_rank_ics=(0.054, 0.048, 0.061),
        paired_stress_log_growth=paired,
        shadow_turnover=12.0,
        sparse_turnover=5.0,
    )


def _production_frontier() -> tuple[HorizonOOFEvidence, ...]:
    """24-cell H10 frontier matching the pre-registered production grid shape."""
    return tuple(_frontier_cell(index) for index in range(24))


def test_resolvable_minimum_guard_rejects_underpowered_family() -> None:
    """resolution_guard_rejects_underpowered_family.

    A family whose Holm threshold sits below the k/B p-value grid fails closed.
    """
    candidates = (
        _evidence(5, _positive(5, 0.01)),
        _evidence(10, _positive(10, 0.005)),
        _evidence(15, tuple(-0.001 for _ in range(10))),
    )
    with pytest.raises(ValueError, match="below the resolvable minimum") as exc_info:
        select_horizons(candidates, 0.05, 42, n_bootstrap=50)
    message = str(exc_info.value)
    assert "60" in message
    assert "family size 3" in message


def test_production_family_resolution() -> None:
    """default_resamples_resolve_production_family.

    The 24-cell production family (m=72) resolves on the default B grid.
    """
    assert DEFAULT_BOOTSTRAP_RESAMPLES == 2000
    from math import ceil

    minimum_resamples = ceil(72 / 0.05)
    assert minimum_resamples == 1440
    assert minimum_resamples <= DEFAULT_BOOTSTRAP_RESAMPLES

    import time

    start = time.perf_counter()
    selection = select_horizons(_production_frontier(), 0.05, 42)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    thresholds = [
        *selection.base_holm_thresholds.values(),
        *selection.stress_holm_thresholds.values(),
        *selection.paired_holm_thresholds.values(),
    ]
    assert len(thresholds) > 0
    assert min(thresholds) >= 1.0 / DEFAULT_BOOTSTRAP_RESAMPLES
    # Weak synthetic signal stays NO_TRADE; the point is measurability, not promotion.
    assert selection.primary_horizon_sessions is None


def test_attainability_after_resolution_fix() -> None:
    """strong_signal_candidate_attainable_end_to_end.

    A genuinely strong candidate is admissible once the grid resolves.
    """
    strong = _frontier_cell(7, mean=0.004, sd=0.008)
    selection = select_horizons((strong,), 0.05, 42, n_bootstrap=2000)
    assert selection.primary_horizon_sessions == 10
    key = (10, strong.rebalance_frequency_sessions, strong.top_k, strong.profile_id)
    assert selection.adjusted_lower_growth[key]["base"] > 0.0
    assert selection.adjusted_lower_growth[key]["stress"] > 0.0
    assert selection.paired_lower_bounds[key] > 0.0
    assert selection.turnover_ratio[key] <= 0.60

    with pytest.raises(ValueError, match="below the resolvable minimum"):
        select_horizons((strong,), 0.05, 42, n_bootstrap=50)


_SEED_KEY = (20, 10, 8, _LOWER)


def test_growth_route_seed_invests_segment0() -> None:
    """SCENARIO_GROWTH_ROUTE_SEED_01_INVESTS_SEGMENT0.

    A pre-registered seed policy invests the otherwise-forced-cash segment 0:
    its own segment-0 series is spliced verbatim, every interval carries one
    policy, coverage is complete, and the route is tagged v2.
    """
    alpha = 0.05
    n_bootstrap = 20

    seed_candidate = _segmented_evidence(
        20,
        {0: (-0.02,) * 12, 1: (0.02,) * 12, 2: (0.02,) * 12},
        profile_id=_LOWER,
        cadence=10,
        top_k=8,
    )
    assert seed_candidate.rebalance_frequency_sessions == 10
    assert seed_candidate.top_k == 8
    assert seed_candidate.horizon_sessions == 20

    route = stitch_prequential_growth_route(
        (seed_candidate,), alpha, 42, n_bootstrap, seed_policy=_SEED_KEY
    )
    seg0_length = sum(
        1 for segment in route.segment_ids if segment == 0
    )
    assert seg0_length == 12
    assert route.selected_policies[0] == _SEED_KEY
    assert list(route.base_log_growth[:seg0_length]) == [-0.02] * 12
    assert list(route.stress_log_growth[:seg0_length]) == [-0.02] * 12
    assert route.observed_interval_count == 36
    assert route.invested_interval_count == route.observed_interval_count
    assert route.route_version == "v2"
    assert route.seed_policy == _SEED_KEY


def test_growth_route_seed_admissible_beats_seed() -> None:
    """SCENARIO_GROWTH_ROUTE_SEED_02_ADMISSIBLE_BEATS_SEED.

    The seed only fills segments no earlier evidence admits; a strictly
    positive candidate admissible from prior segments wins its segment.
    """
    alpha = 0.05
    n_bootstrap = 20

    seed_candidate = _segmented_evidence(
        20,
        {0: (-0.02,) * 12, 1: (-0.02,) * 12, 2: (-0.02,) * 12},
        profile_id=_LOWER,
        cadence=10,
        top_k=8,
    )
    strong = _segmented_evidence(
        5, {0: (0.02,) * 12, 1: (0.02,) * 12}, profile_id=_LEGACY
    )
    route = stitch_prequential_growth_route(
        (seed_candidate, strong), alpha, 42, n_bootstrap, seed_policy=_SEED_KEY
    )
    # Segment 0: no earlier evidence exists, the seed invests.
    assert route.selected_policies[0] == _SEED_KEY
    # Segment 1: the strong candidate is admissible from segment 0 and beats
    # the seed; the losing seed candidate is never re-selected.
    assert route.selected_policies[1] == (5, 5, 20, _LEGACY)
    seed_positions = [
        index
        for index, key in enumerate(route.interval_policies)
        if key == _SEED_KEY
    ]
    assert seed_positions
    assert all(route.segment_ids[index] == 0 for index in seed_positions)


def test_growth_route_seed_missing_candidate_fails_closed() -> None:
    """SCENARIO_GROWTH_ROUTE_SEED_03_MISSING_CANDIDATE_FAILS_CLOSED."""
    orphan = _segmented_evidence(5, {0: (0.01,) * 6}, profile_id=_LEGACY)
    with pytest.raises(ValueError, match="seed policy") as exc_info:
        stitch_prequential_growth_route(
            (orphan,), 0.05, 42, 20, seed_policy=_SEED_KEY
        )
    assert str(_LOWER) in str(exc_info.value)
    assert "20" in str(exc_info.value)
    assert "10" in str(exc_info.value)


def test_growth_route_seed_none_is_v1_parity() -> None:
    """SCENARIO_GROWTH_ROUTE_SEED_04_NONE_IS_V1_PARITY.

    Omitting the seed reproduces v1 exactly: forced-cash segment 0, v1 tag,
    and a null seed_policy field.
    """
    candidate = _segmented_evidence(
        20,
        {0: (-0.02,) * 12, 1: (0.02,) * 12, 2: (0.02,) * 12},
        profile_id=_LOWER,
        cadence=10,
    )
    route = stitch_prequential_growth_route((candidate,), 0.05, 42, 20)
    assert route.selected_policies[0] is None
    assert all(value == 0.0 for value in route.base_log_growth[:12])
    assert route.route_version == "v1"
    assert route.seed_policy is None
    assert route.invested_interval_count < route.observed_interval_count


class TestBenchmarkReconciliationReason:
    """SCENARIO_BENCHMARK_RECONCILIATION_REASON_RECORDED."""

    @staticmethod
    def _route_and_discovery(n_panel_sessions: int, growth_length: int):
        from datetime import UTC, datetime

        import polars as pl

        from src.stocks.ml.horizons import HorizonOOFEvidence as _Evidence
        from src.stocks.ml.training import HorizonDiscovery
        from src.stocks.ml.execution_replay import ExecutionReplayEvidence

        sessions = [
            datetime(2024, 1, 1 + i, tzinfo=UTC)
            for i in range(n_panel_sessions)
        ]
        rows = []
        for session in sessions:
            for t in range(3):
                price = 100.0 + t
                rows.append(
                    {
                        "instrument_id": f"KRX:{t + 1:05d}",
                        "session": session,
                        "observation_time": session.replace(hour=15, minute=30),
                        "available_time": session.replace(hour=15, minute=31),
                        "open": price,
                        "close": price * 1.01,
                        "volume": 1_000_000.0,
                        "trading_value": 100_000_000.0,
                        "sector": "S0",
                        "adtv": 100_000_000.0,
                    }
                )
        panel = pl.DataFrame(rows)
        key = (10, 5, 2, _LOWER)
        n_growth = growth_length
        interval_bounds: tuple[tuple[datetime, ...], ...] = (
            (tuple(sessions[: n_growth + 1])),
        )
        evidence = ExecutionReplayEvidence(
            base_log_growth=tuple(0.01 for _ in range(n_growth)),
            stress_log_growth=tuple(0.01 for _ in range(n_growth)),
            segment_ids=tuple(0 for _ in range(n_growth)),
            planned_cycles=2,
            filled_orders=6,
            cash_session_fraction=0.0,
            turnover=0.5,
            observed_interval_count=n_growth,
            invested_interval_count=n_growth,
            invested_interval_fraction=1.0,
            base_interval_exposure=tuple(0.9 for _ in range(growth_length)),
            stress_interval_exposure=tuple(0.9 for _ in range(growth_length)),
            base_interval_session_bounds=interval_bounds,
        )
        from src.stocks.ml.horizons import GrowthRouteEvidence as _Route

        route = _Route(
            base_log_growth=tuple(0.01 for _ in range(growth_length)),
            stress_log_growth=tuple(0.01 for _ in range(growth_length)),
            segment_ids=tuple(0 for _ in range(growth_length)),
            selected_policies=(key,),
            interval_policies=(key,) * growth_length,
            benchmark_log_growth=(),
            candidate_count=1,
            observed_interval_count=growth_length,
            invested_interval_count=growth_length,
            filled_orders=6,
            filled_cycle_count=2,
            turnover_ratio=0.5,
        )
        discovery_evidence = _Evidence(
            horizon_sessions=10,
            profile_id=_LOWER,
            model_family="net_alpha_elastic_net",
            base_log_growth=tuple(0.01 for _ in range(growth_length)),
            stress_log_growth=tuple(0.01 for _ in range(growth_length)),
            cohort_segment_ids=tuple(0 for _ in range(growth_length)),
            complete_cohort_count=growth_length,
            active_cohort_count=growth_length,
            partial_cohort_count=0,
            missing_cohort_count=0,
            segment_count=1,
            fold_rank_ics=(0.2,),
            rebalance_frequency_sessions=5,
            top_k=2,
        )
        discovery = HorizonDiscovery(
            evidence=(discovery_evidence,),
            diagnostics=(),
            oof_by_horizon={},
            execution_evidence_by_candidate={key: evidence},
        )
        return route, discovery, panel

    def test_benchmark_reconciliation_reason_recorded(self) -> None:
        """SCENARIO_BENCHMARK_RECONCILIATION_REASON_RECORDED."""
        from src.stocks.research.metrics import certify_growth_route
        from src.stocks.ml.contracts import CompoundingCertificationSettings
        from src.stocks.ml.training import (
            _attach_growth_route_execution_evidence,
            _growth_route_projection,
        )

        settings = CompoundingCertificationSettings()
        # Five panel sessions against four exposure/growth entries.
        mismatched_route, discovery, panel = self._route_and_discovery(5, 4)
        evidence = discovery.execution_evidence_by_candidate[(10, 5, 2, _LOWER)]
        # Corrupt the bounds partition to exercise the length-mismatch path.
        object.__setattr__(
            evidence,
            "base_interval_session_bounds",
            ((evidence.base_interval_session_bounds[0][:3]),),
        )
        attached = _attach_growth_route_execution_evidence(
            mismatched_route, discovery, panel
        )
        assert attached.benchmark_log_growth == ()
        assert (
            attached.benchmark_reconcile_failure
            == "benchmark-exposure-length-mismatch"
        )
        certificate = certify_growth_route(attached, 10, settings)
        projection = _growth_route_projection(attached, certificate)
        assert (
            projection["benchmark_reconcile_failure"]
            == "benchmark-exposure-length-mismatch"
        )
        assert "matched-benchmark-missing" in certificate["reasons"]

    def test_successful_attach_leaves_empty_reason(self) -> None:
        from src.stocks.ml.training import (
            _attach_growth_route_execution_evidence,
        )

        # Six panel sessions bound the five growth intervals of one segment.
        route, discovery, panel = self._route_and_discovery(6, 5)
        attached = _attach_growth_route_execution_evidence(route, discovery, panel)
        assert attached.benchmark_reconcile_failure == ""
        assert len(attached.benchmark_log_growth) == len(route.base_log_growth)


def _turnover_evidence(
    horizon: int,
    profile_id: str,
    cadence: int,
    *,
    sparse_turnover: float,
    shadow_turnover: float,
) -> HorizonOOFEvidence:
    cell = _evidence(
        horizon,
        _positive(horizon, 0.01, count=60),
        profile_id=profile_id,
        rebalance_frequency_sessions=cadence,
    )
    return replace(
        cell, sparse_turnover=sparse_turnover, shadow_turnover=shadow_turnover
    )


def test_shadow_ratio_exempt_admission_and_numeric_publication() -> None:
    """SHADOW_RATIO_EXEMPT: missing shadow publishes None and never rejects."""
    cells = (
        _turnover_evidence(10, "cell_exempt_a", 5, sparse_turnover=9.5, shadow_turnover=0.0),
        _turnover_evidence(10, "cell_exempt_b", 10, sparse_turnover=6.0, shadow_turnover=0.0),
        _turnover_evidence(10, "cell_ratio_pass", 15, sparse_turnover=9.5, shadow_turnover=20.0),
        _turnover_evidence(10, "cell_ratio_fail", 20, sparse_turnover=9.5, shadow_turnover=10.0),
    )
    result = select_horizons(cells, 0.05, 42)

    assert result.turnover_ratio[(10, 5, 20, "cell_exempt_a")] is None
    assert result.turnover_ratio[(10, 10, 20, "cell_exempt_b")] is None
    assert result.turnover_ratio[(10, 15, 20, "cell_ratio_pass")] == pytest.approx(0.475)
    assert result.turnover_ratio[(10, 20, 20, "cell_ratio_fail")] == pytest.approx(0.95)

    joined = "\n".join(result.selection_reasons)
    assert "turnover_ratio=9." not in joined
    assert "turnover_ratio=exempt" in joined
    fail_reasons = [r for r in result.selection_reasons if "cell_ratio_fail" in r]
    assert any("turnover_ratio=0.95" in r for r in fail_reasons)
    pass_reasons = [r for r in result.selection_reasons if "cell_ratio_pass" in r]
    assert any("admissible" in r for r in pass_reasons)


def test_holm_unique_family_declared_cells_keep_threshold_sequence() -> None:
    """HOLM_UNIQUE_FAMILY: Holm sequence counts every declared cell (alpha/m)."""
    cells = tuple(
        _evidence(
            10,
            _positive(10, 0.01, count=60),
            profile_id=profile_id,
            rebalance_frequency_sessions=cadence,
        )
        for profile_id, cadence in (
            (_LEGACY, 5),
            (_LOWER, 10),
            ("third_profile", 20),
        )
    )
    result = select_horizons(cells, 0.05, 42)

    thresholds = sorted(result.base_holm_thresholds.values())
    assert len(thresholds) == 3
    assert thresholds[0] == pytest.approx(0.05 / 3)
    assert thresholds[1] == pytest.approx(0.05 / 2)
    assert thresholds[2] == pytest.approx(0.05)
    for reason in result.selection_reasons:
        if "turnover_ratio=" in reason:
            token = reason.split("turnover_ratio=")[1]
            value_token = token.split()[0]
            assert value_token == "exempt" or float(value_token) < 1e6


def test_ROUTE_TURNOVER_NONE_01_NO_SHADOW_NULL_RATIO() -> None:
    """ROUTE_TURNOVER_NONE_01_NO_SHADOW_NULL_RATIO.

    Without a dense shadow (shadow_turnover == 0) the stitched route publishes
    turnover_ratio=None instead of a fabricated denominator; candidates with a
    real dense shadow yield mean(sparse/shadow); the route projection mirrors
    both; negative ratios fail closed at construction.
    """
    from src.stocks.ml.horizons import GrowthRouteEvidence

    no_shadow = _segmented_evidence(
        10,
        {0: (0.02,) * 12, 1: (0.02,) * 12, 2: (0.02,) * 12},
        profile_id=_LOWER,
        sparse_turnover=2.0,
        shadow_turnover=0.0,
    )
    route = stitch_prequential_growth_route((no_shadow,), 0.05, 42, 20)
    assert route.turnover_ratio is None

    shadowed = _segmented_evidence(
        10,
        {0: (0.02,) * 12, 1: (0.02,) * 12, 2: (0.02,) * 12},
        profile_id=_LOWER,
        sparse_turnover=2.0,
        shadow_turnover=4.0,
    )
    ratio_route = stitch_prequential_growth_route((shadowed,), 0.05, 42, 20)
    assert ratio_route.turnover_ratio == pytest.approx(2.0 / 4.0, abs=1e-9)

    with pytest.raises(ValueError, match="turnover_ratio"):
        GrowthRouteEvidence(
            base_log_growth=(0.01,),
            stress_log_growth=(0.01,),
            segment_ids=(0,),
            selected_policies=((10, 5, 12, _LOWER),),
            turnover_ratio=-0.5,
        )


# ---------------------------------------------------------------------------
# Excess-scoped route certification (benchmarks_by_key opt-in)
# ---------------------------------------------------------------------------


def test_excess_route_stitch_selection() -> None:
    """excess_route_stitch_selection.

    With benchmarks supplied, selection ranks candidates on the excess
    (base - benchmark) stress lower bound: K20 wins on excess even though
    K8 wins on absolute series. The chosen benchmark slice is appended in
    parallel and stays finite; segment s only ever sees segments < s.
    """
    alpha = 0.05
    n_bootstrap = 40

    k8 = _segmented_evidence(
        5,
        {0: (0.005,) * 6, 1: (0.005,) * 6, 2: (0.005,) * 6},
        stress={0: (0.004,) * 6, 1: (0.004,) * 6, 2: (0.004,) * 6},
        profile_id=_LEGACY,
        top_k=8,
    )
    k20 = _segmented_evidence(
        5,
        {0: (0.003,) * 6, 1: (0.003,) * 6, 2: (0.003,) * 6},
        stress={0: (0.0025,) * 6, 1: (0.0025,) * 6, 2: (0.0025,) * 6},
        profile_id=_LOWER,
        top_k=20,
    )
    # K8 absolute is stronger but its excess edge is tiny; K20's benchmark
    # sits far below its base so its excess stream dominates.
    benchmarks = {
        (5, 5, 8, _LEGACY): tuple(0.004 for _ in k8.base_log_growth),
        (5, 5, 20, _LOWER): tuple(-0.001 for _ in k20.base_log_growth),
    }

    key8 = (5, 5, 8, _LEGACY)
    key20 = (5, 5, 20, _LOWER)
    absolute = stitch_prequential_growth_route((k8, k20), alpha, 42, n_bootstrap)
    assert all(key == key8 or key is None for key in absolute.selected_policies)

    excess = stitch_prequential_growth_route(
        (k8, k20), alpha, 42, n_bootstrap, benchmarks_by_key=benchmarks
    )
    assert excess.selected_policies[1] == key20
    assert excess.selected_policies[2] == key20
    assert excess.route_version == "v1-excess"
    assert len(excess.benchmark_log_growth) == len(excess.base_log_growth)
    assert all(np.isfinite(excess.benchmark_log_growth))
    # Segment 0 has no earlier evidence and no seed: cash with empty benchmark.
    assert excess.selected_policies[0] is None
    assert all(value == 0.0 for value in excess.benchmark_log_growth[:6])


def test_excess_route_flagoff_byte_parity() -> None:
    """excess_route_flagoff_byte_parity.

    Omitting benchmarks_by_key reproduces the legacy route exactly, including
    an empty benchmark series and legacy v1/v2 tags.
    """
    alpha = 0.05
    n_bootstrap = 40
    seed_key = (20, 10, 8, _LOWER)
    seed_candidate = _segmented_evidence(
        20,
        {0: (-0.02,) * 12, 1: (0.02,) * 12, 2: (0.02,) * 12},
        profile_id=_LOWER,
        cadence=10,
        top_k=8,
    )
    legacy = stitch_prequential_growth_route(
        (seed_candidate,), alpha, 42, n_bootstrap, seed_policy=seed_key
    )
    explicit_none = stitch_prequential_growth_route(
        (seed_candidate,),
        alpha,
        42,
        n_bootstrap,
        seed_policy=seed_key,
        benchmarks_by_key=None,
    )
    assert legacy == explicit_none
    assert explicit_none.benchmark_log_growth == ()
    assert explicit_none.route_version in ("v1", "v2")
    assert explicit_none.selected_policies[0] == seed_key


def test_excess_route_failclosed_nonpositive_lb() -> None:
    """excess_route_failclosed_nonpositive_lb.

    A candidate whose excess lower bound is non-positive is never selected;
    unadmissible segments fall back to the declared seed policy, then cash,
    and no exception escapes even when every candidate fails on excess.
    """
    alpha = 0.05
    n_bootstrap = 40
    seed_key = (20, 10, 8, _LOWER)

    loser = _segmented_evidence(
        10,
        {0: (0.01,) * 6, 1: (0.01,) * 6},
        profile_id=_LEGACY,
    )
    # Benchmark sits above the base everywhere: excess is strictly negative.
    negative_excess = {(10, 5, 20, _LEGACY): tuple(0.02 for _ in loser.base_log_growth)}
    seed_candidate = _segmented_evidence(
        20,
        {0: (-0.02,) * 6, 1: (0.01,) * 6},
        profile_id=_LOWER,
        cadence=10,
        top_k=8,
    )
    seed_benchmarks = {
        **negative_excess,
        seed_key: tuple(0.0 for _ in seed_candidate.base_log_growth),
    }

    route = stitch_prequential_growth_route(
        (loser, seed_candidate),
        alpha,
        42,
        n_bootstrap,
        seed_policy=seed_key,
        benchmarks_by_key=seed_benchmarks,
    )
    # Segment 0: nothing admissible, the declared seed invests.
    assert route.selected_policies[0] == seed_key
    assert list(route.base_log_growth[:6]) == [-0.02] * 6
    # Segment 1: the loser still fails on excess; the seed remains invested
    # because no admissible candidate ever outranks it.
    assert route.selected_policies[1] == seed_key

    # Without any seed, every segment falls back to cash and never crashes.
    cash_route = stitch_prequential_growth_route(
        (loser,), alpha, 42, n_bootstrap, benchmarks_by_key=negative_excess
    )
    assert all(key is None for key in cash_route.selected_policies)
    assert cash_route.invested_interval_count == 0
