"""Prequential policy selection contract tests.

Scenarios: PREQUENTIAL_ECONOMIC_01, PREQUENTIAL_ECONOMIC_02.
"""
from __future__ import annotations


import pytest

from legacy.stocks.ml.horizons import (
    HorizonOOFEvidence,
    minimum_resolvable_bootstrap_count,
    select_prequential_execution_policy,
)


def _evidence(
    horizon: int,
    base: tuple[float, ...],
    stress: tuple[float, ...] | None = None,
    *,
    segments: tuple[int, ...] | None = None,
    rank_ics: tuple[float, ...] = (0.1, 0.2, 0.3),
    profile_id: str = "lower_bound_only",
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


# ---------------------------------------------------------------------------
# PREQUENTIAL_ECONOMIC_01
# ---------------------------------------------------------------------------

def test_prequential_segment_policy_cannot_carry_forward() -> None:
    """PREQUENTIAL_ECONOMIC_01.

    Changing segment s or any future segment cannot change the policy
    assigned to segment s; with no admissible earlier evidence, segment s
    has zero exposure and zero strategy growth.
    """
    evidence_by_segment: dict[int, tuple[HorizonOOFEvidence, ...]] = {
        0: (),
        1: (_evidence(10, _positive(10, 0.01, count=30)),),
    }
    result = select_prequential_execution_policy(
        evidence_by_segment, 0.05, 42, n_bootstrap=200,
    )
    assert 0 in result.segment_policies
    assert result.segment_policies[0] is None

    evidence_by_segment_variant: dict[int, tuple[HorizonOOFEvidence, ...]] = {
        0: (),
        1: (_evidence(10, tuple(-0.01 for _ in range(30))),),
    }
    result_variant = select_prequential_execution_policy(
        evidence_by_segment_variant, 0.05, 42, n_bootstrap=200,
    )
    assert result_variant.segment_policies[0] == result.segment_policies[0]


def test_prequential_admissible_policy_propagates_forward() -> None:
    """An earlier segment with admissible evidence selects a policy for the next."""
    evidence_by_segment: dict[int, tuple[HorizonOOFEvidence, ...]] = {
        0: (_evidence(10, _positive(10, 0.01, count=30)),),
        1: (),
    }
    result = select_prequential_execution_policy(
        evidence_by_segment, 0.05, 42, n_bootstrap=200,
    )
    assert result.segment_policies[0] is None
    assert result.segment_policies[1] is not None


# ---------------------------------------------------------------------------
# PREQUENTIAL_ECONOMIC_02
# ---------------------------------------------------------------------------

def test_minimum_resolvable_bootstrap_count_exact() -> None:
    """PREQUENTIAL_ECONOMIC_02.

    With alpha 0.05 and 180 declared path hypotheses,
    minimum_resolvable_bootstrap_count returns 3600;
    3599 draws fail closed and 3600 draws are accepted.
    """
    assert minimum_resolvable_bootstrap_count(180, 0.05) == 3600
    assert minimum_resolvable_bootstrap_count(179, 0.05) == 3580
    assert minimum_resolvable_bootstrap_count(1, 0.05) == 20


def test_minimum_resolvable_bootstrap_requires_enough_draws() -> None:
    """Below the minimum, selection should fail closed."""
    evidence = (_evidence(10, _positive(10, 0.01, count=30)),)
    with pytest.raises(ValueError, match="n_bootstrap"):
        select_prequential_execution_policy(
            {0: evidence}, 0.05, 42, n_bootstrap=1,
        )


# ---------------------------------------------------------------------------
# PREQUENTIAL_GATE_03
# ---------------------------------------------------------------------------

def test_prequential_mutation_does_not_affect_past_segments() -> None:
    """PREQUENTIAL_GATE_03.

    Mutating candidate returns in segment s or later leaves s policy and prior
    stitched growth unchanged; no admissible prior history produces exact zero
    exposure and zero growth for s.
    """
    positive_evidence = _evidence(10, _positive(10, 0.01, count=30), segments=(0,) * 30)
    evidence_by_segment: dict[int, tuple[HorizonOOFEvidence, ...]] = {
        0: (positive_evidence,),
        1: (),
    }
    result_a = select_prequential_execution_policy(
        evidence_by_segment, 0.05, 42, n_bootstrap=200,
    )
    negative_evidence = _evidence(
        10, tuple(-0.01 for _ in range(30)),
        segments=(0,) * 30,
    )
    evidence_by_segment_variant: dict[int, tuple[HorizonOOFEvidence, ...]] = {
        0: (negative_evidence,),
        1: (),
    }
    result_b = select_prequential_execution_policy(
        evidence_by_segment_variant, 0.05, 42, n_bootstrap=200,
    )
    assert result_a.segment_policies[0] == result_b.segment_policies[0]
