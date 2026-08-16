"""Immutable execution-outcome policy contract tests."""
from __future__ import annotations

import pytest

from src.stocks.domain.execution_policy import (
    FIRST_TRADABLE_OPEN_POLICY_ID,
    SCHEDULED_OPEN_POLICY_ID,
    SCHEDULED_OPEN_V1,
    ExecutionOutcomePolicy,
)


def test_scheduled_open_v1_is_default_and_fully_fail_closed() -> None:
    policy = SCHEDULED_OPEN_V1
    assert policy.policy_id == SCHEDULED_OPEN_POLICY_ID
    assert policy.entry_offset_sessions == 1
    assert policy.max_entry_delay_sessions == 0
    assert policy.max_exit_delay_sessions == 0
    assert policy.permits_deferral is False
    assert policy.canonical_hash


def test_canonical_hash_is_deterministic_and_semantic() -> None:
    first = ExecutionOutcomePolicy(policy_id=SCHEDULED_OPEN_POLICY_ID)
    second = ExecutionOutcomePolicy(policy_id=SCHEDULED_OPEN_POLICY_ID)
    assert first.canonical_hash == second.canonical_hash
    deferred = ExecutionOutcomePolicy(
        policy_id=FIRST_TRADABLE_OPEN_POLICY_ID,
        max_entry_delay_sessions=2,
        max_exit_delay_sessions=5,
    )
    assert deferred.canonical_hash != first.canonical_hash


def test_deferred_policy_requires_explicit_non_negative_delays() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        ExecutionOutcomePolicy(
            policy_id=FIRST_TRADABLE_OPEN_POLICY_ID, max_entry_delay_sessions=-1
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        ExecutionOutcomePolicy(
            policy_id=FIRST_TRADABLE_OPEN_POLICY_ID, max_exit_delay_sessions=-3
        )


def test_scheduled_policy_must_not_carry_delay_bounds() -> None:
    with pytest.raises(ValueError, match="fully fail-closed"):
        ExecutionOutcomePolicy(
            policy_id=SCHEDULED_OPEN_POLICY_ID, max_exit_delay_sessions=2
        )


def test_permits_deferral_only_when_a_delay_bound_is_set() -> None:
    assert ExecutionOutcomePolicy(
        policy_id=FIRST_TRADABLE_OPEN_POLICY_ID, max_entry_delay_sessions=1
    ).permits_deferral
    assert ExecutionOutcomePolicy(
        policy_id=FIRST_TRADABLE_OPEN_POLICY_ID, max_exit_delay_sessions=4
    ).permits_deferral
    assert not ExecutionOutcomePolicy(
        policy_id=FIRST_TRADABLE_OPEN_POLICY_ID
    ).permits_deferral


def test_rejects_empty_policy_id_and_bad_entry_offset() -> None:
    with pytest.raises(ValueError, match="policy_id must be non-empty"):
        ExecutionOutcomePolicy(policy_id="")
    with pytest.raises(ValueError, match="entry_offset_sessions"):
        ExecutionOutcomePolicy(policy_id=SCHEDULED_OPEN_POLICY_ID, entry_offset_sessions=0)
