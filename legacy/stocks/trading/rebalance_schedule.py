"""Deterministic rebalance cadence kernel for v5 sparse-growth policy.

The single pure scheduling function ``rebalance_session_indices`` selects the
exact decision indices from an ordered eligible session sequence given a
frequency.  Both the OOF replay path (``_replay_costs``) and the independent
backtester (``simulate_portfolio``) consume this same kernel so the cadence is
never silently divergent.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime


def rebalance_session_indices(
    sessions: Sequence[datetime],
    eligible_from: datetime,
    eligible_to: datetime,
    frequency_sessions: int,
    *,
    legacy_daily: bool = False,
) -> tuple[int, ...]:
    """Return the ordered decision indices within the eligible window.

    ``sessions`` must be a sorted sequence of naive or aware datetimes.  Only
    sessions ``>= eligible_from`` and ``<= eligible_to`` are considered
    eligible.  From the eligible subsequence the first index ``a`` is chosen
    and every ``a + k * frequency_sessions`` (``k >= 0``) within the eligible
    subsequence is returned.

    When *legacy_daily* is ``True`` every eligible index is returned,
    preserving v1-v4 daily semantics.

    Raises ``ValueError`` when ``frequency_sessions <= 0`` (unless legacy
    daily mode is active), ``sessions`` is unsorted, or ``sessions`` contain
    naive timestamps while ``eligible_from``/``eligible_to`` are aware (or
    vice-versa).
    """
    if not sessions:
        return ()
    if not legacy_daily and frequency_sessions <= 0:
        raise ValueError(
            f"frequency_sessions must be positive, got {frequency_sessions}"
        )
    _validate_sorted(sessions)
    _validate_tz_consistency(sessions, eligible_from, eligible_to)

    eligible_indices = [
        i
        for i, session in enumerate(sessions)
        if eligible_from <= session <= eligible_to
    ]
    if not eligible_indices:
        return ()
    if legacy_daily:
        return tuple(eligible_indices)

    first = eligible_indices[0]
    eligible_set = set(eligible_indices)
    result: list[int] = []
    idx = first
    while idx <= eligible_indices[-1]:
        if idx in eligible_set:
            result.append(idx)
        idx += frequency_sessions
    return tuple(result)


def _validate_sorted(sessions: Sequence[datetime]) -> None:
    for i in range(1, len(sessions)):
        if sessions[i] < sessions[i - 1]:
            raise ValueError("sessions must be sorted in ascending order")


def _validate_tz_consistency(
    sessions: Sequence[datetime],
    eligible_from: datetime,
    eligible_to: datetime,
) -> None:
    if not sessions:
        return
    sample = sessions[0]
    from_naive = sample.tzinfo is None
    from_to_naive = eligible_from.tzinfo is None
    to_naive = eligible_to.tzinfo is None
    if from_naive != from_to_naive or from_naive != to_naive:
        raise ValueError(
            "sessions and eligible_from/eligible_to must have consistent "
            "timezone awareness"
        )
