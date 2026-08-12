"""Deterministic multi-fidelity selection policy widths and versioning."""
from __future__ import annotations

import pytest

from src.stocks.workflows.economic_selection import (
    SELECTION_POLICY_VERSION,
    SelectionPolicy,
)


def test_for_budget_produces_27_to_6_to_2_profile() -> None:
    policy = SelectionPolicy.for_budget(total_trials=81, route_count=3, fold_count=3)
    assert policy.widths == (27, 6, 2)
    assert policy.route_budget == 27
    assert policy.promotion_width == 6
    assert policy.finalist_width == 2
    assert policy.fold_count == 3


def test_for_budget_never_exceeds_positive_screen_candidates() -> None:
    policy = SelectionPolicy.for_budget(total_trials=3, route_count=1, fold_count=3)
    assert policy.promotion_width == 2
    assert policy.finalist_width == 1


def test_for_budget_shrinks_finalists_by_fold_count() -> None:
    one_fold = SelectionPolicy.for_budget(total_trials=81, route_count=3, fold_count=1)
    assert one_fold.finalist_width == 6
    three_folds = SelectionPolicy.for_budget(total_trials=81, route_count=3, fold_count=3)
    assert three_folds.finalist_width == 2


def test_for_budget_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SelectionPolicy.for_budget(total_trials=0, route_count=3, fold_count=3)
    with pytest.raises(ValueError, match="must be positive"):
        SelectionPolicy.for_budget(total_trials=81, route_count=0, fold_count=3)
    with pytest.raises(ValueError, match="must be positive"):
        SelectionPolicy.for_budget(total_trials=81, route_count=3, fold_count=0)


def test_to_json_safe_records_version_and_widths() -> None:
    payload = SelectionPolicy.for_budget(
        total_trials=81, route_count=3, fold_count=3
    ).to_json_safe()
    assert payload["selection_policy_version"] == SELECTION_POLICY_VERSION
    assert payload["route_budget"] == 27
    assert payload["promotion_width"] == 6
    assert payload["finalist_width"] == 2
