"""Deterministic v4 proxy-selection policy widths and versioning."""
from __future__ import annotations

import math

import pytest

from src.stocks.workflows.economic_selection import (
    SELECTION_POLICY_VERSION,
    ScreenFidelityPolicy,
    SelectionPolicy,
)


def test_for_budget_produces_27_to_6_to_6_to_3_profile() -> None:
    policy = ScreenFidelityPolicy.for_budget(total_trials=81, route_count=3, fold_count=3)
    assert policy.widths == (27, 6, 6, 3)
    assert policy.route_budget == 27
    assert policy.proxy_session_stride == 6
    assert policy.promotion_width == 6
    assert policy.economic_finalist_width == 3
    assert SELECTION_POLICY_VERSION == "economic-selection-v4-stability"


def test_for_budget_never_exceeds_positive_screen_candidates() -> None:
    policy = ScreenFidelityPolicy.for_budget(total_trials=3, route_count=1, fold_count=3)
    assert policy.promotion_width == 2
    assert policy.economic_finalist_width == 2


def test_proxy_stride_grows_with_sqrt_of_route_budget() -> None:
    small = ScreenFidelityPolicy.for_budget(total_trials=12, route_count=1, fold_count=3)
    assert small.proxy_session_stride == 4
    larger = ScreenFidelityPolicy.for_budget(total_trials=81, route_count=1, fold_count=3)
    assert larger.proxy_session_stride == 9
    assert larger.promotion_width == 9


def test_economic_finalist_width_is_sqrt_of_promotion_width_per_route() -> None:
    for trials in (12, 27, 81):
        policy = ScreenFidelityPolicy.for_budget(total_trials=trials, route_count=1, fold_count=3)
        assert policy.economic_finalist_width == min(
            policy.promotion_width,
            max(1, math.ceil(math.sqrt(policy.promotion_width))),
        )


def test_for_budget_rejects_non_positive_inputs() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ScreenFidelityPolicy.for_budget(total_trials=0, route_count=3, fold_count=3)
    with pytest.raises(ValueError, match="must be positive"):
        ScreenFidelityPolicy.for_budget(total_trials=81, route_count=0, fold_count=3)
    with pytest.raises(ValueError, match="must be positive"):
        ScreenFidelityPolicy.for_budget(total_trials=81, route_count=3, fold_count=0)


def test_to_json_safe_records_version_widths_and_one_policy_cell() -> None:
    payload = ScreenFidelityPolicy.for_budget(
        total_trials=81, route_count=3, fold_count=3
    ).to_json_safe()
    assert payload["selection_policy_version"] == SELECTION_POLICY_VERSION
    assert payload["route_budget"] == 27
    assert payload["proxy_session_stride"] == 6
    assert payload["promotion_width"] == 6
    assert payload["economic_finalist_width"] == 3
    assert payload["configured_compounding_policy_cells"] == 1


def test_legacy_selection_policy_stays_versioned_as_v1() -> None:
    policy = SelectionPolicy.for_budget(total_trials=81, route_count=3, fold_count=3)
    assert policy.widths == (27, 6, 2)
    assert policy.to_json_safe()["selection_policy_version"] == "economic-selection-v1"
