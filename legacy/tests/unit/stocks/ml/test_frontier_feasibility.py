"""Frontier feasibility contract tests.

Scenarios: FRONTIER_FEASIBILITY_01.
"""
from __future__ import annotations

import pytest

from legacy.stocks.ml.contracts import ExecutionFrontierSettings


class TestFrontierFeasibility:
    """FRONTIER_FEASIBILITY_01."""

    def test_h3_with_legacy_cadence_raises_value_error(self) -> None:
        """H3_FRONTIER_CLI_02: legacy C=(5,10,20) fails for H=(3,)."""
        frontier = ExecutionFrontierSettings(
            candidate_horizon_sessions=(3,),
            candidate_rebalance_frequency_sessions=(5, 10, 20),
            candidate_top_k=(12, 16, 20, 24),
        )
        with pytest.raises(ValueError, match="H=3"):
            frontier.require_feasible_horizons(0.90, 0.08)

    def test_h3_with_valid_cadence_returns_12_cells(self) -> None:
        """H=(3,), C=(1,2,3), K=(12,16,20,24): returns exactly 12 feasible cells."""
        frontier = ExecutionFrontierSettings(
            candidate_horizon_sessions=(3,),
            candidate_rebalance_frequency_sessions=(1, 2, 3),
            candidate_top_k=(12, 16, 20, 24),
        )
        cells = frontier.require_feasible_horizons(0.90, 0.08)
        assert len(cells) == 12
        for h, c, k in cells:
            assert h == 3
            assert c <= h
            assert k >= 12

    def test_feasible_cells_works_as_before(self) -> None:
        """Existing feasible_cells method is unaffected."""
        frontier = ExecutionFrontierSettings(
            candidate_horizon_sessions=(3,),
            candidate_rebalance_frequency_sessions=(1, 2, 3),
            candidate_top_k=(12, 16, 20, 24),
        )
        cells = frontier.feasible_cells(0.90, 0.08)
        assert len(cells) == 12

    def test_h10_h20_with_default_cadence(self) -> None:
        """H=(10,20) with default cadence (5,10,20) works correctly."""
        frontier = ExecutionFrontierSettings(
            candidate_horizon_sessions=(10, 20),
            candidate_rebalance_frequency_sessions=(5, 10, 20),
            candidate_top_k=(12, 16, 20, 24),
        )
        cells = frontier.require_feasible_horizons(0.90, 0.08)
        for h, c, k in cells:
            assert c <= h
            assert k >= 12
        assert len(cells) > 0
