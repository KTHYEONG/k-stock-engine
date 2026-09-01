from __future__ import annotations

from legacy.stocks.trading.allocation import AllocationDecision


def test_allocation_decision_defaults_to_empty() -> None:
    decision = AllocationDecision()

    assert decision.allocations == ()
    assert decision.selected_count == 0
