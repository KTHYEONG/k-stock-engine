from __future__ import annotations

from src.stocks.trading.policy import ExecutionUtility, SizingMethod


def test_policy_modes_are_explicit() -> None:
    assert ExecutionUtility.SPARSE_HOLD_REPLACE.value.endswith("v2")
    assert SizingMethod.RISK_BALANCED_WATERFILL.value.endswith("v2")
