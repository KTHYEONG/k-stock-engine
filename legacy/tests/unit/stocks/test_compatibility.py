from __future__ import annotations

import pytest

from legacy.stocks.compatibility import parse_execution_utility
from legacy.stocks.trading.policy import ExecutionUtility


def test_old_execution_alias_maps_to_semantic_enum() -> None:
    assert parse_execution_utility("sparse_hold_replace_v2") is ExecutionUtility.SPARSE_HOLD_REPLACE


def test_unknown_execution_alias_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown execution"):
        parse_execution_utility("unknown")
