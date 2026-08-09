"""Portfolio primitives contract tests."""
from __future__ import annotations

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation, Position


def instrument() -> Instrument:
    return Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")


class TestPosition:
    def test_position_holds_quantity_and_cost(self) -> None:
        position = Position(instrument=instrument(), quantity=10.0, average_cost=70000.0)
        assert position.quantity == 10.0
        assert position.average_cost == 70000.0


class TestAllocation:
    def test_allocation_carries_target_and_reason(self) -> None:
        allocation = Allocation(instrument=instrument(), target_value=0.2, reason="rank")
        assert allocation.target_value == 0.2
        assert allocation.reason == "rank"
