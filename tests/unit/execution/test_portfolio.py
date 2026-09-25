"""Portfolio primitives contract tests."""
from __future__ import annotations

from src.core.instruments import AssetKind, Instrument
from src.execution.domain.portfolio import Allocation, Position


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


class TestPositionValidation:
    def test_position_rejects_negative_quantity(self) -> None:
        import pytest

        from src.execution.domain.portfolio import Position

        with pytest.raises(ValueError, match="non-negative"):
            Position(instrument=instrument(), quantity=-1.0, average_cost=1.0)

    def test_position_rejects_non_finite_quantity(self) -> None:
        import math

        import pytest

        from src.execution.domain.portfolio import Position

        with pytest.raises(ValueError, match="finite"):
            Position(instrument=instrument(), quantity=math.inf, average_cost=1.0)


class TestAllocationValidation:
    def test_allocation_rejects_negative_target_value(self) -> None:
        import pytest

        from src.execution.domain.portfolio import Allocation

        with pytest.raises(ValueError, match="non-negative finite"):
            Allocation(instrument=instrument(), target_value=-0.1)

    def test_allocation_rejects_non_integer_target_quantity(self) -> None:
        import pytest

        from src.execution.domain.portfolio import Allocation

        with pytest.raises(ValueError, match="non-negative integer"):
            Allocation(instrument=instrument(), target_value=1.0, target_quantity=True)  # type: ignore[arg-type]

    def test_allocation_rejects_negative_target_quantity(self) -> None:
        import pytest

        from src.execution.domain.portfolio import Allocation

        with pytest.raises(ValueError, match="non-negative integer"):
            Allocation(instrument=instrument(), target_value=1.0, target_quantity=-1)

    def test_allocation_rejects_off_lot_target_quantity(self) -> None:
        import pytest

        from src.core.instruments import AssetKind, Instrument
        from src.execution.domain.portfolio import Allocation

        odd_lot = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW", lot_size=10)
        with pytest.raises(ValueError, match="multiple of lot_size"):
            Allocation(instrument=odd_lot, target_value=1.0, target_quantity=3)


class TestPortfolioSnapshotValidation:
    def test_snapshot_rejects_empty_account_id(self) -> None:
        from datetime import UTC, datetime

        import pytest

        from src.execution.domain.portfolio import PortfolioSnapshot

        with pytest.raises(ValueError, match="non-empty"):
            PortfolioSnapshot(account_snapshot_id="", as_of=datetime(2024, 1, 1, tzinfo=UTC), settled_cash=0.0, unsettled_cash=0.0, positions=())

    def test_snapshot_rejects_non_finite_cash(self) -> None:
        import math
        from datetime import UTC, datetime

        import pytest

        from src.execution.domain.portfolio import PortfolioSnapshot

        with pytest.raises(ValueError, match="finite"):
            PortfolioSnapshot(account_snapshot_id="a", as_of=datetime(2024, 1, 1, tzinfo=UTC), settled_cash=math.inf, unsettled_cash=0.0, positions=())

    def test_snapshot_rejects_duplicate_positions(self) -> None:
        from datetime import UTC, datetime

        import pytest

        from src.execution.domain.portfolio import PortfolioSnapshot, Position

        position = Position(instrument=instrument(), quantity=1.0, average_cost=1.0)
        with pytest.raises(ValueError, match="duplicate position"):
            PortfolioSnapshot(account_snapshot_id="a", as_of=datetime(2024, 1, 1, tzinfo=UTC), settled_cash=0.0, unsettled_cash=0.0, positions=(position, position))

    def test_snapshot_values_holdings_at_mark_prices(self) -> None:
        from datetime import UTC, datetime

        from src.execution.domain.portfolio import PortfolioSnapshot, Position

        snapshot = PortfolioSnapshot(account_snapshot_id="a", as_of=datetime(2024, 1, 1, tzinfo=UTC), settled_cash=100.0, unsettled_cash=50.0, positions=(Position(instrument=instrument(), quantity=10.0, average_cost=5.0),))
        assert snapshot.quantity_of("KRX:005930") == 10
        assert snapshot.quantity_of("KRX:000000") == 0
        assert snapshot.equity({"KRX:005930": 7.0}) == 220.0

    def test_snapshot_equity_requires_price_for_every_holding(self) -> None:
        from datetime import UTC, datetime

        import pytest

        from src.execution.domain.portfolio import PortfolioSnapshot, Position

        snapshot = PortfolioSnapshot(account_snapshot_id="a", as_of=datetime(2024, 1, 1, tzinfo=UTC), settled_cash=0.0, unsettled_cash=0.0, positions=(Position(instrument=instrument(), quantity=1.0, average_cost=1.0),))
        with pytest.raises(ValueError, match="no price"):
            snapshot.equity({})
        with pytest.raises(ValueError, match="invalid mark price"):
            snapshot.equity({"KRX:005930": 0.0})

    def test_snapshot_rejects_newer_as_of_than_decision_time(self) -> None:
        from datetime import UTC, datetime

        import pytest

        from src.execution.domain.portfolio import PortfolioSnapshot

        snapshot = PortfolioSnapshot(account_snapshot_id="a", as_of=datetime(2024, 1, 2, tzinfo=UTC), settled_cash=0.0, unsettled_cash=0.0, positions=())
        with pytest.raises(ValueError, match="newer than decision_time"):
            snapshot.validate_as_of(datetime(2024, 1, 1, tzinfo=UTC))
        snapshot.validate_as_of(datetime(2024, 1, 2, tzinfo=UTC))
