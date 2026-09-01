"""generate_intents: allocations -> target-position intents."""
from __future__ import annotations

from datetime import UTC, datetime

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation
from legacy.stocks.workflows.generate_intents import generate_intents


def allocation(instrument_id: str = "KRX:005930", target_value: float = 0.5) -> Allocation:
    return Allocation(
        instrument=Instrument(
            instrument_id=instrument_id,
            asset_kind=AssetKind.STOCK,
            exchange="KRX",
            symbol=instrument_id.split(":")[-1],
            currency="KRW",
        ),
        target_value=target_value,
        reason="rank",
    )


class TestGenerateIntents:
    def test_intents_carry_account_snapshot_identity(self) -> None:
        decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
        intents = generate_intents(
            [allocation()],
            strategy_id="stock_alpha_v1",
            decision_time=decision,
            execution_time=decision,
            account_snapshot_id="acc-1",
        )
        assert len(intents) == 1
        assert intents[0].account_snapshot_id == "acc-1"
        assert intents[0].target_value == 0.5
        assert intents[0].asset_kind is AssetKind.STOCK

    def test_zero_target_allocation_produces_exit_intent(self) -> None:
        decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
        intents = generate_intents(
            [allocation(target_value=0.0)],
            strategy_id="stock_alpha_v1",
            decision_time=decision,
            execution_time=decision,
            account_snapshot_id="acc-1",
        )
        assert intents[0].target_value == 0.0
