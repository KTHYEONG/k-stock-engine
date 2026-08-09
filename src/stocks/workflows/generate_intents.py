"""Stock workflow: turn allocations into TradeIntents for execution."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.core.portfolio import Allocation
from src.execution.domain.intents import TradeIntent


def generate_intents(
    allocations: Sequence[Allocation],
    *,
    strategy_id: str,
    decision_time: datetime,
    execution_time: datetime,
    account_snapshot_id: str,
) -> list[TradeIntent]:
    """Produce validated TradeIntent objects.

    Execution consumes approved ``TradeIntent`` objects; it does not calculate
    features, select a model, or contain strategy exit rules.
    """
    intents: list[TradeIntent] = []
    for i, allocation in enumerate(allocations):
        intents.append(
            TradeIntent(
                intent_id=f"{strategy_id}:{allocation.instrument.instrument_id}:{decision_time.isoformat()}:{i}",
                asset_kind=allocation.instrument.asset_kind,
                instrument_id=allocation.instrument.instrument_id,
                target_value=allocation.target_value,
                decision_time=decision_time,
                execution_time=execution_time,
                strategy_id=strategy_id,
                reason=allocation.reason or "score-rank-policy",
                idempotency_key=f"{strategy_id}:{allocation.instrument.instrument_id}:{decision_time.date().isoformat()}",
                account_snapshot_id=account_snapshot_id,
            )
        )
    return intents
