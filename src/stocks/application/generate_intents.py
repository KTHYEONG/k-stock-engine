"""Stock application: turn allocations into TradeIntents for execution."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation
from src.execution.domain.intent import TradeIntent


def generate_intents(
    allocations: Sequence[Allocation],
    *,
    strategy_id: str,
    decision_time: datetime,
    execution_time: datetime,
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
            )
        )
    return intents


def main(args: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate trade intents from allocations")
    parser.add_argument("--allocations", required=True, type=Path)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parsed = parser.parse_args(args)

    frame = pl.read_parquet(parsed.allocations)
    allocations = [
        Allocation(
            instrument=_instrument_from_row(row),
            target_value=float(row["target_value"]),
            reason=str(row.get("reason", "")),
        )
        for row in frame.iter_rows(named=True)
    ]
    decision_time = datetime.fromisoformat(str(frame["decision_time"][0]))
    intents = generate_intents(
        allocations,
        strategy_id=parsed.strategy_id,
        decision_time=decision_time,
        execution_time=decision_time,
    )
    rows = [
        {
            "intent_id": i.intent_id,
            "asset_kind": i.asset_kind.value,
            "instrument_id": i.instrument_id,
            "target_value": i.target_value,
            "decision_time": i.decision_time.isoformat(),
            "strategy_id": i.strategy_id,
            "reason": i.reason,
            "idempotency_key": i.idempotency_key,
        }
        for i in intents
    ]
    pl.DataFrame(rows).write_parquet(parsed.out)
    return 0


def _instrument_from_row(row: dict[str, object]) -> Instrument:
    return Instrument(
        instrument_id=str(row["instrument_id"]),
        asset_kind=AssetKind(str(row["asset_kind"])),
        exchange="KRX",
        symbol=str(row["instrument_id"]).split(":")[-1],
        currency="KRW",
    )
