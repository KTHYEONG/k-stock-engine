"""Stock intents CLI: parse allocations, invoke generate_intents, serialize."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation
from src.stocks.workflows.generate_intents import generate_intents


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate trade intents from allocations")
    parser.add_argument("--allocations", required=True, type=Path)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execution-time", type=datetime.fromisoformat, default=None)
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
    execution_time = parsed.execution_time or decision_time
    intents = generate_intents(
        allocations,
        strategy_id=parsed.strategy_id,
        decision_time=decision_time,
        execution_time=execution_time,
    )
    rows = [
        {
            "intent_id": i.intent_id,
            "asset_kind": i.asset_kind.value,
            "instrument_id": i.instrument_id,
            "target_value": i.target_value,
            "decision_time": i.decision_time.isoformat(),
            "execution_time": i.execution_time.isoformat(),
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


if __name__ == "__main__":
    raise SystemExit(main())
