"""Stock intents CLI: run a planning cycle, generate intents, serialize.

Two paths are supported:

- ``--cycle``: load a snapshot and portfolio, invoke the pure
  ``run_trading_cycle`` planner, and derive target-position intents from its
  allocations;
- ``--allocations``: the legacy path that reads a precomputed allocation frame
  and turns it into intents without a planner.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.settings import REFERENCE_DATETIME
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.generate_intents import generate_intents
from src.stocks.workflows.trading_cycle import (
    TradingCycleRequest,
    run_trading_cycle,
)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate trade intents from allocations")
    parser.add_argument("--allocations", type=Path, default=None)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execution-time", type=datetime.fromisoformat, default=None)
    parser.add_argument("--cycle", action="store_true", help="run a full planning cycle")
    parser.add_argument("--snapshot", type=Path, default=None, help="snapshot parquet")
    parser.add_argument("--registry", type=Path, default=None, help="artifact registry root")
    parser.add_argument("--artifact-id", default=None)
    parser.add_argument("--dataset-id", default="cycle")
    parser.add_argument("--decision-time", type=datetime.fromisoformat, default=None)
    parser.add_argument("--account-snapshot-id", default="paper")
    parsed = parser.parse_args(args)

    if parsed.cycle:
        if parsed.snapshot is None or parsed.registry is None or parsed.artifact_id is None:
            parser.error("--cycle requires --snapshot, --registry, and --artifact-id")
        cycle = run_trading_cycle(
            _snapshot(parsed.snapshot, parsed.artifact_id),
            ModelArtifactRegistry(parsed.registry),
            {},
            _portfolio(parsed),
            TradingCycleRequest(
                strategy_id=parsed.strategy_id,
                artifact_id=parsed.artifact_id,
                dataset_id=parsed.dataset_id,
                decision_time=parsed.decision_time or REFERENCE_DATETIME,
                execution_time=parsed.execution_time or REFERENCE_DATETIME,
                risk_policy=StockRiskPolicy(),
                mode="plan",
            ),
        )
        intents = generate_intents(
            cycle.allocations,
            strategy_id=parsed.strategy_id,
            decision_time=cycle.decision_time,
            execution_time=parsed.execution_time or cycle.decision_time,
            account_snapshot_id=parsed.account_snapshot_id,
        )
    else:
        frame = pl.read_parquet(parsed.allocations)
        allocations = [
            Allocation(
                instrument=_instrument_from_row(row),
                target_value=float(row["target_value"]),
                reason=str(row.get("reason", "")),
                target_quantity=int(row["target_quantity"]) if row.get("target_quantity") is not None else None,
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
            account_snapshot_id=parsed.account_snapshot_id,
        )
    rows = [
        {
            "intent_id": i.intent_id,
            "asset_kind": i.asset_kind.value,
            "instrument_id": i.instrument_id,
            "target_value": i.target_value,
            "target_quantity": i.target_quantity,
            "decision_time": i.decision_time.isoformat(),
            "execution_time": i.execution_time.isoformat(),
            "strategy_id": i.strategy_id,
            "reason": i.reason,
            "idempotency_key": i.idempotency_key,
            "account_snapshot_id": i.account_snapshot_id,
        }
        for i in intents
    ]
    pl.DataFrame(rows).write_parquet(parsed.out)
    return 0


def _snapshot(path: Path, artifact_id: str) -> DatasetSnapshot:
    from src.core.datasets import make_manifest

    frame = pl.read_parquet(path)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="stock_alpha_v1",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        time_start=datetime(2024, 1, 1),
        time_end=datetime(2024, 3, 1),
        provider_version=artifact_id,
        universe_policy_version="fixture",
        row_count=frame.height,
    )
    return DatasetSnapshot(manifest=manifest, frame=frame)


def _portfolio(parsed: argparse.Namespace) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        account_snapshot_id=parsed.account_snapshot_id,
        as_of=parsed.decision_time or REFERENCE_DATETIME,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )


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
