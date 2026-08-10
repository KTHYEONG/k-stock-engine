"""Stock portfolio-simulation workflow: artifact -> cycle -> replay ledger.

The simulation workflow replays the *same* pure planner used by paper and live
paths through ``StockBacktester``, so a historical replay step and a paper
planning cycle produce identical target allocations for identical inputs.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    BacktestResult,
    StockBacktester,
)
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.costs import CostEvidence
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.contracts import SimulationRequest


def simulate_portfolio(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: SimulationRequest,
    cost_evidence: CostEvidence | None = None,
) -> BacktestResult:
    """Replay the trading cycle over the snapshot and return the ledger result.

    ``cost_evidence`` is the hash-bound cost artifact resolved from the research
    snapshot; when supplied the replay uses the dynamic liquidity slippage model
    and statutory sell taxes instead of the static base/stress schedules.
    """
    manifest = snapshot.manifest
    artifact_manifest = registry.read_manifest(request.artifact_id)
    eligible_from = datetime.fromisoformat(artifact_manifest.eligible_from)
    eligible_to = datetime.fromisoformat(artifact_manifest.eligible_to)

    frame = snapshot.frame
    sessions = sorted(frame["session"].unique().to_list())
    instruments = _instruments_from_frame(frame)
    policy = StockRiskPolicy(
        top_k=request.top_k,
        gross_cap=request.max_exposure,
        single_name_cap=request.max_single_weight,
        participation_limit=request.participation_limit,
    )
    base = request.cost_schedule or default_base_schedule()
    stress = request.stress_cost_schedule or default_stress_schedule()
    decision_indices = _decision_indices(sessions, eligible_from, eligible_to)

    initial_portfolio = PortfolioSnapshot(
        account_snapshot_id="backtest",
        as_of=datetime(2000, 1, 1, tzinfo=UTC),
        settled_cash=request.initial_cash,
        unsettled_cash=0.0,
        positions=(),
    )
    backtest_request = BacktestRequest(
        strategy_id="stock_alpha_v1",
        start_time=eligible_from,
        end_time=eligible_to,
        decision_session_indices=decision_indices,
        cost_schedule=base,
        stress_cost_schedule=stress,
        risk_policy=policy,
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=eligible_from,
                eligible_to=eligible_to,
                artifact_id=request.artifact_id,
            ),
        )
    )
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=manifest,
        cost_schedule=base,
        stress_cost_schedule=stress,
        cost_evidence=cost_evidence,
    )
    return backtester.run(
        frame, artifacts, initial_portfolio, backtest_request
    )


def _decision_indices(
    sessions: list[object],
    eligible_from: datetime,
    eligible_to: datetime,
) -> tuple[int, ...]:
    indices: list[int] = []
    for index, session in enumerate(sessions):
        if eligible_from <= cast(datetime, session) <= eligible_to:
            indices.append(index)
    return tuple(indices)


def _instruments_from_frame(frame: pl.DataFrame) -> dict[str, Instrument]:
    instruments: dict[str, Instrument] = {}
    for row in frame.select("instrument_id").unique().iter_rows(named=True):
        instrument_id = str(row["instrument_id"])
        instruments[instrument_id] = Instrument(
            instrument_id=instrument_id,
            asset_kind=AssetKind.STOCK,
            exchange="KRX",
            symbol=instrument_id.split(":")[-1],
            currency="KRW",
        )
    return instruments
