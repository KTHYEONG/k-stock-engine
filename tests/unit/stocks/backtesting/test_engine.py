"""Backtest result provenance contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    BacktestResult,
    PreparedReplayDecision,
    StockBacktester,
)
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import (
    StockRiskPolicy,
    construct_target_allocations,
)
from src.stocks.workflows.trading_cycle import (
    CycleStatus,
    TradingCycleResult,
    _build_intents,
)
from tests.fixtures.stocks.helpers import (
    stock_instrument_df,
    stock_manifest,
)


def test_backtest_result_preserves_data_quality_evidence() -> None:
    evidence = {
        "dataset_content_hash": "dataset-hash",
        "quality_report_hash": "quality-hash",
        "master_hash": "master-hash",
        "calendar_hash": "calendar-hash",
        "action_hash": "action-hash",
        "cost_hash": "cost-hash",
    }
    result = BacktestResult(
        ledger=(),
        trades=(),
        final_value=100.0,
        total_return=0.0,
        metrics={},
        data_quality=evidence,
    )

    assert result.data_quality == evidence


def _paired_inputs():
    df = stock_instrument_df(n_sessions=80, n_tickers=3, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry("mem://paired")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    scored = df.with_columns(
        pl.col("feature_momentum_5d")
        .rank("dense")
        .over("session")
        .cast(pl.Float64)
        .alias("pred_score")
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                artifact_id="a001",
            ),
        )
    )
    request = BacktestRequest(
        strategy_id="paired",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(10, 20, 30),
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=policy,
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="promotion",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    return df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio


def _prepare(decision_time: datetime, execution_time: datetime, scored) -> PreparedReplayDecision:
    visible = scored.filter(pl.col("available_time") <= decision_time)
    return PreparedReplayDecision(decision_time, execution_time, visible)


def _scenario_planner(prepared, portfolio, cycle_request, instruments, policy):
    visible = prepared.visible
    if visible.is_empty():
        return TradingCycleResult(
            status=CycleStatus.NO_TRADE, cycle_id="stub",
            decision_time=cycle_request.decision_time, dataset_hash="d",
            artifact_id=cycle_request.artifact_id,
            account_snapshot_id=portfolio.account_snapshot_id,
            allocations=(), intents=(), selected_instruments=(),
            reasons=("empty-scored-cross-section",),
        )
    allocations = construct_target_allocations(visible, instruments, portfolio, policy)
    if not allocations:
        return TradingCycleResult(
            status=CycleStatus.NO_TRADE, cycle_id="stub",
            decision_time=cycle_request.decision_time, dataset_hash="d",
            artifact_id=cycle_request.artifact_id,
            account_snapshot_id=portfolio.account_snapshot_id,
            allocations=(), intents=(), selected_instruments=(),
            reasons=("no-feasible-allocation",),
        )
    intents = _build_intents(tuple(allocations), portfolio, cycle_request)
    return TradingCycleResult(
        status=CycleStatus.PLANNED, cycle_id="stub",
        decision_time=cycle_request.decision_time, dataset_hash="d",
        artifact_id=cycle_request.artifact_id,
        account_snapshot_id=portfolio.account_snapshot_id,
        allocations=tuple(allocations), intents=intents,
        selected_instruments=tuple(
            sorted({a.instrument.instrument_id for a in allocations})
        ),
        reasons=("scored-plan",),
    )


def test_paired_replay_matches_separate_ledgers_and_prepares_decision_once() -> None:
    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()

    def separate_planner(snapshot_inner, reg, inst, port, creq):
        del snapshot_inner, reg, inst
        prepared = _prepare(creq.decision_time, creq.execution_time, scored)
        return _scenario_planner(prepared, port, creq, instruments, policy)

    separate = StockBacktester(
        planner=separate_planner,
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    reference = separate.run(df, artifacts, portfolio, request)

    prepare_calls = {"count": 0}

    def counting_provider(decision_time: datetime, execution_time: datetime):
        prepare_calls["count"] += 1
        return _prepare(decision_time, execution_time, scored)

    paired = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=counting_provider,
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    optimized = paired.run(df, artifacts, portfolio, request)

    assert reference.ledger == optimized.ledger
    assert reference.trades == optimized.trades
    assert reference.metrics == optimized.metrics
    assert reference.stress_metrics == optimized.stress_metrics
    assert reference.stress_final_value == optimized.stress_final_value
    assert reference.attempted_orders == optimized.attempted_orders
    assert reference.filled_orders == optimized.filled_orders
    assert reference.planned_cycles == optimized.planned_cycles
    assert reference.no_trade_reasons == optimized.no_trade_reasons
    assert prepare_calls["count"] == len(request.decision_session_indices)
    assert paired.prepared_decision_count == len(request.decision_session_indices)
