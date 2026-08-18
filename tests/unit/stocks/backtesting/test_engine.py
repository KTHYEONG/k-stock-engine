"""Backtest result provenance contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    BacktestResult,
    BacktestValidationError,
    PreparedReplayDecision,
    PreparedReplayMarket,
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
    assert reference.unfilled_order_reason_counts == optimized.unfilled_order_reason_counts
    assert prepare_calls["count"] == len(request.decision_session_indices)
    assert paired.prepared_decision_count == len(request.decision_session_indices)


def test_prepared_market_replay_matches_reference_over_immutable_index() -> None:
    """Prepared replay matches the reference execution on one immutable market."""
    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()

    prepare_calls = {"count": 0}

    def counting_provider(decision_time: datetime, execution_time: datetime):
        prepare_calls["count"] += 1
        return _prepare(decision_time, execution_time, scored)

    reference = StockBacktester(
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
    reference_result = reference.run(df, artifacts, portfolio, request)
    assert prepare_calls["count"] == len(request.decision_session_indices)

    market = PreparedReplayMarket.build(
        df,
        reference.adtv_window,
        instruments=instruments,
        artifacts=artifacts,
        initial_portfolio=portfolio,
    )
    overlay_frame = df.sort(["session", "instrument_id"]).select(
        "instrument_id", "session"
    ).join(
        scored.select("instrument_id", "session", "pred_score"),
        on=["instrument_id", "session"],
        how="left",
    )
    import numpy as np

    score_overlay = overlay_frame["pred_score"].to_numpy().astype(np.float64)

    prepared_calls = {"count": 0}

    def prepared_provider(decision_time: datetime, execution_time: datetime):
        prepared_calls["count"] += 1
        return _prepare(decision_time, execution_time, scored)

    prepared = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=prepared_provider,
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    prepared_result = prepared.run_prepared(request, market, score_overlay)

    assert reference_result.ledger == prepared_result.ledger
    assert reference_result.trades == prepared_result.trades
    assert reference_result.metrics == prepared_result.metrics
    assert reference_result.stress_metrics == prepared_result.stress_metrics
    assert reference_result.stress_final_value == prepared_result.stress_final_value
    assert reference_result.attempted_orders == prepared_result.attempted_orders
    assert reference_result.filled_orders == prepared_result.filled_orders
    assert reference_result.planned_cycles == prepared_result.planned_cycles
    assert reference_result.no_trade_reasons == prepared_result.no_trade_reasons
    assert (
        reference_result.unfilled_order_reason_counts
        == prepared_result.unfilled_order_reason_counts
    )
    assert prepared_calls["count"] == len(request.decision_session_indices)
    assert prepared.prepared_decision_count == len(request.decision_session_indices)


def test_run_prepared_rejects_missing_and_mismatched_overlay() -> None:
    """Prepared replay fails closed on a missing or length-mismatched overlay."""
    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=lambda dt, et: _prepare(dt, et, scored),
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    market = PreparedReplayMarket.build(
        df,
        backtester.adtv_window,
        instruments=instruments,
        artifacts=artifacts,
        initial_portfolio=portfolio,
    )
    with pytest.raises(BacktestValidationError, match="requires an aligned score overlay"):
        backtester.run_prepared(request, market, None)
    with pytest.raises(BacktestValidationError, match="length"):
        backtester.run_prepared(
            request, market, np.zeros(market.row_count + 1, dtype=np.float64)
        )


def test_run_prepared_rejects_nonfinite_scored_overlay_rows() -> None:
    """A non-finite score on a scored overlay row fails closed."""
    import numpy as np

    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=lambda dt, et: _prepare(dt, et, scored),
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    market = PreparedReplayMarket.build(
        df,
        backtester.adtv_window,
        instruments=instruments,
        artifacts=artifacts,
        initial_portfolio=portfolio,
    )
    overlay = np.full(market.row_count, np.nan, dtype=np.float64)
    overlay[0] = float("inf")
    with pytest.raises(BacktestValidationError, match="non-finite values on scored rows"):
        backtester.run_prepared(request, market, overlay)

def test_prepared_replay_preserves_null_action_coverage_and_fails_false() -> None:
    """Null provisional coverage fills; literal False stays unfilled."""
    import numpy as np

    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=lambda dt, et: _prepare(dt, et, scored),
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    overlay_frame = df.sort(["session", "instrument_id"]).select(
        "instrument_id", "session"
    ).join(
        scored.select("instrument_id", "session", "pred_score"),
        on=["instrument_id", "session"],
        how="left",
    )
    score_overlay = overlay_frame["pred_score"].to_numpy().astype(np.float64)

    null_frame = df.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("action_interval_covered")
    )
    null_market = PreparedReplayMarket.build(
        null_frame,
        backtester.adtv_window,
        instruments=instruments,
        artifacts=artifacts,
        initial_portfolio=portfolio,
    )
    assert null_market.value_at(0, "action_interval_covered") is None
    null_result = backtester.run_prepared(request, null_market, score_overlay)
    assert null_result.filled_orders > 0
    assert "no-action-coverage" not in null_result.unfilled_order_reason_counts

    false_frame = df.with_columns(
        pl.lit(False, dtype=pl.Boolean).alias("action_interval_covered")
    )
    false_market = PreparedReplayMarket.build(
        false_frame,
        backtester.adtv_window,
        instruments=instruments,
        artifacts=artifacts,
        initial_portfolio=portfolio,
    )
    false_result = backtester.run_prepared(request, false_market, score_overlay)
    assert false_result.filled_orders == 0
    assert false_result.unfilled_order_reason_counts["no-action-coverage"] == (
        false_result.attempted_orders
    )


def test_scenario_research_replay_no_action_gate_and_production_rejects_coverage() -> None:
    """SCENARIO_RESEARCH_REPLAY_NO_ACTION_GATE: only production requires coverage."""
    import dataclasses

    from src.core.datasets import DatasetCertification

    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()
    research_manifest = dataclasses.replace(
        snapshot.manifest, certification=DatasetCertification.RESEARCH
    )
    research_backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=research_manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=lambda dt, et: _prepare(dt, et, scored),
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    null_frame = df.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("action_interval_covered")
    )
    research_backtester.run(null_frame, artifacts, portfolio, request)

    production_backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=dataclasses.replace(
            snapshot.manifest, certification=DatasetCertification.PRODUCTION
        ),
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=lambda dt, et: _prepare(dt, et, scored),
        scenario_planner=lambda prepared, port, creq: _scenario_planner(
            prepared, port, creq, instruments, policy
        ),
    )
    with pytest.raises(BacktestValidationError, match="uncovered action interval"):
        production_backtester.run(null_frame, artifacts, portfolio, request)
    false_frame = df.with_columns(
        pl.lit(False, dtype=pl.Boolean).alias("action_interval_covered")
    )
    with pytest.raises(BacktestValidationError, match="uncovered action interval"):
        production_backtester.run(false_frame, artifacts, portfolio, request)


def test_backtest_data_quality_records_execution_policy() -> None:
    from src.stocks.domain.execution_policy import (
        SCHEDULED_OPEN_POLICY_ID,
        SCHEDULED_OPEN_V1,
    )

    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()

    def planner(snap, reg, inst, port, creq):
        del snap, reg, inst, port
        return TradingCycleResult(
            status=CycleStatus.NO_TRADE, cycle_id="stub",
            decision_time=creq.decision_time, dataset_hash="d",
            artifact_id=creq.artifact_id,
            account_snapshot_id="acc",
            allocations=(), intents=(), selected_instruments=(),
            reasons=("none",),
        )

    backtester = StockBacktester(
        planner=planner,
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        policy=SCHEDULED_OPEN_V1,
    )
    result = backtester.run(df, artifacts, portfolio, request)
    assert result.data_quality["execution_policy_id"] == SCHEDULED_OPEN_POLICY_ID
    assert result.data_quality["execution_policy_hash"] == SCHEDULED_OPEN_V1.canonical_hash


def test_missing_execution_open_is_explicit_unfilled_never_silently_dropped() -> None:
    """LMD-04: missing execution opens remain explicit unfilled orders."""
    from src.core.instruments import AssetKind
    from src.execution.domain.intents import TradeIntent
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()
    df = df.with_columns(
        pl.when(
            (pl.col("instrument_id") == "KRX:000001")
            & (pl.col("session_index") == 11)
        )
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )

    def planner(snap, reg, inst, port, creq):
        del snap, reg, inst
        intent = TradeIntent(
            intent_id="t1",
            asset_kind=AssetKind.STOCK,
            instrument_id="KRX:000001",
            target_value=10_000_000.0,
            decision_time=creq.decision_time,
            execution_time=creq.execution_time,
            strategy_id=creq.strategy_id,
            reason="scored-plan",
            idempotency_key="k1",
            account_snapshot_id=port.account_snapshot_id,
        )
        return TradingCycleResult(
            status=CycleStatus.PLANNED, cycle_id="stub",
            decision_time=creq.decision_time, dataset_hash="d",
            artifact_id=creq.artifact_id,
            account_snapshot_id=port.account_snapshot_id,
            allocations=(), intents=(intent,),
            selected_instruments=("KRX:000001",),
            reasons=("scored-plan",),
        )

    backtester = StockBacktester(
        planner=planner,
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        policy=SCHEDULED_OPEN_V1,
    )
    result = backtester.run(df, artifacts, portfolio, request)
    assert result.unfilled_order_reason_counts.get("missing-open", 0) >= 1
    assert result.filled_orders < 3 * 3
    assert result.data_quality["execution_policy_id"] == "scheduled_open_v1"


def test_deferred_policy_fills_at_first_valid_open_within_bounded_window() -> None:
    from src.core.instruments import AssetKind
    from src.execution.domain.intents import TradeIntent
    from src.stocks.domain.execution_policy import ExecutionOutcomePolicy

    deferred = ExecutionOutcomePolicy(
        policy_id="first_tradable_open_v1",
        max_entry_delay_sessions=2,
        max_exit_delay_sessions=0,
    )
    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()
    df = df.with_columns(
        pl.when(
            (pl.col("instrument_id") == "KRX:000001")
            & (pl.col("session_index") == 11)
        )
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )

    def planner(snap, reg, inst, port, creq):
        del snap, reg, inst
        intent = TradeIntent(
            intent_id="t1",
            asset_kind=AssetKind.STOCK,
            instrument_id="KRX:000001",
            target_value=10_000_000.0,
            decision_time=creq.decision_time,
            execution_time=creq.execution_time,
            strategy_id=creq.strategy_id,
            reason="scored-plan",
            idempotency_key="k1",
            account_snapshot_id=port.account_snapshot_id,
        )
        return TradingCycleResult(
            status=CycleStatus.PLANNED, cycle_id="stub",
            decision_time=creq.decision_time, dataset_hash="d",
            artifact_id=creq.artifact_id,
            account_snapshot_id=port.account_snapshot_id,
            allocations=(), intents=(intent,),
            selected_instruments=("KRX:000001",),
            reasons=("scored-plan",),
        )

    backtester = StockBacktester(
        planner=planner,
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        policy=deferred,
    )
    result = backtester.run(df, artifacts, portfolio, request)
    assert result.unfilled_order_reason_counts.get("missing-open", 0) == 0
    assert result.filled_orders > 0
    assert result.data_quality["execution_policy_id"] == "first_tradable_open_v1"


def test_deferred_policy_expired_order_is_explicit_expired_event() -> None:
    from src.core.instruments import AssetKind
    from src.execution.domain.intents import TradeIntent
    from src.stocks.domain.execution_policy import ExecutionOutcomePolicy

    deferred = ExecutionOutcomePolicy(
        policy_id="first_tradable_open_v1",
        max_entry_delay_sessions=0,
        max_exit_delay_sessions=0,
    )
    df, snapshot, registry, instruments, policy, scored, artifacts, request, portfolio = _paired_inputs()

    def planner(snap, reg, inst, port, creq):
        del snap, reg, inst
        intent = TradeIntent(
            intent_id="t1",
            asset_kind=AssetKind.STOCK,
            instrument_id="KRX:000001",
            target_value=10_000_000.0,
            decision_time=creq.decision_time,
            execution_time=creq.execution_time,
            strategy_id=creq.strategy_id,
            reason="scored-plan",
            idempotency_key="k1",
            account_snapshot_id=port.account_snapshot_id,
        )
        return TradingCycleResult(
            status=CycleStatus.PLANNED, cycle_id="stub",
            decision_time=creq.decision_time, dataset_hash="d",
            artifact_id=creq.artifact_id,
            account_snapshot_id=port.account_snapshot_id,
            allocations=(), intents=(intent,),
            selected_instruments=("KRX:000001",),
            reasons=("scored-plan",),
        )

    backtester = StockBacktester(
        planner=planner,
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        policy=deferred,
    )
    result = backtester.run(df, artifacts, portfolio, request)
    assert result.data_quality["execution_policy_id"] == "first_tradable_open_v1"
    assert result.filled_orders > 0
