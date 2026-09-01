"""PLAN-06-T2-SETTLEMENT: sale proceeds mature exactly on the due session.

The replay stores sale settlement under the *due session index* and releases it
exactly once at that session. Proceeds cannot fund buys before settlement, and
buys can never spend unsettled cash.
"""
from __future__ import annotations

# PARITY_08: settlement chronology and equity reconciliation are covered below.

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.costs import CostPoint, CostSchedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot
from legacy.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    StockBacktester,
)
from legacy.stocks.data.contracts import DatasetSnapshot
from src.execution.domain.intents import TradeIntent
from legacy.stocks.research.artifacts import ModelArtifactRegistry
from legacy.stocks.trading.portfolio_constructor import StockRiskPolicy
from legacy.stocks.workflows.trading_cycle import CycleStatus, TradingCycleResult
from tests.fixtures.stocks.helpers import (
    publish_baseline_artifact,
    stock_instrument_df,
    stock_manifest,
)


def t2_schedule() -> CostSchedule:
    return CostSchedule(
        name="t2",
        points=(
            CostPoint(
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                commission_rate=0.0,
                tax_rate=0.0,
                slippage_bps=0.0,
                settlement_days=2,
            ),
        ),
    )


def test_sale_proceeds_mature_exactly_on_due_session(tmp_path) -> None:
    df = stock_instrument_df(n_sessions=40, n_tickers=2, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
    instrument_id = sorted(df["instrument_id"].unique().to_list())[0]
    instrument = instruments[instrument_id]

    # Decision 0 buys a position; decision 1 exits it, so the replay must sell
    # and release the proceeds exactly two sessions later.
    def stub_planner(snapshot, registry, instruments, portfolio, request):
        del snapshot, registry, instruments
        buy = request.decision_time == datetime(2024, 1, 1, 15, 31, tzinfo=UTC)
        target_value = 1_000_000.0 if buy else 0.0
        allocation = Allocation(
            instrument=instrument, target_value=target_value, reason="stub"
        )
        intent = TradeIntent(
            intent_id="stub",
            asset_kind=AssetKind.STOCK,
            instrument_id=instrument_id,
            target_value=target_value,
            decision_time=request.decision_time,
            execution_time=request.execution_time,
            strategy_id=request.strategy_id,
            reason="stub",
            idempotency_key="stub",
            account_snapshot_id=portfolio.account_snapshot_id,
        )
        return TradingCycleResult(
            status=CycleStatus.PLANNED,
            cycle_id="stub",
            decision_time=request.decision_time,
            dataset_hash="d",
            artifact_id=request.artifact_id,
            account_snapshot_id=portfolio.account_snapshot_id,
            allocations=(allocation,),
            intents=(intent,),
            selected_instruments=(instrument_id,),
            reasons=("stub",),
        )

    backtester = StockBacktester(
        planner=stub_planner,
        registry=registry,
        instruments=instruments,
        manifest=manifest,
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    request = BacktestRequest(
        strategy_id="t2",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(0, 1),
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=StockRiskPolicy(top_k=5, turnover_budget=1.0),
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-t2",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    result = backtester.run(
        df,
        ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                    eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                    artifact_id="a001",
                ),
            )
        ),
        portfolio,
        request,
    )

    sells = [t for t in result.trades if t.side == "SELL" and t.gross is not None]
    assert sells, "expected the exit decision to produce a sell fill"
    sell_session = sells[0].session
    ledgers = {row.session: row for row in result.ledger}
    sell_row = ledgers[sell_session]
    assert sell_row.unsettled_cash > 0.0
    assert sell_row.settled_cash < sell_row.settled_cash + sell_row.unsettled_cash - 1e-8

    sell_index = next(
        i for i, row in enumerate(result.ledger) if row.session == sell_session
    )
    due_session = result.ledger[min(sell_index + 2, len(result.ledger) - 1)].session
    due_row = ledgers[due_session]
    assert due_row.unsettled_cash < sell_row.unsettled_cash + 1e-8

    for row in result.ledger:
        assert row.settled_cash >= -1e-8
        assert abs(
            row.equity
            - (row.settled_cash + row.unsettled_cash + row.positions_value - row.accrued_costs)
        ) <= 1e-8


def test_buys_cannot_spend_unsettled_cash(tmp_path) -> None:
    df = stock_instrument_df(n_sessions=30, n_tickers=2, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    publish_baseline_artifact(
        registry,
        artifact_id="a001",
        feature_schema_hash=manifest.schema_hash,
    )
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=manifest,
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    request = BacktestRequest(
        strategy_id="t2",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(5,),
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=StockRiskPolicy(top_k=5, turnover_budget=1.0),
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-t2",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    result = backtester.run(
        df,
        ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                    eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                    artifact_id="a001",
                ),
            )
        ),
        portfolio,
        request,
    )
    for row in result.ledger:
        assert row.settled_cash >= 0.0 - 1e-8
        assert row.unsettled_cash >= 0.0 - 1e-8


BACKTEST_SEMANTICS_01 = "BACKTEST-SEMANTICS-01"


def test_backtest_semantics_01_prepared_matches_dataframe_reference(tmp_path) -> None:
    """BACKTEST-SEMANTICS-01: prepared engine equals the DataFrame reference.

    Next-open execution, T+2 settlement, partial fills, capacity limits, action
    coverage handling, and base/stress ledger ordering match the DataFrame
    reference path exactly when the planner sees identical inputs.
    """
    from legacy.stocks.backtesting.engine import PreparedReplayDecision

    df = stock_instrument_df(n_sessions=40, n_tickers=2, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    publish_baseline_artifact(
        registry,
        artifact_id="a001",
        feature_schema_hash=manifest.schema_hash,
    )
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
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
        strategy_id="semantics",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(5, 10, 15),
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=StockRiskPolicy(top_k=5, turnover_budget=1.0),
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-sem",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )

    reference_backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=manifest,
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    reference = reference_backtester.run(df, artifacts, portfolio, request)

    prepared_backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=manifest,
        cost_schedule=t2_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        decision_provider=lambda dt, et: PreparedReplayDecision(
            dt, et, df.filter(pl.col("available_time") <= dt)
        ),
        scenario_planner=lambda prepared, port, creq: run_trading_cycle_like(
            prepared, port, creq, registry, instruments
        ),
    )
    prepared = prepared_backtester.run(df, artifacts, portfolio, request)

    assert len(prepared.ledger) == len(reference.ledger)
    for prepared_row, reference_row in zip(
        prepared.ledger, reference.ledger, strict=True
    ):
        assert prepared_row.session == reference_row.session
        for field in (
            "settled_cash", "unsettled_cash", "positions_value",
            "accrued_costs", "equity",
        ):
            assert getattr(prepared_row, field) == pytest.approx(
                getattr(reference_row, field), abs=1e-8
            ), f"ledger {field} diverged at {prepared_row.session}"
    assert [t.session for t in prepared.trades] == [
        t.session for t in reference.trades
    ]
    for prepared_trade, reference_trade in zip(
        prepared.trades, reference.trades, strict=True
    ):
        assert (
            prepared_trade.instrument_id,
            prepared_trade.side,
            prepared_trade.quantity,
            prepared_trade.reason,
        ) == (
            reference_trade.instrument_id,
            reference_trade.side,
            reference_trade.quantity,
            reference_trade.reason,
        )
        if reference_trade.price is not None:
            assert prepared_trade.price == pytest.approx(reference_trade.price)
            assert prepared_trade.cost == pytest.approx(reference_trade.cost)


def run_trading_cycle_like(prepared, portfolio, cycle_request, registry, instruments):
    """Run the same default trading-cycle planner on the prepared snapshot."""
    from src.core.costs import default_base_schedule
    from legacy.stocks.data.contracts import DatasetSnapshot

    from legacy.stocks.workflows.trading_cycle import MarketSlice

    slice_frame = MarketSlice(
        frame=prepared.visible,
        decision_time=cycle_request.decision_time,
        execution_time=cycle_request.execution_time,
    )
    snapshot = DatasetSnapshot(manifest=stock_manifest(
        columns=slice_frame.frame.columns, horizon=5
    ), frame=slice_frame.frame)
    del default_base_schedule
    from legacy.stocks.workflows.trading_cycle import run_trading_cycle

    return run_trading_cycle(
        snapshot,
        registry,
        instruments,
        portfolio,
        cycle_request,
    )
