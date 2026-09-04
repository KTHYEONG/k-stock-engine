"""Single-instrument smoke tests on historical 2016 market data.

Validates end-to-end execution, fill model, cost schedule, slippage,
T+2 settlement, accounting identity, and result artifact generation.
"""
from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.core.costs import (
    LiquiditySlippageModel,
    TickSizeRule,
    TickSizeSchedule,
    default_base_schedule,
)
from src.core.instruments import AssetKind, Instrument
from src.core.time import SessionCalendar
from src.data.backtest_runner import run_managed_backtest, verify_accounting_identity
from src.data.backtest_sessions import build_backtest_sessions
from src.data.schemas import SilverTable
from src.data.snapshot import PITSnapshotRepository
from src.engine.backtest import BacktestConfig, BacktestSession
from src.engine.decision import DecisionContext
from src.engine.fill_model import ExecutionScenario, HistoricalFillModel
from src.execution.domain.intents import TradeIntent

KST = ZoneInfo("Asia/Seoul")
UTC = UTC


@pytest.fixture
def samsung_2016_january_sessions(tmp_path: Path) -> tuple[tuple[BacktestSession, ...], SessionCalendar, Instrument]:
    dm_root = Path("data/silver/stocks/daily_market")
    if not dm_root.exists():
        pytest.skip("Silver daily_market data not present")
    files = list(dm_root.rglob("*.parquet"))
    if not files:
        pytest.skip("No parquet files in daily_market")

    df_market = (
        pl.scan_parquet(files)
        .filter(
            (pl.col("instrument_id") == "KRX:005930")
            & (pl.col("session") >= datetime(2016, 1, 1, tzinfo=KST))
            & (pl.col("session") <= datetime(2016, 2, 5, tzinfo=KST))
        )
        .sort("session")
        .collect()
    )
    if df_market.height < 10:
        pytest.skip("Insufficient Samsung 2016-01 market bars")

    df_market_pit = df_market.with_columns(
        pl.col("session").dt.replace(hour=15, minute=30, second=0).alias("available_at")
    )
    sessions_list = tuple(sorted(df_market_pit["session"].to_list()))
    calendar = SessionCalendar(sessions_list)
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: df_market_pit}, root=tmp_path)

    start = sessions_list[0]
    end = sessions_list[-2]
    sessions = build_backtest_sessions(
        snapshot_repository=repo,
        calendar=calendar,
        start=start,
        end=end,
        decision_time_of=lambda s: s.replace(hour=15, minute=30, second=0),
    )
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    return sessions, calendar, instrument


def test_samsung_2016_smoke_roundtrip_settlement_and_accounting(
    samsung_2016_january_sessions: tuple[tuple[BacktestSession, ...], SessionCalendar, Instrument],
    tmp_path: Path,
) -> None:
    sessions, calendar, instrument = samsung_2016_january_sessions
    costs = default_base_schedule()
    ticks = TickSizeSchedule((
        TickSizeRule("all", datetime(2000, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1000.0),
    ))
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.1, ticks),
        ExecutionScenario.BASE,
        target_participation_cap=0.1,
        hard_participation_cap=0.2,
    )

    sessions_ordered = [s.session_open for s in sessions]

    class RoundtripStrategy:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            idx = sessions_ordered.index(context.decision_time.replace(hour=9, minute=0, second=0))
            if idx == 0:
                return (
                    TradeIntent(
                        intent_id="buy-samsung-smoke",
                        asset_kind=AssetKind.STOCK,
                        instrument_id=instrument.instrument_id,
                        target_value=50_000_000.0,
                        decision_time=context.decision_time,
                        execution_time=sessions_ordered[1],
                        strategy_id="champion-v1",
                        reason="smoke_entry",
                        idempotency_key="smoke_buy_key",
                        account_snapshot_id=context.portfolio.account_snapshot_id,
                    ),
                )
            if idx == 8:
                return (
                    TradeIntent(
                        intent_id="sell-samsung-smoke",
                        asset_kind=AssetKind.STOCK,
                        instrument_id=instrument.instrument_id,
                        target_value=0.0,
                        decision_time=context.decision_time,
                        execution_time=sessions_ordered[9],
                        strategy_id="champion-v1",
                        reason="smoke_exit",
                        idempotency_key="smoke_sell_key",
                        account_snapshot_id=context.portfolio.account_snapshot_id,
                    ),
                )
            return ()

    config = BacktestConfig(
        ledger_id="smoke-samsung-roundtrip",
        initial_cash=100_000_000.0,
        instruments={instrument.instrument_id: instrument},
        scenario=ExecutionScenario.BASE,
        cost_schedule=costs,
        calendar=calendar,
        fill_model=fill_model,
    )

    result, manifest = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=RoundtripStrategy(),
        artifact_root=tmp_path / "artifacts",
        dataset_hash="test_samsung_jan_2016",
        smoke_symbol="KRX:005930",
    )

    assert len(result.fills) == 2
    buy_fill = result.fills[0]
    sell_fill = result.fills[1]
    assert buy_fill.side.value == "BUY"
    assert sell_fill.side.value == "SELL"
    assert buy_fill.quantity == sell_fill.quantity
    assert buy_fill.quantity > 0

    assert buy_fill.commission > 0
    assert buy_fill.tax == 0.0
    assert buy_fill.slippage_cost > 0
    assert sell_fill.commission > 0
    assert sell_fill.tax > 0.0
    assert sell_fill.slippage_cost > 0

    assert verify_accounting_identity(result.daily_nav) is True
    final_nav = result.daily_nav[-1]
    assert final_nav.unsettled_cash == pytest.approx(0.0, abs=1e-5)
    assert final_nav.marked_value == pytest.approx(0.0, abs=1e-5)
    assert final_nav.nav == pytest.approx(final_nav.settled_cash, abs=1e-5)

    assert manifest["content_hash"] != ""
    assert manifest["fill_count"] == 2
    assert manifest["reject_count"] == 0
    assert manifest["accounting_reconciled"] is True
    assert "performance" in manifest
    perf = manifest["performance"]
    assert perf["initial_nav"] == 100_000_000.0
    assert perf["mdd"] >= 0.0


def test_samsung_smoke_reproducibility_identical_hash(
    samsung_2016_january_sessions: tuple[tuple[BacktestSession, ...], SessionCalendar, Instrument],
    tmp_path: Path,
) -> None:
    sessions, calendar, instrument = samsung_2016_january_sessions
    costs = default_base_schedule()
    ticks = TickSizeSchedule((
        TickSizeRule("all", datetime(2000, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1000.0),
    ))
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.1, ticks),
        ExecutionScenario.BASE,
        target_participation_cap=0.1,
        hard_participation_cap=0.2,
    )

    class StaticStrategy:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            return ()

    config = BacktestConfig(
        ledger_id="smoke-reproducibility",
        initial_cash=100_000_000.0,
        instruments={instrument.instrument_id: instrument},
        scenario=ExecutionScenario.BASE,
        cost_schedule=costs,
        calendar=calendar,
        fill_model=fill_model,
    )

    _, manifest_run1 = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=StaticStrategy(),
        artifact_root=tmp_path / "artifacts1",
        dataset_hash="reproducibility_test_v1",
        smoke_symbol="KRX:005930",
    )

    _, manifest_run2 = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=StaticStrategy(),
        artifact_root=tmp_path / "artifacts2",
        dataset_hash="reproducibility_test_v1",
        smoke_symbol="KRX:005930",
    )

    assert manifest_run1["content_hash"] == manifest_run2["content_hash"]
    assert manifest_run1["performance"] == manifest_run2["performance"]


def test_compute_backtest_performance_empty() -> None:
    from src.data.backtest_runner import compute_backtest_performance
    perf = compute_backtest_performance(())
    assert perf["initial_nav"] == 0.0
    assert perf["cagr"] == 0.0


def test_verify_accounting_identity_valid() -> None:
    from datetime import UTC, datetime
    from src.core.ledger import LedgerNav
    from src.data.backtest_runner import verify_accounting_identity

    nav = LedgerNav(
        mark_id="mark-1",
        as_of=datetime(2020, 1, 1, 9, 0, tzinfo=UTC),
        nav=100.0,
        settled_cash=50.0,
        unsettled_cash=20.0,
        marked_value=30.0,
    )
    assert verify_accounting_identity((nav,)) is True
