"""Integration: dynamic cost evidence drives engine and simulator fills.

Covers the fail-closed missing-liquidity gate, per-fill cost tracing, buy/sell
tax separation, manifest lineage, run reproducibility, and engine/simulator
parity through the shared ``resolve_fill_cost`` helper.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot
from src.core.datasets import DatasetManifest
from src.execution.domain.intents import TradeIntent
from src.stocks.backtesting.engine import (
    ArtifactSchedule,
    ArtifactSlot,
    BacktestRequest,
    StockBacktester,
)
from src.stocks.data.costs import resolve_fill_cost
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.allocation_policy import AllocationPolicy
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.trading.simulator import StockSimulator
from src.stocks.workflows.trading_cycle import CycleStatus, TradingCycleResult
from tests.fixtures.stocks.helpers import (
    cost_evidence_fixture,
    stock_manifest,
)

INSTRUMENT_ID = "KRX:000050"
INSTRUMENT = Instrument(INSTRUMENT_ID, AssetKind.STOCK, "KRX", "000050", "KRW", lot_size=1)
ADTV_TARGET = 100_000_000.0


def dynamic_panel(n_sessions: int = 12) -> pl.DataFrame:
    """Deterministic panel with constant trading value and point-in-time vol."""
    rows = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for s in range(n_sessions):
        obs = start + timedelta(days=s)
        close = 100.0 + float((s % 5) * 2.0)
        rows.append(
            {
                "session_index": s,
                "session": obs,
                "instrument_id": INSTRUMENT_ID,
                "observation_time": obs.replace(hour=15, minute=30, tzinfo=UTC),
                "available_time": obs.replace(hour=15, minute=31, tzinfo=UTC),
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 1.5,
                "close": close,
                "volume": ADTV_TARGET / close,
                "trading_value": ADTV_TARGET,
                "adtv": ADTV_TARGET,
                "feature__volatility_20d": 0.02,
                "pred_score": 1.0,
                "sector": "S1",
            }
        )
    return pl.DataFrame(rows)


def stub_planner(plan_steps: list[float]):
    """Return a planner that emits fixed target values per decision session.

    Base and stress ledger runs both consume the same step sequence, so the
    steps are cycled deterministically across calls.
    """
    state = {"index": 0}

    def plan(snapshot, registry, instruments, portfolio, request):
        target_value = plan_steps[state["index"] % len(plan_steps)]
        state["index"] += 1
        allocation = Allocation(
            instrument=INSTRUMENT, target_value=target_value, reason="stub"
        )
        intent = TradeIntent(
            intent_id="stub",
            asset_kind=AssetKind.STOCK,
            instrument_id=INSTRUMENT_ID,
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
            selected_instruments=(INSTRUMENT_ID,),
            reasons=("stub",),
        )

    return plan


def build_backtester(
    tmp_path,
    panel: pl.DataFrame,
    plan_steps: list[float],
    *,
    evidence=None,
    manifest: DatasetManifest | None = None,
    decision_session_indices: tuple[int, ...] = (0, 1),
) -> tuple[StockBacktester, BacktestRequest, PortfolioSnapshot, ArtifactSchedule]:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = manifest or stock_manifest(columns=panel.columns, horizon=5)
    backtester = StockBacktester(
        planner=stub_planner(plan_steps),
        registry=registry,
        instruments={INSTRUMENT_ID: INSTRUMENT},
        manifest=manifest,
        cost_schedule=default_base_schedule(),
        cost_evidence=evidence,
    )
    request = BacktestRequest(
        strategy_id="cost_evidence_bt",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 2, 1, tzinfo=UTC),
        decision_session_indices=decision_session_indices,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=StockRiskPolicy(top_k=1),
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="cost-evidence",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                eligible_to=datetime(2024, 2, 1, tzinfo=UTC),
                artifact_id="a001",
            ),
        )
    )
    return backtester, request, portfolio, artifacts


def test_engine_dynamic_fill_records_cost_breakdown(tmp_path) -> None:
    evidence = cost_evidence_fixture()
    panel = dynamic_panel()
    backtester, request, portfolio, artifacts = build_backtester(
        tmp_path, panel, [1e12, 0.0], evidence=evidence
    )
    result = backtester.run(panel, artifacts, portfolio, request)

    buy = [t for t in result.trades if t.side == "BUY" and t.quantity > 0]
    assert buy, "expected a filled buy"
    fill = buy[0]
    assert fill.cost_breakdown is not None
    recomputed, _ = resolve_fill_cost(
        evidence,
        side="BUY",
        market="KOSPI",
        price=float(fill.price),
        notional=float(fill.gross),
        adtv_20d=ADTV_TARGET,
        daily_volatility=0.02,
        effective_time=fill.session,
    )
    assert fill.cost_breakdown == recomputed.to_dict(artifact_hash=evidence.content_hash)
    assert fill.cost_breakdown["sell_tax_rate"] > 0.0
    assert fill.cost_breakdown["tick_rule_id"] == evidence.tick_schedule.rule_for(
        float(fill.price), fill.session
    ).rule_id
    assert fill.cost_breakdown["model_id"] == "sqrt_impact_v1"

    sell = [t for t in result.trades if t.side == "SELL" and t.quantity > 0]
    assert sell, "expected a filled sell"
    sell_fill = sell[0]
    assert sell_fill.cost_breakdown is not None
    sell_recomputed, _ = resolve_fill_cost(
        evidence,
        side="SELL",
        market="KOSPI",
        price=float(sell_fill.price),
        notional=float(sell_fill.gross),
        adtv_20d=ADTV_TARGET,
        daily_volatility=0.02,
        effective_time=sell_fill.session,
    )
    assert sell_fill.cost_breakdown == sell_recomputed.to_dict(
        artifact_hash=evidence.content_hash
    )
    assert sell_fill.cost > 0.0
    assert sell_fill.cost > buy[0].cost


def test_engine_missing_liquidity_input_is_unfilled(tmp_path) -> None:
    evidence = cost_evidence_fixture()
    panel = dynamic_panel().drop("feature__volatility_20d")
    backtester, request, portfolio, artifacts = build_backtester(
        tmp_path, panel, [1e12, 0.0], evidence=evidence
    )
    result = backtester.run(panel, artifacts, portfolio, request)
    reasons = [t.reason for t in result.trades]
    assert "missing-liquidity-input" in reasons
    assert all(t.quantity == 0 for t in result.trades)


def test_engine_manifest_records_cost_artifact(tmp_path) -> None:
    evidence = cost_evidence_fixture()
    panel = dynamic_panel()
    backtester, request, portfolio, artifacts = build_backtester(
        tmp_path, panel, [1e12, 0.0], evidence=evidence
    )
    result = backtester.run(panel, artifacts, portfolio, request)
    assert result.data_quality["cost_artifact_hash"] == evidence.content_hash
    assert result.data_quality["cost_model_id"] == "sqrt_impact_v1"
    assert result.data_quality["cost_params_hash"] == evidence.base_liquidity_model.params_hash


def test_engine_run_is_reproducible(tmp_path) -> None:
    evidence = cost_evidence_fixture()
    panel = dynamic_panel()
    backtester, request, portfolio, artifacts = build_backtester(
        tmp_path, panel, [1e12, 0.0], evidence=evidence
    )
    first = backtester.run(panel, artifacts, portfolio, request)
    second = backtester.run(panel, artifacts, portfolio, request)
    assert first.trades == second.trades
    assert first.ledger == second.ledger


def test_simulator_matches_shared_cost_helper_and_engine(tmp_path) -> None:
    evidence = cost_evidence_fixture()
    panel = dynamic_panel()

    simulator = StockSimulator(
        cost_schedule=default_base_schedule(), cost_evidence=evidence
    )
    sim_result = simulator.simulate(
        panel, AllocationPolicy(top_k=1, max_single_weight=0.2), AssetKind.STOCK
    )
    sim_buy = [
        t for t in sim_result.trades if t.get("side") == "buy" and t.get("quantity", 0) > 0
    ]
    assert sim_buy, "expected a simulator buy fill"
    sim_fill = sim_buy[0]
    assert "cost_breakdown" in sim_fill
    recomputed, _ = resolve_fill_cost(
        evidence,
        side="BUY",
        market="KOSPI",
        price=float(sim_fill["price"]),
        notional=float(sim_fill["gross"]),
        adtv_20d=ADTV_TARGET,
        daily_volatility=0.02,
        effective_time=sim_fill["session"],
    )
    assert sim_fill["cost_breakdown"] == recomputed.to_dict(
        artifact_hash=evidence.content_hash
    )

    backtester, request, portfolio, artifacts = build_backtester(
        tmp_path, panel, [1e12], evidence=evidence, decision_session_indices=(1,)
    )
    engine_result = backtester.run(panel, artifacts, portfolio, request)
    engine_buy = [t for t in engine_result.trades if t.side == "BUY" and t.quantity > 0]
    assert engine_buy
    # The engine embeds spread/impact in the adverse tick-rounded fill price, so
    # its recorded breakdown is traced at the fill price rather than the open.
    engine_recomputed, _ = resolve_fill_cost(
        evidence,
        side="BUY",
        market="KOSPI",
        price=float(engine_buy[0].price),
        notional=float(engine_buy[0].gross),
        adtv_20d=ADTV_TARGET,
        daily_volatility=0.02,
        effective_time=engine_buy[0].session,
    )
    assert engine_buy[0].cost_breakdown == engine_recomputed.to_dict(
        artifact_hash=evidence.content_hash
    )
    session_open = panel.filter(pl.col("session") == engine_buy[0].session)[
        "open"
    ][0]
    assert float(engine_buy[0].price) > float(session_open)


def test_simulator_missing_volatility_is_unfilled(tmp_path) -> None:
    evidence = cost_evidence_fixture()
    panel = dynamic_panel().drop("feature__volatility_20d")
    simulator = StockSimulator(
        cost_schedule=default_base_schedule(), cost_evidence=evidence
    )
    result = simulator.simulate(
        panel, AllocationPolicy(top_k=1, max_single_weight=0.2), AssetKind.STOCK
    )
    reasons = [t.get("reason") for t in result.trades]
    assert "missing-liquidity-input" in reasons
    assert all(t.get("quantity", 0) == 0 for t in result.trades)
