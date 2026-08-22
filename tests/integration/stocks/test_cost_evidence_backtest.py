"""Integration: dynamic cost evidence drives engine and simulator fills.

Covers the fail-closed missing-liquidity gate, per-fill cost tracing, buy/sell
tax separation, manifest lineage, run reproducibility, and engine/simulator
parity through the shared ``resolve_fill_cost`` helper.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import tempfile

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


REPLAY_ECONOMIC_PARITY_01 = "REPLAY-ECONOMIC-PARITY-01"


def test_replay_economic_parity_01_streaming_matches_batch_reference() -> None:
    """REPLAY-ECONOMIC-PARITY-01: segment-major streaming parity.

    Base/stress returns, fills, costs evidence, settlement-driven equity, and
    candidate ordering are exact (or within 1e-12) between the streaming path
    and the prepared-batch reference across multiple candidates.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    import polars as pl
    import pytest

    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    from src.core.portfolio import PortfolioSnapshot
    from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
    from src.stocks.ml.contracts import NetAlphaTrainingRequest
    from src.stocks.ml.execution_replay import (
        ExecutionEquivalentReplayRequest,
        ExecutionReplayContext,
        instruments_from_frame,
        prepare_execution_replay_batch,
        replay_execution_equivalent_batch,
        stream_execution_replay_batch,
    )
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from src.stocks.trading.portfolio_constructor import StockRiskPolicy
    from tests.fixtures.stocks.helpers import stock_liquidity_model

    market_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    segments: dict[int, list[datetime]] = {}
    for seg in range(2):
        for idx in range(8):
            session = datetime(2024, 1, 1 + seg * 12 + idx, tzinfo=UTC)
            segments.setdefault(seg, []).append(session)
            for t in range(3):
                price = 100.0 + t + idx * 0.25
                market_rows.append(
                    {
                        "instrument_id": f"KRX:{t + 1:05d}",
                        "session": session,
                        "observation_time": session.replace(hour=15, minute=30),
                        "available_time": session.replace(hour=15, minute=31),
                        "open": price,
                        "close": price * 1.01,
                        "volume": 1e6,
                        "trading_value": price * 1e6,
                        "sector": f"S{t % 2}",
                    }
                )
                score = 0.012 - t * 0.002 + idx * 1e-5
                score_rows.append(
                    {
                        "instrument_id": f"KRX:{t + 1:05d}",
                        "session": session,
                        "oof_segment_id": seg,
                        "predicted_net_alpha": score,
                        "expected_active_alpha": score,
                        "alpha_lower_bound": score * 0.5,
                        "expected_net_alpha": score * 0.8,
                        "net_alpha_lower_bound": score * 0.3,
                        "exit_cost_rate": 0.0012,
                    }
                )
    market = pl.DataFrame(market_rows)
    scores = pl.DataFrame(score_rows)
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK, schema_version="v1", schema_hash="h",
        provider_version="p", universe_policy_version="u",
        universe_policy_hash="u", feature_set="stock_net_alpha_v1",
        feature_set_hash="f", label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 2, 6, tzinfo=UTC),
        generated_time=datetime(2024, 2, 6, tzinfo=UTC),
        row_count=market.height,
    )
    request = NetAlphaTrainingRequest(artifact_id="parity", candidate_horizon_sessions=(10,))
    context = ExecutionReplayContext(
        registry=ModelArtifactRegistry(Path(tempfile.mkdtemp(prefix="parity-"))),
        manifest=manifest,
        instruments=instruments_from_frame(market),
        artifact_id="parity",
        strategy_id="parity",
        initial_portfolio=PortfolioSnapshot(
            account_snapshot_id="oof",
            as_of=min(segments[0]),
            settled_cash=request.portfolio.initial_cash,
            unsettled_cash=0.0,
            positions=(),
        ),
        risk_policy=StockRiskPolicy(
            top_k=3, gross_cap=0.9, single_name_cap=0.3, sector_cap=0.5,
            participation_limit=0.01, no_trade_band_bps=0.0,
        ),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=stock_liquidity_model(),
        stress_liquidity_model=stock_liquidity_model(stress_multiplier=1.5),
        execution_policy=SCHEDULED_OPEN_V1,
        seed=42,
    )

    def _req(seed_offset: int) -> ExecutionEquivalentReplayRequest:
        ctx = context if seed_offset == 0 else ExecutionReplayContext(
            registry=context.registry,
            manifest=context.manifest,
            instruments=context.instruments,
            artifact_id=context.artifact_id,
            strategy_id=context.strategy_id,
            initial_portfolio=context.initial_portfolio,
            risk_policy=context.risk_policy,
            base_cost_schedule=context.base_cost_schedule,
            stress_cost_schedule=context.stress_cost_schedule,
            liquidity_model=context.liquidity_model,
            stress_liquidity_model=context.stress_liquidity_model,
            execution_policy=context.execution_policy,
            seed=context.seed + seed_offset,
        )
        return ExecutionEquivalentReplayRequest(
            context=ctx,
            market_frame=market,
            score_frame=scores,
            segment_column="oof_segment_id",
            decision_sessions_by_segment={s: tuple(v) for s, v in segments.items()},
            horizon_sessions=10,
        )

    requests = [_req(0), _req(1), _req(2)]
    batch = prepare_execution_replay_batch(requests[0])
    reference = replay_execution_equivalent_batch(requests, prepared_batch=batch)
    stats: dict[str, int] = {}
    streamed = stream_execution_replay_batch(tuple(requests), stats=stats)

    assert [ev.segment_ids for ev in streamed] == [
        ev.segment_ids for ev in reference
    ]
    for streamed_ev, reference_ev in zip(streamed, reference, strict=True):
        assert streamed_ev.base_log_growth == pytest.approx(
            reference_ev.base_log_growth, abs=1e-12
        )
        assert streamed_ev.stress_log_growth == pytest.approx(
            reference_ev.stress_log_growth, abs=1e-12
        )
        assert streamed_ev.planned_cycles == reference_ev.planned_cycles
        assert streamed_ev.filled_orders == reference_ev.filled_orders
        assert streamed_ev.turnover == pytest.approx(reference_ev.turnover, abs=1e-12)
        assert streamed_ev.cash_session_fraction == pytest.approx(
            reference_ev.cash_session_fraction, abs=1e-12
        )
        assert streamed_ev.invested_interval_count == (
            reference_ev.invested_interval_count
        )
        assert streamed_ev.filled_cycle_count == reference_ev.filled_cycle_count
        assert streamed_ev.unfilled_order_reason_counts == (
            reference_ev.unfilled_order_reason_counts
        )
        assert streamed_ev.base_interval_exposure == pytest.approx(
            reference_ev.base_interval_exposure, abs=1e-12
        )
