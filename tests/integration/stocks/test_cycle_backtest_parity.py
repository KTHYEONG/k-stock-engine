"""PLAN-03-BACKTEST-LIVE-PARITY: replay and paper cycle produce identical targets."""
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
    StockBacktester,
)
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.trading_cycle import TradingCycleRequest, run_trading_cycle
from tests.fixtures.stocks.helpers import (
    publish_baseline_artifact,
    stock_instrument_df,
    stock_manifest,
)


def build_inputs(tmp_path):
    df = stock_instrument_df(n_sessions=80, n_tickers=3, horizon=5)
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
    return snapshot, registry, instruments


def decision() -> datetime:
    return datetime(2024, 2, 20, 15, 31, tzinfo=UTC)


def test_identical_snapshot_yields_identical_targets_in_replay_and_cycle(tmp_path) -> None:
    snapshot, registry, instruments = build_inputs(tmp_path)
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    decision_time = decision()
    execution_time = datetime(2024, 2, 21, 0, 0, tzinfo=UTC)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-parity",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )

    cycle = run_trading_cycle(
        snapshot,
        registry,
        instruments,
        portfolio,
        TradingCycleRequest(
            strategy_id="parity",
            artifact_id="a001",
            dataset_id="d",
            decision_time=decision_time,
            execution_time=execution_time,
            risk_policy=policy,
            mode="plan",
        ),
    )

    sessions = sorted(snapshot.frame["session"].unique().to_list())
    visible = snapshot.frame.filter(snapshot.frame["available_time"] <= decision_time)
    decision_session = visible["session"].max()
    decision_index = sessions.index(decision_session)
    backtest_request = BacktestRequest(
        strategy_id="parity",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(decision_index,),
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=policy,
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
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    result = backtester.run(snapshot.frame, artifacts, portfolio, backtest_request)
    replay_cycles = backtester.cycles_at((decision_index,))
    assert decision_index in replay_cycles
    replay_allocs = replay_cycles[decision_index]
    cycle_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6)) for a in cycle.allocations
    }
    replay_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6)) for a in replay_allocs.allocations
    }
    assert cycle_targets == replay_targets
    assert cycle.cycle_id == replay_allocs.cycle_id


def test_replay_ledger_reconciles_after_every_event(tmp_path) -> None:
    snapshot, registry, instruments = build_inputs(tmp_path)
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    sessions = sorted(snapshot.frame["session"].unique().to_list())
    backtest_request = BacktestRequest(
        strategy_id="parity",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=(10, 20, 30),
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        risk_policy=policy,
    )
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-parity",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    result = backtester.run(
        snapshot.frame,
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
        backtest_request,
    )
    assert result.ledger
    for row in result.ledger:
        assert abs(row.equity - (row.settled_cash + row.unsettled_cash + row.positions_value - row.accrued_costs)) <= 1e-8
    del sessions


class _CalibratedRankingModel:
    """Test-only model emitting deterministic net-alpha prediction columns.

    ``predict`` ranks ``feature_momentum_5d`` per session and attaches positive
    cost-adjusted expected net alpha so the economic allocation gate accepts
    entries. The production planner and the replay planner both consume the
    exact same scored columns through ``construct_target_allocations``.
    """

    no_trade = False

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        scored = frame.with_columns(
            pl.col("feature_momentum_5d")
            .rank("dense")
            .over("session")
            .cast(pl.Float64)
            .alias("pred_score")
        )
        return scored.with_columns(
            pl.lit(0.02, dtype=pl.Float64).alias("expected_active_alpha"),
            pl.lit(0.015, dtype=pl.Float64).alias("expected_net_alpha"),
            pl.lit(0.005, dtype=pl.Float64).alias("alpha_lower_bound"),
            pl.lit(0.002, dtype=pl.Float64).alias("exit_cost_rate"),
        )

    def manifest(self) -> ModelManifest:
        raise NotImplementedError


def test_calibrated_replay_and_cycle_produce_identical_targets(tmp_path) -> None:
    """Economic columns flow identically through replay and the planner."""
    df = stock_instrument_df(n_sessions=80, n_tickers=3, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    model_manifest = ModelManifest(
        artifact_id="a002",
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        universe_policy_hash=manifest.universe_policy_hash,
        label_definition=manifest.label_definition,
        label_horizon_sessions=manifest.label_horizon_sessions,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-03-31T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
    )
    registry.publish(_CalibratedRankingModel(), model_manifest)
    registry.write_metrics("a002", {"promoted": True})
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    instruments = {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in sorted(df["instrument_id"].unique().to_list())
    }
    policy = StockRiskPolicy(top_k=5, turnover_budget=1.0)
    decision_time = decision()
    execution_time = datetime(2024, 2, 21, 0, 0, tzinfo=UTC)
    portfolio = PortfolioSnapshot(
        account_snapshot_id="acc-calib-parity",
        as_of=decision_time,
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )

    cycle = run_trading_cycle(
        snapshot,
        registry,
        instruments,
        portfolio,
        TradingCycleRequest(
            strategy_id="parity-calib",
            artifact_id="a002",
            dataset_id="d",
            decision_time=decision_time,
            execution_time=execution_time,
            risk_policy=policy,
            mode="plan",
        ),
    )

    sessions = sorted(snapshot.frame["session"].unique().to_list())
    visible = snapshot.frame.filter(snapshot.frame["available_time"] <= decision_time)
    decision_session = visible["session"].max()
    decision_index = sessions.index(decision_session)
    backtester = StockBacktester(
        registry=registry,
        instruments=instruments,
        manifest=snapshot.manifest,
        cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
    )
    backtester.run(
        snapshot.frame,
        ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=datetime(2024, 1, 1, tzinfo=UTC),
                    eligible_to=datetime(2024, 3, 31, tzinfo=UTC),
                    artifact_id="a002",
                ),
            )
        ),
        portfolio,
        BacktestRequest(
            strategy_id="parity-calib",
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            end_time=datetime(2024, 3, 31, tzinfo=UTC),
            decision_session_indices=(decision_index,),
            cost_schedule=default_base_schedule(),
            stress_cost_schedule=default_stress_schedule(),
            risk_policy=policy,
        ),
    )
    replay_allocs = backtester.cycles_at((decision_index,))
    assert decision_index in replay_allocs
    cycle_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6)) for a in cycle.allocations
    }
    replay_targets = {
        (a.instrument.instrument_id, round(a.target_value, 6))
        for a in replay_allocs[decision_index].allocations
    }
    assert cycle_targets == replay_targets
    assert cycle.allocations
