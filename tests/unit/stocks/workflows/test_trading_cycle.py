"""PLAN-02-KRX-PIT-CYCLE / PLAN-02-PROVISIONAL-NO-LIVE: pure trading cycle."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import polars as pl

from src.core.datasets import (
    DatasetCertification,
    validate_production_manifest,
)
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot, Position
from src.core.time import TemporalViolationError
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.trading_cycle import (
    CycleStatus,
    TradingCycleNotReadyError,
    TradingCycleRequest,
    run_trading_cycle,
)
from tests.fixtures.stocks.helpers import (
    publish_baseline_artifact,
    stock_instrument_df,
    stock_manifest,
)


def decision_time() -> datetime:
    return datetime(2024, 2, 20, 15, 31, tzinfo=UTC)


def execution_time() -> datetime:
    return datetime(2024, 2, 21, 0, 0, tzinfo=UTC)


def instruments_of(frame) -> dict[str, Instrument]:
    ids = sorted(frame["instrument_id"].unique().to_list())
    return {
        i: Instrument(i, AssetKind.STOCK, "KRX", i.split(":")[-1], "KRW", lot_size=1)
        for i in ids
    }


def empty_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        account_snapshot_id="acc-1",
        as_of=decision_time(),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )


def build_snapshot(tmp_path, n_sessions: int = 80) -> tuple[DatasetSnapshot, ModelArtifactRegistry]:
    df = stock_instrument_df(n_sessions=n_sessions, n_tickers=3, horizon=5)
    manifest = stock_manifest(columns=df.columns, horizon=5)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    publish_baseline_artifact(
        registry,
        artifact_id="a001",
        feature_schema_hash=manifest.schema_hash,
    )
    return DatasetSnapshot(manifest=manifest, frame=df), registry


def cycle_request(manifest, **overrides) -> TradingCycleRequest:
    values = {
        "strategy_id": "stock_alpha_v1",
        "artifact_id": "a001",
        "dataset_id": "dataset-a",
        "decision_time": decision_time(),
        "execution_time": execution_time(),
        "risk_policy": StockRiskPolicy(),
        "mode": "plan",
    }
    values.update(overrides)
    return TradingCycleRequest(**values)


class TestTradingCycleRequest:
    def test_rejects_decision_at_or_after_execution(self) -> None:
        with pytest.raises(ValueError, match="decision_time"):
            TradingCycleRequest(
                strategy_id="s",
                artifact_id="a",
                dataset_id="d",
                decision_time=decision_time(),
                execution_time=decision_time(),
                risk_policy=StockRiskPolicy(),
            )

    def test_rejects_unsupported_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            TradingCycleRequest(
                strategy_id="s",
                artifact_id="a",
                dataset_id="d",
                decision_time=decision_time(),
                execution_time=execution_time(),
                risk_policy=StockRiskPolicy(),
                mode="nope",
            )


class TestKrXPitCycle:
    def test_planning_cycle_is_deterministic_and_repeatable(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        request = cycle_request(snapshot.manifest)
        first = run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), request)
        second = run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), request)
        assert first.status is CycleStatus.PLANNED
        assert first.cycle_id == second.cycle_id
        assert first.dataset_hash == second.dataset_hash
        assert first.artifact_id == second.artifact_id
        assert first.account_snapshot_id == second.account_snapshot_id
        assert [a.instrument.instrument_id for a in first.allocations] == [
            a.instrument.instrument_id for a in second.allocations
        ]
        assert [i.target_value for i in first.intents] == [i.target_value for i in second.intents]
        assert first.cycle_id

    def test_result_carries_fingerprints(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        result = run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), cycle_request(snapshot.manifest))
        assert result.universe_hash
        assert result.feature_hash
        assert result.label_hash
        assert result.risk_policy_hash
        assert result.dataset_hash != snapshot.manifest.schema_hash

    def test_cycle_id_changes_when_snapshot_rows_change(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        instruments = instruments_of(snapshot.frame)
        request = cycle_request(snapshot.manifest)
        changed_frame = snapshot.frame.with_columns(
            pl.when(pl.col("instrument_id") == "KRX:000001")
            .then(pl.col("close") + 1.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        changed = DatasetSnapshot(manifest=snapshot.manifest, frame=changed_frame)
        first = run_trading_cycle(snapshot, registry, instruments, empty_portfolio(), request)
        second = run_trading_cycle(changed, registry, instruments, empty_portfolio(), request)
        assert first.dataset_hash != second.dataset_hash
        assert first.cycle_id != second.cycle_id

    def test_unpromoted_artifact_is_rejected(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        snapshot = replace(
            snapshot,
            manifest=replace(snapshot.manifest, certification=DatasetCertification.RESEARCH),
        )
        publish_baseline_artifact(
            registry,
            artifact_id="candidate",
            feature_schema_hash=snapshot.manifest.schema_hash,
            promoted=False,
        )
        with pytest.raises(TradingCycleNotReadyError, match="promoted=true"):
            run_trading_cycle(
                snapshot,
                registry,
                instruments_of(snapshot.frame),
                empty_portfolio(),
                cycle_request(snapshot.manifest, artifact_id="candidate", mode="paper"),
            )

    def test_cycle_id_key_changes_with_each_input_component(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        publish_baseline_artifact(
            registry,
            artifact_id="a002",
            feature_schema_hash=snapshot.manifest.schema_hash,
        )
        instruments = instruments_of(snapshot.frame)
        base = run_trading_cycle(snapshot, registry, instruments, empty_portfolio(), cycle_request(snapshot.manifest))
        keys = set()
        keys.add(base.cycle_id)
        keys.add(run_trading_cycle(snapshot, registry, instruments, empty_portfolio(), cycle_request(snapshot.manifest, strategy_id="other")).cycle_id)
        keys.add(run_trading_cycle(snapshot, registry, instruments, empty_portfolio(), cycle_request(snapshot.manifest, artifact_id="a002")).cycle_id)
        keys.add(run_trading_cycle(snapshot, registry, instruments, empty_portfolio(), cycle_request(snapshot.manifest, decision_time=decision_time().replace(hour=16))).cycle_id)
        other_account = replace(empty_portfolio(), account_snapshot_id="acc-2")
        keys.add(run_trading_cycle(snapshot, registry, instruments, other_account, cycle_request(snapshot.manifest)).cycle_id)
        assert len(keys) == 5

    def test_future_availability_raises_temporal_violation(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        # rows observe after their availability time (point-in-time inversion)
        bad = snapshot.frame.with_columns(
            (snapshot.frame["available_time"] + timedelta(hours=3)).alias("observation_time")
        )
        bad_snapshot = DatasetSnapshot(manifest=snapshot.manifest, frame=bad)
        portfolio = replace(empty_portfolio(), as_of=decision_time().replace(hour=20))
        request = cycle_request(snapshot.manifest, decision_time=decision_time().replace(hour=20))
        with pytest.raises(TemporalViolationError):
            run_trading_cycle(bad_snapshot, registry, instruments_of(snapshot.frame), portfolio, request)

    def test_missing_artifact_is_rejected(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        request = cycle_request(snapshot.manifest, artifact_id="missing-v1")
        with pytest.raises(FileNotFoundError):
            run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), request)

    def test_account_snapshot_newer_than_decision_rejected(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        portfolio = replace(empty_portfolio(), as_of=decision_time().replace(hour=18))
        with pytest.raises(ValueError, match="newer than decision_time"):
            run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), portfolio, cycle_request(snapshot.manifest))

    def test_held_position_exits_with_zero_target_intent(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        instruments = instruments_of(snapshot.frame)
        visible = snapshot.frame.filter(snapshot.frame["available_time"] <= decision_time())
        latest = visible["session"].max()
        cross = visible.filter(visible["session"] == latest)
        # hold the lowest-momentum instrument so the ranking excludes it
        held_id = cross.sort("feature_momentum_5d")["instrument_id"][0]
        instrument = instruments[held_id]
        price = cross.filter(cross["instrument_id"] == held_id)["close"][0]
        portfolio = PortfolioSnapshot(
            account_snapshot_id="acc-held",
            as_of=decision_time(),
            settled_cash=50_000_000.0,
            unsettled_cash=0.0,
            positions=(
                Position(instrument=instrument, quantity=100, average_cost=price),
            ),
        )
        policy = StockRiskPolicy(top_k=1, turnover_budget=1.0)
        result = run_trading_cycle(
            snapshot, registry, instruments, portfolio,
            cycle_request(snapshot.manifest, risk_policy=policy),
        )
        assert result.status is CycleStatus.PLANNED
        assert any(i.instrument_id == held_id and i.target_value == 0.0 for i in result.intents)

    def test_infeasible_current_position_returns_de_risk_status(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        instruments = instruments_of(snapshot.frame)
        visible = snapshot.frame.filter(snapshot.frame["available_time"] <= decision_time())
        latest = visible["session"].max()
        cross = visible.filter(visible["session"] == latest)
        held_id = cross["instrument_id"][0]
        price = float(cross.filter(cross["instrument_id"] == held_id)["close"][0])
        quantity = 100
        position_value = quantity * price
        portfolio = PortfolioSnapshot(
            account_snapshot_id="acc-over-cap",
            as_of=decision_time(),
            settled_cash=position_value * 3.0,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=instruments[held_id],
                    quantity=quantity,
                    average_cost=price,
                ),
            ),
        )
        result = run_trading_cycle(
            snapshot,
            registry,
            instruments,
            portfolio,
            cycle_request(snapshot.manifest, risk_policy=StockRiskPolicy(top_k=1)),
        )
        assert result.status is CycleStatus.DE_RISK
        assert any("constraints=de-risk" in reason for reason in result.reasons)


class TestProvisionalNoLive:
    def test_provisional_plan_produces_diagnostics_only(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        assert snapshot.manifest.certification is DatasetCertification.PROVISIONAL
        result = run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), cycle_request(snapshot.manifest))
        assert result.status is CycleStatus.PLANNED
        assert any("certification=provisional" in r for r in result.reasons)

    def test_provisional_paper_mode_is_rejected(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        request = cycle_request(snapshot.manifest, mode="paper")
        with pytest.raises(TradingCycleNotReadyError, match="RESEARCH or PRODUCTION"):
            run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), request)

    def test_provisional_live_mode_is_rejected(self, tmp_path) -> None:
        snapshot, registry = build_snapshot(tmp_path)
        request = cycle_request(snapshot.manifest, mode="live")
        with pytest.raises(TradingCycleNotReadyError, match="PRODUCTION"):
            run_trading_cycle(snapshot, registry, instruments_of(snapshot.frame), empty_portfolio(), request)

    def test_production_missing_hashes_fails_closed(self) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=2)
        manifest = stock_manifest(columns=df.columns, horizon=5)
        production = replace(manifest, certification=DatasetCertification.PRODUCTION)
        with pytest.raises(ValueError, match="calendar_hash"):
            validate_production_manifest(production)

    def test_live_mode_accepts_complete_production_manifest(self, tmp_path) -> None:
        df = stock_instrument_df(n_sessions=60, n_tickers=2)
        manifest = stock_manifest(columns=df.columns, horizon=5)
        production = replace(
            manifest,
            certification=DatasetCertification.PRODUCTION,
            calendar_hash="cal",
            corporate_action_hash="ca",
            cost_source_hash="cost",
        )
        validate_production_manifest(production)
        assert production.certification is DatasetCertification.PRODUCTION
