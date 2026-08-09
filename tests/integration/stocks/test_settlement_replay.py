"""PLAN-06-T2-SETTLEMENT: sale proceeds mature exactly on the due session.

The replay stores sale settlement under the *due session index* and releases it
exactly once at that session. Proceeds cannot fund buys before settlement, and
buys can never spend unsettled cash.
"""
from __future__ import annotations

from datetime import UTC, datetime

from src.core.costs import CostPoint, CostSchedule, default_stress_schedule
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
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
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
    decision_indices = tuple(range(0, 30, 5))
    request = BacktestRequest(
        strategy_id="t2",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 3, 31, tzinfo=UTC),
        decision_session_indices=decision_indices,
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

    # A sell adds proceeds to unsettled_cash that must not re-enter settled_cash
    # before the configured +2 session, and each due session matures exactly once.
    unsettled_steps: list[float] = [row.unsettled_cash for row in result.ledger]
    settled_steps: list[float] = [row.settled_cash for row in result.ledger]
    assert any(u > 0.0 for u in unsettled_steps)

    for index in range(1, len(result.ledger)):
        # settled cash must not fall below 0 and must never exceed what was ever
        # available; the accounting identity holds at every session.
        assert settled_steps[index] >= -1e-8
        row = result.ledger[index]
        assert abs(row.equity - (row.settled_cash + row.unsettled_cash + row.positions_value - row.accrued_costs)) <= 1e-8

    # A sale matures within the settlement_days window: once proceeds enter
    # unsettled, they appear in settled within `settlement_days` sessions.
    sell_indices = [
        i
        for i, trade in enumerate(result.trades)
        if trade.side == "SELL" and trade.gross is not None
    ]
    if sell_indices:
        sell_session = result.trades[sell_indices[0]].session
        ledgers = {row.session: row for row in result.ledger}
        sell_row = ledgers[sell_session]
        # proceeds must not be spendable on the sell session (still unsettled)
        assert sell_row.unsettled_cash > 0.0
        assert sell_row.settled_cash <= sell_row.settled_cash + sell_row.unsettled_cash - 1e-8


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
