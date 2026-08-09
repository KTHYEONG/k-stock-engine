"""Multi-session economic ledger simulator tests."""
from __future__ import annotations

import polars as pl
import pytest

from src.core.costs import default_base_schedule
from src.core.instruments import AssetKind
from src.stocks.research.metrics import max_drawdown
from src.stocks.trading.allocation_policy import AllocationPolicy
from src.stocks.trading.simulator import StockSimulator
from tests.fixtures.stocks.helpers import stock_instrument_df


def scored_panel(n_sessions: int = 30, n_tickers: int = 5) -> pl.DataFrame:
    return stock_instrument_df(n_sessions=n_sessions, n_tickers=n_tickers).with_columns(
        pl.lit(1.0).alias("pred_score")
    )


def test_every_session_reconciles_and_fills_are_integer() -> None:
    simulator = StockSimulator(cost_schedule=default_base_schedule())
    policy = AllocationPolicy(top_k=5, max_exposure=1.0)
    result = simulator.simulate(scored_panel(), policy, AssetKind.STOCK)
    assert result.ledger
    for row in result.ledger:
        assert abs(
            float(row["equity"])
            - (
                float(row["settled_cash"])
                + float(row["unsettled_cash"])
                + float(row["positions_value"])
                - float(row["accrued_costs"])
            )
        ) < 1e-8
    for fill in result.trades:
        if fill.get("quantity") is not None:
            assert int(fill["quantity"]) == fill["quantity"]


def test_missing_open_produces_unfilled_reason() -> None:
    panel = scored_panel(n_sessions=20).with_columns(
        pl.when(pl.col("session_index") == 5)
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    simulator = StockSimulator(cost_schedule=default_base_schedule())
    policy = AllocationPolicy(top_k=5, max_exposure=1.0)
    result = simulator.simulate(panel, policy, AssetKind.STOCK)
    assert any(t.get("reason") for t in result.trades)


def test_metrics_derive_only_from_ledger() -> None:
    simulator = StockSimulator(cost_schedule=default_base_schedule())
    policy = AllocationPolicy(top_k=5)
    result = simulator.simulate(scored_panel(), policy, AssetKind.STOCK)
    assert result.metrics["max_drawdown"] == pytest.approx(max_drawdown(result.equity_curve))
    assert result.metrics["turnover"] >= 0.0


def test_rejects_missing_required_columns() -> None:
    simulator = StockSimulator(cost_schedule=default_base_schedule())
    with pytest.raises(ValueError, match="panel must carry"):
        simulator.simulate(
            scored_panel().drop("volume"), AllocationPolicy(top_k=5), AssetKind.STOCK
        )
