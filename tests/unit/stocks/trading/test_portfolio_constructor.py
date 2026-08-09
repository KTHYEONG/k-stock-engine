"""PLAN-03-CONSTRAINED-SIZING: randomized feasible inputs preserve every constraint."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot, Position
from src.stocks.trading.portfolio_constructor import (
    PortfolioConstraintError,
    StockRiskPolicy,
    construct_target_allocations,
)


def instruments_for(n_tickers: int) -> dict[str, Instrument]:
    return {
        f"KRX:{t:06d}": Instrument(
            f"KRX:{t:06d}", AssetKind.STOCK, "KRX", f"{t:06d}", "KRW", lot_size=1
        )
        for t in range(1, n_tickers + 1)
    }


def scored_panel(
    n_sessions: int = 80,
    n_tickers: int = 10,
    seed: int = 7,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(1, n_tickers + 1):
        drift = rng.normal(0.0002, 0.0005)
        price = 50_000.0
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            ret = drift + rng.normal(0.0, 0.02)
            price = max(1_000.0, price * (1.0 + ret))
            rows.append(
                {
                    "session": obs,
                    "instrument_id": f"KRX:{t:06d}",
                    "pred_score": float(t) * 0.1 + rng.normal(0.0, 0.05),
                    "sector": f"S{t % 3}",
                    "adtv": price * (1_000_000.0 + float(t) * 50_000.0),
                    "close": price,
                    "ret": float(ret),
                }
            )
    return pl.DataFrame(rows)


def empty_portfolio(as_of: datetime | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        account_snapshot_id="acc-empty",
        as_of=as_of or datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=100_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )


def equity_of(panel: pl.DataFrame, portfolio: PortfolioSnapshot) -> float:
    latest = panel.select(pl.col("session").max()).to_series()[0]
    cross = panel.filter(pl.col("session") == latest)
    prices = {
        str(r["instrument_id"]): float(r["close"])
        for r in cross.select(["instrument_id", "close"]).to_dicts()
        if r["close"] is not None
    }
    return portfolio.equity(prices)


class TestStockRiskPolicy:
    def test_seed_profile_is_internally_consistent(self) -> None:
        policy = StockRiskPolicy()
        assert 0.0 < policy.single_name_cap <= policy.sector_cap <= policy.gross_cap <= 1.0
        assert 0.0 <= policy.participation_limit <= 1.0

    def test_rejects_invalid_cap_ordering(self) -> None:
        with pytest.raises(ValueError, match="caps"):
            StockRiskPolicy(single_name_cap=0.2, sector_cap=0.1)

    def test_rejects_negative_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookbacks"):
            StockRiskPolicy(volatility_lookback_sessions=0)


class TestConstrainedSizing:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_random_feasible_inputs_preserve_all_constraints(self, seed: int) -> None:
        policy = StockRiskPolicy(top_k=20, participation_limit=0.01)
        panel = scored_panel(seed=seed)
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        equity = equity_of(panel, portfolio)

        assert allocations
        assert all(a.target_value >= 0.0 for a in allocations)
        assert sum(a.target_value for a in allocations) <= equity * policy.gross_cap + 1e-8
        for a in allocations:
            assert a.target_value <= equity * policy.single_name_cap + 1e-8

        latest = panel.select(pl.col("session").max()).to_series()[0]
        cross = panel.filter(pl.col("session") == latest)
        sector_of = {str(r["instrument_id"]): r["sector"] for r in cross.to_dicts()}
        sector_total: dict[object, float] = {}
        for a in allocations:
            sector = sector_of[a.instrument.instrument_id]
            sector_total[sector] = sector_total.get(sector, 0.0) + a.target_value
        for total in sector_total.values():
            assert total <= equity * policy.sector_cap + 1e-8

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_capacity_cap_holds_at_low_equity(self, seed: int) -> None:
        policy = StockRiskPolicy(top_k=20, participation_limit=0.01)
        panel = scored_panel(seed=seed)
        instruments = instruments_for(10)
        portfolio = PortfolioSnapshot(
            account_snapshot_id="small",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=1_000_000.0,
            unsettled_cash=0.0,
            positions=(),
        )
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        equity = equity_of(panel, portfolio)
        latest = panel.select(pl.col("session").max()).to_series()[0]
        cross = panel.filter(pl.col("session") == latest)
        adtv_of = {str(r["instrument_id"]): float(r["adtv"]) for r in cross.to_dicts()}
        for a in allocations:
            assert (
                a.target_value
                <= policy.participation_limit * adtv_of[a.instrument.instrument_id] + 1e-8
            )

    def test_empty_panel_returns_empty_tuple(self) -> None:
        policy = StockRiskPolicy()
        panel = scored_panel(n_sessions=5)
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        result = construct_target_allocations(
            panel.filter(pl.col("pred_score").is_null()), instruments, portfolio, policy
        )
        assert result == ()

    def test_deterministic_tie_handling(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel_a = scored_panel(seed=21)
        panel_b = scored_panel(seed=21)
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        alloc_a = construct_target_allocations(panel_a, instruments, portfolio, policy)
        alloc_b = construct_target_allocations(panel_b, instruments, portfolio, policy)
        assert [(a.instrument.instrument_id, a.target_value) for a in alloc_a] == [
            (a.instrument.instrument_id, a.target_value) for a in alloc_b
        ]

    def test_returns_sorted_by_instrument_id(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = scored_panel(seed=31)
        instruments = instruments_for(10)
        allocations = construct_target_allocations(panel, instruments, empty_portfolio(), policy)
        ids = [a.instrument.instrument_id for a in allocations]
        assert ids == sorted(ids)

    def test_turnover_interpolation_preserves_gross_for_feasible_current(self) -> None:
        policy = StockRiskPolicy(top_k=20, turnover_budget=0.2)
        panel = scored_panel(seed=41)
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        current_position = next(iter(instruments.values()))
        price = panel.filter(pl.col("instrument_id") == current_position.instrument_id).sort(
            "session"
        )["close"][-1]
        portfolio = PortfolioSnapshot(
            account_snapshot_id="held",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=equity - 0.03 * equity,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=current_position,
                    quantity=int(0.03 * equity // price),
                    average_cost=price,
                ),
            ),
        )
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        new_equity = equity_of(panel, portfolio)
        assert sum(a.target_value for a in allocations) <= new_equity * policy.gross_cap + 1e-8

    def test_unknown_instrument_raises_constraint_error(self) -> None:
        policy = StockRiskPolicy()
        panel = scored_panel(seed=51)
        instruments = instruments_for(5)  # only 5 of 10 present
        with pytest.raises(PortfolioConstraintError, match="unknown instruments"):
            construct_target_allocations(panel, instruments, empty_portfolio(), policy)


class TestDeRisk:
    def test_infeasible_current_produces_only_reductions(self) -> None:
        policy = StockRiskPolicy(top_k=20, single_name_cap=0.08)
        panel = scored_panel(seed=61)
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        instrument = next(iter(instruments.values()))
        price = panel.filter(pl.col("instrument_id") == instrument.instrument_id).sort(
            "session"
        )["close"][-1]
        over_cap_quantity = int(0.25 * equity // price)
        portfolio = PortfolioSnapshot(
            account_snapshot_id="over",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=0.0,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=instrument,
                    quantity=over_cap_quantity,
                    average_cost=price,
                ),
            ),
        )
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert allocations
        for a in allocations:
            assert a.target_value <= equity * policy.single_name_cap + 1e-8
