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


def _economic_panel(
    n_sessions: int = 30,
    n_tickers: int = 10,
    seed: int = 3,
    *,
    positive_top: int = 3,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "session": start + timedelta(days=s),
            "instrument_id": f"KRX:{t:06d}",
            "pred_score": float(t) * 0.1 + s % 2,
            "sector": f"S{t % 3}",
            "adtv": 1e9 * t,
            "close": 50_000.0 + t + s,
            "ret": float(rng.normal(0.0002, 0.01)),
            "expected_active_alpha": 0.01 if t <= positive_top else -0.01,
            "expected_net_alpha": 0.008 if t <= positive_top else -0.012,
            "alpha_lower_bound": 0.004 if t <= positive_top else -0.002,
            "exit_cost_rate": 0.002,
        }
        for s in range(n_sessions)
        for t in range(1, n_tickers + 1)
    ]
    return pl.DataFrame(rows)


def _holding_portfolio(
    instrument_id: str,
    price: float,
    equity: float,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        account_snapshot_id="held",
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        settled_cash=equity - 0.05 * equity,
        unsettled_cash=0.0,
        positions=(
            Position(
                instrument=Instrument(
                    instrument_id, AssetKind.STOCK, "KRX", instrument_id.split(":")[-1], "KRW", lot_size=1
                ),
                quantity=int(0.05 * equity // price),
                average_cost=price,
            ),
        ),
    )


class TestEconomicAllocation:
    def test_only_positive_net_alpha_names_enter(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _economic_panel(positive_top=3)
        instruments = instruments_for(10)
        allocations = construct_target_allocations(panel, instruments, empty_portfolio(), policy)
        assert [a.instrument.instrument_id for a in allocations] == [
            f"KRX:{t:06d}" for t in (1, 2, 3)
        ]

    def test_non_positive_net_alpha_panel_returns_empty_cash(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _economic_panel(positive_top=0)
        instruments = instruments_for(10)
        assert (
            construct_target_allocations(panel, instruments, empty_portfolio(), policy)
            == ()
        )

    def test_missing_alpha_columns_preserves_legacy_rank_behavior(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _economic_panel().drop(
            [
                "expected_active_alpha",
                "expected_net_alpha",
                "alpha_lower_bound",
                "exit_cost_rate",
            ]
        )
        instruments = instruments_for(10)
        allocations = construct_target_allocations(panel, instruments, empty_portfolio(), policy)
        assert allocations
        assert len(allocations) <= policy.top_k

    def test_hard_caps_are_preserved_with_economic_columns(self) -> None:
        policy = StockRiskPolicy(top_k=20, single_name_cap=0.08, gross_cap=0.9)
        panel = _economic_panel(positive_top=8)
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        allocations = construct_target_allocations(panel, instruments, empty_portfolio(), policy)
        assert allocations
        assert sum(a.target_value for a in allocations) <= equity * policy.gross_cap + 1e-8
        for a in allocations:
            assert a.target_value <= equity * policy.single_name_cap + 1e-8

    def test_incumbent_with_positive_keep_benefit_is_retained(self) -> None:
        policy = StockRiskPolicy(top_k=20, turnover_budget=0.5)
        panel = _economic_panel(positive_top=3)
        instruments = instruments_for(10)
        held_id = "KRX:000002"
        equity = equity_of(panel, empty_portfolio())
        price = panel.filter(pl.col("instrument_id") == held_id).sort("session")["close"][-1]
        portfolio = _holding_portfolio(held_id, price, equity)
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert any(a.instrument.instrument_id == held_id for a in allocations)

    def test_incumbent_failing_keep_gate_is_exited(self) -> None:
        policy = StockRiskPolicy(top_k=20, turnover_budget=0.0)
        panel = _economic_panel(positive_top=3)
        instruments = instruments_for(10)
        held_id = "KRX:000009"
        equity = equity_of(panel, empty_portfolio())
        price = panel.filter(pl.col("instrument_id") == held_id).sort("session")["close"][-1]
        portfolio = _holding_portfolio(held_id, price, equity)
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert all(a.instrument.instrument_id != held_id for a in allocations)


def _with_economic(panel: pl.DataFrame, *, positive: bool = True) -> pl.DataFrame:
    """Attach cost-adjusted net-alpha evidence to a scored panel."""
    rows = panel.to_dicts()
    for row in rows:
        row["expected_active_alpha"] = 0.02 if positive else -0.01
        row["expected_net_alpha"] = 0.015 if positive else -0.015
        row["alpha_lower_bound"] = 0.005 if positive else -0.003
        row["exit_cost_rate"] = 0.002
    return pl.DataFrame(rows)


class TestEconomicGating:
    def test_only_positive_net_alpha_entries_are_allocated(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _with_economic(scored_panel(seed=71), positive=True)
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert allocations
        assert all(a.reason == "inverse-vol-constrained" for a in allocations)

    def test_positive_gross_but_non_positive_net_lower_bound_cannot_enter(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _with_economic(scored_panel(seed=76), positive=True).with_columns(
            pl.lit(-0.001, dtype=pl.Float64).alias("net_alpha_lower_bound")
        )
        allocations = construct_target_allocations(
            panel, instruments_for(10), empty_portfolio(), policy
        )
        assert allocations == ()

    def test_positive_net_lower_bound_allows_entry(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _with_economic(scored_panel(seed=77), positive=True).with_columns(
            pl.lit(0.001, dtype=pl.Float64).alias("net_alpha_lower_bound")
        )
        allocations = construct_target_allocations(
            panel, instruments_for(10), empty_portfolio(), policy
        )
        assert allocations
        assert all(a.reason == "inverse-vol-constrained" for a in allocations)

    def test_non_positive_net_alpha_yields_no_synthetic_long(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _with_economic(scored_panel(seed=72), positive=False)
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert allocations == ()

    def test_missing_or_non_finite_alpha_is_never_a_buy(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = scored_panel(seed=73).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("expected_active_alpha"),
            pl.lit(None, dtype=pl.Float64).alias("expected_net_alpha"),
            pl.lit(None, dtype=pl.Float64).alias("alpha_lower_bound"),
            pl.lit(0.002, dtype=pl.Float64).alias("exit_cost_rate"),
        )
        allocations = construct_target_allocations(
            panel, instruments_for(10), empty_portfolio(), policy
        )
        assert allocations == ()

    def test_incumbent_kept_when_net_benefit_positive_and_exited_otherwise(self) -> None:
        policy = StockRiskPolicy(top_k=20)
        panel = _with_economic(scored_panel(seed=74), positive=True)
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        held = next(iter(instruments.values()))
        price = panel.filter(pl.col("instrument_id") == held.instrument_id).sort(
            "session"
        )["close"][-1]
        portfolio = PortfolioSnapshot(
            account_snapshot_id="held",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=equity - 0.03 * equity,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=held,
                    quantity=int(0.03 * equity // price),
                    average_cost=price,
                ),
            ),
        )
        kept = construct_target_allocations(panel, instruments, portfolio, policy)
        assert any(a.instrument.instrument_id == held.instrument_id for a in kept)

        exiting_panel = _with_economic(scored_panel(seed=75), positive=False).with_columns(
            pl.when(pl.col("instrument_id") == held.instrument_id)
            .then(pl.lit(0.0))
            .otherwise(pl.col("expected_net_alpha"))
            .alias("expected_net_alpha")
        )
        exiting = construct_target_allocations(exiting_panel, instruments, portfolio, policy)
        assert all(
            a.instrument.instrument_id != held.instrument_id for a in exiting
        )


def test_prepared_allocations_match_reference_constructor() -> None:
    """The array-backed prepared constructor is bit-identical to the reference."""
    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    policy = StockRiskPolicy(top_k=20, participation_limit=0.01)
    instruments = instruments_for(10)
    portfolio = empty_portfolio()

    reference = construct_target_allocations(panel, instruments, portfolio, policy)
    assert reference

    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        construct_target_allocations_prepared,
    )

    market = PreparedAllocationMarket.build(panel)
    overlay = (
        panel.sort(["session", "instrument_id"])["pred_score"].to_numpy().astype(float)
    )
    allocations = construct_target_allocations_prepared(
        market,
        len(market.sessions) - 1,
        overlay,
        None,
        instruments,
        portfolio,
        policy,
    )
    assert allocations == reference


def test_prepared_allocations_reject_overlay_length_mismatch() -> None:
    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        construct_target_allocations_prepared,
    )

    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    market = PreparedAllocationMarket.build(panel)
    with pytest.raises(ValueError, match="row count"):
        construct_target_allocations_prepared(
            market,
            len(market.sessions) - 1,
            np.asarray([0.5] * (market.row_count - 1)),
            None,
            instruments_for(10),
            empty_portfolio(),
            StockRiskPolicy(top_k=20),
        )
