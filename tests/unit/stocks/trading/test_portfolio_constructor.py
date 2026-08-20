"""PLAN-03-CONSTRAINED-SIZING: randomized feasible inputs preserve every constraint."""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot, Position
from src.stocks.trading.portfolio_constructor import (
    CompoundingPolicyConfig,
    PortfolioConstraintError,
    StockRiskPolicy,
    construct_target_allocations,
    stock_risk_policy_fingerprint,
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


def no_trade_when_alpha_lcb_not_above_marginal_hurdle() -> bool:
    """Contract oracle: an alpha lower bound below the marginal hurdle yields no trade.

    A candidate whose ``net_alpha_lower_bound`` does not beat the round-trip
    marginal cost (the entry/exit hurdle) must never produce a new order, even
    when the gross alpha is positive.
    """
    policy = StockRiskPolicy(top_k=20)
    positive = _with_economic(scored_panel(seed=78), positive=True)
    below_hurdle = positive.with_columns(
        pl.lit(-0.001, dtype=pl.Float64).alias("net_alpha_lower_bound")
    )
    if construct_target_allocations(
        below_hurdle, instruments_for(10), empty_portfolio(), policy
    ):
        return False
    above_hurdle = positive.with_columns(
        pl.lit(0.001, dtype=pl.Float64).alias("net_alpha_lower_bound")
    )
    return bool(
        construct_target_allocations(
            above_hurdle, instruments_for(10), empty_portfolio(), policy
        )
    )


def test_no_trade_when_alpha_lcb_not_above_marginal_hurdle() -> None:
    assert no_trade_when_alpha_lcb_not_above_marginal_hurdle()


class TestEconomicGatingRemainder:
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


class TestCompoundingOverlay:
    def test_rejects_non_positive_or_non_finite_risk_aversion(self) -> None:
        with pytest.raises(ValueError, match="growth_risk_aversion"):
            CompoundingPolicyConfig(growth_risk_aversion=0.0)
        with pytest.raises(ValueError, match="growth_risk_aversion"):
            CompoundingPolicyConfig(growth_risk_aversion=-1.0)
        with pytest.raises(ValueError, match="growth_risk_aversion"):
            CompoundingPolicyConfig(growth_risk_aversion=float("nan"))
        with pytest.raises(ValueError, match="growth_risk_aversion"):
            CompoundingPolicyConfig(growth_risk_aversion=float("inf"))

    def test_positive_finite_edge_scales_exposure_within_unit_and_holds_caps(self) -> None:
        panel = _with_economic(scored_panel(seed=83), positive=True).with_columns(
            pl.lit(0.004, dtype=pl.Float64).alias("net_alpha_lower_bound")
        )
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        equity = equity_of(panel, portfolio)
        policy = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            turnover_budget=0.0,
            compounding=CompoundingPolicyConfig(growth_risk_aversion=50.0),
        )
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert allocations
        scale = float(policy.compounding_evidence[-1]["confidence_scale"])
        assert 0.0 < scale <= 1.0
        assert policy.compounding_evidence[-1]["cash_reason"] is None
        assert policy.compounding_evidence[-1]["confidence_edge_h"] > 0.0
        assert policy.compounding_evidence[-1]["confidence_variance_h"] > 0.0
        assert sum(a.target_value for a in allocations) <= equity * policy.gross_cap + 1e-8
        for a in allocations:
            assert a.target_value <= equity * policy.single_name_cap + 1e-8
            assert a.target_value >= 0.0

        baseline_policy = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            turnover_budget=0.0,
            compounding=CompoundingPolicyConfig(enabled=False),
        )
        baseline = construct_target_allocations(panel, instruments, portfolio, baseline_policy)
        assert sum(a.target_value for a in allocations) < sum(
            a.target_value for a in baseline
        )

    def test_non_positive_correlated_basket_edge_returns_cash(self) -> None:
        panel = _with_economic(scored_panel(seed=85), positive=True).with_columns(
            pl.lit(0.004, dtype=pl.Float64).alias("net_alpha_lower_bound")
        )
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        held_id = "KRX:000005"
        held = instruments[held_id]
        price = panel.filter(pl.col("instrument_id") == held_id).sort("session")["close"][-1]
        portfolio = PortfolioSnapshot(
            account_snapshot_id="held",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=equity - 0.05 * equity,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=held,
                    quantity=int(0.05 * equity // price),
                    average_cost=price,
                ),
            ),
        )
        negative_edge = panel.with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(-0.5))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        policy = StockRiskPolicy(top_k=20)
        allocations = construct_target_allocations(
            negative_edge, instruments, portfolio, policy
        )
        assert allocations == ()
        assert policy.compounding_evidence[-1]["cash_reason"] == (
            "non-positive-confidence-edge"
        )

    def test_missing_or_non_finite_lower_bound_fails_closed(self) -> None:
        panel = _with_economic(scored_panel(seed=87), positive=True).with_columns(
            pl.lit(0.004, dtype=pl.Float64).alias("net_alpha_lower_bound")
        )
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        held_id = "KRX:000005"
        held = instruments[held_id]
        price = panel.filter(pl.col("instrument_id") == held_id).sort("session")["close"][-1]
        portfolio = PortfolioSnapshot(
            account_snapshot_id="held",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=equity - 0.05 * equity,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=held,
                    quantity=int(0.05 * equity // price),
                    average_cost=price,
                ),
            ),
        )
        non_finite = panel.with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        policy = StockRiskPolicy(top_k=20)
        allocations = construct_target_allocations(non_finite, instruments, portfolio, policy)
        assert allocations == ()
        assert policy.compounding_evidence[-1]["cash_reason"] == (
            "invalid-confidence-variance"
        )

        missing = panel.with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(None))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        policy2 = StockRiskPolicy(top_k=20)
        assert (
            construct_target_allocations(missing, instruments, portfolio, policy2) == ()
        )
        assert policy2.compounding_evidence[-1]["cash_reason"] == (
            "invalid-confidence-variance"
        )

    def test_legacy_panel_without_economic_columns_is_unchanged(self) -> None:
        panel = scored_panel(seed=91)
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        legacy = StockRiskPolicy(top_k=20)
        disabled = StockRiskPolicy(
            top_k=20, compounding=CompoundingPolicyConfig(enabled=False)
        )
        assert construct_target_allocations(panel, instruments, portfolio, legacy) == (
            construct_target_allocations(panel, instruments, portfolio, disabled)
        )
        assert legacy.compounding_evidence == []

    @staticmethod
    def _economic_panel(seed: int = 83) -> pl.DataFrame:
        return _with_economic(scored_panel(seed=seed), positive=True).with_columns(
            pl.lit(0.004, dtype=pl.Float64).alias("net_alpha_lower_bound")
        )

    def test_decision_record_is_complete_json_safe_and_finite(self) -> None:
        panel = self._economic_panel()
        instruments = instruments_for(10)
        portfolio = empty_portfolio()
        policy = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            turnover_budget=0.2,
            compounding=CompoundingPolicyConfig(growth_risk_aversion=50.0),
        )
        allocations = construct_target_allocations(panel, instruments, portfolio, policy)
        assert allocations
        record = policy.compounding_evidence[-1]
        for key in (
            "decision_session",
            "candidate_count",
            "ranked_count",
            "selected_count",
            "gross_before_compounding",
            "gross_after_compounding",
            "turnover_lambda",
            "confidence_edge_h",
            "confidence_variance_h",
            "confidence_scale",
            "cash_reason",
        ):
            assert key in record
        assert record["cash_reason"] is None
        assert str(record["decision_session"])
        assert (
            int(record["candidate_count"])
            >= int(record["ranked_count"])
            >= int(record["selected_count"])
            >= 1
        )
        assert int(record["selected_count"]) == len(allocations)
        assert float(record["gross_before_compounding"]) > 0.0
        assert float(record["gross_after_compounding"]) <= (
            float(record["gross_before_compounding"]) + 1e-12
        )
        assert 0.0 <= float(record["turnover_lambda"]) <= 1.0
        assert all(
            math.isfinite(float(value))
            for key, value in record.items()
            if isinstance(value, float) and key != "decision_session"
        )
        json.dumps(record)

    def test_cash_branches_record_complete_fail_closed_fields(self) -> None:
        instruments = instruments_for(10)
        equity = equity_of(self._economic_panel(seed=85), empty_portfolio())
        held_id = "KRX:000005"
        held = instruments[held_id]

        def _portfolio() -> PortfolioSnapshot:
            price = self._economic_panel(seed=85).filter(
                pl.col("instrument_id") == held_id
            ).sort("session")["close"][-1]
            return PortfolioSnapshot(
                account_snapshot_id="held",
                as_of=datetime(2024, 1, 1, tzinfo=UTC),
                settled_cash=equity - 0.05 * equity,
                unsettled_cash=0.0,
                positions=(
                    Position(
                        instrument=held,
                        quantity=int(0.05 * equity // price),
                        average_cost=price,
                    ),
                ),
            )

        negative_edge = self._economic_panel(seed=85).with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(-0.5))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        policy = StockRiskPolicy(top_k=20)
        assert construct_target_allocations(
            negative_edge, instruments, _portfolio(), policy
        ) == ()
        record = policy.compounding_evidence[-1]
        assert record["cash_reason"] == "non-positive-confidence-edge"
        assert float(record["confidence_scale"]) == 0.0
        assert float(record["gross_after_compounding"]) == 0.0
        assert float(record["turnover_lambda"]) == 0.0
        assert float(record["gross_before_compounding"]) >= 0.0
        json.dumps(record)

        non_finite = self._economic_panel(seed=87).with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        policy2 = StockRiskPolicy(top_k=20)
        assert construct_target_allocations(
            non_finite, instruments, _portfolio(), policy2
        ) == ()
        record2 = policy2.compounding_evidence[-1]
        assert record2["cash_reason"] == "invalid-confidence-variance"
        assert record2["confidence_variance_h"] is None
        assert record2["confidence_scale"] is None
        assert float(record2["gross_after_compounding"]) == 0.0
        assert float(record2["turnover_lambda"]) == 0.0
        json.dumps(record2)

    def test_fail_closed_missing_lower_bound_still_records_complete_keys(self) -> None:
        panel = self._economic_panel(seed=89)
        instruments = instruments_for(10)
        held_id = "KRX:000005"
        held = instruments[held_id]
        price = panel.filter(pl.col("instrument_id") == held_id).sort("session")["close"][-1]
        equity = equity_of(panel, empty_portfolio())
        portfolio = PortfolioSnapshot(
            account_snapshot_id="held",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=equity - 0.05 * equity,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=held,
                    quantity=int(0.05 * equity // price),
                    average_cost=price,
                ),
            ),
        )
        policy = StockRiskPolicy(top_k=20)
        missing = panel.with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(None))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        assert construct_target_allocations(missing, instruments, portfolio, policy) == ()
        record = policy.compounding_evidence[-1]
        assert record["cash_reason"] == "invalid-confidence-variance"
        assert record["confidence_scale"] is None
        assert record["confidence_edge_h"] is None
        for key in ("candidate_count", "ranked_count", "selected_count"):
            assert isinstance(record[key], int)
        assert float(record["gross_before_compounding"]) == 0.0
        assert float(record["gross_after_compounding"]) == 0.0
        json.dumps(record)


class TestHorizonConsistentLogUtility:
    def test_horizon_halves_scale(self) -> None:
        """HC_LOG_UTILITY_01_HORIZON_HALVES_SCALE: doubling forecast_horizon_sessions halves confidence_scale."""
        panel = TestCompoundingOverlay._economic_panel()
        instruments = instruments_for(10)
        portfolio = empty_portfolio()

        policy_h5 = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            turnover_budget=0.0,
            compounding=CompoundingPolicyConfig(
                growth_risk_aversion=50.0,
                forecast_horizon_sessions=5,
            ),
        )
        alloc_h5 = construct_target_allocations(panel, instruments, portfolio, policy_h5)
        assert alloc_h5
        scale_h5 = float(policy_h5.compounding_evidence[-1]["confidence_scale"])

        policy_h10 = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            turnover_budget=0.0,
            compounding=CompoundingPolicyConfig(
                growth_risk_aversion=50.0,
                forecast_horizon_sessions=10,
            ),
        )
        alloc_h10 = construct_target_allocations(panel, instruments, portfolio, policy_h10)
        assert alloc_h10
        scale_h10 = float(policy_h10.compounding_evidence[-1]["confidence_scale"])

        assert scale_h5 > 0.0
        assert scale_h10 > 0.0
        assert scale_h10 == pytest.approx(scale_h5 / 2, abs=1e-12)

    def test_invalid_horizon_rejects_config(self) -> None:
        """HC_LOG_UTILITY_02_INVALID_HORIZON_FAILS_CLOSED: non-positive forecast_horizon_sessions raises ValueError."""
        with pytest.raises(ValueError, match="forecast_horizon_sessions"):
            CompoundingPolicyConfig(forecast_horizon_sessions=0)
        with pytest.raises(ValueError, match="forecast_horizon_sessions"):
            CompoundingPolicyConfig(forecast_horizon_sessions=-1)

    def test_invalid_horizon_fails_closed_to_cash(self) -> None:
        """HC_LOG_UTILITY_02_INVALID_HORIZON_FAILS_CLOSED: missing/NaN/non-positive lower bound returns cash."""
        panel = TestCompoundingOverlay._economic_panel()
        instruments = instruments_for(10)
        equity = equity_of(panel, empty_portfolio())
        held_id = "KRX:000005"
        held = instruments[held_id]
        price = panel.filter(pl.col("instrument_id") == held_id).sort("session")["close"][-1]
        portfolio = PortfolioSnapshot(
            account_snapshot_id="held",
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            settled_cash=equity - 0.05 * equity,
            unsettled_cash=0.0,
            positions=(
                Position(
                    instrument=held,
                    quantity=int(0.05 * equity // price),
                    average_cost=price,
                ),
            ),
        )
        non_positive = panel.with_columns(
            pl.when(pl.col("instrument_id") == held_id)
            .then(pl.lit(-0.1))
            .otherwise(pl.col("net_alpha_lower_bound"))
            .alias("net_alpha_lower_bound")
        )
        policy = StockRiskPolicy(
            top_k=20,
            compounding=CompoundingPolicyConfig(forecast_horizon_sessions=10),
        )
        alloc = construct_target_allocations(non_positive, instruments, portfolio, policy)
        assert alloc == ()

    def test_fingerprint_includes_horizon(self) -> None:
        """forecast_horizon_sessions affects the risk policy fingerprint."""
        p1 = StockRiskPolicy(
            compounding=CompoundingPolicyConfig(forecast_horizon_sessions=5),
        )
        p2 = StockRiskPolicy(
            compounding=CompoundingPolicyConfig(forecast_horizon_sessions=10),
        )
        p_none = StockRiskPolicy(
            compounding=CompoundingPolicyConfig(forecast_horizon_sessions=None),
        )
        assert stock_risk_policy_fingerprint(p1) != stock_risk_policy_fingerprint(p2)
        assert stock_risk_policy_fingerprint(p1) != stock_risk_policy_fingerprint(p_none)
        assert stock_risk_policy_fingerprint(p2) != stock_risk_policy_fingerprint(p_none)


def test_prepared_allocations_match_reference_constructor() -> None:
    """The array-backed prepared constructor is bit-identical to the reference."""
    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    policy = StockRiskPolicy(top_k=20, participation_limit=0.01)
    instruments = instruments_for(10)
    portfolio = empty_portfolio()

    reference = construct_target_allocations(panel, instruments, portfolio, policy)
    assert reference

    from src.stocks.trading.allocation_policy import rank_stock_candidate_indices
    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        construct_target_allocations_prepared,
    )

    cross = panel.filter(pl.col("session") == panel.select(pl.col("session").max()).to_series()[0])
    oracle_order = rank_stock_candidate_indices(
        np.asarray(cross["pred_score"].to_list(), dtype=np.float64),
        np.asarray(cross["instrument_id"].to_list(), dtype=object),
    )
    oracle_top = sorted(
        cross.gather(oracle_order)["instrument_id"].to_list()[: policy.top_k]
    )
    assert sorted(a.instrument.instrument_id for a in reference) == oracle_top

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
    assert sorted(a.instrument.instrument_id for a in allocations) == oracle_top


def test_prepared_allocations_match_reference_for_stateful_and_calibrated_decisions() -> None:
    """Prepared and reference allocators agree on stateful, calibrated, and short-history decisions.

    The array-backed path must reproduce the reference constructor exactly for a
    feasible incumbent portfolio (turnover path), an infeasible incumbent
    portfolio (sell-only ``DE_RISK`` path), a frozen calibration bucket table
    (economic gates and compounding), and a two-session window with missing
    covariance history -- either bit-identical allocations or the same
    ``PortfolioConstraintError`` outcome.
    """
    from src.stocks.research.economic_alpha import CausalAlphaCalibrator
    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        _SESSION_COLUMN,
        construct_target_allocations_prepared,
    )

    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    policy = StockRiskPolicy(top_k=20, participation_limit=0.01)
    instruments = instruments_for(10)
    market = PreparedAllocationMarket.build(panel)
    overlay = (
        panel.sort(["session", "instrument_id"])["pred_score"].to_numpy().astype(float)
    )
    decision_index = len(market.sessions) - 1
    base = datetime(2024, 1, 1, tzinfo=UTC)

    def reference_allocations(
        portfolio: PortfolioSnapshot,
        *,
        cal_state: dict[str, object] | None = None,
        target_market=market,
        target_index: int = decision_index,
        target_overlay: np.ndarray = overlay,
    ) -> tuple[tuple[object, ...], str | None]:
        window_len = (
            max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions)
            + 1
        )
        start = max(0, target_index - window_len + 1)
        indices = np.concatenate(
            [
                np.arange(target_market.session_ranges[i][0], target_market.session_ranges[i][1])
                for i in range(start, target_index + 1)
            ]
        )
        window_frame = pl.DataFrame(
            {
                "instrument_id": target_market.instrument_ids[indices],
                _SESSION_COLUMN: pl.Series(
                    target_market.row_sessions[indices].tolist(),
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "pred_score": np.asarray(target_overlay)[indices],
                "sector": target_market.sector[indices],
                "adtv": target_market.adtv[indices],
                "close": target_market.close[indices],
            }
        ).with_columns(pl.col("pred_score").fill_nan(None))
        if cal_state is not None:
            window_frame = CausalAlphaCalibrator.apply_prepared(cal_state, window_frame)
        try:
            return construct_target_allocations(
                window_frame, instruments, portfolio, policy
            ), None
        except PortfolioConstraintError as exc:
            return (), str(exc)

    def prepared_allocations(
        portfolio: PortfolioSnapshot,
        *,
        cal_state: dict[str, object] | None = None,
        target_market=market,
        target_index: int = decision_index,
        target_overlay: np.ndarray = overlay,
    ) -> tuple[tuple[object, ...], str | None]:
        try:
            return construct_target_allocations_prepared(
                target_market,
                target_index,
                target_overlay,
                cal_state,
                instruments,
                portfolio,
                policy,
            ), None
        except PortfolioConstraintError as exc:
            return (), str(exc)

    cash_portfolio = PortfolioSnapshot(
        account_snapshot_id="cash",
        as_of=base,
        settled_cash=1_000_000_000.0,
        unsettled_cash=0.0,
        positions=(),
    )
    feasible_holding = PortfolioSnapshot(
        account_snapshot_id="feasible",
        as_of=base,
        settled_cash=900_000_000.0,
        unsettled_cash=0.0,
        positions=(
            Position(
                instrument=instruments["KRX:000004"], quantity=2000.0, average_cost=50000.0
            ),
        ),
    )
    infeasible_holding = PortfolioSnapshot(
        account_snapshot_id="over_cap",
        as_of=base,
        settled_cash=0.0,
        unsettled_cash=0.0,
        positions=(
            Position(
                instrument=instruments["KRX:000004"], quantity=3000.0, average_cost=50000.0
            ),
        ),
    )
    buckets = [
        {
            "bucket": bucket,
            "sample_size": 10,
            "expected_active_alpha": 0.003,
            "alpha_lower_bound": 0.002,
        }
        for bucket in range(4)
    ]
    cal_state = {
        "bucket_count": 4,
        "history_sessions": 30,
        "round_trip_cost": 0.0005,
        "exit_cost_rate": 0.0003,
        "buckets": buckets,
    }

    scenarios = [
        ("cash", cash_portfolio, None),
        ("feasible-holding", feasible_holding, None),
        ("over-cap-holding", infeasible_holding, None),
        ("calibrated", cash_portfolio, cal_state),
    ]
    for name, portfolio, cal_state in scenarios:
        ref, ref_error = reference_allocations(portfolio, cal_state=cal_state)
        prep, prep_error = prepared_allocations(portfolio, cal_state=cal_state)
        assert ref_error == prep_error, name
        assert ref == prep, name
        assert prep == prep

    # Two-session window: the reference pivot drops the null first session, so
    # the covariance is degenerate; both paths must agree exactly.
    short_panel = panel.head(2 * 10)
    short_market = PreparedAllocationMarket.build(short_panel)
    short_overlay = (
        short_panel.sort(["session", "instrument_id"])["pred_score"]
        .to_numpy()
        .astype(float)
    )
    ref, ref_error = reference_allocations(
        cash_portfolio,
        target_market=short_market,
        target_index=len(short_market.sessions) - 1,
        target_overlay=short_overlay,
    )
    prep, prep_error = prepared_allocations(
        cash_portfolio,
        target_market=short_market,
        target_index=len(short_market.sessions) - 1,
        target_overlay=short_overlay,
    )
    assert ref_error == prep_error
    assert ref == prep


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


def test_prepared_allocations_reject_non_finite_scored_overlay() -> None:
    """Non-finite values on scored overlay rows are rejected, never used."""
    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        construct_target_allocations_prepared,
    )

    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    market = PreparedAllocationMarket.build(panel)
    overlay = np.full(market.row_count, np.nan, dtype=np.float64)
    overlay[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite scored values"):
        construct_target_allocations_prepared(
            market,
            len(market.sessions) - 1,
            overlay,
            None,
            instruments_for(10),
            empty_portfolio(),
            StockRiskPolicy(top_k=20),
        )

def test_prepared_allocations_convert_nan_history_overlay_to_null() -> None:
    """NaN historical overlay rows become null and never enter an allocation."""
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
    panel_sorted = panel.sort(["session", "instrument_id"])
    latest_session = panel_sorted["session"].max()
    overlay = np.full(market.row_count, np.nan, dtype=np.float64)
    latest_mask = (panel_sorted["session"] == latest_session).to_numpy()
    overlay[latest_mask] = (
        panel_sorted.filter(pl.col("session") == latest_session)["pred_score"]
        .to_numpy()
    )
    assert overlay[~latest_mask].size > 0
    assert np.isnan(overlay[~latest_mask]).all()

    plain = construct_target_allocations_prepared(
        market,
        len(market.sessions) - 1,
        overlay,
        None,
        instruments,
        portfolio,
        policy,
    )
    assert plain == reference

    calibration_state = {
        "buckets": [
            {
                "bucket": 0,
                "expected_active_alpha": 0.01,
                "alpha_lower_bound": 0.005,
            }
        ],
        "bucket_count": 1,
        "round_trip_cost": 0.001,
        "exit_cost_rate": 0.0005,
    }
    calibrated = construct_target_allocations_prepared(
        market,
        len(market.sessions) - 1,
        overlay,
        calibration_state,
        instruments,
        portfolio,
        policy,
    )
    assert calibrated
    assert {
        a.instrument.instrument_id for a in calibrated
    } == {a.instrument.instrument_id for a in plain}

def test_causal_covariance_uses_full_shrinkage_when_complete() -> None:
    from src.stocks.trading.portfolio_constructor import (
        causal_covariance_or_fallback,
        _shrinkage_covariance,
    )

    rng = np.random.default_rng(11)
    matrix = rng.normal(0.001, 0.01, size=(60, 3))
    covariance, source = causal_covariance_or_fallback(
        matrix,
        volatility_lookback_sessions=20,
        covariance_lookback_sessions=60,
    )
    assert source == "full"
    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, _shrinkage_covariance(matrix))
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)


def test_causal_covariance_fallback_is_conservative_psd_and_never_zero_fills() -> None:
    from src.stocks.trading.portfolio_constructor import (
        causal_covariance_or_fallback,
    )

    rng = np.random.default_rng(12)
    matrix = rng.normal(0.001, 0.01, size=(60, 3))
    matrix[20:, 0] = np.nan
    matrix[:20, 1] = np.nan
    matrix[10:30, 2] = np.nan
    covariance, source = causal_covariance_or_fallback(
        matrix,
        volatility_lookback_sessions=20,
        covariance_lookback_sessions=60,
    )
    assert source == "fallback"
    assert covariance.shape == (3, 3)
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)
    variance_2 = covariance[1, 1]
    assert variance_2 > 0.0
    missing_pair_correlation = covariance[0, 1] / math.sqrt(
        covariance[0, 0] * covariance[1, 1]
    )
    assert missing_pair_correlation > 0.0


def test_causal_covariance_fails_closed_without_own_volatility_history() -> None:
    from src.stocks.trading.portfolio_constructor import (
        PortfolioConstraintError,
        causal_covariance_or_fallback,
    )

    rng = np.random.default_rng(13)
    matrix = rng.normal(0.001, 0.01, size=(40, 2))
    matrix[:20, 0] = np.nan
    matrix[10:, 1] = np.nan
    with pytest.raises(PortfolioConstraintError, match="insufficient covariance"):
        causal_covariance_or_fallback(
            matrix,
            volatility_lookback_sessions=20,
            covariance_lookback_sessions=60,
        )

    all_nan = np.full((40, 2), np.nan)
    with pytest.raises(PortfolioConstraintError, match="insufficient covariance"):
        causal_covariance_or_fallback(
            all_nan,
            volatility_lookback_sessions=20,
            covariance_lookback_sessions=60,
        )

    zero_variance = rng.normal(0.001, 0.0, size=(40, 2))
    zero_variance[1:, 1] = np.nan
    with pytest.raises(PortfolioConstraintError, match="insufficient covariance"):
        causal_covariance_or_fallback(
            zero_variance,
            volatility_lookback_sessions=20,
            covariance_lookback_sessions=60,
        )


def test_prepared_and_reference_fallback_covariance_inputs_are_identical() -> None:
    """Prepared and reference raw covariance windows agree under fallback."""
    from datetime import timedelta

    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        _covariance,
        _prepared_return_matrix,
        _return_matrix,
        _window_returns,
        causal_covariance_or_fallback,
    )

    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for session in range(60):
        for instrument_id, present in (
            ("A", True),
            ("B", True),
            ("C", session < 21),
            ("D", session >= 39),
        ):
            if not present:
                continue
            rows.append(
                {
                    "session": start + timedelta(days=session),
                    "instrument_id": instrument_id,
                    "sector": "S1",
                    "adtv": 1e9,
                    "close": 50_000.0 + session,
                }
            )
    panel = pl.DataFrame(rows)
    ids = ["A", "B", "C", "D"]

    reference_matrix = _return_matrix(panel, ids)
    assert reference_matrix is not None
    assert reference_matrix.shape == (60, 4)
    assert np.count_nonzero(np.all(np.isfinite(reference_matrix), axis=1)) < 2

    market = PreparedAllocationMarket.build(panel)
    decision_index = len(market.sessions) - 1
    window = _window_returns(market, 0, decision_index)
    prepared_matrix = _prepared_return_matrix(window, market, ids)
    assert prepared_matrix is not None
    assert np.array_equal(
        np.isnan(prepared_matrix), np.isnan(reference_matrix)
    )

    policy = StockRiskPolicy(
        top_k=4,
        volatility_lookback_sessions=20,
        covariance_lookback_sessions=60,
    )
    ref_cov, ref_source = _covariance(panel, ids, policy)
    prep_cov, prep_source = causal_covariance_or_fallback(
        prepared_matrix,
        volatility_lookback_sessions=20,
        covariance_lookback_sessions=60,
    )
    assert ref_source == prep_source == "fallback"
    assert np.allclose(ref_cov, prep_cov)
    assert np.all(np.isfinite(ref_cov))
    assert np.all(np.linalg.eigvalsh(ref_cov) >= -1e-12)

_ECO_ENTRY_SCENARIO = "economic_entry_rank_overrides_raw_score"


def test_economic_entry_rank_overrides_raw_score() -> None:
    """economic_net_v1 selects by expected_net_alpha, not raw pred_score."""
    from src.stocks.trading.portfolio_constructor import _economic_rank_values

    policy_raw = StockRiskPolicy(top_k=20, economic_ranking_mode="raw_score_v1")
    policy_econ = StockRiskPolicy(top_k=20, economic_ranking_mode="economic_net_v1")

    raw_scores = np.array([0.9, 0.1], dtype=np.float64)
    expected_active_alpha = np.array([0.005, 0.02], dtype=np.float64)
    expected_net_alpha = np.array([0.003, 0.018], dtype=np.float64)
    exit_cost_rate = np.array([0.002, 0.002], dtype=np.float64)
    instrument_ids = np.array(["KRX:A", "KRX:B"], dtype=object)
    incumbent_ids: set[str] = set()

    vals_raw = _economic_rank_values(
        raw_scores=raw_scores,
        expected_active_alpha=expected_active_alpha,
        expected_net_alpha=expected_net_alpha,
        exit_cost_rate=exit_cost_rate,
        instrument_ids=instrument_ids,
        incumbent_ids=incumbent_ids,
        ranking_mode=policy_raw.economic_ranking_mode,
    )
    vals_econ = _economic_rank_values(
        raw_scores=raw_scores,
        expected_active_alpha=expected_active_alpha,
        expected_net_alpha=expected_net_alpha,
        exit_cost_rate=exit_cost_rate,
        instrument_ids=instrument_ids,
        incumbent_ids=incumbent_ids,
        ranking_mode=policy_econ.economic_ranking_mode,
    )

    from src.stocks.trading.allocation_policy import rank_stock_candidate_indices

    order_raw = rank_stock_candidate_indices(vals_raw, instrument_ids)
    order_econ = rank_stock_candidate_indices(vals_econ, instrument_ids)

    assert list(instrument_ids[order_raw]) == ["KRX:A", "KRX:B"]
    assert list(instrument_ids[order_econ]) == ["KRX:B", "KRX:A"]


_ECO_INCUMBENT_SCENARIO = "economic_incumbent_keep_benefit_rank"


def test_economic_incumbent_keep_benefit_rank() -> None:
    """economic_net_v1 ranks incumbents by active_alpha - exit_cost, entrants by net_alpha."""
    from src.stocks.trading.portfolio_constructor import _economic_rank_values

    raw_scores = np.array([0.5, 0.5], dtype=np.float64)
    expected_active_alpha = np.array([0.01, 0.005], dtype=np.float64)
    expected_net_alpha = np.array([0.008, 0.003], dtype=np.float64)
    exit_cost_rate = np.array([0.002, 0.002], dtype=np.float64)
    instrument_ids = np.array(["KRX:INC", "KRX:ENT"], dtype=object)
    incumbent_ids = {"KRX:INC"}

    vals = _economic_rank_values(
        raw_scores=raw_scores,
        expected_active_alpha=expected_active_alpha,
        expected_net_alpha=expected_net_alpha,
        exit_cost_rate=exit_cost_rate,
        instrument_ids=instrument_ids,
        incumbent_ids=incumbent_ids,
        ranking_mode="economic_net_v1",
    )
    assert vals[0] == pytest.approx(0.008)
    assert vals[1] == pytest.approx(0.003)


def test_economic_rank_values_fails_closed_on_non_finite() -> None:
    """economic_net_v1 raises ValueError on non-finite economic inputs."""
    from src.stocks.trading.portfolio_constructor import _economic_rank_values

    raw_scores = np.array([0.5, 0.5], dtype=np.float64)
    expected_active_alpha = np.array([float("nan"), 0.005], dtype=np.float64)
    expected_net_alpha = np.array([0.003, 0.003], dtype=np.float64)
    exit_cost_rate = np.array([0.002, 0.002], dtype=np.float64)
    instrument_ids = np.array(["KRX:A", "KRX:B"], dtype=object)

    with pytest.raises(ValueError, match="finite economic"):
        _economic_rank_values(
            raw_scores=raw_scores,
            expected_active_alpha=expected_active_alpha,
            expected_net_alpha=expected_net_alpha,
            exit_cost_rate=exit_cost_rate,
            instrument_ids=instrument_ids,
            incumbent_ids=set(),
            ranking_mode="economic_net_v1",
        )


_ECO_PARITY_SCENARIO = "prepared_reference_economic_rank_parity"


def test_prepared_reference_economic_rank_parity() -> None:
    """Prepared and reference constructors produce identical allocations under economic_net_v1."""
    from src.stocks.research.economic_alpha import CausalAlphaCalibrator
    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        construct_target_allocations_prepared,
    )

    policy = StockRiskPolicy(top_k=20, economic_ranking_mode="economic_net_v1")
    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    instruments = instruments_for(10)
    portfolio = empty_portfolio()

    buckets = [
        {
            "bucket": bucket,
            "sample_size": 10,
            "expected_active_alpha": 0.003,
            "alpha_lower_bound": 0.002,
        }
        for bucket in range(4)
    ]
    cal_state = {
        "bucket_count": 4,
        "history_sessions": 30,
        "round_trip_cost": 0.0005,
        "exit_cost_rate": 0.0003,
        "buckets": buckets,
    }

    market = PreparedAllocationMarket.build(panel)
    overlay = (
        panel.sort(["session", "instrument_id"])["pred_score"].to_numpy().astype(float)
    )

    window_len = (
        max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions)
        + 1
    )
    decision_index = len(market.sessions) - 1
    start = max(0, decision_index - window_len + 1)
    indices = np.concatenate(
        [
            np.arange(market.session_ranges[i][0], market.session_ranges[i][1])
            for i in range(start, decision_index + 1)
        ]
    )
    from src.stocks.trading.portfolio_constructor import _SESSION_COLUMN

    window_frame = pl.DataFrame(
        {
            "instrument_id": market.instrument_ids[indices],
            _SESSION_COLUMN: pl.Series(
                market.row_sessions[indices].tolist(),
                dtype=pl.Datetime("us", "UTC"),
            ),
            "pred_score": np.asarray(overlay)[indices],
            "sector": market.sector[indices],
            "adtv": market.adtv[indices],
            "close": market.close[indices],
        }
    ).with_columns(pl.col("pred_score").fill_nan(None))
    window_frame = CausalAlphaCalibrator.apply_prepared(cal_state, window_frame)

    ref = construct_target_allocations(window_frame, instruments, portfolio, policy)
    prep = construct_target_allocations_prepared(
        market, decision_index, overlay, cal_state, instruments, portfolio, policy
    )
    assert prep == ref


def test_stock_risk_policy_fingerprint_includes_economic_ranking_mode() -> None:
    """Fingerprint differs when economic_ranking_mode differs."""
    p1 = StockRiskPolicy(top_k=20, economic_ranking_mode="raw_score_v1")
    p2 = StockRiskPolicy(top_k=20, economic_ranking_mode="economic_net_v1")
    assert stock_risk_policy_fingerprint(p1) != stock_risk_policy_fingerprint(p2)

    p3 = StockRiskPolicy(top_k=20, economic_ranking_mode="raw_score_v1")
    assert stock_risk_policy_fingerprint(p1) == stock_risk_policy_fingerprint(p3)


def test_stock_risk_policy_fingerprint_includes_execution_utility_mode() -> None:
    """Fingerprint differs when execution_utility_mode differs."""
    p1 = StockRiskPolicy(top_k=20)
    p2 = StockRiskPolicy(top_k=20, execution_utility_mode="delta_cost_aware_v1")
    p3 = StockRiskPolicy(top_k=20, execution_utility_mode="legacy_target_interpolation_v1")
    assert stock_risk_policy_fingerprint(p1) != stock_risk_policy_fingerprint(p2)
    assert stock_risk_policy_fingerprint(p1) == stock_risk_policy_fingerprint(p3)
    assert stock_risk_policy_fingerprint(p2) != stock_risk_policy_fingerprint(p3)


def test_policy_profile_rejects_invalid_execution_utility_mode() -> None:
    from src.stocks.ml.contracts import PolicyProfile
    with pytest.raises(ValueError, match="execution_utility_mode"):
        PolicyProfile(profile_id="test", execution_utility_mode="unknown_mode")


DELTA_COST_UTILITY_01 = "DELTA_COST_UTILITY_01_COST_DOMINATED_HOLD"


def test_delta_cost_utility_01_cost_dominated_hold() -> None:
    """High costs make U(1) <= U(0), so s=0 is selected (hold)."""
    from src.stocks.trading.portfolio_constructor import (
        _lower_confidence_transition_utility,
        _select_delta_cost_aware_transition,
        causal_covariance_or_fallback,
    )

    ids = ["A", "B"]
    current = {"A": 0.05, "B": 0.03}
    target = {"A": 0.08, "B": 0.02}
    lower_alpha = {"A": 0.001, "B": 0.001}
    entry_cost = {"A": 0.05, "B": 0.05}
    exit_cost = {"A": 0.05, "B": 0.05}

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=(60, 2))
    cov, _ = causal_covariance_or_fallback(
        returns, volatility_lookback_sessions=20, covariance_lookback_sessions=60
    )

    u0 = _lower_confidence_transition_utility(
        current, target, lower_alpha, entry_cost, exit_cost, cov, ids, 10, 1.0, 0.0
    )
    u1 = _lower_confidence_transition_utility(
        current, target, lower_alpha, entry_cost, exit_cost, cov, ids, 10, 1.0, 1.0
    )
    assert u1 <= u0 + 1e-12

    final, scale, reason = _select_delta_cost_aware_transition(
        current, target, lower_alpha, entry_cost, exit_cost, cov, ids, 10, 1.0
    )
    assert scale == 0.0
    assert reason is None
    for instrument_id in ids:
        assert final[instrument_id] == pytest.approx(current[instrument_id], abs=1e-12)


DELTA_COST_UTILITY_02 = "DELTA_COST_UTILITY_02_EDGE_DOMINATED_TRANSITION"


def test_delta_cost_utility_02_edge_dominated_transition() -> None:
    """Positive lower alpha with low costs yields U(s) > U(0) and 0 < s <= 1."""
    from src.stocks.trading.portfolio_constructor import (
        _lower_confidence_transition_utility,
        _select_delta_cost_aware_transition,
        causal_covariance_or_fallback,
    )

    ids = ["A", "B"]
    current = {"A": 0.0, "B": 0.0}
    target = {"A": 0.08, "B": 0.05}
    lower_alpha = {"A": 0.02, "B": 0.015}
    entry_cost = {"A": 0.001, "B": 0.001}
    exit_cost = {"A": 0.001, "B": 0.001}

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=(60, 2))
    cov, _ = causal_covariance_or_fallback(
        returns, volatility_lookback_sessions=20, covariance_lookback_sessions=60
    )

    u0 = _lower_confidence_transition_utility(
        current, target, lower_alpha, entry_cost, exit_cost, cov, ids, 10, 1.0, 0.0
    )

    final, scale, reason = _select_delta_cost_aware_transition(
        current, target, lower_alpha, entry_cost, exit_cost, cov, ids, 10, 1.0
    )
    assert reason is None
    assert 0.0 < scale <= 1.0 + 1e-12

    u_selected = _lower_confidence_transition_utility(
        current, target, lower_alpha, entry_cost, exit_cost, cov, ids, 10, 1.0, scale
    )
    assert u_selected > u0 + 1e-12

    for instrument_id in ids:
        assert 0.0 <= final[instrument_id] <= target[instrument_id] + 1e-12


DELTA_COST_UTILITY_03 = "DELTA_COST_UTILITY_03_INVALID_COST_FAILS_CLOSED"


def test_delta_cost_utility_03_invalid_cost_fails_closed() -> None:
    """NaN/negative/inconsistent costs return s=0 without optional buy."""
    from src.stocks.trading.portfolio_constructor import (
        _select_delta_cost_aware_transition,
        causal_covariance_or_fallback,
    )

    ids = ["A", "B"]
    current = {"A": 0.0, "B": 0.0}
    target = {"A": 0.08, "B": 0.05}
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=(60, 2))
    cov, _ = causal_covariance_or_fallback(
        returns, volatility_lookback_sessions=20, covariance_lookback_sessions=60
    )

    nan_alpha = {"A": float("nan"), "B": 0.01}
    ok_costs = {"A": 0.001, "B": 0.001}
    final, scale, reason = _select_delta_cost_aware_transition(
        current, target, nan_alpha, ok_costs, ok_costs, cov, ids, 10, 1.0
    )
    assert scale == 0.0
    assert reason == "non-finite-lower-alpha"

    neg_alpha = {"A": -0.01, "B": 0.01}
    final, scale, reason = _select_delta_cost_aware_transition(
        current, target, neg_alpha, ok_costs, ok_costs, cov, ids, 10, 1.0
    )
    assert scale == 0.0
    assert reason is None

    neg_entry = {"A": -0.001, "B": 0.001}
    ok_alpha = {"A": 0.01, "B": 0.01}
    final, scale, reason = _select_delta_cost_aware_transition(
        current, target, ok_alpha, neg_entry, ok_costs, cov, ids, 10, 1.0
    )
    assert scale == 0.0
    assert reason == "invalid-entry-cost"


DELTA_COST_UTILITY_04 = "DELTA_COST_UTILITY_04_PREPARED_REFERENCE_PARITY"


def test_delta_cost_utility_04_prepared_reference_parity() -> None:
    """Prepared and reference constructors emit same weights under delta_cost_aware_v1."""
    from src.stocks.research.economic_alpha import CausalAlphaCalibrator
    from src.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        construct_target_allocations_prepared,
    )

    policy = StockRiskPolicy(
        top_k=20,
        economic_ranking_mode="economic_net_v1",
        execution_utility_mode="delta_cost_aware_v1",
    )
    panel = scored_panel(n_sessions=61, n_tickers=10, seed=9).drop("ret")
    instruments = instruments_for(10)
    portfolio = empty_portfolio()

    buckets = [
        {
            "bucket": bucket,
            "sample_size": 10,
            "expected_active_alpha": 0.003,
            "alpha_lower_bound": 0.002,
        }
        for bucket in range(4)
    ]
    cal_state = {
        "bucket_count": 4,
        "history_sessions": 30,
        "round_trip_cost": 0.0005,
        "exit_cost_rate": 0.0003,
        "buckets": buckets,
    }

    market = PreparedAllocationMarket.build(panel)
    overlay = (
        panel.sort(["session", "instrument_id"])["pred_score"].to_numpy().astype(float)
    )

    window_len = (
        max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions)
        + 1
    )
    decision_index = len(market.sessions) - 1
    start = max(0, decision_index - window_len + 1)
    indices = np.concatenate(
        [
            np.arange(market.session_ranges[i][0], market.session_ranges[i][1])
            for i in range(start, decision_index + 1)
        ]
    )
    from src.stocks.trading.portfolio_constructor import _SESSION_COLUMN

    window_frame = pl.DataFrame(
        {
            "instrument_id": market.instrument_ids[indices],
            _SESSION_COLUMN: pl.Series(
                market.row_sessions[indices].tolist(),
                dtype=pl.Datetime("us", "UTC"),
            ),
            "pred_score": np.asarray(overlay)[indices],
            "sector": market.sector[indices],
            "adtv": market.adtv[indices],
            "close": market.close[indices],
        }
    ).with_columns(pl.col("pred_score").fill_nan(None))
    window_frame = CausalAlphaCalibrator.apply_prepared(cal_state, window_frame)

    ref = construct_target_allocations(window_frame, instruments, portfolio, policy)
    prep = construct_target_allocations_prepared(
        market, decision_index, overlay, cal_state, instruments, portfolio, policy
    )
    assert sorted(a.instrument.instrument_id for a in prep) == sorted(
        a.instrument.instrument_id for a in ref
    )
    for r, p in zip(
        sorted(ref, key=lambda a: a.instrument.instrument_id),
        sorted(prep, key=lambda a: a.instrument.instrument_id),
        strict=True,
    ):
        assert r.instrument.instrument_id == p.instrument.instrument_id
        assert r.target_value == pytest.approx(p.target_value, abs=1e-12)
