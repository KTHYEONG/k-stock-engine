"""Cost-model contract tests."""
from __future__ import annotations

from datetime import UTC, datetime
from math import inf

import pytest

from src.core.costs import (
    CostModel,
    CostPoint,
    CostSchedule,
    FillCostBreakdown,
    LiquiditySlippageModel,
    TickSizeRule,
    TickSizeSchedule,
)


def tick_schedule() -> TickSizeSchedule:
    return TickSizeSchedule(
        rules=(
            TickSizeRule("r0", datetime(2020, 1, 1, tzinfo=UTC), 0.0, 1000.0, 1.0),
            TickSizeRule("r1", datetime(2020, 1, 1, tzinfo=UTC), 1000.0, 5000.0, 5.0),
            TickSizeRule("r2", datetime(2020, 1, 1, tzinfo=UTC), 5000.0, 10000.0, 10.0),
            TickSizeRule("r3", datetime(2020, 1, 1, tzinfo=UTC), 10000.0, 50000.0, 50.0),
            TickSizeRule("r4", datetime(2020, 1, 1, tzinfo=UTC), 50000.0, 100000.0, 100.0),
            TickSizeRule("r5", datetime(2020, 1, 1, tzinfo=UTC), 100000.0, 500000.0, 500.0),
            TickSizeRule("r6", datetime(2020, 1, 1, tzinfo=UTC), 500000.0, inf, 1000.0),
        )
    )


class TestCostModel:
    def test_round_trip_cost_models_commission_tax_slippage(self) -> None:
        model = CostModel(commission_rate=0.00015, tax_rate=0.0023, slippage_bps=5.0)
        cost = model.round_trip_cost(1_000_000.0)
        expected = 1_000_000.0 * (0.00015 * 2 + 0.0023 + 5.0 / 10_000)
        assert cost == pytest.approx(expected)

    def test_zero_notional_has_zero_cost(self) -> None:
        assert CostModel().round_trip_cost(0.0) == 0.0

    def test_negative_rates_rejected(self) -> None:
        with pytest.raises(ValueError, match="commission_rate"):
            CostModel(commission_rate=-0.1)
        with pytest.raises(ValueError, match="tax_rate"):
            CostModel(tax_rate=-0.1)
        with pytest.raises(ValueError, match="slippage_bps"):
            CostModel(slippage_bps=-1.0)


class TestCostSchedule:
    def _schedule(self) -> CostSchedule:
        return CostSchedule(
            name="test",
            points=(
                CostPoint(
                    effective_from=datetime(2020, 1, 1, tzinfo=UTC),
                    commission_rate=0.00015,
                    tax_rate=0.0023,
                    slippage_bps=5.0,
                    settlement_days=2,
                ),
                CostPoint(
                    effective_from=datetime(2024, 1, 1, tzinfo=UTC),
                    commission_rate=0.0001,
                    tax_rate=0.0018,
                    slippage_bps=3.0,
                    settlement_days=2,
                ),
            ),
        )

    def test_cost_for_resolves_by_effective_time(self) -> None:
        schedule = self._schedule()
        early = schedule.cost_for(datetime(2022, 6, 1, tzinfo=UTC))
        late = schedule.cost_for(datetime(2025, 6, 1, tzinfo=UTC))
        assert early.commission_rate == pytest.approx(0.00015)
        assert late.commission_rate == pytest.approx(0.0001)
        assert late.commission_rate >= 0.0

    def test_lookup_before_coverage_fails_closed(self) -> None:
        schedule = self._schedule()
        with pytest.raises(ValueError, match="coverage"):
            schedule.cost_for(datetime(2010, 1, 1, tzinfo=UTC))

    def test_overlapping_points_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            CostSchedule(
                name="bad",
                points=(
                    CostPoint(effective_from=datetime(2020, 1, 1, tzinfo=UTC), commission_rate=0.0, tax_rate=0.0, slippage_bps=0.0),
                    CostPoint(effective_from=datetime(2020, 1, 1, tzinfo=UTC), commission_rate=0.0, tax_rate=0.0, slippage_bps=0.0),
                ),
            )

    def test_unsorted_points_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            CostSchedule(
                name="bad",
                points=(
                    CostPoint(effective_from=datetime(2024, 1, 1, tzinfo=UTC), commission_rate=0.0, tax_rate=0.0, slippage_bps=0.0),
                    CostPoint(effective_from=datetime(2020, 1, 1, tzinfo=UTC), commission_rate=0.0, tax_rate=0.0, slippage_bps=0.0),
                ),
            )

    def test_empty_schedule_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            CostSchedule(name="empty", points=())

    def test_negative_rate_point_rejected(self) -> None:
        with pytest.raises(ValueError, match="commission_rate"):
            CostPoint(
                effective_from=datetime(2020, 1, 1, tzinfo=UTC),
                commission_rate=-0.1,
                tax_rate=0.0,
                slippage_bps=0.0,
            )


class TestTickSizeSchedule:
    def test_resolves_tick_by_price_band(self) -> None:
        when = datetime(2024, 1, 1, tzinfo=UTC)
        schedule = tick_schedule()
        assert schedule.tick_size(990.0, when) == 1.0
        assert schedule.tick_size(1000.0, when) == 5.0
        assert schedule.tick_size(10000.0, when) == 50.0
        assert schedule.tick_size(500000.0, when) == 1000.0

    def test_band_gap_rejected(self) -> None:
        with pytest.raises(ValueError, match="gap or overlap"):
            TickSizeSchedule(
                rules=(
                    TickSizeRule("a", datetime(2020, 1, 1, tzinfo=UTC), 0.0, 1000.0, 1.0),
                    TickSizeRule("b", datetime(2020, 1, 1, tzinfo=UTC), 1500.0, inf, 5.0),
                )
            )

    def test_nonzero_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="start at zero"):
            TickSizeSchedule(
                rules=(
                    TickSizeRule("a", datetime(2020, 1, 1, tzinfo=UTC), 100.0, 1000.0, 1.0),
                    TickSizeRule("b", datetime(2020, 1, 1, tzinfo=UTC), 1000.0, inf, 5.0),
                )
            )

    def test_uncovered_tail_rejected(self) -> None:
        with pytest.raises(ValueError, match="cover all prices"):
            TickSizeSchedule(
                rules=(
                    TickSizeRule("a", datetime(2020, 1, 1, tzinfo=UTC), 0.0, 1000.0, 1.0),
                )
            )

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            tick_schedule().rule_for(-1.0, datetime(2024, 1, 1, tzinfo=UTC))


class TestLiquiditySlippageModel:
    def _model(self) -> LiquiditySlippageModel:
        return LiquiditySlippageModel(
            impact_coefficient=0.1,
            tick_schedule=tick_schedule(),
            stress_multiplier=1.0,
        )

    def test_slippage_is_half_spread_plus_sqrt_impact(self) -> None:
        model = self._model()
        when = datetime(2024, 1, 1, tzinfo=UTC)
        half_spread = model.half_spread_bps(10_000.0, when)
        expected = half_spread + 0.1 * 0.02 * 10_000 * (5_000_000.0 / 1e9) ** 0.5
        assert model.slippage_bps(
            notional=5_000_000.0,
            adtv_20d=1e9,
            daily_volatility=0.02,
            reference_price=10_000.0,
            effective_time=when,
        ) == pytest.approx(expected)

    def test_slippage_is_monotonic_in_notional(self) -> None:
        model = self._model()
        when = datetime(2024, 1, 1, tzinfo=UTC)
        small = model.slippage_bps(
            notional=1e6, adtv_20d=1e9, daily_volatility=0.02,
            reference_price=10_000.0, effective_time=when,
        )
        large = model.slippage_bps(
            notional=1e8, adtv_20d=1e9, daily_volatility=0.02,
            reference_price=10_000.0, effective_time=when,
        )
        assert large > small

    def test_slippage_is_anti_monotonic_in_adtv(self) -> None:
        model = self._model()
        when = datetime(2024, 1, 1, tzinfo=UTC)
        illiquid = model.slippage_bps(
            notional=1e6, adtv_20d=1e8, daily_volatility=0.02,
            reference_price=10_000.0, effective_time=when,
        )
        liquid = model.slippage_bps(
            notional=1e6, adtv_20d=1e10, daily_volatility=0.02,
            reference_price=10_000.0, effective_time=when,
        )
        assert illiquid > liquid

    def test_stress_uses_explicit_multiplier(self) -> None:
        base = self._model()
        stress = LiquiditySlippageModel(
            impact_coefficient=0.1,
            tick_schedule=tick_schedule(),
            stress_multiplier=1.5,
        )
        when = datetime(2024, 1, 1, tzinfo=UTC)
        kwargs = {
            "notional": 1e6,
            "adtv_20d": 1e9,
            "daily_volatility": 0.02,
            "reference_price": 10_000.0,
            "effective_time": when,
        }
        base_slippage = base.slippage_bps(**kwargs)
        stress_slippage = stress.slippage_bps(**kwargs)
        assert stress_slippage - base_slippage == pytest.approx(
            0.1 * 0.5 * 0.02 * 10_000 * (1e6 / 1e9) ** 0.5
        )
        assert base.params_hash != stress.params_hash

    def test_non_positive_inputs_fail_closed(self) -> None:
        model = self._model()
        when = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="adtv_20d"):
            model.slippage_bps(
                notional=1e6, adtv_20d=0.0, daily_volatility=0.02,
                reference_price=10_000.0, effective_time=when,
            )
        with pytest.raises(ValueError, match="daily_volatility"):
            model.slippage_bps(
                notional=1e6, adtv_20d=1e9, daily_volatility=-0.01,
                reference_price=10_000.0, effective_time=when,
            )
        with pytest.raises(ValueError, match="reference_price"):
            model.slippage_bps(
                notional=1e6, adtv_20d=1e9, daily_volatility=0.02,
                reference_price=0.0, effective_time=when,
            )


class TestFillCostBreakdown:
    def test_total_rate_separates_buy_sell_tax(self) -> None:
        breakdown = FillCostBreakdown(
            commission_rate=0.000036396,
            securities_transaction_tax_rate=0.0003,
            rural_special_tax_rate=0.0015,
            sell_tax_rate=0.0018,
            slippage_bps=10.0,
            tick_rule_id="krx_test_3",
            model_id="sqrt_impact_v1",
            params_hash="p",
        )
        buy_rate = breakdown.total_rate(side="BUY")
        sell_rate = breakdown.total_rate(side="SELL")
        assert buy_rate == pytest.approx(0.000036396 + 10.0 / 10_000)
        assert sell_rate == pytest.approx(buy_rate + 0.0018)
        assert breakdown.sell_tax_rate == pytest.approx(0.0003 + 0.0015)

    def test_tracing_record_binds_artifact_hash(self) -> None:
        breakdown = FillCostBreakdown(
            commission_rate=0.0,
            securities_transaction_tax_rate=0.0,
            rural_special_tax_rate=0.0,
            sell_tax_rate=0.0,
            slippage_bps=0.0,
            tick_rule_id="r",
            model_id="m",
            params_hash="p",
        )
        record = breakdown.to_dict(artifact_hash="abc")
        assert record["artifact_hash"] == "abc"
        assert set(record) == {
            "artifact_hash",
            "commission_rate",
            "securities_transaction_tax_rate",
            "rural_special_tax_rate",
            "sell_tax_rate",
            "slippage_bps",
            "tick_rule_id",
            "model_id",
            "params_hash",
        }

    def test_unknown_side_rejected(self) -> None:
        breakdown = FillCostBreakdown(
            commission_rate=0.0,
            securities_transaction_tax_rate=0.0,
            rural_special_tax_rate=0.0,
            sell_tax_rate=0.0,
            slippage_bps=0.0,
            tick_rule_id="r",
            model_id="m",
            params_hash="p",
        )
        with pytest.raises(ValueError, match="side"):
            breakdown.total_rate(side="HOLD")
