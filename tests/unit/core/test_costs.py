"""Cost-model contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.costs import CostModel, CostPoint, CostSchedule


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
