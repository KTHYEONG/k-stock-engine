"""Cost-model contract tests."""
from __future__ import annotations

import pytest

from src.core.costs import CostModel


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
