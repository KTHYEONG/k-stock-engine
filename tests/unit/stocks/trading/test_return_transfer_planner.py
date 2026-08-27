"""Contract tests for sparse return-transfer transitions."""

from __future__ import annotations

from src.stocks.ml.return_transfer import ReturnDistributionForecast, TransitionCost
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.trading.return_transfer_planner import plan_return_transfer_transition


def test_RETURN_TRANSFER_05_SPARSE_HOLD_REPLACE() -> None:
    """RETURN_TRANSFER_05_SPARSE_HOLD_REPLACE: cost-neutral hold, gain replace."""
    policy = StockRiskPolicy(
        top_k=2, single_name_cap=0.5, sector_cap=0.5, gross_cap=0.9
    )
    forecasts = {
        "A": ReturnDistributionForecast(mu=0.01, q20=-0.01, residual_rank=0.5),
        "B": ReturnDistributionForecast(mu=0.02, q20=-0.01, residual_rank=0.6),
    }
    costs = {
        "A": TransitionCost(enter=0.005, exit=0.005, hold=0.001),
        "B": TransitionCost(enter=0.005, exit=0.005, hold=0.001),
    }
    replaced = plan_return_transfer_transition({"A": 0.4}, forecasts, costs, policy)
    assert "B" in replaced
    assert "A" not in replaced
    assert sum(replaced.values()) <= policy.gross_cap

    # B's predicted benefit does not cover its entry plus A's exit cost.
    weak = dict(forecasts)
    weak["B"] = ReturnDistributionForecast(mu=0.014, q20=-0.02, residual_rank=0.6)
    held = plan_return_transfer_transition({"A": 0.4}, weak, costs, policy)
    assert "A" in held
    assert all(weight <= policy.single_name_cap for weight in held.values())
