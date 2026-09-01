from __future__ import annotations


def test_portfolio_allocation_stock_only_history_mode() -> None:
    from legacy.stocks.trading.portfolio_allocation import net_exposure_gate_scale
    from legacy.stocks.trading.policy import StockRiskPolicy

    policy = StockRiskPolicy(
        net_exposure_gate_mode="trend_vol_v1",
        gate_floor=0.0,
        gate_history_mode="cash_on_insufficient_v1",
    )
    scale, detail = net_exposure_gate_scale([0.001] * 20, policy)
    assert scale == 0.0
    assert detail == {"reason": "gate-history-insufficient-cash"}
