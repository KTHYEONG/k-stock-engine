from __future__ import annotations

from src.stocks.backtesting.metrics import build_backtest_attribution


def test_empty_attribution_is_zero() -> None:
    attribution = build_backtest_attribution([], [])

    assert attribution.base_total == 0.0
