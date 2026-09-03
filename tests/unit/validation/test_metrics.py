from __future__ import annotations


def test_calculate_ledger_metrics_uses_hand_computed_log_growth_and_drawdown() -> None:
    from datetime import UTC, datetime, timedelta
    from math import expm1, log

    import pytest

    from src.core.ledger import LedgerNav
    from src.validation.metrics import calculate_ledger_metrics

    start = datetime(2024, 1, 2, tzinfo=UTC)
    nav = tuple(
        LedgerNav(f"mark-{index}", start + timedelta(days=index), value, value, 0.0, 0.0)
        for index, value in enumerate((100.0, 110.0, 99.0, 108.9))
    )

    metrics = calculate_ledger_metrics(nav, sessions_per_year=3)

    expected_log_growth = log(1.1) + log(0.9) + log(1.1)
    assert metrics.annualized_log_growth == pytest.approx(expected_log_growth)
    assert metrics.cagr == pytest.approx(expm1(expected_log_growth))
    assert metrics.max_drawdown == pytest.approx(0.1)
