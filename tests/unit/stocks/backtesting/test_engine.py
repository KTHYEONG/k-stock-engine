"""Backtest result provenance contract tests."""
from __future__ import annotations

from src.stocks.backtesting.engine import BacktestResult


def test_backtest_result_preserves_data_quality_evidence() -> None:
    evidence = {
        "dataset_content_hash": "dataset-hash",
        "quality_report_hash": "quality-hash",
        "master_hash": "master-hash",
        "calendar_hash": "calendar-hash",
        "action_hash": "action-hash",
        "cost_hash": "cost-hash",
    }
    result = BacktestResult(
        ledger=(),
        trades=(),
        final_value=100.0,
        total_return=0.0,
        metrics={},
        data_quality=evidence,
    )

    assert result.data_quality == evidence
