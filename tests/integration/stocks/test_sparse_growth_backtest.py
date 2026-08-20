"""Sparse-growth backtest contract smoke coverage."""

from datetime import UTC, datetime, timedelta

from src.stocks.trading.rebalance_schedule import rebalance_session_indices


def test_sparse_growth_v5_end_to_end_cadence_contract() -> None:
    """SPARSE_GROWTH_V5_09_END_TO_END_PARITY."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = tuple(start + timedelta(days=i) for i in range(8))
    assert rebalance_session_indices(
        sessions, sessions[0], sessions[-1], 3, legacy_daily=False
    ) == (0, 3, 6)
