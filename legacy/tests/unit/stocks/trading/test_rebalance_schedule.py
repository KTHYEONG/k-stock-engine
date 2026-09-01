"""SPARSE_GROWTH_V5_02_CADENCE: rebalance_session_indices cadence kernel."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from legacy.stocks.trading.rebalance_schedule import rebalance_session_indices


class TestRebalanceScheduleCadence:
    def test_exact_indices_for_13_sessions_frequency_5(self) -> None:
        """SPARSE_GROWTH_V5_02_CADENCE: 13 eligible sessions, f=5 -> (0,5,10)."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(13))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[-1],
            frequency_sessions=5,
        )
        assert result == (0, 5, 10)

    def test_frequency_1_returns_all_eligible(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(5))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[-1],
            frequency_sessions=1,
        )
        assert result == (0, 1, 2, 3, 4)

    def test_frequency_larger_than_window_returns_single(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(3))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[-1],
            frequency_sessions=10,
        )
        assert result == (0,)

    def test_empty_sessions_returns_empty(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        result = rebalance_session_indices(
            (),
            eligible_from=base,
            eligible_to=base + timedelta(days=10),
            frequency_sessions=5,
        )
        assert result == ()

    def test_no_eligible_sessions_returns_empty(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(5))
        result = rebalance_session_indices(
            sessions,
            eligible_from=base + timedelta(days=100),
            eligible_to=base + timedelta(days=200),
            frequency_sessions=5,
        )
        assert result == ()

    def test_frequency_zero_raises_value_error(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(5))
        with pytest.raises(ValueError, match="frequency_sessions must be positive"):
            rebalance_session_indices(
                sessions,
                eligible_from=sessions[0],
                eligible_to=sessions[-1],
                frequency_sessions=0,
            )

    def test_frequency_negative_raises_value_error(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(5))
        with pytest.raises(ValueError, match="frequency_sessions must be positive"):
            rebalance_session_indices(
                sessions,
                eligible_from=sessions[0],
                eligible_to=sessions[-1],
                frequency_sessions=-1,
            )

    def test_unsorted_sessions_raises_value_error(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = (base + timedelta(days=5), base + timedelta(days=1))
        with pytest.raises(ValueError, match="sorted"):
            rebalance_session_indices(
                sessions,
                eligible_from=base,
                eligible_to=base + timedelta(days=10),
                frequency_sessions=5,
            )

    def test_naive_sessions_with_aware_bounds_raises(self) -> None:
        naive_sessions = (datetime(2024, 1, 1), datetime(2024, 1, 6))
        aware_from = datetime(2024, 1, 1, tzinfo=UTC)
        aware_to = datetime(2024, 1, 10, tzinfo=UTC)
        with pytest.raises(ValueError, match="timezone"):
            rebalance_session_indices(
                naive_sessions,
                eligible_from=aware_from,
                eligible_to=aware_to,
                frequency_sessions=5,
            )

    def test_legacy_daily_returns_all_eligible_indices(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(13))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[-1],
            frequency_sessions=5,
            legacy_daily=True,
        )
        assert result == tuple(range(13))

    def test_legacy_daily_allows_zero_frequency(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(3))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[-1],
            frequency_sessions=0,
            legacy_daily=True,
        )
        assert result == (0, 1, 2)

    def test_partial_eligibility(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(20))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[3],
            eligible_to=sessions[12],
            frequency_sessions=5,
        )
        assert result == (3, 8)

    def test_exact_boundary_eligible(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        sessions = tuple(base + timedelta(days=i) for i in range(10))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[0],
            frequency_sessions=5,
        )
        assert result == (0,)

    def test_naive_sessions_with_naive_bounds(self) -> None:
        sessions = tuple(datetime(2024, 1, 1) + timedelta(days=i) for i in range(13))
        result = rebalance_session_indices(
            sessions,
            eligible_from=sessions[0],
            eligible_to=sessions[-1],
            frequency_sessions=5,
        )
        assert result == (0, 5, 10)
