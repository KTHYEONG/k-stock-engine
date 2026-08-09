"""Point-in-time stock universe policy tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from src.stocks.domain.universe import PointInTimeUniverse, UniversePolicy


def test_universe_members_are_sorted_and_available_time_scoped() -> None:
    policy = UniversePolicy(version="v1", min_close=0.0, require_operating_income=False)
    decision = datetime(2024, 1, 6, 8, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "code": ["000002", "000001"],
            "available_time": [decision] * 2,
            "close": [10.0, 20.0],
        }
    )
    result = PointInTimeUniverse(policy).apply(frame, decision)
    assert result.members == ("000001", "000002")
