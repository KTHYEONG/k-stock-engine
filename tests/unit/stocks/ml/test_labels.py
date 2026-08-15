"""Net-alpha ML label contracts: canonical columns and horizon partitioning."""
from __future__ import annotations

import pytest

from src.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    HORIZON_COLUMN,
    ID_COLUMN,
    REALIZED_RETURN_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
    TARGET_COLUMN,
    partition_labels_by_horizon,
)


def test_canonical_column_names_are_decimal_and_causal() -> None:
    assert TARGET_COLUMN == "net_alpha_target"
    assert RISK_RESIDUAL_COLUMN == "risk_residual"
    assert REFERENCE_COST_COLUMN == "reference_cost"
    assert REALIZED_RETURN_COLUMN == "realized_net_return"
    assert ID_COLUMN == "instrument_id"
    assert SESSION_COLUMN == "session"
    assert AVAILABLE_COLUMN == "label_available_time"


def test_partition_labels_by_horizon_preserves_independent_universes() -> None:
    from datetime import UTC, datetime

    import polars as pl

    session = datetime(2024, 1, 2, tzinfo=UTC)
    labels = pl.DataFrame(
        {
            ID_COLUMN: ["KRX:00001", "KRX:00002"] * 2,
            SESSION_COLUMN: [session] * 4,
            HORIZON_COLUMN: [3, 3, 5, 5],
            TARGET_COLUMN: [0.1, 0.2, 0.3, 0.4],
            AVAILABLE_COLUMN: [session] * 4,
            RISK_RESIDUAL_COLUMN: [0.01, 0.02, 0.03, 0.04],
            REFERENCE_COST_COLUMN: [0.001] * 4,
        }
    )
    partitioned = partition_labels_by_horizon(labels, (3, 5))
    assert set(partitioned) == {3, 5}
    assert partitioned[3].height == 2
    assert partitioned[5].height == 2
    assert HORIZON_COLUMN not in partitioned[3].columns


def test_partition_rejects_missing_horizon() -> None:
    from datetime import UTC, datetime

    import polars as pl

    session = datetime(2024, 1, 2, tzinfo=UTC)
    labels = pl.DataFrame(
        {
            ID_COLUMN: ["KRX:00001"],
            SESSION_COLUMN: [session],
            HORIZON_COLUMN: [3],
            TARGET_COLUMN: [0.1],
            AVAILABLE_COLUMN: [session],
            RISK_RESIDUAL_COLUMN: [0.01],
            REFERENCE_COST_COLUMN: [0.001],
        }
    )
    with pytest.raises(ValueError, match="no rows for horizons"):
        partition_labels_by_horizon(labels, (3, 5))
