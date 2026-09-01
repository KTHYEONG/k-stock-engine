"""Net-alpha ML label contracts: canonical columns and horizon partitioning."""
from __future__ import annotations

import pytest

from legacy.stocks.ml.labels import (
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


def test_build_partitioned_labels_with_status_emits_sidecar() -> None:
    import polars as pl
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule
    from legacy.stocks.data.quality import KRXSessionCalendar
    from legacy.stocks.ml.contracts import OUTCOME_STATUS_COLUMN
    from legacy.stocks.ml.labels import (
        HORIZON_COLUMN,
        ID_COLUMN,
        SESSION_COLUMN,
        build_partitioned_net_alpha_labels_with_status,
    )
    from tests.fixtures.stocks.helpers import stock_liquidity_model

    start = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = tuple(start + timedelta(days=i) for i in range(40))
    calendar = KRXSessionCalendar(
        version="fixture",
        sessions=tuple(s.date() for s in sessions),
        generated_time=start,
    )
    rows: list[dict] = []
    for t in range(20):
        rows.extend(
            {
                ID_COLUMN: f"KRX:{t + 1:06d}",
                SESSION_COLUMN: sessions[s],
                "open": 100.0,
                "sector": f"S{t % 4}",
                "adtv": 1.0e8,
                "market_cap": 1.0e11,
                "beta": 1.0,
                "volatility": 0.02,
            }
            for s in range(40)
        )
    base = pl.DataFrame(rows)
    cost_schedule = CostSchedule(
        name="fixture",
        points=(
            CostPoint(
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                commission_rate=0.00015,
                tax_rate=0.0023,
                slippage_bps=5.0,
            ),
        ),
    )
    labels, status = build_partitioned_net_alpha_labels_with_status(
        base,
        calendar,
        cost_schedule,
        stock_liquidity_model(),
        horizon_sessions=(3, 5),
        reference_notional=1.0e6,
    )
    assert OUTCOME_STATUS_COLUMN in status.columns
    assert set(status[HORIZON_COLUMN].unique().to_list()) == {3, 5}
    assert status.filter(pl.col(OUTCOME_STATUS_COLUMN).is_null()).height == 0
    # The sidecar covers every decision key x horizon.
    assert status.height == 2 * base.height


def test_publish_outcome_status_sidecar_binds_schema_and_content_hash(tmp_path) -> None:
    import polars as pl
    from datetime import UTC, datetime, timedelta

    from legacy.stocks.ml.labels import (
        HORIZON_COLUMN,
        ID_COLUMN,
        OUTCOME_STATUS_DATASET_SUFFIX,
        SESSION_COLUMN,
        publish_outcome_status_sidecar,
    )
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    start = datetime(2024, 1, 1, tzinfo=UTC)
    status = pl.DataFrame(
        {
            ID_COLUMN: ["KRX:00001"] * 2 + ["KRX:00002"] * 2,
            SESSION_COLUMN: [start, start + timedelta(days=1)] * 2,
            HORIZON_COLUMN: [3, 3, 5, 5],
            "outcome_status": ["REALIZED", "REALIZED", "PARTIAL_TAIL", "REALIZED"],
        }
    ).sort([ID_COLUMN, SESSION_COLUMN, HORIZON_COLUMN])
    result = publish_outcome_status_sidecar(
        status,
        destination_root=tmp_path / "labels",
        dataset_id="na_labels_outcome_status",
        base_panel_hash="base-hash",
        calendar_hash="cal-hash",
        horizon_sessions=(3, 5),
        generated_time=datetime(2024, 2, 1, tzinfo=UTC),
    )
    store = ParquetDatasetStore(tmp_path / "labels")
    manifest = store.read_manifest(result.dataset_id)
    assert manifest.content_hash == canonical_content_hash(
        status, status.columns
    )
    assert manifest.schema_hash
    assert manifest.label_horizon_sessions == 3
    from src.core.instruments import AssetKind

    reread = store.read(
        result.dataset_id,
        AssetKind.STOCK,
        "labels",
        datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert reread.select(status.columns).sort(
        [ID_COLUMN, SESSION_COLUMN, HORIZON_COLUMN]
    ).equals(status)
    assert result.dataset_id.endswith(OUTCOME_STATUS_DATASET_SUFFIX)
    # Snapshot-pinned: the declared content hash binds the exact spine and the
    # sidecar emits exactly one row per (instrument_id, session, horizon).
    duplicate_count = reread.group_by(
        [ID_COLUMN, SESSION_COLUMN, HORIZON_COLUMN]
    ).len().filter(pl.col("len") > 1).height
    assert duplicate_count == 0
    assert reread.height == status.height


def test_publish_outcome_status_sidecar_rejects_duplicate_identity_keys(tmp_path) -> None:
    import polars as pl
    from datetime import UTC, datetime

    from legacy.stocks.ml.labels import (
        HORIZON_COLUMN,
        ID_COLUMN,
        SESSION_COLUMN,
        publish_outcome_status_sidecar,
    )

    start = datetime(2024, 1, 1, tzinfo=UTC)
    duplicate_status = pl.DataFrame(
        {
            ID_COLUMN: ["KRX:00001", "KRX:00001"],
            SESSION_COLUMN: [start, start],
            HORIZON_COLUMN: [3, 3],
            "outcome_status": ["REALIZED", "PARTIAL_TAIL"],
        }
    )
    with pytest.raises(ValueError, match="exactly one row"):
        publish_outcome_status_sidecar(
            duplicate_status,
            destination_root=tmp_path / "labels",
            dataset_id="na_labels_outcome_status",
            base_panel_hash="base-hash",
            calendar_hash="cal-hash",
            horizon_sessions=(3,),
            generated_time=datetime(2024, 2, 1, tzinfo=UTC),
        )
