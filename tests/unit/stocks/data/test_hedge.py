"""Hedge overlay data hash tests."""
from __future__ import annotations

def test_load_executable_overlay_rejects_hash_mismatch(tmp_path) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    from dataclasses import replace
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.stocks.data.hedge import load_executable_overlay_data
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    sessions = [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)]
    frame = pl.DataFrame({"instrument_id": ["KRX:252670"] * 2, "session": sessions, "open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0], "close": [100.0, 101.0], "volume": [1_000_000.0] * 2, "available_time": sessions})
    manifest = make_manifest(asset_kind=AssetKind.ETF, columns=list(frame.columns), feature_set="hedge_bars_v1", label_definition="none", label_horizon_sessions=1, time_start=sessions[0], time_end=sessions[-1], provider_version="test", universe_policy_version="test", row_count=2, generated_time=sessions[-1], storage_layout=HIVE_PARTITION_LAYOUT, schema_version="v2")
    manifest = replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))
    ParquetDatasetStore(tmp_path).write_partitioned(frame, dataset_id="hedge_v1", manifest=manifest, expected_feature_set="hedge_bars_v1", decision_time=datetime(2024, 1, 4, tzinfo=UTC))

    with pytest.raises(ValueError, match="hedge-content-hash-mismatch"):
        load_executable_overlay_data(root=tmp_path, dataset_id="hedge_v1", instrument_id="KRX:252670", beta=-2.0, decision_time=datetime(2024, 1, 4, tzinfo=UTC), expected_content_hash="f" * 64, base_cost_schedule=default_base_schedule(), stress_cost_schedule=default_stress_schedule())
