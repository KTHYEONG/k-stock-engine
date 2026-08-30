"""Active research data pipeline tests."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.stocks.data.active import ActiveResearchDataRequest, resolve_active_research_data
from src.stocks.data.catalog import ActiveDatasetPolicy, CatalogEntry, CatalogKind, CatalogStore, EvidenceCompleteness
from src.stocks.data.contracts import CoverageRange
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash
import polars as pl


def _make_manifest(dataset_id: str, frame: pl.DataFrame, feature_set: str) -> DatasetManifest:
    from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=list(frame.columns),
        feature_set=feature_set,
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 31, tzinfo=UTC),
        provider_version="test",
        universe_policy_version="test",
        row_count=frame.height,
        generated_time=datetime.now(UTC),
        schema_version="v2",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    from dataclasses import replace
    return replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))


def _write_dataset(root: Path, dataset_id: str, frame: pl.DataFrame, feature_set: str) -> DatasetManifest:
    store = ParquetDatasetStore(root)
    manifest = _make_manifest(dataset_id, frame, feature_set)
    store.write_partitioned(frame, dataset_id=dataset_id, manifest=manifest, expected_feature_set=feature_set, decision_time=datetime(2024, 3, 31, tzinfo=UTC))
    return manifest


def _setup_catalog(tmp_path: Path):
    catalog_root = tmp_path / "catalog"
    base_root = tmp_path / "base"
    feature_root = tmp_path / "feature"
    label_root = tmp_path / "label"
    for p in (base_root, feature_root, label_root):
        p.mkdir(parents=True, exist_ok=True)
    # cost evidence file
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(json.dumps({"cost": 1}), encoding="utf-8")
    cost_hash = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    # build frames
    sessions = [datetime(2024, 1, 10, tzinfo=UTC), datetime(2024, 2, 10, tzinfo=UTC)]
    base_frame = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00002"], "session": sessions, "open": [100.0, 101.0], "close": [101.0, 102.0], "volume": [1e6, 1e6], "trading_value": [1e8, 1e8]})
    feat_frame = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00002"], "session": sessions, "feature__x": [0.1, 0.2]})
    label_frame = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00002"], "session": sessions, "horizon_sessions": [10, 10], "net_alpha_target": [0.01, 0.02], "label_available_time": sessions})
    base_manifest = _write_dataset(base_root, "base_v1", base_frame, "base_panel")
    feat_manifest = _write_dataset(feature_root, "feat_v1", feat_frame, "stock_net_alpha_v1")
    label_manifest = _write_dataset(label_root, "label_v1", label_frame, "labels")
    store = CatalogStore(catalog_root)
    rng = CoverageRange(start=date(2024, 1, 1), end=date(2024, 3, 31))
    for kind, name, manifest in [
        (CatalogKind.BASE_PANEL, "base_v1", base_manifest),
        (CatalogKind.FEATURES, "feat_v1", feat_manifest),
        (CatalogKind.LABELS, "label_v1", label_manifest),
    ]:
        entry = CatalogEntry(kind=kind, name=name, content_hash=manifest.content_hash, schema_hash=manifest.schema_hash, registered_at=datetime(2024, 1, 1, tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str((base_root if kind == CatalogKind.BASE_PANEL else feature_root if kind == CatalogKind.FEATURES else label_root) / name))
        store.register(entry)
    costs_entry = CatalogEntry(kind=CatalogKind.COSTS, name="costs_v1", content_hash=cost_hash, schema_hash=hashlib.sha256(cost_path.read_bytes()).hexdigest(), registered_at=datetime(2024, 1, 1, tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(cost_path))
    store.register(costs_entry)
    policy = ActiveDatasetPolicy(entries=((CatalogKind.BASE_PANEL, "base_v1"), (CatalogKind.FEATURES, "feat_v1"), (CatalogKind.LABELS, "label_v1"), (CatalogKind.COSTS, "costs_v1")))
    store.save_active_policy(policy)
    return catalog_root, base_root, feature_root, label_root, cost_path


def test_active_selection_resolves_catalogued_datasets_without_snapshot_manifest(tmp_path) -> None:
    catalog_root, base_root, feature_root, label_root, cost_path = _setup_catalog(tmp_path)
    request = ActiveResearchDataRequest(start=date(2024, 1, 15), end=date(2024, 2, 15), candidate_horizon_sessions=(10,))
    selection = resolve_active_research_data(catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, request=request)
    assert selection.direct_request.base_dataset_id == "base_v1"
    assert selection.direct_request.feature_dataset_id == "feat_v1"
    assert selection.direct_request.label_dataset_id == "label_v1"
    assert selection.data_inputs["base_content_hash"] != ""
    assert (catalog_root / "snapshots").exists() is False or not any((catalog_root / "snapshots").iterdir())


def test_active_selection_rejects_missing_or_hash_mismatched_operational_entry_before_load(tmp_path) -> None:
    catalog_root, base_root, feature_root, label_root, _ = _setup_catalog(tmp_path)
    # corrupt label manifest hash mismatch: change catalog entry hash
    store = CatalogStore(catalog_root)
    # create mismatched entry by directly editing catalog.jsonl
    log_path = store.log_path
    text = log_path.read_text(encoding="utf-8")
    text = text.replace(store.get(CatalogKind.LABELS, "label_v1").content_hash, "0"*64)
    log_path.write_text(text, encoding="utf-8")
    request = ActiveResearchDataRequest(start=date(2024, 1, 15), end=date(2024, 2, 15), candidate_horizon_sessions=(10,))
    with pytest.raises(ValueError, match="labels"):
        resolve_active_research_data(catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, request=request)


def test_active_selection_rejects_out_of_range_or_costless_policy(tmp_path) -> None:
    catalog_root, base_root, feature_root, label_root, _ = _setup_catalog(tmp_path)
    store = CatalogStore(catalog_root)
    # out of range request
    request = ActiveResearchDataRequest(start=date(2025, 1, 1), end=date(2025, 1, 10), candidate_horizon_sessions=(10,))
    with pytest.raises(ValueError, match="does not contain"):
        resolve_active_research_data(catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, request=request)
    # costless policy: remove costs
    policy = ActiveDatasetPolicy(entries=((CatalogKind.BASE_PANEL, "base_v1"), (CatalogKind.FEATURES, "feat_v1"), (CatalogKind.LABELS, "label_v1")))
    store.save_active_policy(policy)
    request2 = ActiveResearchDataRequest(start=date(2024, 1, 15), end=date(2024, 2, 15), candidate_horizon_sessions=(10,))
    with pytest.raises(ValueError, match="costs"):
        resolve_active_research_data(catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, request=request2)
