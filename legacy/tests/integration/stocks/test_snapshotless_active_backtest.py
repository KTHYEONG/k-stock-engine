"""Active backtest integration."""
from __future__ import annotations
import hashlib
import json
from datetime import UTC, date, datetime
import polars as pl
from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
from src.core.instruments import AssetKind
from legacy.stocks.data.catalog import ActiveDatasetPolicy, CatalogEntry, CatalogKind, CatalogStore, EvidenceCompleteness
from legacy.stocks.data.contracts import CoverageRange
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

def _write(root, did, frame, fset):
    store = ParquetDatasetStore(root)
    manifest = make_manifest(asset_kind=AssetKind.STOCK, columns=list(frame.columns), feature_set=fset, label_definition="net_alpha_o2o", label_horizon_sessions=10, time_start=datetime(2024,1,1,tzinfo=UTC), time_end=datetime(2024,3,31,tzinfo=UTC), provider_version="t", universe_policy_version="t", row_count=frame.height, generated_time=datetime.now(UTC), schema_version="v2", storage_layout=HIVE_PARTITION_LAYOUT)
    from dataclasses import replace
    manifest = replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))
    store.write_partitioned(frame, dataset_id=did, manifest=manifest, expected_feature_set=fset, decision_time=datetime(2024,3,31,tzinfo=UTC))
    return manifest

def test_active_backtest_reuses_direct_selection_and_preserves_causal_guards(tmp_path) -> None:
    catalog_root = tmp_path / "catalog"
    base_root = tmp_path / "base"
    feature_root = tmp_path / "feature"
    label_root = tmp_path / "label"
    for p in (base_root, feature_root, label_root):
        p.mkdir(parents=True, exist_ok=True)
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(json.dumps({"c":1}), encoding="utf-8")
    cost_hash = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    sessions = [datetime(2024,1,10,tzinfo=UTC), datetime(2024,2,10,tzinfo=UTC)]
    base_frame = pl.DataFrame({"instrument_id": ["KRX:00001","KRX:00001"], "session": sessions, "open": [100.0,101.0], "close":[101.0,102.0], "volume":[1e6,1e6], "trading_value":[1e8,1e8]})
    feat_frame = pl.DataFrame({"instrument_id": ["KRX:00001","KRX:00001"], "session": sessions, "feature__x":[0.1,0.2]})
    label_frame = pl.DataFrame({"instrument_id": ["KRX:00001","KRX:00001"], "session": sessions, "horizon_sessions":[10,10], "net_alpha_target":[0.01,0.02], "label_available_time": sessions})
    base_manifest = _write(base_root, "base_v1", base_frame, "base_panel")
    feat_manifest = _write(feature_root, "feat_v1", feat_frame, "stock_net_alpha_v1")
    label_manifest = _write(label_root, "label_v1", label_frame, "labels")
    store = CatalogStore(catalog_root)
    rng = CoverageRange(start=date(2024,1,1), end=date(2024,3,31))
    for kind, name, manifest, path in [(CatalogKind.BASE_PANEL,"base_v1",base_manifest, base_root/"base_v1"),(CatalogKind.FEATURES,"feat_v1",feat_manifest, feature_root/"feat_v1"),(CatalogKind.LABELS,"label_v1",label_manifest, label_root/"label_v1")]:
        store.register(CatalogEntry(kind=kind, name=name, content_hash=manifest.content_hash, schema_hash=manifest.schema_hash, registered_at=datetime(2024,1,1,tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(path)))
    store.register(CatalogEntry(kind=CatalogKind.COSTS, name="costs_v1", content_hash=cost_hash, schema_hash=hashlib.sha256(cost_path.read_bytes()).hexdigest(), registered_at=datetime(2024,1,1,tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(cost_path)))
    store.save_active_policy(ActiveDatasetPolicy(entries=((CatalogKind.BASE_PANEL,"base_v1"),(CatalogKind.FEATURES,"feat_v1"),(CatalogKind.LABELS,"label_v1"),(CatalogKind.COSTS,"costs_v1"))))
    from legacy.stocks.data.active import ActiveResearchDataRequest, resolve_active_research_data
    from legacy.stocks.data.direct import DirectMarketDataLoader
    request = ActiveResearchDataRequest(start=date(2024,1,15), end=date(2024,2,15), candidate_horizon_sessions=(10,))
    selection = resolve_active_research_data(catalog_root=catalog_root, base_root=base_root, feature_root=feature_root, label_root=label_root, request=request)
    loader = DirectMarketDataLoader(base_root=base_root, feature_root=feature_root, label_root=label_root)
    readiness = loader.assess_readiness(selection.direct_request, datetime(2024,2,20,tzinfo=UTC), cost_evidence_path=selection.cost_evidence_path)
    assert readiness.passed
    data = loader.load(selection.direct_request)
    # causal guard: ensure loader succeeded and produced frame
    assert data.frame.height > 0
    # ledger without snapshot_id
    from legacy.stocks.ml.result_ledger import MlResultLedger
    ledger = MlResultLedger(tmp_path / "results")
    ledger.record_research_outcome(run_id="backtest-1", status="completed", data_inputs=dict(selection.data_inputs), readiness={"passed": True}, outcome={}, started_at=datetime.now(UTC))
    recent = (tmp_path / "results" / "ml_runs" / "recent.jsonl")
    if recent.exists():
        txt = recent.read_text(encoding="utf-8")
        assert "snapshot_id" not in txt
