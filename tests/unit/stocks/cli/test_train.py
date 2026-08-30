# ruff: noqa
"""Train CLI snapshotless tests."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.cli import train


def test_train_parser_exposes_no_snapshot_or_dataset_override_flags() -> None:
    parser = train.build_parser()
    for flag in ["--snapshot-id", "--as-of", "--base-dataset-id", "--feature-dataset-id", "--label-dataset-id", "--cost-snapshot-id"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["--artifact-id", "a1", flag, "val"])
    # research start/end remain accepted
    args = parser.parse_args(["--artifact-id", "a1", "--research-start", "2024-01-01", "--research-end", "2024-02-01"])
    assert args.research_start == date(2024, 1, 1)


def test_model_selection_uses_active_selection_once_and_records_reproducible_inputs(monkeypatch, tmp_path) -> None:
    # setup minimal catalog and datasets via helper from active test
    from src.stocks.data.active import ActiveResearchDataRequest, ActiveResearchDataSelection
    from src.stocks.data.direct import DirectDataRequest
    import hashlib, json as _json
    from src.stocks.data.catalog import ActiveDatasetPolicy, CatalogEntry, CatalogKind, CatalogStore, EvidenceCompleteness
    from src.stocks.data.contracts import CoverageRange
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash
    import polars as pl
    from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind

    catalog_root = tmp_path / "catalog"
    base_root = tmp_path / "base"
    feature_root = tmp_path / "feature"
    label_root = tmp_path / "label"
    for p in (base_root, feature_root, label_root):
        p.mkdir(parents=True, exist_ok=True)
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(_json.dumps({"c":1}), encoding="utf-8")
    cost_hash = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    sessions = [datetime(2024,1,10,tzinfo=UTC), datetime(2024,2,10,tzinfo=UTC)]
    base_frame = pl.DataFrame({"instrument_id": ["KRX:00001","KRX:00001"], "session": sessions, "open": [100.0,101.0], "close":[101.0,102.0], "volume":[1e6,1e6], "trading_value":[1e8,1e8]})
    feat_frame = pl.DataFrame({"instrument_id": ["KRX:00001","KRX:00001"], "session": sessions, "feature__x":[0.1,0.2]})
    label_frame = pl.DataFrame({"instrument_id": ["KRX:00001","KRX:00001"], "session": sessions, "horizon_sessions":[10,10], "net_alpha_target":[0.01,0.02], "label_available_time": sessions})
    def _write(root, did, frame, fset):
        store = ParquetDatasetStore(root)
        manifest = make_manifest(asset_kind=AssetKind.STOCK, columns=list(frame.columns), feature_set=fset, label_definition="net_alpha_o2o", label_horizon_sessions=10, time_start=datetime(2024,1,1,tzinfo=UTC), time_end=datetime(2024,3,31,tzinfo=UTC), provider_version="t", universe_policy_version="t", row_count=frame.height, generated_time=datetime.now(UTC), schema_version="v2", storage_layout=HIVE_PARTITION_LAYOUT)
        from dataclasses import replace
        manifest = replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))
        store.write_partitioned(frame, dataset_id=did, manifest=manifest, expected_feature_set=fset, decision_time=datetime(2024,3,31,tzinfo=UTC))
        return manifest
    base_manifest = _write(base_root, "base_v1", base_frame, "base_panel")
    feat_manifest = _write(feature_root, "feat_v1", feat_frame, "stock_net_alpha_v1")
    label_manifest = _write(label_root, "label_v1", label_frame, "labels")
    store = CatalogStore(catalog_root)
    rng = CoverageRange(start=date(2024,1,1), end=date(2024,3,31))
    for kind, name, manifest, path in [(CatalogKind.BASE_PANEL,"base_v1",base_manifest, base_root/"base_v1"),(CatalogKind.FEATURES,"feat_v1",feat_manifest, feature_root/"feat_v1"),(CatalogKind.LABELS,"label_v1",label_manifest, label_root/"label_v1")]:
        store.register(CatalogEntry(kind=kind, name=name, content_hash=manifest.content_hash, schema_hash=manifest.schema_hash, registered_at=datetime(2024,1,1,tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(path)))
    store.register(CatalogEntry(kind=CatalogKind.COSTS, name="costs_v1", content_hash=cost_hash, schema_hash=hashlib.sha256(cost_path.read_bytes()).hexdigest(), registered_at=datetime(2024,1,1,tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(cost_path)))
    store.save_active_policy(ActiveDatasetPolicy(entries=((CatalogKind.BASE_PANEL,"base_v1"),(CatalogKind.FEATURES,"feat_v1"),(CatalogKind.LABELS,"label_v1"),(CatalogKind.COSTS,"costs_v1"))))

    # mock loader to capture calls
    from src.stocks.data.direct import DirectMarketDataLoader

    orig_resolve = train.resolve_active_research_data
    call_count = {"resolve":0}
    orig_resolve_fn = orig_resolve
    def counting_resolve(**kwargs):
        call_count["resolve"]+=1
        return orig_resolve_fn(**kwargs)
    monkeypatch.setattr(train, "resolve_active_research_data", counting_resolve)

    # mock MlResultLedger to capture data_selection without snapshot_id - use simple pass-through
    class FakeLedger:
        def __init__(self, root): pass
        def record_research_outcome(self, **kwargs):
            data_inputs = kwargs.get("data_inputs", {})
            # ensure no snapshot_id
            if "snapshot_id" in data_inputs:
                raise AssertionError("snapshot_id should not be in data_inputs")
            FakeLedger.captured = data_inputs

    monkeypatch.setattr("src.stocks.cli.train.MlResultLedger", FakeLedger)
    monkeypatch.setattr("src.stocks.ml.result_ledger.MlResultLedger", FakeLedger)
    monkeypatch.setattr(train, "MlResultLedger", FakeLedger)

    # mock load_cost_evidence to avoid needing real cost file
    def fake_load_cost(path, rng):
        from types import SimpleNamespace as _NS
        return _NS(base_schedule=lambda: _NS(kind="base"), stress_schedule=lambda: _NS(kind="stress"), base_liquidity_model=_NS(), stress_liquidity_model=_NS())
    monkeypatch.setattr("src.stocks.cli.train.load_cost_evidence", fake_load_cost)
    # mock validate_ml_snapshot to pass
    class _PassAudit:
        passed = True
        checks = []
    monkeypatch.setattr("src.stocks.data.ml_integrity.validate_ml_snapshot", lambda *a, **k: _PassAudit())
    # mock model selection evaluate to avoid heavy compute
    def fake_evaluate(data, request, settings, registry=None):
        return {"status":"RESEARCH_ONLY","next_action":"done","candidates":[]}
    monkeypatch.setattr("src.stocks.ml.model_selection.evaluate_model_selection_study", fake_evaluate)
    monkeypatch.setattr("src.stocks.ml.model_selection.build_model_selection_study_settings", lambda parsed, req: SimpleNamespace())

    # need to provide minimal data via loader mock: patch loader.load_training_data to return dummy data
    class DummyData:
        feature_frame = base_frame
        manifest = feat_manifest
        join_evidence = ()
    orig_load = DirectMarketDataLoader.load_training_data
    def fake_load(self, req, decision_time, readiness=None, checkpoint=None, rescope=None):
        # check called with selection.direct_request
        assert req.base_dataset_id == "base_v1"
        return DummyData()
    monkeypatch.setattr(DirectMarketDataLoader, "load_training_data", fake_load)
    # patch assess readiness to pass
    orig_assess = DirectMarketDataLoader.assess_readiness
    def fake_assess(self, req, dt, cost_evidence_path=None):
        from src.stocks.data.direct import DirectReadinessReport, DirectInputReference
        ref = DirectInputReference(base_dataset_id=req.base_dataset_id, base_content_hash=base_manifest.content_hash, feature_dataset_id=req.feature_dataset_id, feature_content_hash=feat_manifest.content_hash, feature_schema_hash=feat_manifest.schema_hash, label_dataset_id=req.label_dataset_id, label_content_hash=label_manifest.content_hash, label_schema_hash=label_manifest.schema_hash, start=req.start, end=req.end, cost_evidence_path=str(cost_path), cost_evidence_hash=cost_hash)
        return DirectReadinessReport(input_reference=ref, errors=(), warnings=(), excluded_sources=())
    monkeypatch.setattr(DirectMarketDataLoader, "assess_readiness", fake_assess)

    parser = train.build_parser()
    args = parser.parse_args(["--artifact-id","test01","--catalog-root",str(catalog_root),"--base-root",str(base_root),"--feature-root",str(feature_root),"--label-root",str(label_root),"--research-start","2024-01-15","--research-end","2024-02-15","--results-root",str(tmp_path/"results")])
    from src.stocks.ml.contracts import NetAlphaTrainingRequest
    request = NetAlphaTrainingRequest(artifact_id="test01")
    result = train.run_research_only_model_selection_study(args, request)
    assert call_count["resolve"] == 1
    assert hasattr(FakeLedger,"captured")
    assert "snapshot_id" not in FakeLedger.captured
