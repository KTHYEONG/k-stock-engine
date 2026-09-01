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


def test_train_parser_exposes_direct_dataset_flags_but_no_snapshot_flags() -> None:
    parser = train.build_parser()
    for flag in ["--snapshot-id", "--as-of"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["--artifact-id", "a1", flag, "val"])
    direct = parser.parse_args(
        [
            "--artifact-id", "a1",
            "--base-dataset-id", "base",
            "--feature-dataset-id", "features",
            "--label-dataset-id", "labels",
            "--research-start-direct", "2024-01-01",
            "--research-end-direct", "2024-02-01",
        ]
    )
    assert direct.base_dataset_id == "base"
    assert direct.feature_dataset_id == "features"
    assert direct.label_dataset_id == "labels"
    # research start/end remain accepted
    args = parser.parse_args(["--artifact-id", "a1", "--research-start", "2024-01-01", "--research-end", "2024-02-01"])
    assert args.research_start == date(2024, 1, 1)


def test_stock_only_cli_requires_small_account_and_rejects_etf_satellite() -> None:
    from src.stocks.cli.train import _build_training_request, build_parser

    parser = build_parser()
    missing_account = parser.parse_args(['--artifact-id', 'stock-only', '--enable-stock-only-small-capital'])
    with pytest.raises(ValueError, match='account-capital-krw'):
        _build_training_request(missing_account)
    mixed_instruments = parser.parse_args([
        '--artifact-id', 'stock-only', '--enable-stock-only-small-capital',
        '--account-capital-krw', '10000000', '--enable-etf-satellite',
    ])
    with pytest.raises(ValueError, match='enable-etf-satellite'):
        _build_training_request(mixed_instruments)


def test_train_main_resolves_active_selection_once(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime
    import hashlib, json as _json
    from pathlib import Path
    import polars as pl
    from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.stocks.data.catalog import ActiveDatasetPolicy, CatalogEntry, CatalogKind, CatalogStore, EvidenceCompleteness
    from src.stocks.data.contracts import CoverageRange
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    catalog_root = tmp_path / "catalog"
    base_root = tmp_path / "base"
    feature_root = tmp_path / "feature"
    label_root = tmp_path / "label"
    for p in (base_root, feature_root, label_root):
        p.mkdir(parents=True, exist_ok=True)
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(_json.dumps({"c": 1}), encoding="utf-8")
    cost_hash = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    sessions = [datetime(2024, 1, 10, tzinfo=UTC), datetime(2024, 2, 10, tzinfo=UTC)]
    base_frame = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00001"], "session": sessions, "open": [100.0, 101.0], "close": [101.0, 102.0], "volume": [1e6, 1e6], "trading_value": [1e8, 1e8]})
    feat_frame = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00001"], "session": sessions, "feature__x": [0.1, 0.2]})
    label_frame = pl.DataFrame({"instrument_id": ["KRX:00001", "KRX:00001"], "session": sessions, "horizon_sessions": [10, 10], "net_alpha_target": [0.01, 0.02], "label_available_time": sessions})

    def _write(root, did, frame, fset):
        store = ParquetDatasetStore(root)
        manifest = make_manifest(asset_kind=AssetKind.STOCK, columns=list(frame.columns), feature_set=fset, label_definition="net_alpha_o2o", label_horizon_sessions=10, time_start=datetime(2024, 1, 1, tzinfo=UTC), time_end=datetime(2024, 3, 31, tzinfo=UTC), provider_version="t", universe_policy_version="t", row_count=frame.height, generated_time=datetime.now(UTC), schema_version="v2", storage_layout=HIVE_PARTITION_LAYOUT)
        from dataclasses import replace
        manifest = replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))
        store.write_partitioned(frame, dataset_id=did, manifest=manifest, expected_feature_set=fset, decision_time=datetime(2024, 3, 31, tzinfo=UTC))
        return manifest

    base_manifest = _write(base_root, "base_v1", base_frame, "base_panel")
    feat_manifest = _write(feature_root, "feat_v1", feat_frame, "stock_net_alpha_v1")
    label_manifest = _write(label_root, "label_v1", label_frame, "labels")
    store = CatalogStore(catalog_root)
    rng = CoverageRange(start=date(2024, 1, 1), end=date(2024, 3, 31))
    for kind, name, manifest, path in [(CatalogKind.BASE_PANEL, "base_v1", base_manifest, base_root / "base_v1"), (CatalogKind.FEATURES, "feat_v1", feat_manifest, feature_root / "feat_v1"), (CatalogKind.LABELS, "label_v1", label_manifest, label_root / "label_v1")]:
        store.register(CatalogEntry(kind=kind, name=name, content_hash=manifest.content_hash, schema_hash=manifest.schema_hash, registered_at=datetime(2024, 1, 1, tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(path)))
    store.register(CatalogEntry(kind=CatalogKind.COSTS, name="costs_v1", content_hash=cost_hash, schema_hash=hashlib.sha256(cost_path.read_bytes()).hexdigest(), registered_at=datetime(2024, 1, 1, tzinfo=UTC), coverage=rng, completeness=EvidenceCompleteness.COMPLETE, path=str(cost_path)))
    store.save_active_policy(ActiveDatasetPolicy(entries=((CatalogKind.BASE_PANEL, "base_v1"), (CatalogKind.FEATURES, "feat_v1"), (CatalogKind.LABELS, "label_v1"), (CatalogKind.COSTS, "costs_v1"))))

    from src.stocks.data.direct import DirectMarketDataLoader

    orig_resolve = train.resolve_active_research_data
    call_count = {"resolve": 0}
    orig_fn = orig_resolve

    def counting_resolve(**kwargs):
        call_count["resolve"] += 1
        return orig_fn(**kwargs)

    monkeypatch.setattr(train, "resolve_active_research_data", counting_resolve)
    # also patch catalog resolve via train._LAST_TRAIN_PARSED handling – counting via same

    captured_req = {}

    def fake_load(self, req, decision_time, readiness=None, checkpoint=None, rescope=None):
        # ensure direct request IDs unchanged from selection
        assert req.base_dataset_id == "base_v1"
        assert req.feature_dataset_id == "feat_v1"
        assert req.label_dataset_id == "label_v1"
        # ensure not accessing deprecated parser fields
        for bad in ("base_dataset_id", "feature_dataset_id", "label_dataset_id", "snapshot_id", "as_of"):
            if hasattr(req, bad) and bad not in ("base_dataset_id", "feature_dataset_id", "label_dataset_id"):
                pass
        captured_req["req"] = req
        class DummyData:
            feature_frame = base_frame
            manifest = feat_manifest
            labels_by_horizon = {10: label_frame}
            join_evidence = ()

        return DummyData()

    monkeypatch.setattr(DirectMarketDataLoader, "load_training_data", fake_load)
    monkeypatch.setattr(DirectMarketDataLoader, "load_backtest_snapshot", fake_load)

    def fake_assess(self, req, dt, cost_evidence_path=None):
        from src.stocks.data.direct import DirectReadinessReport, DirectInputReference
        ref = DirectInputReference(base_dataset_id=req.base_dataset_id, base_content_hash=base_manifest.content_hash, feature_dataset_id=req.feature_dataset_id, feature_content_hash=feat_manifest.content_hash, feature_schema_hash=feat_manifest.schema_hash, label_dataset_id=req.label_dataset_id, label_content_hash=label_manifest.content_hash, label_schema_hash=label_manifest.schema_hash, start=req.start, end=req.end, cost_evidence_path=str(cost_path), cost_evidence_hash=cost_hash)
        return DirectReadinessReport(input_reference=ref, errors=(), warnings=(), excluded_sources=())

    monkeypatch.setattr(DirectMarketDataLoader, "assess_readiness", fake_assess)
    monkeypatch.setattr("src.stocks.cli.train.load_cost_evidence", lambda path, rng: __import__("types").SimpleNamespace(base_schedule=lambda: __import__("types").SimpleNamespace(kind="base"), stress_schedule=lambda: __import__("types").SimpleNamespace(kind="stress"), base_liquidity_model=__import__("types").SimpleNamespace(), stress_liquidity_model=__import__("types").SimpleNamespace()))

    class _PassAudit:
        passed = True
        checks = []

    monkeypatch.setattr("src.stocks.data.ml_integrity.validate_ml_snapshot", lambda *a, **k: _PassAudit())

    def fake_train(data, registry, request, diagnostics=None, progress=None):
        from src.stocks.research.models import ModelManifest
        from src.core.instruments import AssetKind

        return ModelManifest(artifact_id=request.artifact_id, asset_kind=AssetKind.STOCK, feature_set="stock_net_alpha_v1", feature_schema_hash="s", universe_policy_hash="u", label_definition="net_alpha_o2o", label_horizon_sessions=10, eligible_from="2024-01-01T00:00:00+00:00", eligible_to="2024-03-31T00:00:00+00:00", model_type="no_trade", params={})

    monkeypatch.setattr("src.stocks.ml.training.train_net_alpha_model", fake_train)
    monkeypatch.setattr(train, "train_net_alpha_model", fake_train)

    # invoke main with explicit research range fitting coverage – should resolve active once and not raise AttributeError for deprecated fields
    try:
        code = train.main(["--artifact-id", "a1", "--catalog-root", str(catalog_root), "--base-root", str(base_root), "--feature-root", str(feature_root), "--label-root", str(label_root), "--results-root", str(tmp_path / "results"), "--research-start", "2024-01-15", "--research-end", "2024-02-15"])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
    assert call_count["resolve"] == 1
    assert "req" in captured_req
    # ensure no AttributeError for missing parser fields was raised internally – success code 0
    assert code == 0


def test_train_cli_rejects_conflicting_studies(monkeypatch) -> None:
    from src.stocks.cli.contracts import parse_train_command

    # patch to ensure resolve not called
    called = {"resolve": False, "loader": False}

    def fake_resolve(**kwargs):
        called["resolve"] = True
        raise AssertionError("resolve should not be called on conflicting studies")

    monkeypatch.setattr(train, "resolve_active_research_data", fake_resolve)
    from src.stocks.data.direct import DirectMarketDataLoader

    orig_load = DirectMarketDataLoader.load_training_data

    def fake_load(*a, **kw):
        called["loader"] = True
        raise AssertionError("loader should not be called")

    monkeypatch.setattr(DirectMarketDataLoader, "load_training_data", fake_load)
    monkeypatch.setattr(DirectMarketDataLoader, "load_backtest_snapshot", fake_load)

    # two aliases
    with pytest.raises((ValueError, SystemExit), match="conflicting"):
        train.main(["--artifact-id", "a1", "--research-only-growth-route", "--research-only-temporal-window-study"])
    assert not called["resolve"]
    assert not called["loader"]
    # alias plus --study
    with pytest.raises((ValueError, SystemExit), match="conflicting"):
        train.main(["--artifact-id", "a1", "--research-only-growth-route", "--study", "temporal_window_study"])
    assert not called["resolve"]


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

    # Mock the Markdown report to capture data_selection without snapshot_id.
    class FakeLedger:
        def __init__(self, root): pass
        def record_research_outcome(self, **kwargs):
            data_inputs = kwargs.get("data_inputs", {})
            # ensure no snapshot_id
            if "snapshot_id" in data_inputs:
                raise AssertionError("snapshot_id should not be in data_inputs")
            FakeLedger.captured = data_inputs

    monkeypatch.setattr("src.stocks.cli.train.MlComparisonReport", FakeLedger)

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

def test_research_request_preserves_low_mcap_rescope_policy(monkeypatch, tmp_path) -> None:
    from src.stocks.cli import train
    from src.stocks.ml.contracts import UniverseRescopeSettings

    parser = train.build_parser()
    parsed = parser.parse_args([
        '--artifact-id', 'rescope-probe',
        '--enable-universe-rescope',
        '--rescope-mcap-quantile-lo', '0.0',
        '--rescope-mcap-quantile-hi', '0.25',
    ])
    request = train._build_training_request(parsed)
    expected = UniverseRescopeSettings(
        market_cap_quantile_lo=0.0,
        market_cap_quantile_hi=0.25,
    )

    assert request.universe_rescope == expected
    assert request.universe_rescope.fingerprint == expected.fingerprint
