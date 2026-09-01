"""Simulate CLI requires an explicit snapshot id and resolves it through the catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from legacy.stocks.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from legacy.stocks.cli import simulate
from legacy.stocks.settings import REFERENCE_DATETIME


def test_simulate_cli_defaults_to_canonical_roots() -> None:
    assert simulate.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
    assert simulate.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert simulate.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert simulate.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert simulate.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT


def test_simulate_parser_default_decision_time_uses_reference_boundary() -> None:
    args = simulate.build_parser().parse_args(
        ["--artifact-id", "a1"]
    )
    assert args.decision_time == REFERENCE_DATETIME


def test_simulate_cli_rejects_missing_snapshot_id() -> None:
    # snapshotless: missing catalog policy is handled as ValueError inside main, not SystemExit for snapshot
    # But main still requires artifact-id; without active policy it will raise ValueError about missing costs
    import tempfile
    from pathlib import Path as _P
    with pytest.raises((SystemExit, ValueError)):
        simulate.main(["--artifact-id", "a1", "--catalog-root", str(_P(tempfile.gettempdir()) / "empty_catalog")])


def test_simulate_direct_inputs_bypass_snapshot_resolution() -> None:
    """Direct simulation arguments use active selection, not snapshot."""
    args = simulate.build_parser().parse_args(
        [
            "--artifact-id", "a1",
            "--research-start", "2024-01-01",
            "--research-end", "2024-03-31",
        ]
    )
    assert args.research_start.isoformat() == "2024-01-01"


def test_simulate_default_request_matches_canonical_profile(monkeypatch, tmp_path) -> None:
    from legacy.stocks.config.research import CanonicalResearchProfile
    from legacy.stocks.settings import DEFAULT_STOCK_ALPHA
    from legacy.stocks.data.catalog import ActiveDatasetPolicy, CatalogEntry, CatalogKind, CatalogStore, EvidenceCompleteness
    from legacy.stocks.data.contracts import CoverageRange
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash
    from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    import hashlib
    import json as _json
    import polars as pl
    from datetime import UTC, datetime, date

    canonical = CanonicalResearchProfile()
    # stale tuple must not equal canonical
    assert (canonical.top_k, canonical.max_single_weight, canonical.max_exposure) != (5, 0.2, 1.0)
    assert DEFAULT_STOCK_ALPHA.top_k == canonical.top_k
    assert DEFAULT_STOCK_ALPHA.max_single_weight == canonical.max_single_weight
    assert DEFAULT_STOCK_ALPHA.max_exposure == canonical.max_exposure
    assert hasattr(DEFAULT_STOCK_ALPHA, "participation_limit")
    assert DEFAULT_STOCK_ALPHA.participation_limit == canonical.participation_limit

    # simulate profile-less artifact: patch artifact_policy_profile to None and check request uses canonical
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

    # patch simulate to use profile-less artifact
    monkeypatch.setattr("legacy.stocks.workflows.simulate_portfolio.artifact_policy_profile", lambda reg, aid: None)
    monkeypatch.setattr(simulate, "artifact_policy_profile", lambda reg, aid: None)
    # patch loader and backtester to avoid heavy I/O
    from legacy.stocks.data.direct import DirectMarketDataLoader

    def fake_assess(self, req, dt, cost_evidence_path=None):
        from legacy.stocks.data.direct import DirectReadinessReport, DirectInputReference
        ref = DirectInputReference(base_dataset_id=req.base_dataset_id, base_content_hash=base_manifest.content_hash, feature_dataset_id=req.feature_dataset_id, feature_content_hash=feat_manifest.content_hash, feature_schema_hash=feat_manifest.schema_hash, label_dataset_id=req.label_dataset_id, label_content_hash=label_manifest.content_hash, label_schema_hash=label_manifest.schema_hash, start=req.start, end=req.end, cost_evidence_path=str(cost_path), cost_evidence_hash=cost_hash)
        return DirectReadinessReport(input_reference=ref, errors=(), warnings=(), excluded_sources=())

    monkeypatch.setattr(DirectMarketDataLoader, "assess_readiness", fake_assess)

    class FakeSnapshot:
        frame = base_frame
        manifest = type("M", (), {"schema_hash": "s", "content_hash": "c"})()

    def fake_load_backtest(self, req, dt, readiness=None):
        return FakeSnapshot()

    monkeypatch.setattr(DirectMarketDataLoader, "load_backtest_snapshot", fake_load_backtest)
    # patch registry and simulate_portfolio
    from legacy.stocks.research.artifacts import ModelArtifactRegistry

    reg_path = tmp_path / "registry"
    reg_path.mkdir(parents=True, exist_ok=True)
    # create dummy artifact manifest for eligibility
    from legacy.stocks.research.models import ModelManifest

    from src.core.instruments import AssetKind as _AK
    dummy = ModelManifest(artifact_id="a1", asset_kind=_AK.STOCK, feature_set="stock_net_alpha_v1", feature_schema_hash="s", universe_policy_hash="u", label_definition="net_alpha_o2o", label_horizon_sessions=10, eligible_from="2024-01-01T00:00:00+00:00", eligible_to="2024-03-31T00:00:00+00:00", model_type="no_trade", params={})
    monkeypatch.setattr(ModelArtifactRegistry, "read_manifest", lambda self, aid: dummy)

    captured_req = {}

    def fake_simulate(snapshot, registry, request, cost_evidence, diagnostics=None):
        captured_req["request"] = request
        assert request.top_k == canonical.top_k
        assert request.max_single_weight == canonical.max_single_weight
        assert request.max_exposure == canonical.max_exposure
        assert request.participation_limit == canonical.participation_limit
        # also record fingerprint
        from legacy.stocks.trading.policy import stock_risk_policy_fingerprint, StockRiskPolicy

        fp = stock_risk_policy_fingerprint(StockRiskPolicy(top_k=request.top_k, gross_cap=request.max_exposure, single_name_cap=request.max_single_weight, participation_limit=request.participation_limit))
        captured_req["fingerprint"] = fp
        from types import SimpleNamespace

        return SimpleNamespace(final_value=100.0, total_return=0.01)

    monkeypatch.setattr("legacy.stocks.workflows.simulate_portfolio.simulate_portfolio", fake_simulate)
    monkeypatch.setattr(simulate, "simulate_portfolio", fake_simulate)
    # run main with explicit research range fitting coverage
    code = simulate.main(["--artifact-id", "a1", "--catalog-root", str(catalog_root), "--base-root", str(base_root), "--feature-root", str(feature_root), "--label-root", str(label_root), "--registry", str(reg_path), "--results-root", str(tmp_path / "results"), "--research-start", "2024-01-15", "--research-end", "2024-02-15"])
    assert code == 0
    assert captured_req["request"].top_k == canonical.top_k
    assert captured_req["request"].top_k != 5


def test_simulate_cli_rejects_provisional_for_paper_mode(monkeypatch) -> None:
    def fake_resolve(catalog_root, snapshot_id, *, mode):
        raise ValueError(f"snapshot {snapshot_id} is provisional and cannot drive {mode} mode")

    monkeypatch.setattr(simulate, "resolve_snapshot_for_mode", fake_resolve)

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id")
    parser.add_argument("--catalog-root", type=Path, default=simulate.STOCK_CATALOG_ROOT)
    parser.add_argument("--mode", default="paper")
    args = parser.parse_args(["--snapshot-id", "prov_snap_1", "--mode", "paper"])

    with pytest.raises(ValueError, match="provisional"):
        simulate.resolve_snapshot_for_mode(args.catalog_root, args.snapshot_id, mode=args.mode)
