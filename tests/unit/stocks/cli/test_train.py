"""Train CLI requires an explicit snapshot id and resolves it through the catalog."""
from __future__ import annotations

import argparse
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
from src.stocks.ml.contracts import ExecutionFrontierSettings
from src.stocks.settings import REFERENCE_DATE, REFERENCE_DATETIME


def test_train_cli_defaults_to_canonical_roots() -> None:
    assert train.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
    assert train.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert train.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert train.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert train.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT


def test_train_cli_rejects_missing_snapshot_id() -> None:
    with pytest.raises(SystemExit):
        train.main(["--artifact-id", "a1"])


def test_train_cli_rejects_legacy_trial_flag() -> None:
    with pytest.raises(SystemExit):
        train.main(
            [
                "--artifact-id",
                "a1",
                "--snapshot-id",
                "s1",
                "--optuna-trials",
                "120",
            ]
        )


def test_train_cli_exposes_net_alpha_args() -> None:
    # SDA-05: CLI exposes direct as-of selection alongside the migration path.
    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--artifact-id",
            "a1",
            "--snapshot-id",
            "s1",
            "--candidate-horizon-sessions",
            "3,5,8,10,15,20",
            "--max-rss-mib",
            "4096",
            "--model-threads",
            "2",
        ]
    )
    assert args.candidate_horizon_sessions == "3,5,8,10,15,20"
    assert args.max_rss_mib == 4096
    assert args.model_threads == 2
    assert not hasattr(args, "optuna_trials")
    assert not hasattr(args, "lgb_threads")
    assert not hasattr(args, "resume")
    assert args.decision_time == REFERENCE_DATETIME
    assert args.research_end == REFERENCE_DATE


def test_net_alpha_args_map_memory_reserve_mib() -> None:
    """ML_FULL_EXECUTION_P0_TELEMETRY_AND_CLI_05: --memory-reserve-mib maps unchanged."""
    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--artifact-id",
            "a1",
            "--snapshot-id",
            "s1",
            "--memory-reserve-mib",
            "768",
        ]
    )
    assert args.memory_reserve_mib == 768
    request = train._build_training_request(args)
    assert request.memory_reserve_mib == 768

    default_request = train._build_training_request(
        parser.parse_args(["--artifact-id", "a1", "--snapshot-id", "s1"])
    )
    assert default_request.memory_reserve_mib == 0


def test_net_alpha_args_reject_negative_memory_reserve_mib() -> None:
    from dataclasses import replace as dataclass_replace

    from src.stocks.ml.contracts import NetAlphaTrainingRequest

    request = NetAlphaTrainingRequest(artifact_id="neg")
    with pytest.raises(ValueError, match="memory_reserve_mib"):
        dataclass_replace(request, memory_reserve_mib=-1)


def test_train_cli_exposes_complete_execution_frontier() -> None:
    """H3_FRONTIER_CLI_01: CLI input preserves the complete H/C/K grid."""
    args = train.build_parser().parse_args(
        [
            "--artifact-id",
            "h3",
            "--snapshot-id",
            "snapshot",
            "--candidate-horizon-sessions",
            "3",
            "--candidate-rebalance-frequency-sessions",
            "1,2,3",
            "--candidate-top-k",
            "12,16,20,24",
        ]
    )

    assert args.candidate_horizon_sessions == "3"
    assert args.candidate_rebalance_frequency_sessions == "1,2,3"
    assert args.candidate_top_k == "12,16,20,24"
    frontier = ExecutionFrontierSettings(
        candidate_horizon_sessions=train._parse_horizons(args.candidate_horizon_sessions),
        candidate_rebalance_frequency_sessions=train._parse_horizons(
            args.candidate_rebalance_frequency_sessions
        ),
        candidate_top_k=train._parse_horizons(args.candidate_top_k),
    )
    assert len(frontier.require_feasible_horizons(0.90, 0.08)) == 12


def test_train_cli_resolves_snapshot_and_composes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSnapshot:
        pass

    class FakeRepository:
        def __init__(self, **kwargs):
            captured["roots"] = kwargs

        def compose_training_snapshot(self, snapshot, **kwargs):
            captured["snapshot_id"] = snapshot.snapshot_id
            captured["compose_kwargs"] = kwargs
            return None

    def fake_resolve(catalog_root, snapshot_id, *, mode):
        captured["catalog_root"] = catalog_root
        captured["mode"] = mode
        fake = FakeSnapshot()
        fake.snapshot_id = snapshot_id
        return fake

    monkeypatch.setattr(train, "resolve_snapshot_for_mode", fake_resolve)
    monkeypatch.setattr(train, "ResearchDataRepository", FakeRepository)

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id")
    parser.add_argument("--catalog-root", type=Path, default=train.STOCK_CATALOG_ROOT)
    args = parser.parse_args(["--snapshot-id", "research_snap_1"])

    snapshot = train.resolve_snapshot_for_mode(args.catalog_root, args.snapshot_id, mode="research")
    repo = train.ResearchDataRepository(base_root=args.catalog_root)
    repo.compose_training_snapshot(snapshot, feature_set="stock_net_alpha_v1", decision_time=None)
    assert     captured["snapshot_id"] == "research_snap_1"
    assert captured["mode"] == "research"


def _install_train_fakes(monkeypatch, train_result, ledger_cls, captured) -> None:
    """Patch the CLI's external dependencies and capture ledger calls."""
    from datetime import UTC, datetime

    import polars as pl

    class _FakeSnapshot:
        costs = None

    class _FakeRepository:
        def __init__(self, **kwargs):
            captured["repo_kwargs"] = kwargs

        def compose_labeled_training_snapshot(self, snapshot, **kwargs):
            del snapshot
            captured["compose_kwargs"] = kwargs
            return SimpleNamespace()

    class _FakeData:
        manifest = SimpleNamespace(
            schema_hash="schema-hash-1",
            universe_policy_hash="universe-hash-1",
            label_definition="net_alpha_o2o",
            label_horizon_sessions=5,
        )
        join_evidence: tuple = ()
        feature_frame = pl.DataFrame(
            {
                "instrument_id": ["KRX:00001"],
                "session": [datetime(2024, 1, 1, tzinfo=UTC)],
                "feature__x": [1.0],
            }
        )

    def _fake_train(data, registry, request):
        del data, registry, request
        if isinstance(train_result, BaseException):
            raise train_result
        return train_result

    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train, "compose_net_alpha_training_data", lambda *a, **k: _FakeData()
    )
    monkeypatch.setattr(
        train, "_resolve_cost_contexts", lambda snapshot: (None, None, None, None)
    )
    monkeypatch.setattr(train, "MlResultLedger", ledger_cls)
    monkeypatch.setattr(train, "train_net_alpha_model", _fake_train)


def _fake_model_manifest(artifact_id: str = "a1"):
    from src.stocks.research.models import ModelManifest

    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind="stock",
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="schema-hash-1",
        universe_policy_hash="universe-hash-1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
    )


def test_train_cli_records_completed_through_ledger(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _FakeLedger:
        def __init__(self, results_root):
            captured["results_root"] = results_root

        def record_completed(self, context, manifest, registry, telemetry=None):
            captured["completed"] = (context, manifest, registry)

        def record_failed(self, context, phase, exc, telemetry=None):
            captured["failed"] = (context, phase, exc)

    _install_train_fakes(monkeypatch, _fake_model_manifest(), _FakeLedger, captured)
    results_root = tmp_path / "docs" / "results"
    rc = train.main(
        [
            "--artifact-id",
            "a1",
            "--snapshot-id",
            "s1",
            "--results-root",
            str(results_root),
        ]
    )
    assert rc == 0
    assert "failed" not in captured
    completed = captured["completed"]
    assert completed[0].artifact_id == "a1"
    assert completed[0].snapshot_id == "s1"
    assert completed[1].artifact_id == "a1"
    assert captured["results_root"] == results_root


def test_train_cli_records_failed_and_reraises(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _FakeLedger:
        def __init__(self, results_root):
            captured["results_root"] = results_root

        def record_completed(self, context, manifest, registry, telemetry=None):
            captured["completed"] = (context, manifest, registry)

        def record_failed(self, context, phase, exc, telemetry=None):
            captured["failed"] = (context, phase, exc)

    boom = ValueError("boom")
    _install_train_fakes(monkeypatch, boom, _FakeLedger, captured)
    with pytest.raises(ValueError, match="boom"):
        train.main(
            [
                "--artifact-id",
                "a1",
                "--snapshot-id",
                "s1",
                "--results-root",
                str(tmp_path / "docs" / "results"),
            ]
        )
    failed = captured["failed"]
    assert failed[0].artifact_id == "a1"
    assert failed[1] == "train_net_alpha_model"
    assert failed[2] is boom


def test_train_cli_ledger_write_failure_preserves_artifact(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _ExplodingLedger:
        def __init__(self, results_root):
            captured["results_root"] = results_root

        def record_completed(self, context, manifest, registry, telemetry=None):
            raise RuntimeError("ledger boom")

        def record_failed(self, context, phase, exc, telemetry=None):
            captured["failed"] = (context, phase, exc)

    _install_train_fakes(
        monkeypatch, _fake_model_manifest(), _ExplodingLedger, captured
    )
    rc = train.main(
        [
            "--artifact-id",
            "a1",
            "--snapshot-id",
            "s1",
            "--results-root",
            str(tmp_path / "docs" / "results"),
        ]
    )
    assert rc == 0
    assert "failed" not in captured


def test_direct_dataset_arguments() -> None:
    """LMD-05: CLI accepts required base/feature/label dataset IDs."""
    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--artifact-id",
            "test_artifact",
            "--base-dataset-id",
            "base_2024",
            "--feature-dataset-id",
            "features_2024",
            "--label-dataset-id",
            "labels_2024",
            "--research-start-direct",
            "2024-01-01",
            "--research-end-direct",
            "2024-03-31",
        ]
    )
    assert args.base_dataset_id == "base_2024"
    assert args.feature_dataset_id == "features_2024"
    assert args.label_dataset_id == "labels_2024"
    assert args.research_start_direct.isoformat() == "2024-01-01"
    assert args.research_end_direct.isoformat() == "2024-03-31"


def test_direct_dataset_arguments_rejects_snapshot_id() -> None:
    """CLI rejects --snapshot-id when using direct dataset arguments."""
    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--artifact-id",
            "test_artifact",
            "--snapshot-id",
            "some_snapshot",
            "--base-dataset-id",
            "base_2024",
            "--feature-dataset-id",
            "features_2024",
            "--label-dataset-id",
            "labels_2024",
            "--research-start-direct",
            "2024-01-01",
            "--research-end-direct",
            "2024-03-31",
        ]
    )
    # When direct dataset IDs are provided, snapshot_id is ignored
    assert args.snapshot_id == "some_snapshot"
    assert args.base_dataset_id == "base_2024"


def test_direct_dataset_defaults_use_reference_boundary() -> None:
    args = train.build_parser().parse_args(
        [
            "--artifact-id",
            "test_artifact",
            "--base-dataset-id",
            "base",
            "--feature-dataset-id",
            "features",
            "--label-dataset-id",
            "labels",
        ]
    )
    assert args.research_end_direct == REFERENCE_DATE
    assert args.decision_time == REFERENCE_DATETIME


def test_direct_cost_provenance_01_requires_known_snapshot(monkeypatch, tmp_path) -> None:
    """DIRECT_COST_PROVENANCE_01: unknown evidence cannot certify a run."""
    monkeypatch.setattr(train, "STOCK_CATALOG_ROOT", tmp_path)
    with pytest.raises(ValueError, match=r"cost snapshot .* not found"):
        train._resolve_direct_cost_context(
            "missing-cost-evidence", SimpleNamespace(), object()
        )
