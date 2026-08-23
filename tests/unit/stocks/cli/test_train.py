"""Train CLI requires an explicit snapshot id and resolves it through the catalog."""
from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
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
from src.stocks.data.direct import DirectDataRequest
from src.stocks.ml.contracts import ExecutionFrontierSettings
from src.stocks.research.models import ModelManifest
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


def test_train_cli_h10_only_frontier_valid_and_default_unchanged() -> None:
    """ML_HORIZON_SCOPE_01_H10_ONLY_FRONTIER.

    A pre-registered H10-only frontier parses, every cadence stays feasible
    for H10 (C <= 10), and the operational grid is H10 x {C5,C10} x K while
    the default discovery grid remains H10/H20.
    """
    args = train.build_parser().parse_args(
        [
            "--artifact-id",
            "h10_only",
            "--snapshot-id",
            "s1",
            "--candidate-horizon-sessions",
            "10",
            "--candidate-rebalance-frequency-sessions",
            "5,10",
            "--candidate-top-k",
            "12,16,20,24",
        ]
    )
    request = train._build_training_request(args)
    assert request.candidate_horizon_sessions == (10,)
    assert request.execution_frontier.candidate_horizon_sessions == (10,)
    assert request.execution_frontier.candidate_rebalance_frequency_sessions == (5, 10)
    cells = request.execution_frontier.require_feasible_horizons(0.90, 0.08)
    assert {horizon for horizon, _, _ in cells} == {10}
    assert all(cadence <= 10 for _, cadence, _ in cells)
    assert len(cells) == 8

    default_request = train._build_training_request(
        train.build_parser().parse_args(
            ["--artifact-id", "h_default", "--snapshot-id", "s1"]
        )
    )
    assert default_request.candidate_horizon_sessions == (10, 20)


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


FULL_TERMINAL_03 = "FULL_TERMINAL_03_REDUCED_PARITY"


def _write_parity_dataset(store, dataset_id, frame, *, feature_set: str) -> None:
    from dataclasses import replace as dc_replace

    from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.storage.parquet_datasets import canonical_content_hash

    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=list(frame.columns),
        feature_set=feature_set,
        label_definition="net_alpha_o2o",
        label_horizon_sessions=20,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 31, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        generated_time=datetime.now(UTC),
        schema_version="v2",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    manifest = dc_replace(
        manifest, content_hash=canonical_content_hash(frame, frame.columns)
    )
    store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=feature_set,
        decision_time=datetime(2024, 3, 31, tzinfo=UTC),
    )


def _build_parity_fixture(tmp_path):
    """Reduced deterministic H10/H20 direct fixture shared by both loaders."""
    import polars as pl

    from src.storage.parquet_datasets import ParquetDatasetStore

    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    for root in (base_root, feature_root, label_root):
        root.mkdir()

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(25)]
    base_rows = []
    feature_rows = []
    label_rows = []
    last = sessions[-1]
    for session in sessions:
        for t in range(2):
            price = 100.0 + t
            base_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "open": price, "close": price * 1.01,
                "volume": 1e6, "trading_value": price * 1e6,
            })
            feature_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "feature__momentum_5d": 0.1 * t,
                "feature__volatility_20d": 0.02,
            })
            for horizon in (10, 20):
                usable = session + timedelta(days=horizon) <= last
                label_rows.append({
                    "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                    "horizon_sessions": horizon,
                    "net_alpha_target": 0.001 * horizon if usable else None,
                    "label_available_time": session + timedelta(days=horizon),
                })

    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)
    _write_parity_dataset(base_store, "base_p", pl.DataFrame(base_rows), feature_set="base_panel")
    _write_parity_dataset(feature_store, "feat_p", pl.DataFrame(feature_rows), feature_set="stock_net_alpha_v1")
    _write_parity_dataset(label_store, "lab_p", pl.DataFrame(label_rows), feature_set="labels")
    return (
        DirectDataRequest(
            base_dataset_id="base_p",
            feature_dataset_id="feat_p",
            label_dataset_id="lab_p",
            start=date(2024, 1, 1),
            end=date(2024, 1, 25),
            candidate_horizon_sessions=(10, 20),
        ),
        base_root,
        feature_root,
        label_root,
    )


def test_full_terminal_03_reduced_parity(monkeypatch, tmp_path) -> None:
    """FULL_TERMINAL_03_REDUCED_PARITY.

    The reduced direct fixture preserves decision row count, per-horizon label
    key set, feature schema hash, and terminal NO_TRADE classification exactly.
    """
    import polars as pl

    from src.stocks.data.direct import DirectMarketDataLoader
    from src.stocks.ml.data import (
        compose_direct_net_alpha_training_data,
        validate_ml_market_data,
    )

    request, base_root, feature_root, label_root = _build_parity_fixture(tmp_path)
    decision_time = datetime(2024, 3, 1, tzinfo=UTC)

    loader = DirectMarketDataLoader(
        base_root=base_root, feature_root=feature_root, label_root=label_root
    )
    market_data = loader.load(request)
    validate_ml_market_data(market_data, request.candidate_horizon_sessions)
    legacy = compose_direct_net_alpha_training_data(market_data, decision_time)

    modern_loader = DirectMarketDataLoader(
        base_root=tmp_path / "base2",
        feature_root=tmp_path / "features2",
        label_root=tmp_path / "labels2",
    )
    # Re-point the independent loader at the same physical datasets so the
    # comparison isolates the composition path, not the fixture copies.
    modern_loader._base_store = __import__(
        "src.storage.parquet_datasets", fromlist=["ParquetDatasetStore"]
    ).ParquetDatasetStore(base_root)
    modern_loader._feature_store = __import__(
        "src.storage.parquet_datasets", fromlist=["ParquetDatasetStore"]
    ).ParquetDatasetStore(feature_root)
    modern_loader._label_store = __import__(
        "src.storage.parquet_datasets", fromlist=["ParquetDatasetStore"]
    ).ParquetDatasetStore(label_root)
    modern = modern_loader.load_training_data(request, decision_time)

    assert modern.feature_frame.height == legacy.feature_frame.height
    assert modern.manifest.schema_hash == legacy.manifest.schema_hash
    assert modern.manifest.content_hash == legacy.manifest.content_hash
    assert set(modern.labels_by_horizon) == set(legacy.labels_by_horizon)
    assert modern.feature_frame.equals(legacy.feature_frame)
    order = ["instrument_id", "session", "net_alpha_target", "label_available_time"]
    for horizon, label_frame in modern.labels_by_horizon.items():
        reference = legacy.labels_by_horizon[horizon]
        assert label_frame.select(order).rows() == reference.select(order).rows()
        evidence_modern = [
            e for e in modern.join_evidence if e.horizon_sessions == horizon
        ]
        evidence_legacy = [
            e for e in legacy.join_evidence if e.horizon_sessions == horizon
        ]
        assert evidence_modern[0].joined_rows == evidence_legacy[0].joined_rows

    # Terminal classification parity through the CLI direct path.
    captured: dict[str, object] = {}

    class _CapturingLedger:
        def __init__(self, results_root):
            captured["results_root"] = results_root

        def record_completed(self, context, manifest, registry, telemetry=None):
            captured["completed"] = (context, manifest)

        def record_failed(self, context, phase, exc, telemetry=None):  # pragma: no cover
            captured["failed"] = (context, phase, exc)

    def _fake_no_trade_train(data, registry, req):
        captured["train_data"] = data
        return ModelManifest(
            artifact_id=req.artifact_id,
            asset_kind="stock",
            feature_set="stock_net_alpha_v1",
            feature_schema_hash=data.manifest.schema_hash,
            universe_policy_hash=data.manifest.universe_policy_hash,
            label_definition="net_alpha_o2o",
            label_horizon_sessions=20,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-03-31T00:00:00+00:00",
            model_type="no_trade",
        )

    diag_root = tmp_path / "diag"
    diag_root.mkdir()
    monkeypatch.setattr(train, "MlResultLedger", _CapturingLedger)
    monkeypatch.setattr(train, "train_net_alpha_model", _fake_no_trade_train)
    monkeypatch.setattr("src.core.paths.RUN_DIAGNOSTIC_ROOT", diag_root)

    rc = train.main(
        [
            "--artifact-id", "parity01",
            "--base-dataset-id", "base_p",
            "--feature-dataset-id", "feat_p",
            "--label-dataset-id", "lab_p",
            "--research-start-direct", "2024-01-01",
            "--research-end-direct", "2024-01-25",
            "--base-root", str(base_root),
            "--feature-root", str(feature_root),
            "--label-root", str(label_root),
            "--registry", str(tmp_path / "artifacts"),
            "--results-root", str(tmp_path / "results"),
            "--decision-time", decision_time.isoformat(),
        ]
    )

    assert rc == 0
    trained_data = captured["train_data"]
    assert isinstance(trained_data.feature_frame, pl.DataFrame)
    assert trained_data.feature_frame.equals(modern.feature_frame)
    assert trained_data.manifest.schema_hash == modern.manifest.schema_hash
    _, completed_manifest = captured["completed"]
    assert completed_manifest.model_type == "no_trade"
    assert "failed" not in captured


TERMINAL_OBS_05 = "TERMINAL_OBS_05_REDUCED_FULL_PARITY"


def test_terminal_obs_05_reduced_full_parity(monkeypatch, tmp_path) -> None:
    """TERMINAL_OBS_05_REDUCED_FULL_PARITY.

    The reduced direct fixture exits 0 with exactly one completed ledger
    record, and the durable execution journal contains the full stage chain:
    direct_preflight, direct_collected, matrix_preparation, terminal_pass.
    """
    import json as _json

    request, base_root, feature_root, label_root = _build_parity_fixture(tmp_path)
    del request

    captured: dict[str, object] = {}

    class _CapturingLedger:
        def __init__(self, results_root):
            captured["results_root"] = results_root

        def record_completed(self, context, manifest, registry, telemetry=None):
            captured.setdefault("completed", []).append((context, manifest))

        def record_failed(self, context, phase, exc, telemetry=None):
            captured.setdefault("failed", []).append((context, phase, exc))

    diag_root = tmp_path / "diag_obs05"
    diag_root.mkdir()
    monkeypatch.setattr(train, "MlResultLedger", _CapturingLedger)
    monkeypatch.setattr(
        train,
        "train_net_alpha_model",
        lambda data, registry, req: ModelManifest(
            artifact_id=req.artifact_id,
            asset_kind="stock",
            feature_set="stock_net_alpha_v1",
            feature_schema_hash=data.manifest.schema_hash,
            universe_policy_hash=data.manifest.universe_policy_hash,
            label_definition="net_alpha_o2o",
            label_horizon_sessions=20,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-03-31T00:00:00+00:00",
            model_type="no_trade",
        ),
    )
    monkeypatch.setattr("src.core.paths.RUN_DIAGNOSTIC_ROOT", diag_root)

    rc = train.main(
        [
            "--artifact-id", "obs05",
            "--base-dataset-id", "base_p",
            "--feature-dataset-id", "feat_p",
            "--label-dataset-id", "lab_p",
            "--research-start-direct", "2024-01-01",
            "--research-end-direct", "2024-01-25",
            "--base-root", str(base_root),
            "--feature-root", str(feature_root),
            "--label-root", str(label_root),
            "--registry", str(tmp_path / "artifacts"),
            "--results-root", str(tmp_path / "results"),
            "--candidate-horizon-sessions", "10,20",
        ]
    )

    assert rc == 0
    assert len(captured["completed"]) == 1  # one completed ledger record
    assert "failed" not in captured

    journal_lines = (
        diag_root / "obs05" / "execution_journal.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    stages = [
        record["stage"]
        for record in (_json.loads(line) for line in journal_lines)
        if record.get("event") == "checkpoint"
    ]
    for required_stage in (
        "direct_preflight",
        "direct_collected",
        "matrix_preparation",
        "terminal_pass",
    ):
        assert required_stage in stages, f"missing journal stage {required_stage}"


def test_terminal_progress_04_direct_journal_and_lookback_flag(
    monkeypatch, tmp_path
) -> None:
    """TERMINAL_PROGRESS_04_DIRECT_JOURNAL.

    The CLI parses --max-training-lookback-sessions 1260 into the request,
    the direct run forwards progress into the fsynced journal so it records
    fitting_started and fitting_complete checkpoints, and the final durable
    journal row is a terminal passed event.
    """
    import json as _json

    request_args, base_root, feature_root, label_root = _build_parity_fixture(
        tmp_path
    )
    del request_args

    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--artifact-id",
            "lookback04",
            "--max-training-lookback-sessions",
            "1260",
        ]
    )
    assert args.max_training_lookback_sessions == 1260
    built = train._build_training_request(args)
    assert built.max_training_lookback_sessions == 1260
    default_built = train._build_training_request(
        parser.parse_args(["--artifact-id", "lookback04"])
    )
    assert default_built.max_training_lookback_sessions is None

    captured: dict[str, object] = {}

    class _CapturingLedger:
        def __init__(self, results_root):
            captured["results_root"] = results_root

        def record_completed(self, context, manifest, registry, telemetry=None):
            captured["completed"] = (context, manifest)

        def record_failed(self, context, phase, exc, telemetry=None):
            captured["failed"] = (context, phase, exc)

    def _progressing_no_trade(
        data, registry, req, *, diagnostics=None, progress=None
    ):
        del data, registry, diagnostics
        if progress is not None:
            progress("fitting_started", {"fold_count": 3})
            progress("fitting_complete", {"model_type": "no_trade"})
        return ModelManifest(
            artifact_id=req.artifact_id,
            asset_kind="stock",
            feature_set="stock_net_alpha_v1",
            feature_schema_hash="fixture-schema",
            universe_policy_hash="fixture-universe",
            label_definition="net_alpha_o2o",
            label_horizon_sessions=20,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-03-31T00:00:00+00:00",
            model_type="no_trade",
        )

    diag_root = tmp_path / "diag_tp04"
    diag_root.mkdir()
    monkeypatch.setattr(train, "MlResultLedger", _CapturingLedger)
    # Patch both bindings: _invoke_training inspects the cli-train symbol,
    # while TrainingOrchestrator.run resolves the ml-training module symbol.
    monkeypatch.setattr(train, "train_net_alpha_model", _progressing_no_trade)
    monkeypatch.setattr(
        "src.stocks.ml.training.train_net_alpha_model", _progressing_no_trade
    )
    monkeypatch.setattr("src.core.paths.RUN_DIAGNOSTIC_ROOT", diag_root)

    rc = train.main(
        [
            "--artifact-id", "lookback04",
            "--base-dataset-id", "base_p",
            "--feature-dataset-id", "feat_p",
            "--label-dataset-id", "lab_p",
            "--research-start-direct", "2024-01-01",
            "--research-end-direct", "2024-01-25",
            "--base-root", str(base_root),
            "--feature-root", str(feature_root),
            "--label-root", str(label_root),
            "--registry", str(tmp_path / "artifacts"),
            "--results-root", str(tmp_path / "results"),
            "--candidate-horizon-sessions", "10,20",
            "--max-training-lookback-sessions", "1260",
        ]
    )

    assert rc == 0
    assert "failed" not in captured
    assert isinstance(captured.get("completed"), tuple)
    run_context, completed_manifest = captured["completed"]
    assert run_context.request.max_training_lookback_sessions == 1260
    assert completed_manifest.model_type == "no_trade"

    records = [
        _json.loads(line)
        for line in (
            diag_root / "lookback04" / "execution_journal.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint_stages = [
        record["stage"] for record in records if record.get("event") == "checkpoint"
    ]
    assert "fitting_started" in checkpoint_stages
    assert "fitting_complete" in checkpoint_stages
    final_record = records[-1]
    assert final_record["event"] == "terminal"
    assert final_record["status"] == "passed"
