"""Terminal memory guard: durable failure evidence for direct training runs."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
from src.core.instruments import AssetKind
from src.stocks.cli import train
from src.stocks.ml.replay_resources import (
    TrainingRunDeniedError,
    TrainingRunGuard,
)
from src.storage.parquet_datasets import (
    ParquetDatasetStore,
    canonical_content_hash,
)

MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _reset_fake_ledger():
    _FakeCompletedLedger.completed_calls = 0
    return


class _RecordingSink:
    """Minimal RunDiagnostics double capturing emitted events."""

    def __init__(self) -> None:
        self.events: list[object] = []
        self.closed_status: str | None = None

    def emit(self, event: object) -> None:
        self.events.append(event)

    def close(self, status: str) -> None:
        self.closed_status = status


def _guard_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cgroup_limit: int | None,
    cgroup_current: int | None,
    mem_available: int | None,
    rss: int | None = None,
) -> None:
    from src.stocks.ml import replay_resources as rr

    monkeypatch.setattr(rr, "read_cgroup_limit_bytes", lambda root=None: cgroup_limit)
    monkeypatch.setattr(
        rr, "read_cgroup_current_bytes", lambda root=None: cgroup_current
    )
    monkeypatch.setattr(rr, "read_host_mem_available_bytes", lambda: mem_available)
    if rss is not None:
        monkeypatch.setattr(rr, "_current_process_rss_bytes", lambda: rss)


FULL_TERMINAL_02 = "FULL_TERMINAL_02_GUARD_DURABLE_FAILURE"


def test_full_terminal_02_guard_boundary_records_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FULL_TERMINAL_02: every boundary records RSS, cgroup current,
    MemAvailable, live owner names, planned bytes, and verdict as scalars."""
    _guard_env(
        monkeypatch,
        cgroup_limit=8192 * MIB,
        cgroup_current=1024 * MIB,
        mem_available=16 * 1024 * MIB,
        rss=512 * MIB,
    )
    sink = _RecordingSink()
    guard = TrainingRunGuard(
        request_limit_bytes=4096 * MIB,
        reserve_bytes=256 * MIB,
        diagnostics=sink,
        run_id="guard_unit",
    )

    envelope = guard.boundary(
        "direct_load", planned_bytes=1000 * MIB, live_owners=("decision_frame",)
    )

    assert envelope.ok
    assert not guard.denied
    record = guard.records[-1]
    assert record.verdict == "ok"
    assert record.current_rss_bytes == 512 * MIB
    assert record.cgroup_current_bytes == 1024 * MIB
    assert record.mem_available_bytes == 16 * 1024 * MIB
    assert record.live_owners == ("decision_frame",)
    assert record.planned_bytes == 1000 * MIB
    payload = sink.events[-1].payload
    assert payload["verdict"] == "ok"
    assert payload["planned_bytes"] == 1000 * MIB
    assert payload["current_rss_bytes"] == 512 * MIB
    assert payload["cgroup_current_bytes"] == 1024 * MIB
    assert payload["mem_available_bytes"] == 16 * 1024 * MIB
    assert payload["live_owners"] == "decision_frame"


def test_full_terminal_02_guard_denial_is_typed_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planned allocation larger than any supplied headroom fails closed."""
    _guard_env(
        monkeypatch,
        cgroup_limit=1024 * MIB,
        cgroup_current=768 * MIB,
        mem_available=800 * MIB,
        rss=900 * MIB,
    )
    sink = _RecordingSink()
    guard = TrainingRunGuard(
        request_limit_bytes=1000 * MIB,
        reserve_bytes=0,
        diagnostics=sink,
        run_id="guard_unit",
    )

    with pytest.raises(TrainingRunDeniedError) as excinfo:
        guard.boundary(
            "matrix_preparation",
            planned_bytes=5000 * MIB,
            live_owners=("learner_matrix",),
        )

    assert excinfo.value.stage == "matrix_preparation"
    assert guard.denied
    assert guard.records[-1].verdict == "denied"
    assert str(excinfo.value.planned_bytes) == str(5000 * MIB)
    assert sink.events[-1].status.value == "FAIL"
    assert sink.events[-1].payload["verdict"] == "denied"


def _write_fixture_dataset(
    store: ParquetDatasetStore,
    dataset_id: str,
    frame: pl.DataFrame,
    *,
    feature_set: str,
) -> None:
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=list(frame.columns),
        feature_set=feature_set,
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 31, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        generated_time=datetime.now(UTC),
        schema_version="v2",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    manifest = replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))
    store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=feature_set,
        decision_time=datetime(2024, 3, 31, tzinfo=UTC),
    )


def _build_reduced_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    for root in (base_root, feature_root, label_root):
        root.mkdir()
    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
    base_rows = []
    feature_rows = []
    label_rows = []
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
            })
            label_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "horizon_sessions": 10,
                "net_alpha_target": 0.001 * t,
                "label_available_time": session + timedelta(days=10),
            })
    _write_fixture_dataset(base_store, "base_g", pl.DataFrame(base_rows), feature_set="base_panel")
    _write_fixture_dataset(feature_store, "feat_g", pl.DataFrame(feature_rows), feature_set="stock_net_alpha_v1")
    _write_fixture_dataset(label_store, "lab_g", pl.DataFrame(label_rows), feature_set="labels")
    return base_root, feature_root, label_root


def test_full_terminal_02_cli_durable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FULL_TERMINAL_02: a guard denial under a tiny budget writes one terminal
    failure record, closes diagnostics FAIL, and returns non-zero without ever
    invoking training."""
    base_root, feature_root, label_root = _build_reduced_fixture(tmp_path)
    diag_root = tmp_path / "diag"
    diag_root.mkdir()

    invoked: list[bool] = []

    def _forbidden_train(data: object, registry: object, request: object):
        invoked.append(True)
        raise AssertionError("training must not run after a guard denial")

    monkeypatch.setattr(train, "train_net_alpha_model", _forbidden_train)
    monkeypatch.setattr("src.core.paths.RUN_DIAGNOSTIC_ROOT", diag_root)

    rc = train.main(
        [
            "--artifact-id", "guarda1",
            "--base-dataset-id", "base_g",
            "--feature-dataset-id", "feat_g",
            "--label-dataset-id", "lab_g",
            "--research-start-direct", "2024-01-01",
            "--research-end-direct", "2024-01-05",
            "--base-root", str(base_root),
            "--feature-root", str(feature_root),
            "--label-root", str(label_root),
            "--registry", str(tmp_path / "artifacts"),
            "--results-root", str(tmp_path / "results"),
            "--max-rss-mib", "1",
        ]
    )

    assert rc == 1
    assert not invoked

    latest_path = tmp_path / "results" / "ml_runs" / "latest.json"
    assert latest_path.exists(), "terminal failed ledger record must be durable"
    record = json.loads(latest_path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["artifact_id"] == "guarda1"
    assert record["failure"]["phase"] == "training_run_guard:direct_load"
    assert "exceeds" in record["failure"]["message"].lower()

    manifest_path = diag_root / "guarda1" / "manifest.json"
    assert manifest_path.exists(), "diagnostics must be closed with terminal evidence"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "FAIL"

    # Exactly one terminal failure record was written.
    recent_lines = (
        tmp_path / "results" / "ml_runs" / "recent.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    failed_records = [
        line for line in recent_lines if json.loads(line)["status"] == "failed"
    ]
    assert len(failed_records) == 1


def test_full_terminal_02_unbounded_budget_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a finite budget the same fixture trains normally (exit 0)."""
    base_root, feature_root, label_root = _build_reduced_fixture(tmp_path)
    diag_root = tmp_path / "diag_ok"
    diag_root.mkdir()

    def _fake_train(data: object, registry: object, request: object):
        from src.stocks.research.models import ModelManifest

        return ModelManifest(
            artifact_id=request.artifact_id,
            asset_kind="stock",
            feature_set="stock_net_alpha_v1",
            feature_schema_hash="h",
            universe_policy_hash="u",
            label_definition="net_alpha_o2o",
            label_horizon_sessions=10,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-03-31T00:00:00+00:00",
            model_type="no_trade",
        )

    monkeypatch.setattr(train, "train_net_alpha_model", _fake_train)
    monkeypatch.setattr("src.core.paths.RUN_DIAGNOSTIC_ROOT", diag_root)
    monkeypatch.setattr(
        train,
        "MlResultLedger",
        _FakeCompletedLedger,
    )

    rc = train.main(
        [
            "--artifact-id", "guarda2",
            "--base-dataset-id", "base_g",
            "--feature-dataset-id", "feat_g",
            "--label-dataset-id", "lab_g",
            "--research-start-direct", "2024-01-01",
            "--research-end-direct", "2024-01-05",
            "--base-root", str(base_root),
            "--feature-root", str(feature_root),
            "--label-root", str(label_root),
            "--registry", str(tmp_path / "artifacts"),
            "--results-root", str(tmp_path / "results"),
            "--candidate-horizon-sessions", "10",
        ]
    )

    assert rc == 0
    assert _FakeCompletedLedger.completed_calls == 1
    manifest_path = diag_root / "guarda2" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"


class _FakeCompletedLedger:
    completed_calls = 0

    def __init__(self, results_root: Path) -> None:
        del results_root

    def record_completed(self, context, manifest, registry, telemetry=None):
        type(self).completed_calls += 1

    def record_failed(self, context, phase, exc, telemetry=None):  # pragma: no cover
        raise AssertionError("unexpected failure record")


TERMINAL_OBS_03 = "TERMINAL_OBS_03_CGROUP_SUBPATH_METRICS"


def test_terminal_obs_03_cgroup_subpath_metrics(tmp_path: Path) -> None:
    """TERMINAL_OBS_03_CGROUP_SUBPATH_METRICS.

    A /proc/self/cgroup mapping to /init.scope resolves the subpath under the
    cgroup mount and reports fixture current/peak/max and oom_kill values
    exactly; unavailable files read as null instead of fabricated zeros.
    """
    from src.stocks.cli.train import resolve_process_cgroup_root, sample_process_cgroup_memory

    proc_cgroup = tmp_path / "self_cgroup"
    mount = tmp_path / "cgroup"
    scope = mount / "init.scope"
    scope.mkdir(parents=True)
    proc_cgroup.write_text(
        "12:pids:/user.slice/user-1000.slice/user@1000.service\n"
        "0::/init.scope\n",
        encoding="utf-8",
    )
    (scope / "memory.current").write_text("1024\n", encoding="utf-8")
    (scope / "memory.peak").write_text("2048\n", encoding="utf-8")
    (scope / "memory.max").write_text("4096\n", encoding="utf-8")
    (scope / "memory.events").write_text(
        "low 0\nhigh 1\nmax 2\noom_kill 3\n", encoding="utf-8"
    )

    resolved = resolve_process_cgroup_root(
        proc_cgroup_path=proc_cgroup, cgroup_mount=mount
    )
    assert resolved == scope

    sample = sample_process_cgroup_memory(resolved)
    assert sample.current_bytes == 1024
    assert sample.peak_bytes == 2048
    assert sample.limit_bytes == 4096
    assert sample.oom_kill_count == 3


def test_terminal_obs_03_cgroup_unavailable_values_are_null(tmp_path: Path) -> None:
    """Missing/unlimited cgroup scalars stay null; v1 memory line is a fallback."""
    from src.stocks.cli.train import resolve_process_cgroup_root, sample_process_cgroup_memory

    mount = tmp_path / "mount"
    group = mount / "memory" / "system.slice"
    group.mkdir(parents=True)
    (group / "memory.usage_in_bytes").write_text("512\n", encoding="utf-8")
    (group / "memory.limit_in_bytes").write_text("max\n", encoding="utf-8")
    proc_cgroup = tmp_path / "v1_cgroup"
    proc_cgroup.write_text("4:memory:/system.slice\n", encoding="utf-8")

    resolved = resolve_process_cgroup_root(
        proc_cgroup_path=proc_cgroup, cgroup_mount=mount
    )
    assert resolved == group
    sample = sample_process_cgroup_memory(resolved)
    assert sample.current_bytes == 512
    assert sample.limit_bytes is None
    assert sample.peak_bytes is None
    assert sample.oom_kill_count is None

    missing = resolve_process_cgroup_root(
        proc_cgroup_path=tmp_path / "absent", cgroup_mount=mount
    )
    assert missing is None
