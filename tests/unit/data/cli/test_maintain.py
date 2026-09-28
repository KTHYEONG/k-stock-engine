"""Maintain-area CLI commands (verify, prune, datasets, audit, scope info)."""
from __future__ import annotations

from pathlib import Path

import json


from tests.fixtures.cli_fixtures import (
    _LIVE_SUBCOMMANDS,
    _capture_subparsers,
    _dataset_runtime,
    _publish_cli_dataset,
    _dataset_cli_args,
)


def test_ordinary_universe_price_audit_command_emits_report_and_handles_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    import src.data.ordinary_universe_price_audit as audit_mod
    from src.data.cli import main
    from src.data.ordinary_universe_price_audit import OrdinaryUniversePriceAudit
    from src.core.pit import PITDataError

    monkeypatch.setattr(
        audit_mod,
        "audit_ordinary_universe_price_availability",
        lambda **_kwargs: OrdinaryUniversePriceAudit(
            dataset_id="audit-1",
            universe_dataset_id="universe-1",
            sessions=1,
            universe_rows=1,
            eligible_rows=1,
            price_rows=1,
            tradable_rows=1,
            missing_price_rows=0,
            invalid_price_rows=0,
            zero_volume_rows=0,
            report_hash="r" * 64,
        ),
    )
    import src.data.cli as cli_mod

    monkeypatch.setattr(cli_mod.DatasetRegistry, "require", lambda _self, _kind: "universe-1")
    arguments = [
        "audit-ordinary-universe-prices",
        "--scope-config",
        "config/research/kr_swing_2019_v1.toml",
        "--data-root",
        str(tmp_path / "data"),
    ]
    assert main(arguments) == 0
    assert '"dataset_id": "audit-1"' in capsys.readouterr().out

    def fail(**_kwargs):
        raise PITDataError("audit fixture failure")

    monkeypatch.setattr(audit_mod, "audit_ordinary_universe_price_availability", fail)
    assert main(arguments) == 2



def test_cli_registers_only_live_subcommands(monkeypatch, capsys) -> None:
    """Only live subcommands are registered."""
    parsers = _capture_subparsers(monkeypatch, capsys)
    assert set(parsers) == set(_LIVE_SUBCOMMANDS)



def test_cli_has_no_legacy_path_defaults(monkeypatch, capsys) -> None:
    """No legacy path defaults."""
    parsers = _capture_subparsers(monkeypatch, capsys)
    assert parsers
    for name, parser in parsers.items():
        for action in parser._actions:
            default = action.default
            for banned in ("stocks", "data/evidence", "data/artifacts"):
                assert banned not in str(default), f"{name}.{action.dest}={default!r}"



def test_verify_datasets_exits_nonzero_on_tampering(tmp_path, capsys, caplog) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    caplog.set_level("INFO")
    runtime = _dataset_runtime(tmp_path)
    published = _publish_cli_dataset(runtime, "daily_market")
    DatasetRegistry(runtime.workspace.state_root).register("daily_market", published.dataset_id)
    partition = published.path / "part.parquet"
    partition.write_bytes(partition.read_bytes() + b"tampered")

    exit_code = main(_dataset_cli_args("verify-datasets", runtime))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert exit_code == 1
    assert any(line.get("dataset_id") == published.dataset_id and line["status"] == "failed" for line in lines)
    assert lines[-1]["type"] == "summary"
    assert lines[-1]["failed"] == 1
    assert any("[DATA] command=verify_datasets" in message for message in caplog.messages)



def test_verify_datasets_warns_but_does_not_fail_on_stale_lineage(tmp_path, capsys) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    old_input = _publish_cli_dataset(runtime, "ordinary_universe", params={"generation": 1})
    current_input = _publish_cli_dataset(runtime, "ordinary_universe", params={"generation": 2})
    dependent = _publish_cli_dataset(
        runtime,
        "daily_market",
        inputs={"universe": old_input.dataset_id},
    )
    registry = DatasetRegistry(runtime.workspace.state_root)
    registry.register("ordinary_universe", current_input.dataset_id)
    registry.register("daily_market", dependent.dataset_id)

    exit_code = main(_dataset_cli_args("verify-datasets", runtime))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    dependent_line = next(line for line in lines if line.get("dataset_id") == dependent.dataset_id)

    assert exit_code == 0
    assert dependent_line["stale_inputs"] == [
        {
            "role": "universe",
            "input_kind": "ordinary_universe",
            "input_id": old_input.dataset_id,
            "registered_id": current_input.dataset_id,
        }
    ]
    assert lines[-1]["stale"] == 1



def test_verify_datasets_all_includes_unregistered_datasets(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    anchor = _publish_cli_dataset(runtime, "anchor")
    from src.data.dataset_registry import DatasetRegistry
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    published = _publish_cli_dataset(runtime, "unregistered")

    exit_code = main(_dataset_cli_args("verify-datasets", runtime, "--all"))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert exit_code == 0
    assert any(line.get("dataset_id") == published.dataset_id for line in lines)



def test_prune_datasets_keeps_transitive_lineage_and_apply_removes_only_orphan(tmp_path, capsys) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    root_input = _publish_cli_dataset(runtime, "source")
    middle = _publish_cli_dataset(runtime, "middle", inputs={"source": root_input.dataset_id})
    orphan = _publish_cli_dataset(runtime, "orphan")
    DatasetRegistry(runtime.workspace.state_root).register("middle", middle.dataset_id)

    plan_exit = main(_dataset_cli_args("prune-datasets", runtime))
    plan_lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    listed = {line["dataset_id"] for line in plan_lines if line.get("status") == "prunable"}

    assert plan_exit == 0
    assert listed == {orphan.dataset_id}
    assert (runtime.workspace.silver_root / root_input.dataset_id).is_dir()
    assert (runtime.workspace.silver_root / middle.dataset_id).is_dir()

    apply_exit = main(_dataset_cli_args("prune-datasets", runtime, "--apply"))

    assert apply_exit == 0
    assert not (runtime.workspace.silver_root / orphan.dataset_id).exists()
    assert (runtime.workspace.silver_root / root_input.dataset_id).is_dir()
    assert (runtime.workspace.silver_root / middle.dataset_id).is_dir()



def test_prune_datasets_never_deletes_unreadable_manifest(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    state_before = sorted(path.name for path in runtime.workspace.state_root.iterdir())
    unreadable = runtime.workspace.silver_root / "broken_0123456789abcdef"
    unreadable.mkdir()

    exit_code = main(_dataset_cli_args("prune-datasets", runtime, "--apply"))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert exit_code == 1
    assert any(line.get("dataset_id") == unreadable.name and line["status"] == "unreadable_manifest" for line in lines)
    assert unreadable.is_dir()
    assert sorted(path.name for path in runtime.workspace.state_root.iterdir()) == state_before



def test_verify_datasets_accepts_retired_lineage_and_reports_missing_registration(tmp_path, capsys) -> None:
    import json as json_module

    from src.data.cli import main
    from src.data.dataset_registry import REGISTRY_NAME
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    import polars as pl

    runtime = _dataset_runtime(tmp_path)
    retired_id = "ordinary_universe_0123456789abcdef"
    identity = DatasetIdentity(
        kind="daily_market",
        layer=DatasetLayer.SILVER,
        policy_version="cli-fixture-v1",
        inputs={"universe": retired_id},
        params={},
    )
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=identity,
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    registry_path = runtime.workspace.state_root / REGISTRY_NAME
    registry_path.write_text(
        json_module.dumps(
            {
                "current": {"daily_market": published.dataset_id},
                "retired": {retired_id: "superseded upstream generation"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert main(_dataset_cli_args("verify-datasets", runtime)) == 0
    capsys.readouterr()
    registry_path.write_text(
        json_module.dumps(
            {
                "current": {"daily_market": "daily_market_1111111111111111"},
                "retired": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert main(_dataset_cli_args("verify-datasets", runtime)) == 1
    lines = [json_module.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any("missing or ambiguous" in failure for line in lines for failure in line.get("failures", []))



def test_dataset_commands_ignore_missing_layer_root_and_bronze_lineage(tmp_path, capsys) -> None:
    import shutil as shutil_module

    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    source = _publish_cli_dataset(runtime, "source")
    dependent = _publish_cli_dataset(
        runtime,
        "dependent",
        inputs={"bronze": f"bronze:{'b' * 64}", "source": source.dataset_id},
    )
    DatasetRegistry(runtime.workspace.state_root).register("dependent", dependent.dataset_id)
    DatasetRegistry(runtime.workspace.state_root).register("source", source.dataset_id)
    shutil_module.rmtree(runtime.workspace.gold_root)

    assert main(_dataset_cli_args("verify-datasets", runtime)) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    dependent_line = next(line for line in lines if line["dataset_id"] == dependent.dataset_id)
    assert dependent_line["stale_inputs"] == []



def test_verify_datasets_reports_configuration_and_manifest_read_failures(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    published = _publish_cli_dataset(runtime, "daily_market")
    (published.path / "manifest.json").write_text("not-json", encoding="utf-8")
    args = _dataset_cli_args("verify-datasets", runtime)
    registry_path = runtime.workspace.state_root / "datasets.json"
    registry_path.write_text(
        json.dumps(
            {
                "current": {"daily_market": published.dataset_id},
                "retired": {},
            }
        ),
        encoding="utf-8",
    )

    assert main(args) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["status"] == "failed"
    assert lines[0]["stale_inputs"] == []

    invalid_args = [
        "verify-datasets",
        "--scope-config",
        "missing-scope.toml",
        "--data-root",
        str(runtime.workspace.root),
    ]
    assert main(invalid_args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert "error" in summary



def test_prune_datasets_fails_closed_for_registered_unreadable_dataset(tmp_path, capsys) -> None:
    from src.data.cli import main
    from src.data.dataset_registry import DatasetRegistry

    runtime = _dataset_runtime(tmp_path)
    registered = _publish_cli_dataset(runtime, "registered")
    orphan = _publish_cli_dataset(
        runtime,
        "orphan",
        inputs={"bronze": f"bronze:{'c' * 64}"},
    )
    DatasetRegistry(runtime.workspace.state_root).register("registered", registered.dataset_id)
    (registered.path / "manifest.json").write_text("not-json", encoding="utf-8")

    assert main(_dataset_cli_args("prune-datasets", runtime)) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["type"] == "summary"
    assert "current dataset manifest is invalid" in summary["error"]
    assert orphan.path.is_dir()



def test_prune_datasets_reports_configuration_failure(tmp_path, capsys) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    args = [
        "prune-datasets",
        "--scope-config",
        "missing-scope.toml",
        "--data-root",
        str(runtime.workspace.root),
    ]

    assert main(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert "error" in summary



def test_prune_apply_rechecks_candidate_set_before_deletion(tmp_path, capsys, monkeypatch) -> None:
    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    orphan = _publish_cli_dataset(runtime, "orphan")
    from src.data.dataset_maintenance import prune_plan as original_plan
    calls = 0

    def _no_longer_prunable(current_runtime, registry):
        nonlocal calls
        calls += 1
        return original_plan(current_runtime, registry) if calls == 1 else ([], [])

    monkeypatch.setattr("src.data.dataset_maintenance.prune_plan", _no_longer_prunable)
    assert main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["candidates"] == 1
    assert summary["deleted"] == 0
    assert orphan.path.is_dir()



def test_prune_apply_never_deletes_symlink_replacement(tmp_path, capsys, monkeypatch) -> None:
    import shutil

    from src.data.cli import main

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    orphan = _publish_cli_dataset(runtime, "orphan")
    outside = tmp_path / "outside"
    outside.mkdir()
    from src.data.dataset_maintenance import prune_plan as original_plan
    calls = 0

    def _replace_with_symlink(current_runtime, registry):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_plan(current_runtime, registry)
        shutil.rmtree(orphan.path)
        orphan.path.symlink_to(outside, target_is_directory=True)
        return [orphan.path], []

    monkeypatch.setattr("src.data.dataset_maintenance.prune_plan", _replace_with_symlink)
    assert main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 0
    assert orphan.path.is_symlink()
    assert outside.is_dir()



def test_prune_apply_reports_delete_failure(tmp_path, capsys, monkeypatch, caplog) -> None:
    import shutil

    from src.data.cli import main

    caplog.set_level("INFO")
    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    orphan = _publish_cli_dataset(runtime, "orphan")
    monkeypatch.setattr(
        shutil,
        "rmtree",
        lambda path, **_kwargs: (_ for _ in ()).throw(OSError(f"cannot remove {path}")),
    )

    assert main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(line.get("dataset_id") == orphan.dataset_id and line["status"] == "delete_failed" for line in lines)
    assert orphan.path.is_dir()
    assert any("action=keep status=failed" in message for message in caplog.messages)



def test_dataset_cli_directory_manifest_and_current_validation_boundaries(tmp_path, monkeypatch, capsys) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.dataset_maintenance import dataset_directories as _dataset_directories, legacy_lineage as _legacy_lineage, legacy_prunable_manifest as _legacy_prunable_manifest, prune_plan as _prune_plan, validated_current_directories as _validated_current_directories
    from src.data.dataset_registry import REGISTRY_NAME, DatasetRegistry
    import src.data.cli as cli_module

    runtime = _dataset_runtime(tmp_path)
    silver = runtime.workspace.silver_root
    (silver / ".hidden").mkdir()
    (silver / "not-a-directory").write_text("x", encoding="utf-8")
    nested = silver / "legacy-table" / "legacy_0123456789abcdef"
    nested.mkdir(parents=True)
    (nested / "dataset_manifest.json").write_text(
        json.dumps({"inputs": {"upstream": "ordinary_universe_0123456789abcdef"}}),
        encoding="utf-8",
    )
    (nested / "content_manifest.json").write_text(
        json.dumps({"partitions": [{"path": "part.parquet"}]}), encoding="utf-8"
    )
    assert nested in _dataset_directories(runtime)
    assert _legacy_prunable_manifest(nested) is not None

    malformed_root = tmp_path / "legacy-malformed"
    malformed = malformed_root / "legacy_0123456789abcdef"
    malformed.mkdir(parents=True)
    (malformed / "manifest.json").write_text("[]", encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    (malformed / "manifest.json").write_text(json.dumps({"dataset_id": "wrong", "partitions": [{"path": "x"}]}), encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    (malformed / "manifest.json").write_text(json.dumps({"dataset_id": malformed.name, "partitions": []}), encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    (malformed / "manifest.json").write_text(json.dumps({"dataset_id": malformed.name, "partitions": [1]}), encoding="utf-8")
    assert _legacy_prunable_manifest(malformed) is None
    content_root = tmp_path / "legacy-content"
    content_dir = content_root / "legacy_0123456789abcdef"
    content_dir.mkdir(parents=True)
    (content_dir / "dataset_manifest.json").write_text(json.dumps({"inputs": {}}), encoding="utf-8")
    (content_dir / "content_manifest.json").write_text("[]", encoding="utf-8")
    assert _legacy_prunable_manifest(content_dir) is None
    (content_dir / "content_manifest.json").write_text(json.dumps({"partitions": [{"path": "part.parquet"}]}), encoding="utf-8")
    assert _legacy_prunable_manifest(content_dir) is not None
    missing_content = tmp_path / "legacy-missing-content" / "legacy_0123456789abcdef"
    missing_content.mkdir(parents=True)
    (missing_content / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    assert _legacy_prunable_manifest(missing_content) is None

    valid_id = "ordinary_universe_0123456789abcdef"
    other_id = "daily_market_1111111111111111"
    assert _legacy_lineage(
        {
            "inputs": {"a": valid_id, "b": f"bronze:{'a' * 64}"},
            "daily_market_dataset_id": other_id,
            "universe_dataset_id": "not-an-id",
        }
    ) == {valid_id, other_id}

    empty_registry = DatasetRegistry(runtime.workspace.state_root)
    with pytest.raises(PITDataError, match="no current datasets"):
        _validated_current_directories(runtime, empty_registry)
    assert cli_module.main(_dataset_cli_args("verify-datasets", runtime)) == 1
    assert json.loads(capsys.readouterr().out)["failed"] == 1

    missing_id = "daily_market_0123456789abcdef"
    (runtime.workspace.state_root / REGISTRY_NAME).write_text(
        json.dumps({"current": {"daily_market": missing_id}, "retired": {}}), encoding="utf-8"
    )
    with pytest.raises(PITDataError, match="missing or ambiguous"):
        _validated_current_directories(runtime, DatasetRegistry(runtime.workspace.state_root))

    mismatch = _publish_cli_dataset(runtime, "daily_market")
    (runtime.workspace.state_root / REGISTRY_NAME).write_text(
        json.dumps({"current": {"market_panel": mismatch.dataset_id}, "retired": {}}), encoding="utf-8"
    )
    with pytest.raises(PITDataError, match="kind mismatch"):
        _validated_current_directories(runtime, DatasetRegistry(runtime.workspace.state_root))

    class _SnapshotRegistry:
        def snapshot(self):
            return {"market_panel": mismatch.dataset_id}

        def retired(self):
            return {}

    with pytest.raises(PITDataError, match="kind mismatch"):
        _validated_current_directories(runtime, _SnapshotRegistry())  # type: ignore[arg-type]

    failed = _publish_cli_dataset(runtime, "daily_market", params={"case": "failed"})
    registry_path = runtime.workspace.state_root / REGISTRY_NAME
    registry_path.write_text(json.dumps({"current": {"daily_market": failed.dataset_id}, "retired": {}}), encoding="utf-8")
    (failed.path / "part.parquet").write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="failed verification"):
        _validated_current_directories(runtime, DatasetRegistry(runtime.workspace.state_root))

    clean = _publish_cli_dataset(runtime, "anchor")
    registry_path.write_text(json.dumps({"current": {"anchor": clean.dataset_id}, "retired": {}}), encoding="utf-8")
    dependent = _publish_cli_dataset(runtime, "dependent", inputs={"bronze": f"bronze:{'b' * 64}"})
    _prune_plan(runtime, DatasetRegistry(runtime.workspace.state_root))
    assert dependent.path.is_dir()

    legacy_orphan = silver / "legacy-table" / "orphan_0123456789abcdef"
    legacy_orphan.mkdir()
    (legacy_orphan / "dataset_manifest.json").write_text(
        json.dumps({"inputs": {"source": "daily_market_1111111111111111"}}),
        encoding="utf-8",
    )
    (legacy_orphan / "content_manifest.json").write_text(
        json.dumps({"partitions": [{"path": "part.parquet"}]}), encoding="utf-8"
    )
    candidates, unreadable = _prune_plan(runtime, DatasetRegistry(runtime.workspace.state_root))
    assert legacy_orphan in candidates
    assert not unreadable

    broken = silver / "broken_0123456789abcdef"
    broken.mkdir()
    monkeypatch.setattr("src.data.dataset_maintenance.validated_current_directories", lambda *_args: ({"daily_market": broken.name}, set()))
    monkeypatch.setattr("src.data.dataset_maintenance.dataset_directories", lambda _runtime: (broken,))
    result = _prune_plan(runtime, DatasetRegistry(runtime.workspace.state_root))
    assert result[0] == []
    assert result[1][0][0] == broken



def test_cli_scope_commands_and_builder_dispatch_paths(tmp_path, monkeypatch, capsys) -> None:
    import src.data.cli as cli_module
    import src.data.ordinary_universe as ordinary_module
    import src.data.dividend_events as dividend_module

    data_root = tmp_path / "data"
    args = [
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(data_root),
    ]
    assert cli_module.main(["scope-info", *args]) == 0
    scope_payload = json.loads(capsys.readouterr().out)
    assert scope_payload["scope_id"] == "kr_swing_2019_v1"
    assert cli_module.main(["init-workspace", *args]) == 0
    assert json.loads(capsys.readouterr().out)["scope_id"] == "kr_swing_2019_v1"

    runtime = _dataset_runtime(tmp_path)
    monkeypatch.setattr("src.data.cli.build.register_dataset", lambda *_args: None)
    monkeypatch.setattr(ordinary_module, "catalog_master_sessions", lambda _catalog: ())
    monkeypatch.setattr(
        ordinary_module,
        "materialize_ordinary_universe_from_catalog",
        lambda **_kwargs: tmp_path / "ordinary_universe_0123456789abcdef",
    )
    assert cli_module.main(_dataset_cli_args("build-ordinary-universe", runtime)) == 0
    capsys.readouterr()
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(json.dumps(["2024-01-02"]), encoding="utf-8")
    assert cli_module.main(_dataset_cli_args("build-ordinary-universe", runtime, "--sessions", str(sessions_file))) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == 1
    sessions_file.write_text(json.dumps({"session": "2024-01-02"}), encoding="utf-8")
    assert cli_module.main(_dataset_cli_args("build-ordinary-universe", runtime, "--sessions", str(sessions_file))) == 2
    assert "JSON list" in json.loads(capsys.readouterr().out)["error"]

    dividend_calls: list[dict[str, object]] = []

    def _fake_dividends(**kwargs: object) -> Path:
        dividend_calls.append(kwargs)
        return tmp_path / "dividend_events_0123456789abcdef"

    monkeypatch.setattr(dividend_module, "materialize_dividend_events", _fake_dividends)
    # 시세 데이터셋이 없으면 배당 타당성 게이트를 켤 수 없으므로 게시하지 않고 실패한다.
    assert cli_module.main(_dataset_cli_args("build-dividend-events", runtime)) == 2
    capsys.readouterr()
    assert dividend_calls == []
    from src.data.dataset_registry import DatasetRegistry

    daily = _publish_cli_dataset(runtime, "daily_market")
    DatasetRegistry(runtime.workspace.state_root).register("daily_market", daily.dataset_id)
    assert cli_module.main(_dataset_cli_args("build-dividend-events", runtime)) == 0
    assert json.loads(capsys.readouterr().out)["dataset_id"] == "dividend_events_0123456789abcdef"
    assert dividend_calls[0]["daily_market_path"] == runtime.workspace.silver_root / daily.dataset_id
    assert dividend_calls[0]["policy"] is not None



def test_cli_parse_and_prune_unreadable_apply_boundaries(tmp_path, monkeypatch, capsys) -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.cli import _parse_decision_time
    import src.data.cli as cli_module

    with pytest.raises(PITDataError, match="timezone-aware"):
        _parse_decision_time("2024-01-01T00:00:00")

    runtime = _dataset_runtime(tmp_path)
    from src.data.dataset_registry import DatasetRegistry
    anchor = _publish_cli_dataset(runtime, "anchor")
    DatasetRegistry(runtime.workspace.state_root).register("anchor", anchor.dataset_id)
    broken = runtime.workspace.silver_root / "broken_0123456789abcdef"
    broken.mkdir()
    monkeypatch.setattr("src.data.dataset_maintenance.prune_plan", lambda *_args: ([broken], []))
    assert cli_module.main(_dataset_cli_args("prune-datasets", runtime, "--apply")) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(line.get("status") == "delete_failed" for line in lines)
