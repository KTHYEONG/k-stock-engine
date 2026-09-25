"""Identity-bound dataset publication and fail-closed verification tests."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.data.datasets import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    DatasetCheck,
    DatasetIdentity,
    DatasetLayer,
    dataset_id_for,
    dataset_kind_from_id,
    load_manifest,
    publish_dataset,
    read_dataset,
    universe_sessions,
    verify_dataset,
)

_UPSTREAM = "daily_market_0123456789abcdef"
_BRONZE = f"bronze:{'a' * 64}"


def _identity(
    *,
    kind: str = "market_panel",
    layer: DatasetLayer = DatasetLayer.GOLD,
    inputs: dict[str, str] | None = None,
    params: dict[str, str | int | float | bool | None] | None = None,
) -> DatasetIdentity:
    return DatasetIdentity(
        kind=kind,
        layer=layer,
        policy_version="fixture-v1",
        inputs=inputs or {},
        params=params or {},
    )


def _frame(value: int = 1) -> pl.DataFrame:
    return pl.DataFrame({"instrument_id": [f"KRX:{value:06d}"], "session": [date(2024, 1, 2)]})


def _file_state(dataset_dir: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(dataset_dir)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file()
    }


def _valid_manifest(published) -> dict[str, object]:
    return json.loads((published.path / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_raw_manifest(published, raw: object) -> None:
    (published.path / MANIFEST_NAME).write_text(json.dumps(raw), encoding="utf-8")


def test_dataset_id_is_stable_and_identity_only(tmp_path: Path) -> None:
    identity = _identity(params={"threshold": 0.1, "enabled": True})
    reordered = _identity(params={"enabled": True, "threshold": 0.1})
    first = publish_dataset(
        layer_root=tmp_path / "gold",
        identity=identity,
        partitions={"part.parquet": _frame()},
        details={"counter": 1},
    )
    second = publish_dataset(
        layer_root=tmp_path / "other",
        identity=reordered,
        partitions={"part.parquet": _frame()},
        details={"counter": 2},
    )

    assert first.dataset_id == second.dataset_id
    assert dataset_id_for(replace(identity, params={"threshold": 0.2, "enabled": True})) != first.dataset_id
    assert dataset_id_for(replace(identity, inputs={"upstream": _UPSTREAM})) != first.dataset_id
    assert dataset_kind_from_id(first.dataset_id) == "market_panel"
    with pytest.raises(PITDataError, match="invalid dataset id"):
        dataset_kind_from_id("not-a-dataset")


def test_dataset_id_canonicalizes_timezone_aware_datetimes() -> None:
    decision_time = datetime(2024, 1, 2, 9, tzinfo=UTC)
    first = _identity(params=cast("dict[str, str | int | float | bool | None]", {"at": decision_time}))
    second = _identity(params=cast("dict[str, str | int | float | bool | None]", {"at": decision_time}))

    assert dataset_id_for(first) == dataset_id_for(second)
    with pytest.raises(PITDataError, match="require an offset"):
        _identity(params=cast("dict[str, str | int | float | bool | None]", {"at": datetime(2024, 1, 2, 9)}))


def test_deterministic_republish_is_byte_and_mtime_noop(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market", params={"window": 20}),
        partitions={"year=2024/part.parquet": _frame()},
    )
    before = _file_state(published.path)

    repeated = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market", params={"window": 20}),
        partitions={"year=2024/part.parquet": _frame()},
    )

    assert repeated == published
    assert _file_state(published.path) == before


def test_non_deterministic_republish_preserves_original(tmp_path: Path) -> None:
    identity = _identity(kind="daily_market")
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=identity,
        partitions={"part.parquet": _frame(1)},
    )
    before = _file_state(published.path)

    with pytest.raises(PITDataError, match="non-deterministic rebuild"):
        publish_dataset(
            layer_root=tmp_path / "silver",
            identity=identity,
            partitions={"part.parquet": _frame(2)},
        )

    assert _file_state(published.path) == before


def test_tampered_partition_is_named_and_read_fails_closed(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market"),
        partitions={"year=2024/part.parquet": _frame()},
    )
    partition = published.path / "year=2024/part.parquet"
    partition.write_bytes(partition.read_bytes() + b"tampered")

    verification = verify_dataset(published.path, known_ids=lambda _dataset_id: True)

    assert not verification.passed
    assert any("year=2024/part.parquet" in failure for failure in verification.failures)
    with pytest.raises(PITDataError, match="dataset verification failed"):
        read_dataset(published.path)


def test_missing_input_fails_unless_retired_is_known(tmp_path: Path) -> None:
    identity = _identity(kind="daily_market", inputs={"upstream": _UPSTREAM})
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=identity,
        partitions={"part.parquet": _frame()},
    )

    missing = verify_dataset(published.path, known_ids=lambda _dataset_id: False)
    retired = verify_dataset(published.path, known_ids=lambda dataset_id: dataset_id in {_UPSTREAM})

    assert any(_UPSTREAM in failure for failure in missing.failures)
    assert retired.passed


def test_failed_check_fails_verification(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "gold",
        identity=_identity(),
        partitions={"part.parquet": _frame()},
        checks=[DatasetCheck(name="identity_violations", value=1.0, limit=0.0, passed=False)],
    )

    verification = verify_dataset(published.path, known_ids=lambda _dataset_id: True)

    assert not verification.passed
    assert verification.failed_checks[0].name == "identity_violations"
    assert "quality check failed" in verification.failures[0]


def test_partition_path_escape_writes_nothing(tmp_path: Path) -> None:
    layer_root = tmp_path / "silver"

    with pytest.raises(PITDataError, match="partition path"):
        publish_dataset(
            layer_root=layer_root,
            identity=_identity(kind="daily_market"),
            partitions={"../x.parquet": _frame()},
        )

    assert not (tmp_path / "x.parquet").exists()
    assert not any(layer_root.iterdir())


def test_bronze_digest_does_not_require_lineage_lookup(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", inputs={"bronze": _BRONZE}),
        partitions={"part.parquet": _frame()},
    )

    def _unexpected_lookup(_dataset_id: str) -> bool:
        raise AssertionError("Bronze digest was treated as dataset lineage")

    assert verify_dataset(published.path, known_ids=_unexpected_lookup).passed


def test_zero_row_partition_is_hashed_and_readable(tmp_path: Path) -> None:
    empty = pl.DataFrame({"instrument_id": pl.Series([], dtype=pl.String), "value": pl.Series([], dtype=pl.Int64)})
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market"),
        partitions={"empty.parquet": empty},
    )
    manifest = load_manifest(published.path)

    assert published.rows == 0
    assert manifest.partitions[0].rows == 0
    assert len(manifest.partitions[0].sha256) == 64
    assert read_dataset(published.path, columns=["value"]).collect().height == 0


def test_manifest_identity_can_be_tampered(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "gold",
        identity=_identity(),
        partitions={"part.parquet": _frame()},
    )
    manifest_path = published.path / MANIFEST_NAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["policy_version"] = "tampered"
    manifest_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")

    verification = verify_dataset(published.path, known_ids=lambda _dataset_id: True)

    assert not verification.passed
    assert "does not match identity" in verification.failures[0]


def test_unreadable_manifest_is_reported_as_verification_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "market_panel_0123456789abcdef"
    dataset.mkdir()
    (dataset / MANIFEST_NAME).write_text("not-json", encoding="utf-8")

    verification = verify_dataset(dataset, known_ids=lambda _dataset_id: True)

    assert not verification.passed
    assert "unreadable dataset manifest" in verification.failures[0]


def test_dataset_identity_rejects_invalid_kind_and_input(tmp_path: Path) -> None:
    with pytest.raises(PITDataError, match="path-safe"):
        DatasetIdentity("../bad", DatasetLayer.SILVER, "v1", {}, {})
    with pytest.raises(PITDataError, match="invalid dataset input"):
        DatasetIdentity("daily_market", DatasetLayer.SILVER, "v1", {"upstream": "bad"}, {})
    with pytest.raises(PITDataError, match="unsupported dataset parameter"):
        DatasetIdentity("daily_market", DatasetLayer.SILVER, "v1", {}, {"bad": object()})


def test_publish_rejects_an_invalid_existing_manifest(tmp_path: Path) -> None:
    identity = _identity()
    published = publish_dataset(
        layer_root=tmp_path / "gold",
        identity=identity,
        partitions={"part.parquet": _frame()},
    )
    (published.path / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(PITDataError, match="existing dataset has an invalid manifest"):
        publish_dataset(
            layer_root=tmp_path / "gold",
            identity=identity,
            partitions={"part.parquet": _frame()},
        )


def test_universe_sessions_reads_v2_and_legacy_manifests(tmp_path: Path) -> None:
    v2 = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER),
        partitions={
            "session=2024-01-02/part.parquet": pl.DataFrame({"session": [date(2024, 1, 2)]}),
            "session=2024-01-03/part.parquet": pl.DataFrame({"session": [date(2024, 1, 3)]}),
        },
    )
    assert universe_sessions(tmp_path / "silver", v2.dataset_id) == (
        v2.dataset_id,
        (date(2024, 1, 2), date(2024, 1, 3)),
    )

    legacy_id = "ordinary_universe_1111111111111111"
    legacy = tmp_path / "silver" / legacy_id
    legacy.mkdir()
    (legacy / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "dataset_id": legacy_id,
                "partitions": [{"session": "2024-01-02"}, {"session": "2024-01-03"}],
            }
        ),
        encoding="utf-8",
    )
    assert universe_sessions(tmp_path / "silver", legacy_id) == (
        legacy_id,
        (date(2024, 1, 2), date(2024, 1, 3)),
    )


def test_manifest_schema_is_v2_and_json_is_readable(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "gold",
        identity=_identity(),
        partitions={"part.parquet": _frame()},
        details={"counter": 3},
    )
    raw = json.loads((published.path / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = load_manifest(published.path)

    assert raw["schema"] == MANIFEST_SCHEMA
    assert manifest.details == {"counter": 3}
    assert manifest.created_at.utcoffset() is not None


def test_identity_rejects_invalid_runtime_shapes() -> None:
    with pytest.raises(PITDataError, match="DatasetLayer"):
        DatasetIdentity("daily_market", cast("DatasetLayer", "silver"), "v1", {}, {})
    with pytest.raises(PITDataError, match="input roles"):
        DatasetIdentity("daily_market", DatasetLayer.SILVER, "v1", {"": _UPSTREAM}, {})
    with pytest.raises(PITDataError, match="parameter names"):
        DatasetIdentity("daily_market", DatasetLayer.SILVER, "v1", {}, {"": 1})
    with pytest.raises(PITDataError, match="finite"):
        DatasetIdentity("daily_market", DatasetLayer.SILVER, "v1", {}, {"threshold": float("nan")})


def test_manifest_loader_rejects_each_invalid_v2_field(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "gold",
        identity=_identity(),
        partitions={"part.parquet": _frame()},
        checks=[DatasetCheck(name="rows", value=1.0, limit=1.0, passed=True)],
    )
    valid = _valid_manifest(published)

    def _manifest_with(**updates: object) -> dict[str, object]:
        raw = json.loads(json.dumps(valid))
        raw.update(updates)
        return raw

    def _with_partition_rows(rows: int) -> dict[str, object]:
        raw = json.loads(json.dumps(valid))
        raw["partitions"][0]["rows"] = rows
        raw["rows"] = rows
        return raw

    def _with_check(**updates: object) -> dict[str, object]:
        raw = json.loads(json.dumps(valid))
        raw["checks"][0].update(updates)
        return raw

    duplicate = json.loads(json.dumps(valid))
    duplicate["partitions"].append(json.loads(json.dumps(duplicate["partitions"][0])))
    cases = [
        (_manifest_with(schema="legacy-v1"), "unsupported dataset manifest schema"),
        (_manifest_with(dataset_id="invalid"), "invalid dataset id"),
        (_manifest_with(inputs=[]), "field 'inputs' must be an object"),
        (_manifest_with(partitions={}), "partitions must be a list"),
        (_manifest_with(partitions=[1]), "partition must be an object"),
        (duplicate, "duplicate dataset partition path"),
        (_with_partition_rows(-1), "partition rows must be a non-negative integer"),
        (_manifest_with(rows=2), "row total does not match partitions"),
        (_manifest_with(checks={}), "checks must be a list"),
        (_manifest_with(checks=[1]), "check must be an object"),
        (_with_check(value=True), "value must be numeric"),
        (_with_check(limit="1"), "limit must be numeric"),
        (_with_check(passed=1), "passed must be boolean"),
        (_manifest_with(created_at=1), "created_at must be an ISO-8601 string"),
        (_manifest_with(created_at="not-a-time"), "created_at must be an ISO-8601 string"),
        (_manifest_with(created_at="2024-01-02T09:00:00"), "created_at requires an offset"),
        (_manifest_with(layer="bronze"), "invalid dataset manifest"),
    ]
    for raw, message in cases:
        _write_raw_manifest(published, raw)
        with pytest.raises(PITDataError, match=message):
            load_manifest(published.path)

    invalid_sha = json.loads(json.dumps(valid))
    invalid_sha["partitions"][0]["sha256"] = "bad"
    _write_raw_manifest(published, invalid_sha)
    with pytest.raises(PITDataError, match="invalid partition sha256"):
        load_manifest(published.path)

    _write_raw_manifest(published, [])
    with pytest.raises(PITDataError, match="invalid dataset manifest"):
        load_manifest(published.path)


def test_manifest_loader_rejects_directory_mismatch_and_symlink_escape(tmp_path: Path) -> None:
    import shutil

    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={"nested/part.parquet": _frame()},
    )
    moved = published.path.with_name("market_panel_0123456789abcdef")
    shutil.copytree(published.path, moved)
    with pytest.raises(PITDataError, match="does not match directory"):
        load_manifest(moved)

    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(published.path / "nested")
    (published.path / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PITDataError, match="escapes its directory"):
        load_manifest(published.path)


def test_verifier_reports_missing_unreadable_and_wrong_row_partitions(tmp_path: Path, monkeypatch) -> None:
    missing = publish_dataset(
        layer_root=tmp_path / "missing",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={"part.parquet": _frame()},
    )
    (missing.path / "part.parquet").unlink()
    result = verify_dataset(missing.path, known_ids=lambda _dataset_id: True)
    assert any("missing dataset partition" in failure for failure in result.failures)

    unreadable = publish_dataset(
        layer_root=tmp_path / "unreadable",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={"part.parquet": _frame()},
    )
    partition = unreadable.path / "part.parquet"
    partition.write_bytes(b"not parquet")
    raw = _valid_manifest(unreadable)
    raw["partitions"][0]["sha256"] = __import__("hashlib").sha256(partition.read_bytes()).hexdigest()
    _write_raw_manifest(unreadable, raw)
    result = verify_dataset(unreadable.path, known_ids=lambda _dataset_id: True)
    assert any("unreadable dataset partition" in failure for failure in result.failures)

    wrong_rows = publish_dataset(
        layer_root=tmp_path / "wrong_rows",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={"part.parquet": _frame()},
    )
    raw = _valid_manifest(wrong_rows)
    raw["partitions"][0]["rows"] = 2
    raw["rows"] = 2
    _write_raw_manifest(wrong_rows, raw)
    result = verify_dataset(wrong_rows.path, known_ids=lambda _dataset_id: True)
    assert any("partition row mismatch" in failure for failure in result.failures)

    import src.data.datasets as dataset_module

    io_failure = publish_dataset(
        layer_root=tmp_path / "io_failure",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={"part.parquet": _frame()},
    )
    monkeypatch.setattr(
        dataset_module,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(OSError("hash read")),
    )
    result = verify_dataset(io_failure.path, known_ids=lambda _dataset_id: True)
    assert any("unreadable dataset partition" in failure for failure in result.failures)


def test_publish_write_failures_are_clean_and_atomic(tmp_path: Path, monkeypatch) -> None:
    identity = _identity(kind="daily_market", layer=DatasetLayer.SILVER)
    layer_root = tmp_path / "silver"
    monkeypatch.setattr(pl.DataFrame, "write_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(PITDataError, match="cannot write dataset partition"):
        publish_dataset(layer_root=layer_root, identity=identity, partitions={"part.parquet": _frame()})
    assert not any(layer_root.iterdir())

    monkeypatch.undo()
    import src.data.datasets as dataset_module

    monkeypatch.setattr(dataset_module.os, "rename", lambda *_args: (_ for _ in ()).throw(OSError("rename")))
    with pytest.raises(PITDataError, match="cannot publish dataset"):
        publish_dataset(layer_root=layer_root, identity=identity, partitions={"part.parquet": _frame()})
    assert not any(layer_root.iterdir())

    monkeypatch.undo()
    blocked_root = tmp_path / "blocked-root"
    blocked_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PITDataError, match="cannot stage or publish dataset"):
        publish_dataset(layer_root=blocked_root, identity=identity, partitions={"part.parquet": _frame()})


def test_publish_handles_concurrent_atomic_rename_as_identical_republish(tmp_path: Path, monkeypatch) -> None:
    import src.data.datasets as dataset_module

    original_rename = dataset_module.os.rename
    calls = 0

    def _rename_then_fail(source, target) -> None:
        nonlocal calls
        calls += 1
        original_rename(source, target)
        raise OSError("simulated rename notification failure")

    monkeypatch.setattr(dataset_module.os, "rename", _rename_then_fail)
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={"part.parquet": _frame()},
    )

    assert calls == 1
    assert published.path.is_dir()


def test_empty_partition_set_reads_as_empty_lazy_frame(tmp_path: Path) -> None:
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
        partitions={},
    )

    assert read_dataset(published.path).collect().height == 0


def test_invalid_partition_names_and_manifest_details_fail_cleanly(tmp_path: Path) -> None:
    for path in ("", "part\\file.parquet", "/absolute.parquet", "a//part.parquet", MANIFEST_NAME):
        with pytest.raises(PITDataError, match="partition path"):
            publish_dataset(
                layer_root=tmp_path / "silver",
                identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER),
                partitions={path: _frame()},
            )
    with pytest.raises(PITDataError, match="cannot write dataset manifest"):
        publish_dataset(
            layer_root=tmp_path / "silver",
            identity=_identity(),
            partitions={"part.parquet": _frame()},
            details={"invalid": float("nan")},
        )


def test_universe_sessions_rejects_unreadable_and_non_ordered_manifests(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    missing_id = "ordinary_universe_1111111111111111"
    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        universe_sessions(silver, missing_id)

    dataset = silver / missing_id
    dataset.mkdir(parents=True)
    manifest_path = dataset / MANIFEST_NAME
    manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        universe_sessions(silver, missing_id)

    manifest_path.write_text(json.dumps({"dataset_id": missing_id, "partitions": []}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        universe_sessions(silver, missing_id)

    manifest_path.write_text(json.dumps({"dataset_id": missing_id, "partitions": [1]}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid ordinary-universe partition"):
        universe_sessions(silver, missing_id)

    manifest_path.write_text(
        json.dumps({"dataset_id": missing_id, "partitions": [{"session": "bad"}]}),
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="partition session"):
        universe_sessions(silver, missing_id)

    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": missing_id,
                "partitions": [{"session": "2024-01-03"}, {"session": "2024-01-02"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="strictly ordered"):
        universe_sessions(silver, missing_id)


def test_dataset_compatibility_readers_and_digest_boundaries(tmp_path: Path, monkeypatch) -> None:
    import src.data.datasets as dataset_module
    from src.data.datasets import (
        dataset_partition_paths,
        dataset_reference,
        normalize_bronze_digest,
        read_dataset_compat,
        resolve_bronze_digest,
        _legacy_partition_sessions,
    )

    with pytest.raises(PITDataError, match="kind mismatch"):
        dataset_reference(_UPSTREAM, kind="market_panel")
    with pytest.raises(PITDataError, match="invalid dataset reference"):
        dataset_reference("", kind="market_panel")

    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market"),
        partitions={"part.parquet": _frame()},
    )
    assert read_dataset(published.path, columns=["missing"]).collect().columns == ["missing"]
    monkeypatch.setattr(dataset_module.pl, "read_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")))
    with pytest.raises(PITDataError, match="unreadable dataset partition"):
        read_dataset(published.path)
    monkeypatch.undo()

    legacy_id = "daily_market_1111111111111111"
    legacy = tmp_path / "legacy" / legacy_id
    legacy.mkdir(parents=True)
    manifest = legacy / MANIFEST_NAME
    manifest.write_text(json.dumps({"dataset_id": legacy_id, "partitions": {}}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid dataset partitions"):
        dataset_partition_paths(legacy, allow_legacy=True)
    manifest.write_text(json.dumps({"dataset_id": legacy_id, "partitions": [1]}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid dataset partition"):
        dataset_partition_paths(legacy, allow_legacy=True)
    manifest.write_text(json.dumps({"dataset_id": legacy_id, "partitions": [{"path": "x", "sha256": "bad"}]}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid dataset partition metadata"):
        dataset_partition_paths(legacy, allow_legacy=True)
    part = legacy / "x"
    pl.DataFrame({"value": [1]}).write_parquet(part)
    digest = __import__("hashlib").sha256(part.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps({"dataset_id": legacy_id, "partitions": [{"path": "x", "sha256": digest}, {"path": "x", "sha256": digest}]}),
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="duplicate dataset partition"):
        dataset_partition_paths(legacy, allow_legacy=True)
    manifest.write_text(json.dumps({"dataset_id": legacy_id, "partitions": [{"path": "x", "sha256": digest}]}), encoding="utf-8")
    assert read_dataset_compat(legacy).collect().height == 1
    assert read_dataset_compat(legacy, columns=["value"]).collect().height == 1
    empty_legacy = tmp_path / "legacy-empty" / legacy_id
    empty_legacy.mkdir(parents=True)
    (empty_legacy / MANIFEST_NAME).write_text(json.dumps({"dataset_id": legacy_id, "partitions": []}), encoding="utf-8")
    assert read_dataset_compat(empty_legacy).collect().height == 0

    with pytest.raises(PITDataError, match="does not match"):
        resolve_bronze_digest("a" * 64, ["b"])
    assert resolve_bronze_digest(None, ["b"]).startswith("bronze:")
    assert resolve_bronze_digest(normalize_bronze_digest(None, ["b"]), ["b"]).startswith("bronze:")
    assert normalize_bronze_digest("b" * 64, ["b"]) == f"bronze:{'b' * 64}"
    with pytest.raises(PITDataError, match="non-empty"):
        normalize_bronze_digest("", ["b"])
    with pytest.raises(PITDataError, match="bronze:"):
        normalize_bronze_digest("bronze:bad", ["b"])
    with pytest.raises(PITDataError, match="SHA-256"):
        normalize_bronze_digest("bad", ["b"])

    def _legacy_partitions(raw: object, dataset_id: str = legacy_id) -> None:
        manifest.write_text(json.dumps({"dataset_id": dataset_id, "partitions": raw}), encoding="utf-8")

    _legacy_partitions({})
    with pytest.raises(PITDataError, match="invalid ordinary-universe partitions"):
        _legacy_partition_sessions({})
    _legacy_partitions([1])
    with pytest.raises(PITDataError, match="invalid ordinary-universe partition"):
        _legacy_partition_sessions([1])
    _legacy_partitions([{"session": 1}])
    with pytest.raises(PITDataError, match="partition session"):
        _legacy_partition_sessions([{"session": 1}])
    _legacy_partitions([{"path": "part.parquet"}])
    with pytest.raises(PITDataError, match="lacks session"):
        _legacy_partition_sessions([{"path": "part.parquet"}])
    _legacy_partitions([{"session": "bad"}])
    with pytest.raises(PITDataError, match="partition session"):
        _legacy_partition_sessions([{"session": "bad"}])
    assert _legacy_partition_sessions([{"path": "session=2024-01-02/part.parquet"}]) == [date(2024, 1, 2)]


def test_dataset_v2_universe_and_republish_error_boundaries(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import src.data.datasets as dataset_module
    from src.data.datasets import _require_identical_partitions

    valid = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER, params={"case": "valid"}),
        partitions={"session=2024-01-02/part.parquet": pl.DataFrame({"session": [date(2024, 1, 2)]})},
    )
    monkeypatch.setattr(dataset_module, "verify_dataset", lambda *_args, **_kwargs: (_ for _ in ()).throw(PITDataError("verify")))
    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        universe_sessions(tmp_path / "silver", valid.dataset_id)
    monkeypatch.undo()

    monkeypatch.setattr(
        dataset_module,
        "verify_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(passed=False, failures=("bad",)),
    )
    with pytest.raises(PITDataError, match="bad"):
        universe_sessions(tmp_path / "silver", valid.dataset_id)
    monkeypatch.undo()

    wrong_kind = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="daily_market", layer=DatasetLayer.SILVER, params={"case": "wrong-kind"}),
        partitions={"session=2024-01-02/part.parquet": pl.DataFrame({"session": [date(2024, 1, 2)]})},
    )
    with pytest.raises(PITDataError, match="ordinary-universe kind"):
        universe_sessions(tmp_path / "silver", wrong_kind.dataset_id)

    empty_partition = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER, params={"case": "empty"}),
        partitions={"session=2024-01-02/part.parquet": pl.DataFrame({"session": pl.Series([], dtype=pl.Date)})},
    )
    assert universe_sessions(tmp_path / "silver", empty_partition.dataset_id)[1] == (date(2024, 1, 2),)
    bad_partition = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER, params={"case": "bad-session"}),
        partitions={"session=bad/part.parquet": pl.DataFrame({"session": pl.Series([], dtype=pl.Date)})},
    )
    with pytest.raises(PITDataError, match="invalid ordinary-universe partition session"):
        universe_sessions(tmp_path / "silver", bad_partition.dataset_id)
    non_date = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER, params={"case": "non-date"}),
        partitions={"part.parquet": pl.DataFrame({"session": ["bad"]})},
    )
    with pytest.raises(PITDataError, match="invalid ordinary-universe partition session"):
        universe_sessions(tmp_path / "silver", non_date.dataset_id)
    unordered = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER, params={"case": "unordered"}),
        partitions={
            "a/part.parquet": pl.DataFrame({"session": [date(2024, 1, 3)]}),
            "z/part.parquet": pl.DataFrame({"session": [date(2024, 1, 2)]}),
        },
    )
    with pytest.raises(PITDataError, match="strictly ordered"):
        universe_sessions(tmp_path / "silver", unordered.dataset_id)

    no_session_path = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=_identity(kind="ordinary_universe", layer=DatasetLayer.SILVER, params={"case": "no-session-path"}),
        partitions={"part.parquet": pl.DataFrame({"session": pl.Series([], dtype=pl.Date)})},
    )
    with pytest.raises(PITDataError, match="lacks session"):
        universe_sessions(tmp_path / "silver", no_session_path.dataset_id)

    monkeypatch.setattr(dataset_module.pl, "read_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")))
    with pytest.raises(PITDataError, match="invalid ordinary-universe partition"):
        universe_sessions(tmp_path / "silver", valid.dataset_id)
    monkeypatch.undo()

    expected = load_manifest(valid.path)
    original_dataset_id_for = dataset_module.dataset_id_for
    identity_checks = {"count": 0}

    def _different_after_load(identity):
        identity_checks["count"] += 1
        return "different" if identity_checks["count"] == 2 else original_dataset_id_for(identity)

    monkeypatch.setattr(dataset_module, "dataset_id_for", _different_after_load)
    with pytest.raises(PITDataError, match="non-deterministic rebuild"):
        _require_identical_partitions(valid.path, expected)
    monkeypatch.undo()
    original_read_bytes = Path.read_bytes

    def fail_manifest_read(path: Path) -> bytes:
        if path == valid.path / MANIFEST_NAME:
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_manifest_read)
    with pytest.raises(PITDataError, match="manifest is unreadable"):
        _require_identical_partitions(valid.path, expected)
    monkeypatch.undo()
    monkeypatch.setattr(dataset_module, "verify_dataset", lambda *_args, **_kwargs: SimpleNamespace(passed=False, failures=("bad",)))
    with pytest.raises(PITDataError, match="failed verification"):
        _require_identical_partitions(valid.path, expected)
    monkeypatch.undo()


def test_dataset_check_validation_rejects_invalid_gate_shapes(tmp_path: Path) -> None:
    from src.data.datasets import DatasetCheck

    identity = _identity(kind="daily_market")
    invalid_checks = [
        [object()],
        [DatasetCheck(name="", value=1.0, limit=1.0, passed=True)],
        [DatasetCheck(name="x", value=1.0, limit=1.0, passed=True), DatasetCheck(name="x", value=1.0, limit=1.0, passed=True)],
        [DatasetCheck(name="x", value=float("nan"), limit=1.0, passed=True)],
        [DatasetCheck(name="x", value=1.0, limit=1.0, passed=1)],  # type: ignore[arg-type]
    ]
    for checks in invalid_checks:
        with pytest.raises(PITDataError):
            publish_dataset(layer_root=tmp_path / "silver", identity=identity, partitions={"part.parquet": _frame()}, checks=checks)
