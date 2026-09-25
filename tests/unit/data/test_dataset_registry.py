"""Verified, concurrent dataset registry tests."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.data.dataset_registry import REGISTRY_NAME, DatasetRegistry, RetiredInput
from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset


def _identity(kind: str, *, inputs: dict[str, str] | None = None) -> DatasetIdentity:
    return DatasetIdentity(
        kind=kind,
        layer=DatasetLayer.SILVER,
        policy_version="fixture-v1",
        inputs=inputs or {},
        params={},
    )


def _publish(layer_root: Path, kind: str, *, inputs: dict[str, str] | None = None) -> str:
    return publish_dataset(
        layer_root=layer_root,
        identity=_identity(kind, inputs=inputs),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    ).dataset_id


def _register_in_process(
    state_root: Path,
    kind: str,
    dataset_id: str,
    start: multiprocessing.synchronize.Event,
) -> None:
    start.wait(timeout=10)
    DatasetRegistry(state_root).register(kind, dataset_id)


def test_register_requires_a_verified_dataset_and_is_unchanged_on_failure(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    state_root = data_root / "state" / "scope"
    silver_root = data_root / "silver" / "scope"
    dataset_id = _publish(silver_root, "daily_market")
    registry = DatasetRegistry(state_root)
    before = registry.snapshot()
    partition = silver_root / dataset_id / "part.parquet"
    partition.write_bytes(partition.read_bytes() + b"tampered")

    with pytest.raises(PITDataError, match="dataset verification failed"):
        registry.register("daily_market", dataset_id)

    assert registry.snapshot() == before


def test_register_rejects_missing_input_until_upstream_exists(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    state_root = data_root / "state" / "scope"
    silver_root = data_root / "silver" / "scope"
    upstream_id = _publish(silver_root, "ordinary_universe")
    target_id = _publish(silver_root, "daily_market", inputs={"universe": upstream_id})
    registry = DatasetRegistry(state_root)

    (silver_root / upstream_id).rename(silver_root / "temporarily_unavailable")
    with pytest.raises(PITDataError, match="missing dataset input"):
        registry.register("daily_market", target_id)

    (silver_root / "temporarily_unavailable").rename(silver_root / upstream_id)
    registry.register("daily_market", target_id)
    assert registry.require("daily_market") == target_id


def test_kind_must_match_id_prefix(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    silver_root = data_root / "silver" / "scope"
    dataset_id = _publish(silver_root, "daily_market")
    registry = DatasetRegistry(data_root / "state" / "scope")

    with pytest.raises(PITDataError, match="dataset kind mismatch"):
        registry.register("market_panel", dataset_id)


def test_require_fails_closed_and_snapshots_are_immutable(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path / "data" / "state" / "scope")

    assert registry.current("market_panel") is None
    with pytest.raises(PITDataError, match="not registered"):
        registry.require("market_panel")
    with pytest.raises(TypeError):
        registry.snapshot()["market_panel"] = "changed"  # type: ignore[index]


def test_concurrent_registers_keep_both_pointers(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    state_root = data_root / "state" / "scope"
    silver_root = data_root / "silver" / "scope"
    first_id = _publish(silver_root, "daily_market")
    second_id = _publish(silver_root, "market_panel")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_register_in_process, args=(state_root, "daily_market", first_id, start)),
        context.Process(target=_register_in_process, args=(state_root, "market_panel", second_id, start)),
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0]
    assert DatasetRegistry(state_root).snapshot() == {
        "daily_market": first_id,
        "market_panel": second_id,
    }


def test_registry_reads_retired_lineage_and_writes_human_readable_json(tmp_path: Path) -> None:
    state_root = tmp_path / "data" / "state" / "scope"
    state_root.mkdir(parents=True)
    retired_id = "ordinary_universe_0123456789abcdef"
    (state_root / REGISTRY_NAME).write_text(
        json.dumps(
            {
                "current": {},
                "retired": {retired_id: "superseded before verified lineage"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = DatasetRegistry(state_root)

    retired = registry.retired()

    assert retired == {
        retired_id: RetiredInput(
            dataset_id=retired_id,
            reason="superseded before verified lineage",
        )
    }
    assert (state_root / REGISTRY_NAME).read_text(encoding="utf-8").startswith("{\n")


def test_register_rejects_missing_or_ambiguous_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    state_root = data_root / "state" / "scope"
    registry = DatasetRegistry(state_root)
    missing_id = "daily_market_0123456789abcdef"

    with pytest.raises(PITDataError, match="directory does not exist"):
        registry.register("daily_market", missing_id)

    silver_dir = data_root / "silver" / "scope" / missing_id
    gold_dir = data_root / "gold" / "scope" / missing_id
    silver_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)
    with pytest.raises(PITDataError, match="ambiguous"):
        registry.register("daily_market", missing_id)


def test_invalid_registry_documents_fail_closed(tmp_path: Path) -> None:
    state_root = tmp_path / "data" / "state" / "scope"
    state_root.mkdir(parents=True)
    registry = DatasetRegistry(state_root)

    (state_root / REGISTRY_NAME).write_text("not-json", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable dataset registry"):
        registry.snapshot()

    (state_root / REGISTRY_NAME).write_text(
        json.dumps({"current": {"daily_market": "market_panel_0123456789abcdef"}, "retired": {}}),
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="kind mismatch"):
        registry.snapshot()


def test_registry_rejects_invalid_document_shapes(tmp_path: Path) -> None:
    state_root = tmp_path / "data" / "state" / "scope"
    state_root.mkdir(parents=True)
    path = state_root / REGISTRY_NAME
    registry = DatasetRegistry(state_root)

    path.write_text(json.dumps({"current": {}}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid dataset registry"):
        registry.snapshot()

    invalid_documents = [
        ({"current": [], "retired": {}}, "current must be an object"),
        ({"current": {"daily_market": 1}, "retired": {}}, "map strings to strings"),
        ({"current": {}, "retired": []}, "retired must be an object"),
        ({"current": {}, "retired": {"ordinary_universe_0123456789abcdef": ""}}, "non-empty reasons"),
    ]
    for raw, message in invalid_documents:
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(PITDataError, match=message):
            registry.snapshot()


def test_register_rejects_layer_root_mismatch(tmp_path: Path) -> None:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    data_root = tmp_path / "data"
    silver_root = data_root / "silver" / "scope"
    published = publish_dataset(
        layer_root=silver_root,
        identity=DatasetIdentity("daily_market", DatasetLayer.GOLD, "fixture-v1", {}, {}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )

    with pytest.raises(PITDataError, match="does not match its root"):
        DatasetRegistry(data_root / "state" / "scope").register("daily_market", published.dataset_id)


def test_registry_write_failure_preserves_snapshot(tmp_path: Path, monkeypatch) -> None:
    import src.data.dataset_registry as registry_module

    data_root = tmp_path / "data"
    state_root = data_root / "state" / "scope"
    dataset_id = _publish(data_root / "silver" / "scope", "daily_market")
    registry = DatasetRegistry(state_root)
    before = registry.snapshot()
    monkeypatch.setattr(registry_module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace")))

    with pytest.raises(PITDataError, match="cannot write dataset registry"):
        registry.register("daily_market", dataset_id)

    assert registry.snapshot() == before


def test_registry_retirement_and_nested_discovery_guards(tmp_path: Path, monkeypatch) -> None:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    import src.data.dataset_registry as registry_module

    data_root = tmp_path / "data"
    state_root = data_root / "state" / "scope"
    registry = DatasetRegistry(state_root)
    retired_id = "ordinary_universe_0123456789abcdef"
    with pytest.raises(PITDataError, match="reason"):
        registry.retire(retired_id, "")
    registry.retire(retired_id, "superseded")
    registry.retire(retired_id, "superseded")
    with pytest.raises(PITDataError, match="different reason"):
        registry.retire(retired_id, "changed")
    assert registry.retired()[retired_id].reason == "superseded"

    nested_root = data_root / "silver" / "scope" / "table"
    (data_root / "silver" / "scope" / ".hidden").mkdir(parents=True)
    nested = publish_dataset(
        layer_root=nested_root,
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    registry.register("daily_market", nested.dataset_id)
    assert registry.require("daily_market") == nested.dataset_id

    outside = tmp_path / "outside"
    outside_dataset = publish_dataset(
        layer_root=outside,
        identity=DatasetIdentity("daily_market", DatasetLayer.SILVER, "fixture-v1", {}, {}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    monkeypatch.setattr(registry, "_candidate_directories", lambda _dataset_id: (outside_dataset.path,))
    with pytest.raises(PITDataError, match="outside the data root"):
        registry.register("daily_market", outside_dataset.dataset_id)
    _ = registry_module
