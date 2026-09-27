"""Verified dataset inspection and unreferenced-dataset pruning for one scope."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

from src.core.pit import PITDataError
from src.data.dataset_registry import DatasetRegistry
from src.data.datasets import dataset_kind_from_id, load_manifest, verify_dataset
from src.data.runtime import DataRuntime

__all__ = [
    "dataset_directories",
    "prune_datasets",
    "verify_datasets",
]

_LOG = logging.getLogger(__name__)


def dataset_directories(runtime: DataRuntime) -> tuple[Path, ...]:
    """Flat and one-level table dataset directories in a scope."""
    dataset_name = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*_[0-9a-f]{16}\Z")
    paths: list[Path] = []
    for root in (runtime.workspace.silver_root, runtime.workspace.gold_root):
        if not root.is_dir() or root.is_symlink():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
                continue
            if dataset_name.fullmatch(path.name) or (path / "manifest.json").is_file():
                paths.append(path)
                continue
            paths.extend(
                nested
                for nested in sorted(path.iterdir())
                if nested.is_dir()
                and not nested.is_symlink()
                and (
                    (nested / "manifest.json").is_file()
                    or (nested / "dataset_manifest.json").is_file()
                    or (nested / "content_manifest.json").is_file()
                )
            )
    return tuple(paths)


def stale_inputs_for_directory(
    *,
    dataset_dir: Path | None,
    dataset_id: str,
    registered_ids: set[str],
    current: Mapping[str, str],
    retired_ids: set[str],
) -> list[dict[str, object]]:
    """Stale lineage warnings for one verified dataset directory."""
    if dataset_dir is None or dataset_id not in registered_ids:
        return []
    try:
        manifest = load_manifest(dataset_dir)
    except PITDataError:
        return []
    stale: list[dict[str, object]] = []
    for role, input_id in sorted(manifest.inputs.items()):
        if input_id in retired_ids:
            continue
        try:
            input_kind = dataset_kind_from_id(input_id)
        except PITDataError:
            continue
        registered_id = current.get(input_kind)
        if input_id != registered_id:
            stale.append(
                {
                    "role": role,
                    "input_kind": input_kind,
                    "input_id": input_id,
                    "registered_id": registered_id,
                }
            )
    return stale


def legacy_prunable_manifest(path: Path) -> Mapping[str, object] | None:
    """A structurally valid pre-v2 dataset manifest eligible for pruning."""
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = path / "dataset_manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if manifest_path.name == "dataset_manifest.json":
        content_path = path / "content_manifest.json"
        try:
            content = json.loads(content_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(content, dict):
            return None
        merged = dict(content)
        merged.update(raw)
        merged["dataset_id"] = path.name
        raw = merged
    if raw.get("dataset_id") != path.name:
        return None
    partitions = raw.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        return None
    if any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in partitions):
        return None
    return raw


def legacy_lineage(manifest: Mapping[str, object]) -> set[str]:
    """Dataset ids referenced by a legacy manifest, ignoring unknown shapes."""
    values: list[object] = []
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        values.extend(inputs.values())
    for name in (
        "daily_market_dataset_id",
        "universe_dataset_id",
        "ls_dataset_id",
        "kis_supplement_dataset_id",
        "market_panel_dataset_id",
        "source_financial_dataset_id",
    ):
        value = manifest.get(name)
        if isinstance(value, str):
            values.append(value)
    result: set[str] = set()
    for value in values:
        try:
            dataset_kind_from_id(str(value))
        except PITDataError:
            continue
        result.add(str(value))
    return result


def validated_current_directories(
    runtime: DataRuntime, registry: DatasetRegistry
) -> tuple[dict[str, str], set[str]]:
    """Every current pointer, validated before retention can classify anything."""
    current = registry.snapshot()
    if not current:
        raise PITDataError("dataset registry has no current datasets")
    directories = dataset_directories(runtime)
    by_id: dict[str, list[Path]] = {}
    for directory in directories:
        by_id.setdefault(directory.name, []).append(directory)
    retired_ids = set(registry.retired())
    known_ids = set(by_id) | retired_ids
    for kind, dataset_id in sorted(current.items()):
        matches = by_id.get(dataset_id, [])
        if len(matches) != 1:
            raise PITDataError(f"current dataset is missing or ambiguous: {dataset_id}")
        dataset_dir = matches[0]
        try:
            manifest = load_manifest(dataset_dir)
        except PITDataError as exc:
            raise PITDataError(f"current dataset manifest is invalid: {dataset_id}") from exc
        if manifest.kind != kind:
            raise PITDataError(f"current dataset kind mismatch: {dataset_id}")
        verification = verify_dataset(dataset_dir, known_ids=known_ids.__contains__)
        if not verification.passed:
            raise PITDataError(f"current dataset failed verification: {dataset_id}: {verification.failures}")
    return dict(current), retired_ids


def prune_plan(
    runtime: DataRuntime, registry: DatasetRegistry
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Readable orphan paths and unreadable manifests, failing closed."""
    current, _retired_ids = validated_current_directories(runtime, registry)
    directories = dataset_directories(runtime)
    graph: dict[str, set[str]] = {}
    unreadable: list[tuple[Path, str]] = []
    unreadable_ids: set[str] = set()
    for dataset_dir in directories:
        try:
            manifest = load_manifest(dataset_dir)
            upstream = graph.setdefault(manifest.dataset_id, set())
            for input_id in manifest.inputs.values():
                try:
                    dataset_kind_from_id(input_id)
                except PITDataError:
                    continue
                upstream.add(input_id)
        except PITDataError as exc:
            legacy = legacy_prunable_manifest(dataset_dir)
            if legacy is None:
                unreadable.append((dataset_dir, str(exc)))
                unreadable_ids.add(dataset_dir.name)
                continue
            upstream = graph.setdefault(dataset_dir.name, set())
            upstream.update(legacy_lineage(legacy))

    registered_ids = set(current.values())
    if registered_ids & unreadable_ids:
        return [], sorted(unreadable, key=lambda item: str(item[0]))
    reachable = set(registered_ids)
    pending = list(registered_ids)
    while pending:
        dataset_id = pending.pop()
        for input_id in graph.get(dataset_id, set()):
            if input_id not in reachable:
                reachable.add(input_id)
                pending.append(input_id)

    orphans = [
        dataset_dir
        for dataset_dir in directories
        if dataset_dir.name not in reachable and dataset_dir.name not in unreadable_ids
    ]
    return sorted(orphans, key=lambda path: str(path)), sorted(unreadable, key=lambda item: str(item[0]))


def verify_datasets(
    runtime: DataRuntime, *, verify_all: bool, emit: Callable[[Mapping[str, object]], None]
) -> dict[str, object]:
    """Verify registered datasets, or every dataset with ``verify_all``."""
    registry = DatasetRegistry(runtime.workspace.state_root)
    current = registry.snapshot()
    if not current:
        raise PITDataError("dataset registry has no current datasets")
    retired_ids = set(registry.retired())
    directories = dataset_directories(runtime)
    known_ids = {path.name for path in directories} | retired_ids
    targets: list[tuple[str, Path | None]]
    if verify_all:
        targets = [(path.name, path) for path in directories]
    else:
        targets = []
        for _kind, dataset_id in sorted(current.items()):
            matches = [path for path in directories if path.name == dataset_id]
            targets.append((dataset_id, matches[0] if len(matches) == 1 else None))

    registered_ids = set(current.values())
    failed = 0
    stale_count = 0
    for dataset_id, dataset_dir in targets:
        if dataset_dir is None:
            failures = [f"registered dataset directory is missing or ambiguous: {dataset_id}"]
            verification = None
            rows = 0
        else:
            verification = verify_dataset(dataset_dir, known_ids=known_ids.__contains__)
            failures = list(verification.failures)
            rows = verification.rows
        failed_checks = [] if verification is None else [check.name for check in verification.failed_checks]
        stale = stale_inputs_for_directory(
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            registered_ids=registered_ids,
            current=current,
            retired_ids=retired_ids,
        )
        stale_count += int(bool(stale))
        status = "failed" if failures else "ok"
        failed += int(bool(failures))
        emit(
            {
                "type": "dataset",
                "dataset_id": dataset_id,
                "path": None if dataset_dir is None else str(dataset_dir),
                "status": status,
                "rows": rows,
                "failures": failures,
                "failed_checks": failed_checks,
                "stale_inputs": stale,
            }
        )
        _LOG.info(
            "[DATA] command=verify_datasets dataset=%s status=%s stale=%s",
            dataset_id,
            status,
            bool(stale),
        )

    summary: dict[str, object] = {"type": "summary", "datasets": len(targets), "failed": failed, "stale": stale_count}
    _LOG.info(
        "[DATA] command=verify_datasets action=summary datasets=%d failed=%d stale=%d",
        len(targets),
        failed,
        stale_count,
    )
    return summary


def prune_datasets(
    runtime: DataRuntime,
    registry: DatasetRegistry,
    *,
    apply: bool,
    emit: Callable[[Mapping[str, object]], None],
) -> dict[str, object]:
    """List and optionally delete only revalidated unreferenced datasets."""
    candidates, unreadable = prune_plan(runtime, registry)

    for dataset_dir in candidates:
        emit(
            {
                "type": "dataset",
                "dataset_id": dataset_dir.name,
                "path": str(dataset_dir),
                "status": "prunable",
                "action": "delete" if apply else "plan",
            }
        )
        _LOG.info("[DATA] command=prune_datasets dataset=%s action=prune", dataset_dir.name)
    for dataset_dir, error in unreadable:
        emit(
            {
                "type": "dataset",
                "dataset_id": dataset_dir.name,
                "path": str(dataset_dir),
                "status": "unreadable_manifest",
                "action": "keep",
                "error": error,
            }
        )
        _LOG.warning(
            "[DATA] command=prune_datasets dataset=%s action=keep reason=unreadable_manifest", dataset_dir.name
        )

    deleted = 0
    delete_failed = 0
    if apply:
        revalidated, revalidated_unreadable = prune_plan(runtime, registry)
        unreadable.extend(item for item in revalidated_unreadable if item not in unreadable)
        revalidated_paths = set(revalidated)
        for dataset_dir in candidates:
            if dataset_dir not in revalidated_paths:
                continue
            if dataset_dir.is_symlink() or not dataset_dir.is_dir():
                continue
            try:
                try:
                    load_manifest(dataset_dir)
                except PITDataError:
                    if legacy_prunable_manifest(dataset_dir) is None:
                        raise
                shutil.rmtree(dataset_dir)
            except (PITDataError, OSError) as exc:
                delete_failed += 1
                emit(
                    {
                        "type": "dataset",
                        "dataset_id": dataset_dir.name,
                        "path": str(dataset_dir),
                        "status": "delete_failed",
                        "action": "keep",
                        "error": str(exc),
                    }
                )
                _LOG.error(
                    "[DATA] command=prune_datasets dataset=%s action=keep status=failed error=%s",
                    dataset_dir.name,
                    exc,
                )
                continue
            deleted += 1
            _LOG.info("[DATA] command=prune_datasets dataset=%s action=deleted", dataset_dir.name)

    summary = {
        "type": "summary",
        "candidates": len(candidates),
        "unreadable": len(unreadable),
        "applied": bool(apply),
        "deleted": deleted,
        "failed": delete_failed,
    }
    _LOG.info(
        "[DATA] command=prune_datasets action=summary candidates=%d deleted=%d failed=%d",
        len(candidates),
        deleted,
        delete_failed,
    )
    return summary
