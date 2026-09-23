"""Retention planning and compaction for storage that grows without bound.

Two storage paths publish a brand-new full snapshot on every write and never
remove the superseded one: the Bronze receipt catalog (a new revision file
per ``publish()`` call) and ``ParquetDatasetStore``-based Silver tables (a new
generation directory per refresh). Both are reproducible, non-authoritative
artifacts, so a plan-then-apply compaction tool is safe: plan first (read
only), apply only behind an explicit call that re-verifies the plan against
current directory state immediately before deleting.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.schemas import PITDataError
from src.storage.parquet_datasets import MANIFEST_NAME

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{64}\.json$")


@dataclass(frozen=True, slots=True)
class CatalogRevisionRetentionPlan:
    """Dry-run classification of one receipt-catalog directory's revisions.

    Only files matching the catalog's own ``<64-hex-sha256>.json`` revision
    naming are ever classified as reclaimable; any other file in the
    directory (including ``latest.json`` itself) is left untouched.
    """

    catalog_root: Path
    live_revision: str
    reclaimable_revisions: tuple[str, ...]
    reclaimable_bytes: int
    deletion_eligible: bool
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableGenerationRetentionPlan:
    """Dry-run classification of one ParquetDatasetStore table's generations."""

    table_root: Path
    live_generation: str
    reclaimable_generations: tuple[str, ...]
    reclaimable_bytes: int
    deletion_eligible: bool
    blocking_reasons: tuple[str, ...]


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def discover_generation_tables(silver_root: Path) -> tuple[Path, ...]:
    """Find every ParquetDatasetStore-style multi-generation table under a scope's Silver root.

    A candidate is an immediate subdirectory of ``silver_root`` whose own
    immediate children are all directories that each carry a
    ``dataset_manifest.json`` (the ``ParquetDatasetStore`` generation
    manifest). Detection is purely structural so it never depends on a
    hardcoded table-name list and never matches a flat content-hash dataset
    directory (which carries its own manifest at its own root, not a nested
    generation per child).

    Args:
        silver_root: A scope's Silver root (one level above table directories).

    Returns:
        Sorted table root paths with at least one generation.
    """
    root = Path(silver_root)
    if not root.exists():
        return ()
    tables: list[Path] = []
    for candidate in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        children = [c for c in candidate.iterdir() if c.is_dir() and not c.name.startswith(".")]
        if children and all((child / MANIFEST_NAME).is_file() for child in children):
            tables.append(candidate)
    return tuple(tables)


def _read_generated_time(manifest_path: Path) -> tuple[datetime, str] | None:
    try:
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    raw_generated = data.get("generated_time")
    if not isinstance(raw_generated, str):
        return None
    try:
        generated = datetime.fromisoformat(raw_generated)
    except ValueError:
        return None
    if generated.tzinfo is None:
        return None
    content_hash = data.get("content_hash")
    return (generated, str(content_hash) if content_hash else manifest_path.parent.name)


def plan_catalog_revision_retention(catalog_root: Path) -> CatalogRevisionRetentionPlan:
    """Classify a receipt-catalog directory's revisions as live or reclaimable.

    Args:
        catalog_root: A scope's ``bronze/<scope>/catalog`` directory.

    Returns:
        The plan. ``deletion_eligible`` is true only when the live revision
        was independently verified readable and holds a JSON list, and every
        other ``<hex>.json`` file in the directory is being proposed for
        deletion; otherwise ``blocking_reasons`` explains why and
        ``reclaimable_revisions`` is empty.
    """
    root = Path(catalog_root)
    pointer_path = root / "latest.json"  # mirrors ReceiptCatalog._POINTER_NAME
    blocking: list[str] = []
    live_revision = ""
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        live_revision = str(pointer["revision"])
        if not live_revision:
            raise ValueError("empty revision")
    except (OSError, ValueError, KeyError):
        blocking.append("missing_or_invalid_pointer")
    if not blocking:
        live_path = root / live_revision
        try:
            live_payload = json.loads(live_path.read_text(encoding="utf-8"))
            if not isinstance(live_payload, list):
                raise ValueError("live revision is not a list")
        except (OSError, ValueError):
            blocking.append("unreadable_live_revision")
    reclaimable: tuple[str, ...] = ()
    reclaimable_bytes = 0
    if not blocking:
        reclaimable = tuple(
            sorted(
                p.name
                for p in root.iterdir()
                if p.is_file() and _REVISION_PATTERN.match(p.name) and p.name != live_revision
            )
        )
        reclaimable_bytes = sum((root / name).stat().st_size for name in reclaimable)
    return CatalogRevisionRetentionPlan(
        catalog_root=root,
        live_revision=live_revision,
        reclaimable_revisions=reclaimable,
        reclaimable_bytes=reclaimable_bytes,
        deletion_eligible=not blocking,
        blocking_reasons=tuple(blocking),
    )


def plan_table_generation_retention(table_root: Path) -> TableGenerationRetentionPlan:
    """Classify one multi-generation table's directories as live or reclaimable.

    Args:
        table_root: A path returned by :func:`discover_generation_tables`.

    Returns:
        The plan, using the same latest-generation selection as
        ``load_latest_silver_table``: the generation with the greatest
        ``(generated_time, content_hash)``. ``deletion_eligible`` is false and
        ``blocking_reasons`` explains why whenever any generation's manifest
        is unreadable, malformed, or has a missing/naive ``generated_time`` —
        an unreadable manifest blocks the whole table rather than being
        silently skipped, because it could itself be the true live generation.
    """
    root = Path(table_root)
    generations = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    blocking: list[str] = []
    keys: dict[str, tuple[datetime, str]] = {}
    for generation in generations:
        resolved = _read_generated_time(generation / MANIFEST_NAME)
        if resolved is None:
            blocking.append(f"unreadable_manifest:{generation.name}")
            continue
        keys[generation.name] = resolved
    if blocking:
        return TableGenerationRetentionPlan(
            table_root=root,
            live_generation="",
            reclaimable_generations=(),
            reclaimable_bytes=0,
            deletion_eligible=False,
            blocking_reasons=tuple(blocking),
        )
    live_name = max(keys, key=lambda name: keys[name])
    reclaimable = tuple(sorted(name for name in keys if name != live_name))
    reclaimable_bytes = sum(_dir_bytes(root / name) for name in reclaimable)
    return TableGenerationRetentionPlan(
        table_root=root,
        live_generation=live_name,
        reclaimable_generations=reclaimable,
        reclaimable_bytes=reclaimable_bytes,
        deletion_eligible=True,
        blocking_reasons=(),
    )


def apply_catalog_revision_retention(catalog_root: Path) -> int:
    """Delete a catalog's reclaimable revisions after re-verifying the plan.

    Recomputes :func:`plan_catalog_revision_retention` immediately before
    deleting so a revision published between planning and applying is never
    destroyed as a side effect of stale input.

    Returns:
        Total bytes freed.

    Raises:
        PITDataError: the freshly recomputed plan is not ``deletion_eligible``.
    """
    plan = plan_catalog_revision_retention(catalog_root)
    if not plan.deletion_eligible:
        raise PITDataError(f"catalog revision retention is not deletion-eligible: {plan.blocking_reasons}")
    freed = 0
    for name in plan.reclaimable_revisions:
        path = plan.catalog_root / name
        size = path.stat().st_size
        try:
            path.unlink()
        except OSError:
            continue
        freed += size
    return freed


def apply_table_generation_retention(table_root: Path) -> int:
    """Delete a table's reclaimable generations after re-verifying the plan.

    Returns:
        Total bytes freed.

    Raises:
        PITDataError: the freshly recomputed plan is not ``deletion_eligible``.
    """
    plan = plan_table_generation_retention(table_root)
    if not plan.deletion_eligible:
        raise PITDataError(f"table generation retention is not deletion-eligible: {plan.blocking_reasons}")
    freed = 0
    for name in plan.reclaimable_generations:
        path = plan.table_root / name
        size = _dir_bytes(path)
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            freed += size
    return freed
