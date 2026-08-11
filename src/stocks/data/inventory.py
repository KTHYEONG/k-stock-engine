"""Read-only filesystem inventory for the stock data root.

The inventory classifies every regular file beneath a data root into exactly
one lifecycle, streams a bounded-chunk SHA-256, and annotates catalog and
snapshot references. It is deliberately non-mutating: ``plan`` output is the
only artifact, and ``apply`` is out of scope. A file that is unreadable or that
changes while being hashed raises ``ValueError`` so no plan is ever produced
from partial evidence. Parquet files are inspected via footer metadata/schema
only; full frames are never collected.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as pq

from src.core.paths import PROJECT_ROOT
from src.stocks.data.catalog import (
    SNAPSHOT_MANIFEST_NAME,
    CatalogEntry,
    CatalogKind,
    CatalogStore,
)

INVENTORY_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024

_LEGACY_DIR_PREFIXES = ("processed", "etf_daily", "market_index", "model_features")
_ROOT_LEGACY_FILES = frozenset({"financials.parquet"})
_RUNTIME_ROOT_FILES = frozenset({"trading_state.db"})


class InventoryLifecycle(StrEnum):
    """Exactly one lifecycle per file; catalog/snapshot metadata is ``unknown``."""

    CANONICAL = "canonical"
    DERIVED = "derived"
    EVIDENCE = "evidence"
    ACTIVE_LEGACY = "active_legacy"
    ARCHIVE_CANDIDATE = "archive_candidate"
    RUNTIME_STATE = "runtime_state"
    UNKNOWN = "unknown"


class Recommendation(StrEnum):
    """Read-only disposition; only ``ARCHIVE_CANDIDATE`` files are eligible to move."""

    RETAIN = "retain"
    CANDIDATE = "candidate"


_OWNER_BY_LIFECYCLE = {
    InventoryLifecycle.CANONICAL: "catalog",
    InventoryLifecycle.DERIVED: "catalog",
    InventoryLifecycle.EVIDENCE: "evidence",
    InventoryLifecycle.ACTIVE_LEGACY: "legacy",
    InventoryLifecycle.ARCHIVE_CANDIDATE: "migration",
    InventoryLifecycle.RUNTIME_STATE: "runtime",
    InventoryLifecycle.UNKNOWN: "unknown",
}

_CANDIDATE_ELIGIBLE = frozenset(
    {
        InventoryLifecycle.CANONICAL,
        InventoryLifecycle.DERIVED,
        InventoryLifecycle.EVIDENCE,
        InventoryLifecycle.ACTIVE_LEGACY,
    }
)


def classify_lifecycle(rel_path: str) -> InventoryLifecycle:
    """Map a data-root-relative POSIX path to its lifecycle.

    Catalog and snapshot files are registry metadata, not datasets: they are
    classified ``unknown`` and are never archive candidates. Root-level legacy
    files and ``trading_state.db`` keep their legacy/runtime classification
    until a separate migration retires them.
    """
    parts = rel_path.split("/")
    first = parts[0]
    if len(parts) == 1 and first in _ROOT_LEGACY_FILES:
        return InventoryLifecycle.ACTIVE_LEGACY
    if len(parts) == 1 and first in _RUNTIME_ROOT_FILES:
        return InventoryLifecycle.RUNTIME_STATE
    if first in _LEGACY_DIR_PREFIXES:
        return InventoryLifecycle.ACTIVE_LEGACY
    if first == "runtime":
        return InventoryLifecycle.RUNTIME_STATE
    if first in ("canonical", "derived", "evidence"):
        return InventoryLifecycle(first)
    return InventoryLifecycle.UNKNOWN


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    """One deterministic, immutable record for a regular file."""

    path: str
    byte_count: int
    sha256: str
    extension: str
    lifecycle: InventoryLifecycle
    owner: str
    catalog_reference: str
    snapshot_reference: str
    recommendation: Recommendation
    candidate_reason: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "extension": self.extension,
            "lifecycle": self.lifecycle.value,
            "owner": self.owner,
            "catalog_reference": self.catalog_reference,
            "snapshot_reference": self.snapshot_reference,
            "recommendation": self.recommendation.value,
            "candidate_reason": self.candidate_reason,
        }


@dataclass(frozen=True, slots=True)
class InventoryReport:
    """The complete, sorted inventory of one data root."""

    data_root: str
    records: tuple[InventoryRecord, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "inventory_version": INVENTORY_VERSION,
            "data_root": self.data_root,
            "records": [record.to_json() for record in self.records],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, object]:
        by_lifecycle: dict[str, dict[str, int]] = {}
        for lifecycle in InventoryLifecycle:
            matching = [r for r in self.records if r.lifecycle is lifecycle]
            by_lifecycle[lifecycle.value] = {
                "files": len(matching),
                "bytes": sum(r.byte_count for r in matching),
            }
        candidates = [
            {"path": r.path, "reason": r.candidate_reason}
            for r in self.records
            if r.lifecycle is InventoryLifecycle.ARCHIVE_CANDIDATE
        ]
        return {
            "files": len(self.records),
            "bytes": sum(r.byte_count for r in self.records),
            "by_lifecycle": by_lifecycle,
            "candidates": candidates,
        }


@dataclass(frozen=True, slots=True)
class _ReferenceIndex:
    """Pre-resolved catalog and snapshot references keyed by data-relative path."""

    catalog_paths: dict[str, tuple[str, ...]]
    snapshot_paths: dict[str, tuple[str, ...]]
    content_hashes: frozenset[str]
    pinned_paths: frozenset[str]


def scan_inventory(data_root: Path, store: CatalogStore) -> InventoryReport:
    """Scan ``data_root`` and classify every regular file without mutation.

    Raises ``ValueError`` on any unreadable file, any file that changes while
    being hashed, or an unreadable Parquet footer. No report is returned when
    the scan fails, so the caller can never act on partial evidence.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"data root not found: {root}")
    index = _build_reference_index(store, root)
    records: list[InventoryRecord] = []
    for rel_path, full_path in _iter_files(root):
        byte_count, sha256 = _stream_sha256(full_path)
        if full_path.suffix.lower() == ".parquet":
            _verify_parquet_footer(full_path)
        catalog_reference, snapshot_reference = _references_for(rel_path, index)
        pinned = _is_pinned(rel_path, index)
        lifecycle, recommendation, reason = _classify(
            rel_path, sha256, catalog_reference, snapshot_reference, pinned, index
        )
        records.append(
            InventoryRecord(
                path=rel_path,
                byte_count=byte_count,
                sha256=sha256,
                extension=full_path.suffix.lower().lstrip("."),
                lifecycle=lifecycle,
                owner=_OWNER_BY_LIFECYCLE[lifecycle],
                catalog_reference=catalog_reference,
                snapshot_reference=snapshot_reference,
                recommendation=recommendation,
                candidate_reason=reason,
            )
        )
    records.sort(key=lambda record: record.path)
    return InventoryReport(data_root=str(root.resolve()), records=tuple(records))


def _iter_files(root: Path) -> Iterator[tuple[str, Path]]:
    def on_error(error: OSError) -> None:
        raise ValueError(f"unreadable directory: {root}: {error}") from error

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        dirnames.sort()
        for filename in sorted(filenames):
            full_path = Path(dirpath) / filename
            try:
                if full_path.is_symlink() or not full_path.is_file():
                    continue
            except OSError as error:
                raise ValueError(f"unreadable path: {full_path}: {error}") from error
            yield full_path.relative_to(root).as_posix(), full_path


def _stream_sha256(path: Path) -> tuple[int, str]:
    """Stream a bounded-chunk SHA-256 and fail closed if the file mutates."""
    try:
        before = path.stat()
    except OSError as error:
        raise ValueError(f"unreadable file: {path}: {error}") from error
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
                total += len(chunk)
    except OSError as error:
        raise ValueError(f"unreadable file: {path}: {error}") from error
    try:
        after = path.stat()
    except OSError as error:
        raise ValueError(f"unreadable file: {path}: {error}") from error
    if after.st_size != before.st_size or after.st_mtime != before.st_mtime or total != before.st_size:
        raise ValueError(f"file changed during inventory: {path}")
    return before.st_size, digest.hexdigest()


def _verify_parquet_footer(path: Path) -> None:
    """Read Parquet footer metadata/schema only; never collect a frame."""
    try:
        _ = pq.ParquetFile(path).metadata
    except Exception as error:
        raise ValueError(f"unreadable parquet: {path}: {error}") from error


def _build_reference_index(store: CatalogStore, data_root: Path) -> _ReferenceIndex:
    catalog_paths: dict[str, list[str]] = {}
    pinned_paths: set[str] = set()
    content_hashes: set[str] = set()
    for entry in store.list():
        content_hashes.add(entry.content_hash)
        rel_path = _entry_relative_path(entry.path, data_root)
        if rel_path is None:
            continue
        catalog_paths.setdefault(rel_path, []).append(f"{entry.kind.value}:{entry.name}")
        if entry.pinned:
            pinned_paths.add(rel_path)

    snapshot_paths: dict[str, list[str]] = {}
    snapshots_root = store.root / "snapshots"
    snapshot_root_rel = _directory_relative_path(snapshots_root, data_root)
    if snapshots_root.is_dir():
        for snapshot_dir in sorted(p for p in snapshots_root.iterdir() if p.is_dir()):
            manifest_path = snapshot_dir / SNAPSHOT_MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_id = snapshot_dir.name
            if snapshot_root_rel is not None:
                snapshot_paths.setdefault(f"{snapshot_root_rel}/{snapshot_id}", []).append(snapshot_id)
            raw_references = payload.get("references")
            if not isinstance(raw_references, dict):
                continue
            for kind, raw_ref in raw_references.items():
                if not isinstance(raw_ref, dict):
                    continue
                ref_hash = str(raw_ref.get("content_hash", ""))
                content_hashes.add(ref_hash)
                manifest_entry = _resolve_manifest_reference(store, kind, str(raw_ref.get("name", "")))
                if manifest_entry is None:
                    continue
                rel_path = _entry_relative_path(manifest_entry.path, data_root)
                if rel_path is not None:
                    snapshot_paths.setdefault(rel_path, []).append(snapshot_id)

    return _ReferenceIndex(
        catalog_paths={key: tuple(value) for key, value in catalog_paths.items()},
        snapshot_paths={key: tuple(value) for key, value in snapshot_paths.items()},
        content_hashes=frozenset(content_hashes),
        pinned_paths=frozenset(pinned_paths),
    )


def _resolve_manifest_reference(store: CatalogStore, kind: str, name: str) -> CatalogEntry | None:
    try:
        return store.get(CatalogKind(kind), name)
    except ValueError:
        return None


def _references_for(rel_path: str, index: _ReferenceIndex) -> tuple[str, str]:
    catalog_refs: list[str] = []
    for candidate, refs in index.catalog_paths.items():
        if _is_under(rel_path, candidate):
            catalog_refs.extend(refs)
    snapshot_refs: list[str] = []
    for candidate, refs in index.snapshot_paths.items():
        if _is_under(rel_path, candidate):
            snapshot_refs.extend(refs)
    return ",".join(dict.fromkeys(catalog_refs)), ",".join(dict.fromkeys(snapshot_refs))


def _is_pinned(rel_path: str, index: _ReferenceIndex) -> bool:
    return any(_is_under(rel_path, candidate) for candidate in index.pinned_paths)


def _classify(
    rel_path: str,
    sha256: str,
    catalog_reference: str,
    snapshot_reference: str,
    pinned: bool,
    index: _ReferenceIndex,
) -> tuple[InventoryLifecycle, Recommendation, str]:
    lifecycle = classify_lifecycle(rel_path)
    if (
        lifecycle in _CANDIDATE_ELIGIBLE
        and not (catalog_reference or snapshot_reference or pinned)
        and sha256 in index.content_hashes
    ):
        reason = f"byte-identical to a cataloged/referenced content hash {sha256[:12]}"
        return InventoryLifecycle.ARCHIVE_CANDIDATE, Recommendation.CANDIDATE, reason
    return lifecycle, Recommendation.RETAIN, ""


def _is_under(rel_path: str, candidate: str) -> bool:
    return rel_path == candidate or rel_path.startswith(candidate + "/")


def _entry_relative_path(raw_path: str, data_root: Path) -> str | None:
    """Resolve a catalog entry ``path`` to a data-root-relative POSIX path.

    Real entries store absolute data paths or ``data/...`` paths relative to the
    repository root; tests store paths relative to a temporary data root. All
    three conventions resolve to the same data-relative key.
    """
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = (path,) if path.is_absolute() else (PROJECT_ROOT / path, data_root / path)
    resolved_root = data_root.resolve()
    for candidate in candidates:
        try:
            return candidate.resolve().relative_to(resolved_root).as_posix()
        except ValueError:
            continue
    return None


def _directory_relative_path(path: Path, data_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return None
