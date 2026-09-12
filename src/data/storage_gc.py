"""Silver/Gold storage-root retention planning (audit-only, no deletion)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_STAGING_MARKER = "staging"


@dataclass(frozen=True, slots=True)
class StorageRootRetentionPlan:
    retained_silver_roots: tuple[str, ...]
    reclaimable_silver_roots: tuple[str, ...]
    retained_gold_roots: tuple[str, ...]
    reclaimable_gold_roots: tuple[str, ...]
    orphaned_staging_paths: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    deletion_eligible: bool


def find_orphaned_staging_paths(base: Path) -> tuple[str, ...]:
    """List directories left behind by an interrupted ParquetDatasetStore.write.

    A completed write always removes its own staging directory on both the
    success and failure paths, so any hidden directory whose name still
    contains "staging" can only be the residue of a process that was killed
    mid-write (crash or disconnect) and is always safe to report as reclaimable.
    """
    root = Path(base)
    if not root.exists():
        return ()
    return tuple(
        sorted(
            str(path)
            for path in root.rglob("*")
            if path.is_dir() and path.name.startswith(".") and _STAGING_MARKER in path.name
        )
    )


def _immediate_subdirectory_names(base: Path) -> tuple[str, ...]:
    if not base.exists():
        return ()
    return tuple(sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")))


def _scan_referenced_root_names(
    *, artifact_root: Path, candidate_names: frozenset[str], read_size: int
) -> tuple[frozenset[str], tuple[str, ...]]:
    base = Path(artifact_root)
    if not base.exists() or not candidate_names:
        return frozenset(), ()
    longest = max((len(name) for name in candidate_names), default=0)
    pattern = re.compile("|".join(re.escape(name) for name in sorted(candidate_names, key=len, reverse=True)))
    found: set[str] = set()
    blocking: list[str] = []
    for artifact_path in sorted(base.rglob("*.json")):
        try:
            handle = artifact_path.open("r", encoding="utf-8")
        except OSError:
            blocking.append(f"unreadable_artifact:{artifact_path.as_posix()}")
            continue
        with handle:
            tail = ""
            while True:
                try:
                    chunk = handle.read(read_size)
                except (OSError, ValueError):
                    blocking.append(f"unreadable_artifact:{artifact_path.as_posix()}")
                    break
                if not chunk:
                    break
                window = tail + chunk
                found.update(name for name in pattern.findall(window) if name in candidate_names)
                tail = window[-(longest - 1):] if longest > 1 and len(window) >= longest - 1 else window
    return frozenset(found), tuple(blocking)


def plan_storage_root_retention(
    *,
    silver_base: Path,
    gold_base: Path,
    artifact_root: Path,
    keep_root_names: frozenset[str] = frozenset({"stocks"}),
    read_size: int = 65536,
) -> StorageRootRetentionPlan:
    """Classify Silver/Gold root namespaces as referenced or reclaimable.

    A root is retained when its directory name is the reserved canonical
    name (``keep_root_names``, the CLI's own default target even before any
    manifest references it) or appears verbatim inside any artifact JSON
    under ``artifact_root``. Every other immediate root directory is
    reclaimable. An unreadable artifact file blocks deletion entirely
    (rather than being silently skipped) because a missed reference would
    make a genuinely referenced root look orphaned.
    """
    if not isinstance(read_size, int) or isinstance(read_size, bool) or read_size < 1:
        raise ValueError("read_size must be a positive integer")
    silver_names = _immediate_subdirectory_names(Path(silver_base))
    gold_names = _immediate_subdirectory_names(Path(gold_base))
    candidate_names = frozenset(silver_names) | frozenset(gold_names)
    referenced, blocking = _scan_referenced_root_names(
        artifact_root=Path(artifact_root), candidate_names=candidate_names, read_size=read_size
    )
    retained_silver = tuple(sorted(n for n in silver_names if n in keep_root_names or n in referenced))
    reclaimable_silver = tuple(sorted(n for n in silver_names if n not in keep_root_names and n not in referenced))
    retained_gold = tuple(sorted(n for n in gold_names if n in keep_root_names or n in referenced))
    reclaimable_gold = tuple(sorted(n for n in gold_names if n not in keep_root_names and n not in referenced))
    orphaned_staging = find_orphaned_staging_paths(Path(silver_base)) + find_orphaned_staging_paths(Path(gold_base))
    return StorageRootRetentionPlan(
        retained_silver_roots=retained_silver,
        reclaimable_silver_roots=reclaimable_silver,
        retained_gold_roots=retained_gold,
        reclaimable_gold_roots=reclaimable_gold,
        orphaned_staging_paths=orphaned_staging,
        blocking_reasons=tuple(blocking),
        deletion_eligible=not blocking,
    )
