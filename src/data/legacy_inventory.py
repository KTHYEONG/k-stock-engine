"""Legacy inventory, migration artifact, and storage reset."""
from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.data.schemas import BronzeReceipt, EvidenceKind


class LegacyDisposition(StrEnum):
    REUSE_AS_BRONZE = "reuse_as_bronze"
    COLLECT_MISSING = "collect_missing"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class LegacyInventoryItem:
    relative_path: str
    disposition: LegacyDisposition


@dataclass(frozen=True, slots=True)
class LegacyInventory:
    entries: tuple[LegacyInventoryItem, ...]


@dataclass(frozen=True, slots=True)
class MigrationEntry:
    source_path: str
    source_hash: str
    receipt_path: str
    coverage: str
    decision: str


@dataclass(frozen=True, slots=True)
class MigrationArtifact:
    receipts: dict[EvidenceKind, BronzeReceipt]
    entries: tuple[MigrationEntry, ...]
    content_hash: str
    verified: bool = True

    @classmethod
    def empty_verified(cls, data_root: Path) -> MigrationArtifact:
        seed = f"empty-verified:{Path(data_root).as_posix()}".encode()
        return cls(
            receipts={},
            entries=(),
            content_hash=hashlib.sha256(seed).hexdigest(),
            verified=True,
        )


_REUSE_SEEDS: frozenset[str] = frozenset(
    {
        "evidence/stocks/calendar_20131213_20260311.json",
        "evidence/stocks/master_20160104_20260310_historical_v1.json",
        "evidence/stocks/krx-bars-20160104-20260310_backfill_v1.json",
        "evidence/stocks/dart_disclosures_20160101_20260310_v1.json",
        "evidence/stocks/corporate_actions_20160104_20260310_v2.json",
        "evidence/stocks/costs/kis_lifetime_preferential_counterfactual_v1.json",
    }
)

_PURGE_RELATIVES: tuple[str, ...] = (
    "canonical",
    "derived",
    "catalog",
    "evidence",
    "trading_state.db",
)


def inspect_legacy_data(data_root: Path) -> LegacyInventory:
    root = Path(data_root)
    items: dict[str, LegacyInventoryItem] = {}
    if root.exists():
        found: set[str] = set()
        for child in sorted(root.iterdir(), key=lambda p: p.as_posix()):
            rel = child.relative_to(root).as_posix()
            if child.is_dir() and not child.is_symlink():
                for sub in sorted(child.rglob("*"), key=lambda p: p.as_posix()):
                    sub_rel = sub.relative_to(root).as_posix()
                    found.add(sub_rel)
                    if sub_rel in _REUSE_SEEDS:
                        items[sub_rel] = LegacyInventoryItem(sub_rel, LegacyDisposition.REUSE_AS_BRONZE)
                    elif sub.is_dir():
                        items.setdefault(
                            sub_rel, LegacyInventoryItem(sub_rel, LegacyDisposition.REMOVE)
                        )
                    else:
                        items[sub_rel] = LegacyInventoryItem(sub_rel, LegacyDisposition.REMOVE)
                items.setdefault(rel, LegacyInventoryItem(rel, LegacyDisposition.REMOVE))
                if rel in _REUSE_SEEDS:
                    items[rel] = LegacyInventoryItem(rel, LegacyDisposition.REUSE_AS_BRONZE)
            else:
                found.add(rel)
                if rel in _REUSE_SEEDS:
                    items[rel] = LegacyInventoryItem(rel, LegacyDisposition.REUSE_AS_BRONZE)
                else:
                    items[rel] = LegacyInventoryItem(rel, LegacyDisposition.REMOVE)
        for seed in sorted(_REUSE_SEEDS):
            if seed not in found and seed not in items and not (root / seed).exists():
                items[seed] = LegacyInventoryItem(seed, LegacyDisposition.COLLECT_MISSING)
    else:
        for seed in sorted(_REUSE_SEEDS):
            items[seed] = LegacyInventoryItem(seed, LegacyDisposition.COLLECT_MISSING)
    ordered = tuple(items[key] for key in sorted(items))
    return LegacyInventory(entries=ordered)


def _has_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    if path.is_dir() and not path.is_symlink():
        for child in path.iterdir():
            if child.is_symlink() or (child.is_dir() and _has_symlink(child)):
                return True
    return False


def purge_legacy_data(
    data_root: Path, migration: MigrationArtifact, *, confirm_purge: bool
) -> tuple[Path, ...]:
    if not confirm_purge:
        raise ValueError("confirm_purge must be True to delete legacy outputs")
    if not migration.verified or not migration.content_hash:
        raise ValueError("unverified migration artifact: missing receipts verification")
    receipts: Mapping[EvidenceKind, BronzeReceipt] = migration.receipts
    if receipts is None:
        raise ValueError("missing migration receipts")
    for receipt in receipts.values():
        if not receipt.payload_path.exists() or not receipt.metadata_path.exists():
            raise ValueError(f"missing migration receipt payload: {receipt.payload_path}")
    root = Path(data_root)
    root_resolved = root.resolve()
    removed: list[Path] = []
    for rel in _PURGE_RELATIVES:
        target = root / rel
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink() or _has_symlink(target):
            raise ValueError(f"refusing to purge symlink target: {target}")
        try:
            resolved = target.resolve()
        except OSError as exc:
            raise ValueError(f"unresolvable purge target: {target}") from exc
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise ValueError(f"purge target outside data_root: {target}")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(target)
    return tuple(sorted(removed, key=lambda p: p.as_posix()))
