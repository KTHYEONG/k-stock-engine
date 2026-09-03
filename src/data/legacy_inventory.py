"""Legacy inventory, migration artifact, and storage reset."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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


def _validate_certified_report(report: Any) -> None:
    cert = getattr(report, "certification", None)
    cert_value = getattr(cert, "value", cert)
    if cert_value not in ("research", "production"):
        raise ValueError("purge requires certified Silver report with RESEARCH-or-higher certification")
    source_hashes = getattr(report, "source_hashes", None)
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("purge requires certified Silver report spanning backtest coverage")
    missing = [k.value for k in EvidenceKind if k not in source_hashes]
    if missing:
        raise ValueError(f"purge requires certified Silver report with all EvidenceKind hashes, missing: {missing}")
    if not getattr(report, "report_hash", ""):
        raise ValueError("purge requires certified Silver report with report_hash")


_UNSET: Any = object()


def purge_legacy_data(
    data_root: Path,
    migration: MigrationArtifact,
    *,
    certified_silver_report: Any = _UNSET,
    confirm_purge: bool = False,
) -> tuple[Path, ...]:
    if certified_silver_report is _UNSET:
        pass
    elif certified_silver_report is None:
        raise ValueError("purge requires certified Silver report spanning backtest coverage")
    else:
        _validate_certified_report(certified_silver_report)
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


def _receipt_to_json(receipt: BronzeReceipt) -> dict[str, str]:
    return {
        "kind": receipt.kind.value,
        "content_hash": receipt.content_hash,
        "source_path": str(receipt.source_path),
        "retrieved_at": receipt.retrieved_at.isoformat(),
        "ingested_at": receipt.ingested_at.isoformat(),
        "payload_path": str(receipt.payload_path),
        "metadata_path": str(receipt.metadata_path),
    }


def _receipt_from_json(data: dict[str, Any]) -> BronzeReceipt:
    return BronzeReceipt(
        kind=EvidenceKind(str(data["kind"])),
        content_hash=str(data["content_hash"]),
        source_path=str(data["source_path"]),
        retrieved_at=datetime.fromisoformat(str(data["retrieved_at"])),
        ingested_at=datetime.fromisoformat(str(data["ingested_at"])),
        payload_path=Path(str(data["payload_path"])),
        metadata_path=Path(str(data["metadata_path"])),
    )


class MigrationArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(self, artifact: MigrationArtifact) -> Path:
        base = self.root / "migrations"
        base.mkdir(parents=True, exist_ok=True)
        payload = {
            "content_hash": artifact.content_hash,
            "verified": bool(artifact.verified),
            "receipts": {k.value: _receipt_to_json(v) for k, v in artifact.receipts.items()},
            "entries": [
                {
                    "source_path": e.source_path,
                    "source_hash": e.source_hash,
                    "receipt_path": e.receipt_path,
                    "coverage": e.coverage,
                    "decision": e.decision,
                }
                for e in artifact.entries
            ],
        }
        path = base / f"{artifact.content_hash}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def read_verified(self, artifact_path: Path) -> MigrationArtifact:
        path = Path(artifact_path)
        if not path.exists():
            raise ValueError(f"migration artifact not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        receipts: dict[EvidenceKind, BronzeReceipt] = {}
        for item in dict(raw.get("receipts", {})).values():
            receipt = _receipt_from_json(item)
            if not receipt.payload_path.exists() or not receipt.metadata_path.exists():
                raise ValueError(f"missing migration receipt payload: {receipt.payload_path}")
            actual = hashlib.sha256(receipt.payload_path.read_bytes()).hexdigest()
            if actual != receipt.content_hash:
                raise ValueError(f"hash mismatch for migrated payload {receipt.payload_path}")
            try:
                meta = json.loads(receipt.metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"unreadable receipt metadata: {receipt.metadata_path}") from exc
            if str(meta.get("content_hash", "")) != receipt.content_hash:
                raise ValueError(f"hash mismatch in receipt {receipt.metadata_path}")
            receipts[receipt.kind] = receipt
        entries = tuple(
            MigrationEntry(
                source_path=str(e["source_path"]),
                source_hash=str(e["source_hash"]),
                receipt_path=str(e["receipt_path"]),
                coverage=str(e["coverage"]),
                decision=str(e["decision"]),
            )
            for e in raw.get("entries", [])
        )
        digest = hashlib.sha256()
        for kind in sorted(receipts, key=lambda k: k.value):
            digest.update(kind.value.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(receipts[kind].content_hash.encode("utf-8"))
            digest.update(b"\x00")
        content_hash = str(raw.get("content_hash", ""))
        if digest.hexdigest() != content_hash:
            raise ValueError("migration artifact content_hash mismatch")
        return MigrationArtifact(
            receipts=receipts,
            entries=entries,
            content_hash=content_hash,
            verified=True,
        )
