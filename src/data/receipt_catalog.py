"""Scope-local receipt index with latest-state lookup by source natural key."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from src.data.schemas import PITDataError

__all__ = [
    "CatalogRevision",
    "EvidenceStatus",
    "ReceiptCatalog",
    "ReceiptIndexEntry",
]

_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")
_POINTER_NAME = "latest.json"


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


class EvidenceStatus(StrEnum):
    """Observed provider outcome retained separately from successful evidence coverage."""

    SUCCESS = "success"
    EMPTY = "empty"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass(frozen=True, slots=True)
class ReceiptIndexEntry:
    """Natural-key index row for one retained raw provider response.

    The index makes collection planning depend on validated receipt state rather
    than repeated recursive reads of content-addressed JSON payloads.
    """

    source: str
    natural_key: str
    as_of: date | None
    fiscal_period: str | None
    status: EvidenceStatus
    content_hash: str
    retrieved_at: datetime
    payload_path: Path


@dataclass(frozen=True, slots=True)
class CatalogRevision:
    """Immutable summary of one atomically published receipt-catalog revision."""

    content_hash: str
    row_count: int
    path: Path


def _entry_to_dict(entry: ReceiptIndexEntry) -> dict[str, str | None]:
    return {
        "source": entry.source,
        "natural_key": entry.natural_key,
        "as_of": entry.as_of.isoformat() if entry.as_of is not None else None,
        "fiscal_period": entry.fiscal_period,
        "status": entry.status.value,
        "content_hash": entry.content_hash,
        "retrieved_at": entry.retrieved_at.isoformat(),
        "payload_path": str(entry.payload_path),
    }


def _entry_from_dict(raw: dict[str, object]) -> ReceiptIndexEntry:
    as_of_raw = raw.get("as_of")
    retrieved_raw = raw.get("retrieved_at")
    if not isinstance(retrieved_raw, str) or not retrieved_raw.strip():
        raise PITDataError("receipt catalog entry is missing retrieved_at")
    return ReceiptIndexEntry(
        source=str(raw.get("source") or ""),
        natural_key=str(raw.get("natural_key") or ""),
        as_of=date.fromisoformat(str(as_of_raw)) if isinstance(as_of_raw, str) and as_of_raw.strip() else None,
        fiscal_period=str(raw["fiscal_period"]) if raw.get("fiscal_period") not in (None, "") else None,
        status=EvidenceStatus(str(raw.get("status") or "")),
        content_hash=str(raw.get("content_hash") or ""),
        retrieved_at=datetime.fromisoformat(retrieved_raw),
        payload_path=Path(str(raw.get("payload_path") or "")),
    )


class ReceiptCatalog:
    """Scope-local receipt index with latest-state lookup by source natural key."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _pointer_path(self) -> Path:
        return self._root / _POINTER_NAME

    def _snapshot(self) -> dict[tuple[str, str], ReceiptIndexEntry]:
        pointer = self._pointer_path()
        if not pointer.exists():
            return {}
        try:
            raw = json.loads(pointer.read_text(encoding="utf-8"))
            revision_path = self._root / str(raw["revision"])
            payload = json.loads(revision_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PITDataError("receipt catalog revision is unreadable") from exc
        if not isinstance(payload, list):
            raise PITDataError("receipt catalog revision has invalid schema")
        snapshot: dict[tuple[str, str], ReceiptIndexEntry] = {}
        for item in payload:
            if not isinstance(item, dict):
                raise PITDataError("receipt catalog revision has invalid schema")
            entry = _entry_from_dict(item)
            snapshot[(entry.source, entry.natural_key)] = entry
        return snapshot

    def publish(self, entries: Sequence[ReceiptIndexEntry]) -> CatalogRevision:
        for entry in entries:
            if not entry.source.strip() or not entry.natural_key.strip():
                raise PITDataError("receipt catalog entry requires source and natural key")
            payload_path = Path(entry.payload_path)
            if not payload_path.is_file():
                raise PITDataError(f"receipt catalog payload is missing for {entry.natural_key!r}")
            if hashlib.sha256(payload_path.read_bytes()).hexdigest() != entry.content_hash:
                raise PITDataError(f"receipt catalog hash mismatch for {entry.natural_key!r}")
        snapshot = self._snapshot()
        for entry in entries:
            key = (entry.source, entry.natural_key)
            current = snapshot.get(key)
            if current is not None and current.retrieved_at == entry.retrieved_at and current.content_hash != entry.content_hash:
                raise PITDataError(f"receipt catalog conflict for {entry.natural_key!r}")
            if current is None or entry.retrieved_at >= current.retrieved_at:
                snapshot[key] = entry
        ordered = [_entry_to_dict(snapshot[key]) for key in sorted(snapshot)]
        canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._root.mkdir(parents=True, exist_ok=True)
        revision_path = self._root / f"{content_hash}.json"
        tmp_revision = self._root / f".{content_hash}.tmp"
        tmp_revision.write_text(json.dumps(ordered, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_revision, revision_path)
        tmp_pointer = self._root / ".latest.json.tmp"
        tmp_pointer.write_text(json.dumps({"revision": revision_path.name}, sort_keys=True), encoding="utf-8")
        os.replace(tmp_pointer, self._pointer_path())
        return CatalogRevision(content_hash=content_hash, row_count=len(ordered), path=revision_path)

    def latest(self, *, source: str, natural_keys: Collection[str]) -> Mapping[str, ReceiptIndexEntry]:
        snapshot = self._snapshot()
        wanted = set(natural_keys)
        return {
            key: entry
            for (entry_source, key), entry in snapshot.items()
            if entry_source == source and key in wanted
        }

    def successful_keys(self, *, source: str, fiscal_start: str | None = None) -> frozenset[str]:
        snapshot = self._snapshot()
        floor = _fiscal_key(fiscal_start) if fiscal_start is not None else None
        keys: set[str] = set()
        for (entry_source, key), entry in snapshot.items():
            if entry_source != source or entry.status != EvidenceStatus.SUCCESS:
                continue
            if floor is not None:
                if entry.fiscal_period is None or not _FISCAL_PATTERN.fullmatch(entry.fiscal_period):
                    continue
                if _fiscal_key(entry.fiscal_period) < floor:
                    continue
            keys.add(key)
        return frozenset(keys)
