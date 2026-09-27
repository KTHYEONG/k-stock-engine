"""Immutable Bronze persistence for legacy DART document archives."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.core.pit import EvidenceKind
from src.data.evidence_sources import DART_DOCUMENT_SOURCE
from src.data.receipt_catalog import BlobEntry, CatalogRevision, ReceiptCatalog


@dataclass(frozen=True, slots=True)
class DartDocumentReceipt:
    rcept_no: str
    content_hash: str
    payload_path: Path
    metadata_path: Path
    byte_length: int
    retrieved_at: datetime
    catalog_revision: CatalogRevision


@dataclass(frozen=True, slots=True)
class _StoredArchive:
    """One archive on disk, before its blob is registered."""

    rcept_no: str
    content_hash: str
    payload_path: Path
    metadata_path: Path
    byte_length: int
    retrieved_at: datetime


class DartDocumentStore:
    """Persist legacy ZIP archives under dart_documents/<sha256>/payload.zip.

    Every stored archive is registered as a usable blob of the
    ``dart_documents`` source in the same commit, so the catalog answers
    "which filing documents exist" without a filesystem scan.
    """

    def __init__(self, bronze_root: Path | str, *, catalog: ReceiptCatalog) -> None:
        self.bronze_root = Path(bronze_root)
        self._catalog = catalog

    def store_archive(
        self, archive_bytes: bytes, *, rcept_no: str, retrieved_at: datetime
    ) -> DartDocumentReceipt:
        receipt_no = str(rcept_no or "").strip()
        if len(receipt_no) != 14 or not receipt_no.isdigit():
            raise ValueError("rcept_no must be a 14-digit receipt number")
        if not archive_bytes:
            raise ValueError("archive_bytes must not be empty")
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=UTC)
        content_hash = hashlib.sha256(archive_bytes).hexdigest()
        payload_dir = self.bronze_root / DART_DOCUMENT_SOURCE / content_hash
        payload_path = payload_dir / "payload.zip"
        metadata_path = payload_dir / "receipt.json"
        stored = self._reuse_existing(
            payload_dir, payload_path, metadata_path, content_hash
        ) or self._write(
            payload_dir,
            payload_path,
            metadata_path,
            archive_bytes=archive_bytes,
            content_hash=content_hash,
            receipt_no=receipt_no,
            retrieved_at=retrieved_at,
        )
        revision = self._catalog.publish(
            (),
            blobs=[
                BlobEntry(
                    content_hash=stored.content_hash,
                    kind=EvidenceKind.DISCLOSURES,
                    source=DART_DOCUMENT_SOURCE,
                    usable=True,
                    unusable_reason=None,
                    retrieved_at=stored.retrieved_at,
                    payload_path=stored.payload_path,
                )
            ],
        )
        return DartDocumentReceipt(
            rcept_no=stored.rcept_no,
            content_hash=stored.content_hash,
            payload_path=stored.payload_path,
            metadata_path=stored.metadata_path,
            byte_length=stored.byte_length,
            retrieved_at=stored.retrieved_at,
            catalog_revision=revision,
        )

    def _reuse_existing(
        self, payload_dir: Path, payload_path: Path, metadata_path: Path, content_hash: str
    ) -> _StoredArchive | None:
        """Return the verified stored archive, or ``None`` when it must be written."""
        if not (payload_dir.exists() and payload_path.exists() and metadata_path.exists()):
            return None
        existing = payload_path.read_bytes()
        if hashlib.sha256(existing).hexdigest() != content_hash:
            raise ValueError(f"hash mismatch for existing payload {payload_path}")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Identical content is reused only after receipt/hash verification.
        if meta.get("sha256") != content_hash:
            raise ValueError(f"hash mismatch in receipt {metadata_path}")
        return _StoredArchive(
            rcept_no=str(meta.get("rcept_no", "")),
            content_hash=content_hash,
            payload_path=payload_path,
            metadata_path=metadata_path,
            byte_length=int(meta.get("byte_length", len(existing))),
            retrieved_at=datetime.fromisoformat(str(meta["retrieved_at"])),
        )

    def _write(
        self,
        payload_dir: Path,
        payload_path: Path,
        metadata_path: Path,
        *,
        archive_bytes: bytes,
        content_hash: str,
        receipt_no: str,
        retrieved_at: datetime,
    ) -> _StoredArchive:
        payload_dir.mkdir(parents=True, exist_ok=True)
        if payload_path.exists():
            if hashlib.sha256(payload_path.read_bytes()).hexdigest() != content_hash:
                raise ValueError(f"hash mismatch for existing payload {payload_path}")
        else:
            payload_path.write_bytes(archive_bytes)
        metadata_path.write_text(
            json.dumps(
                {
                    "rcept_no": receipt_no,
                    "source": "document.xml",
                    "retrieved_at": retrieved_at.isoformat(),
                    "sha256": content_hash,
                    "byte_length": len(archive_bytes),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return _StoredArchive(
            rcept_no=receipt_no,
            content_hash=content_hash,
            payload_path=payload_path,
            metadata_path=metadata_path,
            byte_length=len(archive_bytes),
            retrieved_at=retrieved_at,
        )
