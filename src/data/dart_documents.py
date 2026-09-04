"""Immutable Bronze persistence for legacy DART document archives."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DartDocumentReceipt:
    rcept_no: str
    content_hash: str
    payload_path: Path
    metadata_path: Path
    byte_length: int
    retrieved_at: datetime


class DartDocumentStore:
    """Persist legacy ZIP archives under dart_documents/<sha256>/payload.zip."""

    def __init__(self, bronze_root: Path | str) -> None:
        self.bronze_root = Path(bronze_root)

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
        payload_dir = self.bronze_root / "dart_documents" / content_hash
        payload_path = payload_dir / "payload.zip"
        metadata_path = payload_dir / "receipt.json"
        if payload_dir.exists() and payload_path.exists() and metadata_path.exists():
            existing = payload_path.read_bytes()
            if hashlib.sha256(existing).hexdigest() != content_hash:
                raise ValueError(f"hash mismatch for existing payload {payload_path}")
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            # Identical content reused only after receipt/hash verification.
            if meta.get("sha256") != content_hash:
                raise ValueError(f"hash mismatch in receipt {metadata_path}")
            return DartDocumentReceipt(
                rcept_no=str(meta.get("rcept_no", receipt_no)),
                content_hash=content_hash,
                payload_path=payload_path,
                metadata_path=metadata_path,
                byte_length=int(meta.get("byte_length", len(archive_bytes))),
                retrieved_at=datetime.fromisoformat(str(meta["retrieved_at"])),
            )
        payload_dir.mkdir(parents=True, exist_ok=True)
        if payload_path.exists():
            if hashlib.sha256(payload_path.read_bytes()).hexdigest() != content_hash:
                raise ValueError(f"hash mismatch for existing payload {payload_path}")
        else:
            payload_path.write_bytes(archive_bytes)
        meta_obj = {
            "rcept_no": receipt_no,
            "source": "document.xml",
            "retrieved_at": retrieved_at.isoformat(),
            "sha256": content_hash,
            "byte_length": len(archive_bytes),
        }
        metadata_path.write_text(json.dumps(meta_obj, indent=2, sort_keys=True), encoding="utf-8")
        return DartDocumentReceipt(
            rcept_no=receipt_no,
            content_hash=content_hash,
            payload_path=payload_path,
            metadata_path=metadata_path,
            byte_length=len(archive_bytes),
            retrieved_at=retrieved_at,
        )
