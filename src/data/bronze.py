"""Content-addressed Bronze ingestion."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from src.core.pit import BronzeReceipt, EvidenceKind, PITDataError


class BronzeStore:
    """Append-only, content-addressed Bronze ingestion."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def import_bytes(
        self, payload: bytes, *, kind: EvidenceKind, retrieved_at: datetime, source_label: str
    ) -> BronzeReceipt:
        """Persist provider bytes without an external temporary file."""
        if not payload:
            raise PITDataError("cannot import empty Bronze payload")
        content_hash = hashlib.sha256(payload).hexdigest()
        payload_dir = self.root / kind.value / content_hash
        payload_path = payload_dir / "payload.json"
        metadata_path = payload_dir / "receipt.json"
        if payload_dir.exists() and payload_path.exists() and metadata_path.exists():
            if hashlib.sha256(payload_path.read_bytes()).hexdigest() != content_hash:
                raise PITDataError(f"hash mismatch for existing payload {payload_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("content_hash") != content_hash:
                raise PITDataError(f"hash mismatch in receipt {metadata_path}")
            return BronzeReceipt(
                kind=kind,
                content_hash=content_hash,
                source_path=str(metadata.get("source_path", source_label)),
                retrieved_at=datetime.fromisoformat(metadata["retrieved_at"]),
                ingested_at=datetime.fromisoformat(metadata["ingested_at"]),
                payload_path=payload_path,
                metadata_path=metadata_path,
            )
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload)
        ingested_at = datetime.now(UTC)
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=UTC)
        metadata = {
            "kind": kind.value,
            "content_hash": content_hash,
            "source_path": source_label,
            "retrieved_at": retrieved_at.isoformat(),
            "ingested_at": ingested_at.isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return BronzeReceipt(
            kind=kind,
            content_hash=content_hash,
            source_path=source_label,
            retrieved_at=retrieved_at,
            ingested_at=ingested_at,
            payload_path=payload_path,
            metadata_path=metadata_path,
        )
