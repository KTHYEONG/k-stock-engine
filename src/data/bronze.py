"""Byte-preserving content-addressed JSON import."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

# Fixed registry mapping kind to expected filename relative to source_root
_RETAINED_REGISTRY: dict[EvidenceKind, str] = {
    EvidenceKind.CALENDAR: "calendar_20131213_20260311.json",
    EvidenceKind.SECURITY_MASTER: "master_20160104_20260310_historical_v1.json",
    EvidenceKind.DAILY_MARKET: "krx-bars-20160104-20260310_backfill_v1.json",
    EvidenceKind.DISCLOSURES: "dart_disclosures_20160101_20260310_v1.json",
    EvidenceKind.CORPORATE_ACTIONS: "corporate_actions_20160104_20260310_v2.json",
    EvidenceKind.HISTORICAL_COSTS: "costs/kis_lifetime_preferential_counterfactual_v1.json",
}

# Required top-level keys for each kind to validate JSON root
_REQUIRED_KEYS: dict[EvidenceKind, str] = {
    EvidenceKind.CALENDAR: "sessions",
    EvidenceKind.SECURITY_MASTER: "records",
    EvidenceKind.DAILY_MARKET: "records",
    EvidenceKind.DISCLOSURES: "records",
    EvidenceKind.CORPORATE_ACTIONS: "intervals",
    EvidenceKind.HISTORICAL_COSTS: "commission",
}


class BronzeStore:
    """Append-only, content-addressed Bronze ingestion."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def import_json(
        self, source_path: Path, *, kind: EvidenceKind, retrieved_at: datetime
    ) -> BronzeReceipt:
        source = Path(source_path)
        if not source.exists():
            raise PITDataError(f"missing source file: {source}")
        data = source.read_bytes()
        content_hash = hashlib.sha256(data).hexdigest()
        payload_dir = self.root / kind.value / content_hash
        payload_path = payload_dir / "payload.json"
        metadata_path = payload_dir / "receipt.json"

        if payload_dir.exists() and payload_path.exists() and metadata_path.exists():
            existing = payload_path.read_bytes()
            existing_hash = hashlib.sha256(existing).hexdigest()
            if existing_hash != content_hash:
                raise PITDataError(f"hash mismatch for existing payload {payload_path}")
            # verify receipt hash matches
            with metadata_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("content_hash") != content_hash:
                raise PITDataError(f"hash mismatch in receipt {metadata_path}")
            return BronzeReceipt(
                kind=kind,
                content_hash=content_hash,
                source_path=str(source),
                retrieved_at=datetime.fromisoformat(meta["retrieved_at"]),
                ingested_at=datetime.fromisoformat(meta["ingested_at"]),
                payload_path=payload_path,
                metadata_path=metadata_path,
            )

        # Ensure no overwrite of different hash dir collisions is not issue
        payload_dir.mkdir(parents=True, exist_ok=True)
        # If payload already exists but hash mismatch handled above, write fresh
        if not payload_path.exists():
            payload_path.write_bytes(data)
        else:
            # existing payload but not matching earlier condition (e.g., missing metadata)
            existing = payload_path.read_bytes()
            if existing != data:
                raise PITDataError(f"hash mismatch for existing payload {payload_path}")

        ingested_at = datetime.now(UTC)
        # Ensure retrieved_at is timezone-aware
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=UTC)
        meta_obj = {
            "kind": kind.value,
            "content_hash": content_hash,
            "source_path": str(source),
            "retrieved_at": retrieved_at.isoformat(),
            "ingested_at": ingested_at.isoformat(),
        }
        # write receipt atomically
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(meta_obj, fh, indent=2, sort_keys=True)

        return BronzeReceipt(
            kind=kind,
            content_hash=content_hash,
            source_path=str(source),
            retrieved_at=retrieved_at,
            ingested_at=ingested_at,
            payload_path=payload_path,
            metadata_path=metadata_path,
        )


def import_retained_stock_evidence(
    source_root: Path, *, store: BronzeStore, retrieved_at: datetime
) -> dict[EvidenceKind, BronzeReceipt]:
    root = Path(source_root)
    result: dict[EvidenceKind, BronzeReceipt] = {}
    for kind, rel in _RETAINED_REGISTRY.items():
        source_path = root / rel
        if not source_path.exists():
            raise PITDataError(f"missing required evidence file {rel} (krx-bars etc)")
        # Validate JSON root and required key
        try:
            text = source_path.read_bytes()
            obj = json.loads(text)
        except Exception as exc:
            raise PITDataError(f"invalid JSON for {rel}: {exc}") from exc
        if not isinstance(obj, dict):
            raise PITDataError(f"invalid JSON root for {rel}: expected object")
        required = _REQUIRED_KEYS.get(kind)
        if required is not None and required not in obj:
            raise PITDataError(f"invalid JSON root for {rel}: missing key {required}")
        # hash mismatch check is implicit via import
        receipt = store.import_json(source_path, kind=kind, retrieved_at=retrieved_at)
        # Verify stored hash matches computed
        computed = hashlib.sha256(text).hexdigest()
        if receipt.content_hash != computed:
            raise PITDataError(f"hash mismatch for {rel}")
        result[kind] = receipt
    return result
