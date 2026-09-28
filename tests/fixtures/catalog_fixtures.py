"""Catalog seeding helpers for tests that need catalog state without a collector."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from src.core.pit import EvidenceKind
from src.data.receipt_catalog import (
    BlobEntry,
    CatalogRevision,
    EvidenceStatus,
    ReceiptCatalog,
    ReceiptIndexEntry,
)

__all__ = ["blob_for", "register_fact_page", "seed_corp_code_bridge", "seed_receipts"]


def blob_for(entry: ReceiptIndexEntry, *, usable: bool = True) -> BlobEntry:
    """The blob a receipt must reference, derived from the same stored payload."""
    return BlobEntry(
        content_hash=entry.content_hash,
        kind=EvidenceKind.DAILY_MARKET,
        source=entry.source,
        usable=usable,
        unusable_reason=None if usable else "seeded unusable blob",
        retrieved_at=entry.retrieved_at,
        payload_path=entry.payload_path,
    )


def seed_receipts(
    catalog: ReceiptCatalog,
    entries: Sequence[ReceiptIndexEntry],
    *,
    retrieved_at: datetime | None = None,
    extra: Sequence[ReceiptIndexEntry] = (),
) -> CatalogRevision:
    """Publish receipts together with the blobs they reference.

    Tests seed the catalog with receipts whose payload files already exist, so
    the blob is derived from the receipt rather than written separately.
    """
    published = [*entries, *extra]
    normalized = [
        entry if retrieved_at is None else replace(entry, retrieved_at=retrieved_at) for entry in published
    ]
    return catalog.publish(normalized, blobs=[blob_for(entry) for entry in normalized])


def seed_corp_code_bridge(
    bronze_root: Path, raw: bytes, *, retrieved_at: datetime = datetime(2025, 1, 1, tzinfo=UTC)
) -> str:
    """Store a corp-code bridge payload and publish it as the catalog's bridge receipt.

    Readers take the bridge only from the catalog, so a fixture that just wrote
    the file would look like a scope without a bridge.
    """
    digest = hashlib.sha256(raw).hexdigest()
    payload_path = Path(bronze_root) / "dart_corp_codes" / digest / "payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    if not payload_path.exists():
        payload_path.write_bytes(raw)
    entry = ReceiptIndexEntry(
        source="dart_corp_codes",
        natural_key="dart_corp_codes",
        as_of=None,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        content_hash=digest,
        retrieved_at=retrieved_at,
        payload_path=payload_path,
    )
    ReceiptCatalog(Path(bronze_root) / "catalog").publish(
        [entry], blobs=[replace(blob_for(entry), kind=EvidenceKind.SECURITY_MASTER)]
    )
    return digest


def register_fact_page(bronze_root: Path, receipt_dir: Path, *, natural_key: str | None = None) -> str:
    """Catalog the ``payload.json`` stored in one on-disk ``financial_facts`` page directory.

    Fact refresh reads only catalogued pages. The catalog hashes the actual
    payload, so a fixture may still hold a ``receipt.json`` that disagrees with it
    (tampered, malformed): the refresh must reject that at read time.
    """
    payload_path = Path(receipt_dir) / "payload.json"
    digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    moment = datetime(2016, 1, 1, tzinfo=UTC)
    entry = ReceiptIndexEntry(
        source="financial_facts", natural_key=natural_key or Path(receipt_dir).name, as_of=None,
        fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=digest,
        retrieved_at=moment, payload_path=payload_path,
    )
    ReceiptCatalog(Path(bronze_root) / "catalog").publish(
        [entry], blobs=[replace(blob_for(entry), kind=EvidenceKind.FINANCIAL_FACTS)]
    )
    return digest
