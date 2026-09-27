"""Catalog seeding helpers for tests that need catalog state without a collector."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from src.core.pit import EvidenceKind
from src.data.receipt_catalog import (
    BlobEntry,
    CatalogRevision,
    ReceiptCatalog,
    ReceiptIndexEntry,
)

__all__ = ["blob_for", "seed_receipts"]


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
