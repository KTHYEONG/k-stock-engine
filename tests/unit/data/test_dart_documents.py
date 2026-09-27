from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data.dart_documents import DartDocumentStore
from src.data.evidence_sources import DART_DOCUMENT_SOURCE
from src.data.receipt_catalog import ReceiptCatalog


def _store(root: Path) -> tuple[DartDocumentStore, ReceiptCatalog]:
    catalog = ReceiptCatalog(root / "catalog")
    return DartDocumentStore(root, catalog=catalog), catalog


def test_store_archive_is_content_addressed_and_idempotent(tmp_path) -> None:
    store, catalog = _store(tmp_path / "bronze")
    retrieved_at = datetime(2020, 1, 2, 3, 4, tzinfo=UTC)
    archive = b"zip-fixture"

    first = store.store_archive(archive, rcept_no="20200102000001", retrieved_at=retrieved_at)
    second = store.store_archive(
        archive,
        rcept_no="20200102000001",
        retrieved_at=datetime(2020, 1, 3, tzinfo=UTC),
    )

    expected_hash = hashlib.sha256(archive).hexdigest()
    assert first.content_hash == expected_hash
    assert first.payload_path.read_bytes() == archive
    assert first.metadata_path.exists()
    assert second.rcept_no == first.rcept_no
    assert second.content_hash == first.content_hash
    assert second.byte_length == first.byte_length
    # Identical content is re-verified and re-registered, never written twice.
    assert second.catalog_revision.sequence == first.catalog_revision.sequence + 1
    assert [blob.content_hash for blob in catalog.blobs(source=DART_DOCUMENT_SOURCE)] == [expected_hash]


def test_store_archive_rejects_invalid_receipt_and_empty_payload(tmp_path) -> None:
    store, _catalog = _store(tmp_path / "bronze")
    now = datetime.now(UTC)

    with pytest.raises(ValueError, match="14-digit"):
        store.store_archive(b"archive", rcept_no="bad", retrieved_at=now)
    with pytest.raises(ValueError, match="must not be empty"):
        store.store_archive(b"", rcept_no="20200102000001", retrieved_at=now)


def test_store_archive_detects_tampered_existing_payload(tmp_path) -> None:
    store, _catalog = _store(tmp_path / "bronze")
    receipt = store.store_archive(
        b"archive",
        rcept_no="20200102000001",
        retrieved_at=datetime(2020, 1, 2, tzinfo=UTC),
    )
    receipt.payload_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        store.store_archive(
            b"archive",
            rcept_no="20200102000001",
            retrieved_at=datetime(2020, 1, 2, tzinfo=UTC),
        )


def test_store_archive_detects_a_receipt_naming_another_hash(tmp_path) -> None:
    store, _catalog = _store(tmp_path / "bronze")
    receipt = store.store_archive(
        b"archive",
        rcept_no="20200102000001",
        retrieved_at=datetime(2020, 1, 2, tzinfo=UTC),
    )
    metadata = json.loads(receipt.metadata_path.read_text(encoding="utf-8"))
    metadata["sha256"] = "f" * 64
    receipt.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch in receipt"):
        store.store_archive(
            b"archive",
            rcept_no="20200102000001",
            retrieved_at=datetime(2020, 1, 2, tzinfo=UTC),
        )


def test_store_archive_accepts_naive_retrieved_at_as_utc(tmp_path) -> None:
    store, catalog = _store(tmp_path / "bronze")

    receipt = store.store_archive(b"archive", rcept_no="20200102000001", retrieved_at=datetime(2020, 1, 2))

    assert receipt.retrieved_at.tzinfo is not None
    assert [blob.usable for blob in catalog.blobs(source=DART_DOCUMENT_SOURCE)] == [True]
