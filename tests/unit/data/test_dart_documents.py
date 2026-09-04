from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from src.data.dart_documents import DartDocumentStore


def test_store_archive_is_content_addressed_and_idempotent(tmp_path) -> None:
    store = DartDocumentStore(tmp_path / "bronze")
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
    assert second == first
    assert first.payload_path.read_bytes() == archive
    assert first.metadata_path.exists()


def test_store_archive_rejects_invalid_receipt_and_empty_payload(tmp_path) -> None:
    store = DartDocumentStore(tmp_path / "bronze")
    now = datetime.now(UTC)

    with pytest.raises(ValueError, match="14-digit"):
        store.store_archive(b"archive", rcept_no="bad", retrieved_at=now)
    with pytest.raises(ValueError, match="must not be empty"):
        store.store_archive(b"", rcept_no="20200102000001", retrieved_at=now)


def test_store_archive_detects_tampered_existing_payload(tmp_path) -> None:
    store = DartDocumentStore(tmp_path / "bronze")
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
