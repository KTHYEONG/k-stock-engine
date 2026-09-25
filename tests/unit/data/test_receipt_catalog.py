from __future__ import annotations

import hashlib
import multiprocessing
import shutil
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.schemas import PITDataError


def _payload_file(root: Path, name: str, data: bytes) -> tuple[str, Path]:
    path = root / "payloads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), path


def _entry(
    root: Path,
    *,
    source: str = "krx_daily_market",
    natural_key: str = "2024-01-02",
    as_of: date | None = date(2024, 1, 2),
    fiscal_period: str | None = None,
    status: EvidenceStatus = EvidenceStatus.SUCCESS,
    body: bytes = b'{"records": [1]}',
    retrieved_at: datetime = datetime(2024, 1, 3, tzinfo=UTC),
) -> ReceiptIndexEntry:
    digest = hashlib.sha256(body).hexdigest()
    content_hash, payload_path = _payload_file(
        root,
        f"{source}-{natural_key}-{retrieved_at.isoformat()}-{digest}.json",
        body,
    )
    return ReceiptIndexEntry(
        source=source,
        natural_key=natural_key,
        as_of=as_of,
        fiscal_period=fiscal_period,
        status=status,
        content_hash=content_hash,
        retrieved_at=retrieved_at,
        payload_path=payload_path,
    )


def test_publish_latest_and_entry_stream_use_sqlite_catalog(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    entry = _entry(tmp_path)

    revision = catalog.publish((entry,))

    assert revision.sequence == 1
    assert revision.row_count == 1
    assert revision.path == catalog_root.resolve() / "catalog.sqlite3"
    found = catalog.latest(source=entry.source, natural_keys=[entry.natural_key, "missing"])
    assert found[entry.natural_key] == replace(entry, payload_path=entry.payload_path.resolve())
    assert catalog.latest(source="other", natural_keys=[entry.natural_key]) == {}
    assert list(catalog.entries(source=entry.source)) == list(found.values())
    assert list(catalog.entries(source="other")) == []
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        stored_path = connection.execute("SELECT payload_path FROM receipts").fetchone()[0]
    assert not Path(stored_path).is_absolute()
    assert not (catalog_root / "latest.json").exists()
    assert not (catalog_root / ".publish.lock").exists()
    assert list(catalog_root.glob("*.json")) == []
    assert list(catalog_root.glob(".*.tmp")) == []


def test_missing_database_is_an_empty_catalog(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)

    assert catalog.latest(source="source", natural_keys=["key"]) == {}
    assert catalog.successful_keys(source="source") == frozenset()
    assert list(catalog.entries(source="source")) == []
    assert not catalog_root.exists()


def test_unsuccessful_receipt_is_not_successful_coverage(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entries = tuple(
        _entry(
            tmp_path,
            natural_key=f"key-{index}",
            status=status,
            body=f"body-{index}".encode(),
        )
        for index, status in enumerate(
            (
                EvidenceStatus.EMPTY,
                EvidenceStatus.PROVIDER_UNAVAILABLE,
                EvidenceStatus.PROVIDER_ERROR,
                EvidenceStatus.EXTRACTION_FAILED,
            )
        )
    )
    catalog.publish(entries)

    assert catalog.successful_keys(source="krx_daily_market") == frozenset()
    assert catalog.latest(source="krx_daily_market", natural_keys=["key-0"])["key-0"].status == EvidenceStatus.EMPTY


def test_newer_receipt_replaces_older_and_older_is_ignored(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    older = _entry(tmp_path, body=b"v1", retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))
    newer = _entry(tmp_path, body=b"v2", retrieved_at=datetime(2024, 1, 4, tzinfo=UTC))
    catalog.publish((older,))

    replacement = catalog.publish((newer,))
    ignored = catalog.publish((older,))

    found = catalog.latest(source=newer.source, natural_keys=[newer.natural_key])[newer.natural_key]
    assert found == replace(newer, payload_path=newer.payload_path.resolve())
    assert replacement.sequence == 2
    assert replacement.row_count == 1
    assert ignored.sequence == 3
    assert ignored.row_count == 1


def test_same_instant_different_hash_conflicts_without_mutation(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    moment = datetime(2024, 1, 3, tzinfo=UTC)
    first = _entry(tmp_path, body=b"first", retrieved_at=moment)
    original = catalog.publish((first,))
    conflicting = _entry(tmp_path, body=b"second", retrieved_at=moment)

    with pytest.raises(PITDataError, match="conflict"):
        catalog.publish((conflicting,))

    assert catalog.latest(source=first.source, natural_keys=[first.natural_key])[first.natural_key].content_hash == first.content_hash
    follow_up = catalog.publish((_entry(tmp_path, natural_key="new-key", retrieved_at=moment),))
    assert follow_up.sequence == original.sequence + 1


def test_batch_validation_failure_is_atomic(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    original = _entry(tmp_path, natural_key="original")
    first_revision = catalog.publish((original,))
    valid = _entry(tmp_path, natural_key="valid")
    invalid = replace(_entry(tmp_path, natural_key="invalid"), content_hash="f" * 64)

    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog.publish((valid, invalid))

    assert catalog.latest(
        source=original.source,
        natural_keys=[original.natural_key, valid.natural_key, invalid.natural_key],
    ) == {original.natural_key: replace(original, payload_path=original.payload_path.resolve())}
    follow_up = catalog.publish((_entry(tmp_path, natural_key="after-failure"),))
    assert follow_up.sequence == first_revision.sequence + 1


def test_paths_are_relative_and_survive_catalog_move(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    moved_root = tmp_path / "moved"
    entry = _entry(original_root)
    ReceiptCatalog(original_root / "catalog").publish((entry,))
    shutil.copytree(original_root, moved_root)

    found = ReceiptCatalog(moved_root / "catalog").latest(
        source=entry.source,
        natural_keys=[entry.natural_key],
    )[entry.natural_key]

    assert found.payload_path == moved_root / "payloads" / found.payload_path.name
    assert found.payload_path.is_absolute()
    assert found.payload_path.is_file()


def test_payload_outside_bronze_root_is_rejected_before_transaction(tmp_path: Path) -> None:
    scope_root = tmp_path / "scope"
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    entry = replace(
        _entry(scope_root, natural_key="inside"),
        payload_path=outside,
    )
    catalog = ReceiptCatalog(scope_root / "catalog")

    with pytest.raises(PITDataError, match="outside the Bronze root"):
        catalog.publish((entry,))

    assert not (scope_root / "catalog" / "catalog.sqlite3").exists()


def test_successful_keys_apply_fiscal_floor_only_to_valid_periods(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entries = (
        _entry(
            tmp_path,
            source="financial_facts",
            natural_key="2015Q4",
            as_of=date(2016, 1, 2),
            fiscal_period="2015Q4",
            retrieved_at=datetime(2016, 1, 3, tzinfo=UTC),
        ),
        _entry(
            tmp_path,
            source="financial_facts",
            natural_key="2016Q1",
            as_of=date(2016, 5, 16),
            fiscal_period="2016Q1",
            retrieved_at=datetime(2016, 5, 17, tzinfo=UTC),
        ),
        _entry(
            tmp_path,
            source="financial_facts",
            natural_key="no-period",
            as_of=date(2016, 5, 16),
            fiscal_period=None,
            retrieved_at=datetime(2016, 5, 17, tzinfo=UTC),
        ),
        _entry(
            tmp_path,
            source="financial_facts",
            natural_key="failed",
            fiscal_period="2016Q2",
            status=EvidenceStatus.EXTRACTION_FAILED,
            retrieved_at=datetime(2016, 8, 1, tzinfo=UTC),
        ),
    )
    catalog.publish(entries)

    assert catalog.successful_keys(source="financial_facts", fiscal_start="2016Q1") == frozenset({"2016Q1"})
    assert catalog.successful_keys(source="financial_facts") == frozenset({"2015Q4", "2016Q1", "no-period"})


def test_publish_rejects_invalid_identity_missing_payload_and_hash_mismatch(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    with pytest.raises(PITDataError, match="source and natural key"):
        catalog.publish((_entry(tmp_path, source=" "),))
    with pytest.raises(PITDataError, match="source and natural key"):
        catalog.publish((_entry(tmp_path, natural_key=" "),))
    with pytest.raises(PITDataError, match="missing"):
        catalog.publish((replace(_entry(tmp_path), payload_path=tmp_path / "absent.json"),))
    content_hash, payload_path = _payload_file(tmp_path, "tampered.json", b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog.publish((replace(_entry(tmp_path), content_hash="f" * 64, payload_path=payload_path),))
    assert content_hash
    assert not catalog.latest(source="krx_daily_market", natural_keys=["absent", " "])
    assert not (tmp_path / "catalog").exists()


def test_database_path_must_be_a_file(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    (catalog_root / "catalog.sqlite3").mkdir(parents=True)

    with pytest.raises(PITDataError, match="not a file"):
        ReceiptCatalog(catalog_root).latest(source="source", natural_keys=["key"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("DROP TABLE catalog_metadata", "metadata schema"),
        ("DELETE FROM catalog_metadata", "metadata is missing"),
        ("ALTER TABLE receipts ADD COLUMN extra TEXT", "entry schema"),
        ("DROP INDEX idx_receipts_source_status", "status index"),
        ("UPDATE catalog_metadata SET row_count = -1", "metadata is invalid"),
    ],
)
def test_invalid_catalog_schema_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    catalog.publish((_entry(tmp_path),))
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute(mutation)

    with pytest.raises(PITDataError, match=message):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])


def test_corrupt_row_values_fail_closed(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    catalog.publish((_entry(tmp_path),))
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE receipts SET status = 'unknown'")

    with pytest.raises(PITDataError, match="corrupt entry"):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])


def test_payload_directory_is_not_a_valid_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload-directory"
    payload_path.mkdir()
    entry = replace(_entry(tmp_path), payload_path=payload_path)

    with pytest.raises(PITDataError, match="missing"):
        ReceiptCatalog(tmp_path / "catalog").publish((entry,))


def test_payload_open_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(tmp_path)
    real_open = Path.open

    def _fail_for_payload(path: Path, *args: object, **kwargs: object) -> object:
        if path == entry.payload_path:
            raise PermissionError("simulated payload read denial")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", _fail_for_payload)
    with pytest.raises(PITDataError, match="payload is missing"):
        ReceiptCatalog(tmp_path / "catalog").publish((entry,))


def test_catalog_directory_creation_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(tmp_path)
    catalog_root = tmp_path / "catalog"
    real_mkdir = Path.mkdir

    def _fail_for_catalog(path: Path, *args: object, **kwargs: object) -> None:
        if path == catalog_root:
            raise OSError("simulated directory failure")
        real_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", _fail_for_catalog)
    with pytest.raises(PITDataError, match="directory could not be created"):
        ReceiptCatalog(catalog_root).publish((entry,))


def test_duplicate_keys_in_one_batch_keep_the_latest_and_reject_equal_hash_conflicts(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    older = _entry(tmp_path, body=b"older", retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))
    newer = _entry(tmp_path, body=b"newer", retrieved_at=datetime(2024, 1, 4, tzinfo=UTC))
    same_time = datetime(2024, 1, 4, tzinfo=UTC)
    conflict = _entry(tmp_path, body=b"conflict", retrieved_at=same_time)

    revision = catalog.publish((older, newer))

    assert revision.row_count == 1
    assert catalog.latest(source=newer.source, natural_keys=[newer.natural_key])[newer.natural_key].content_hash == newer.content_hash
    with pytest.raises(PITDataError, match="conflict"):
        catalog.publish((newer, conflict))


def test_equal_timestamp_and_hash_is_idempotent(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)

    first = catalog.publish((entry,))
    second = catalog.publish((entry,))

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.row_count == 1


def test_unknown_schema_fails_closed_for_every_operation(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    catalog.publish((_entry(tmp_path),))
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE catalog_metadata SET schema_version = 999")

    with pytest.raises(PITDataError, match="schema_version"):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])
    with pytest.raises(PITDataError, match="schema_version"):
        catalog.successful_keys(source="krx_daily_market")
    with pytest.raises(PITDataError, match="schema_version"):
        catalog.entries(source="krx_daily_market")
    with pytest.raises(PITDataError, match="schema_version"):
        catalog.publish(())


def test_corrupt_database_fails_closed(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    (catalog_root / "catalog.sqlite3").write_bytes(b"not a sqlite database")

    with pytest.raises(PITDataError, match=r"unreadable|corrupt"):
        ReceiptCatalog(catalog_root).latest(source="source", natural_keys=["key"])


def _publish_concurrent_batch(
    catalog_root: str,
    payload_root: str,
    worker_index: int,
    count: int,
    barrier: object,
) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

    root = Path(catalog_root)
    payloads = Path(payload_root)
    entries = []
    for offset in range(count):
        key = f"worker-{worker_index}-key-{offset:03d}"
        payload_path = payloads / f"{key}.json"
        entries.append(
            ReceiptIndexEntry(
                source="concurrent",
                natural_key=key,
                as_of=None,
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                payload_path=payload_path,
            )
        )
    barrier.wait(timeout=20)  # type: ignore[attr-defined]
    ReceiptCatalog(root).publish(entries)


def test_concurrent_publishers_do_not_lose_entries(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    payload_root = tmp_path / "payloads"
    payload_root.mkdir()
    per_worker = 200
    for worker_index in range(2):
        for offset in range(per_worker):
            (payload_root / f"worker-{worker_index}-key-{offset:03d}.json").write_bytes(b'{"records": []}')

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_publish_concurrent_batch,
            args=(str(catalog_root), str(payload_root), worker_index, per_worker, barrier),
        )
        for worker_index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
    assert [process.exitcode for process in processes] == [0, 0]

    catalog = ReceiptCatalog(catalog_root)
    keys = [f"worker-{worker}-key-{offset:03d}" for worker in range(2) for offset in range(per_worker)]
    assert len(catalog.latest(source="concurrent", natural_keys=keys)) == 400
    assert sum(1 for _ in catalog.entries(source="concurrent")) == 400
    assert catalog.publish(()).sequence == 3


def test_publish_with_empty_batch_creates_database_and_increments_sequence(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)

    first = catalog.publish(())
    second = catalog.publish(())

    assert first.sequence == 1
    assert first.row_count == 0
    assert first.path == catalog_root.resolve() / "catalog.sqlite3"
    assert second.sequence == 2
    assert second.row_count == 0
    assert catalog.successful_keys(source="source") == frozenset()
