from __future__ import annotations

import hashlib
import multiprocessing
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.receipt_catalog import (
    BlobEntry,
    CatalogRevision,
    CoverageRange,
    EvidenceStatus,
    ReceiptCatalog,
    ReceiptIndexEntry,
)
from src.core.digest import dataset_digest
from src.core.pit import EvidenceKind, PITDataError


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


def _blob(entry: ReceiptIndexEntry, *, usable: bool = True) -> BlobEntry:
    """The blob every receipt must reference, derived from the same stored payload."""
    return BlobEntry(
        content_hash=entry.content_hash,
        kind=EvidenceKind.DAILY_MARKET,
        source=entry.source,
        usable=usable,
        unusable_reason=None if usable else "provider page is not evidence",
        retrieved_at=entry.retrieved_at,
        payload_path=entry.payload_path,
    )


def _publish(
    catalog: ReceiptCatalog,
    entries: Sequence[ReceiptIndexEntry],
    *,
    ranges: Sequence[CoverageRange] = (),
) -> CatalogRevision:
    """Publish receipts together with the blobs they reference."""
    return catalog.publish(entries, blobs=[_blob(entry) for entry in entries], ranges=ranges)


def _range(
    subject: str,
    start: date,
    end: date,
    *,
    source: str = "ls_investor_flow",
    status: EvidenceStatus = EvidenceStatus.EMPTY,
    content_hash: str | None = None,
    retrieved_at: datetime = datetime(2024, 1, 3, tzinfo=UTC),
) -> CoverageRange:
    return CoverageRange(
        source=source,
        subject=subject,
        start=start,
        end=end,
        status=status,
        content_hash=content_hash,
        retrieved_at=retrieved_at,
    )


def test_publish_latest_and_entry_stream_use_sqlite_catalog(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    entry = _entry(tmp_path)

    revision = _publish(catalog, (entry,))

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
    _publish(catalog, entries)

    assert catalog.successful_keys(source="krx_daily_market") == frozenset()
    assert catalog.latest(source="krx_daily_market", natural_keys=["key-0"])["key-0"].status == EvidenceStatus.EMPTY


def test_newer_receipt_replaces_older_and_older_is_ignored(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    older = _entry(tmp_path, body=b"v1", retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))
    newer = _entry(tmp_path, body=b"v2", retrieved_at=datetime(2024, 1, 4, tzinfo=UTC))
    _publish(catalog, (older,))

    replacement = _publish(catalog, (newer,))
    ignored = _publish(catalog, (older,))

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
    original = _publish(catalog, (first,))
    conflicting = _entry(tmp_path, body=b"second", retrieved_at=moment)

    with pytest.raises(PITDataError, match="conflict"):
        _publish(catalog, (conflicting,))

    assert catalog.latest(source=first.source, natural_keys=[first.natural_key])[first.natural_key].content_hash == first.content_hash
    follow_up = _publish(catalog, (_entry(tmp_path, natural_key="new-key", retrieved_at=moment),))
    assert follow_up.sequence == original.sequence + 1


def test_batch_validation_failure_is_atomic(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    original = _entry(tmp_path, natural_key="original")
    first_revision = _publish(catalog, (original,))
    valid = _entry(tmp_path, natural_key="valid")
    invalid = replace(_entry(tmp_path, natural_key="invalid"), content_hash="f" * 64)

    with pytest.raises(PITDataError, match="hash mismatch"):
        _publish(catalog, (valid, invalid))

    assert catalog.latest(
        source=original.source,
        natural_keys=[original.natural_key, valid.natural_key, invalid.natural_key],
    ) == {original.natural_key: replace(original, payload_path=original.payload_path.resolve())}
    follow_up = _publish(catalog, (_entry(tmp_path, natural_key="after-failure"),))
    assert follow_up.sequence == first_revision.sequence + 1


def test_paths_are_relative_and_survive_catalog_move(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    moved_root = tmp_path / "moved"
    entry = _entry(original_root)
    _publish(ReceiptCatalog(original_root / "catalog"), (entry,))
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
        _publish(catalog, (entry,))

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
    _publish(catalog, entries)

    assert catalog.successful_keys(source="financial_facts", fiscal_start="2016Q1") == frozenset({"2016Q1"})
    assert catalog.successful_keys(source="financial_facts") == frozenset({"2015Q4", "2016Q1", "no-period"})


def test_publish_rejects_invalid_identity_missing_payload_and_hash_mismatch(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    with pytest.raises(PITDataError, match="source and natural key"):
        _publish(catalog, (_entry(tmp_path, source=" "),))
    with pytest.raises(PITDataError, match="source and natural key"):
        _publish(catalog, (_entry(tmp_path, natural_key=" "),))
    with pytest.raises(PITDataError, match="missing"):
        _publish(catalog, (replace(_entry(tmp_path), payload_path=tmp_path / "absent.json"),))
    content_hash, payload_path = _payload_file(tmp_path, "tampered.json", b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        _publish(catalog, (replace(_entry(tmp_path), content_hash="f" * 64, payload_path=payload_path),))
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
        ("ALTER TABLE receipts ADD COLUMN extra TEXT", "receipts schema"),
        ("DROP INDEX idx_receipts_source_status", "receipts index is missing"),
        ("DROP TABLE blobs", "blobs schema"),
        ("DROP INDEX idx_blobs_source_usable", "blobs index is missing"),
        ("ALTER TABLE ranges ADD COLUMN extra TEXT", "ranges schema"),
        ("DROP INDEX idx_ranges_source_subject_start", "ranges index is missing"),
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
    _publish(catalog, (_entry(tmp_path),))
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute(mutation)

    with pytest.raises(PITDataError, match=message):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])


def test_corrupt_row_values_fail_closed(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    _publish(catalog, (_entry(tmp_path),))
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE receipts SET status = 'unknown'")

    with pytest.raises(PITDataError, match="corrupt entry"):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])


def test_corrupt_blob_and_range_rows_fail_closed(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    _publish(
        catalog,
        (_entry(tmp_path),),
        ranges=(_range("005930", date(2024, 1, 2), date(2024, 1, 2)),),
    )
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE blobs SET kind = 'not-a-kind'")
        connection.execute('UPDATE ranges SET "end" = \'not-a-date\'')

    with pytest.raises(PITDataError, match="corrupt blob"):
        list(catalog.blobs(source="krx_daily_market"))
    with pytest.raises(PITDataError, match="corrupt range"):
        list(catalog.ranges(source="ls_investor_flow"))


def test_payload_directory_is_not_a_valid_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload-directory"
    payload_path.mkdir()
    entry = replace(_entry(tmp_path), payload_path=payload_path)

    with pytest.raises(PITDataError, match="missing"):
        _publish(ReceiptCatalog(tmp_path / "catalog"), (entry,))


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
        _publish(ReceiptCatalog(tmp_path / "catalog"), (entry,))


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
        _publish(ReceiptCatalog(catalog_root), (entry,))


def test_duplicate_keys_in_one_batch_keep_the_latest_and_reject_equal_hash_conflicts(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    older = _entry(tmp_path, body=b"older", retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))
    newer = _entry(tmp_path, body=b"newer", retrieved_at=datetime(2024, 1, 4, tzinfo=UTC))
    same_time = datetime(2024, 1, 4, tzinfo=UTC)
    conflict = _entry(tmp_path, body=b"conflict", retrieved_at=same_time)

    revision = _publish(catalog, (older, newer))

    assert revision.row_count == 1
    assert catalog.latest(source=newer.source, natural_keys=[newer.natural_key])[newer.natural_key].content_hash == newer.content_hash
    with pytest.raises(PITDataError, match="conflict"):
        _publish(catalog, (newer, conflict))


def test_equal_timestamp_and_hash_is_idempotent(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)

    first = _publish(catalog, (entry,))
    second = _publish(catalog, (entry,))

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.row_count == 1


def test_unknown_schema_fails_closed_for_every_operation(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)
    _publish(catalog, (_entry(tmp_path),))
    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        connection.execute("UPDATE catalog_metadata SET schema_version = 999")

    with pytest.raises(PITDataError, match="schema_version"):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])
    with pytest.raises(PITDataError, match="schema_version"):
        catalog.successful_keys(source="krx_daily_market")
    with pytest.raises(PITDataError, match="schema_version"):
        catalog.entries(source="krx_daily_market")
    with pytest.raises(PITDataError, match="schema_version"):
        _publish(catalog, ())


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

    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

    root = Path(catalog_root)
    payloads = Path(payload_root)
    entries = []
    blobs = []
    for offset in range(count):
        key = f"worker-{worker_index}-key-{offset:03d}"
        payload_path = payloads / f"{key}.json"
        content_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        entries.append(
            ReceiptIndexEntry(
                source="concurrent",
                natural_key=key,
                as_of=None,
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=content_hash,
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                payload_path=payload_path,
            )
        )
        blobs.append(
            BlobEntry(
                content_hash=content_hash,
                kind=EvidenceKind.DAILY_MARKET,
                source="concurrent",
                usable=True,
                unusable_reason=None,
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
                payload_path=payload_path,
            )
        )
    barrier.wait(timeout=20)  # type: ignore[attr-defined]
    ReceiptCatalog(root).publish(entries, blobs=blobs)


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
    assert _publish(catalog, ()).sequence == 3


def test_publish_with_empty_batch_creates_database_and_increments_sequence(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog = ReceiptCatalog(catalog_root)

    first = _publish(catalog, ())
    second = _publish(catalog, ())

    assert first.sequence == 1
    assert first.row_count == 0
    assert first.path == catalog_root.resolve() / "catalog.sqlite3"
    assert second.sequence == 2
    assert second.row_count == 0
    assert catalog.successful_keys(source="source") == frozenset()


def test_latest_scales_with_requested_keys_not_catalog_size(tmp_path: Path) -> None:
    """A bad SQLite join order made this quadratic (minutes for 20k keys on the real catalog)."""
    import time

    catalog = ReceiptCatalog(tmp_path / "catalog")
    _publish(catalog, [_entry(tmp_path, natural_key="seed", as_of=None)])
    database = tmp_path / "catalog" / "catalog.sqlite3"
    rows = [
        ("investor_flow", f"{index:06d}:2026-01-02", "2026-01-02", None, "success", "a" * 64, "2026-01-03T00:00:00+00:00", "p")
        for index in range(60_000)
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany("INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    requested = [f"{index:06d}:2026-01-02" for index in range(0, 60_000, 6)] + ["absent:2026-01-02"]

    started = time.monotonic()
    found = catalog.latest(source="investor_flow", natural_keys=requested)
    elapsed = time.monotonic() - started

    assert len(found) == 10_000
    assert elapsed < 5.0


# --- schema v2: blobs and answered ranges ---------------------------------


def _legacy_catalog(catalog_root: Path, entries: Sequence[ReceiptIndexEntry]) -> None:
    """Write a ``v1`` catalog by hand: receipts, metadata and no blob/range tables."""
    connection = sqlite3.connect(catalog_root / "catalog.sqlite3")
    with connection:
        connection.execute(
            """
            CREATE TABLE catalog_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                row_count INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE receipts (
                source TEXT NOT NULL,
                natural_key TEXT NOT NULL,
                as_of TEXT,
                fiscal_period TEXT,
                status TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                payload_path TEXT NOT NULL,
                PRIMARY KEY (source, natural_key)
            ) WITHOUT ROWID
            """
        )
        connection.execute("CREATE INDEX idx_receipts_source_status ON receipts(source, status)")
        for entry in entries:
            connection.execute(
                "INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.source,
                    entry.natural_key,
                    entry.as_of.isoformat() if entry.as_of is not None else None,
                    entry.fiscal_period,
                    entry.status.value,
                    entry.content_hash,
                    entry.retrieved_at.isoformat(),
                    entry.payload_path.relative_to(catalog_root.parent).as_posix(),
                ),
            )
        connection.execute(
            "INSERT INTO catalog_metadata VALUES (1, 1, 7, ?)", (len(entries),)
        )
    connection.close()


def test_v1_database_is_migrated_in_place_by_the_next_publish(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir(parents=True)
    legacy = tuple(_entry(tmp_path, natural_key=f"2024-01-0{index}") for index in (2, 3, 4))
    _legacy_catalog(catalog_root, legacy)
    catalog = ReceiptCatalog(catalog_root)

    with pytest.raises(PITDataError, match="needs migration"):
        catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])
    with pytest.raises(PITDataError, match="needs migration"):
        list(catalog.blobs(source="krx_daily_market"))

    upgrade = _publish(catalog, ())

    with sqlite3.connect(catalog_root / "catalog.sqlite3") as connection:
        tables = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"blobs", "ranges"} <= tables
        version = connection.execute("SELECT schema_version FROM catalog_metadata").fetchone()[0]
    assert int(version) == 2
    assert upgrade.row_count == 3
    assert sorted(catalog.latest(source="krx_daily_market", natural_keys=list("123"))) == []


def test_migration_preserves_every_legacy_receipt(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir(parents=True)
    legacy = tuple(_entry(tmp_path, natural_key=f"2024-01-0{index}") for index in (2, 3, 4))
    _legacy_catalog(catalog_root, legacy)
    catalog = ReceiptCatalog(catalog_root)

    _publish(catalog, ())

    found = catalog.latest(source="krx_daily_market", natural_keys=[e.natural_key for e in legacy])
    assert {key: entry.content_hash for key, entry in found.items()} == {
        entry.natural_key: entry.content_hash for entry in legacy
    }


def test_dangling_receipt_reference_is_rejected_and_commits_nothing(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    orphan = _entry(tmp_path, natural_key="orphan")

    with pytest.raises(PITDataError, match="unstored blob"):
        catalog.publish((orphan,))

    assert list(catalog.blobs(source=orphan.source)) == []
    assert catalog.latest(source=orphan.source, natural_keys=[orphan.natural_key]) == {}
    assert list(catalog.ranges(source="ls_investor_flow")) == []
    assert catalog.blob_digest(source=orphan.source) == dataset_digest([])
    assert catalog.successful_keys(source=orphan.source) == frozenset()


def test_dangling_success_range_reference_is_rejected(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")

    with pytest.raises(PITDataError, match="unstored blob"):
        catalog.publish(
            (),
            ranges=[
                _range("005930", date(2024, 1, 2), date(2024, 1, 2), status=EvidenceStatus.SUCCESS, content_hash="a" * 64)
            ],
        )

    assert list(catalog.ranges(source="ls_investor_flow")) == []


def test_empty_range_needs_no_blob_and_success_range_does(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    answer = _entry(tmp_path, source="ls_investor_flow", natural_key="005930:2024-01-02")

    catalog.publish((), ranges=[_range("005930", date(2024, 1, 2), date(2024, 1, 2))])

    with pytest.raises(PITDataError, match="unstored blob"):
        catalog.publish(
            (),
            ranges=[
                _range(
                    "005930",
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    status=EvidenceStatus.SUCCESS,
                    content_hash=answer.content_hash,
                    retrieved_at=datetime(2024, 1, 4, tzinfo=UTC),
                )
            ],
        )

    accepted = catalog.publish(
        (),
        blobs=[_blob(answer)],
        ranges=[
            _range(
                "005930",
                date(2024, 1, 2),
                date(2024, 1, 2),
                status=EvidenceStatus.SUCCESS,
                content_hash=answer.content_hash,
                retrieved_at=datetime(2024, 1, 4, tzinfo=UTC),
            )
        ],
    )

    stored = list(catalog.ranges(source="ls_investor_flow"))
    assert accepted.sequence == 2
    assert [item.status for item in stored] == [EvidenceStatus.SUCCESS]
    assert stored[0].content_hash == answer.content_hash


def test_non_success_range_may_not_reference_a_blob(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    answer = _entry(tmp_path, source="ls_investor_flow", natural_key="005930:2024-01-02")

    with pytest.raises(PITDataError, match="must not reference a blob"):
        catalog.publish(
            (),
            blobs=[_blob(answer)],
            ranges=[
                _range(
                    "005930",
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    status=EvidenceStatus.EMPTY,
                    content_hash=answer.content_hash,
                )
            ],
        )


def test_unusable_blobs_are_hidden_from_reads_and_digest(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    keep = _entry(tmp_path, natural_key="keep", body=b"keep")
    drop = _entry(tmp_path, natural_key="drop", body=b"drop")

    catalog.publish(
        (),
        blobs=[_blob(keep), _blob(drop, usable=False)],
    )

    assert [item.content_hash for item in catalog.blobs(source="krx_daily_market")] == [keep.content_hash]
    assert [item.content_hash for item in catalog.blobs(source="krx_daily_market", usable=False)] == [
        drop.content_hash
    ]
    assert catalog.blob_digest(source="krx_daily_market") == dataset_digest([keep.content_hash])


def test_mark_unusable_changes_only_the_verdict(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)
    _publish(catalog, (entry,))

    revision = catalog.mark_unusable([entry.content_hash], reason="provider page is not evidence")

    assert revision.sequence == 2
    assert [item.content_hash for item in catalog.blobs(source="krx_daily_market")] == []
    assert entry.payload_path.is_file()
    with pytest.raises(PITDataError, match="no stored blob"):
        catalog.mark_unusable(["f" * 64], reason="absent")
    with pytest.raises(PITDataError, match="requires a reason"):
        catalog.mark_unusable([entry.content_hash], reason="  ")
    with pytest.raises(PITDataError, match="at least one content hash"):
        catalog.mark_unusable([], reason="anything")


def test_blob_digest_reads_no_payload_file(tmp_path: Path) -> None:
    """A Silver identity check must survive a Bronze tree it cannot open."""
    catalog = ReceiptCatalog(tmp_path / "catalog")
    first = _entry(tmp_path, natural_key="first", body=b"first")
    second = _entry(tmp_path, natural_key="second", body=b"second")
    _publish(catalog, (first, second))
    expected = dataset_digest(sorted([first.content_hash, second.content_hash]))
    for entry in (first, second):
        entry.payload_path.chmod(0o000)

    try:
        assert catalog.blob_digest(source="krx_daily_market") == expected
    finally:
        for entry in (first, second):
            entry.payload_path.chmod(0o644)


def test_delete_receipts_removes_catalog_rows_only(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)
    other = _entry(tmp_path, source="financial_facts", natural_key="00126380:2019:11013")
    _publish(catalog, (entry, other))

    revision = catalog.delete_receipts(source="krx_daily_market")

    assert revision.row_count == 1
    assert catalog.latest(source="krx_daily_market", natural_keys=[entry.natural_key]) == {}
    assert len(list(catalog.blobs(source="krx_daily_market"))) == 1
    assert entry.payload_path.is_file()
    with pytest.raises(PITDataError, match="requires a source"):
        catalog.delete_receipts(source="  ")


def test_latest_range_replaces_an_earlier_answer(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    first = _entry(tmp_path, source="ls_investor_flow", natural_key="page-1", body=b"one")
    second = _entry(tmp_path, source="ls_investor_flow", natural_key="page-2", body=b"two")
    catalog.publish(
        (),
        blobs=[_blob(first), _blob(second)],
        ranges=[
            _range(
                "005930",
                date(2024, 1, 2),
                date(2024, 1, 2),
                status=EvidenceStatus.SUCCESS,
                content_hash=first.content_hash,
                retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
            )
        ],
    )

    catalog.publish(
        (),
        blobs=[_blob(second)],
        ranges=[
            _range(
                "005930",
                date(2024, 1, 2),
                date(2024, 1, 2),
                status=EvidenceStatus.SUCCESS,
                content_hash=second.content_hash,
                retrieved_at=datetime(2024, 1, 4, tzinfo=UTC),
            )
        ],
    )

    stored = list(catalog.ranges(source="ls_investor_flow", subjects=["005930"]))
    assert len(stored) == 1
    assert stored[0].content_hash == second.content_hash
    assert stored[0].retrieved_at == datetime(2024, 1, 4, tzinfo=UTC)


def test_same_instant_range_conflict_is_rejected(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    first = _entry(tmp_path, source="ls_investor_flow", natural_key="page-1", body=b"one")
    second = _entry(tmp_path, source="ls_investor_flow", natural_key="page-2", body=b"two")
    catalog.publish(
        (),
        blobs=[_blob(first), _blob(second)],
        ranges=[
            _range(
                "005930",
                date(2024, 1, 2),
                date(2024, 1, 2),
                status=EvidenceStatus.SUCCESS,
                content_hash=first.content_hash,
            )
        ],
    )

    with pytest.raises(PITDataError, match="conflict"):
        catalog.publish(
            (),
            blobs=[_blob(second)],
            ranges=[
                _range(
                    "005930",
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    status=EvidenceStatus.SUCCESS,
                    content_hash=second.content_hash,
                )
            ],
        )

    assert next(iter(catalog.ranges(source="ls_investor_flow"))).content_hash == first.content_hash


def test_ranges_stream_filtered_by_subject_and_reject_reversed_spans(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    catalog.publish(
        (),
        ranges=[
            _range("005930", date(2024, 1, 2), date(2024, 1, 3)),
            _range("000660", date(2024, 1, 2), date(2024, 1, 2)),
        ],
    )

    assert [item.subject for item in catalog.ranges(source="ls_investor_flow", subjects=["005930"])] == ["005930"]
    assert [item.subject for item in catalog.ranges(source="ls_investor_flow")] == ["000660", "005930"]
    assert list(catalog.ranges(source="other_source")) == []
    with pytest.raises(PITDataError, match="ends before it starts"):
        catalog.publish((), ranges=[_range("005930", date(2024, 1, 3), date(2024, 1, 2))])


def test_blob_and_range_identity_is_checked_before_the_transaction(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)

    with pytest.raises(PITDataError, match="requires a content hash and a source"):
        catalog.publish((), blobs=[replace(_blob(entry), content_hash=" ")])
    with pytest.raises(PITDataError, match="requires a content hash and a source"):
        catalog.publish((), blobs=[replace(_blob(entry), source=" ")])
    with pytest.raises(PITDataError, match="requires a reason"):
        catalog.publish((), blobs=[replace(_blob(entry, usable=False), unusable_reason=" ")])
    with pytest.raises(PITDataError, match="hash mismatch for blob"):
        catalog.publish((), blobs=[replace(_blob(entry), content_hash="f" * 64)])
    with pytest.raises(PITDataError, match="payload is missing"):
        catalog.publish((), blobs=[replace(_blob(entry), payload_path=tmp_path / "absent.json")])
    with pytest.raises(PITDataError, match="requires a source and a subject"):
        catalog.publish(
            (), ranges=[replace(_range("005930", date(2024, 1, 2), date(2024, 1, 2), source=" "))]
        )
    with pytest.raises(PITDataError, match="requires a source and a subject"):
        catalog.publish(
            (),
            ranges=[
                replace(
                    _range("005930", date(2024, 1, 2), date(2024, 1, 2)),
                    subject=" ",
                )
            ],
        )

    assert list(catalog.blobs(source="krx_daily_market")) == []


def test_same_instant_conflict_inside_one_batch_is_rejected(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    first = _entry(tmp_path, source="ls_investor_flow", natural_key="page-1", body=b"one")
    second = _entry(tmp_path, source="ls_investor_flow", natural_key="page-2", body=b"two")
    same_instant = [
        _range(
            "005930",
            date(2024, 1, 2),
            date(2024, 1, 2),
            status=EvidenceStatus.SUCCESS,
            content_hash=first.content_hash,
        ),
        _range(
            "005930",
            date(2024, 1, 2),
            date(2024, 1, 2),
            status=EvidenceStatus.SUCCESS,
            content_hash=second.content_hash,
        ),
    ]

    with pytest.raises(PITDataError, match="range conflict"):
        catalog.publish((), blobs=[_blob(first), _blob(second)], ranges=same_instant)

    assert list(catalog.ranges(source="ls_investor_flow")) == []


def test_older_answers_never_replace_a_committed_one(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")
    first = _entry(tmp_path, source="ls_investor_flow", natural_key="page-1", body=b"one")
    newer = _entry(tmp_path, source="ls_investor_flow", natural_key="page-2", body=b"two")
    newer_range = _range(
        "005930",
        date(2024, 1, 2),
        date(2024, 1, 2),
        status=EvidenceStatus.SUCCESS,
        content_hash=newer.content_hash,
        retrieved_at=datetime(2024, 1, 4, tzinfo=UTC),
    )
    catalog.publish((), blobs=[_blob(first), _blob(newer)], ranges=[newer_range])

    catalog.publish((), ranges=[replace(newer_range, retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))])
    catalog.publish((), blobs=[_blob(first)])

    assert next(iter(catalog.ranges(source="ls_investor_flow"))).content_hash == newer.content_hash
    assert next(iter(catalog.blobs(source="ls_investor_flow"))).content_hash == newer.content_hash


def test_blob_digest_of_a_missing_catalog_is_the_empty_digest(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")

    assert catalog.blob_digest(source="krx_daily_market") == dataset_digest([])


def test_republishing_the_same_payload_updates_its_verdict(tmp_path: Path) -> None:
    """One payload, one blob row: a later verdict replaces an earlier one."""
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path, retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))
    _publish(catalog, (entry,))

    catalog.mark_unusable([entry.content_hash], reason="provider page is not evidence")
    later = replace(entry, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog.publish((), blobs=[replace(_blob(later), usable=True, unusable_reason=None)])

    stored = list(catalog.blobs(source="krx_daily_market", usable=False))
    assert [item.content_hash for item in stored] == []
    assert next(iter(catalog.blobs(source="krx_daily_market"))).retrieved_at == datetime(
        2024, 1, 5, tzinfo=UTC
    )
    assert catalog.blob_digest(source="krx_daily_market") == dataset_digest([entry.content_hash])


def test_deleting_receipts_on_a_missing_catalog_creates_an_empty_one(tmp_path: Path) -> None:
    catalog = ReceiptCatalog(tmp_path / "catalog")

    revision = catalog.delete_receipts(source="krx_daily_market")

    assert (revision.sequence, revision.row_count) == (1, 0)
    assert list(catalog.blobs(source="krx_daily_market")) == []
    assert catalog.successful_keys(source="krx_daily_market") == frozenset()


def test_deleting_receipts_migrates_a_legacy_catalog(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir(parents=True)
    legacy = tuple(_entry(tmp_path, natural_key=f"2024-01-0{index}") for index in (2, 3))
    _legacy_catalog(catalog_root, legacy)
    catalog = ReceiptCatalog(catalog_root)

    revision = catalog.delete_receipts(source="krx_daily_market")

    assert revision.row_count == 0
    assert catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02", "2024-01-03"]) == {}
    assert catalog.blob_digest(source="krx_daily_market") == dataset_digest([])



def test_streaming_rows_performs_no_path_resolution(tmp_path: Path, monkeypatch) -> None:
    """Many rows stream with stored paths and no per-row filesystem resolution."""
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entries = tuple(_entry(tmp_path, natural_key=f"2024-01-{day:02d}") for day in range(2, 6))
    _publish(catalog, entries)

    resolve_calls = 0
    real_resolve = Path.resolve

    def _counted_resolve(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal resolve_calls
        resolve_calls += 1
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _counted_resolve)
    streamed_entries = list(catalog.entries(source="krx_daily_market"))
    streamed_blobs = list(catalog.blobs(source="krx_daily_market"))
    assert resolve_calls == 0
    bronze_root = (tmp_path / "catalog").resolve().parent
    for item in (*streamed_entries, *streamed_blobs):
        assert item.payload_path.is_absolute()
        assert str(item.payload_path).startswith(str(bronze_root))


def test_stored_path_outside_bronze_root_is_rejected_on_read(tmp_path: Path) -> None:
    """A stored path escaping the root through .. fails with a containment error."""
    import sqlite3

    catalog = ReceiptCatalog(tmp_path / "catalog")
    _publish(catalog, (_entry(tmp_path),))
    bronze_root = (tmp_path / "catalog").resolve().parent
    escaped = f"{bronze_root}/../outside/payload.json"
    with sqlite3.connect(tmp_path / "catalog" / "catalog.sqlite3") as connection:
        connection.execute("UPDATE receipts SET payload_path = ?", (escaped,))
        connection.commit()
    with pytest.raises(PITDataError, match="outside the Bronze root"):
        list(catalog.entries(source="krx_daily_market"))


def test_relative_stored_paths_join_bronze_root_and_publish_verifies_content(tmp_path: Path) -> None:
    """Relative stored paths resolve under the root; bad publish bytes still fail."""
    catalog = ReceiptCatalog(tmp_path / "catalog")
    entry = _entry(tmp_path)
    _publish(catalog, (entry,))
    bronze_root = (tmp_path / "catalog").resolve().parent
    (streamed,) = list(catalog.entries(source="krx_daily_market"))
    assert streamed.payload_path == bronze_root / streamed.payload_path.relative_to(bronze_root)
    tampered = replace(entry, natural_key="2024-01-09", content_hash="0" * 64)
    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog.publish((tampered,), blobs=[_blob(tampered)])
    outside = replace(entry, natural_key="2024-01-10",
                      payload_path=tmp_path.parent / "outside-payload.json")
    with pytest.raises(PITDataError, match=r"outside the Bronze root|missing"):
        catalog.publish((outside,), blobs=[replace(_blob(outside), payload_path=outside.payload_path)])


def test_stored_blob_path_outside_bronze_root_is_rejected_on_read(tmp_path: Path) -> None:
    """A blob row escaping the root through .. fails with a containment error."""
    import sqlite3

    catalog = ReceiptCatalog(tmp_path / "catalog")
    _publish(catalog, (_entry(tmp_path),))
    bronze_root = (tmp_path / "catalog").resolve().parent
    escaped = f"{bronze_root}/../outside/payload.zip"
    with sqlite3.connect(tmp_path / "catalog" / "catalog.sqlite3") as connection:
        connection.execute("UPDATE blobs SET payload_path = ?", (escaped,))
        connection.commit()
    with pytest.raises(PITDataError, match=r"outside the Bronze root"):
        list(catalog.blobs(source="krx_daily_market"))
