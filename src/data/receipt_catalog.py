"""Scope-local receipt index with latest-state lookup by source natural key."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from src.core.digest import dataset_digest
from src.core.pit import EvidenceKind, PITDataError

__all__ = [
    "BlobEntry",
    "CatalogRevision",
    "CoverageRange",
    "EvidenceStatus",
    "ReceiptCatalog",
    "ReceiptIndexEntry",
]

_CATALOG_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_SQLITE_TIMEOUT_SECONDS = 30.0
_BUSY_TIMEOUT_MS = 30_000
_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")
_RECEIPT_COLUMNS = (
    "source",
    "natural_key",
    "as_of",
    "fiscal_period",
    "status",
    "content_hash",
    "retrieved_at",
    "payload_path",
)
_BLOB_COLUMNS = (
    "content_hash",
    "kind",
    "source",
    "usable",
    "unusable_reason",
    "retrieved_at",
    "payload_path",
)
_RANGE_COLUMNS = (
    "source",
    "subject",
    "start",
    "end",
    "status",
    "content_hash",
    "retrieved_at",
)
_METADATA_COLUMNS = ("singleton", "schema_version", "sequence", "row_count")


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


class EvidenceStatus(StrEnum):
    """Observed provider outcome retained separately from successful evidence coverage."""

    SUCCESS = "success"
    EMPTY = "empty"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass(frozen=True, slots=True)
class ReceiptIndexEntry:
    """Natural-key index row for one retained raw provider response.

    The index makes collection planning depend on validated receipt state rather
    than repeated recursive reads of content-addressed JSON payloads.
    """

    source: str
    natural_key: str
    as_of: date | None
    fiscal_period: str | None
    status: EvidenceStatus
    content_hash: str
    retrieved_at: datetime
    payload_path: Path


@dataclass(frozen=True, slots=True)
class CatalogRevision:
    """Summary of one committed publish."""

    sequence: int
    row_count: int
    path: Path


@dataclass(frozen=True, slots=True)
class BlobEntry:
    """One stored Bronze payload and its usability verdict."""

    content_hash: str
    kind: EvidenceKind
    source: str
    usable: bool
    unusable_reason: str | None
    retrieved_at: datetime
    payload_path: Path


@dataclass(frozen=True, slots=True)
class CoverageRange:
    """An answered provider request over consecutive sessions of one subject.

    ``success`` ranges reference the blob holding the rows. ``empty`` ranges
    record that the provider answered with no rows and have no blob. A session
    inside an answered range is never requested again from that source.
    Whether it has a value is decided by Silver from the blob's rows.
    """

    source: str
    subject: str
    start: date
    end: date
    status: EvidenceStatus
    content_hash: str | None
    retrieved_at: datetime


class ReceiptCatalog:
    """Scope-local evidence index: blobs, keyed receipts and answered ranges.

    Backed by ``<root>/catalog.sqlite3``. A publish is one transaction, so
    readers see either all or none of a batch, and concurrent collectors are
    serialized by SQLite's write lock instead of rewriting a snapshot.
    Payload paths are stored relative to the Bronze root (the catalog root's
    parent), so a moved data root keeps a valid catalog.

    The catalog is the only answer to "what evidence exists": every stored
    payload is a ``blobs`` row carrying its source and usability verdict, and a
    range-shaped source records answered ``[start, end]`` session ranges instead
    of one receipt per cell. Deleting rows never deletes Bronze files.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._bronze_root = self._root.parent
        self._database_path = self._root / "catalog.sqlite3"

    def _database_is_missing(self) -> bool:
        if not self._database_path.exists():
            return True
        if not self._database_path.is_file():
            raise PITDataError(f"receipt catalog database is not a file: {self._database_path}")
        return False

    @staticmethod
    def _enable_wal(connection: sqlite3.Connection) -> None:
        deadline = time.monotonic() + _SQLITE_TIMEOUT_SECONDS
        while True:
            try:
                journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
                if journal_mode is not None and str(journal_mode[0]).lower() == "wal":
                    return
                raise PITDataError("receipt catalog could not enable WAL mode")  # pragma: no cover - filesystem guard
            except sqlite3.OperationalError as exc:  # pragma: no cover - child-process concurrency is not traced
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _connect(self, *, write: bool) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
                isolation_level=None,
            )
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            if write:
                self._enable_wal(connection)
                connection.execute("PRAGMA synchronous = NORMAL")
            else:
                connection.execute("PRAGMA query_only = ON")
        except PITDataError:  # pragma: no cover - connection cleanup guard
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:  # pragma: no cover - connection cleanup guard
            if connection is not None:
                connection.close()
            raise PITDataError("receipt catalog database is unreadable") from exc
        return connection

    def _connect_existing(self) -> sqlite3.Connection | None:
        if self._database_is_missing():
            return None
        connection = self._connect(write=False)
        try:
            uninitialized = self._is_uninitialized(connection)
            if not uninitialized:
                self._validate_schema(connection, allow_legacy=False)
        except sqlite3.Error as exc:
            connection.close()
            raise PITDataError("receipt catalog database is corrupt") from exc
        except Exception:
            connection.close()
            raise
        if uninitialized:
            connection.close()
            return None
        return connection

    @staticmethod
    def _is_uninitialized(connection: sqlite3.Connection) -> bool:
        """True for a database file that holds no table at all.

        SQLite creates the file and its WAL sidecars before the first statement
        runs, so a rolled-back first publish leaves an empty file behind. That
        is an unwritten catalog, not a corrupt one, and reads must see it as
        empty instead of failing closed on a schema that does not exist yet.
        """
        row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
        return row is not None and int(row[0]) == 0

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return tuple(str(row[1]) for row in rows)

    @staticmethod
    def _require_columns(connection: sqlite3.Connection, table: str, expected: tuple[str, ...]) -> None:
        if ReceiptCatalog._table_columns(connection, table) != expected:
            raise PITDataError(f"receipt catalog {table} schema is invalid")

    @staticmethod
    def _require_index(connection: sqlite3.Connection, table: str, index: str) -> None:
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?", (table,)
            ).fetchall()
        }
        if index not in index_names:
            raise PITDataError(f"receipt catalog {table} index is missing")

    def _validate_schema(self, connection: sqlite3.Connection, *, allow_legacy: bool) -> tuple[int, int]:
        """Validate metadata plus every table the current schema version owns.

        Returns:
            The committed sequence and row count.

        Raises:
            PITDataError: the database is not a catalog, the version is
                unknown, a ``v1`` database is read before it is migrated, or a
                table is missing its columns or indexes.
        """
        try:
            self._require_columns(connection, "catalog_metadata", _METADATA_COLUMNS)
            metadata_rows = connection.execute(
                "SELECT schema_version, sequence, row_count "
                "FROM catalog_metadata "
                "WHERE singleton = 1"
            ).fetchall()
            if len(metadata_rows) != 1:
                raise PITDataError("receipt catalog metadata is missing")
            schema_version, sequence, row_count = metadata_rows[0]
            version = int(schema_version)
            if version not in (_LEGACY_SCHEMA_VERSION, _CATALOG_SCHEMA_VERSION):
                raise PITDataError(f"unsupported receipt catalog schema_version: {schema_version!r}")
            if version == _LEGACY_SCHEMA_VERSION and not allow_legacy:
                raise PITDataError("receipt catalog needs migration")
            self._require_columns(connection, "receipts", _RECEIPT_COLUMNS)
            self._require_index(connection, "receipts", "idx_receipts_source_status")
            if version >= _CATALOG_SCHEMA_VERSION:
                self._require_columns(connection, "blobs", _BLOB_COLUMNS)
                self._require_index(connection, "blobs", "idx_blobs_source_usable")
                self._require_columns(connection, "ranges", _RANGE_COLUMNS)
                self._require_index(connection, "ranges", "idx_ranges_source_subject_start")
        except sqlite3.Error as exc:
            raise PITDataError("receipt catalog database is corrupt") from exc
        sequence_int = int(sequence)
        row_count_int = int(row_count)
        if sequence_int < 0 or row_count_int < 0:
            raise PITDataError("receipt catalog metadata is invalid")
        return sequence_int, row_count_int

    @staticmethod
    def _create_blobs_and_ranges(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                content_hash TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                usable INTEGER NOT NULL,
                unusable_reason TEXT,
                retrieved_at TEXT NOT NULL,
                payload_path TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_blobs_source_usable ON blobs(source, usable)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ranges (
                source TEXT NOT NULL,
                subject TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                status TEXT NOT NULL,
                content_hash TEXT,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (source, subject, start, end)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ranges_source_subject_start "
            "ON ranges(source, subject, start)"
        )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                row_count INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
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
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_receipts_source_status ON receipts(source, status)"
        )
        self._create_blobs_and_ranges(connection)
        connection.execute(
            "INSERT OR IGNORE INTO catalog_metadata(singleton, schema_version, sequence, row_count) "
            "VALUES (1, ?, 0, 0)",
            (_CATALOG_SCHEMA_VERSION,),
        )

    @staticmethod
    def _migrate_legacy_schema(connection: sqlite3.Connection) -> None:
        """Bring a ``v1`` catalog to the current version in the caller's transaction."""
        ReceiptCatalog._create_blobs_and_ranges(connection)
        connection.execute(
            "UPDATE catalog_metadata SET schema_version = ? WHERE singleton = 1",
            (_CATALOG_SCHEMA_VERSION,),
        )

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            with suppress(sqlite3.Error):
                connection.execute("ROLLBACK")

    def _payload_path(self, raw_path: Path, *, must_exist: bool) -> Path:
        candidate = raw_path if raw_path.is_absolute() else self._bronze_root / raw_path
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise PITDataError(f"receipt catalog payload is missing for {raw_path!s}") from exc
        try:
            resolved.relative_to(self._bronze_root)
        except ValueError as exc:
            raise PITDataError(f"receipt catalog payload is outside the Bronze root: {resolved}") from exc
        if must_exist and not resolved.is_file():
            raise PITDataError(f"receipt catalog payload is missing for {raw_path!s}")
        return resolved

    def _prepare_entries(self, entries: Sequence[ReceiptIndexEntry]) -> tuple[ReceiptIndexEntry, ...]:
        prepared: dict[tuple[str, str], ReceiptIndexEntry] = {}
        for entry in entries:
            if not entry.source.strip() or not entry.natural_key.strip():
                raise PITDataError("receipt catalog entry requires source and natural key")
            payload_path = self._payload_path(Path(entry.payload_path), must_exist=True)
            try:
                with payload_path.open("rb") as handle:
                    actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            except OSError as exc:  # pragma: no cover - external file-race guard
                raise PITDataError(f"receipt catalog payload is missing for {entry.natural_key!r}") from exc
            if actual_hash != entry.content_hash:
                raise PITDataError(f"receipt catalog hash mismatch for {entry.natural_key!r}")
            candidate = replace(entry, payload_path=payload_path)
            key = (candidate.source, candidate.natural_key)
            current = prepared.get(key)
            if current is None or candidate.retrieved_at > current.retrieved_at:
                prepared[key] = candidate
            elif (
                candidate.retrieved_at == current.retrieved_at
                and candidate.content_hash != current.content_hash
            ):
                raise PITDataError(f"receipt catalog conflict for {candidate.natural_key!r}")
        return tuple(prepared.values())

    def _entry_from_row(self, row: Sequence[object]) -> ReceiptIndexEntry:
        try:
            raw_as_of = row[2]
            raw_retrieved_at = row[6]
            return ReceiptIndexEntry(
                source=str(row[0]),
                natural_key=str(row[1]),
                as_of=date.fromisoformat(str(raw_as_of)) if raw_as_of is not None else None,
                fiscal_period=str(row[3]) if row[3] is not None else None,
                status=EvidenceStatus(str(row[4])),
                content_hash=str(row[5]),
                retrieved_at=datetime.fromisoformat(str(raw_retrieved_at)),
                payload_path=self._payload_path(Path(str(row[7])), must_exist=False),
            )
        except (TypeError, ValueError) as exc:
            raise PITDataError("receipt catalog contains a corrupt entry") from exc

    def _blob_from_row(self, row: Sequence[object]) -> BlobEntry:
        try:
            return BlobEntry(
                content_hash=str(row[0]),
                kind=EvidenceKind(str(row[1])),
                source=str(row[2]),
                usable=bool(row[3]),
                unusable_reason=str(row[4]) if row[4] is not None else None,
                retrieved_at=datetime.fromisoformat(str(row[5])),
                payload_path=self._payload_path(Path(str(row[6])), must_exist=False),
            )
        except (TypeError, ValueError) as exc:
            raise PITDataError("receipt catalog contains a corrupt blob") from exc

    @staticmethod
    def _range_from_row(row: Sequence[object]) -> CoverageRange:
        try:
            return CoverageRange(
                source=str(row[0]),
                subject=str(row[1]),
                start=date.fromisoformat(str(row[2])),
                end=date.fromisoformat(str(row[3])),
                status=EvidenceStatus(str(row[4])),
                content_hash=str(row[5]) if row[5] is not None else None,
                retrieved_at=datetime.fromisoformat(str(row[6])),
            )
        except (TypeError, ValueError) as exc:
            raise PITDataError("receipt catalog contains a corrupt range") from exc

    def _prepare_blobs(self, blobs: Sequence[BlobEntry]) -> tuple[BlobEntry, ...]:
        prepared: dict[str, BlobEntry] = {}
        for blob in blobs:
            if not blob.content_hash.strip() or not blob.source.strip():
                raise PITDataError("receipt catalog blob requires a content hash and a source")
            if not blob.usable and not (blob.unusable_reason or "").strip():
                raise PITDataError(f"unusable blob {blob.content_hash[:12]} requires a reason")
            payload_path = self._payload_path(Path(blob.payload_path), must_exist=True)
            with payload_path.open("rb") as handle:
                actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual_hash != blob.content_hash:
                raise PITDataError(f"receipt catalog hash mismatch for blob {blob.content_hash[:12]}")
            candidate = replace(blob, payload_path=payload_path)
            current = prepared.get(candidate.content_hash)
            if current is None or candidate.retrieved_at > current.retrieved_at:
                prepared[candidate.content_hash] = candidate
        return tuple(prepared.values())

    @staticmethod
    def _prepare_ranges(ranges: Sequence[CoverageRange]) -> tuple[CoverageRange, ...]:
        prepared: dict[tuple[str, str, date, date], CoverageRange] = {}
        for item in ranges:
            if not item.source.strip() or not item.subject.strip():
                raise PITDataError("receipt catalog range requires a source and a subject")
            if item.end < item.start:
                raise PITDataError(f"receipt catalog range ends before it starts for {item.subject!r}")
            if item.status is not EvidenceStatus.SUCCESS and item.content_hash is not None:
                raise PITDataError(f"{item.status.value} range for {item.subject!r} must not reference a blob")
            key = (item.source, item.subject, item.start, item.end)
            current = prepared.get(key)
            if current is None or item.retrieved_at > current.retrieved_at:
                prepared[key] = item
            elif item.retrieved_at == current.retrieved_at and item.content_hash != current.content_hash:
                raise PITDataError(f"receipt catalog range conflict for {item.subject!r} {item.start!s}..{item.end!s}")
        return tuple(prepared.values())

    @staticmethod
    def _require_blob_reference(connection: sqlite3.Connection, content_hash: str, label: str) -> None:
        stored = connection.execute(
            "SELECT 1 FROM blobs WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if stored is None:
            raise PITDataError(f"receipt catalog reference to an unstored blob for {label}")

    def _write_blob(self, connection: sqlite3.Connection, blob: BlobEntry) -> None:
        stored_path = blob.payload_path.relative_to(self._bronze_root).as_posix()
        current = connection.execute(
            "SELECT retrieved_at, usable, unusable_reason FROM blobs WHERE content_hash = ?",
            (blob.content_hash,),
        ).fetchone()
        if current is None:
            connection.execute(
                """
                INSERT INTO blobs(
                    content_hash, kind, source, usable, unusable_reason, retrieved_at, payload_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    blob.content_hash,
                    blob.kind.value,
                    blob.source,
                    1 if blob.usable else 0,
                    blob.unusable_reason,
                    blob.retrieved_at.isoformat(),
                    stored_path,
                ),
            )
            return
        stored_retrieved_at = datetime.fromisoformat(str(current[0]))
        upgrade_unreferenced = (
            blob.usable
            and int(current[1]) == 0
            and str(current[2] or "") == "unreferenced"
            and stored_retrieved_at <= blob.retrieved_at
        )
        if stored_retrieved_at >= blob.retrieved_at and not upgrade_unreferenced:
            return
        if True:
            connection.execute(
                """
                UPDATE blobs
                SET kind = ?, source = ?, usable = ?, unusable_reason = ?, retrieved_at = ?, payload_path = ?
                WHERE content_hash = ?
                """,
                (
                    blob.kind.value,
                    blob.source,
                    1 if blob.usable else 0,
                    blob.unusable_reason,
                    blob.retrieved_at.isoformat(),
                    stored_path,
                    blob.content_hash,
                ),
            )

    def _write_range(self, connection: sqlite3.Connection, item: CoverageRange) -> None:
        key = (item.source, item.subject, item.start.isoformat(), item.end.isoformat())
        current = connection.execute(
            'SELECT retrieved_at, content_hash FROM ranges '
            'WHERE source = ? AND subject = ? AND start = ? AND "end" = ?',
            key,
        ).fetchone()
        if current is None:
            connection.execute(
                """
                INSERT INTO ranges(source, subject, start, "end", status, content_hash, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (*key[:2], key[2], key[3], item.status.value, item.content_hash, item.retrieved_at.isoformat()),
            )
            return
        current_retrieved_at = datetime.fromisoformat(str(current[0]))
        if item.retrieved_at == current_retrieved_at and item.content_hash != str(current[1] or ""):
            raise PITDataError(
                f"receipt catalog range conflict for {item.subject!r} {item.start.isoformat()}..{item.end.isoformat()}"
            )
        if item.retrieved_at <= current_retrieved_at:
            return
        connection.execute(
            """
            UPDATE ranges
            SET status = ?, content_hash = ?, retrieved_at = ?
            WHERE source = ? AND subject = ? AND start = ? AND "end" = ?
            """,
            (item.status.value, item.content_hash, item.retrieved_at.isoformat(), *key),
        )

    def publish(
        self,
        entries: Sequence[ReceiptIndexEntry],
        *,
        blobs: Sequence[BlobEntry] = (),
        ranges: Sequence[CoverageRange] = (),
    ) -> CatalogRevision:
        """Validate a complete batch, then atomically publish it to SQLite.

        Blobs, keyed receipts and answered ranges commit in one transaction, so
        a reader never sees a receipt whose payload is not yet registered. Every
        receipt and every ``success`` range must reference a blob that exists in
        ``blobs`` or in this batch; nothing is committed when one does not.

        Returns:
            The committed revision, whose ``row_count`` counts keyed receipts.

        Raises:
            PITDataError: a payload is missing, a hash does not match, a
                reference dangles, or a same-instant row conflicts with a
                different hash.
        """
        prepared_entries = self._prepare_entries(entries)
        prepared_blobs = self._prepare_blobs(blobs)
        prepared_ranges = self._prepare_ranges(ranges)
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - external filesystem guard
            raise PITDataError("receipt catalog directory could not be created") from exc
        database_existed = not self._database_is_missing()
        connection = self._connect(write=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not database_existed or self._is_uninitialized(connection):
                self._create_schema(connection)
            else:
                self._validate_schema(connection, allow_legacy=True)
                self._migrate_legacy_schema(connection)
            _, row_count = self._validate_schema(connection, allow_legacy=False)
            for blob in prepared_blobs:
                self._write_blob(connection, blob)
            for entry in prepared_entries:
                self._require_blob_reference(connection, entry.content_hash, entry.natural_key)
            for item in prepared_ranges:
                if item.content_hash is not None:
                    self._require_blob_reference(
                        connection, item.content_hash, f"{item.subject} {item.start!s}..{item.end!s}"
                    )
            for entry in prepared_entries:
                current = connection.execute(
                    "SELECT retrieved_at, content_hash FROM receipts "
                    "WHERE source = ? AND natural_key = ?",
                    (entry.source, entry.natural_key),
                ).fetchone()
                if current is None:
                    connection.execute(
                        """
                        INSERT INTO receipts(
                            source, natural_key, as_of, fiscal_period, status,
                            content_hash, retrieved_at, payload_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry.source,
                            entry.natural_key,
                            entry.as_of.isoformat() if entry.as_of is not None else None,
                            entry.fiscal_period,
                            entry.status.value,
                            entry.content_hash,
                            entry.retrieved_at.isoformat(),
                            entry.payload_path.relative_to(self._bronze_root).as_posix(),
                        ),
                    )
                    row_count += 1
                    continue
                current_retrieved_at = datetime.fromisoformat(str(current[0]))
                if entry.retrieved_at == current_retrieved_at and entry.content_hash != str(current[1]):
                    raise PITDataError(f"receipt catalog conflict for {entry.natural_key!r}")
                if entry.retrieved_at <= current_retrieved_at:
                    continue
                connection.execute(
                    """
                    UPDATE receipts
                    SET as_of = ?, fiscal_period = ?, status = ?, content_hash = ?,
                        retrieved_at = ?, payload_path = ?
                    WHERE source = ? AND natural_key = ?
                    """,
                    (
                        entry.as_of.isoformat() if entry.as_of is not None else None,
                        entry.fiscal_period,
                        entry.status.value,
                        entry.content_hash,
                        entry.retrieved_at.isoformat(),
                        entry.payload_path.relative_to(self._bronze_root).as_posix(),
                        entry.source,
                        entry.natural_key,
                    ),
                )
            for item in prepared_ranges:
                self._write_range(connection, item)
            connection.execute(
                "UPDATE catalog_metadata SET sequence = sequence + 1, row_count = ? WHERE singleton = 1",
                (row_count,),
            )
            revision_row = connection.execute(
                "SELECT sequence, row_count FROM catalog_metadata WHERE singleton = 1"
            ).fetchone()
            sequence, committed_row_count = cast(tuple[int, int], revision_row)
            connection.execute("COMMIT")
        except PITDataError:
            self._rollback(connection)
            raise
        except sqlite3.Error as exc:  # pragma: no cover - transactional infrastructure guard
            self._rollback(connection)
            raise PITDataError("receipt catalog publish failed") from exc
        finally:
            connection.close()
        return CatalogRevision(sequence=sequence, row_count=committed_row_count, path=self._database_path)

    def latest(self, *, source: str, natural_keys: Collection[str]) -> Mapping[str, ReceiptIndexEntry]:
        """Return entries for the requested keys without scanning the full catalog.

        ``CROSS JOIN`` pins the join order: the requested keys drive primary-key
        lookups. Left to itself SQLite may scan every receipt and search the key
        list linearly per row, which is quadratic (about 5 minutes for 20k keys
        against 400k receipts versus milliseconds here).
        """
        if self._database_is_missing():
            return {}
        requested_json = json.dumps(list(dict.fromkeys(natural_keys)), separators=(",", ":"))
        connection = self._connect_existing()
        if connection is None:  # pragma: no cover - deletion race guard
            return {}
        try:
            rows = connection.execute(
                """
                SELECT r.source, r.natural_key, r.as_of, r.fiscal_period,
                       r.status, r.content_hash, r.retrieved_at, r.payload_path
                FROM json_each(?) AS requested
                CROSS JOIN receipts AS r
                    ON r.source = ? AND r.natural_key = requested.value
                """,
                (requested_json, source),
            ).fetchall()
            return {str(row[1]): self._entry_from_row(row) for row in rows}
        except sqlite3.Error as exc:  # pragma: no cover - query infrastructure guard
            raise PITDataError("receipt catalog lookup failed") from exc
        finally:
            connection.close()

    def successful_keys(self, *, source: str, fiscal_start: str | None = None) -> frozenset[str]:
        """Return successful natural keys, optionally applying a fiscal-period floor."""
        floor = _fiscal_key(fiscal_start) if fiscal_start is not None else None
        if self._database_is_missing():
            return frozenset()
        connection = self._connect_existing()
        if connection is None:  # pragma: no cover - deletion race guard
            return frozenset()
        try:
            rows = connection.execute(
                "SELECT natural_key, fiscal_period FROM receipts WHERE source = ? AND status = ?",
                (source, EvidenceStatus.SUCCESS.value),
            )
            keys: set[str] = set()
            for natural_key, fiscal_period in rows:
                if floor is not None:
                    raw_period = str(fiscal_period) if fiscal_period is not None else ""
                    if not _FISCAL_PATTERN.fullmatch(raw_period) or _fiscal_key(raw_period) < floor:
                        continue
                keys.add(str(natural_key))
            return frozenset(keys)
        except sqlite3.Error as exc:  # pragma: no cover - query infrastructure guard
            raise PITDataError("receipt catalog successful-key lookup failed") from exc
        finally:
            connection.close()

    def entries(self, *, source: str) -> Iterator[ReceiptIndexEntry]:
        """Stream all entries for one source from a single indexed query."""
        if self._database_is_missing():
            return iter(())
        validation_connection = self._connect_existing()
        if validation_connection is None:  # pragma: no cover - deletion race guard
            return iter(())
        validation_connection.close()
        return self._stream_entries(source=source)

    def _stream_entries(self, *, source: str) -> Iterator[ReceiptIndexEntry]:
        connection = self._connect_existing()
        if connection is None:  # pragma: no cover - deletion race guard
            return
        try:
            rows = connection.execute(
                """
                SELECT source, natural_key, as_of, fiscal_period,
                       status, content_hash, retrieved_at, payload_path
                FROM receipts
                WHERE source = ?
                """,
                (source,),
            )
            for row in rows:
                yield self._entry_from_row(row)
        except sqlite3.Error as exc:  # pragma: no cover - stream infrastructure guard
            raise PITDataError("receipt catalog entry stream failed") from exc
        finally:
            connection.close()

    def blobs(self, *, source: str, usable: bool = True) -> Iterator[BlobEntry]:
        """Stream stored payloads of one source from a single indexed query.

        The verdict, not the file, decides what a reader sees: a blob marked
        unusable is never yielded here and never enters ``blob_digest``.
        """
        if self._database_is_missing():
            return iter(())
        validation_connection = self._connect_existing()
        if validation_connection is None:  # pragma: no cover - deletion race guard
            return iter(())
        validation_connection.close()
        return self._stream_blobs(source=source, usable=usable)

    def _stream_blobs(self, *, source: str, usable: bool) -> Iterator[BlobEntry]:
        connection = self._connect_existing()
        if connection is None:  # pragma: no cover - deletion race guard
            return
        try:
            rows = connection.execute(
                """
                SELECT content_hash, kind, source, usable, unusable_reason, retrieved_at, payload_path
                FROM blobs
                WHERE source = ? AND usable = ?
                ORDER BY content_hash
                """,
                (source, 1 if usable else 0),
            )
            for row in rows:
                yield self._blob_from_row(row)
        except sqlite3.Error as exc:  # pragma: no cover - stream infrastructure guard
            raise PITDataError("receipt catalog blob stream failed") from exc
        finally:
            connection.close()

    def blob_digest(self, *, source: str) -> str:
        """Return the Silver ``bronze_*`` identity of one source's usable blobs.

        Computed in SQL order from catalog columns alone, so a preview or a
        freshness check never opens a payload file.
        """
        if self._database_is_missing():
            return dataset_digest([])
        connection = self._connect_existing()
        if connection is None:  # pragma: no cover - deletion race guard
            return dataset_digest([])
        try:
            rows = connection.execute(
                "SELECT content_hash FROM blobs WHERE source = ? AND usable = 1 ORDER BY content_hash",
                (source,),
            )
            return dataset_digest([str(row[0]) for row in rows])
        except sqlite3.Error as exc:  # pragma: no cover - query infrastructure guard
            raise PITDataError("receipt catalog blob digest failed") from exc
        finally:
            connection.close()

    def ranges(
        self, *, source: str, subjects: Collection[str] | None = None
    ) -> Iterator[CoverageRange]:
        """Stream answered session ranges of one source from a single indexed query.

        Args:
            source: Registered source whose answered ranges are read.
            subjects: Restrict the stream to these subjects; every subject of
                the source is streamed when omitted.
        """
        if self._database_is_missing():
            return iter(())
        validation_connection = self._connect_existing()
        if validation_connection is None:  # pragma: no cover - deletion race guard
            return iter(())
        validation_connection.close()
        return self._stream_ranges(source=source, subjects=subjects)

    def _stream_ranges(
        self, *, source: str, subjects: Collection[str] | None
    ) -> Iterator[CoverageRange]:
        connection = self._connect_existing()
        if connection is None:  # pragma: no cover - deletion race guard
            return
        try:
            if subjects is None:
                rows = connection.execute(
                    """
                    SELECT source, subject, start, "end", status, content_hash, retrieved_at
                    FROM ranges
                    WHERE source = ?
                    ORDER BY subject, start
                    """,
                    (source,),
                )
            else:
                requested_json = json.dumps(list(dict.fromkeys(subjects)), separators=(",", ":"))
                rows = connection.execute(
                    """
                    SELECT r.source, r.subject, r.start, r."end", r.status, r.content_hash, r.retrieved_at
                    FROM json_each(?) AS requested
                    CROSS JOIN ranges AS r
                        ON r.source = ? AND r.subject = requested.value
                    ORDER BY r.subject, r.start
                    """,
                    (requested_json, source),
                )
            for row in rows:
                yield self._range_from_row(row)
        except sqlite3.Error as exc:  # pragma: no cover - stream infrastructure guard
            raise PITDataError("receipt catalog range stream failed") from exc
        finally:
            connection.close()

    def mark_unusable(self, content_hashes: Collection[str], *, reason: str) -> CatalogRevision:
        """Change the usability verdict of stored blobs without touching disk.

        Raises:
            PITDataError: the reason is empty, no hash was requested, or none of
                the requested hashes is a stored blob.
        """
        requested = list(dict.fromkeys(str(value) for value in content_hashes))
        if not (reason or "").strip():
            raise PITDataError("marking blobs unusable requires a reason")
        if not requested:
            raise PITDataError("marking blobs unusable requires at least one content hash")
        requested_json = json.dumps(requested, separators=(",", ":"))

        def _mark(connection: sqlite3.Connection) -> None:
            stored = connection.execute(
                "SELECT COUNT(*) FROM blobs "
                "WHERE content_hash IN (SELECT value FROM json_each(?))",
                (requested_json,),
            ).fetchone()
            if stored is None or int(stored[0]) == 0:
                raise PITDataError("marking blobs unusable found no stored blob")
            connection.execute(
                "UPDATE blobs SET usable = 0, unusable_reason = ? "
                "WHERE content_hash IN (SELECT value FROM json_each(?))",
                (reason, requested_json),
            )

        return self._commit(_mark)

    def delete_receipts(self, *, source: str) -> CatalogRevision:
        """Remove the keyed receipts of one source. Bronze files are never deleted.

        Raises:
            PITDataError: the source is blank.
        """
        if not source.strip():
            raise PITDataError("deleting receipts requires a source")

        def _delete(connection: sqlite3.Connection) -> None:
            connection.execute("DELETE FROM receipts WHERE source = ?", (source,))

        return self._commit(_delete)

    def _commit(self, work: Callable[[sqlite3.Connection], None]) -> CatalogRevision:
        """Run one catalog mutation, migrating a ``v1`` database first, and commit it."""
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - external filesystem guard
            raise PITDataError("receipt catalog directory could not be created") from exc
        database_existed = not self._database_is_missing()
        connection = self._connect(write=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not database_existed or self._is_uninitialized(connection):
                self._create_schema(connection)
            else:
                self._validate_schema(connection, allow_legacy=True)
                self._migrate_legacy_schema(connection)
            self._validate_schema(connection, allow_legacy=False)
            work(connection)
            connection.execute(
                "UPDATE catalog_metadata SET sequence = sequence + 1, row_count = ? WHERE singleton = 1",
                (self._row_count(connection),),
            )
            revision_row = connection.execute(
                "SELECT sequence, row_count FROM catalog_metadata WHERE singleton = 1"
            ).fetchone()
            sequence, committed_row_count = cast(tuple[int, int], revision_row)
            connection.execute("COMMIT")
        except PITDataError:
            self._rollback(connection)
            raise
        except sqlite3.Error as exc:  # pragma: no cover - transactional infrastructure guard
            self._rollback(connection)
            raise PITDataError("receipt catalog update failed") from exc
        finally:
            connection.close()
        return CatalogRevision(sequence=sequence, row_count=committed_row_count, path=self._database_path)

    @staticmethod
    def _row_count(connection: sqlite3.Connection) -> int:
        stored = connection.execute("SELECT COUNT(*) FROM receipts").fetchone()
        return int(stored[0]) if stored is not None else 0
