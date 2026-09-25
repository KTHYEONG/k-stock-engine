"""Scope-local receipt index with latest-state lookup by source natural key."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from src.data.schemas import PITDataError

__all__ = [
    "CatalogRevision",
    "EvidenceStatus",
    "ReceiptCatalog",
    "ReceiptIndexEntry",
]

_CATALOG_SCHEMA_VERSION = 1
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


class ReceiptCatalog:
    """Scope-local receipt index with latest-state lookup by source natural key.

    Backed by ``<root>/catalog.sqlite3``. A publish is one transaction, so
    readers see either all or none of a batch, and concurrent collectors are
    serialized by SQLite's write lock instead of rewriting a snapshot.
    Payload paths are stored relative to the Bronze root (the catalog root's
    parent), so a moved data root keeps a valid catalog.
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
        if self._database_is_missing():  # pragma: no cover - deletion race guard
            return None
        connection = self._connect(write=False)
        try:
            self._validate_schema(connection)
        except Exception:
            connection.close()
            raise
        return connection

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return tuple(str(row[1]) for row in rows)

    def _validate_schema(self, connection: sqlite3.Connection) -> tuple[int, int]:
        try:
            if self._table_columns(connection, "catalog_metadata") != _METADATA_COLUMNS:
                raise PITDataError("receipt catalog metadata schema is invalid")
            metadata_rows = connection.execute(
                "SELECT schema_version, sequence, row_count "
                "FROM catalog_metadata "
                "WHERE singleton = 1"
            ).fetchall()
            if len(metadata_rows) != 1:
                raise PITDataError("receipt catalog metadata is missing")
            schema_version, sequence, row_count = metadata_rows[0]
            if int(schema_version) != _CATALOG_SCHEMA_VERSION:
                raise PITDataError(f"unsupported receipt catalog schema_version: {schema_version!r}")
            if self._table_columns(connection, "receipts") != _RECEIPT_COLUMNS:
                raise PITDataError("receipt catalog entry schema is invalid")
            index_names = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'receipts'"
                ).fetchall()
            }
            if "idx_receipts_source_status" not in index_names:
                raise PITDataError("receipt catalog status index is missing")
        except sqlite3.Error as exc:
            raise PITDataError("receipt catalog database is corrupt") from exc
        sequence_int = int(sequence)
        row_count_int = int(row_count)
        if sequence_int < 0 or row_count_int < 0:
            raise PITDataError("receipt catalog metadata is invalid")
        return sequence_int, row_count_int

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
        connection.execute(
            "INSERT OR IGNORE INTO catalog_metadata(singleton, schema_version, sequence, row_count) "
            "VALUES (1, ?, 0, 0)",
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

    def publish(self, entries: Sequence[ReceiptIndexEntry]) -> CatalogRevision:
        """Validate a complete batch, then atomically publish it to SQLite."""
        prepared = self._prepare_entries(entries)
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - external filesystem guard
            raise PITDataError("receipt catalog directory could not be created") from exc
        database_existed = not self._database_is_missing()
        connection = self._connect(write=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if database_existed:
                _, row_count = self._validate_schema(connection)
            else:
                self._create_schema(connection)
                _, row_count = self._validate_schema(connection)
            for entry in prepared:
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
        """Return entries for the requested keys without scanning the full catalog."""
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
                FROM receipts AS r
                JOIN json_each(?) AS requested ON requested.value = r.natural_key
                WHERE r.source = ?
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
