"""Content-hash cache of derived fact-page header fields.

Reparse planning, document fetch planning and benchmark sampling only test a
handful of header fields of each latest fact page. Fact blobs are immutable
and content-addressed, so those fields are derived once per content hash and
reused without ever going stale.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.data.receipt_catalog import ReceiptIndexEntry

__all__ = [
    "REVISION",
    "FactPageMeta",
    "FactPageMetaStore",
]

REVISION: Final[int] = 1

_BATCH_SIZE: Final[int] = 500
_SQLITE_TIMEOUT_SECONDS: Final[float] = 30.0
_BUSY_TIMEOUT_MS: Final[int] = 30_000

_LOG = logging.getLogger(__name__)

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS fact_page_meta (
    content_hash TEXT PRIMARY KEY,
    derivation_version INTEGER NOT NULL,
    source_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    raw_document_hash TEXT NOT NULL,
    document_not_found INTEGER NOT NULL,
    has_labels INTEGER NOT NULL,
    is_financial INTEGER NOT NULL,
    reprt_code TEXT NOT NULL,
    biz_year TEXT NOT NULL,
    rcept_no TEXT NOT NULL
) WITHOUT ROWID
"""


@dataclass(frozen=True, slots=True)
class FactPageMeta:
    """Header facts of one fact page, derived once per content hash."""

    content_hash: str
    source_kind: str
    status: str
    parser_version: str
    raw_document_hash: str
    document_not_found: bool
    has_labels: bool
    is_financial: bool
    reprt_code: str
    biz_year: str
    rcept_no: str


def _read_page_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def _derive_meta(entry: ReceiptIndexEntry, page: Mapping[str, Any]) -> FactPageMeta:
    """Derive header facts with the same field rules the consumers use."""
    from src.data.dart_document_benchmark import (
        _is_financial_page,
        _page_field,
        _page_receipt,
        standard_labels,
    )
    from src.data.jobs.dart_documents import _is_document_not_found

    parts = entry.natural_key.split(":")
    return FactPageMeta(
        content_hash=entry.content_hash,
        source_kind=str(page.get("source_kind") or ""),
        status=str(page.get("status") or ""),
        parser_version=str(page.get("parser_version") or ""),
        raw_document_hash=str(page.get("raw_document_hash") or ""),
        document_not_found=bool(_is_document_not_found(page)),
        has_labels=bool(standard_labels(page)),
        is_financial=bool(_is_financial_page(page)),
        reprt_code=_page_field(page, "reprt_code") or (parts[-1] if parts else ""),
        biz_year=_page_field(page, "biz_year") or (parts[1] if len(parts) > 1 else ""),
        rcept_no=_page_receipt(page),
    )


def _meta_from_row(row: Sequence[Any]) -> FactPageMeta:
    return FactPageMeta(
        content_hash=str(row[0]),
        source_kind=str(row[1]),
        status=str(row[2]),
        parser_version=str(row[3]),
        raw_document_hash=str(row[4]),
        document_not_found=bool(row[5]),
        has_labels=bool(row[6]),
        is_financial=bool(row[7]),
        reprt_code=str(row[8]),
        biz_year=str(row[9]),
        rcept_no=str(row[10]),
    )


class FactPageMetaStore:
    """SQLite cache of ``FactPageMeta`` next to the receipt catalog.

    The cache is derived data: every row is a pure function of an immutable page blob and of
    ``REVISION``, so it can be deleted at any time and is rebuilt on demand. It is not part of
    the receipt catalog schema or digest.

    Args:
        catalog_root: Directory of the receipt catalog; the cache file lives inside it.
    """

    def __init__(self, catalog_root: Path) -> None:
        self._db_path = Path(catalog_root).expanduser() / "fact_page_meta.sqlite3"

    def resolve(
        self, entries: Iterable[ReceiptIndexEntry]
    ) -> Iterator[tuple[ReceiptIndexEntry, FactPageMeta]]:
        """Yield each entry with its page metadata, reading a page only on a cache miss.

        Entries are processed in batches: cached rows are looked up together, missing pages are read,
        derived and stored in one transaction, and a ``[DATA]`` heartbeat with hit and miss counts is
        logged per batch. Entries whose payload is unreadable or is not a JSON object are skipped and
        never cached.

        Args:
            entries: Latest fact receipts, typically from ``catalog.entries(source="financial_facts")``.

        Returns:
            An iterator of ``(entry, meta)`` in input order.
        """
        batch: list[ReceiptIndexEntry] = []
        for entry in entries:
            batch.append(entry)
            if len(batch) >= _BATCH_SIZE:
                yield from self._resolve_batch(batch)
                batch = []
        if batch:
            yield from self._resolve_batch(batch)

    def _resolve_batch(
        self, batch: Sequence[ReceiptIndexEntry]
    ) -> Iterator[tuple[ReceiptIndexEntry, FactPageMeta]]:
        hashes = list(dict.fromkeys(entry.content_hash for entry in batch))
        cached = self._lookup(hashes)
        missing = [content_hash for content_hash in hashes if content_hash not in cached]
        derived: dict[str, FactPageMeta] = {}
        if missing:
            by_hash: dict[str, ReceiptIndexEntry] = {}
            for entry in batch:
                by_hash.setdefault(entry.content_hash, entry)
            for content_hash in missing:
                representative = by_hash[content_hash]
                page = _read_page_file(Path(str(representative.payload_path)))
                if page is None:
                    continue
                derived[content_hash] = _derive_meta(representative, page)
            self._store(tuple(derived.values()))
        _LOG.info("[DATA] stage=fact_page_meta hits=%d misses=%d", len(cached), len(missing))
        for entry in batch:
            meta = cached.get(entry.content_hash, derived.get(entry.content_hash))
            if meta is None:
                continue
            yield entry, meta

    def _lookup(self, hashes: Sequence[str]) -> dict[str, FactPageMeta]:
        if not hashes or not self._db_path.exists():
            return {}
        requested_json = json.dumps(list(hashes), separators=(",", ":"))
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self._db_path, timeout=_SQLITE_TIMEOUT_SECONDS)
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            rows = connection.execute(
                "SELECT content_hash, source_kind, status, parser_version, raw_document_hash,"
                " document_not_found, has_labels, is_financial, reprt_code, biz_year, rcept_no"
                " FROM fact_page_meta WHERE content_hash IN (SELECT value FROM json_each(?))"
                " AND derivation_version = ?",
                (requested_json, REVISION),
            ).fetchall()
            return {str(row[0]): _meta_from_row(row) for row in rows}
        except sqlite3.OperationalError:  # pragma: no cover - lock or missing table falls back to reads
            return {}
        except sqlite3.DatabaseError:
            with suppress(OSError):
                self._db_path.unlink()
            return {}
        except (sqlite3.Error, OSError):  # pragma: no cover - transient lookup falls back to reads
            return {}
        finally:
            if connection is not None:
                connection.close()

    def _store(self, metas: Sequence[FactPageMeta]) -> None:
        if not metas:
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - unwritable catalog directory skips caching
            return
        try:
            self._write_rows(metas)
        except sqlite3.OperationalError:  # pragma: no cover - a busy cache skips this batch
            return
        except sqlite3.DatabaseError:
            with suppress(OSError):
                self._db_path.unlink()
            with suppress(sqlite3.Error, OSError):
                self._write_rows(metas)
        except (sqlite3.Error, OSError):
            return

    def _write_rows(self, metas: Sequence[FactPageMeta]) -> None:
        connection = sqlite3.connect(
            self._db_path, timeout=_SQLITE_TIMEOUT_SECONDS, isolation_level=None
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            with suppress(sqlite3.Error):
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    "INSERT OR REPLACE INTO fact_page_meta("
                    "content_hash, derivation_version, source_kind, status, parser_version,"
                    " raw_document_hash, document_not_found, has_labels, is_financial,"
                    " reprt_code, biz_year, rcept_no"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            meta.content_hash,
                            REVISION,
                            meta.source_kind,
                            meta.status,
                            meta.parser_version,
                            meta.raw_document_hash,
                            1 if meta.document_not_found else 0,
                            1 if meta.has_labels else 0,
                            1 if meta.is_financial else 0,
                            meta.reprt_code,
                            meta.biz_year,
                            meta.rcept_no,
                        )
                        for meta in metas
                    ],
                )
                connection.execute("COMMIT")
            except sqlite3.Error:  # pragma: no cover - transactional rollback guard
                with suppress(sqlite3.Error):
                    connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()
