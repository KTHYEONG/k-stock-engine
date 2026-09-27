"""One-pass streaming index of every stored Bronze payload into the v2 catalog."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIS_FLOW_SOURCE, KIS_INDUSTRY_SOURCE, LS_FLOW_SOURCE
from src.data.receipt_catalog import (
    BlobEntry,
    CoverageRange,
    EvidenceStatus,
    ReceiptCatalog,
    ReceiptIndexEntry,
)
from src.data.runtime import DataRuntime

__all__ = ["IndexReport", "index_bronze"]

_LOG = logging.getLogger(__name__)

_LEGACY_FLOW_RECEIPTS = "investor_flow"
_INDUSTRY_ENDPOINTS = frozenset({"inquire-price", "search-stock-info"})


@dataclass(frozen=True, slots=True)
class IndexReport:
    """Outcome of one Bronze index pass over every stored payload."""

    scanned: int
    registered: int  # new blob rows
    already_registered: int
    usable: Mapping[str, int]  # source -> usable blobs
    unusable: Mapping[str, int]  # reason -> blobs
    ranges_published: Mapping[str, int]  # source -> ranges
    receipts_removed: Mapping[str, int]  # source -> per-cell receipts replaced by ranges


@dataclass(frozen=True, slots=True)
class _ReferenceMaps:
    """Catalog state read once before the scan, without opening payload files."""

    flow_hashes: frozenset[str]  # blobs referenced by legacy per-cell investor_flow receipts
    flow_receipt_count: int  # legacy per-cell investor_flow receipts
    receipt_source: Mapping[str, str]  # content_hash -> source of its latest receipt
    stored_blobs: frozenset[str]  # blobs already registered, skipped without file reads


@dataclass(frozen=True, slots=True)
class _Decision:
    """One blob's catalog rows plus the counters the report aggregates."""

    blob: BlobEntry
    ranges: tuple[CoverageRange, ...]
    entries: tuple[ReceiptIndexEntry, ...]
    usable_source: str | None
    unusable_reason: str | None


def _catalog_database(runtime: DataRuntime) -> Path:
    return runtime.workspace.bronze_root / "catalog" / "catalog.sqlite3"


def _load_reference_maps(database: Path) -> _ReferenceMaps:
    """Snapshot receipt references and stored blobs from the catalog file alone."""
    empty = _ReferenceMaps(
        flow_hashes=frozenset(), flow_receipt_count=0, receipt_source={}, stored_blobs=frozenset()
    )
    if not database.is_file():
        return empty
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "receipts" not in tables:
                return empty
            flow_hashes = frozenset(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT content_hash FROM receipts WHERE source = ?",
                    (_LEGACY_FLOW_RECEIPTS,),
                ).fetchall()
            )
            count_row = connection.execute(
                "SELECT COUNT(*) FROM receipts WHERE source = ?", (_LEGACY_FLOW_RECEIPTS,)
            ).fetchone()
            latest: dict[str, tuple[datetime, str]] = {}
            for content_hash, source, retrieved_at in connection.execute(
                "SELECT content_hash, source, retrieved_at FROM receipts"
            ).fetchall():
                try:
                    moment = datetime.fromisoformat(str(retrieved_at))
                except ValueError:
                    continue
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=UTC)
                key = str(content_hash)
                current = latest.get(key)
                if current is None or moment > current[0]:
                    latest[key] = (moment, str(source))
            stored = (
                frozenset(
                    str(row[0]) for row in connection.execute("SELECT content_hash FROM blobs").fetchall()
                )
                if "blobs" in tables
                else frozenset()
            )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise PITDataError("receipt catalog database is unreadable") from exc
    return _ReferenceMaps(
        flow_hashes=flow_hashes,
        flow_receipt_count=int(count_row[0]) if count_row is not None else 0,
        receipt_source={key: source for key, (_, source) in latest.items()},
        stored_blobs=stored,
    )


def _kind_for_dir(name: str) -> EvidenceKind | None:
    """Evidence kind stored under one Bronze directory, if it is a known layout."""
    try:
        return EvidenceKind(name)
    except ValueError:
        if name == "dart_documents":
            return EvidenceKind.DISCLOSURES
        if name == "dart_corp_codes":
            return EvidenceKind.SECURITY_MASTER
        return None


def _payload_file(blob_dir: Path) -> Path | None:
    """Payload file of one blob directory without opening anything."""
    candidate = blob_dir / "payload.json"
    if candidate.is_file():
        return candidate
    archive = blob_dir / "payload.zip"
    return archive if archive.is_file() else None


def _stored_retrieved_at(blob_dir: Path) -> datetime | None:
    """Retrieved moment from the blob's own receipt, or None when it is missing."""
    try:
        meta = json.loads((blob_dir / "receipt.json").read_text(encoding="utf-8"))
        moment = datetime.fromisoformat(str(meta["retrieved_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment


def _parse_day(value: object) -> date | None:
    """Session day from an ISO or compact provider date, without validating anything else."""
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, TypeError):
        return None


def _parse_day_list(values: object) -> list[date]:
    """Parseable session days of a listed provider answer, dropping nothing else."""
    if not isinstance(values, list):
        return []
    days: list[date] = []
    for value in values:
        day = _parse_day(value)
        if day is not None:
            days.append(day)
    return days


def _nonblank(value: object) -> bool:
    return bool(str(value or "").strip())


def _fallback_source(kind_dir: str, document: object) -> str:
    """Best-effort source label for a blob that earns no verdict of its own."""
    provider = document.get("provider") if isinstance(document, Mapping) else None
    if kind_dir == "investor_flow":
        if provider == "KIS":
            return KIS_FLOW_SOURCE
        if provider in ("LS", "ls"):
            return LS_FLOW_SOURCE
        return _LEGACY_FLOW_RECEIPTS
    if kind_dir == "industry":
        return KIS_INDUSTRY_SOURCE if provider == "KIS" else kind_dir
    return kind_dir


def _unusable_blob(
    *,
    kind: EvidenceKind,
    content_hash: str,
    payload_path: Path,
    retrieved_at: datetime,
    source: str,
    reason: str,
) -> _Decision:
    return _Decision(
        blob=BlobEntry(
            content_hash=content_hash,
            kind=kind,
            source=source,
            usable=False,
            unusable_reason=reason,
            retrieved_at=retrieved_at,
            payload_path=payload_path,
        ),
        ranges=(),
        entries=(),
        usable_source=None,
        unusable_reason=reason,
    )


def _classify_investor_flow(
    *,
    kind: EvidenceKind,
    kind_dir: str,
    content_hash: str,
    payload_path: Path,
    retrieved_at: datetime,
    document: object,
    flow_hashes: frozenset[str],
) -> _Decision:
    """Apply the investor_flow classification rows in order; the first match wins."""
    info = document if isinstance(document, Mapping) else {}
    provider = info.get("provider")
    rows = info.get("rows")
    has_rows = isinstance(rows, list) and len(rows) > 0
    raw_query = info.get("query")
    query = raw_query if isinstance(raw_query, Mapping) else {}
    symbol = str(query.get("symbol") or "").strip()
    if provider == "LS" and has_rows and symbol and _nonblank(query.get("start")) and _nonblank(query.get("end")):
        start = _parse_day(query.get("start"))
        end = _parse_day(query.get("end"))
        if start is None or end is None or end < start:
            return _unusable_blob(
                kind=kind, content_hash=content_hash, payload_path=payload_path,
                retrieved_at=retrieved_at, source=LS_FLOW_SOURCE, reason="unrecognized",
            )
        return _Decision(
            blob=BlobEntry(
                content_hash=content_hash, kind=kind, source=LS_FLOW_SOURCE, usable=True,
                unusable_reason=None, retrieved_at=retrieved_at, payload_path=payload_path,
            ),
            ranges=(
                CoverageRange(
                    source=LS_FLOW_SOURCE, subject=symbol, start=start, end=end,
                    status=EvidenceStatus.SUCCESS, content_hash=content_hash, retrieved_at=retrieved_at,
                ),
            ),
            entries=(),
            usable_source=LS_FLOW_SOURCE,
            unusable_reason=None,
        )
    if provider == "LS" and info.get("status") == "missing_sessions" and content_hash in flow_hashes:
        page_symbol = str(info.get("symbol") or "").strip()
        days = _parse_day_list(info.get("missing_sessions"))
        if not page_symbol or not days:
            return _unusable_blob(
                kind=kind, content_hash=content_hash, payload_path=payload_path,
                retrieved_at=retrieved_at, source=LS_FLOW_SOURCE, reason="unrecognized",
            )
        return _Decision(
            blob=BlobEntry(
                content_hash=content_hash, kind=kind, source=LS_FLOW_SOURCE, usable=False,
                unusable_reason="negative_marker", retrieved_at=retrieved_at, payload_path=payload_path,
            ),
            ranges=(
                CoverageRange(
                    source=LS_FLOW_SOURCE, subject=page_symbol, start=min(days), end=max(days),
                    status=EvidenceStatus.EMPTY, content_hash=None, retrieved_at=retrieved_at,
                ),
            ),
            entries=(),
            usable_source=None,
            unusable_reason="negative_marker",
        )
    if provider == "KIS" and has_rows and symbol and _nonblank(query.get("anchor")):
        anchor = _parse_day(query.get("anchor"))
        row_days: list[date] = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    day = _parse_day(row.get("stck_bsop_date"))
                    if day is not None:
                        row_days.append(day)
        if anchor is None or not row_days or min(row_days) > anchor:
            return _unusable_blob(
                kind=kind, content_hash=content_hash, payload_path=payload_path,
                retrieved_at=retrieved_at, source=KIS_FLOW_SOURCE, reason="unrecognized",
            )
        return _Decision(
            blob=BlobEntry(
                content_hash=content_hash, kind=kind, source=KIS_FLOW_SOURCE, usable=True,
                unusable_reason=None, retrieved_at=retrieved_at, payload_path=payload_path,
            ),
            ranges=(
                CoverageRange(
                    source=KIS_FLOW_SOURCE, subject=symbol, start=min(row_days), end=anchor,
                    status=EvidenceStatus.SUCCESS, content_hash=content_hash, retrieved_at=retrieved_at,
                ),
            ),
            entries=(),
            usable_source=KIS_FLOW_SOURCE,
            unusable_reason=None,
        )
    if provider == "KIS" and not has_rows:
        return _unusable_blob(
            kind=kind, content_hash=content_hash, payload_path=payload_path,
            retrieved_at=retrieved_at, source=KIS_FLOW_SOURCE, reason="kis_mapped_only",
        )
    if provider == "LS" and not has_rows:
        return _unusable_blob(
            kind=kind, content_hash=content_hash, payload_path=payload_path,
            retrieved_at=retrieved_at, source=LS_FLOW_SOURCE, reason="ls_mapped_only",
        )
    if provider == "ls":
        return _unusable_blob(
            kind=kind, content_hash=content_hash, payload_path=payload_path,
            retrieved_at=retrieved_at, source=LS_FLOW_SOURCE, reason="legacy_error_page",
        )
    return _unusable_blob(
        kind=kind, content_hash=content_hash, payload_path=payload_path,
        retrieved_at=retrieved_at, source=_fallback_source(kind_dir, document), reason="unrecognized",
    )


def _classify_industry(
    *,
    kind: EvidenceKind,
    kind_dir: str,
    content_hash: str,
    payload_path: Path,
    retrieved_at: datetime,
    document: object,
) -> _Decision:
    """Register one KIS classification page with its keyed receipt, or mark it unrecognized."""
    info = document if isinstance(document, Mapping) else {}
    endpoint = str(info.get("endpoint") or "").strip()
    symbol = str(info.get("symbol") or "").strip()
    if info.get("provider") == "KIS" and endpoint in _INDUSTRY_ENDPOINTS and symbol:
        try:
            collected = datetime.fromisoformat(str(info.get("collected_at")))
        except (ValueError, TypeError):
            collected = None
        if collected is not None:
            if collected.tzinfo is None:
                collected = collected.replace(tzinfo=UTC)
            collected_day = collected.date()
            records = info.get("records")
            status = EvidenceStatus.SUCCESS if isinstance(records, list) and records else EvidenceStatus.EMPTY
            natural_key = f"{endpoint}:{symbol}:{collected_day.isoformat()}"
            return _Decision(
                blob=BlobEntry(
                    content_hash=content_hash, kind=kind, source=KIS_INDUSTRY_SOURCE, usable=True,
                    unusable_reason=None, retrieved_at=retrieved_at, payload_path=payload_path,
                ),
                ranges=(),
                entries=(
                    ReceiptIndexEntry(
                        source=KIS_INDUSTRY_SOURCE, natural_key=natural_key, as_of=collected_day,
                        fiscal_period=None, status=status, content_hash=content_hash,
                        retrieved_at=retrieved_at, payload_path=payload_path,
                    ),
                ),
                usable_source=KIS_INDUSTRY_SOURCE,
                unusable_reason=None,
            )
    return _unusable_blob(
        kind=kind, content_hash=content_hash, payload_path=payload_path,
        retrieved_at=retrieved_at, source=_fallback_source(kind_dir, document), reason="unrecognized",
    )


def _classify_blob(
    *,
    kind: EvidenceKind,
    kind_dir: str,
    content_hash: str,
    payload_path: Path,
    maps: _ReferenceMaps,
) -> _Decision:
    """Read one payload, verify its hash, and classify it exactly once.

    Raises:
        PITDataError: the payload's bytes do not hash to its directory name.
    """
    raw = payload_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != content_hash:
        raise PITDataError(f"Bronze payload hash mismatch for {payload_path}")
    document: object | None = None
    if payload_path.suffix == ".json":
        try:
            document = json.loads(raw)
        except ValueError:
            document = None
    retrieved_at = _stored_retrieved_at(payload_path.parent)
    if retrieved_at is None:
        fallback = datetime.fromtimestamp(payload_path.stat().st_mtime, tz=UTC)
        return _unusable_blob(
            kind=kind, content_hash=content_hash, payload_path=payload_path,
            retrieved_at=fallback, source=_fallback_source(kind_dir, document), reason="missing_receipt",
        )
    if kind_dir == "investor_flow":
        return _classify_investor_flow(
            kind=kind, kind_dir=kind_dir, content_hash=content_hash, payload_path=payload_path,
            retrieved_at=retrieved_at, document=document, flow_hashes=maps.flow_hashes,
        )
    if kind_dir == "industry":
        return _classify_industry(
            kind=kind, kind_dir=kind_dir, content_hash=content_hash, payload_path=payload_path,
            retrieved_at=retrieved_at, document=document,
        )
    receipt_source = maps.receipt_source.get(content_hash)
    if receipt_source is not None:
        return _Decision(
            blob=BlobEntry(
                content_hash=content_hash, kind=kind, source=receipt_source, usable=True,
                unusable_reason=None, retrieved_at=retrieved_at, payload_path=payload_path,
            ),
            ranges=(),
            entries=(),
            usable_source=receipt_source,
            unusable_reason=None,
        )
    return _unusable_blob(
        kind=kind, content_hash=content_hash, payload_path=payload_path,
        retrieved_at=retrieved_at, source=kind_dir, reason="unreferenced",
    )


def index_bronze(
    runtime: DataRuntime, *, dry_run: bool, batch_size: int = 1000, emit: Callable[[Mapping[str, object]], None]
) -> IndexReport:
    """Register every stored Bronze payload in the catalog and classify it once.

    Payload files are read one at a time and released before the next is
    opened, so memory does not grow with the size of Bronze. Blobs already in
    the ``blobs`` table are skipped without reading their files, so a rerun
    after an interruption continues where it stopped. Classification never
    rewrites or deletes a payload; an unusable blob keeps its bytes and gets a
    reason.

    Raises:
        PITDataError: a payload's bytes do not hash to its directory name.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise PITDataError("batch_size must be a positive integer")
    bronze_root = runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    maps = _load_reference_maps(_catalog_database(runtime))
    started = time.monotonic()
    scanned = 0
    already_registered = 0
    registered = 0
    usable: dict[str, int] = {}
    unusable: dict[str, int] = {}
    ranges_published: dict[str, int] = {}
    pending_blobs: list[BlobEntry] = []
    pending_ranges: list[CoverageRange] = []
    pending_entries: list[ReceiptIndexEntry] = []

    def _flush() -> None:
        nonlocal registered
        if not pending_blobs:
            return
        if not dry_run:
            catalog.publish(pending_entries, blobs=pending_blobs, ranges=pending_ranges)
        registered += len(pending_blobs)
        emit({
            "type": "progress",
            "scanned": scanned,
            "registered": registered,
            "elapsed_s": time.monotonic() - started,
        })
        pending_blobs.clear()
        pending_ranges.clear()
        pending_entries.clear()

    with os.scandir(bronze_root) as kind_scan:
        for kind_entry in kind_scan:
            if kind_entry.name == "catalog" or kind_entry.name.startswith("."):
                continue
            if not kind_entry.is_dir(follow_symlinks=False):
                continue
            kind = _kind_for_dir(kind_entry.name)
            if kind is None:
                _LOG.debug("[DATA] stage=bronze_index status=skipped_directory directory=%s", kind_entry.name)
                continue
            with os.scandir(kind_entry.path) as blob_scan:
                for blob_entry in blob_scan:
                    if blob_entry.name.startswith(".") or not blob_entry.is_dir(follow_symlinks=False):
                        continue
                    content_hash = blob_entry.name
                    if content_hash in maps.stored_blobs:
                        scanned += 1
                        already_registered += 1
                        continue
                    payload_path = _payload_file(Path(blob_entry.path))
                    if payload_path is None:
                        continue
                    scanned += 1
                    decision = _classify_blob(
                        kind=kind,
                        kind_dir=kind_entry.name,
                        content_hash=content_hash,
                        payload_path=payload_path,
                        maps=maps,
                    )
                    pending_blobs.append(decision.blob)
                    pending_ranges.extend(decision.ranges)
                    pending_entries.extend(decision.entries)
                    if decision.usable_source is not None:
                        usable[decision.usable_source] = usable.get(decision.usable_source, 0) + 1
                    elif decision.unusable_reason is not None:
                        reason = decision.unusable_reason
                        unusable[reason] = unusable.get(reason, 0) + 1
                    for item in decision.ranges:
                        ranges_published[item.source] = ranges_published.get(item.source, 0) + 1
                    if len(pending_blobs) >= batch_size:
                        _flush()
    _flush()
    receipts_removed: dict[str, int] = {}
    if maps.flow_receipt_count > 0 and scanned > 0:
        receipts_removed = {_LEGACY_FLOW_RECEIPTS: maps.flow_receipt_count}
        if not dry_run:
            catalog.delete_receipts(source=_LEGACY_FLOW_RECEIPTS)
    _LOG.info(
        "[DATA] stage=bronze_index scanned=%d registered=%d already_registered=%d ranges=%d receipts_removed=%d dry_run=%s",
        scanned,
        registered,
        already_registered,
        sum(ranges_published.values()),
        sum(receipts_removed.values()),
        dry_run,
    )
    return IndexReport(
        scanned=scanned,
        registered=registered,
        already_registered=already_registered,
        usable=usable,
        unusable=unusable,
        ranges_published=ranges_published,
        receipts_removed=receipts_removed,
    )
