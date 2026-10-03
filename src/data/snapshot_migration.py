"""Re-wrap legacy snapshot KRX session pages as scoped Bronze receipts."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KRX_DAILY_MARKET_SOURCE, KRX_SECURITY_MASTER_SOURCE
from src.data.jobs.krx import (
    _ANSWERED,
    krx_daily_market_scoped_payload,
    krx_security_master_scoped_payload,
)
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

__all__ = ["SnapshotMigrationReport", "migrate_snapshot_krx"]

_LOG = logging.getLogger(__name__)

_LIVE_MARKETS = frozenset({"KOSPI", "KOSDAQ"})


@dataclass(frozen=True, slots=True)
class SnapshotMigrationReport:
    """Outcome of one snapshot migration run.

    Attributes:
        kind: ``daily_market`` or ``security_master``.
        migrated: Sessions newly persisted as scoped receipts.
        already_answered: Sessions skipped because the scoped catalog already holds an answered receipt.
        skipped_blobs: Legacy blobs that are not single-session pages (manifests, warm-ups, aggregates),
            keyed by reason.
        rejected_sessions: Sessions whose candidate pages failed validation, keyed by session ISO date with
            the reason; never persisted.
    """

    kind: str
    migrated: int
    already_answered: int
    skipped_blobs: Mapping[str, int]
    rejected_sessions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Candidate:
    session: date
    retrieved_at: datetime
    content_hash: str
    payload_path: Path


_SOURCES: Mapping[EvidenceKind, str] = {
    EvidenceKind.DAILY_MARKET: KRX_DAILY_MARKET_SOURCE,
    EvidenceKind.SECURITY_MASTER: KRX_SECURITY_MASTER_SOURCE,
}


def _read_verified(payload_path: Path, content_hash: str) -> Any:
    raw = payload_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != content_hash:
        raise PITDataError(f"legacy payload hash mismatch for {payload_path}")
    return json.loads(raw)


def _parse_day_text(text: object) -> date | None:
    """Session day from compact ``YYYYMMDD`` or ISO provider dates."""
    raw = str(text or "").strip()
    if not raw:
        return None
    compact = raw.replace("-", "")
    if len(compact) == 8 and compact.isdigit():
        try:
            return date(int(compact[:4]), int(compact[4:6]), int(compact[6:8]))
        except ValueError:
            return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _extract_records(document: Any) -> list[dict[str, Any]] | None:
    if not isinstance(document, dict):
        return None
    records = document.get("records")
    if isinstance(records, list):
        return records
    legacy = document.get("OutBlock_1")
    if isinstance(legacy, list):
        return legacy
    return None


def _page_session(kind: EvidenceKind, document: Any) -> date | None:
    """Single session a legacy page answers, or None when it is not a one-session page."""
    if not isinstance(document, dict):
        return None
    records = _extract_records(document)
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        return None
    if kind is EvidenceKind.DAILY_MARKET:
        days = {_parse_day_text(r.get("BAS_DD")) for r in records}
        if len(days) != 1:
            return None
        return days.pop()
    as_of = document.get("as_of", document.get("session"))
    return _parse_day_text(as_of) if as_of not in (None, "") else None


def _live_records(kind: EvidenceKind, document: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = document.get("records", document.get("OutBlock_1", []))
    records = [dict(r) for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    if kind is EvidenceKind.DAILY_MARKET:
        return [r for r in records if str(r.get("MKT_NM") or "").strip() in _LIVE_MARKETS or not str(r.get("MKT_NM") or "").strip()]
    kept: list[dict[str, Any]] = []
    for record in records:
        market = str(record.get("MKT_TP_NM") or record.get("MKT_NM") or "").strip()
        if market and market not in _LIVE_MARKETS:
            continue
        kept.append(record)
    return kept


def _wrap(kind: EvidenceKind, records: Sequence[Mapping[str, Any]], session: date, retrieved_at: datetime) -> ScopedRawPayload:
    if kind is EvidenceKind.DAILY_MARKET:
        return krx_daily_market_scoped_payload(records=records, session=session, retrieved_at=retrieved_at)
    return krx_security_master_scoped_payload(records=records, session=session, retrieved_at=retrieved_at)


def _coerce_retrieved_at(value: object) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    # A naive stamp has no provable zone (UTC vs KST is a 9-hour PIT difference); skip rather than guess.
    return None if moment.tzinfo is None else moment


def _session_universe(*, evidence_start: date, last: date) -> frozenset[date]:
    """KRX sessions in the closed scope window; empty when the window holds none."""
    from src.core.krx_calendar import xkrx_session_calendar
    from src.core.time import KRX_TZ

    try:
        calendar = xkrx_session_calendar(start=evidence_start, end=last)
    except ValueError:
        return frozenset()
    return frozenset(instant.astimezone(KRX_TZ).date() for instant in calendar.sessions)


def _scan(
    kind: EvidenceKind, kind_root: Path, *, evidence_start: date
) -> tuple[dict[date, _Candidate], dict[str, int]]:
    """Best candidate per session (latest retrieval, then lowest hash) and skip counts by reason."""
    best: dict[date, _Candidate] = {}
    skipped: dict[str, int] = {}
    provisional: list[_Candidate] = []

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for blob_dir in sorted(p for p in kind_root.iterdir() if p.is_dir()):
        receipt_path = blob_dir / "receipt.json"
        payload_path = blob_dir / "payload.json"
        if not receipt_path.is_file() or not payload_path.is_file():
            _skip("missing_receipt_or_payload")
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _skip("invalid_receipt")
            continue
        source_path = str(receipt.get("source_path", ""))
        lowered = source_path.lower()
        if source_path.startswith("manifest:") or lowered.startswith("manifest:"):
            _skip("manifest")
            continue
        if "warmup" in lowered:
            _skip("warmup")
            continue
        if lowered.startswith("data/evidence") or "aggregate" in lowered:
            _skip("aggregate")
            continue
        raw = payload_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != blob_dir.name:
            raise PITDataError(f"legacy payload hash mismatch for {payload_path}")
        try:
            document = json.loads(raw)
        except ValueError:
            _skip("invalid_json")
            continue
        session = _page_session(kind, document)
        del document
        if session is None:
            _skip("not_single_session_page")
            continue
        retrieved_at = _coerce_retrieved_at(receipt.get("retrieved_at"))
        if retrieved_at is None:
            _skip("invalid_retrieved_at")
            continue
        provisional.append(_Candidate(session, retrieved_at, blob_dir.name, payload_path))
    if not provisional:
        return best, skipped
    last = max(candidate.session for candidate in provisional)
    if last < evidence_start:
        for _ in provisional:
            _skip("outside_scope_sessions")
        return best, skipped
    sessions = _session_universe(evidence_start=evidence_start, last=last)
    for candidate in provisional:
        if candidate.session < evidence_start or candidate.session not in sessions:
            _skip("outside_scope_sessions")
            continue
        current = best.get(candidate.session)
        if current is None or (candidate.retrieved_at, _neg(candidate.content_hash)) > (
            current.retrieved_at,
            _neg(current.content_hash),
        ):
            best[candidate.session] = candidate
    return best, skipped


def _neg(text: str) -> tuple[int, ...]:
    """Sort key making the lexicographically smaller hash win a max() tie-break."""
    return tuple(-ord(ch) for ch in text)


def migrate_snapshot_krx(
    runtime: DataRuntime,
    *,
    snapshot_root: Path,
    kinds: Sequence[EvidenceKind] = (EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER),
    dry_run: bool,
    batch_sessions: int = 200,
    emit: Callable[[Mapping[str, object]], None],
) -> tuple[SnapshotMigrationReport, ...]:
    """Persist legacy snapshot KRX session pages as scoped Bronze receipts with their original retrieval time.

    Why: the snapshot already holds every KRX session page the scope needs, but without catalog receipts the KRX
    jobs would re-request ~5,000 sessions. Re-wrapping through the live scoped-payload constructors and the
    ``ScopedBronzeWriter`` contract keeps one validation path for migrated and freshly collected pages.

    PIT: ``retrieved_at`` is the legacy receipt's ``retrieved_at`` (never the migration time, never
    ``ingested_at``), so availability semantics match what was actually observed.

    Args:
        runtime: Scoped runtime whose catalog and Bronze root receive the receipts.
        snapshot_root: Legacy ``data/bronze/stocks`` directory (read-only).
        kinds: KRX evidence kinds to migrate.
        dry_run: Classify and validate only; persist nothing.
        batch_sessions: Sessions per ``persist_many`` call (one catalog revision each).
        emit: Progress sink; one structured line per batch and one summary per kind.

    Returns:
        One report per requested kind, in ``kinds`` order.

    Raises:
        PITDataError: a legacy payload's bytes do not hash to its directory name, or ``snapshot_root`` is the
            scoped Bronze root itself.
        ValueError: ``batch_sessions < 1`` or an unsupported kind.
    """
    if isinstance(batch_sessions, bool) or int(batch_sessions) < 1:
        raise ValueError(f"invalid batch_sessions {batch_sessions!r}: must be a positive integer")
    for kind in kinds:
        if kind not in _SOURCES:
            raise ValueError(f"unsupported kind {kind!r}: expected daily_market or security_master")
    bronze_root = runtime.workspace.bronze_root
    if Path(snapshot_root).expanduser().resolve() == bronze_root.expanduser().resolve():
        raise PITDataError("snapshot_root must not be the scoped Bronze root")
    catalog = ReceiptCatalog(bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    started = time.monotonic()
    reports: list[SnapshotMigrationReport] = []
    for kind in kinds:
        source = _SOURCES[EvidenceKind(kind)]
        kind_root = Path(snapshot_root) / EvidenceKind(kind).value
        if not kind_root.is_dir():
            report = SnapshotMigrationReport(EvidenceKind(kind).value, 0, 0, {}, {})
            emit({"type": "summary", "kind": report.kind, "migrated": 0, "already_answered": 0,
                  "skipped_blobs": {}, "rejected_sessions": 0, "dry_run": dry_run})
            _LOG.info("[DATA] stage=snapshot_migration kind=%s migrated=0 pending=0 elapsed_s=%.1f", report.kind, time.monotonic() - started)
            reports.append(report)
            continue
        best, skipped = _scan(EvidenceKind(kind), kind_root, evidence_start=runtime.scope.evidence_start)
        answered = catalog.latest(source=source, natural_keys={day.isoformat() for day in best})
        todo = [
            day
            for day in sorted(best)
            if not (answered.get(day.isoformat()) is not None and answered[day.isoformat()].status in _ANSWERED)
        ]
        already = len(best) - len(todo)
        rejected: dict[str, str] = {}
        migrated = 0
        step = int(batch_sessions)
        for start in range(0, len(todo), step):
            window = todo[start : start + step]
            batch: list[ScopedRawPayload] = []
            for day in window:
                candidate = best[day]
                document = _read_verified(candidate.payload_path, candidate.content_hash)
                records = _live_records(EvidenceKind(kind), document)
                del document
                if not records:
                    rejected[day.isoformat()] = "no_live_market_records"
                    continue
                batch.append(_wrap(EvidenceKind(kind), records, day, candidate.retrieved_at))
            accepted = _persist_batch(writer, batch, rejected=rejected, dry_run=dry_run)
            migrated += accepted
            pending = max(len(todo) - (start + len(window)), 0)
            emit({
                "type": "progress", "kind": EvidenceKind(kind).value, "migrated": migrated,
                "pending": pending, "elapsed_s": round(time.monotonic() - started, 1),
            })
            _LOG.info(
                "[DATA] stage=snapshot_migration kind=%s migrated=%d pending=%d elapsed_s=%.1f",
                EvidenceKind(kind).value, migrated, pending, time.monotonic() - started,
            )
        report = SnapshotMigrationReport(EvidenceKind(kind).value, migrated, already, dict(skipped), dict(rejected))
        emit({"type": "summary", "kind": report.kind, "migrated": migrated, "already_answered": already,
              "skipped_blobs": dict(skipped), "rejected_sessions": len(rejected), "dry_run": dry_run})
        _LOG.info(
            "[DATA] stage=snapshot_migration kind=%s migrated=%d already_answered=%d skipped=%d rejected=%d",
            report.kind, migrated, already, sum(skipped.values()), len(rejected),
        )
        reports.append(report)
    return tuple(reports)


def _persist_batch(
    writer: ScopedBronzeWriter, batch: Sequence[ScopedRawPayload], *, rejected: dict[str, str], dry_run: bool
) -> int:
    """Persist one batch; on a contract failure fall back to per-session writes so one bad page is isolated."""
    if not batch:
        return 0
    if dry_run:
        accepted = 0
        for payload in batch:
            try:
                writer.validate(payload)
            except PITDataError as exc:
                rejected[payload.natural_key] = str(exc)
            else:
                accepted += 1
        return accepted
    try:
        writer.persist_many(batch)
        return len(batch)
    except PITDataError:
        accepted = 0
        for payload in batch:
            try:
                writer.persist(payload)
            except PITDataError as exc:
                rejected[payload.natural_key] = str(exc)
            else:
                accepted += 1
        return accepted
