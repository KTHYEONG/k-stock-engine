"""KRX KIND exchange-notice collection: title-keyword search windows plus notice bodies.

KIND carries the KOSPI exchange notices OpenDART omits (delisting decisions with
liquidation-trading dates, administrative designations/releases). Search windows
store every raw result page so Silver re-parses from Bronze, plus the total KIND
declared, so an incomplete completed window can never be accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.config.providers import KindPolicy, ProviderPolicy
from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE, KIND_NOTICE_SEARCH_SOURCE
from src.data.jobs.dart import disclosure_windows
from src.data.jobs.runner import JobContext, JobSpec, JobUnit
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.krx.kind import KindClient, KindNoticeRow, parse_kind_search_page
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "KIND_CHUNK_SIZE",
    "KIND_JOBS",
    "KindNoticeDocumentJob",
    "KindNoticeSearchJob",
    "build_kind_job_context",
    "kind_document_targets",
    "normalize_kind_title",
    "resolve_kind_job",
]

KIND_CHUNK_SIZE = 5
_SEARCH_MAX_PAGES = 20
_DOCUMENT_MAX_REQUESTS = 3
_KST = timedelta(hours=9)

_BRACKET_TAG = re.compile(r"\[[^\[\]]*\]")
_PAREN_QUALIFIER = re.compile(r"\([^()]*\)")
_WHITESPACE = re.compile(r"\s+")

_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED})


def _kst_today(now: datetime) -> date:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return (moment.astimezone(UTC) + _KST).date()


def _window_key(keyword: str, start: date, end: date) -> str:
    return f"{keyword}:{start.isoformat()}..{end.isoformat()}"


def normalize_kind_title(title: str) -> str:
    """Title with ``[…]`` tags, ``(…)`` qualifiers and all whitespace removed.

    ``관리종목 지정`` and ``관리종목지정해제(회생절차 종결결정)`` normalize to ``관리종목지정`` and
    ``관리종목지정해제``, so a qualifier never hides an exchange form from body collection.
    """
    text = _BRACKET_TAG.sub("", title or "")
    text = _PAREN_QUALIFIER.sub("", text)
    return _WHITESPACE.sub("", text)


def _read_search_document(entry_content_hash: str, payload_path: Path) -> dict[str, object]:
    try:
        raw = Path(payload_path).read_bytes()
    except OSError as exc:
        raise PITDataError(f"retained KIND search payload is missing for {entry_content_hash[:12]}") from exc
    if hashlib.sha256(raw).hexdigest() != entry_content_hash:
        raise PITDataError(f"retained KIND search payload hash mismatch for {entry_content_hash[:12]}")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"retained KIND search payload is unreadable for {entry_content_hash[:12]}") from exc
    if not isinstance(document, dict):
        raise PITDataError(f"retained KIND search payload is unreadable for {entry_content_hash[:12]}")
    return document


def _search_pages(document: dict[str, object], *, label: str) -> tuple[str, ...]:
    pages = document.get("pages")
    if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
        raise PITDataError(f"retained KIND search payload carries no pages for {label}")
    return tuple(pages)


def _search_receipt_complete(entry: ReceiptIndexEntry) -> bool:
    """True only when the stored window's parsed rows match its stored declared total."""
    try:
        document = _read_search_document(entry.content_hash, Path(entry.payload_path))
        pages = _search_pages(document, label=entry.natural_key)
        reported_total = document.get("reported_total")
        if isinstance(reported_total, bool) or not isinstance(reported_total, int):
            return False
        rows = sum(len(parse_kind_search_page(page).rows) for page in pages)
    except PITDataError:
        return False
    return rows == reported_total


def _search_max_requests(entry: ReceiptIndexEntry) -> int:
    try:
        document = _read_search_document(entry.content_hash, Path(entry.payload_path))
        reported_total = document.get("reported_total")
        if isinstance(reported_total, bool) or not isinstance(reported_total, int) or reported_total < 0:
            return _SEARCH_MAX_PAGES
        return min(_SEARCH_MAX_PAGES, max(1, -(-reported_total // 100)))
    except PITDataError:
        return _SEARCH_MAX_PAGES


def _search_payload(
    *, keyword: str, start: date, end: date, reported_total: int, pages: Sequence[str], retrieved_at: datetime
) -> ScopedRawPayload:
    body = json.dumps(
        {
            "keyword": keyword,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "reported_total": reported_total,
            "pages": list(pages),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return ScopedRawPayload(
        kind=EvidenceKind.DISCLOSURES,
        source=KIND_NOTICE_SEARCH_SOURCE,
        natural_key=_window_key(keyword, start, end),
        as_of=end,
        fiscal_period=None,
        status=EvidenceStatus.EMPTY if reported_total == 0 else EvidenceStatus.SUCCESS,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"kind:search:{keyword}:{start.isoformat()}..{end.isoformat()}",
    )


class KindNoticeSearchJob:
    """Quarterly KIND title-keyword windows from the scope's evidence start.

    A window receipt stores every raw result page so Silver re-parses from Bronze, plus the
    total KIND declared, so an incomplete completed window can never be accepted.
    """

    name = "kind_notice_search"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        today = _kst_today(ctx.now())
        units: list[JobUnit] = []
        for keyword in ctx.provider.kind.search_keywords:
            for w_start, w_end in disclosure_windows(ctx.runtime.scope.evidence_start, today):
                key = _window_key(keyword, w_start, w_end)
                if w_end >= today:
                    units.append(
                        JobUnit(
                            source=KIND_NOTICE_SEARCH_SOURCE,
                            natural_key=key,
                            payload={
                                "keyword": keyword,
                                "window_start": w_start.isoformat(),
                                "window_end": w_end.isoformat(),
                            },
                            max_requests=_SEARCH_MAX_PAGES,
                        )
                    )
                    continue
                answered = ctx.catalog.latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={key})
                entry = answered.get(key)
                if entry is not None and entry.status in _ANSWERED and _search_receipt_complete(entry):
                    continue
                max_requests = _search_max_requests(entry) if entry is not None else _SEARCH_MAX_PAGES
                units.append(
                    JobUnit(
                        source=KIND_NOTICE_SEARCH_SOURCE,
                        natural_key=key,
                        payload={
                            "keyword": keyword,
                            "window_start": w_start.isoformat(),
                            "window_end": w_end.isoformat(),
                        },
                        max_requests=max_requests,
                    )
                )
        units.sort(key=lambda unit: unit.natural_key)
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        today = _kst_today(ctx.now())
        out: list[ScopedRawPayload] = []
        for unit in units:
            keyword = unit.payload["keyword"]
            w_start = date.fromisoformat(unit.payload["window_start"])
            w_end = date.fromisoformat(unit.payload["window_end"])
            first_html = ctx.collector.search_page(keyword, w_start, w_end, page_index=1)
            first = parse_kind_search_page(first_html)
            reported_total = first.reported_total
            need = max(1, -(-reported_total // 100))
            if need > _SEARCH_MAX_PAGES:
                raise PITDataError(
                    f"KIND search window {unit.natural_key} needs {need} pages"
                    f" over the bound {_SEARCH_MAX_PAGES}; refusing to truncate"
                )
            pages = [first_html]
            pages.extend(
                ctx.collector.search_page(keyword, w_start, w_end, page_index=page_index)
                for page_index in range(2, need + 1)
            )
            parsed = [parse_kind_search_page(page) for page in pages]
            if w_end < today:
                expected = parsed[0].reported_total
                if any(page.reported_total != expected for page in parsed):
                    raise PITDataError(
                        f"KIND search window {unit.natural_key} declares contradictory totals"
                    )
                rows = [row for page in parsed for row in page.rows]
                if len(rows) != reported_total:
                    raise PITDataError(
                        f"KIND search window {unit.natural_key} is incomplete:"
                        f" rows={len(rows)} reported_total={reported_total}"
                    )
                seen: set[str] = set()
                for row in rows:
                    if row.acptno in seen:
                        raise PITDataError(
                            f"KIND search window {unit.natural_key} carries duplicate {row.acptno}"
                        )
                    seen.add(row.acptno)
            out.append(
                _search_payload(
                    keyword=keyword,
                    start=w_start,
                    end=w_end,
                    reported_total=reported_total,
                    pages=pages,
                    retrieved_at=ctx.now(),
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def kind_document_targets(catalog: ReceiptCatalog, policy: KindPolicy) -> tuple[KindNoticeRow, ...]:
    """Search rows that need a body: exchange submitter and a normalized title in ``document_titles``.

    Raises:
        PITDataError: a stored search page is unreadable, fails its hash, or does not parse.
    """
    wanted = {normalize_kind_title(title) for title in policy.document_titles}
    rows: list[KindNoticeRow] = []
    for entry in catalog.entries(source=KIND_NOTICE_SEARCH_SOURCE):
        document = _read_search_document(entry.content_hash, Path(entry.payload_path))
        for page in _search_pages(document, label=entry.natural_key):
            rows.extend(parse_kind_search_page(page).rows)
    targets = [
        row
        for row in rows
        if row.submitter.endswith("시장본부") and normalize_kind_title(row.title) in wanted
    ]
    seen: dict[str, KindNoticeRow] = {}
    for row in targets:
        seen.setdefault(row.acptno, row)
    return tuple(seen.values())


class KindNoticeDocumentJob:
    """Bodies of exchange-filed notices whose normalized title is in ``document_titles``."""

    name = "kind_notice_documents"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        targets = kind_document_targets(ctx.catalog, ctx.provider.kind)
        wanted = [row.acptno for row in targets]
        answered = ctx.catalog.latest(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_keys=set(wanted))
        units = [
            JobUnit(
                source=KIND_NOTICE_DOCUMENT_SOURCE,
                natural_key=acptno,
                payload={"acptno": acptno},
                max_requests=_DOCUMENT_MAX_REQUESTS,
            )
            for acptno in wanted
            if acptno not in answered or answered[acptno].status not in _ANSWERED
        ]
        units.sort(key=lambda unit: unit.natural_key)
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        targets = {row.acptno: row for row in kind_document_targets(ctx.catalog, ctx.provider.kind)}
        out: list[ScopedRawPayload] = []
        for unit in units:
            acptno = unit.payload["acptno"]
            row = targets.get(acptno)
            if row is None:
                raise PITDataError(f"KIND document target {acptno} has no search row")
            document = ctx.collector.fetch_document(acptno)
            body = json.dumps(
                {
                    "acptno": document.acptno,
                    "doc_no": document.doc_no,
                    "template": document.template,
                    "html": document.html,
                    "disclosed_at": row.disclosed_at.isoformat(),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            out.append(
                ScopedRawPayload(
                    kind=EvidenceKind.DISCLOSURES,
                    source=KIND_NOTICE_DOCUMENT_SOURCE,
                    natural_key=acptno,
                    as_of=row.disclosed_at.date(),
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS,
                    payload=body,
                    retrieved_at=ctx.now(),
                    source_label=f"kind:document:{acptno}",
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


KIND_JOBS: Mapping[str, JobSpec] = {
    KindNoticeSearchJob.name: KindNoticeSearchJob(),
    KindNoticeDocumentJob.name: KindNoticeDocumentJob(),
}


def resolve_kind_job(name: str) -> JobSpec:
    """Return the job spec registered under ``name``.

    Raises:
        PITDataError: no KIND job is registered under ``name``.
    """
    try:
        return KIND_JOBS[str(name)]
    except KeyError:
        raise PITDataError(f"unknown KIND job {name!r}: expected one of {sorted(KIND_JOBS)}") from None


def build_kind_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: KindClient | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    """Build the execution context for one KIND job without issuing any provider request."""
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return JobContext(
        runtime=runtime,
        catalog=catalog,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        provider=provider,
        quota_store=quota_store,
        runner=provider.runner("kind"),
        key_env="KIND",
        collector=collector,
        now=now or (lambda: datetime.now(UTC)),
        sleep=sleep or time.sleep,
    )
