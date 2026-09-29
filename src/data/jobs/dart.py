"""DART collection jobs: market-wide disclosures, periodic facts, dividend decisions.

The three jobs share one budgeted runner. Disclosure windows feed the fact and
dividend jobs: window receipts and the retained per-corp pages are both valid
inputs, so collection resumes from either shape.
"""
from __future__ import annotations

import base64
import calendar as _calendar
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.core.pit import EvidenceKind, PITDataError
from src.data.collection import dart_fact_scoped_payload
from src.data.dart_disclosures import (
    iter_disclosure_records,
    per_corp_coverage,
    periodic_filing_identities,
)
from src.data.dart_documents import DartDocumentStore
from src.data.evidence_sources import (
    DART_DISCLOSURE_WINDOWS_SOURCE,
    DIVIDEND_DECISION_SOURCE,
    EARNINGS_RELEASE_SOURCE,
)
from src.data.jobs.runner import JobContext, JobSpec, JobUnit
from src.data.jobs.universe import corp_code_bridge, eligible_tickers
from src.data.receipt_catalog import EvidenceStatus, ReceiptIndexEntry
from src.data.scoped_ingestion import FACT_SOURCE, ScopedRawPayload, dart_fact_natural_key
from src.integrations.dart.dividend_decision import is_dividend_decision_title
from src.integrations.dart.earnings_release import classify_earnings_release_title
from src.integrations.dart.xbrl import is_transport_failure_page
from src.integrations.errors import ProviderQuotaExhaustedError, ProviderRetryableError

__all__ = [
    "DART_JOBS",
    "DartDisclosuresJob",
    "DartFactsJob",
    "DisclosureCoverageError",
    "DividendDecisionsJob",
    "EarningsReleasesJob",
    "disclosure_windows",
    "last_completed_kst_day",
    "resolve_dart_job",
]

_LOG = logging.getLogger(__name__)

DISCLOSURE_WINDOW_SOURCE = DART_DISCLOSURE_WINDOWS_SOURCE
_FACT_MAX_REQUESTS = 3
_DIVIDEND_MAX_REQUESTS = 1
_DISCLOSURE_WINDOW_MAX_REQUESTS = 10
_WINDOW_MONTHS = 3
_KST = timedelta(hours=9)

_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED})
_COMPLETED_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EXTRACTION_FAILED})


class DisclosureCoverageError(PITDataError):
    """A required fiscal period whose filing deadline has passed has no discoverable periodic filing."""

    def __init__(self, message: str, *, periods: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.periods = tuple(periods)


def _kst_today(now: datetime) -> date:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return (moment.astimezone(UTC) + _KST).date()


def last_completed_kst_day(now: datetime) -> date:
    """Last fully completed KST calendar day (windows never cover today)."""
    return _kst_today(now) - timedelta(days=1)


def disclosure_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Three-month calendar windows covering ``[start, end]`` in order."""
    if end < start:
        return []
    out: list[tuple[date, date]] = []
    year, month = start.year, start.month
    while True:
        first = date(year, month, 1)
        last_month = month + _WINDOW_MONTHS - 1
        last_year = year + (last_month - 1) // 12
        last_month = (last_month - 1) % 12 + 1
        w_end = date(last_year, last_month, _calendar.monthrange(last_year, last_month)[1])
        if w_end > end:
            w_end = end
        w_start = max(first, start)
        if w_start <= w_end:
            out.append((w_start, w_end))
        if w_end >= end:
            break
        month = last_month + 1
        year = last_year
        if month > 12:
            month = 1
            year += 1
    return out


def _window_key(detail_type: str, start: date, end: date) -> str:
    return f"{detail_type}:{start.isoformat()}..{end.isoformat()}"


def _window_payload(
    *,
    detail_type: str,
    start: date,
    end: date,
    records: list[dict[str, str]],
    retrieved_at: datetime,
    reported_total: int,
    raw_rows: int,
) -> ScopedRawPayload:
    body = json.dumps(
        {
            "detail_type": detail_type,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "records": records,
            "reported_total": reported_total,
            "raw_rows": raw_rows,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return ScopedRawPayload(
        kind=EvidenceKind.DISCLOSURES,
        source=DISCLOSURE_WINDOW_SOURCE,
        natural_key=_window_key(detail_type, start, end),
        as_of=end,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS if records else EvidenceStatus.EMPTY,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"opendart:list:{detail_type}:{start.isoformat()}..{end.isoformat()}",
    )


def _window_receipt_complete(entry: ReceiptIndexEntry) -> bool:
    """True only when the stored window payload is complete against DART's declared total."""
    try:
        raw = Path(entry.payload_path).read_bytes()
    except OSError:
        return False
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        return False
    try:
        document = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(document, dict):
        return False
    reported_total = document.get("reported_total")
    raw_rows = document.get("raw_rows")
    if isinstance(reported_total, bool) or isinstance(raw_rows, bool):
        return False
    if not isinstance(reported_total, int) or not isinstance(raw_rows, int):
        return False
    return raw_rows == reported_total


def _window_covered(
    coverage: Mapping[str, Sequence[tuple[date, date]]],
    corps: frozenset[str],
    start: date,
    end: date,
) -> bool:
    return all(
        any(page_start <= start and page_end >= end for page_start, page_end in coverage.get(corp, ()))
        for corp in corps
    )


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


def _prev_quarter(period: str) -> str:
    total = int(period[:4]) * 4 + (int(period[5]) - 1) - 1
    return f"{total // 4}Q{(total % 4) + 1}"


def _publication_cutoff(period: str) -> date:
    year = int(period[:4])
    quarter = int(period[5])
    if quarter == 1:
        return date(year, 5, 15)
    if quarter == 2:
        return date(year, 8, 15)
    if quarter == 3:
        return date(year, 11, 15)
    return date(year + 1, 3, 30)


def _latest_available_quarter(today: date) -> str:
    quarter = (today.month - 1) // 3 + 1
    current = f"{today.year}Q{quarter}"
    for _ in range(12):
        if _publication_cutoff(current) < today:
            return current
        current = _prev_quarter(current)
    return current  # pragma: no cover - publication cutoffs strictly decrease walking back


def required_periods(*, fiscal_start: str, today: date) -> tuple[str, ...]:
    """Every fiscal quarter from the scope floor through the latest published one."""
    latest = _latest_available_quarter(today)
    if _fiscal_key(fiscal_start) > _fiscal_key(latest):
        return ()
    out = [fiscal_start]
    while _fiscal_key(out[-1]) < _fiscal_key(latest):
        year = int(out[-1][:4])
        quarter = int(out[-1][5])
        if quarter == 4:
            out.append(f"{year + 1}Q1")
        else:
            out.append(f"{year}Q{quarter + 1}")
    return tuple(out)


def _unanswered(
    ctx: JobContext, *, source: str, keys: Sequence[str]
) -> set[str]:
    answered = ctx.catalog.latest(source=source, natural_keys=set(keys))
    return {key for key in keys if key not in answered or answered[key].status not in _ANSWERED}


class DartDisclosuresJob:
    """Market-wide ``list.json`` windows for the types the other jobs consume."""

    name = "dart_disclosures"
    # 창 하나가 수십~수백 페이지라 청크를 작게 잡아 완료된 창을 바로 저장한다(중단 시 손실 최소화).
    max_chunk_units = 4

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        bridge = corp_code_bridge(ctx)
        tickers = eligible_tickers(ctx)
        eligible = frozenset(code for code, ticker in bridge.items() if ticker in tickers)
        coverage = per_corp_coverage(ctx.catalog)
        today = _kst_today(ctx.now())
        units: list[JobUnit] = []
        replanned = 0
        for disclosure_filter in ctx.provider.dart.disclosure_filters:
            detail_type = disclosure_filter.code
            for w_start, w_end in disclosure_windows(ctx.runtime.scope.evidence_start, today):
                if w_end < today and _window_covered(coverage, eligible, w_start, w_end):
                    continue
                key = _window_key(detail_type, w_start, w_end)
                if w_end >= today:
                    units.append(
                        JobUnit(
                            source=DISCLOSURE_WINDOW_SOURCE,
                            natural_key=key,
                            payload={
                                "detail_type": detail_type,
                                "window_start": w_start.isoformat(),
                                "window_end": w_end.isoformat(),
                            },
                            max_requests=_DISCLOSURE_WINDOW_MAX_REQUESTS,
                        )
                    )
                    continue
                answered = ctx.catalog.latest(source=DISCLOSURE_WINDOW_SOURCE, natural_keys={key})
                entry = answered.get(key)
                if entry is not None and entry.status in _COMPLETED_ANSWERED:
                    if _window_receipt_complete(entry):
                        continue
                    replanned += 1
                units.append(
                    JobUnit(
                        source=DISCLOSURE_WINDOW_SOURCE,
                        natural_key=key,
                        payload={
                            "detail_type": detail_type,
                            "window_start": w_start.isoformat(),
                            "window_end": w_end.isoformat(),
                        },
                        max_requests=_DISCLOSURE_WINDOW_MAX_REQUESTS,
                    )
                )
        units.sort(key=lambda unit: unit.natural_key)
        if replanned:
            _LOG.info("[DATA] job=dart_disclosures action=replan_incomplete windows=%d", replanned)
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        from src.config.providers import disclosure_filter_for_code

        out: list[ScopedRawPayload] = []
        today = _kst_today(ctx.now())
        for unit in units:
            w_start = date.fromisoformat(unit.payload["window_start"])
            w_end = date.fromisoformat(unit.payload["window_end"])
            disclosure_filter = disclosure_filter_for_code(unit.payload["detail_type"])
            listing = ctx.collector.list_disclosure_window(w_start, w_end, disclosure_filter=disclosure_filter)
            rows = list(listing.records)
            if w_end < today and listing.reported_total == 0:
                raise PITDataError(
                    f"DART list returned no filings for completed window {unit.natural_key}"
                )
            if w_end < today and listing.raw_rows != listing.reported_total:
                raise PITDataError(
                    f"DART list window {unit.natural_key} is incomplete:"
                    f" raw_rows={listing.raw_rows} reported_total={listing.reported_total}"
                )
            out.append(
                _window_payload(
                    detail_type=unit.payload["detail_type"],
                    start=w_start,
                    end=w_end,
                    records=[{str(k): str(v) for k, v in item.items()} for item in rows],
                    retrieved_at=ctx.now(),
                    reported_total=listing.reported_total,
                    raw_rows=listing.raw_rows,
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _identity_fiscal_period(item: Mapping[str, str]) -> str | None:
    raw = str(item.get("fiscal_period") or "").strip()
    if raw:
        return raw
    biz_year = str(item.get("biz_year") or "").strip()
    reprt_code = str(item.get("reprt_code") or "").strip()
    quarter = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Q4"}.get(reprt_code)
    if not biz_year or quarter is None:
        return None
    return f"{biz_year}{quarter}"


def _check_disclosure_coverage(
    identities: Sequence[Mapping[str, str]], *, periods: frozenset[str], today: date
) -> None:
    """Fail when a published period has no discoverable periodic filing."""
    have: set[str] = set()
    for item in identities:
        period = _identity_fiscal_period(item)
        if period is not None:
            have.add(period)
    missing = sorted(
        period for period in periods if _publication_cutoff(period) < today and period not in have
    )
    if missing:
        raise DisclosureCoverageError(
            f"no discoverable periodic filing for {', '.join(missing)}",
            periods=tuple(missing),
        )


class DartFactsJob:
    """Periodic-report facts for eligible corp codes, latest filing wins."""

    name = "dart_facts"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        bridge = corp_code_bridge(ctx)
        tickers = eligible_tickers(ctx)
        eligible = frozenset(code for code, ticker in bridge.items() if ticker in tickers)
        scope = ctx.runtime.scope
        end = last_completed_kst_day(ctx.now())
        today = _kst_today(ctx.now())
        periods = frozenset(required_periods(fiscal_start=scope.features.fundamental_fiscal_start, today=end))
        records = iter_disclosure_records(ctx.catalog)
        identities = periodic_filing_identities(
            records,
            start=scope.evidence_start,
            end=end,
            ticker_by_corp_code={code: bridge[code] for code in eligible},
            required_periods=periods,
            corp_codes=eligible,
        )
        _check_disclosure_coverage(identities, periods=periods, today=today)
        latest: dict[str, dict[str, str]] = {}
        for item in identities:
            key = dart_fact_natural_key(corp_code=item["corp_code"], biz_year=item["biz_year"], reprt_code=item["reprt_code"])
            current = latest.get(key)
            if current is None or (item["published_at"], item["filing_id"]) > (
                current["published_at"],
                current["filing_id"],
            ):
                latest[key] = dict(item)
        pending_keys = _unanswered(ctx, source=FACT_SOURCE, keys=sorted(latest))
        units = [
            JobUnit(
                source=FACT_SOURCE,
                natural_key=key,
                payload=dict(latest[key]),
                max_requests=_FACT_MAX_REQUESTS,
            )
            for key in sorted(pending_keys, key=lambda k: (latest[k]["published_at"], latest[k]["filing_id"]))
        ]
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        documents = DartDocumentStore(ctx.runtime.workspace.bronze_root, catalog=ctx.catalog)
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            identity = dict(unit.payload)
            pages = list(ctx.collector.fetch_financial_fact_sources((identity,)))
            for page in pages:
                if page.get("source_kind") == "blocked":
                    raise ProviderQuotaExhaustedError(f"DART quota exhausted for {unit.natural_key}")
                if is_transport_failure_page(page):
                    raise ProviderRetryableError(f"DART transport failed for {unit.natural_key}")
                serializable = {key: value for key, value in dict(page).items() if key != "raw_archive"}
                archive = page.get("raw_archive")
                if isinstance(archive, (bytes, bytearray)) and len(archive) > 0:
                    rcept_no = str(
                        identity.get("rcept_no") or identity.get("filing_id") or ""
                    ).strip()
                    receipt = documents.store_archive(bytes(archive), rcept_no=rcept_no, retrieved_at=retrieved_at)
                    serializable["document_receipt"] = str(receipt.metadata_path)
                    if not serializable.get("raw_document_hash"):
                        serializable["raw_document_hash"] = receipt.content_hash
                out.append(dart_fact_scoped_payload(page=serializable, retrieved_at=retrieved_at))
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _archive_envelope(*, rcept_no: str, corp_code: str, received_on: str, report_nm: str, archive: bytes | None) -> bytes:
    if archive is None:
        return json.dumps(
            {
                "rcept_no": rcept_no,
                "corp_code": corp_code,
                "received_on": received_on,
                "report_nm": report_nm,
                "document_status": "unavailable",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    return json.dumps(
        {
            "rcept_no": rcept_no,
            "corp_code": corp_code,
            "received_on": received_on,
            "report_nm": report_nm,
            "archive_b64": base64.b64encode(archive).decode("ascii"),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


class DividendDecisionsJob:
    """Decision-titled filings fetched as archives; ``014`` bodies are ``empty``."""

    name = "dividend_decisions"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        bridge = corp_code_bridge(ctx)
        tickers = eligible_tickers(ctx)
        eligible = frozenset(code for code, ticker in bridge.items() if ticker in tickers)
        scope = ctx.runtime.scope
        end = last_completed_kst_day(ctx.now())
        matches: dict[str, dict[str, str]] = {}
        for record in iter_disclosure_records(ctx.catalog):
            if record.corp_code not in eligible:
                continue
            as_of = record.rcept_dt
            if as_of < scope.evidence_start or as_of > end:
                continue
            if not is_dividend_decision_title(record.report_nm):
                continue
            receipt_day = as_of.strftime("%Y%m%d")
            matches[record.rcept_no] = {
                "rcept_no": record.rcept_no,
                "corp_code": record.corp_code,
                "rcept_dt": receipt_day,
                "report_nm": record.report_nm,
            }
        pending_keys = _unanswered(ctx, source=DIVIDEND_DECISION_SOURCE, keys=sorted(matches))
        return [
            JobUnit(
                source=DIVIDEND_DECISION_SOURCE,
                natural_key=key,
                payload=dict(matches[key]),
                max_requests=_DIVIDEND_MAX_REQUESTS,
            )
            for key in sorted(pending_keys, key=lambda k: (matches[k]["rcept_dt"], matches[k]["rcept_no"]))
        ]

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            archive = bytes(ctx.collector.fetch_document_archive(unit.payload["rcept_no"]))
            is_zip = archive[:2] == b"PK"
            receipt_day = unit.payload["rcept_dt"]
            as_of = date(int(receipt_day[:4]), int(receipt_day[4:6]), int(receipt_day[6:8]))
            out.append(
                ScopedRawPayload(
                    kind=EvidenceKind.CORPORATE_ACTIONS,
                    source=DIVIDEND_DECISION_SOURCE,
                    natural_key=unit.natural_key,
                    as_of=as_of,
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS if is_zip else EvidenceStatus.EMPTY,
                    payload=_archive_envelope(
                        rcept_no=unit.payload["rcept_no"],
                        corp_code=unit.payload["corp_code"],
                        received_on=as_of.isoformat(),
                        report_nm=unit.payload["report_nm"],
                        archive=archive if is_zip else None,
                    ),
                    retrieved_at=retrieved_at,
                    source_label=f"{DIVIDEND_DECISION_SOURCE}:{unit.natural_key}",
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


class EarningsReleasesJob:
    """Early-earnings release filings fetched as archives; ``014`` bodies are ``empty``."""

    name = "earnings_releases"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        bridge = corp_code_bridge(ctx)
        tickers = eligible_tickers(ctx)
        eligible = frozenset(code for code, ticker in bridge.items() if ticker in tickers)
        scope = ctx.runtime.scope
        end = last_completed_kst_day(ctx.now())
        matches: dict[str, dict[str, str]] = {}
        for record in iter_disclosure_records(ctx.catalog):
            if record.corp_code not in eligible:
                continue
            as_of = record.rcept_dt
            if as_of < scope.evidence_start or as_of > end:
                continue
            if classify_earnings_release_title(record.report_nm) is None:
                continue
            receipt_day = as_of.strftime("%Y%m%d")
            matches[record.rcept_no] = {
                "rcept_no": record.rcept_no,
                "corp_code": record.corp_code,
                "rcept_dt": receipt_day,
                "report_nm": record.report_nm,
            }
        pending_keys = _unanswered(ctx, source=EARNINGS_RELEASE_SOURCE, keys=sorted(matches))
        return [
            JobUnit(
                source=EARNINGS_RELEASE_SOURCE,
                natural_key=key,
                payload=dict(matches[key]),
                max_requests=_DIVIDEND_MAX_REQUESTS,
            )
            for key in sorted(pending_keys, key=lambda k: (matches[k]["rcept_dt"], matches[k]["rcept_no"]))
        ]

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            archive = bytes(ctx.collector.fetch_document_archive(unit.payload["rcept_no"]))
            is_zip = archive[:2] == b"PK"
            receipt_day = unit.payload["rcept_dt"]
            as_of = date(int(receipt_day[:4]), int(receipt_day[4:6]), int(receipt_day[6:8]))
            out.append(
                ScopedRawPayload(
                    kind=EvidenceKind.DISCLOSURES,
                    source=EARNINGS_RELEASE_SOURCE,
                    natural_key=unit.natural_key,
                    as_of=as_of,
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS if is_zip else EvidenceStatus.EMPTY,
                    payload=_archive_envelope(
                        rcept_no=unit.payload["rcept_no"],
                        corp_code=unit.payload["corp_code"],
                        received_on=as_of.isoformat(),
                        report_nm=unit.payload["report_nm"],
                        archive=archive if is_zip else None,
                    ),
                    retrieved_at=retrieved_at,
                    source_label=f"{EARNINGS_RELEASE_SOURCE}:{unit.natural_key}",
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _dart_corp_codes_job() -> JobSpec:
    from src.data.jobs.corp_codes import DartCorpCodesJob

    return DartCorpCodesJob()


def _dart_document_reparse_job() -> JobSpec:
    from src.data.jobs.dart_documents import DartDocumentReparseJob

    return DartDocumentReparseJob()


def _dart_document_fetch_job() -> JobSpec:
    from src.data.jobs.dart_documents import DartDocumentFetchJob

    return DartDocumentFetchJob()


def _dart_benchmark_document_fetch_job() -> JobSpec:
    from src.data.jobs.dart_documents import DartBenchmarkDocumentFetchJob

    return DartBenchmarkDocumentFetchJob()


DART_JOBS: Mapping[str, JobSpec] = {
    DartDisclosuresJob.name: DartDisclosuresJob(),
    DartFactsJob.name: DartFactsJob(),
    DividendDecisionsJob.name: DividendDecisionsJob(),
    EarningsReleasesJob.name: EarningsReleasesJob(),
    "dart_corp_codes": _dart_corp_codes_job(),
    "dart_document_reparse": _dart_document_reparse_job(),
    "dart_document_fetch": _dart_document_fetch_job(),
    "dart_benchmark_documents": _dart_benchmark_document_fetch_job(),
}


def resolve_dart_job(name: str) -> JobSpec:
    """Return the job spec registered under ``name``.

    Raises:
        PITDataError: no DART job is registered under ``name``.
    """
    try:
        return DART_JOBS[str(name)]
    except KeyError:
        raise PITDataError(f"unknown DART job {name!r}: expected one of {sorted(DART_JOBS)}") from None
