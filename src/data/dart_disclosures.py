"""Catalog-only readers for retained DART disclosure rows and filing identities."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.core.pit import PITDataError
from src.data.evidence_sources import (
    DART_CORP_DISCLOSURES_SOURCE,
    DART_DISCLOSURE_WINDOWS_SOURCE,
)
from src.data.receipt_catalog import ReceiptCatalog

__all__ = [
    "DisclosureRecord",
    "iter_disclosure_records",
    "per_corp_coverage",
    "periodic_filing_identities",
]

_PERIOD = re.compile(r"\((\d{4})\.(\d{2})\)")
_REPORT_CODE_BY_KIND = {
    "사업보고서": "11011",
    "반기보고서": "11012",
    "분기보고서": None,
}
REPRT_QUARTER = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Q4"}

_DISCLOSURE_SOURCES = (DART_DISCLOSURE_WINDOWS_SOURCE, DART_CORP_DISCLOSURES_SOURCE)


@dataclass(frozen=True, slots=True)
class DisclosureRecord:
    """One OpenDART list row as retained in Bronze."""

    corp_code: str
    rcept_no: str
    rcept_dt: date
    report_nm: str


def _parse_receipt_day(value: object) -> date | None:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:]))
        except ValueError:
            return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _read_records_payload(entry_content_hash: str, payload_path: Path) -> list[object]:
    try:
        raw = Path(payload_path).read_bytes()
    except OSError as exc:
        raise PITDataError(f"retained disclosure payload is missing for {entry_content_hash[:12]}") from exc
    if hashlib.sha256(raw).hexdigest() != entry_content_hash:
        raise PITDataError(f"retained disclosure payload hash mismatch for {entry_content_hash[:12]}")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"retained disclosure payload is unreadable for {entry_content_hash[:12]}") from exc
    records = document.get("records") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise PITDataError(f"retained disclosure payload carries no records for {entry_content_hash[:12]}")
    return records


_TITLE_MARKER_PREFIX = re.compile(r"^(?:\[[^\]]*\])+")


def _normalized_title(title: str) -> str:
    return " ".join(_TITLE_MARKER_PREFIX.sub("", title).split())


def _same_filing_relabelled(left: DisclosureRecord, right: DisclosureRecord) -> bool:
    """True when two rows of one receipt carry the same title up to DART annotations.

    DART adds leading markers such as ``[첨부추가]`` or ``[기재정정]`` and sometimes a trailing
    note to a title after the original filing; company and receipt date never change. After marker
    and whitespace normalization one title must be a prefix of the other. A different report kind is
    a real conflict.
    """
    if (left.corp_code, left.rcept_no, left.rcept_dt) != (right.corp_code, right.rcept_no, right.rcept_dt):
        return False
    first, second = _normalized_title(left.report_nm), _normalized_title(right.report_nm)
    return first.startswith(second) or second.startswith(first)


def iter_disclosure_records(catalog: ReceiptCatalog) -> Iterator[DisclosureRecord]:
    """Stream every retained disclosure row from the catalog's latest receipts.

    Reads the ``dart_disclosure_windows`` and ``dart_corp_disclosures``
    sources only, one payload at a time. A row missing ``corp_code``,
    ``rcept_no`` or a valid ``rcept_dt`` is skipped, as today.

    Rows of one receipt whose titles differ only by leading DART markers collapse to the longer
    (annotated) title.

    Raises:
        PITDataError: a referenced payload is missing, unreadable or fails its hash, or one receipt
            carries rows that differ beyond title markers.
    """
    seen: dict[str, DisclosureRecord] = {}
    for source in _DISCLOSURE_SOURCES:
        for entry in catalog.entries(source=source):
            records = _read_records_payload(entry.content_hash, Path(entry.payload_path))
            for record in records:
                if not isinstance(record, dict):
                    continue
                corp_code = str(record.get("corp_code") or "").strip()
                rcept_no = str(record.get("rcept_no") or "").strip()
                receipt_day = _parse_receipt_day(record.get("rcept_dt"))
                if not corp_code or not rcept_no or receipt_day is None:
                    continue
                candidate = DisclosureRecord(
                    corp_code=corp_code,
                    rcept_no=rcept_no,
                    rcept_dt=receipt_day,
                    report_nm=str(record.get("report_nm") or ""),
                )
                previous = seen.get(rcept_no)
                if previous is None:
                    seen[rcept_no] = candidate
                elif previous != candidate:
                    if not _same_filing_relabelled(previous, candidate):
                        raise PITDataError(f"conflicting disclosure rows for receipt {rcept_no!r}")
                    # DART가 나중에 제목에 표지·안내를 덧붙이므로 더 긴 제목을 취한다.
                    if len(candidate.report_nm) > len(previous.report_nm):
                        seen[rcept_no] = candidate
    yield from seen.values()


def per_corp_coverage(catalog: ReceiptCatalog) -> dict[str, list[tuple[date, date]]]:
    """Answered per-corp ranges, from the ``dart_corp_disclosures`` natural keys only (no payload read)."""
    coverage: dict[str, list[tuple[date, date]]] = {}
    for entry in catalog.entries(source=DART_CORP_DISCLOSURES_SOURCE):
        key = entry.natural_key
        try:
            corp, window = key.split(":", 1)
            start_raw, end_raw = window.split("..", 1)
            start = date.fromisoformat(start_raw)
            end = date.fromisoformat(end_raw)
        except ValueError:
            continue
        if not corp.strip() or end < start:
            continue
        coverage.setdefault(corp, []).append((start, end))
    return coverage


def periodic_filing_identities(
    records: Iterable[DisclosureRecord],
    *,
    start: date,
    end: date,
    ticker_by_corp_code: Mapping[str, str] | None,
    required_periods: frozenset[str] | None,
    corp_codes: frozenset[str] | None,
) -> tuple[dict[str, str], ...]:
    """Periodic-report identities (moved unchanged from ``DartXbrlCollector.filing_identities_from_bronze``)."""
    target_codes = frozenset(corp_codes) if corp_codes is not None else None
    identities: list[dict[str, str]] = []
    for record in records:
        corp_code = record.corp_code.strip()
        filing_id = record.rcept_no.strip()
        if target_codes is not None and corp_code not in target_codes:
            continue
        if not start <= record.rcept_dt <= end:
            continue
        name = record.report_nm or ""
        matched = _PERIOD.search(name)
        if not matched or not corp_code or not filing_id:
            continue
        year, month = matched.groups()
        if "사업보고서" in name:
            report_code = _REPORT_CODE_BY_KIND["사업보고서"]
        elif "반기보고서" in name:
            report_code = _REPORT_CODE_BY_KIND["반기보고서"]
        elif "분기보고서" in name and month == "03":
            report_code = "11013"
        elif "분기보고서" in name and month == "09":
            report_code = "11014"
        else:
            continue
        published_at = record.rcept_dt.isoformat()
        quarter = REPRT_QUARTER.get(str(report_code), "")
        fiscal_period = f"{year}{quarter}" if quarter else ""
        if required_periods is not None and fiscal_period not in required_periods:
            continue
        entry: dict[str, str] = {
            "corp_code": corp_code,
            "filing_id": filing_id,
            "rcept_no": filing_id,
            "biz_year": year,
            "reprt_code": str(report_code),
            "fs_div": "CFS",
            "published_at": published_at,
            "correction_of": "",
        }
        if fiscal_period:
            entry["fiscal_period"] = fiscal_period
        if ticker_by_corp_code is not None:
            ticker = ticker_by_corp_code.get(corp_code)
            if ticker is None:
                continue
            entry["ticker"] = str(ticker)
        identities.append(entry)
    unique: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for identity in identities:
        unique[tuple(sorted(identity.items()))] = identity
    return tuple(unique.values())
