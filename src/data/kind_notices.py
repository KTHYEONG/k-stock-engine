"""Point-in-time KIND exchange-notice readers for the Silver market-actions builder.

Consumes the KIND Bronze written by the KIND collection jobs (spec 2): quarterly
title-keyword search windows plus notice-body documents. Tickers, sessions and
market actions are never touched here; this module only reads KIND Bronze.
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from src.core.digest import dataset_digest
from src.core.pit import PITDataError
from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE, KIND_NOTICE_SEARCH_SOURCE
from src.data.jobs.dart import disclosure_windows
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.integrations.krx.kind import KindNoticeRow, parse_kind_search_page

__all__ = [
    "KindAdministrativeForm",
    "KindDelistingForm",
    "iter_kind_notices",
    "kind_source_digest",
    "parse_kind_administrative_form",
    "parse_kind_delisting_form",
    "read_kind_document",
    "require_kind_coverage",
]

_ANSWERED: Final = frozenset(
    {EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED}
)

_TAG_RE: Final = re.compile(r"<[^>]+>")
# 단축코드는 숫자로 시작하는 6자리이며 2024년 이후 신규 종목은 영문을 섞는다(예: A0001A0).
_SHORT_CODE_RE: Final = re.compile(r"\bA([0-9][0-9A-Z]{5})\b")
_LIQUIDATION_LABEL_RE: Final = re.compile(r"정리매매\s*(?:허용\s*)?기간")
_ISIN_AFTER_LABEL_RE: Final = re.compile(r"종목코드\s*[:\uFF1A]?\s*KR7([0-9][0-9A-Z]{5})[0-9A-Z]{3}\b")  # noqa: RUF001
# 유가증권 양식은 `1.종목명 성지건설보통주`, 코스닥 양식은 `1.대상종목 하이에이아이1호스팩 주권 보통주`로
# 라벨과 띄어쓰기가 다르다. 종목 표기는 다음 항목 번호(`2.`)나 다음 필드 라벨 앞까지 읽는다.
_ISSUE_NAME_RE: Final = re.compile(
    r"(?:종목명|대상종목)\s*[:\uFF1A]?\s*(.+?)"  # noqa: RUF001
    r"\s*(?=\b\d{1,2}\.\S|관리종목|지정일|지정사유|해제일|해제사유|$)"
)
_ADMIN_DATE_LABELS: Final = ("지정일", "해제일", "효력발생일", "효력 발생일", "시행일", "발생일")

_YMD_DASH_RE: Final = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_YMD_DOT_RE: Final = re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})$")
_YMD_KOR_RE: Final = re.compile(r"^(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일$")
_DATE_FIND_RES: Final = (
    re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"),
    re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})"),
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
)


@dataclass(frozen=True, slots=True)
class KindDelistingForm:
    """Fields of the exchange ``상장폐지`` form.

    Attributes:
        tickers: Six-character short codes listed in ``단축코드`` (leading ``A`` removed), one per
            delisted share class.
        liquidation_start: First liquidation-trading session, or None when the form states ``-``.
        liquidation_end: Last liquidation-trading session, or None when the form states ``-``.
        delisting_date: Effective delisting date; the instrument trades on no session on or after it.
            None when the exchange has decided but deferred the date (injunction, pending resolution).
    """

    tickers: tuple[str, ...]
    liquidation_start: date | None
    liquidation_end: date | None
    delisting_date: date | None


@dataclass(frozen=True, slots=True)
class KindAdministrativeForm:
    """Fields of an exchange administrative designation or release form.

    Attributes:
        issue_name: ``종목명`` or ``대상종목`` as printed, whitespace collapsed (for example
            ``성지건설보통주`` or ``하이에이아이1호스팩 주권 보통주``).
        is_common: True only when ``issue_name`` ends with ``보통주``; preferred and other classes
            are never mapped onto the common short code.
        effective_on: Stated designation or release date, or None when the form omits it.
    """

    issue_name: str
    is_common: bool
    effective_on: date | None


def _plain_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html or "")
    return _html.unescape(text)


def _parse_kind_date(value: str) -> date:
    """Parse one of the three accepted KIND form date shapes.

    Raises:
        PITDataError: the value matches none of ``YYYY-MM-DD``, ``YYYY.MM.DD`` and
            ``YYYY년 MM월 DD일``.
    """
    text = (value or "").strip()
    for pattern in (_YMD_DASH_RE, _YMD_DOT_RE, _YMD_KOR_RE):
        match = pattern.match(text)
        if match is not None:
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError as exc:
                raise PITDataError(f"KIND form carries an invalid date {value!r}") from exc
    raise PITDataError(f"KIND form carries an unparseable date {value!r}")


def _find_date(text: str) -> date | None:
    for pattern in _DATE_FIND_RES:
        match = pattern.search(text)
        if match is not None:
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError as exc:
                raise PITDataError(f"KIND form carries an invalid date {match.group(0)!r}") from exc
    return None


def _extract_dated_value(text: str, label: str) -> str | None:
    index = text.find(label)
    if index < 0:
        return None
    window = text[index + len(label): index + len(label) + 40]
    stripped = window.strip()
    if stripped[:1] in ("-", "–", "—"):  # noqa: RUF001 - dash variants state "no period"
        return "-"
    found = _find_date(window)
    if found is None:
        return None
    return found.isoformat()


def parse_kind_delisting_form(html: str) -> KindDelistingForm:
    """Parse a ``상장폐지`` body by its field labels, not by template number.

    Raises:
        PITDataError: no ``A``-prefixed short code in ``단축코드`` and no ``KR7`` ISIN after
            ``종목코드``, no ``상장폐지일`` field, an unparseable date, or a
            liquidation start without an end (or the reverse).
    """
    text = _plain_text(html)
    # 단축코드는 '단축코드' 칸부터 '상장폐지 사유' 항목 전까지에서만 읽는다(본문 서술의 영문 토큰 오인 방지).
    code_start = text.find("단축코드")
    code_end = text.find("상장폐지 사유", code_start) if code_start >= 0 else -1
    code_section = text[code_start: code_end if code_end >= 0 else None] if code_start >= 0 else ""
    tickers = tuple(dict.fromkeys(_SHORT_CODE_RE.findall(code_section)))
    if not tickers:
        # ETF·ETN 양식은 단축코드 칸 없이 `종목코드 : KR7200050003`(ISIN)만 싣는다. ISIN 4~9번째 글자가 단축코드다.
        tickers = tuple(dict.fromkeys(_ISIN_AFTER_LABEL_RE.findall(text)))
    if not tickers:
        raise PITDataError("KIND delisting form carries no A-prefixed short code")
    # 가처분 등으로 정리매매가 보류된 결정은 `상장폐지일 -`로 온다. 결정 사실은 유효하므로 종료일 없는 결정으로 본다.
    delisting_raw = _extract_dated_value(text, "상장폐지일")
    if delisting_raw is None:
        raise PITDataError("KIND delisting form carries no 상장폐지일 field")
    delisting_date = None if delisting_raw == "-" else _parse_kind_date(delisting_raw)
    # 양식은 '상장폐지 예고기간'과 '정리매매 허용기간'에 같은 시작일/종료일 라벨을 쓰므로
    # 정리매매 허용기간 라벨부터 상장폐지일 전까지만 읽는다. 라벨이 없으면 정리매매 기간 미기재로 본다.
    liquidation = _LIQUIDATION_LABEL_RE.search(text)
    start_raw: str | None = None
    end_raw: str | None = None
    if liquidation is not None:
        section_end = text.find("상장폐지일", liquidation.end())
        section = text[liquidation.end(): section_end if section_end >= 0 else None]
        start_raw = _extract_dated_value(section, "시작일")
        # 코스닥 양식은 종료 라벨이 `만료일`이다.
        end_raw = _extract_dated_value(section, "종료일")
        if end_raw is None:
            end_raw = _extract_dated_value(section, "만료일")
    start: date | None = None
    end: date | None = None
    if start_raw is not None and start_raw != "-":
        start = _parse_kind_date(start_raw)
    if end_raw is not None and end_raw != "-":
        end = _parse_kind_date(end_raw)
    if (start is None) != (end is None):
        raise PITDataError("KIND delisting form states a liquidation start without an end (or the reverse)")
    return KindDelistingForm(
        tickers=tickers,
        liquidation_start=start,
        liquidation_end=end,
        delisting_date=delisting_date,
    )


def parse_kind_administrative_form(html: str) -> KindAdministrativeForm:
    """Parse a ``관리종목 지정`` or ``관리종목 지정해제`` body by its field labels.

    Raises:
        PITDataError: neither a ``종목명`` nor a ``대상종목`` field is present or it is empty.
    """
    text = re.sub(r"\s+", " ", _plain_text(html))
    match = _ISSUE_NAME_RE.search(text)
    if match is None or not match.group(1).strip():
        raise PITDataError("KIND administrative form carries no 종목명 or 대상종목")
    issue_name = re.sub(r"\s+", " ", match.group(1)).strip()
    effective_on: date | None = None
    for label in _ADMIN_DATE_LABELS:
        raw = _extract_dated_value(text, label)
        if raw is not None and raw != "-":
            effective_on = _parse_kind_date(raw)
            break
    return KindAdministrativeForm(
        issue_name=issue_name,
        is_common=issue_name.endswith("보통주"),
        effective_on=effective_on,
    )


def _read_search_document(content_hash: str, payload_path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = Path(payload_path).read_bytes()
    except OSError as exc:
        raise PITDataError(f"retained KIND search payload is missing for {label}") from exc
    if hashlib.sha256(raw).hexdigest() != content_hash:
        raise PITDataError(f"retained KIND search payload hash mismatch for {label}")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"retained KIND search payload is unreadable for {label}") from exc
    if not isinstance(document, dict):
        raise PITDataError(f"retained KIND search payload is unreadable for {label}")
    return document


def _search_pages(document: dict[str, object], *, label: str) -> tuple[str, ...]:
    pages = document.get("pages")
    if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
        raise PITDataError(f"retained KIND search payload carries no pages for {label}")
    return tuple(pages)


def iter_kind_notices(catalog: ReceiptCatalog) -> Iterator[KindNoticeRow]:
    """Stream every KIND search row once, across keyword windows, in ``(disclosed_at, acptno)`` order.

    Raises:
        PITDataError: a payload is missing, unreadable, fails its hash or does not parse, or one
            ``acptno`` appears with contradictory fields.
    """
    seen: dict[str, KindNoticeRow] = {}
    for entry in catalog.entries(source=KIND_NOTICE_SEARCH_SOURCE):
        document = _read_search_document(entry.content_hash, Path(entry.payload_path), label=entry.natural_key)
        for page in _search_pages(document, label=entry.natural_key):
            try:
                parsed = parse_kind_search_page(page)
            except PITDataError as exc:
                raise PITDataError(
                    f"retained KIND search page does not parse for {entry.natural_key}"
                ) from exc
            for row in parsed.rows:
                previous = seen.get(row.acptno)
                if previous is None:
                    seen[row.acptno] = row
                elif previous != row:
                    raise PITDataError(f"contradictory KIND notice fields for {row.acptno}")
    ordered = sorted(seen.values(), key=lambda row: (row.disclosed_at, row.acptno))
    yield from ordered


def read_kind_document(catalog: ReceiptCatalog, acptno: str) -> str | None:
    """Return the stored body HTML for ``acptno``, or None when no answered receipt exists.

    Raises:
        PITDataError: the payload is unreadable or fails its hash.
    """
    answered = catalog.latest(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_keys={acptno})
    entry = answered.get(acptno)
    if entry is None or entry.status not in _ANSWERED:
        return None
    try:
        raw = Path(entry.payload_path).read_bytes()
    except OSError as exc:
        raise PITDataError(f"retained KIND document payload is unreadable for {acptno}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError(f"retained KIND document payload hash mismatch for {acptno}")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"retained KIND document payload is unreadable for {acptno}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("html"), str):
        raise PITDataError(f"retained KIND document payload is unreadable for {acptno}")
    return str(document["html"])


def kind_source_digest(catalog: ReceiptCatalog) -> str:
    """Stable digest over the latest KIND search and document receipt hashes."""
    hashes = sorted(
        entry.content_hash
        for source in (KIND_NOTICE_SEARCH_SOURCE, KIND_NOTICE_DOCUMENT_SOURCE)
        for entry in catalog.entries(source=source)
    )
    return dataset_digest(hashes)


def _quarter_end(day: date) -> date:
    import calendar as _calendar

    end_month = (day.month - 1) // 3 * 3 + 3
    return date(day.year, end_month, _calendar.monthrange(day.year, end_month)[1])


def _window_complete(entry: ReceiptIndexEntry, *, key: str) -> bool:
    content_hash = entry.content_hash
    payload_path = Path(entry.payload_path)
    try:
        document = _read_search_document(content_hash, payload_path, label=key)
        pages = _search_pages(document, label=key)
        reported_total = document.get("reported_total")
        if isinstance(reported_total, bool) or not isinstance(reported_total, int):
            return False
        rows = sum(len(parse_kind_search_page(page).rows) for page in pages)
    except PITDataError:
        return False
    return rows == reported_total


def require_kind_coverage(
    catalog: ReceiptCatalog, *, keywords: Sequence[str], through: date, start: date
) -> None:
    """Fail unless every completed quarterly window of every keyword in ``[start, through]`` is answered and complete.

    Raises:
        PITDataError: names the first missing or incomplete window key.
    """
    for keyword in keywords:
        for window_start, window_end in disclosure_windows(start, through):
            if _quarter_end(window_start) > through:
                continue
            key = f"{keyword}:{window_start.isoformat()}..{window_end.isoformat()}"
            answered = catalog.latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={key})
            entry = answered.get(key)
            if entry is None or entry.status not in _ANSWERED or not _window_complete(entry, key=key):
                raise PITDataError(f"KIND coverage gap: missing or incomplete window {key}")
