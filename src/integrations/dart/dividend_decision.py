"""Cash/in-kind dividend decision filing parsing (offline, defensive)."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from src.core.pit import PITDataError
from src.integrations.dart.html_tables import decode_member as _decode_member
from src.integrations.dart.html_tables import extract_tables as _extract_tables_shared

__all__ = [
    "DividendDecision", "UndecidedRecordDateError", "is_dividend_decision_title", "parse_dividend_decision"]

_MAX_MEMBERS = 32
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_CORRECTION_MARK = "기재정정"  # noqa: S105 - DART filing marker, not a credential

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{4})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})"),
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"\b(\d{4})(\d{2})(\d{2})\b"),
)

_UNDECIDED_TOKENS = ("미정", "-")


class UndecidedRecordDateError(PITDataError):
    """The filing announces a dividend whose record date is not decided yet ("-" or "미정")."""


@dataclass(frozen=True, slots=True)
class DividendDecision:
    """Dated cash-dividend decision extracted from one filing archive."""

    rcept_no: str
    corp_code: str
    received_on: date
    record_date: date
    pay_date: date | None
    dps_common_krw: int
    is_correction: bool
    agm_date: date | None = None
    dividend_kind: str | None = None
    total_krw: int | None = None
    market_yield_pct: Decimal | None = None


def is_dividend_decision_title(report_nm: str) -> bool:
    """True for cash(/in-kind) dividend decision filings, including corrections."""
    text = str(report_nm or "").strip()
    if not text:
        return False
    text = text.replace(" ", "").replace("\u3000", "")
    while True:
        stripped = re.sub(r"^\[기재정정\]", "", text)
        if stripped == text:
            break
        text = stripped.strip()
    normalized = text.replace("·", "ㆍ").replace("・", "ㆍ").replace(".", "ㆍ").replace("/", "ㆍ")
    return normalized in {"현금ㆍ현물배당결정", "현금배당결정"}


def _extract_tables(text: str) -> list[list[list[str]]]:
    return _extract_tables_shared(text)


def _parse_date_token(text: str) -> date | None:
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            candidate = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        return candidate
    return None


def _cell_has_date(text: str) -> bool:
    return _parse_date_token(text) is not None


def _flatten(tables: list[list[list[str]]]) -> list[list[str]]:
    return [row for table in tables for row in table]


def _is_body_field_row(row: list[str]) -> bool:
    """True for a form field row, False for correction-notice rows.

    A correction filing prepends a reason row ("배당금지급 예정일자 확정") and a
    before/after table whose rows repeat the field labels with three cells; both
    would otherwise be read as the field itself, and the "before" cell is stale.
    The restated form body that follows carries the corrected value.
    """
    non_empty = [cell for cell in row if cell.strip()]
    if len(non_empty) == 1 and not _cell_has_date(non_empty[0]):
        return False  # 라벨만 있는 안내 문장(예: "배당금지급 예정일자 기입")은 필드가 아니다
    return len(non_empty) <= 2 and "정정" not in "".join(row[:1])


def _find_labeled_date(rows: list[list[str]], *, labels: tuple[str, ...]) -> str | None:
    for row in rows:
        if not _is_body_field_row(row):
            continue
        joined = "".join(row)
        if any(label in joined for label in labels):
            for cell in row[1:]:
                if cell.strip():
                    return cell
            # Label and value share one cell (e.g. "배당기준일 2017-12-31").
            return row[0]
    return None


def _parse_amount_cell(cell: str) -> int | None:
    text = cell.strip().replace("원", "").strip()
    if not text or _cell_has_date(cell):
        return None
    m = re.fullmatch(r"-?[\d,]+", text)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _row_label_text(row: list[str]) -> str:
    return "".join(row).replace(" ", "").replace("　", "")


def _row_has_per_share_label(row: list[str]) -> bool:
    """True when the row is labelled as a per-share dividend.

    Accepts the current ``1주당 배당금`` form and the pre-2020 ``주당배당금``
    form (including variants such as ``1주당 현금배당금``); rows labelled as a
    total, a yield or a share count are never per-share labels.
    """
    text = _row_label_text(row)
    if "총액" in text or "시가배당율" in text or "주식수" in text:
        return False
    return "주당" in text and "배당금" in text


def _extract_dps(tables: list[list[list[str]]], full_text: str) -> int | None:
    for table in tables:
        common_col: int | None = None
        for row in table:
            for idx, cell in enumerate(row):
                if "보통주" in cell:
                    common_col = idx
                    break
            if common_col is not None:
                break
        for row in table:
            if not _row_has_per_share_label(row):
                continue
            if "보통주" in "".join(row):
                for cell in row[1:]:
                    amount = _parse_amount_cell(cell)
                    if amount is not None and amount > 0:
                        return amount
                tail = row[0].split("보통주", 1)[-1]
                won = re.search(r"([\d,]+)\s*원", tail)
                candidates = [won.group(1)] if won else []
                candidates.extend(re.findall(r"[\d,]+", re.sub(r"1?주당.*?배당금", "", tail)))
                for token in candidates:
                    digits = token.replace(",", "")
                    if digits.isdigit():
                        amount = int(digits)
                        if amount > 0:
                            return amount
            elif common_col is not None and common_col < len(row):
                # Column-style layout: header names 보통주 once, DPS rows
                # carry only the per-share 배당 label (e.g. the pre-2020 form).
                amount = _parse_amount_cell(row[common_col])
                if amount is not None and amount > 0:
                    return amount
    # Fallback: free-text scan for a per-share 보통주 amount outside tables.
    for m in re.finditer(r"보통주\s*1?주당[^0-9]*?배당금[^0-9]{0,30}?(\d[\d,]*)\s*원?", full_text):
        amount = int(m.group(1).replace(",", ""))
        if amount > 0 and not _cell_has_date(m.group(0)):
            return amount
    return None


# 양식 필드 라벨만 인정한다. 공시 하단 주석("1. 상기 4항의 시가배당율은 ...")도 같은 단어를
# 포함하므로, 라벨 셀 전체가 필드명 형태일 때만 값 행으로 본다.
_TOTAL_LABEL_RE = re.compile(r"^(?:\d+\.)?배당금?총액(?:\(원\))?$")
_YIELD_LABEL_RE = re.compile(r"^(?:\d+\.)?시가배당[율률](?:\(%\))?$")


def _label_cell(row: list[str]) -> str:
    return row[0].replace(" ", "").replace("\u3000", "") if row else ""


def _extract_total_krw(tables: list[list[list[str]]]) -> int | None:
    """Filing's printed dividend total (common + preferred as printed).

    Only a row whose label cell is the form field itself is read, and only from its value cells;
    footnotes that mention the total are ignored. The last value cell wins so a correction notice's
    "after" column and the restated body both yield the corrected total.
    """
    total: int | None = None
    for table in tables:
        for row in table:
            if _TOTAL_LABEL_RE.fullmatch(_label_cell(row)) is None:
                continue
            values = [amount for cell in row[1:] if (amount := _parse_amount_cell(cell)) is not None]
            if values:
                total = values[-1]
    return total


def _parse_yield_cell(cell: str) -> Decimal | None:
    text = cell.strip().replace("%", "").replace("\uff05", "").replace(",", "").strip()
    if not text or _cell_has_date(cell):
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text) is None:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:  # pragma: no cover - guarded by the pattern above
        return None


def _extract_market_yield_pct(tables: list[list[list[str]]]) -> Decimal | None:
    """Filing's printed 시가배당율 for common shares, if the form states one.

    Only the form field row is read. When the row names the share class (``보통주식``), the value
    to its right is used; when a header row names the common-share column, that column is used;
    otherwise the last numeric value cell (a correction notice's "after" column). Footnotes are
    ignored.
    """
    market_yield: Decimal | None = None
    for table in tables:
        # 구양식은 머리글 행에 "보통주|우선주" 열을 한 번만 두고, 값 행에는 계층명이 없다.
        header_col = next(
            (idx for row in table for idx, cell in enumerate(row) if idx > 0 and "보통주" in cell),
            None,
        )
        for row in table:
            if _YIELD_LABEL_RE.fullmatch(_label_cell(row)) is None:
                continue
            cells = row[1:]
            found: Decimal | None = None
            common = next((i for i, cell in enumerate(cells) if "보통주" in cell), None)
            if common is not None:
                for cell in cells[common + 1 :]:
                    if (found := _parse_yield_cell(cell)) is not None:
                        break
            elif header_col is not None and header_col < len(row):
                found = _parse_yield_cell(row[header_col])
            if found is None and common is None:
                values = [value for cell in cells if (value := _parse_yield_cell(cell)) is not None]
                found = values[-1] if values else None
            if found is not None:
                market_yield = found
    return market_yield


def _extract_agm_date(rows: list[list[str]]) -> date | None:
    """Scheduled shareholders' meeting date; an unparseable or undecided value is simply absent."""
    for row in rows:
        if not _is_body_field_row(row):
            continue
        if "주주총회예정일" in "".join(row).replace(" ", ""):
            for cell in row[1:]:
                if cell.strip():
                    return _parse_date_token(cell.strip())
    return None


def _extract_dividend_kind(rows: list[list[str]]) -> str | None:
    """Dividend classification cell ("결산배당", "중간배당", "분기배당"), if the form states one."""
    for row in rows:
        if not _is_body_field_row(row):
            continue
        head = re.sub(r"^\d+\s*[.)]?\s*", "", "".join(row[:1]).replace(" ", ""))
        if head == "배당구분":
            for cell in row[1:]:
                if cell.strip():
                    return cell.strip()
    return None


def _extract_pay_value(rows: list[list[str]]) -> str | None:
    labels = ("지급예정일", "지급 예정일", "배당금지급", "배당 지급", "지급일")
    for row in rows:
        if not _is_body_field_row(row):
            continue
        # 정정 공시는 "...지급일자 확정에 관한 사항입니다" 같은 안내 문장을 먼저 싣는다.
        # 필드 행은 (번호 다음) 라벨로 시작하므로 라벨이 문장 중간에 있는 행은 건너뛴다.
        head = re.sub(r"^\d+\s*[.)]?\s*", "", "".join(row[:1]).replace(" ", ""))
        if any(head.startswith(label.replace(" ", "")) for label in labels):
            for cell in row[1:]:
                if cell.strip():
                    return cell.strip()
            return row[0]
    return None


def parse_dividend_decision(
    *, archive_bytes: bytes, rcept_no: str, corp_code: str, received_on: date, report_nm: str = ""
) -> DividendDecision:
    """Extract record date, pay date, and common-share DPS from a decision filing's document archive.

    The common-share DPS is read only from a row labelled as a per-share dividend (``1주당 배당금`` or
    the pre-2020 ``주당배당금``) in the common-share column; a total amount, a share count or a yield is
    never accepted as DPS. The total dividend and the market-price yield printed by the same filing are
    returned so the builder can check the DPS against them.

    Args:
        archive_bytes: Raw ``document.xml`` ZIP archive bytes.
        rcept_no: 14-digit DART receipt number owning the filing.
        corp_code: 8-digit DART corporation code.
        received_on: Filing receipt date (calendar date, KST).
        report_nm: Filing title from the disclosure list; ``[기재정정]`` marks a correction.

    Returns:
        Parsed decision with common-share DPS in KRW; undecided pay dates become ``None``.

    Raises:
        PITDataError: the archive is unreadable or lacks a record date or a labelled common-share DPS;
            missing values are never defaulted.
    """
    receipt = str(rcept_no or "").strip()
    corp = str(corp_code or "").strip()
    if len(receipt) != 14 or not receipt.isdigit():
        raise PITDataError(f"malformed rcept_no value: {rcept_no!r}")
    if len(corp) != 8 or not corp.isdigit():
        raise PITDataError(f"malformed corp_code value: {corp_code!r}")
    if not isinstance(received_on, date):
        raise PITDataError(f"malformed received_on value: {received_on!r}")
    if not archive_bytes:
        raise PITDataError("empty dividend-decision archive; certification blocked")
    try:
        buf = io.BytesIO(archive_bytes)
        with zipfile.ZipFile(buf) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        # reason: adapter boundary — hostile archives fail in implementation-specific ways; fail closed.
        raise PITDataError("unreadable dividend-decision archive; certification blocked") from exc
    if len(infos) > _MAX_MEMBERS:
        raise PITDataError("dividend-decision archive has too many members; certification blocked")
    total = 0
    for info in infos:
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/").lstrip("/")
        if not name or ".." in name.split("/"):
            raise PITDataError("unsafe dividend-decision archive member; certification blocked")
        if info.file_size > _MAX_MEMBER_BYTES:
            raise PITDataError("dividend-decision archive member too large; certification blocked")  # pragma: no cover - DoS hardening
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            raise PITDataError("dividend-decision archive too large; certification blocked")  # pragma: no cover - DoS hardening
    texts: list[str] = []
    buf2 = io.BytesIO(archive_bytes)
    with zipfile.ZipFile(buf2) as zf2:
        for info in zf2.infolist():
            if info.is_dir():
                continue
            raw = zf2.read(info.filename)
            if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                raise PITDataError("unsafe dividend-decision archive content; certification blocked")
            text = _decode_member(raw)
            if text is None or "<" not in text:
                raise PITDataError("unreadable dividend-decision archive; certification blocked")
            texts.append(text.replace("&nbsp;", " "))
    full_text = "\n".join(texts)
    tables: list[list[list[str]]] = []
    for text in texts:
        tables.extend(_extract_tables(text))
    rows = _flatten(tables)
    record_cell = _find_labeled_date(rows, labels=("배당기준일", "기준일", "기록일", "결산기준일"))
    record_date = _parse_date_token(record_cell or "") if record_cell else None
    if record_date is None and record_cell is not None and (
        record_cell.strip() in _UNDECIDED_TOKENS or "미정" in record_cell
    ):
        raise UndecidedRecordDateError("dividend-decision archive has an undecided record date")
    if record_date is None:
        # Layouts that render the record date outside a table cell.
        m = re.search(r"(배당기준일|기준일)[^0-9]{0,20}(\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2})", full_text)
        if m:
            record_date = _parse_date_token(m.group(2))
    if record_date is None:
        raise PITDataError("dividend-decision archive lacks a record date; certification blocked")
    pay_cell = _extract_pay_value(rows)
    pay_date: date | None = None
    if pay_cell is not None:
        stripped = pay_cell.strip()
        if stripped in _UNDECIDED_TOKENS or "미정" in stripped:
            pay_date = None
        else:
            pay_date = _parse_date_token(stripped)
            if pay_date is None:
                # A pay row exists but carries no parseable date and no
                # undecided marker; fail closed rather than guessing.
                raise PITDataError("dividend-decision archive lacks a parseable pay date; certification blocked")
    dps = _extract_dps(tables, full_text)
    if dps is None:
        raise PITDataError("dividend-decision archive lacks a common-share DPS; certification blocked")
    title = str(report_nm or "").strip()
    if title.startswith("[기재정정]"):
        is_correction = True
    elif title:
        is_correction = False
    else:
        is_correction = _CORRECTION_MARK in full_text
    return DividendDecision(
        rcept_no=receipt,
        corp_code=corp,
        received_on=received_on,
        record_date=record_date,
        pay_date=pay_date,
        dps_common_krw=dps,
        is_correction=is_correction,
        agm_date=_extract_agm_date(rows),
        dividend_kind=_extract_dividend_kind(rows),
        total_krw=_extract_total_krw(tables),
        market_yield_pct=_extract_market_yield_pct(tables),
    )
