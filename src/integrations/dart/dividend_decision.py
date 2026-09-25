"""Cash/in-kind dividend decision filing parsing (offline, defensive)."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser

from src.core.pit import PITDataError

__all__ = ["DividendDecision", "is_dividend_decision_title", "parse_dividend_decision"]

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


class _TableParser(HTMLParser):
    """Tolerant table reader for DART's non-XML decision forms."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        name = tag.upper()
        if name == "TABLE":
            self._current_table = []
            self.tables.append(self._current_table)
        elif name == "TR":
            self._current_row = []
        elif name in {"TD", "TH"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        name = tag.upper()
        if name in {"TD", "TH"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(" ".join("".join(self._current_cell).split()))
            self._current_cell = None
        elif name == "TR" and self._current_row is not None:
            if self._current_table is None:
                self._current_table = []
                self.tables.append(self._current_table)
            self._current_table.append(self._current_row)
            self._current_row = None


def _decode_member(raw: bytes) -> str | None:
    for codec in ("utf-8", "cp949"):
        try:
            return raw.decode(codec).lstrip("\ufeff")
        except UnicodeDecodeError:
            continue
    return None  # pragma: no cover - cp949 maps nearly all byte values


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


def _extract_tables(text: str) -> list[list[list[str]]]:
    parser = _TableParser()
    parser.feed(text)
    return [table for table in parser.tables if table]


def _flatten(tables: list[list[list[str]]]) -> list[list[str]]:
    return [row for table in tables for row in table]


def _find_labeled_date(rows: list[list[str]], *, labels: tuple[str, ...]) -> str | None:
    for row in rows:
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
            joined = "".join(row)
            if "배당" not in joined and "주당" not in joined:
                continue
            if "보통주" in joined:
                for cell in row[1:]:
                    amount = _parse_amount_cell(cell)
                    if amount is not None and amount > 0:
                        return amount
                tail = row[0].split("보통주", 1)[-1]
                for token in re.findall(r"[\d,]+", tail):
                    try:
                        amount = int(token.replace(",", ""))
                    except ValueError:
                        continue
                    if amount > 0:
                        return amount
            elif common_col is not None and common_col < len(row):
                # Column-style layout: header names 보통주 once, DPS rows
                # carry only the 배당 label (e.g. the pre-2020 form).
                amount = _parse_amount_cell(row[common_col])
                if amount is not None and amount > 0:
                    return amount
    # Fallback: free-text scan for "보통주 ... N원" outside tables.
    for m in re.finditer(r"보통주[^0-9]{0,30}(\d[\d,]*)\s*원?", full_text):
        amount = int(m.group(1).replace(",", ""))
        if amount > 0 and not _cell_has_date(m.group(0)):
            return amount
    return None


def _extract_pay_value(rows: list[list[str]]) -> str | None:
    labels = ("지급예정일", "지급 예정일", "배당금지급", "배당 지급", "지급일")
    for row in rows:
        joined = "".join(row).replace(" ", "")
        if any(label.replace(" ", "") in joined for label in labels):
            for cell in row[1:]:
                if cell.strip():
                    return cell.strip()
            return row[0]
    return None


def parse_dividend_decision(
    *, archive_bytes: bytes, rcept_no: str, corp_code: str, received_on: date
) -> DividendDecision:
    """Extract record date, pay date, and common-share DPS from a decision filing's document archive.

    Args:
        archive_bytes: Raw ``document.xml`` ZIP archive bytes.
        rcept_no: 14-digit DART receipt number owning the filing.
        corp_code: 8-digit DART corporation code.
        received_on: Filing receipt date (calendar date, KST).

    Returns:
        Parsed decision with common-share DPS in KRW; undecided pay dates
        become ``None``.

    Raises:
        PITDataError: the archive is unreadable or lacks a record date or
            common-share DPS; missing values are never defaulted.
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
    except Exception as exc:
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
    return DividendDecision(
        rcept_no=receipt,
        corp_code=corp,
        received_on=received_on,
        record_date=record_date,
        pay_date=pay_date,
        dps_common_krw=dps,
        is_correction=_CORRECTION_MARK in full_text,
    )
