"""Early earnings releases: preliminary results (I002) and profit-structure changes (I001).

Preliminary fair disclosures publish the labelled quarter's numbers weeks before the
periodic report; profit-structure changes publish the annual numbers before the audit
report. Both are parsed defensively: anything that cannot be read without guessing is
withheld with a stable reason, never defaulted.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from src.core.pit import PITDataError
from src.integrations.dart.html_tables import decode_member as _decode_member
from src.integrations.dart.html_tables import extract_tables as _extract_tables_shared

__all__ = [
    "EarningsRelease",
    "EarningsReleaseKind",
    "EarningsReleaseParseError",
    "EarningsReleaseValue",
    "ReleaseBasis",
    "ReleaseSpan",
    "classify_earnings_release_title",
    "parse_earnings_release",
]

_MAX_MEMBERS = 32
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_UNIT_MULTIPLIERS: tuple[tuple[str, float], ...] = (
    ("십억원", 1e9),
    ("백만원", 1e6),
    ("천원", 1e3),
    ("억원", 1e8),
)
_BARE_WON_RE = re.compile(r"(?<![천백만십억])원")

_QUARTER_END_MONTH_DAY: dict[int, tuple[int, int]] = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

_CORRECTION_MARKS = ("[기재정정]", "[첨부정정]")
_TITLE_MARKERS = ("[기재정정]", "[첨부정정]", "[첨부추가]")


class EarningsReleaseKind(StrEnum):
    """Which official channel published the early numbers."""

    PRELIMINARY = "preliminary"
    PROFIT_CHANGE = "profit_change"


class ReleaseBasis(StrEnum):
    """Accounting basis the filing states for its numbers."""

    CONSOLIDATED = "consolidated"
    SEPARATE = "separate"


class ReleaseSpan(StrEnum):
    """Which slice of the fiscal year one value covers."""

    QUARTER = "quarter"
    CUMULATIVE = "cumulative"
    ANNUAL = "annual"


class EarningsReleaseParseError(PITDataError):
    """A release archive whose numbers cannot be read without guessing.

    Attributes:
        reason: Stable machine key (``no_result_table``, ``unknown_unit``, ``unrecognized_period``,
            ``basis_conflict``, ``no_metrics``, ``off_season``, ``period_lag``, ``not_an_archive``).
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class EarningsReleaseValue:
    """One metric of a release in KRW, with its prior-year comparator."""

    metric: str
    span: ReleaseSpan
    current_krw: float | None
    prior_year_krw: float | None


@dataclass(frozen=True, slots=True)
class EarningsRelease:
    """Parsed early earnings release for its labelled fiscal period."""

    rcept_no: str
    corp_code: str
    received_on: date
    kind: EarningsReleaseKind
    basis: ReleaseBasis
    is_correction: bool
    fiscal_year: int
    fiscal_quarter: int
    period_label: str
    values: tuple[EarningsReleaseValue, ...]


def _compact_title(report_nm: str) -> str:
    """Title with leading DART markers and all whitespace removed."""
    compact = re.sub(r"\s+", "", str(report_nm or ""))
    while True:
        for marker in _TITLE_MARKERS:
            if compact.startswith(marker):
                compact = compact[len(marker) :]
                break
        else:
            return compact


def classify_earnings_release_title(report_nm: str) -> EarningsReleaseKind | None:
    """Classify a DART list title as an early earnings release, or None.

    Leading DART markers (``[기재정정]``, ``[첨부정정]``, ``[첨부추가]``) and all whitespace are
    ignored. Subsidiary filings (``자회사의 주요경영사항``), guidance (``영업실적등에대한전망``) and
    any other fair disclosure return None, because their numbers do not describe the listed issuer's
    own reported period.
    """
    compact = _compact_title(report_nm)
    if not compact:
        return None
    if "자회사의주요경영사항" in compact or "영업실적등에대한전망" in compact or "수시공시의무관련사항" in compact:
        return None
    if "영업(잠정)실적" in compact:
        return EarningsReleaseKind.PRELIMINARY
    if "매출액또는손익구조" in compact and ("변동" in compact or "변경" in compact):
        return EarningsReleaseKind.PROFIT_CHANGE
    return None


def _normalize_metric_label(cell: str) -> str:
    """Strip whitespace, leading dashes/numbering and a trailing parenthetical."""
    text = re.sub(r"\s+", "", str(cell or ""))
    text = re.sub(r"^[\-△▲\uFF0D\u2500\d.\)\]]+", "", text)
    text = re.sub(r"\(.*\)$", "", text)
    return text


_METRIC_BY_LABEL: dict[str, str] = {
    "매출액": "sales",
    "영업수익": "sales",
    "영업이익": "operating_profit",
    "법인세비용차감전계속사업이익": "pretax_income",
    "법인세차감전계속사업이익": "pretax_income",
    "당기순이익": "net_income",
    "지배기업소유주지분순이익": "controlling_net_income",
}


def _parse_amount_cell(cell: str) -> float | None:
    """Parse a monetary cell; empty, lone dashes and free text become None, never an error."""
    text = re.sub(r"\s+", "", str(cell or ""))
    if not text or text in {"-", "\uFF0D", "\u2014", "--", "△", "▲"}:
        return None
    negative = False
    if text[0] in {"-", "△", "▲", "\uFF0D"}:
        negative = True
        text = text[1:]
    if len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = text.replace(",", "").strip()
    if not text:
        return None
    try:
        amount = float(text)
    except ValueError:
        return None
    return -amount if negative else amount


def _unit_multiplier(unit_text: str) -> float | None:
    """Multiplier from the printed unit text; None when no known unit is stated."""
    text = re.sub(r"\s+", "", str(unit_text or ""))
    for token, multiplier in _UNIT_MULTIPLIERS:
        if token in text:
            return multiplier
    if _BARE_WON_RE.search(text):
        return 1.0
    return None


_RANGE_RE = re.compile(
    r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*~\s*(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})"
)
_QUARTER_FIRST_YEAR_RE = re.compile(r"(?<!\d)(\d{2,4})\s*[.\년]?\s*([1-4])\s*(?:Q|분기)")
_QUARTER_FIRST_RE = re.compile(r"(?<!\d)([1-4])\s*Q\s*(\d{2,4})(?!\d)")


def _full_year(text: str) -> int | None:
    if len(text) == 2:
        return 2000 + int(text)
    if len(text) == 4:
        return int(text)
    return None


def _parse_period_label(label: str) -> tuple[int, int] | None:
    """Parse a period label into ``(fiscal_year, fiscal_quarter)``.

    Accepted forms: a date range covering exactly one calendar quarter (``2025.10.01~2025.12.31``),
    a 2- or 4-digit year followed by a quarter (``2022.4Q``, ``22년 4분기``, ``2024 3Q``), or a
    quarter-first form (``4Q25``). Year-only, month-only and blank labels stay unrecognized because
    the covered quarter would have to be guessed.
    """
    text = str(label or "")
    ranged = _RANGE_RE.search(text)
    if ranged is not None:
        start_year, start_month, start_day, end_year, end_month, end_day = (int(v) for v in ranged.groups())
        quarter, remainder = divmod(end_month, 3)
        if (
            remainder == 0
            and start_year == end_year
            and start_month == end_month - 2
            and start_day == 1
            and (end_month, end_day) == _QUARTER_END_MONTH_DAY[quarter]
        ):
            return end_year, quarter
        return None
    match = _QUARTER_FIRST_YEAR_RE.search(text)
    if match is not None:
        year = _full_year(match.group(1))
        return None if year is None else (year, int(match.group(2)))
    match = _QUARTER_FIRST_RE.search(text)
    if match is not None:
        year = _full_year(match.group(2))
        return None if year is None else (year, int(match.group(1)))
    return None


def _quarter_end(fiscal_year: int, fiscal_quarter: int) -> date:
    month, day = _QUARTER_END_MONTH_DAY[fiscal_quarter]
    return date(fiscal_year, month, day)


def _read_archive_tables(archive_bytes: bytes) -> list[list[list[str]]]:
    """Decode ZIP members and extract their tables, failing closed on hostile archives."""
    if not archive_bytes:
        raise EarningsReleaseParseError("not_an_archive", "empty earnings-release archive")
    try:
        buf = io.BytesIO(archive_bytes)
        with zipfile.ZipFile(buf) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise EarningsReleaseParseError("not_an_archive", "unreadable earnings-release archive") from exc
    if len(infos) > _MAX_MEMBERS:
        raise EarningsReleaseParseError("not_an_archive", "earnings-release archive has too many members")
    total = 0
    for info in infos:
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/").lstrip("/")
        if not name or ".." in name.split("/"):
            raise EarningsReleaseParseError("not_an_archive", "unsafe earnings-release archive member")
        if info.file_size > _MAX_MEMBER_BYTES:
            raise EarningsReleaseParseError("not_an_archive", "earnings-release archive member too large")  # pragma: no cover - DoS hardening
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            raise EarningsReleaseParseError("not_an_archive", "earnings-release archive too large")  # pragma: no cover - DoS hardening
    texts: list[str] = []
    buf2 = io.BytesIO(archive_bytes)
    with zipfile.ZipFile(buf2) as zf2:
        for info in zf2.infolist():
            if info.is_dir():
                continue
            raw = zf2.read(info.filename)
            if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                raise EarningsReleaseParseError("not_an_archive", "unsafe earnings-release archive content")
            text = _decode_member(raw)
            if text is None or "<" not in text:
                raise EarningsReleaseParseError("not_an_archive", "unreadable earnings-release archive member")
            texts.append(text.replace("&nbsp;", " "))
    tables: list[list[list[str]]] = []
    for text in texts:
        tables.extend(_extract_tables_shared(text))
    return tables


def _find_result_table(tables: list[list[list[str]]], *, needle: str) -> list[list[str]] | None:
    for table in tables:
        for row in table:
            if any(needle in cell for cell in row):
                return table
    return None


def _parse_preliminary(
    *,
    tables: list[list[list[str]]],
    title_basis: ReleaseBasis,
    max_filing_lag_days: int,
    received_on: date,
    is_correction: bool,
) -> tuple[ReleaseBasis, int, int, str, tuple[EarningsReleaseValue, ...]]:
    table = _find_result_table(tables, needle="실적내용")
    if table is None:
        raise EarningsReleaseParseError("no_result_table", "preliminary archive carries no result table")
    table_basis = (
        ReleaseBasis.CONSOLIDATED
        if any("연결실적내용" in cell for row in table for cell in row)
        else ReleaseBasis.SEPARATE
    )
    if table_basis is not title_basis:
        raise EarningsReleaseParseError("basis_conflict", "preliminary table basis contradicts its title")
    unit_text = next((cell for row in table for cell in row if "단위" in cell), "")
    multiplier = _unit_multiplier(unit_text)
    if multiplier is None:
        raise EarningsReleaseParseError("unknown_unit", f"preliminary unit is not KRW-based: {unit_text!r}")
    header_at = next(
        (index for index, row in enumerate(table) if "당기실적" in row and "전년동기실적" in row),
        None,
    )
    if header_at is None or header_at + 1 >= len(table):
        raise EarningsReleaseParseError("no_result_table", "preliminary archive carries no period row")
    period_row = table[header_at + 1]
    first_label = next((cell for cell in period_row if cell.strip()), "")
    parsed_period = _parse_period_label(first_label)
    if parsed_period is None:
        raise EarningsReleaseParseError("unrecognized_period", f"unrecognized preliminary period: {first_label!r}")
    fiscal_year, fiscal_quarter = parsed_period
    quarter_end = _quarter_end(fiscal_year, fiscal_quarter)
    if is_correction:
        if not quarter_end < received_on:
            raise EarningsReleaseParseError("period_lag", "correction precedes its labelled quarter end")
    elif not quarter_end < received_on <= date.fromordinal(quarter_end.toordinal() + max_filing_lag_days):
        raise EarningsReleaseParseError("period_lag", "preliminary receipt lies outside its lag window")
    values: list[EarningsReleaseValue] = []
    current_metric: str | None = None
    for row in table[header_at + 2 :]:
        if "당해실적" in row:
            metric = _METRIC_BY_LABEL.get(_normalize_metric_label(row[0]))
            if metric is None:
                continue
            current_metric = metric
            span = ReleaseSpan.QUARTER
            cells = row[row.index("당해실적") + 1 :]
        elif "누계실적" in row:
            if current_metric is None:
                continue
            metric = current_metric
            span = ReleaseSpan.CUMULATIVE
            cells = row[row.index("누계실적") + 1 :]
        else:
            continue
        # 값 열은 당기, 전기, 전기대비(w), 전년동기, 전년동기대비(w) 순이고 w는 서식마다 다르다
        # (증감율만이면 1, 흑자적자전환여부가 붙으면 2). 헤더 열 수가 아니라 값 열 수로 w를 구한다.
        change_width, remainder = divmod(len(cells) - 3, 2)
        if remainder or change_width < 1:
            continue
        current = _parse_amount_cell(cells[0])
        prior_year = _parse_amount_cell(cells[2 + change_width])
        values.append(
            EarningsReleaseValue(
                metric=metric,
                span=span,
                current_krw=current * multiplier if current is not None else None,
                prior_year_krw=prior_year * multiplier if prior_year is not None else None,
            )
        )
    if not values:
        raise EarningsReleaseParseError("no_metrics", "preliminary archive carries no known metric row")
    values.sort(key=lambda item: (item.metric, item.span.value))
    return table_basis, fiscal_year, fiscal_quarter, first_label.strip(), tuple(values)


def _parse_profit_change(
    *,
    tables: list[list[list[str]]],
    received_on: date,
) -> tuple[ReleaseBasis, tuple[EarningsReleaseValue, ...]]:
    if received_on.month < 1 or received_on.month > 4:
        raise EarningsReleaseParseError("off_season", "profit-structure change received outside January-April")
    table = _find_result_table(tables, needle="변동내용")
    if table is None:
        raise EarningsReleaseParseError("no_result_table", "profit-change archive carries no result table")
    basis_cell = ""
    for row in table:
        for index, cell in enumerate(row):
            if "재무제표의종류" in re.sub(r"\s+", "", cell) and index + 1 < len(row):
                basis_cell = row[index + 1]
                break
    compact_basis = re.sub(r"\s+", "", basis_cell)
    if compact_basis == "연결":
        basis = ReleaseBasis.CONSOLIDATED
    elif compact_basis in {"별도", "개별"}:
        basis = ReleaseBasis.SEPARATE
    else:
        raise EarningsReleaseParseError("basis_conflict", f"unknown profit-change basis: {basis_cell!r}")
    unit_text = next((cell for row in table for cell in row if "단위" in cell and "변동내용" in cell), "")
    if not unit_text:
        unit_text = next((cell for row in table for cell in row if "단위" in cell), "")
    multiplier = _unit_multiplier(unit_text)
    if multiplier is None:
        raise EarningsReleaseParseError("unknown_unit", f"profit-change unit is not KRW-based: {unit_text!r}")
    values: list[EarningsReleaseValue] = []
    for row in table:
        if len(row) < 3:
            continue
        metric = _METRIC_BY_LABEL.get(_normalize_metric_label(row[0]))
        if metric is None:
            continue
        current = _parse_amount_cell(row[1])
        prior_year = _parse_amount_cell(row[2])
        values.append(
            EarningsReleaseValue(
                metric=metric,
                span=ReleaseSpan.ANNUAL,
                current_krw=current * multiplier if current is not None else None,
                prior_year_krw=prior_year * multiplier if prior_year is not None else None,
            )
        )
    if not values:
        raise EarningsReleaseParseError("no_metrics", "profit-change archive carries no known metric row")
    values.sort(key=lambda item: (item.metric, item.span.value))
    return basis, tuple(values)


def parse_earnings_release(
    *,
    archive_bytes: bytes,
    rcept_no: str,
    corp_code: str,
    received_on: date,
    report_nm: str,
    max_filing_lag_days: int,
) -> EarningsRelease:
    """Parse one release archive into KRW values for its labelled fiscal period.

    Args:
        archive_bytes: DART ``document.xml`` ZIP archive.
        rcept_no: Receipt number of the filing.
        corp_code: DART corp code of the filer.
        received_on: Receipt date (KST calendar date).
        report_nm: List title; decides kind and correction status.
        max_filing_lag_days: ``EarningsReleasePolicy.max_filing_lag_days``.

    Returns:
        The parsed release; every monetary value converted to KRW by the printed unit.

    Raises:
        EarningsReleaseParseError: the title is not a release, the archive is not a ZIP, no result
            table exists (including correction bodies carrying only a diff table), the unit is not
            one of 원/천원/백만원/억원/십억원, the period label is unrecognized, the stated basis
            contradicts the title, no known metric row is present, a profit-structure change is
            received outside January-April, or an original preliminary filing lies outside
            ``(quarter_end, quarter_end + max_filing_lag_days]``.
    """
    receipt = str(rcept_no or "").strip()
    corp = str(corp_code or "").strip()
    if len(receipt) != 14 or not receipt.isdigit():
        raise EarningsReleaseParseError("not_an_archive", f"malformed rcept_no value: {rcept_no!r}")
    if len(corp) != 8 or not corp.isdigit():
        raise EarningsReleaseParseError("not_an_archive", f"malformed corp_code value: {corp_code!r}")
    if not isinstance(received_on, date):
        raise EarningsReleaseParseError("not_an_archive", f"malformed received_on value: {received_on!r}")
    kind = classify_earnings_release_title(report_nm)
    if kind is None:
        raise EarningsReleaseParseError("not_an_archive", f"not an earnings release title: {report_nm!r}")
    title = str(report_nm or "")
    is_correction = "[기재정정]" in title or "[첨부정정]" in title
    tables = _read_archive_tables(archive_bytes)
    if kind is EarningsReleaseKind.PROFIT_CHANGE:
        basis, values = _parse_profit_change(tables=tables, received_on=received_on)
        return EarningsRelease(
            rcept_no=receipt,
            corp_code=corp,
            received_on=received_on,
            kind=kind,
            basis=basis,
            is_correction=is_correction,
            fiscal_year=received_on.year - 1,
            fiscal_quarter=4,
            period_label="",
            values=values,
        )
    title_basis = (
        ReleaseBasis.CONSOLIDATED
        if _compact_title(report_nm).startswith("연결재무제표기준")
        else ReleaseBasis.SEPARATE
    )
    basis, fiscal_year, fiscal_quarter, period_label, values = _parse_preliminary(
        tables=tables,
        title_basis=title_basis,
        max_filing_lag_days=int(max_filing_lag_days),
        received_on=received_on,
        is_correction=is_correction,
    )
    return EarningsRelease(
        rcept_no=receipt,
        corp_code=corp,
        received_on=received_on,
        kind=kind,
        basis=basis,
        is_correction=is_correction,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        period_label=period_label,
        values=values,
    )
