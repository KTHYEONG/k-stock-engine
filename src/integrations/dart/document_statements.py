"""Section-anchored DART filing-document statement parser (offline, defensive)."""

from __future__ import annotations

import io
import re
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, Literal
from zipfile import ZipInfo

from src.integrations.dart.accounts import map_standardized_account, normalize_statement_label
from src.integrations.dart.html_tables import decode_member, read_blocks

PARSER_VERSION: Final = "dart-document-statements-v1"

_MAX_MEMBERS: Final = 32
_MAX_MEMBER_BYTES: Final = 16 * 1024 * 1024
_MAX_TOTAL_BYTES: Final = 64 * 1024 * 1024

_CFS_CODE: Final = "D-0-3-2-0"
_OFS_CODE: Final = "D-0-3-4-0"

_TITLE_RE: Final = re.compile(r'<TITLE[^>]*AASSOCNOTE\s*=\s*"(D-[\d-]+)"[^>]*>', re.IGNORECASE)
_BS_RE: Final = re.compile(r"재무상태표|대차대조표")
_CF_RE: Final = re.compile(r"현금흐름표")
_SCE_RE: Final = re.compile(r"자본변동표")
_IS_RE: Final = re.compile(r"손익계산서")
_UNIT_RE: Final = re.compile(r"단\s*위\s*[:：]?\s*(백만원|천원|십억원|원)")  # noqa: RUF001
_BARE_UNIT_RE: Final = re.compile(r"백만원|천원|십억원")
_UNIT_MULTIPLIERS: Final = {"원": 1, "천원": 1_000, "백만원": 1_000_000, "십억원": 1_000_000_000}
_PERIOD_RE: Final = re.compile(r"제\s*(\d+)\s*(?:\([^)]*\))?\s*(?:기|반기|분기)")
_DATE_RE: Final = re.compile(r"(20\d{2})\s*[.년]\s*(\d{1,2})\s*[.월]\s*(\d{1,2})")
_AMOUNT_RE: Final = re.compile(r"(?:\([-△]?\d[\d,]*\)|[-△]?\d[\d,]*)")
_ABSENT_CELLS: Final = frozenset({"", "-", "―", "－"})  # noqa: RUF001
_MAIN_SUFFIX_RE: Final = re.compile(r"_\d{5}\.")
_QUARTER_RE: Final = re.compile(r"3개월|분기|반기")
_CUMULATIVE_RE: Final = re.compile(r"누적|누계|6개월|9개월")

_BS_FACTS: Final = frozenset({"assets", "debt", "equity", "cash"})
_IS_FACTS: Final = frozenset({"sales", "gross_profit", "operating_profit", "net_income"})
_CF_FACTS: Final = frozenset({"operating_cash_flow", "capex", "cash"})
_KIND_FACTS: Final = {"BS": _BS_FACTS, "IS": _IS_FACTS, "CF": _CF_FACTS}

_EXPECTED_MONTH_DAY: Final = {"11013": (3, 31), "11012": (6, 30), "11014": (9, 30), "11011": (12, 31)}
_REPORT_KIND: Final = {"11011": "annual", "11012": "half", "11013": "q1", "11014": "q3"}
_WANT_MARKER: Final = {"11013": "q1", "11012": "half", "11014": "q3"}
_QUARTER_ENDS: Final = frozenset({(3, 31), (6, 30), (9, 30), (12, 31)})
_ZIP_ERRORS: Final = (zipfile.BadZipFile, OSError, RuntimeError, ValueError, EOFError, zlib.error)


class PeriodBasis(StrEnum):
    POINT_IN_TIME = "point_in_time"
    QUARTER = "quarter"
    CUMULATIVE = "cumulative"
    ANNUAL = "annual"


@dataclass(frozen=True, slots=True)
class StatementFact:
    fact: str
    value: int
    basis: PeriodBasis
    label: str


@dataclass(frozen=True, slots=True)
class VerifiedStatements:
    """All accepted statements of one basis (consolidated or separate)."""

    consolidated: bool
    period_end: date
    report_kind: Literal["annual", "half", "q1", "q3"]
    unit_multipliers: Mapping[str, int]
    facts: tuple[StatementFact, ...]
    checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentParseResult:
    statements: VerifiedStatements | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _StatementBody:
    region: str
    rows: tuple[tuple[str, ...], ...]


def _is_unsafe_name(name: str) -> bool:
    if not name:
        return True
    normalized = name.replace("\\", "/").lstrip("/")
    return ".." in normalized.split("/")


def _member_guard(infos: list[ZipInfo]) -> str | None:
    """Check archive members without reading them, mirroring the legacy guards."""
    if len(infos) > _MAX_MEMBERS:
        return "too_many_members"
    seen: set[str] = set()
    total = 0
    for info in infos:
        if info.filename in seen:
            return "duplicate_member"
        seen.add(info.filename)
        if info.is_dir():
            continue
        if _is_unsafe_name(info.filename):
            return "unsafe_member_path"
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            return "symlink_member"
        if info.flag_bits & 0x1:
            return "encrypted_member"
        if info.file_size > _MAX_MEMBER_BYTES:
            return "member_too_large"
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            return "expanded_too_large"
    return None


def _main_member_name(infos: list[ZipInfo]) -> str | None:
    """Return the main report member: the first file without a ``_NNNNN`` suffix."""
    for info in infos:
        if info.is_dir():
            continue
        leaf = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if _MAIN_SUFFIX_RE.search(leaf):
            continue
        return info.filename
    return None


def _extract_section(markup: str, code: str) -> str | None:
    """Return the fragment from a DART form section tag to the next section tag."""
    matches = list(_TITLE_RE.finditer(markup))
    for index, match in enumerate(matches):
        if match.group(1) == code:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markup)
            return markup[match.end() : end]
    return None


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _classify_heading(text: str) -> str | None:
    """Name the statement a heading block starts, or None when it starts none."""
    if "주석" in text or len(text) > 60:
        return None
    squeezed = _compact(text)
    if _BS_RE.search(squeezed):
        return "BS"
    if _CF_RE.search(squeezed):
        return "CF"
    if _SCE_RE.search(squeezed):
        return "SCE"
    if _IS_RE.search(squeezed):
        return "IS"
    return None


def _unit_of(region: str) -> int | None:
    match = _UNIT_RE.search(region)
    if match is not None:
        return _UNIT_MULTIPLIERS[match.group(1)]
    bare = _BARE_UNIT_RE.search(region)
    if bare is not None:
        return _UNIT_MULTIPLIERS[bare.group(0)]
    return None


def _scan_section(markup: str) -> dict[str, list[_StatementBody]]:
    """Collect statement bodies inside one form section, in document order."""
    bodies: dict[str, list[_StatementBody]] = {}
    pending: str | None = None
    region_parts: list[str] = []
    for block in read_blocks(markup):
        if block.kind == "text":
            kind = _classify_heading(block.text)
            if kind is not None:
                pending = kind
                region_parts = [block.text]
            elif pending is not None:
                region_parts.append(block.text)
            continue
        grid = block.grid
        first = grid[0][0] if grid[0] else ""
        kind = _classify_heading(first) if len(grid) <= 6 else None
        if kind is not None:
            pending = kind
            region_parts = [" ".join(cell for row in grid for cell in row)]
        elif pending is not None:
            if len(grid) >= 5:
                if pending != "SCE":
                    bodies.setdefault(pending, []).append(_StatementBody(region=" ".join(region_parts), rows=grid))
                pending = None
            else:
                region_parts.append(" ".join(cell for row in grid for cell in row))
        continue
    return bodies


def _current_columns(rows: tuple[tuple[str, ...], ...]) -> tuple[list[int], int]:
    """Return the grid columns carrying the highest ``제 N 기`` number and its row."""
    for row_index, row in enumerate(rows[:3]):
        found = [(pos, int(m.group(1))) for pos, cell in enumerate(row) if (m := _PERIOD_RE.search(cell))]
        if found:
            top = max(num for _, num in found)
            return [pos for pos, num in found if num == top], row_index
    return [], -1


def _parse_amount(cell: str) -> int | None:
    text = _compact(cell)
    if text in _ABSENT_CELLS:
        return None
    if _AMOUNT_RE.fullmatch(text) is None:
        return None
    negative = text.startswith("(") or "△" in text or text.lstrip("(").startswith("-")
    return -int(re.sub(r"\D", "", text)) if negative else int(re.sub(r"\D", "", text))


def _row_value(row: tuple[str, ...], columns: list[int]) -> int | None:
    """Return the single non-empty amount among the current columns, if exactly one."""
    amounts = [_parse_amount(row[pos]) for pos in columns if pos < len(row)]
    present = [value for value in amounts if value is not None]
    if len(present) == 1:
        return present[0]
    return None


def _candidate_dates(
    region: str, rows: tuple[tuple[str, ...], ...], columns: list[int]
) -> list[date]:
    """Collect every plausible balance-sheet date from the heading region and headers."""
    cells = [region] + [row[pos] for row in rows[:3] for pos in columns if pos < len(row)]
    found: list[date] = []
    for cell in cells:
        for match in _DATE_RE.finditer(cell):
            try:
                found.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            except ValueError:
                continue
    return found


def _markers_in(text: str) -> set[str]:
    """Detect report-kind markers (``1분기``, ``반기``, ``3분기`` or a bare ``분기``)."""
    squeezed = _compact(text)
    markers: set[str] = set()
    if "1분기" in squeezed:
        markers.add("q1")
    if "3분기" in squeezed:
        markers.add("q3")
    if "반기" in squeezed or "2분기" in squeezed:
        markers.add("half")
    rest = squeezed.replace("1분기", "").replace("2분기", "").replace("3분기", "")
    if "분기" in rest:
        markers.add("quarter")
    return markers


def _flow_columns(
    kind: str,
    rows: tuple[tuple[str, ...], ...],
    columns: list[int],
    period_index: int,
    reprt_code: str,
) -> tuple[PeriodBasis, list[int]] | None:
    """Choose the value columns and the period basis for one statement body."""
    if kind == "BS":
        return (PeriodBasis.POINT_IN_TIME, columns)
    if reprt_code == "11011":
        return (PeriodBasis.ANNUAL, columns)
    sub = rows[period_index + 1]
    quarter = [pos for pos in columns if pos < len(sub) and _QUARTER_RE.search(_compact(sub[pos]))]
    cumulative = [pos for pos in columns if pos < len(sub) and _CUMULATIVE_RE.search(_compact(sub[pos]))]
    if kind == "CF":
        return (PeriodBasis.CUMULATIVE, cumulative if cumulative else columns)
    if quarter:
        return (PeriodBasis.QUARTER, quarter)
    if reprt_code == "11013":
        return (PeriodBasis.QUARTER, cumulative if cumulative else columns)
    if cumulative:
        return (PeriodBasis.CUMULATIVE, cumulative)
    current = rows[period_index]
    if any(_QUARTER_RE.search(_compact(current[pos])) for pos in columns if pos < len(current)):
        return (PeriodBasis.QUARTER, columns)
    if any(_CUMULATIVE_RE.search(_compact(current[pos])) for pos in columns if pos < len(current)):
        return (PeriodBasis.CUMULATIVE, columns)
    return None


def _extract_kind_facts(
    kind: str, body: _StatementBody, unit: int, reprt_code: str
) -> tuple[dict[str, tuple[int, PeriodBasis, str]], list[str]]:
    """Map one statement body to scaled facts, withholding ambiguous rows and facts."""
    rows = body.rows
    columns, period_index = _current_columns(rows)
    if not columns:
        return {}, [f"missing_current_column:{kind}"]
    flow = _flow_columns(kind, rows, columns, period_index, reprt_code)
    if flow is None:
        return {}, ["missing_period_basis:IS"]
    basis, use_columns = flow
    allowed = _KIND_FACTS[kind]
    facts: dict[str, tuple[int, PeriodBasis, str]] = {}
    dropped: set[str] = set()
    diags: list[str] = []
    for row in rows:
        if not row:
            continue
        label = normalize_statement_label(row[0])
        fact = map_standardized_account(account_nm=label)
        if fact is None:
            # Labels such as "수익(매출액)" lose their closing bracket to
            # normalization, so the vocabulary's own parenthetical stripping
            # cannot see them; retry against the raw cell before giving up.
            fact = map_standardized_account(account_nm=row[0])
        if fact is None or fact not in allowed or fact in dropped:
            continue
        amount = _row_value(row, use_columns)
        if amount is None:
            continue
        scaled = amount * unit
        value = abs(scaled) if fact == "capex" else scaled
        if fact in facts:
            if facts[fact][0] != value:
                del facts[fact]
                dropped.add(fact)
                diags.append(f"ambiguous_fact:{fact}")
            continue
        facts[fact] = (value, basis, label)
    return facts, diags


def _verify_period(body: _StatementBody, reprt_code: str, biz_year: str) -> tuple[date | None, list[str]]:
    """Check the balance-sheet date and the report-kind marker against the identity."""
    month_day = _EXPECTED_MONTH_DAY.get(reprt_code)
    if month_day is None or len(biz_year) != 4 or not biz_year.isdigit():
        return None, ["period_mismatch"]
    try:
        expected = date(int(biz_year), month_day[0], month_day[1])
    except ValueError:
        return None, ["period_mismatch"]
    columns, _ = _current_columns(body.rows)
    candidates = _candidate_dates(body.region, body.rows, columns)
    if not candidates:
        return None, ["missing_period:BS"]
    if expected not in candidates:
        latest = max(candidates)
        if (latest.month, latest.day) not in _QUARTER_ENDS:
            return None, ["non_december_fiscal_year"]
        return None, ["period_mismatch"]
    markers = _markers_in(body.region)
    if not markers:
        header_cells = [row[pos] for row in body.rows[:3] for pos in columns if pos < len(row)]
        markers = _markers_in(" ".join(header_cells))
    if reprt_code == "11011":
        if markers:
            return None, ["period_mismatch"]
        return expected, []
    want = _WANT_MARKER.get(reprt_code, "")
    if want in markers or "quarter" in markers:
        return expected, []
    return None, ["period_mismatch"]


def _report_kind(reprt_code: str) -> Literal["annual", "half", "q1", "q3"]:
    if reprt_code == "11011":
        return "annual"
    if reprt_code == "11012":
        return "half"
    if reprt_code == "11013":
        return "q1"
    return "q3"


def _verify_section(
    markup: str, *, consolidated: bool, reprt_code: str, biz_year: str
) -> tuple[VerifiedStatements | None, list[str]]:
    """Verify one basis (consolidated or separate) inside its form section."""
    bodies = _scan_section(markup)
    diags: list[str] = [f"ambiguous_statement:{kind}" for kind in ("BS", "IS", "CF") if len(bodies.get(kind, ())) > 1]
    bs_list = bodies.get("BS", [])
    if len(bs_list) != 1:
        if not bs_list:
            diags.append("missing_statement:BS")
        return None, diags
    bs = bs_list[0]
    bs_columns, _ = _current_columns(bs.rows)
    if not bs_columns:
        diags.append("missing_current_column:BS")
        return None, diags
    period_end, period_diags = _verify_period(bs, reprt_code, biz_year)
    diags.extend(period_diags)
    if period_end is None:
        return None, diags
    collected: dict[str, dict[str, tuple[int, PeriodBasis, str]]] = {}
    multipliers: dict[str, int] = {}
    for kind in ("BS", "IS", "CF"):
        kind_bodies = bodies.get(kind, [])
        if len(kind_bodies) != 1:
            continue
        unit = _unit_of(kind_bodies[0].region)
        if unit is None:
            diags.append(f"missing_unit:{kind}")
            if kind == "BS":
                return None, diags
            continue
        facts, fact_diags = _extract_kind_facts(kind, kind_bodies[0], unit, reprt_code)
        diags.extend(fact_diags)
        if kind == "BS":
            assets = facts.get("assets")
            debt = facts.get("debt")
            equity = facts.get("equity")
            if assets is None or debt is None or equity is None or assets[0] != debt[0] + equity[0]:
                diags.append("identity_failed:bs_balance")
                return None, diags
        if facts:
            collected[kind] = facts
            multipliers[kind] = unit
    checks = ["bs_balance"]
    bs_cash = collected.get("BS", {}).get("cash")
    cf_cash = collected.get("CF", {}).get("cash")
    if bs_cash is not None and cf_cash is not None:
        if bs_cash[0] != cf_cash[0]:
            diags.append("identity_failed:cf_cash_tieout")
            collected.pop("CF", None)
            multipliers.pop("CF", None)
        else:
            checks.append("cf_cash_tieout")
    ordered = [
        StatementFact(fact=fact, value=value, basis=basis, label=label)
        for kind in ("BS", "IS", "CF")
        for fact, (value, basis, label) in collected.get(kind, {}).items()
    ]
    return VerifiedStatements(
        consolidated=consolidated,
        period_end=period_end,
        report_kind=_report_kind(reprt_code),
        unit_multipliers=multipliers,
        facts=tuple(ordered),
        checks=tuple(checks),
    ), diags


def _parse_archive(archive_bytes: bytes, *, reprt_code: str, biz_year: str) -> DocumentParseResult:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            infos = zf.infolist()
            guard = _member_guard(infos)
            if guard is not None:
                return DocumentParseResult(statements=None, diagnostics=(guard,))
            main = _main_member_name(infos)
            if main is None:
                return DocumentParseResult(statements=None, diagnostics=("missing_main_member",))
            raw = zf.read(main)
    except _ZIP_ERRORS:
        return DocumentParseResult(statements=None, diagnostics=("bad_zip",))
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        return DocumentParseResult(statements=None, diagnostics=("unsafe_xml_declaration",))
    text = decode_member(raw)
    if text is None:
        return DocumentParseResult(statements=None, diagnostics=("decode_failed",))
    diags: list[str] = []
    chosen: VerifiedStatements | None = None
    for code, consolidated in ((_CFS_CODE, True), (_OFS_CODE, False)):
        section = _extract_section(text, code)
        if section is None:
            diags.append(f"missing_section:{code}")
            continue
        verified, section_diags = _verify_section(
            section, consolidated=consolidated, reprt_code=reprt_code, biz_year=biz_year
        )
        diags.extend(section_diags)
        if chosen is None and verified is not None:
            chosen = verified
    seen: set[str] = set()
    unique: list[str] = []
    for diag in diags:
        if diag not in seen:
            seen.add(diag)
            unique.append(diag)
    return DocumentParseResult(statements=chosen, diagnostics=tuple(unique))


def parse_filing_document(archive_bytes: bytes, *, reprt_code: str, biz_year: str) -> DocumentParseResult:
    """Extract verified current-period statements from a DART ``document.xml`` archive.

    A wrong but plausible number is worse than a missing one. Statements are
    located only inside the DART form sections for consolidated
    (``D-0-3-2-0``) and separate (``D-0-3-4-0``) financial statements of the
    main report member, so notes, summaries and audit reports are never read.
    A statement is accepted only when its own unit note, its current-period
    column and its accounting identities all check out; anything else is
    withheld with a diagnostic. The consolidated basis is chosen when it
    verifies, otherwise the separate one, mirroring the standard API path.

    Args:
        archive_bytes: The raw ``document.xml`` ZIP as stored in Bronze.
        reprt_code: The identity's report code (11011, 11012, 11013, 11014).
        biz_year: The identity's business year.

    Returns:
        The verified statements of one basis, or ``None`` with diagnostics.
        Never raises for malformed input.
    """
    try:
        return _parse_archive(archive_bytes, reprt_code=reprt_code, biz_year=biz_year)
    except Exception:
        return DocumentParseResult(statements=None, diagnostics=("parse_failed",))


_REPRT_QUARTER: Final = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Q4"}


def document_verified_page(
    *,
    identity: Mapping[str, str],
    result: DocumentParseResult,
    document_hash: str,
) -> dict[str, object]:
    """Build the ``document_verified`` Bronze page for one filing identity.

    A verified parse becomes a ``000`` page on the chosen basis; anything
    else becomes an ``extraction_failed`` page with no records. Both carry
    the archive hash so the document jobs stay idempotent.
    """
    from src.integrations.dart.accounts import MAPPING_VERSION

    biz_year = str(identity.get("biz_year") or "").strip()
    reprt_code = str(identity.get("reprt_code") or "").strip()
    quarter = _REPRT_QUARTER.get(reprt_code, "Q4")
    fiscal_period = f"{biz_year}{quarter}" if biz_year else str(identity.get("fiscal_period") or "")
    filing_id = str(identity.get("filing_id") or identity.get("rcept_no") or "")
    statements = result.statements
    if statements is None:
        base_identity = {key: value for key, value in dict(identity).items() if key != "raw_document_hash"}
        return {
            "source_kind": "document_verified",
            "status": "extraction_failed",
            "identity": dict(identity),
            "records": [],
            "mapping_version": MAPPING_VERSION,
            "parser_version": PARSER_VERSION,
            "diagnostics": tuple(result.diagnostics),
            **base_identity,
            "raw_document_hash": document_hash,
        }
    fs_div = "CFS" if statements.consolidated else "OFS"
    request_identity = {**{key: value for key, value in dict(identity).items() if key != "raw_document_hash"}, "fs_div": fs_div}
    corp_code = str(identity.get("corp_code") or "")
    ticker = str(identity.get("ticker") or "")
    rcept_no = str(identity.get("rcept_no") or filing_id)
    published_at = str(identity.get("published_at") or "")
    consolidated = statements.consolidated
    records: list[dict[str, object]] = [
        {
            "company_id": corp_code,
            "corp_code": corp_code,
            "ticker": ticker,
            "filing_id": filing_id,
            "rcept_no": rcept_no,
            "biz_year": biz_year,
            "reprt_code": reprt_code,
            "fiscal_period": fiscal_period,
            "published_at": published_at,
            "fact": item.fact,
            "value": float(item.value),
            "unit": "KRW",
            "consolidated": consolidated,
            "period_basis": item.basis.value,
            "restatement_id": "r0",
            "source_kind": "document_verified",
            "mapping_version": MAPPING_VERSION,
            "parser_version": PARSER_VERSION,
            "checks": list(statements.checks),
            "raw_document_hash": document_hash,
        }
        for item in statements.facts
    ]
    return {
        "source_kind": "document_verified",
        "status": "000",
        "identity": request_identity,
        "records": records,
        "mapping_version": MAPPING_VERSION,
        "parser_version": PARSER_VERSION,
        "checks": list(statements.checks),
        "period_end": statements.period_end.isoformat(),
        "unit_multipliers": dict(statements.unit_multipliers),
        "diagnostics": tuple(result.diagnostics),
        **request_identity,
        "raw_document_hash": document_hash,
    }
