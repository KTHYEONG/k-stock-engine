"""Strict re-extraction of current-period statements from legacy filing archives.

A wrong-but-plausible number is worse than a missing one, so every ambiguity
(missing heading, missing unit note, unidentifiable current-period column,
duplicate candidate tables) yields no statement instead of a guess.
"""
from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Final, Literal

__all__ = ["ExtractedStatement", "extract_statements"]

_MAX_MEMBERS: Final = 64
_MAX_MEMBER_BYTES: Final = 32 * 1024 * 1024
_MAX_TOTAL_BYTES: Final = 128 * 1024 * 1024
_CONTEXT_CHARS: Final = 3000
_HEADING_WINDOW: Final = 120

_BS_KEYWORDS: Final = ("재무상태표", "statement of financial position")
_IS_KEYWORDS: Final = ("손익계산서", "포괄손익", "income statement", "profit or loss")

_UNIT_MULTIPLIERS: Final = {"원": 1, "천원": 1000, "백만": 1_000_000, "백만원": 1_000_000}

_PRIOR_WORDS: Final = ("전기", "전년", "전분기", "전반기", "비교", "직전", "작년")
_CURRENT_WORDS: Final = ("당기", "당분", "당반", "금기", "이번")

_LABEL_TO_FACT: Final = {
    "자산총계": "assets",
    "자산총액": "assets",
    "자산계": "assets",
    "부채총계": "debt",
    "부채총액": "debt",
    "부채계": "debt",
    "자본총계": "equity",
    "자본총액": "equity",
    "자본계": "equity",
    "현금및현금성자산": "cash",
    "기말현금및현금성자산": "cash",
    "현금성자산": "cash",
    "매출액": "sales",
    "매출": "sales",
    "영업수익": "sales",
    "매출총이익": "gross_profit",
    "매출총손익": "gross_profit",
    "영업이익": "operating_profit",
    "영업손익": "operating_profit",
    "당기순이익": "net_income",
    "당기순손익": "net_income",
    "분기순이익": "net_income",
    "분기순손익": "net_income",
    "반기순이익": "net_income",
    "반기순손익": "net_income",
    "지배기업소유주지분순이익": "net_income",
    "지배주주지분순이익": "net_income",
}


@dataclass(frozen=True, slots=True)
class ExtractedStatement:
    kind: Literal["BS", "IS"]
    basis: Literal["consolidated", "separate"]
    unit_multiplier: int
    values: Mapping[str, int]  # fact -> integer KRW, current period only
    evidence: str  # statement heading and row labels used


class _TableCollector(HTMLParser):
    """Tolerant table reader recording each table's rows and raw offset."""

    def __init__(self, line_offsets: list[int]) -> None:
        super().__init__(convert_charrefs=True)
        self._line_offsets = line_offsets
        self.tables: list[dict[str, object]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def _offset(self) -> int:
        lineno, col = self.getpos()
        idx = max(0, min(lineno - 1, len(self._line_offsets) - 1))
        return self._line_offsets[idx] + col

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        name = tag.upper()
        if name == "TABLE":
            self._current_table = []
            self.tables.append({"offset": self._offset(), "rows": self._current_table})
        elif name == "TR":
            if self._current_table is not None:
                self._current_row = []
        elif name in {"TD", "TH"}:
            if self._current_row is not None:
                self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        name = tag.upper()
        if name in {"TD", "TH"}:
            if self._current_row is not None and self._current_cell is not None:
                self._current_row.append(" ".join("".join(self._current_cell).split()))
                self._current_cell = None
        elif name == "TR":
            if self._current_table is not None and self._current_row is not None:
                self._current_table.append(self._current_row)
                self._current_row = None
        elif name == "TABLE":
            self._current_table = None
            self._current_row = None
            self._current_cell = None


def _decode_member(raw: bytes) -> str | None:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for codec in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(codec)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _normalize_label(label: str) -> str:
    text = re.sub(r"\([^)]*\)", "", label)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"주\d.*$", "", text)
    text = re.sub(r"주석.*$", "", text)
    return text.strip("·•-*[] ")


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    lowered = haystack.lower()
    return any(n in haystack or n in lowered for n in needles)


def _detect_kind_and_basis(heading: str) -> tuple[str, str] | None:
    is_bs = _contains_any(heading, _BS_KEYWORDS)
    is_is = _contains_any(heading, _IS_KEYWORDS)
    if is_bs == is_is:
        return None
    kind = "BS" if is_bs else "IS"
    has_con = _contains_any(heading, ("연결", "consolidated"))
    has_sep = _contains_any(heading, ("별도", "개별", "separate"))
    if has_con == has_sep:
        return None
    return (kind, "consolidated" if has_con else "separate")


def _detect_unit_multiplier(context: str, table_text: str) -> int | None:
    found: set[int] = set()
    for match in re.finditer(r"단위\s*[:：]?\s*([^\s)<\]]{1,10})", context + "\n" + table_text):  # noqa: RUF001
        token = match.group(1).strip().rstrip("),.]")
        multiplier = _UNIT_MULTIPLIERS.get(token)
        if multiplier is None:
            return None
        found.add(multiplier)
    lowered = (context + "\n" + table_text).lower()
    if re.search(r"in\s+thousands", lowered):
        found.add(1000)
    if re.search(r"in\s+millions", lowered):
        found.add(1_000_000)
    if len(found) != 1:
        return None
    return next(iter(found))


def _classify_header(text: str) -> tuple[str, tuple[int, ...]]:
    if any(w in text for w in _PRIOR_WORDS):
        return ("prior", ())
    if any(w in text for w in _CURRENT_WORDS):
        return ("current", ())
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", text)]
    if years:
        months = [int(m) for m in re.findall(r"(?:\.|년\s*)(\d{1,2})(?:\.|월|일)?", text)]
        month = max(months) if months else 12
        return ("dated", (max(years), month))
    period_no = re.findall(r"제\s*(\d+)\s*기", text)
    if period_no:
        return ("ordinal", (max(int(n) for n in period_no),))
    return ("none", ())


def _select_current_column(header: list[str]) -> int | None:
    """Return the absolute column index of the current-period amount column."""
    if len(header) < 2:
        return None
    kinds: list[str] = []
    keys: list[tuple[int, ...]] = []
    for cell in header[1:]:
        kind, key = _classify_header(cell)
        kinds.append(kind)
        keys.append(key)
    currents = [i for i, k in enumerate(kinds) if k == "current"]
    priors = [i for i, k in enumerate(kinds) if k == "prior"]
    if currents:
        if len(currents) != 1:
            return None
        return currents[0] + 1
    if priors:
        return None
    dated = [(keys[i], i) for i, k in enumerate(kinds) if k in ("dated", "ordinal")]
    if dated:
        best = max(dated, key=lambda item: item[0])
        if sum(1 for key, _ in dated if key == best[0]) != 1:
            return None
        return best[1] + 1
    if len(kinds) == 1:
        return 1
    return None


def _parse_int_amount(text: str) -> int | None:
    token = text.strip().replace(" ", "").replace("원", "").replace("%", "")
    if not token:
        return None
    negative = False
    if token.startswith("(") and token.endswith(")"):
        negative = True
        token = token[1:-1]
    token = token.lstrip("+-−–")  # noqa: RUF001
    if text.strip().startswith(("-", "−", "–", "△")) and not negative:  # noqa: RUF001
        negative = True
    if token.startswith("(") or token.endswith(")"):
        return None
    match = re.fullmatch(r"\d[\d,]*(\.\d+)?", token)
    if match is None:
        return None
    try:
        number = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None
    if number != number.to_integral_value():
        return None
    value = int(number)
    return -value if negative else value


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]*>", " ", text)


def _process_table(
    rows: list[list[str]], unit: int, column: int
) -> dict[str, int] | None:
    facts: dict[str, int] = {}
    for row in rows:
        if not row or not row[0].strip():
            continue
        fact = _LABEL_TO_FACT.get(_normalize_label(row[0]))
        if fact is None:
            continue
        if fact in facts:
            return None
        if column >= len(row):
            continue
        amount = _parse_int_amount(row[column])
        if amount is None:
            continue
        facts[fact] = amount * unit
    if not facts:
        return None
    return facts


def _extract_from_member(text: str) -> list[tuple[tuple[str, str], ExtractedStatement]]:
    line_offsets = [0]
    line_offsets.extend(match.end() for match in re.finditer("\n", text))
    parser = _TableCollector(line_offsets)
    try:
        parser.feed(text)
    except Exception:
        return []
    found: list[tuple[tuple[str, str], ExtractedStatement]] = []
    for table in parser.tables:
        offset = int(table["offset"])  # type: ignore[arg-type]
        rows = [row for row in table["rows"] if any(cell.strip() for cell in row)]  # type: ignore[union-attr]
        if not rows:
            continue
        preceding = text[max(0, offset - _CONTEXT_CHARS) : offset]
        plain_preceding = _strip_tags(preceding)
        heading: str | None = None
        data_start = 0
        kind_matches: list[re.Match[str]] = []
        for keyword in (*_BS_KEYWORDS, *_IS_KEYWORDS):
            pattern = re.escape(keyword)
            kind_matches.extend(re.finditer(pattern, plain_preceding, flags=re.IGNORECASE))
        if kind_matches:
            ordered_matches = sorted(kind_matches, key=lambda m: m.start())
            last = ordered_matches[-1]
            previous_end = (
                max(m.end() for m in ordered_matches[:-1])
                if len(ordered_matches) > 1
                else 0
            )
            start = max(previous_end, last.start() - _HEADING_WINDOW)
            end = min(len(plain_preceding), last.end() + _HEADING_WINDOW)
            heading = plain_preceding[start:end]
        else:
            table_text = " ".join(" ".join(row) for row in rows[:3])
            detected = _detect_kind_and_basis(table_text)
            if detected is None:
                continue
            heading = table_text
            data_start = next(
                (
                    idx + 1
                    for idx, row in enumerate(rows[:3])
                    if _detect_kind_and_basis(" ".join(row)) is not None
                ),
                1,
            )
        detected = _detect_kind_and_basis(heading)
        if detected is None:
            continue
        kind, basis = detected
        table_text = " ".join(" ".join(row) for row in rows)
        unit = _detect_unit_multiplier(plain_preceding[-1500:], table_text)
        if unit is None:
            continue
        data_rows = rows[data_start:]
        if not data_rows:
            continue
        column = _select_current_column(data_rows[0])
        if column is None:
            continue
        facts = _process_table(data_rows[1:], unit, column)
        if facts is None:
            continue
        evidence = f"{' '.join(heading.split())[:200]} | {','.join(sorted(facts))}"
        statement = ExtractedStatement(
            kind=kind,  # type: ignore[arg-type]
            basis=basis,  # type: ignore[arg-type]
            unit_multiplier=unit,
            values=dict(sorted(facts.items())),
            evidence=evidence,
        )
        found.append(((kind, basis), statement))
    return found


def extract_statements(archive_bytes: bytes) -> tuple[ExtractedStatement, ...]:
    """Extract current-period statement values from a legacy filing archive.

    Legacy filings have no machine-readable statement API, so values are read from
    HTML tables. A wrong-but-plausible number is worse than a missing one, so the
    extractor returns nothing for anything it cannot pin down unambiguously.

    Args:
        archive_bytes: The ``document.xml`` ZIP for one receipt.

    Returns:
        Zero or more statements; each fact appears at most once per (kind, basis).
    """
    try:
        if not archive_bytes:
            return ()
        try:
            reader = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except Exception:
            return ()
        with reader:
            infos = [info for info in reader.infolist() if not info.is_dir()]
            if len(infos) > _MAX_MEMBERS:
                return ()
            total = sum(info.file_size for info in infos)
            if total > _MAX_TOTAL_BYTES:
                return ()
            members: list[str] = []
            for info in infos:
                if info.file_size > _MAX_MEMBER_BYTES:
                    continue
                try:
                    raw = reader.read(info.filename)
                except Exception:  # noqa: S112 - unreadable member is skipped; caller fails closed on no statements
                    continue
                if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                    return ()
                decoded = _decode_member(raw)
                if decoded is None or "<" not in decoded:
                    continue
                members.append(decoded)
        grouped: dict[tuple[str, str], list[ExtractedStatement]] = {}
        for member in members:
            for pair, statement in _extract_from_member(member):
                grouped.setdefault(pair, []).append(statement)
        ordered: list[ExtractedStatement] = []
        for pair in sorted(grouped):
            candidates = grouped[pair]
            if len(candidates) != 1:
                continue
            ordered.append(candidates[0])
        return tuple(ordered)
    except Exception:
        return ()
