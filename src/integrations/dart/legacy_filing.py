"""Legacy DART document.xml archive parsing (offline, defensive)."""
from __future__ import annotations

import io
import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree

MAPPING_VERSION = "dart-fact-map-v1"

_LABEL_TO_FACT: dict[str, str] = {
    "매출액": "sales",
    "매출": "sales",
    "영업수익": "sales",
    "수익": "sales",
    "매출총이익": "gross_profit",
    "매출총손익": "gross_profit",
    "영업이익": "operating_profit",
    "영업손익": "operating_profit",
    "당기순이익": "net_income",
    "당기순손익": "net_income",
    "분기순이익": "net_income",
    "분기순손익": "net_income",
    "반기순이익": "net_income",
    "지배기업소유주지분순이익": "net_income",
    "자산총계": "assets",
    "자산총액": "assets",
    "자산계": "assets",
    "자본총계": "equity",
    "자본총액": "equity",
    "자본계": "equity",
    "부채총계": "debt",
    "부채총액": "debt",
    "부채계": "debt",
    "현금및현금성자산": "cash",
    "기말현금및현금성자산": "cash",
    "영업활동현금흐름": "operating_cash_flow",
    "영업활동으로인한현금흐름": "operating_cash_flow",
    "영업활동으로인한순현금흐름": "operating_cash_flow",
    "유형자산취득": "capex",
    "자본적지출": "capex",
    "설비투자": "capex",
    "유형자산의취득": "capex",
    "유무형자산취득": "capex",
}

_ID_TO_FACT: dict[str, str] = {
    "ifrs-full_Revenue": "sales",
    "ifrs-full_RevenueFromContractsWithCustomers": "sales",
    "ifrs-full_SalesRevenue": "sales",
    "ifrs-full_GrossProfit": "gross_profit",
    "ifrs-full_OperatingProfit": "operating_profit",
    "ifrs-full_OperatingIncome": "operating_profit",
    "ifrs-full_ProfitLoss": "net_income",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "net_income",
    "ifrs-full_ProfitLossAttributableToOwners": "net_income",
    "ifrs-full_ComprehensiveIncome": "net_income",
    "ifrs-full_Assets": "assets",
    "ifrs-full_Equity": "equity",
    "ifrs-full_EquityAttributableToOwnersOfParent": "equity",
    "ifrs-full_Liabilities": "debt",
    "ifrs-full_CashAndCashEquivalents": "cash",
    "ifrs-full_CashFlowsFromOperatingActivities": "operating_cash_flow",
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
    "ifrs-full_NetCashFlowsFromOperatingActivities": "operating_cash_flow",
    "ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "ifrs-full_PaymentsToAcquireIntangibleAssets": "capex",
    "dart_OperatingIncome": "operating_profit",
    "dart_Revenue": "sales",
}

_REQUIRED_FACTS = frozenset(
    {"gross_profit", "net_income", "operating_cash_flow", "assets", "equity", "operating_profit", "sales"}
)
_OPTIONAL_FACTS = frozenset({"cash", "debt", "capex"})
_BALANCE_FACTS = frozenset({"assets", "equity", "cash", "debt"})
_FLOW_FACTS = frozenset(
    {"sales", "gross_profit", "operating_profit", "net_income", "operating_cash_flow", "capex"}
)

_MAX_MEMBERS = 32
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_QUARTER_MARK_RE = re.compile(r"(3개월|분기|누계|quarter|cumulative)", re.IGNORECASE)
_NON_XML_TAG_START = re.compile(r"<(?=[^A-Za-z!/ ?])")
_PAREN_NUMBER_RE = re.compile(r"\(?\s*-?\d[\d,]*\.?\d*\s*\)?")
_ROMAN_PREFIX_CHARS = "IVXLCDM\u2160\u2161\u2162\u2163\u2164\u2165\u2166\u2167\u2168\u2169\u216a\u216b"


@dataclass(frozen=True, slots=True)
class LegacyFilingParseResult:
    records: tuple[dict[str, Any], ...]
    status: str
    diagnostics: tuple[str, ...]
    document_hash: str = ""
    mapping_version: str = MAPPING_VERSION


def _is_unsafe_name(name: str) -> bool:
    if not name:
        return True
    # DART document.xml commonly prefixes its single member with '/'.  It is
    # safe here because members are never extracted to the filesystem.
    normalized = name.replace("\\", "/").lstrip("/")
    if not normalized:
        return True
    if ".." in normalized.split("/"):
        return True
    return "../" in normalized or normalized.endswith("/..")


def _decode_member(raw: bytes) -> str | None:
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw[3:].decode("utf-8")
        except UnicodeDecodeError:
            return None
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            return None
    head = raw[:1024].decode("ascii", errors="ignore")
    m = re.search(r"encoding\s*=\s*['\"]([^'\"]+)['\"]", head)
    if m:
        enc = m.group(1).strip().lower()
        normalized = enc.replace("_", "-")
        if normalized in ("utf8", "utf-8"):
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                pass
        if normalized in ("cp949", "euc-kr", "ks-c-5601", "windows-949", "uhc"):
            try:
                codec = "cp949" if normalized in ("cp949", "windows-949", "uhc") else "euc-kr"
                return raw.decode(codec)
            except (UnicodeDecodeError, LookupError):
                pass
    for codec in ("cp949", "euc-kr", "utf-8"):
        try:
            return raw.decode(codec)
        except UnicodeDecodeError:
            continue
    return None


def _fiscal_period_from_identity(identity: Mapping[str, str]) -> str:
    for key in ("fiscal_period", "fiscalPeriod"):
        val = str(identity.get(key, "") or "").strip()
        if val:
            return val
    biz_year = str(identity.get("biz_year") or identity.get("bsns_year") or "").strip()
    reprt = str(identity.get("reprt_code") or identity.get("report_code") or "").strip()
    if len(biz_year) == 4 and reprt:
        mapping = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Q4"}
        suffix = mapping.get(reprt)
        if suffix:
            return f"{biz_year}{suffix}"
        return f"{biz_year}Q4"
    return f"{biz_year}Q4" if biz_year else "unknown"


def map_standardized_account(*, account_id: str = "", account_nm: str = "") -> str | None:
    aid = (account_id or "").strip()
    if aid and aid in _ID_TO_FACT:
        return _ID_TO_FACT[aid]
    label = re.sub(r"\([^)]*\)", "", (account_nm or "")).replace(" ", "").strip()
    if label and label in _LABEL_TO_FACT:
        return _LABEL_TO_FACT[label]
    return None


def parse_legacy_filing_archive(
    *, archive_bytes: bytes, identity: Mapping[str, str], document_hash: str
) -> LegacyFilingParseResult:
    filing_id = str(identity.get("filing_id") or identity.get("rcept_no") or "").strip()
    company_id = str(identity.get("corp_code") or identity.get("company_id") or "").strip()
    fiscal_period = _fiscal_period_from_identity(identity)
    if not archive_bytes:
        return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("empty_archive",), document_hash=document_hash)
    try:
        buf = io.BytesIO(archive_bytes)
        with zipfile.ZipFile(buf) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile:
        return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("bad_zip",), document_hash=document_hash)
    except Exception:
        return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("bad_zip",), document_hash=document_hash)
    if len(infos) > _MAX_MEMBERS:
        return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("too_many_members",), document_hash=document_hash)
    seen: set[str] = set()
    total = 0
    for info in infos:
        name = info.filename
        if name in seen:
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("duplicate_member",), document_hash=document_hash)
        seen.add(name)
        if info.is_dir():
            continue
        if _is_unsafe_name(name):
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("unsafe_member_path",), document_hash=document_hash)
        is_symlink = ((info.external_attr >> 16) & 0o170000) == 0o120000
        if is_symlink:
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("symlink_member",), document_hash=document_hash)
        if info.flag_bits & 0x1:
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("encrypted_member",), document_hash=document_hash)
        if info.file_size > _MAX_MEMBER_BYTES:
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("member_too_large",), document_hash=document_hash)
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("expanded_too_large",), document_hash=document_hash)
    # Re-open to stream one member at a time (O(bytes + rows)).
    records: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    try:
        buf2 = io.BytesIO(archive_bytes)
        with zipfile.ZipFile(buf2) as zf2:
            for info in zf2.infolist():
                if info.is_dir():
                    continue
                raw = zf2.read(info.filename)
                if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                    return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("unsafe_xml_declaration",), document_hash=document_hash)
                text = _decode_member(raw)
                if text is None:
                    return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("extraction_failed",), document_hash=document_hash)
                text = text.replace("&cr;", "\r").replace("&nbsp;", " ")
                # Legacy DART emits a few Korean pseudo-tags such as
                # '<당분기말>' without XML closing tags.  Treat those tokens
                # as text while retaining the surrounding table structure.
                text = _NON_XML_TAG_START.sub("&lt;", text)
                if "<" not in text:
                    return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("extraction_failed",), document_hash=document_hash)
                try:
                    root = ElementTree.fromstring(text)  # noqa: S314
                except ElementTree.ParseError:
                    page_records, page_diags = _extract_legacy_tables(
                        text,
                        company_id=company_id,
                        filing_id=filing_id,
                        fiscal_period=fiscal_period,
                        document_hash=document_hash,
                    )
                    if page_records:
                        records.extend(page_records)
                        diagnostics.extend(page_diags)
                        continue
                    return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("extraction_failed",), document_hash=document_hash)
                page_records, page_diags = _extract_records(
                    root,
                    text,
                    company_id=company_id,
                    filing_id=filing_id,
                    fiscal_period=fiscal_period,
                    document_hash=document_hash,
                )
                records.extend(page_records)
                diagnostics.extend(page_diags)
    except zipfile.BadZipFile:
        return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("extraction_failed",), document_hash=document_hash)
    if not records:
        if any("ambiguous" in d for d in diagnostics):
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=tuple(diagnostics) if diagnostics else ("ambiguous",), document_hash=document_hash)
        if diagnostics:
            return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=tuple(diagnostics), document_hash=document_hash)
        return LegacyFilingParseResult(records=(), status="extraction_failed", diagnostics=("extraction_failed",), document_hash=document_hash)
    return LegacyFilingParseResult(records=tuple(records), status="ok", diagnostics=tuple(diagnostics), document_hash=document_hash)


class _LegacyTableParser(HTMLParser):
    """Tolerant table reader for DART's non-XML pseudo-tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_index = -1
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.tables: dict[int, list[list[str]]] = {}

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        name = tag.upper()
        if name == "TABLE":
            self.table_index += 1
            self.tables[self.table_index] = []
        elif name == "TR":
            self.current_row = []
        elif name in {"TD", "TH"} and self.current_row is not None:
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        name = tag.upper()
        if name in {"TD", "TH"} and self.current_row is not None and self.current_cell is not None:
            self.current_row.append(" ".join("".join(self.current_cell).split()))
            self.current_cell = None
        elif name == "TR" and self.current_row is not None:
            self.tables.setdefault(self.table_index, []).append(self.current_row)
            self.current_row = None


def _parse_legacy_amount(value: str) -> float | None:
    match = _PAREN_NUMBER_RE.search(value.replace(" ", ""))
    if match is None:
        return None
    token = match.group(0).replace(",", "").strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()")
    try:
        number = float(token)
    except ValueError:
        return None
    return -number if negative else number


def _extract_legacy_tables(
    text: str,
    *,
    company_id: str,
    filing_id: str,
    fiscal_period: str,
    document_hash: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    parser = _LegacyTableParser()
    parser.feed(text)
    candidates: dict[str, list[tuple[int, int, float]]] = {}
    for table_index, rows in parser.tables.items():
        table_facts: set[str] = set()
        for row in rows:
            if not row:
                continue
            label = row[0].strip().lstrip("[(")
            label = re.sub(rf"^(?:[{_ROMAN_PREFIX_CHARS}]+\.?|\d+\))\s*", "", label)
            label = label.strip(" ]")
            label = label.replace(" ", "")
            fact = map_standardized_account(account_nm=label)
            if fact is None:
                continue
            values = [value for cell in row[1:] if (value := _parse_legacy_amount(cell)) is not None]
            if not values:
                continue
            table_facts.add(fact)
            candidates.setdefault(fact, []).append((len(table_facts), table_index, values[0]))
    if not candidates:
        return [], ["extraction_failed"]
    # Prefer a coherent statement table (most mapped rows), then the full-KRW
    # rendering over the summary-thousands rendering.
    selected: dict[str, tuple[int, int, float]] = {}
    for fact, fact_candidates in candidates.items():
        selected[fact] = max(
            fact_candidates,
            key=lambda item: (item[0], abs(item[2]), -item[1]),
        )
    diagnostics: list[str] = []
    records: list[dict[str, Any]] = []
    for fact, (_score, _table, value) in selected.items():
        if fact in _FLOW_FACTS and not _QUARTER_MARK_RE.search(text):
            diagnostics.append(f"ambiguous_period_basis:{fact}")
            continue
        records.append(
            {
                "company_id": company_id,
                "fiscal_period": fiscal_period,
                "filing_id": filing_id,
                "fact": fact,
                "value": value,
                "unit": "KRW",
                "consolidated": True,
                "restatement_id": "r0",
                "source_kind": "legacy_document",
                "mapping_version": MAPPING_VERSION,
                "raw_document_hash": document_hash,
            }
        )
    return records, diagnostics


def _extract_records(
    root: ElementTree.Element,
    text: str,
    *,
    company_id: str,
    filing_id: str,
    fiscal_period: str,
    document_hash: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: dict[str, list[tuple[float, str]]] = {}
    diags: list[str] = []
    has_period_basis = bool(_QUARTER_MARK_RE.search(text))
    # Walk elements for structured pairs.
    for elem in root.iter():
        aid = ""
        anm = ""
        val_text = ""
        unit = ""
        for child in elem:
            tag = str(child.tag).lower()
            ctext = (child.text or "").strip()
            if ("account_id" in tag or tag.endswith("id")) and ctext and len(ctext) < 128:
                aid = ctext
            if ("account_nm" in tag or "accountnm" in tag or "label" in tag or "nm" in tag) and ctext and len(ctext) < 64:
                anm = ctext
            if tag in ("amount", "value", "amt", "thstrm_amount"):
                val_text = ctext
            if tag == "unit":
                unit = ctext
        # Also check element text directly for label + sibling number.
        if not anm:
            own = (elem.text or "").strip()
            if own in _LABEL_TO_FACT and len(own) < 32:
                anm = own
                # Look for number in tail or following siblings.
                tail = (elem.tail or "") + "".join((s.tail or "") for s in elem)
                m = _NUMBER_RE.search(tail)
                if m:
                    val_text = m.group(0)
        if not anm and not aid:
            continue
        fact = map_standardized_account(account_id=aid, account_nm=anm)
        if fact is None:
            if anm or aid:
                diags.append(f"unknown_account:{anm or aid}")
            continue
        if not val_text:
            # Try regex around this element's serialized text.
            try:
                blob = ElementTree.tostring(elem, encoding="unicode")
            except Exception:
                blob = ""
            m = _NUMBER_RE.search(blob)
            if not m:
                diags.append(f"unitless:{fact}")
                continue
            val_text = m.group(0)
        cleaned = val_text.replace(",", "").strip()
        try:
            num = float(cleaned)
        except ValueError:
            diags.append(f"non_finite:{fact}")
            continue
        if not math.isfinite(num):
            diags.append(f"non_finite:{fact}")
            continue
        if not unit:
            # Look for unit markers in blob.
            try:
                blob_u = ElementTree.tostring(elem, encoding="unicode")
            except Exception:
                blob_u = ""
            if "KRW" in blob_u or "원" in blob_u:
                unit = "KRW"
            else:
                # Fallback: check page text for KRW; else unitless.
                if "KRW" in text or "원" in text:
                    unit = "KRW"
                else:
                    diags.append(f"unitless:{fact}")
                    continue
        if unit != "KRW":
            diags.append(f"unitless:{fact}")
            continue
        candidates.setdefault(fact, []).append((num, unit))
    # Fallback regex scan when structured walk found nothing.
    if not candidates:
        for label, fact in _LABEL_TO_FACT.items():
            idxs = [m.start() for m in re.finditer(re.escape(label), text)]
            if not idxs:
                continue
            vals: list[tuple[float, str]] = []
            for pos in idxs:
                window = text[pos : pos + 400]
                m = _NUMBER_RE.search(window)
                if not m:
                    continue
                try:
                    num = float(m.group(0).replace(",", ""))
                except ValueError:
                    continue
                if not math.isfinite(num):
                    continue
                unit = "KRW" if ("KRW" in window or "원" in window or "KRW" in text) else ""
                if not unit:
                    continue
                vals.append((num, unit))
            if vals:
                candidates.setdefault(fact, []).extend(vals)
        if not candidates:
            return [], ["extraction_failed"]
    out: list[dict[str, Any]] = []
    for fact, vals in candidates.items():
        if len(vals) > 1:
            first = vals[0][0]
            if any(v != first for v, _ in vals):
                diags.append("ambiguous")
                diags.append(f"ambiguous:{fact}")
                continue
            diags.append("ambiguous")
            diags.append(f"ambiguous:{fact}")
            continue
        if fact in _FLOW_FACTS and not has_period_basis:
            # Ambiguous period basis: emit no flow fact.
            diags.append(f"ambiguous_period_basis:{fact}")
            continue
        num, unit = vals[0]
        out.append(
            {
                "company_id": company_id,
                "fiscal_period": fiscal_period,
                "filing_id": filing_id,
                "fact": fact,
                "value": num,
                "unit": unit,
                "consolidated": True,
                "restatement_id": "r0",
                "source_kind": "legacy_document",
                "mapping_version": MAPPING_VERSION,
                "raw_document_hash": document_hash,
            }
        )
    return out, diags
