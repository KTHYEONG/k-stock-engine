"""Same-filing precision benchmark for filing-document facts.

The benchmark compares the document parser against the same filing's
standard API values. Next-year comparatives are unfit as labels because
restatements rewrite them; correction rows carried into a standard page
from a later filing are excluded by receipt equality.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.data.receipt_catalog import ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import DataRuntime

__all__ = [
    "BENCHMARK_SEED",
    "BenchmarkReport",
    "run_benchmark",
    "select_benchmark_filings",
    "standard_labels",
]

BENCHMARK_SEED: Final = 42

_FACT_SOURCE: Final = "financial_facts"
_DOCUMENT_SOURCE: Final = "dart_documents"

_BS_FACTS: Final = frozenset({"assets", "debt", "equity", "cash"})
_IS_FACTS: Final = frozenset({"sales", "gross_profit", "operating_profit", "net_income"})
_CF_FACTS: Final = frozenset({"operating_cash_flow", "capex", "cash"})

_FINANCIAL_KEYWORDS: Final = ("예치금", "보험", "증권", "은행", "대출", "예수", "캐피탈", "카드", "금융")
_FINANCIAL_ID_KEYWORDS: Final = ("bank", "insur", "securit", "capital", "card", "loan", "deposit")


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Precision of parsed document facts against same-filing standard labels."""

    documents: int
    compared: int
    exact: int
    mismatches: tuple[str, ...]
    withheld_documents: int
    coverage: float

    @property
    def precision(self) -> float:
        """Share of compared pairs that match exactly."""
        if self.compared <= 0:
            return 0.0
        return self.exact / self.compared

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view including the derived precision."""
        return {
            "documents": self.documents,
            "compared": self.compared,
            "exact": self.exact,
            "mismatches": list(self.mismatches),
            "withheld_documents": self.withheld_documents,
            "coverage": self.coverage,
            "precision": self.precision,
        }


def _page_identity(page: Mapping[str, Any]) -> dict[str, Any]:
    raw = page.get("identity")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _page_receipt(page: Mapping[str, Any]) -> str:
    identity = _page_identity(page)
    for key in ("rcept_no", "filing_id"):
        value = str(identity.get(key) or page.get(key) or "").strip()
        if value:
            return value
    return ""


def _page_field(page: Mapping[str, Any], key: str) -> str:
    identity = _page_identity(page)
    return str(identity.get(key) or page.get(key) or "").strip()


def _parse_int_amount(raw: Any) -> int | None:
    """Parse one standard API amount to integer KRW, or ``None`` when absent."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        return None
    text = str(raw).replace(",", "").replace(" ", "").strip()
    if not text or text in {"-", "―"}:
        return None
    negative = text.startswith("-")
    text = text[1:] if negative else text
    if not text.isdigit():
        try:
            as_float = float(text)
        except ValueError:
            return None
        if not as_float.is_integer():
            return None
        value = int(as_float)
        return -value if negative else value
    value = int(text)
    return -value if negative else value


def _thstrm_bases(*, sj_div: str, fact: str, reprt_code: str) -> dict[str, str]:
    """Map amount field name to period basis for one standard row."""
    division = (sj_div or "").strip().upper()
    annual = reprt_code == "11011"
    if division == "BS":
        return {"thstrm_amount": "point_in_time"}
    if division in {"IS", "CIS"}:
        return {
            "thstrm_amount": "annual" if annual else "quarter",
            "thstrm_add_amount": "cumulative",
        }
    if division == "CF":
        return {
            "thstrm_amount": "annual" if annual else "cumulative",
            "thstrm_add_amount": "cumulative",
        }
    if fact in _IS_FACTS:
        return {
            "thstrm_amount": "annual" if annual else "quarter",
            "thstrm_add_amount": "cumulative",
        }
    if fact in _CF_FACTS and fact not in _BS_FACTS:
        return {
            "thstrm_amount": "annual" if annual else "cumulative",
            "thstrm_add_amount": "cumulative",
        }
    if fact in _BS_FACTS:
        return {"thstrm_amount": "point_in_time"}
    return {"thstrm_amount": "annual" if annual else "quarter"}


def standard_labels(page: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    """Clean labels ``(fact, period_basis) -> integer KRW`` from one standard page.

    Only rows whose ``rcept_no`` equals the page identity's receipt are used,
    so values carried over from a later correction never become labels.
    ``SCE`` rows are ignored, and a fact whose rows disagree has no label.
    """
    from src.integrations.dart.accounts import map_standardized_account

    receipt = _page_receipt(page)
    if not receipt:
        return {}
    reprt_code = _page_field(page, "reprt_code")
    records = page.get("records")
    if not isinstance(records, list):
        return {}
    candidates: dict[tuple[str, str], set[int]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        sj_div = str(row.get("sj_div") or "").strip()
        if sj_div.upper() == "SCE":
            continue
        row_receipt = str(row.get("rcept_no") or "").strip()
        if row_receipt != receipt:
            continue
        fact = map_standardized_account(
            account_id=str(row.get("account_id") or row.get("accountId") or ""),
            account_nm=str(row.get("account_nm") or row.get("account") or ""),
        )
        if fact is None:
            continue
        for field, basis in _thstrm_bases(sj_div=sj_div, fact=fact, reprt_code=reprt_code).items():
            amount = _parse_int_amount(row.get(field))
            if amount is None:
                continue
            value = abs(amount) if fact == "capex" else amount
            candidates.setdefault((fact, basis), set()).add(value)
    # 한 계정에 서로 다른 값의 행이 있으면 정답이 모호하므로 비교에서 뺀다.
    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def _is_financial_page(page: Mapping[str, Any]) -> bool:
    identity = _page_identity(page)
    for key in ("is_financial", "financial"):
        if bool(identity.get(key, page.get(key))):
            return True
    records = page.get("records")
    if isinstance(records, list):
        for row in records:
            if not isinstance(row, Mapping):
                continue
            name = f"{row.get('account_nm') or row.get('account') or ''}"
            if any(keyword in name for keyword in _FINANCIAL_KEYWORDS):
                return True
            element = f"{row.get('account_id') or row.get('accountId') or ''}".lower()
            if element and any(keyword in element for keyword in _FINANCIAL_ID_KEYWORDS):
                return True
    return False


def _read_page(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def select_benchmark_filings(catalog: ReceiptCatalog, *, size: int, seed: int) -> tuple[str, ...]:
    """Stratified, deterministic sample of standard-covered fact identities.

    The sample is stratified by report code and by financial versus other
    companies (financial companies only from 2023, when the standard API
    covers them), so it exercises the same layouts as the document path.
    """
    if isinstance(size, bool) or int(size) < 1:
        raise ValueError(f"invalid benchmark size {size!r}: must be a positive integer")
    latest: dict[str, ReceiptIndexEntry] = {}
    for entry in catalog.entries(source=_FACT_SOURCE):
        current = latest.get(entry.natural_key)
        if current is None or entry.retrieved_at > current.retrieved_at:
            latest[entry.natural_key] = entry
    strata: dict[tuple[str, bool], list[str]] = {}
    for natural_key in sorted(latest):
        entry = latest[natural_key]
        page = _read_page(Path(str(entry.payload_path)))
        if page is None:
            continue
        if str(page.get("source_kind") or "") != "opendart_standard":
            continue
        if str(page.get("status") or "") != "000":
            continue
        if not standard_labels(page):
            continue
        reprt_code = _page_field(page, "reprt_code") or natural_key.split(":")[-1]
        biz_year = _page_field(page, "biz_year") or (natural_key.split(":")[1] if len(natural_key.split(":")) > 1 else "")
        financial = _is_financial_page(page)
        if financial:
            try:
                if int(biz_year) < 2023:
                    continue
            except ValueError:
                continue
        strata.setdefault((reprt_code, financial), []).append(natural_key)
    if not strata:
        return ()
    stratum_order = sorted(strata)
    random.Random(int(seed)).shuffle(stratum_order)  # noqa: S311 - deterministic sampling, not cryptography
    per_stratum = {key: list(values) for key, values in strata.items()}
    for key, values in per_stratum.items():
        random.Random(f"{seed}:{key[0]}:{int(key[1])}").shuffle(values)  # noqa: S311 - deterministic sampling
    out: list[str] = []
    positions = dict.fromkeys(stratum_order, 0)
    while len(out) < int(size):
        progressed = False
        for key in stratum_order:
            values = per_stratum[key]
            position = positions[key]
            if position < len(values) and len(out) < int(size):
                out.append(values[position])
                positions[key] = position + 1
                progressed = True
        if not progressed:
            break
    return tuple(out)


def _usable_archive_paths(catalog: ReceiptCatalog) -> dict[str, Path]:
    return {blob.content_hash: Path(str(blob.payload_path)) for blob in catalog.blobs(source=_DOCUMENT_SOURCE)}


def _receipt_archive_index(paths: Mapping[str, Path]) -> dict[str, list[tuple[str, Path]]]:
    index: dict[str, list[tuple[str, Path]]] = {}
    for payload_path in paths.values():
        receipt_path = payload_path.parent / "receipt.json"
        try:
            meta = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, Mapping):
            continue
        rcept_no = str(meta.get("rcept_no") or "").strip()
        if len(rcept_no) != 14 or not rcept_no.isdigit():
            continue
        retrieved = str(meta.get("retrieved_at") or "")
        index.setdefault(rcept_no, []).append((retrieved, payload_path))
    for entries in index.values():
        entries.sort(key=lambda item: item[0])
    return index


def _archive_for_filing(
    *,
    catalog: ReceiptCatalog,
    natural_key: str,
    receipt: str,
    archive_paths: Mapping[str, Path],
    receipt_index: Mapping[str, list[tuple[str, Path]]],
) -> bytes | None:
    for entry in catalog.entries(source=_FACT_SOURCE):
        if entry.natural_key != natural_key:
            continue
        page = _read_page(Path(str(entry.payload_path)))
        if page is None:
            continue
        raw_hash = str(page.get("raw_document_hash") or "")
        if raw_hash and raw_hash in archive_paths:
            try:
                return archive_paths[raw_hash].read_bytes()
            except OSError:  # pragma: no cover - blob removed between plan and read
                continue
    if receipt and receipt in receipt_index:
        for _, payload_path in receipt_index[receipt]:
            try:
                return payload_path.read_bytes()
            except OSError:  # pragma: no cover - blob removed between plan and read
                continue
    return None


def run_benchmark(runtime: DataRuntime, *, filings: Sequence[str]) -> BenchmarkReport:
    """Parse each filing's stored document and compare with its clean standard labels."""
    from src.integrations.dart.document_statements import parse_filing_document

    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    wanted = [str(item).strip() for item in filings if str(item).strip()]
    if not wanted:
        return BenchmarkReport(
            documents=0, compared=0, exact=0, mismatches=(),
            withheld_documents=0, coverage=0.0,
        )
    latest: dict[str, ReceiptIndexEntry] = {}
    for catalog_entry in catalog.entries(source=_FACT_SOURCE):
        if catalog_entry.natural_key not in set(wanted):
            continue
        current = latest.get(catalog_entry.natural_key)
        if current is None or catalog_entry.retrieved_at > current.retrieved_at:
            latest[catalog_entry.natural_key] = catalog_entry
    archive_paths = _usable_archive_paths(catalog)
    receipt_index = _receipt_archive_index(archive_paths)
    documents = 0
    compared = 0
    exact = 0
    mismatches: list[str] = []
    withheld = 0
    labelled_total = 0
    for natural_key in wanted:
        found = latest.get(natural_key)
        if found is None:
            continue
        page = _read_page(Path(str(found.payload_path)))
        if page is None or str(page.get("source_kind") or "") != "opendart_standard":
            continue
        labels = standard_labels(page)
        if not labels:
            continue
        receipt = _page_receipt(page)
        archive = _archive_for_filing(
            catalog=catalog, natural_key=natural_key, receipt=receipt,
            archive_paths=archive_paths, receipt_index=receipt_index,
        )
        if archive is None:
            continue
        parts = natural_key.split(":")
        reprt_code = _page_field(page, "reprt_code") or (parts[2] if len(parts) > 2 else "")
        biz_year = _page_field(page, "biz_year") or (parts[1] if len(parts) > 1 else "")
        parsed = parse_filing_document(bytes(archive), reprt_code=reprt_code, biz_year=biz_year)
        documents += 1
        if parsed.statements is None:
            withheld += 1
            continue
        labelled_total += len(labels)
        emitted: dict[tuple[str, str], int] = {}
        for item in parsed.statements.facts:
            emitted.setdefault((item.fact, item.basis.value), int(item.value))
        for key, parsed_value in emitted.items():
            if key not in labels:
                continue
            label_value = labels[key]
            compared += 1
            if int(parsed_value) == int(label_value):
                exact += 1
            else:
                fact = key[0]
                mismatches.append(f"{natural_key}:{fact}:{int(parsed_value)}:{int(label_value)}")
    mismatches_sorted = tuple(sorted(mismatches))
    coverage = (compared / labelled_total) if labelled_total else 0.0
    return BenchmarkReport(
        documents=documents,
        compared=compared,
        exact=exact,
        mismatches=mismatches_sorted,
        withheld_documents=withheld,
        coverage=coverage,
    )
