"""Invariant scenarios for the same-filing document benchmark."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
RCEPT = "20240315001111"
CORP = "00126380"


def _standard_page(*, rcept_no: str = RCEPT, reprt_code: str = "11011", biz_year: str = "2023",
                   extra_rows: tuple[dict, ...] = ()) -> dict:
    identity = {
        "corp_code": CORP, "filing_id": rcept_no, "rcept_no": rcept_no,
        "biz_year": biz_year, "reprt_code": reprt_code, "fs_div": "CFS",
        "published_at": "2024-03-15", "ticker": "", "fiscal_period": f"{biz_year}Q4",
    }
    return {
        "source_kind": "opendart_standard",
        "status": "000",
        "identity": dict(identity),
        "records": list(extra_rows),
        "mapping_version": "dart-fact-map-v1",
        "diagnostics": (),
        "raw_document_hash": None,
        **identity,
    }


def _row(*, fact_id: str, fact_nm: str, sj_div: str, amount: str, rcept_no: str = RCEPT,
         add_amount: str | None = None) -> dict:
    row: dict[str, object] = {
        "rcept_no": rcept_no, "bsns_year": "2023", "reprt_code": "11011",
        "corp_code": CORP, "fs_div": "CFS", "sj_div": sj_div,
        "account_id": fact_id, "account_nm": fact_nm, "thstrm_amount": amount,
    }
    if add_amount is not None:
        row["thstrm_add_amount"] = add_amount
    return row


def test_standard_labels_excludes_correction_rows() -> None:
    """Correction rows from a later filing never become labels."""
    from src.data.dart_document_benchmark import standard_labels

    page = _standard_page(extra_rows=(
        _row(fact_id="ifrs-full_Revenue", fact_nm="매출액", sj_div="IS",
             amount="9,999", rcept_no="20240415009999"),
        _row(fact_id="ifrs-full_Assets", fact_nm="자산총계", sj_div="BS", amount="1,000"),
    ))
    labels = standard_labels(page)
    assert ("sales", "annual") not in labels
    assert labels[("assets", "point_in_time")] == 1000


def test_standard_labels_drops_disagreeing_rows() -> None:
    """A fact whose same-filing rows disagree has no label; agreeing duplicates keep one."""
    from src.data.dart_document_benchmark import standard_labels

    page = _standard_page(extra_rows=(
        _row(fact_id="", fact_nm="영업수익", sj_div="IS", amount="500"),
        _row(fact_id="", fact_nm="매출액", sj_div="IS", amount="300"),
        _row(fact_id="ifrs-full_Assets", fact_nm="자산총계", sj_div="BS", amount="1,000"),
        _row(fact_id="", fact_nm="자산총계", sj_div="BS", amount="1,000"),
    ))
    labels = standard_labels(page)
    assert ("sales", "annual") not in labels
    assert labels[("assets", "point_in_time")] == 1000


def _stub_catalog(pages: dict[str, dict], directory: Path):
    import hashlib as _hashlib

    class _Entry:
        def __init__(self, natural_key: str, payload_path: Path, content_hash: str) -> None:
            self.natural_key = natural_key
            self.retrieved_at = NOW
            self.payload_path = payload_path
            self.content_hash = content_hash

    class _Catalog:
        def __init__(self) -> None:
            self.root = directory
            self._entries = []
            for key, page in pages.items():
                path = directory / f"{key.replace(':', '_')}.json"
                raw = json.dumps(page, ensure_ascii=False).encode("utf-8")
                path.write_bytes(raw)
                self._entries.append(
                    _Entry(key, path, _hashlib.sha256(raw).hexdigest())
                )

        def entries(self, *, source: str):
            assert source == "financial_facts"
            return iter(self._entries)

    return _Catalog()


def _ten_fact_rows(rcept_no: str = RCEPT) -> tuple[dict, ...]:
    return (
        _row(fact_id="ifrs-full_Revenue", fact_nm="매출액", sj_div="IS", amount="1,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_GrossProfit", fact_nm="매출총이익", sj_div="IS", amount="2,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_OperatingProfit", fact_nm="영업이익", sj_div="IS", amount="3,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_ProfitLoss", fact_nm="당기순이익", sj_div="IS", amount="4,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_Assets", fact_nm="자산총계", sj_div="BS", amount="5,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_Liabilities", fact_nm="부채총계", sj_div="BS", amount="2,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_Equity", fact_nm="자본총계", sj_div="BS", amount="3,000", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_CashAndCashEquivalents", fact_nm="현금및현금성자산",
             sj_div="BS", amount="500", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_CashFlowsFromOperatingActivities", fact_nm="영업활동현금흐름",
             sj_div="CF", amount="600", rcept_no=rcept_no),
        _row(fact_id="ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment",
             fact_nm="유형자산취득", sj_div="CF", amount="700", rcept_no=rcept_no),
    )


def _stratified_pages() -> dict[str, dict]:
    pages: dict[str, dict] = {}
    for reprt_code in ("11011", "11012", "11013", "11014"):
        for financial in (False, True):
            key = f"{CORP}:2023:{reprt_code}{'F' if financial else ''}"
            rows = [dict(row, reprt_code=reprt_code) for row in _ten_fact_rows()]
            if financial:
                rows = [*rows, {"rcept_no": RCEPT, "sj_div": "BS",
                                "account_id": "x", "account_nm": "보험계약부채",
                                "thstrm_amount": "1"}, 0]
            page = _standard_page(reprt_code=reprt_code, extra_rows=tuple(rows))
            if financial:
                page = {**page, "is_financial": True}
            pages[key] = page
    content_rows = [dict(row, reprt_code="11012") for row in _ten_fact_rows()]
    content_rows.append({"rcept_no": RCEPT, "sj_div": "BS", "account_id": "x",
                         "account_nm": "보험계약부채", "thstrm_amount": "1"})
    pages[f"{CORP}:2024:11012C"] = _standard_page(
        reprt_code="11012", biz_year="2024", extra_rows=(0, *tuple(content_rows)))
    element_rows = [dict(row, reprt_code="11013") for row in _ten_fact_rows()]
    element_rows.append({"rcept_no": RCEPT, "sj_div": "BS", "account_id": "ifrs-full_BankDeposits",
                         "account_nm": "미지정계정", "thstrm_amount": "1"})
    pages[f"{CORP}:2024:11013C"] = _standard_page(
        reprt_code="11013", biz_year="2024", extra_rows=tuple(element_rows))
    old_key = f"{CORP}:2020:11011F"
    old_rows = [dict(row, reprt_code="11011") for row in _ten_fact_rows()]
    old_page = _standard_page(reprt_code="11011", biz_year="2020", extra_rows=tuple(old_rows))
    old_page["biz_year"] = "2020"
    old_page["identity"] = {**old_page["identity"], "biz_year": "2020"}
    old_page["is_financial"] = True
    pages[old_key] = old_page
    return pages


def test_select_benchmark_filings_is_deterministic_and_stratified(tmp_path: Path) -> None:
    """The same catalog and seed give identical filings with every stratum present."""
    from src.data.dart_document_benchmark import select_benchmark_filings

    pages = _stratified_pages()
    catalog = _stub_catalog(pages, tmp_path)
    first = select_benchmark_filings(catalog, size=8, seed=7)
    second = select_benchmark_filings(catalog, size=8, seed=7)
    assert first == second
    assert f"{CORP}:2020:11011F" not in select_benchmark_filings(catalog, size=100, seed=7)
    strata = set()
    by_key = dict(pages)
    for key in first:
        page = by_key[key]
        from src.data.dart_document_benchmark import _is_financial_page as _is_financial

        strata.add((str(page["identity"]["reprt_code"]), _is_financial(page)))
    assert len(strata) == 8


def _publish_standard(bronze_root: Path, catalog, *, natural_key: str, page: dict,
                      retrieved_at: datetime = NOW, patch: dict | None = None):
    import hashlib as _hashlib
    import json as _json

    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    if patch:
        page = {**page, **patch}
    raw = _json.dumps(page, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = _hashlib.sha256(raw).hexdigest()
    payload_dir = bronze_root / "financial_facts" / digest
    payload_dir.mkdir(parents=True, exist_ok=True)
    (payload_dir / "payload.json").write_bytes(raw)
    catalog.publish(
        [ReceiptIndexEntry(source="financial_facts", natural_key=natural_key, as_of=date(2024, 3, 15),
                           fiscal_period="2023Q4", status=EvidenceStatus.SUCCESS, content_hash=digest,
                           retrieved_at=retrieved_at, payload_path=payload_dir / "payload.json")],
        blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.FINANCIAL_FACTS, source="financial_facts",
                         usable=True, unusable_reason=None, retrieved_at=retrieved_at,
                         payload_path=payload_dir / "payload.json")],
    )


def _runtime(bronze_root: Path):
    return SimpleNamespace(workspace=SimpleNamespace(bronze_root=bronze_root))


def _ten_labels() -> dict[tuple[str, str], int]:
    return {
        ("sales", "annual"): 1000, ("gross_profit", "annual"): 2000,
        ("operating_profit", "annual"): 3000, ("net_income", "annual"): 4000,
        ("assets", "point_in_time"): 5000, ("debt", "point_in_time"): 2000,
        ("equity", "point_in_time"): 3000, ("cash", "point_in_time"): 500,
        ("operating_cash_flow", "annual"): 600, ("capex", "annual"): 700,
    }


def _statements_for(labels: dict[tuple[str, str], int]):
    from src.integrations.dart.document_statements import PeriodBasis, StatementFact, VerifiedStatements

    facts = tuple(
        StatementFact(fact=fact, value=value, basis=PeriodBasis(basis), label=fact)
        for (fact, basis), value in sorted(labels.items())
    )
    return VerifiedStatements(consolidated=True, period_end=date(2023, 12, 31), report_kind="annual",
                              unit_multipliers={"BS": 1, "IS": 1, "CF": 1},
                              facts=facts, checks=("bs_balance",))


def test_run_benchmark_precision_ignores_withheld_facts(tmp_path: Path, monkeypatch) -> None:
    """Eight emitted matches out of ten labels give precision 1 and coverage 0.8."""
    from src.data.dart_document_benchmark import run_benchmark
    from src.data.dart_documents import DartDocumentStore
    from src.data.receipt_catalog import ReceiptCatalog
    import src.integrations.dart.document_statements as statements_mod

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    natural_key = f"{CORP}:2023:11011"
    page = _standard_page(extra_rows=_ten_fact_rows())
    _publish_standard(bronze_root, catalog, natural_key=natural_key, page=page)
    archive = b"PK\x03\x04benchmark"
    DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        archive, rcept_no=RCEPT, retrieved_at=NOW)
    wanted = {key: value for i, (key, value) in enumerate(sorted(_ten_labels().items())) if i < 8}
    wanted[("sales", "cumulative")] = 999_999

    from src.integrations.dart.document_statements import DocumentParseResult

    monkeypatch.setattr(statements_mod, "parse_filing_document",
                        lambda *args, **kwargs: DocumentParseResult(
                            statements=_statements_for(wanted), diagnostics=()))
    report = run_benchmark(_runtime(bronze_root), filings=[natural_key])
    assert report.precision == 1.0
    assert report.coverage == 0.8
    assert report.compared == 8
    assert report.exact == 8


def test_run_benchmark_mismatch_is_listed(tmp_path: Path, monkeypatch) -> None:
    """One emitted fact off by 1 KRW appears in mismatches with precision below 1."""
    from src.data.dart_document_benchmark import run_benchmark
    from src.data.dart_documents import DartDocumentStore
    from src.data.receipt_catalog import ReceiptCatalog
    import src.integrations.dart.document_statements as statements_mod

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    natural_key = f"{CORP}:2023:11011"
    page = _standard_page(extra_rows=_ten_fact_rows())
    _publish_standard(bronze_root, catalog, natural_key=natural_key, page=page)
    DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        b"PK\x03\x04benchmark", rcept_no=RCEPT, retrieved_at=NOW)
    labels = dict(sorted(_ten_labels().items()))
    first_key = next(iter(labels))
    emitted = dict(labels)
    emitted[first_key] = emitted[first_key] + 1

    from src.integrations.dart.document_statements import DocumentParseResult

    monkeypatch.setattr(statements_mod, "parse_filing_document",
                        lambda *args, **kwargs: DocumentParseResult(
                            statements=_statements_for(emitted), diagnostics=()))
    report = run_benchmark(_runtime(bronze_root), filings=[natural_key])
    assert report.precision < 1
    assert len(report.mismatches) == 1
    fact, parsed, label = first_key[0], emitted[first_key], labels[first_key]
    assert report.mismatches[0] == f"{natural_key}:{fact}:{parsed}:{label}"


def test_benchmark_report_precision_zero_and_dict() -> None:
    """An empty comparison has zero precision and a JSON-ready view."""
    from src.data.dart_document_benchmark import BenchmarkReport

    report = BenchmarkReport(documents=0, compared=0, exact=0, mismatches=(),
                             withheld_documents=0, coverage=0.0)
    assert report.precision == 0.0
    assert report.to_dict() == {"documents": 0, "compared": 0, "exact": 0, "mismatches": [],
                                "withheld_documents": 0, "coverage": 0.0, "precision": 0.0}


def test_parse_int_amount_variants() -> None:
    """Standard amounts parse with commas, signs and floats; absent stays absent."""
    from src.data.dart_document_benchmark import _parse_int_amount

    assert _parse_int_amount(None) is None
    assert _parse_int_amount(1000) == 1000
    assert _parse_int_amount(1000.0) == 1000
    assert _parse_int_amount(1.5) is None
    assert _parse_int_amount("1,000") == 1000
    assert _parse_int_amount("-1,000") == -1000
    assert _parse_int_amount("-") is None
    assert _parse_int_amount("") is None
    assert _parse_int_amount("abc") is None
    assert _parse_int_amount("1.5") is None
    assert _parse_int_amount("1000.0") == 1000


def test_thstrm_bases_cover_statement_kinds() -> None:
    """Every statement division maps its amount fields to the documented basis."""
    from src.data.dart_document_benchmark import _thstrm_bases

    assert _thstrm_bases(sj_div="BS", fact="assets", reprt_code="11011") == {"thstrm_amount": "point_in_time"}
    assert _thstrm_bases(sj_div="IS", fact="sales", reprt_code="11011") == {
        "thstrm_amount": "annual", "thstrm_add_amount": "cumulative"}
    assert _thstrm_bases(sj_div="IS", fact="sales", reprt_code="11014") == {
        "thstrm_amount": "quarter", "thstrm_add_amount": "cumulative"}
    assert _thstrm_bases(sj_div="CF", fact="capex", reprt_code="11014") == {
        "thstrm_amount": "cumulative", "thstrm_add_amount": "cumulative"}
    assert _thstrm_bases(sj_div="CF", fact="capex", reprt_code="11011") == {
        "thstrm_amount": "annual", "thstrm_add_amount": "cumulative"}
    assert _thstrm_bases(sj_div="", fact="sales", reprt_code="11014") == {
        "thstrm_amount": "quarter", "thstrm_add_amount": "cumulative"}
    assert _thstrm_bases(sj_div="", fact="capex", reprt_code="11014") == {
        "thstrm_amount": "cumulative", "thstrm_add_amount": "cumulative"}
    assert _thstrm_bases(sj_div="", fact="cash", reprt_code="11014") == {"thstrm_amount": "point_in_time"}
    assert _thstrm_bases(sj_div="", fact="mystery", reprt_code="11014") == {"thstrm_amount": "quarter"}


def test_standard_labels_edge_cases() -> None:
    """SCE, unknown accounts and absent amounts never become labels."""
    from src.data.dart_document_benchmark import standard_labels

    assert standard_labels({"records": []}) == {}
    assert standard_labels(_standard_page().copy() | {"records": "nope"}) == {}
    page = _standard_page(extra_rows=(
        {"rcept_no": RCEPT, "sj_div": "SCE", "account_id": "ifrs-full_Revenue",
         "account_nm": "매출액", "thstrm_amount": "1"},
        {"rcept_no": RCEPT, "sj_div": "IS", "account_id": "unknown",
         "account_nm": "미지정계정", "thstrm_amount": "1"},
        {"rcept_no": RCEPT, "sj_div": "IS", "account_id": "ifrs-full_Revenue",
         "account_nm": "매출액", "thstrm_amount": ""},
        42,
        _row(fact_id="ifrs-full_Revenue", fact_nm="매출액", sj_div="IS",
             amount="1,000", add_amount="4,000"),
        _row(fact_id="ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment",
             fact_nm="유형자산취득", sj_div="CF", amount="-700"),
        {**_row(fact_id="ifrs-full_Assets", fact_nm="자산총계", sj_div="",
                amount="5,000"), "sj_div": ""},
    ))
    labels = standard_labels(page)
    assert labels[("sales", "annual")] == 1000
    assert labels[("sales", "cumulative")] == 4000
    assert labels[("capex", "annual")] == 700
    assert labels[("assets", "point_in_time")] == 5000


def test_select_benchmark_filings_rejects_invalid_size(tmp_path: Path) -> None:
    """A non-positive sample size fails closed."""
    import pytest

    from src.data.dart_document_benchmark import select_benchmark_filings

    catalog = _stub_catalog({}, tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        select_benchmark_filings(catalog, size=0, seed=7)


def test_select_benchmark_filings_empty_catalog_returns_empty(tmp_path: Path) -> None:
    """No standard-covered identity means no sample."""
    from src.data.dart_document_benchmark import select_benchmark_filings

    assert select_benchmark_filings(_stub_catalog({}, tmp_path), size=10, seed=7) == ()


def test_select_benchmark_filings_skips_unusable_pages(tmp_path: Path) -> None:
    """Missing, corrupt and labelless pages never enter the sample."""
    from src.data.dart_document_benchmark import select_benchmark_filings

    good_key = f"{CORP}:2023:11011"
    pages = {good_key: _standard_page(extra_rows=_ten_fact_rows())}
    catalog = _stub_catalog(pages, tmp_path)

    import hashlib as _hashlib

    class _Entry:
        def __init__(self, natural_key: str, payload_path: Path) -> None:
            self.natural_key = natural_key
            self.retrieved_at = NOW
            self.payload_path = payload_path
            try:
                self.content_hash = _hashlib.sha256(payload_path.read_bytes()).hexdigest()
            except OSError:
                self.content_hash = f"missing-{natural_key}"

    missing = _Entry("missing:2023:11011", tmp_path / "absent.json")
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("not json", encoding="utf-8")
    corrupt = _Entry("corrupt:2023:11011", corrupt_path)
    legacy_page = _standard_page(extra_rows=_ten_fact_rows())
    legacy_page["source_kind"] = "legacy_document"
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_page), encoding="utf-8")
    legacy = _Entry("legacy:2023:11011", legacy_path)
    labelless = _standard_page(extra_rows=(
        _row(fact_id="ifrs-full_Revenue", fact_nm="매출액", sj_div="IS",
             amount="1", rcept_no="20240415009999"),
    ))
    labelless_path = tmp_path / "labelless.json"
    labelless_path.write_text(json.dumps(labelless), encoding="utf-8")
    labelless_entry = _Entry("labelless:2023:11011", labelless_path)
    empty_status = _standard_page(extra_rows=_ten_fact_rows())
    empty_status["status"] = "013"
    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps(empty_status), encoding="utf-8")
    empty_entry = _Entry("empty:2023:11011", empty_path)
    bad_financial = _standard_page(reprt_code="11011", biz_year="abcd",
                                   extra_rows=_ten_fact_rows())
    bad_financial["biz_year"] = "abcd"
    bad_financial["identity"] = {**bad_financial["identity"], "biz_year": "abcd"}
    bad_financial["is_financial"] = True
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad_financial), encoding="utf-8")
    bad_entry = _Entry("bad:abcd:11011", bad_path)

    entries = [*catalog.entries(source="financial_facts"),
               missing, corrupt, legacy, labelless_entry, empty_entry, bad_entry]

    class _Catalog:
        root = tmp_path

        def entries(self, *, source: str):
            assert source == "financial_facts"
            return iter(entries)

    assert select_benchmark_filings(_Catalog(), size=10, seed=7) == (good_key,)


def test_receipt_archive_index_skips_unreadable_receipts(tmp_path: Path) -> None:
    """Foreign blobs without a valid receipt never map to a filing."""
    from src.data.dart_document_benchmark import _receipt_archive_index

    good_dir = tmp_path / "good"
    good_dir.mkdir()
    good_payload = good_dir / "payload.zip"
    good_payload.write_bytes(b"PK\x03\x04")
    (good_dir / "receipt.json").write_text(
        json.dumps({"rcept_no": RCEPT, "retrieved_at": NOW.isoformat()}), encoding="utf-8")
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    (missing_dir / "payload.zip").write_bytes(b"PK\x03\x04")
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "payload.zip").write_bytes(b"PK\x03\x04")
    (corrupt_dir / "receipt.json").write_text("not json", encoding="utf-8")
    list_dir = tmp_path / "listed"
    list_dir.mkdir()
    (list_dir / "payload.zip").write_bytes(b"PK\x03\x04")
    (list_dir / "receipt.json").write_text("[1, 2]", encoding="utf-8")
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "payload.zip").write_bytes(b"PK\x03\x04")
    (bad_dir / "receipt.json").write_text(json.dumps({"rcept_no": "short"}), encoding="utf-8")
    index = _receipt_archive_index({
        "good": good_payload, "missing": missing_dir / "payload.zip",
        "corrupt": corrupt_dir / "payload.zip", "listed": list_dir / "payload.zip",
        "bad": bad_dir / "payload.zip",
    })
    assert set(index) == {RCEPT}


def test_run_benchmark_empty_filings(tmp_path: Path) -> None:
    """No filings give an empty report."""
    from src.data.dart_document_benchmark import run_benchmark

    report = run_benchmark(_runtime(tmp_path / "bronze"), filings=[])
    assert (report.documents, report.compared, report.coverage) == (0, 0, 0.0)


def test_archive_for_filing_prefers_same_key_hash(tmp_path: Path) -> None:
    """A usable hash on the filing's own page beats the receipt scan."""
    from src.data.dart_document_benchmark import _archive_for_filing

    blob_path = tmp_path / "archive.zip"
    blob_path.write_bytes(b"PK\x03\x04same-key")
    archive = _archive_for_filing(raw_document_hash="abc123", receipt="00000000000000",
                                  archive_paths={"abc123": blob_path}, receipt_index={})
    assert archive == b"PK\x03\x04same-key"


def test_archive_for_filing_receipt_fallback_and_missing(tmp_path: Path) -> None:
    """A filing without a usable hash falls back to its receipt archive."""
    from src.data.dart_document_benchmark import _archive_for_filing

    receipt_path = tmp_path / "receipt.zip"
    receipt_path.write_bytes(b"PK\x03\x04receipt")
    receipt_index = {RCEPT: [("2024-03-15T00:00:00+00:00", receipt_path)]}
    archive = _archive_for_filing(raw_document_hash="", receipt=RCEPT,
                                  archive_paths={}, receipt_index=receipt_index)
    assert archive == b"PK\x03\x04receipt"
    assert _archive_for_filing(raw_document_hash="missing", receipt="",
                               archive_paths={}, receipt_index={}) is None
    missing_path = tmp_path / "gone.zip"
    missing_path.write_bytes(b"PK\x03\x04gone")
    missing_path.unlink()
    fallback = _archive_for_filing(raw_document_hash="gone", receipt=RCEPT,
                                   archive_paths={"gone": missing_path},
                                   receipt_index=receipt_index)
    assert fallback == b"PK\x03\x04receipt"


def test_run_benchmark_scans_fact_catalog_once(tmp_path: Path, monkeypatch) -> None:
    """The fact catalog is consumed once no matter how many filings are requested."""
    from src.data.dart_document_benchmark import run_benchmark
    from src.data.dart_documents import DartDocumentStore
    from src.data.receipt_catalog import ReceiptCatalog
    import src.integrations.dart.document_statements as statements_mod
    from src.integrations.dart.document_statements import DocumentParseResult
    from src.data.dart_document_benchmark import standard_labels

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    keys = [f"{CORP}:2023:1101{i}" for i in range(3)]
    page = _standard_page(extra_rows=_ten_fact_rows())
    for key in keys:
        _publish_standard(bronze_root, catalog, natural_key=key, page=page)
    DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        b"PK\x03\x04benchmark", rcept_no=RCEPT, retrieved_at=NOW)
    monkeypatch.setattr(statements_mod, "parse_filing_document",
                        lambda *args, **kwargs: DocumentParseResult(
                            statements=_statements_for(standard_labels(page)), diagnostics=()))
    entry_calls = 0
    real_entries = catalog.entries

    def _counted_entries(*, source: str):
        nonlocal entry_calls
        if source == "financial_facts":
            entry_calls += 1
        return real_entries(source=source)

    monkeypatch.setattr(catalog, "entries", _counted_entries)
    import src.data.dart_document_benchmark as benchmark_mod

    real_catalog = benchmark_mod.ReceiptCatalog
    monkeypatch.setattr(benchmark_mod, "ReceiptCatalog", lambda *args, **kwargs: catalog)
    report = run_benchmark(_runtime(bronze_root), filings=keys)
    assert entry_calls == 1
    assert report.documents == 3


def test_run_benchmark_reads_archive_from_same_key_hash(tmp_path: Path, monkeypatch) -> None:
    """Same-key document hashes resolve the archive without a receipt scan."""
    from datetime import timedelta as _timedelta

    from src.data.dart_document_benchmark import run_benchmark, standard_labels
    from src.data.dart_documents import DartDocumentStore
    from src.data.receipt_catalog import ReceiptCatalog
    import src.integrations.dart.document_statements as statements_mod
    from src.integrations.dart.document_statements import DocumentParseResult

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    natural_key = f"{CORP}:2023:11011"
    page = _standard_page(extra_rows=_ten_fact_rows())
    receipt = DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        b"PK\x03\x04benchmark", rcept_no=RCEPT, retrieved_at=NOW)
    _publish_standard(bronze_root, catalog, natural_key=natural_key, page=page,
                      retrieved_at=NOW - _timedelta(hours=1),
                      patch={"raw_document_hash": receipt.content_hash})
    _publish_standard(bronze_root, catalog, natural_key=natural_key, page=page)
    monkeypatch.setattr(statements_mod, "parse_filing_document",
                        lambda *args, **kwargs: DocumentParseResult(
                            statements=_statements_for(standard_labels(page)), diagnostics=()))
    report = run_benchmark(_runtime(bronze_root), filings=[natural_key])
    assert report.documents == 1
    assert report.precision == 1.0


def test_run_benchmark_skips_filings_without_labels_or_archives(tmp_path: Path) -> None:
    """Labelless pages and missing archives are skipped, never counted."""
    from src.data.dart_document_benchmark import run_benchmark
    from src.data.receipt_catalog import ReceiptCatalog

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    plain_key = f"{CORP}:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=plain_key,
                      page=_standard_page(extra_rows=_ten_fact_rows()))
    labelless_key = "labelless:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=labelless_key,
                      page=_standard_page(extra_rows=(
                          _row(fact_id="ifrs-full_Revenue", fact_nm="매출액", sj_div="IS",
                               amount="1", rcept_no="20240415009999"),
                      )))
    stale_key = "stale:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=stale_key,
                      page=_standard_page(extra_rows=_ten_fact_rows()))
    _publish_standard(bronze_root, catalog, natural_key="ghost:2023:11011",
                      page=_standard_page(extra_rows=_ten_fact_rows()))
    _publish_standard(bronze_root, catalog, natural_key=plain_key,
                      page={**_standard_page(extra_rows=_ten_fact_rows()),
                            "source_kind": "legacy_document", "records": []},
                      retrieved_at=NOW + timedelta(hours=1))
    report = run_benchmark(_runtime(bronze_root),
                           filings=[plain_key, labelless_key, stale_key, "unknown:2023:11011"])
    assert (report.documents, report.compared, report.coverage) == (0, 0, 0.0)
    assert report.withheld_documents == 0


def test_run_benchmark_withheld_document(tmp_path: Path, monkeypatch) -> None:
    """A document with no verified statements counts as withheld."""
    from src.data.dart_document_benchmark import run_benchmark
    from src.data.dart_documents import DartDocumentStore
    from src.data.receipt_catalog import ReceiptCatalog
    import src.integrations.dart.document_statements as statements_mod
    from src.integrations.dart.document_statements import DocumentParseResult

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    natural_key = f"{CORP}:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=natural_key,
                      page=_standard_page(extra_rows=_ten_fact_rows()))
    DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        b"PK\x03\x04benchmark", rcept_no=RCEPT, retrieved_at=NOW)
    monkeypatch.setattr(statements_mod, "parse_filing_document",
                        lambda *args, **kwargs: DocumentParseResult(statements=None, diagnostics=("x",)))
    report = run_benchmark(_runtime(bronze_root), filings=[natural_key])
    assert (report.documents, report.withheld_documents) == (1, 1)
    assert (report.compared, report.coverage, report.precision) == (0, 0.0, 0.0)


SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RCEPT_OTHER = "20240315002222"


def _provider():
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def test_benchmark_job_pending_fetch_and_health(tmp_path: Path) -> None:
    """Benchmark units cost one request each and store before parsing."""
    from src.data.dart_documents import DartDocumentStore
    from src.data.jobs.dart_documents import DartBenchmarkDocumentFetchJob
    from src.data.jobs.runner import JobUnit, build_job_context
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    bronze_root = runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    pending_key = f"{CORP}:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=pending_key,
                      page=_standard_page(extra_rows=_ten_fact_rows()))
    stored_key = "00123456:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=stored_key,
                      page=_standard_page(rcept_no=RCEPT_OTHER, extra_rows=_ten_fact_rows(RCEPT_OTHER)))
    stored_receipt = DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        b"PK\x03\x04stored", rcept_no=RCEPT_OTHER, retrieved_at=NOW)
    hashed_key = "00999999:2023:11011"
    _publish_standard(bronze_root, catalog, natural_key=hashed_key,
                      page=_standard_page(extra_rows=_ten_fact_rows()),
                      patch={"raw_document_hash": stored_receipt.content_hash})
    legacy_page = _standard_page(extra_rows=_ten_fact_rows())
    legacy_page["source_kind"] = "legacy_document"
    _publish_standard(bronze_root, catalog, natural_key="legacy:2023:11011", page=legacy_page)

    calls: list[str] = []

    class _Collector:
        def fetch_document_archive(self, rcept_no: str) -> bytes:
            calls.append(rcept_no)
            return b"PK\x03\x04fetched"

        def health_check(self) -> None:
            calls.append("health")

    ctx = build_job_context(runtime=runtime, provider=_provider(), key_env=None, collector=_Collector())
    job = DartBenchmarkDocumentFetchJob()
    assert job.name == "dart_benchmark_documents"
    units = job.pending(ctx)
    assert [unit.natural_key for unit in units] == [pending_key]
    assert all(unit.max_requests == 1 for unit in units)
    payloads = job.fetch(ctx, units)
    # 벤치마크는 원문 zip만 저장한다: financial_facts 페이지를 쓰면 그 자연키의
    # 최신 항목이 표준 API 페이지에서 이 문서로 바뀌어, 대조에 쓸 정답 자체가 사라진다.
    assert payloads == ()
    assert calls[0] == RCEPT
    assert (bronze_root / "dart_documents").is_dir()
    latest_after = catalog.latest(source="financial_facts", natural_keys={pending_key})
    assert latest_after[pending_key].content_hash != ""
    refreshed_page = json.loads(latest_after[pending_key].payload_path.read_bytes())
    assert refreshed_page["source_kind"] == "opendart_standard"
    assert job.fetch(ctx, [JobUnit(source="financial_facts", natural_key="empty",
                                   payload={"corp_code": "", "biz_year": "", "reprt_code": "",
                                            "filing_id": "", "rcept_no": ""}, max_requests=1)]) == ()

    class _MissingCollector:
        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b"<status>014</status>"

        def health_check(self) -> None:
            return None

    missing_ctx = build_job_context(runtime=runtime, provider=_provider(),
                                    key_env=None, collector=_MissingCollector())
    missing_payloads = job.fetch(missing_ctx, units)
    assert missing_payloads == ()
    job.health_check(ctx)
    assert "health" in calls


def test_cli_collect_benchmark_documents_dry_run(tmp_path: Path, capsys) -> None:
    """The benchmark fetch command plans through the budgeted runner."""
    from src.data.cli import main
    from tests.fixtures.cli_fixtures import _cli_json_lines

    assert main([
        "collect-dart-benchmark-documents",
        "--scope-config", str(SCOPE_CONFIG),
        "--data-root", str(tmp_path / "data"),
        "--dry-run",
    ]) == 0
    summary = _cli_json_lines(capsys)[-1]
    assert summary["job"] == "dart_benchmark_documents"
    assert summary["status"] == "dry_run"


def test_cli_benchmark_documents_gate(tmp_path: Path, capsys, monkeypatch) -> None:
    """The benchmark command exits 1 below min_precision and 0 at the gate."""
    import json as _json

    from src.data.cli import main
    import src.integrations.dart.document_statements as statements_mod
    from src.data.dart_document_benchmark import standard_labels
    from src.data.dart_documents import DartDocumentStore
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.runtime import load_data_runtime
    from src.integrations.dart.document_statements import DocumentParseResult

    base = ["--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data")]
    assert main(["benchmark-dart-documents", *base]) == 1
    gated = _json.loads(capsys.readouterr().out.splitlines()[-1])
    assert gated["precision"] == 0.0

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    bronze_root = runtime.workspace.bronze_root
    catalog = ReceiptCatalog(bronze_root / "catalog")
    natural_key = f"{CORP}:2023:11011"
    page = _standard_page(extra_rows=_ten_fact_rows())
    _publish_standard(bronze_root, catalog, natural_key=natural_key, page=page)
    DartDocumentStore(bronze_root, catalog=catalog).store_archive(
        b"PK\x03\x04benchmark", rcept_no=RCEPT, retrieved_at=NOW)
    monkeypatch.setattr(statements_mod, "parse_filing_document",
                        lambda *args, **kwargs: DocumentParseResult(
                            statements=_statements_for(standard_labels(page)), diagnostics=()))
    assert main(["benchmark-dart-documents", *base]) == 0
    passed = _json.loads(capsys.readouterr().out.splitlines()[-1])
    assert passed["precision"] == 1.0
    assert passed["compared"] == len(standard_labels(page))


def test_select_benchmark_filings_identical_cold_and_warm(tmp_path: Path) -> None:
    """Cold and warm caches sample the same filings as the direct-read path."""
    from src.data.dart_document_benchmark import select_benchmark_filings
    from src.data.receipt_catalog import ReceiptCatalog

    bronze_root = tmp_path / "bronze"
    catalog = ReceiptCatalog(bronze_root / "catalog")
    for key, page in _stratified_pages().items():
        _publish_standard(bronze_root, catalog, natural_key=key, page=page)
    cold = select_benchmark_filings(catalog, size=8, seed=7)
    warm = select_benchmark_filings(catalog, size=8, seed=7)
    assert cold == warm
    assert len(cold) == 8
    assert f"{CORP}:2020:11011F" not in cold
