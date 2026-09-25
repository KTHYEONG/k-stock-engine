"""Invariant scenarios for the one-off legacy recovery tool."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

CORP = "00126380"
PERIOD = "2020Q4"
F1 = "20200101000001"
F_STD = "20210101000001"
PUBLISHED = "2020-11-16T00:00:00+00:00"
PUBLISHED_STD = "2021-04-01T00:00:00+00:00"


def make_archive(*members: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as handle:
        for idx, member in enumerate(members):
            handle.writestr(f"doc{idx}.html", member.encode("utf-8"))
    return buffer.getvalue()


def bs_table(
    *,
    heading: str = "연결재무상태표",
    unit: str = "(단위: 원)",
    headers: tuple[str, str, str] = ("과목", "당기", "전기"),
    assets: tuple[str, str] = ("1,000", "900"),
    debt: tuple[str, str] = ("400", "350"),
    equity: tuple[str, str] = ("600", "550"),
    cash: tuple[str, str] = ("100", "90"),
) -> str:
    return (
        f"<p>{heading} {unit}</p><table>"
        f"<tr><th>{headers[0]}</th><th>{headers[1]}</th><th>{headers[2]}</th></tr>"
        f"<tr><td>자산총계</td><td>{assets[0]}</td><td>{assets[1]}</td></tr>"
        f"<tr><td>부채총계</td><td>{debt[0]}</td><td>{debt[1]}</td></tr>"
        f"<tr><td>자본총계</td><td>{equity[0]}</td><td>{equity[1]}</td></tr>"
        f"<tr><td>현금및현금성자산</td><td>{cash[0]}</td><td>{cash[1]}</td></tr>"
        "</table>"
    )


def is_table(
    *,
    heading: str = "연결손익계산서",
    unit: str = "(단위: 원)",
    sales: tuple[str, str] = ("2,000", "1,800"),
    gross: tuple[str, str] = ("800", "700"),
    operating: tuple[str, str] = ("300", "250"),
    net: tuple[str, str] = ("150", "120"),
) -> str:
    return (
        f"<p>{heading} {unit}</p><table>"
        "<tr><th>과목</th><th>당기</th><th>전기</th></tr>"
        f"<tr><td>매출액</td><td>{sales[0]}</td><td>{sales[1]}</td></tr>"
        f"<tr><td>매출총이익</td><td>{gross[0]}</td><td>{gross[1]}</td></tr>"
        f"<tr><td>영업이익</td><td>{operating[0]}</td><td>{operating[1]}</td></tr>"
        f"<tr><td>당기순이익</td><td>{net[0]}</td><td>{net[1]}</td></tr>"
        "</table>"
    )


def full_archive() -> bytes:
    return make_archive(f"<html><body>{bs_table()}{is_table()}</body></html>")


def _write_receipt(root: Path, kind: str, payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = root / kind / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    stamp = "2021-05-01T00:00:00+00:00"
    (target / "receipt.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "content_hash": digest,
                "source_path": "test",
                "retrieved_at": stamp,
                "ingested_at": stamp,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return digest


def legacy_page(archive_hash: str, *, filing_id: str = F1) -> dict:
    return {
        "source_kind": "legacy_document",
        "identity": {
            "corp_code": CORP,
            "biz_year": "2020",
            "reprt_code": "11011",
            "filing_id": filing_id,
            "rcept_no": filing_id,
            "published_at": PUBLISHED,
            "fiscal_period": PERIOD,
        },
        "records": [
            {
                "company_id": "005930",
                "corp_code": CORP,
                "ticker": "005930",
                "filing_id": filing_id,
                "fiscal_period": PERIOD,
                "fact": "sales",
                "published_at": PUBLISHED,
                "value": 1.0,
                "unit": "KRW",
                "consolidated": True,
            }
        ],
        "mapping_version": "dart-fact-map-v1",
        "raw_document_hash": archive_hash,
    }


def standard_records() -> list[dict]:
    values = {
        "assets": 1000.0,
        "debt": 400.0,
        "equity": 600.0,
        "cash": 100.0,
        "sales": 2000.0,
        "gross_profit": 800.0,
        "operating_profit": 300.0,
        "net_income": 150.0,
    }
    return [
        {
            "corp_code": CORP,
            "filing_id": F_STD,
            "fiscal_period": PERIOD,
            "fact": fact,
            "published_at": PUBLISHED_STD,
            "value": value,
            "unit": "KRW",
            "consolidated": True,
            "source_kind": "opendart_standard",
        }
        for fact, value in values.items()
    ]


def quarantine_entry(*, filing_id: str = F1) -> dict:
    return {
        "company_id": "005930",
        "dart_corp_code": CORP,
        "fiscal_period": PERIOD,
        "filing_id": filing_id,
        "source_kind": "legacy_document",
        "published_at": PUBLISHED,
        "available_at": "2020-11-17T00:00:00+00:00",
    }


def write_bronze(root: Path, *, archive: bytes = b"", standard: bool = True) -> str:
    bronze = root / "bronze"
    archive_hash = ""
    if archive:
        archive_hash = hashlib.sha256(archive).hexdigest()
        target = bronze / "dart_documents" / archive_hash
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.zip").write_bytes(archive)
        (target / "receipt.json").write_text(
            json.dumps(
                {
                    "rcept_no": F1,
                    "source": "document.xml",
                    "retrieved_at": "2021-05-01T00:00:00+00:00",
                    "sha256": archive_hash,
                    "byte_length": len(archive),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_receipt(bronze, "financial_facts", legacy_page(archive_hash))
    if standard:
        _write_receipt(bronze, "financial_facts", {"records": standard_records()})
    quarantine = root / "quarantine.json"
    quarantine.write_text(json.dumps([quarantine_entry()]), encoding="utf-8")
    return archive_hash


class FakeBronze:
    """Content-addressed persist sink mirroring BronzeStore import semantics."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.new_writes = 0
        self.pages: list[dict] = []

    def __call__(self, pages: list[dict]) -> list[str]:
        hashes = []
        for page in pages:
            digest = hashlib.sha256(json.dumps(page, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            if digest not in self.seen:
                self.seen.add(digest)
                self.new_writes += 1
                self.pages.append(page)
            hashes.append(digest)
        return hashes


# --- extract.py scenarios ---


def test_ambiguous_statement_yields_nothing() -> None:
    from tools.legacy_recovery.extract import extract_statements

    archive = make_archive(f"<html><body>{bs_table()}{bs_table()}</body></html>")

    assert extract_statements(archive) == ()


def test_unit_note_required() -> None:
    from tools.legacy_recovery.extract import extract_statements

    archive = make_archive(f"<html><body>{bs_table(unit='')}</body></html>")

    assert extract_statements(archive) == ()


def test_unit_scaling_is_exact() -> None:
    from tools.legacy_recovery.extract import extract_statements

    archive = make_archive(
        "<html><body>"
        + bs_table(
            unit="(단위: 백만원)",
            assets=("12", "11"),
            debt=("5", "4"),
            equity=("7", "7"),
            cash=("1", "1"),
        )
        + "</body></html>"
    )

    (statement,) = extract_statements(archive)
    assert statement.unit_multiplier == 1_000_000
    assert statement.values["assets"] == 12_000_000
    assert statement.values["debt"] == 5_000_000
    assert statement.values["equity"] == 7_000_000
    assert all(isinstance(v, int) for v in statement.values.values())


def test_prior_period_column_never_returned() -> None:
    from tools.legacy_recovery.extract import extract_statements

    archive = make_archive(
        "<html><body>" + bs_table(headers=("과목", "전기", "당기"), assets=("900", "1,000")) + "</body></html>"
    )

    (statement,) = extract_statements(archive)
    assert statement.values["assets"] == 1000


def test_consolidated_and_separate_never_mixed() -> None:
    from tools.legacy_recovery.extract import extract_statements

    archive = make_archive(
        "<html><body>"
        + bs_table(heading="연결재무상태표", assets=("1,000", "900"))
        + bs_table(heading="별도재무상태표", assets=("2,000", "1,900"))
        + "</body></html>"
    )

    statements = extract_statements(archive)
    by_basis = {s.basis: s for s in statements}
    assert set(by_basis) == {"consolidated", "separate"}
    assert by_basis["consolidated"].values["assets"] == 1000
    assert by_basis["separate"].values["assets"] == 2000


def test_malformed_archive_returns_empty() -> None:
    from tools.legacy_recovery.extract import extract_statements

    assert extract_statements(b"") == ()
    assert extract_statements(b"not a zip") == ()


# --- verify.py scenarios ---


def _statement(kind: str, basis: str, unit: int, values: dict) -> object:
    from tools.legacy_recovery.extract import ExtractedStatement

    return ExtractedStatement(kind=kind, basis=basis, unit_multiplier=unit, values=values, evidence="test")


def test_broken_identity_rejected() -> None:
    from tools.legacy_recovery.verify import verify_statement

    statement = _statement("BS", "consolidated", 1, {"assets": 1001, "debt": 400, "equity": 600})

    verdict = verify_statement(statement)  # type: ignore[arg-type]
    assert verdict.accepted is False
    assert verdict.rejected_by == "balance_identity"


def test_income_statement_needs_accepted_balance_sheet() -> None:
    from tools.legacy_recovery.verify import verify_statement

    good_bs = _statement("BS", "consolidated", 1, {"assets": 1000, "debt": 400, "equity": 600})
    lone_is = _statement(
        "IS",
        "consolidated",
        1,
        {"sales": 2000, "gross_profit": 800, "operating_profit": 300},
    )

    assert verify_statement(lone_is).rejected_by == "balance_sheet_link"  # type: ignore[arg-type]
    assert (
        verify_statement(lone_is, siblings=[good_bs]).accepted is True  # type: ignore[arg-type]
    )
    other_unit = _statement("BS", "consolidated", 1000, {"assets": 1000, "debt": 400, "equity": 600})
    assert (
        verify_statement(lone_is, siblings=[other_unit]).rejected_by  # type: ignore[arg-type]
        == "balance_sheet_link"
    )
    bad_order = _statement("IS", "consolidated", 1, {"sales": 100, "gross_profit": 800})
    assert (
        verify_statement(bad_order, siblings=[good_bs]).rejected_by  # type: ignore[arg-type]
        == "profit_ordering"
    )


# --- benchmark.py scenarios ---


def test_gate_blocks_small_samples() -> None:
    from tools.legacy_recovery.benchmark import (
        PromotionGate,
        evaluate,
        filing_key,
    )

    statement = _statement("BS", "consolidated", 1, {"assets": 1000, "debt": 400, "equity": 600})
    extracted = {filing_key(CORP, PERIOD, "consolidated"): [statement]}
    labeled = {(CORP, PERIOD, "consolidated"): {"assets": 1000, "debt": 400, "equity": 600}}

    (result,) = (
        r
        for r in evaluate(
            extracted,  # type: ignore[arg-type]
            labeled,
            PromotionGate(min_exact_match_rate=1.0, min_accepted_filings=2),
        )
        if r.fact_class == "balance_sheet"
    )
    assert result.promotable is False


def test_gate_blocks_low_precision() -> None:
    from tools.legacy_recovery.benchmark import (
        PromotionGate,
        evaluate,
        filing_key,
    )

    good = _statement("BS", "consolidated", 1, {"assets": 1000, "debt": 400, "equity": 600})
    bad = _statement("BS", "consolidated", 1, {"assets": 1002, "debt": 400, "equity": 602})
    extracted = {
        filing_key(CORP, "2020Q4", "consolidated"): [good],
        filing_key(CORP, "2021Q4", "consolidated"): [bad],
    }
    labeled = {
        (CORP, "2020Q4", "consolidated"): {"assets": 1000, "debt": 400, "equity": 600},
        (CORP, "2021Q4", "consolidated"): {"assets": 1000, "debt": 400, "equity": 600},
    }

    (result,) = (
        r
        for r in evaluate(
            extracted,  # type: ignore[arg-type]
            labeled,
            PromotionGate(min_exact_match_rate=0.95, min_accepted_filings=1),
        )
        if r.fact_class == "balance_sheet"
    )
    assert result.accepted_filings == 2
    assert result.exact_matches == 1
    assert result.promotable is False


def test_labeled_set_ignores_untrusted_sources(tmp_path: Path) -> None:
    from tools.legacy_recovery.benchmark import build_labeled_set

    root = tmp_path / "case"
    write_bronze(root, archive=full_archive(), standard=False)

    assert build_labeled_set(root / "bronze", root / "quarantine.json") == {}


# --- run.py scenarios ---


def _gate() -> object:
    from tools.legacy_recovery.benchmark import PromotionGate

    return PromotionGate(min_exact_match_rate=0.5, min_accepted_filings=1)


def test_nothing_promoted_without_promotable_class(tmp_path: Path) -> None:
    from tools.legacy_recovery.benchmark import PromotionGate
    from tools.legacy_recovery.run import run_recovery

    root = tmp_path / "case"
    write_bronze(root, archive=full_archive())
    sink = FakeBronze()

    result = run_recovery(
        bronze_root=root / "bronze",
        quarantine_file=root / "quarantine.json",
        report_dir=root / "reports",
        gate=PromotionGate(min_exact_match_rate=1.0, min_accepted_filings=1000),
        persist_fn=sink,
    )

    assert result.status == "no_promotable_class"
    assert result.persisted_receipts == ()
    assert sink.new_writes == 0


def test_attested_page_complete_and_idempotent(tmp_path: Path) -> None:
    from tools.legacy_recovery.run import run_recovery

    root = tmp_path / "case"
    write_bronze(root, archive=full_archive())
    sink = FakeBronze()
    kwargs = {
        "bronze_root": root / "bronze",
        "quarantine_file": root / "quarantine.json",
        "report_dir": root / "reports",
        "gate": _gate(),
        "persist_fn": sink,
    }

    first = run_recovery(**kwargs)  # type: ignore[arg-type]
    assert first.status == "complete"
    assert sink.new_writes > 0
    assert len(json.loads((root / "reports" / "legacy_recovery_report.json").read_text())) > 0

    from src.data.normalization import _has_valid_attestation

    for page in sink.pages:
        assert page["source_kind"] == "legacy_document_verified"
        assert set(page["verification"]) == {"benchmark_id", "fact_class", "checks"}
        assert page["verification"]["benchmark_id"] == first.benchmark_id
        assert page["verification"]["fact_class"] in ("balance_sheet", "income_statement")
        assert page["verification"]["checks"]
        for record in page["records"]:
            assert record["source_kind"] == "legacy_document_verified"
            assert _has_valid_attestation(record) is True

    before = sink.new_writes
    second = run_recovery(**kwargs)  # type: ignore[arg-type]
    assert second.benchmark_id == first.benchmark_id
    assert sink.new_writes == before


def test_dry_run_makes_no_api_call(tmp_path: Path) -> None:
    from tools.legacy_recovery.run import run_recovery

    root = tmp_path / "case"
    (root / "bronze").mkdir(parents=True)
    (root / "quarantine.json").write_text(json.dumps([quarantine_entry()]))

    class ExplodingCollector:
        def health_check(self) -> None:
            raise AssertionError("no API calls in dry run")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            raise AssertionError("no API calls in dry run")

    result = run_recovery(
        bronze_root=root / "bronze",
        quarantine_file=root / "quarantine.json",
        report_dir=root / "reports",
        gate=_gate(),  # type: ignore[arg-type]
        dry_run=True,
        collector=ExplodingCollector(),  # type: ignore[arg-type]
        headroom_fn=lambda: 0,
    )

    assert result.status == "dry_run"
    assert result.persisted_receipts == ()
    assert (root / "reports" / "legacy_recovery_report.json").exists()
    assert (root / "reports" / "legacy_recovery_report.md").exists()


def test_quota_exhaustion_stops_cleanly(tmp_path: Path) -> None:
    from tools.legacy_recovery.run import run_recovery

    root = tmp_path / "case"
    (root / "bronze").mkdir(parents=True)
    (root / "quarantine.json").write_text(json.dumps([quarantine_entry()]))

    class CountingCollector:
        def __init__(self) -> None:
            self.calls = 0

        def health_check(self) -> None:
            self.calls += 1

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            self.calls += 1
            return b""

    collector = CountingCollector()
    result = run_recovery(
        bronze_root=root / "bronze",
        quarantine_file=root / "quarantine.json",
        report_dir=root / "reports",
        gate=_gate(),  # type: ignore[arg-type]
        collector=collector,  # type: ignore[arg-type]
        headroom_fn=lambda: 0,
        persist_fn=FakeBronze(),
    )

    assert collector.calls == 0
    assert result.remaining_archives == 1
    assert result.report["remaining_archives"] == 1


def test_run_reports_per_year_and_period(tmp_path: Path) -> None:
    from tools.legacy_recovery.run import run_recovery

    root = tmp_path / "case"
    write_bronze(root, archive=full_archive())
    result = run_recovery(
        bronze_root=root / "bronze",
        quarantine_file=root / "quarantine.json",
        report_dir=root / "reports",
        gate=_gate(),  # type: ignore[arg-type]
        persist_fn=FakeBronze(),
    )

    assert result.report["by_year"]["2020"]["balance_sheet"]["accepted"] == 1
    assert result.report["by_period"][PERIOD]["balance_sheet"]["exact"] == 1
    assert result.report["acceptance_rate"]["attempted_filings"] == 1


def test_run_fetch_stores_archive_and_resumes(tmp_path: Path) -> None:
    from tools.legacy_recovery.run import run_recovery

    root = tmp_path / "case"
    write_bronze(root, archive=b"", standard=True)
    archive = full_archive()
    seen: list[str] = []

    class FakeCollector:
        def health_check(self) -> None:
            seen.append("health")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            seen.append(rcept_no)
            return archive

    result = run_recovery(
        bronze_root=root / "bronze",
        quarantine_file=root / "quarantine.json",
        report_dir=root / "reports",
        gate=_gate(),  # type: ignore[arg-type]
        collector=FakeCollector(),  # type: ignore[arg-type]
        headroom_fn=lambda: 100,
        persist_fn=FakeBronze(),
        now=datetime(2021, 5, 1, tzinfo=UTC),
    )

    assert result.fetched_archives == 1
    assert result.remaining_archives == 0
    assert result.status == "complete"
    assert (root / "bronze" / "dart_documents").exists()
