"""Legacy filing archive tests (offline fixtures only)."""
from __future__ import annotations

import io
import zipfile


def identity(rcept_no: str) -> dict[str, str]:
    return {
        "corp_code": "001",
        "filing_id": rcept_no,
        "rcept_no": rcept_no,
        "biz_year": "2015",
        "reprt_code": "11011",
        "report_code": "11011",
        "fs_div": "CFS",
    }


def make_legacy_archive(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()


def legacy_statement_xml() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<document>"
        '<account><account_nm>매출액</account_nm><amount>100</amount><unit>KRW</unit></account>'
        '<account><account_nm>자산총계</account_nm><amount>1000</amount><unit>KRW</unit></account>'
        "</document>"
    )


def ambiguous_statement_xml() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<document>"
        '<account><account_nm>매출액</account_nm><amount>100</amount><unit>KRW</unit></account>'
        '<account><account_nm>매출액</account_nm><amount>200</amount><unit>KRW</unit></account>'
        "</document>"
    )


def test_financial_source_013_uses_legacy_document_not_no_data() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    archive = make_legacy_archive({"20150515001111.xml": legacy_statement_xml()})
    collector = DartXbrlCollector(api_key="test-key", request_json=lambda _endpoint, _params: {"status": "013", "list": []}, request_bytes=lambda _endpoint, _params: archive)
    page = next(collector.fetch_financial_fact_sources((identity("20150515001111"),)))
    assert page["source_kind"] == "legacy_document"
    assert page["status"] == "013"
    assert page["raw_archive"] == archive


def test_legacy_parser_rejects_ambiguous_or_unsafe_archive_without_facts() -> None:
    from src.integrations.dart.legacy_filing import parse_legacy_filing_archive

    unsafe = make_legacy_archive({"../escape.xml": legacy_statement_xml()})
    rejected = parse_legacy_filing_archive(archive_bytes=unsafe, identity=identity("20150515001111"), document_hash="a" * 64)
    assert rejected.records == ()
    assert rejected.status == "extraction_failed"
    ambiguous = make_legacy_archive({"20150515001111.xml": ambiguous_statement_xml()})
    parsed = parse_legacy_filing_archive(archive_bytes=ambiguous, identity=identity("20150515001111"), document_hash="b" * 64)
    assert parsed.records == ()
    assert "ambiguous" in parsed.diagnostics


def test_legacy_parser_recovers_annual_table_facts_when_structured_nodes_are_ambiguous() -> None:
    from src.integrations.dart.legacy_filing import parse_legacy_filing_archive

    statement = """
    <?xml version="1.0" encoding="utf-8"?>
    <document>
      <TABLE>
        <TR><TD>매출액</TD><TD>1,000</TD><TD>900</TD></TR>
        <TR><TD>영업이익</TD><TD>100</TD><TD>90</TD></TR>
        <TR><TD>당기순이익</TD><TD>80</TD><TD>70</TD></TR>
        <TR><TD>자산총계</TD><TD>2,000</TD><TD>1,900</TD></TR>
        <TR><TD>부채총계</TD><TD>800</TD><TD>750</TD></TR>
        <TR><TD>자본총계</TD><TD>1,200</TD><TD>1,150</TD></TR>
      </TABLE>
    </document>
    """
    parsed = parse_legacy_filing_archive(
        archive_bytes=make_legacy_archive({"F1.xml": statement}),
        identity=identity("F1"),
        document_hash="c" * 64,
    )

    assert parsed.status == "ok"
    assert {record["fact"] for record in parsed.records} >= {
        "sales",
        "operating_profit",
        "net_income",
        "assets",
        "debt",
        "equity",
    }
    assert next(record["value"] for record in parsed.records if record["fact"] == "sales") == 1000


def test_legacy_table_skips_account_reference_before_full_amount() -> None:
    from src.integrations.dart.legacy_filing import parse_legacy_filing_archive

    statement = """
    <?xml version="1.0" encoding="utf-8"?>
    <document>
      <TABLE>
        <TR><TD>매출액</TD><TD>4,24</TD><TD>15,826,896,964</TD><TD>17,085,468,266</TD></TR>
      </TABLE>
    </document>
    """
    parsed = parse_legacy_filing_archive(
        archive_bytes=make_legacy_archive({"F2.xml": statement}),
        identity=identity("F2"),
        document_hash="d" * 64,
    )

    assert parsed.status == "ok"
    assert next(record["value"] for record in parsed.records if record["fact"] == "sales") == 15_826_896_964
