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
    collector = DartXbrlCollector(max_workers=1, min_interval=0.0, api_key="test-key", request_json=lambda _endpoint, _params: {"status": "013", "list": []}, request_bytes=lambda _endpoint, _params: archive)
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


def test_archive_with_unexpected_zip_error_is_bad_zip(monkeypatch) -> None:
    import zipfile

    import src.integrations.dart.legacy_filing as legacy

    def _boom(*_args, **_kwargs):
        raise RuntimeError("torn archive")

    monkeypatch.setattr(zipfile, "ZipFile", _boom)

    result = legacy.parse_legacy_filing_archive(
        archive_bytes=b"PK fake", identity=identity("20240101000001"), document_hash="abc"
    )

    assert result.status == "extraction_failed"
    assert result.diagnostics == ("bad_zip",)


def test_shared_decode_member_variants() -> None:
    from src.integrations.dart.html_tables import decode_member, extract_tables

    assert decode_member("가".encode()) == "가"
    assert decode_member(b"\xef\xbb\xbf" + "가".encode()) == "가"
    assert decode_member("가".encode("utf-16")) == "가"
    assert decode_member(b"\xff\xfe\x00") is None
    declared = '<?xml version="1.0" encoding="euc-kr"?><doc>가</doc>'.encode("euc-kr")
    decoded = decode_member(declared)
    assert decoded is not None
    assert "가" in decoded
    assert decode_member("가".encode("utf-16")) == "가"
    assert decode_member("가".encode("cp949")) == "가"
    assert decode_member(b"\xef\xbb\xbf\xff\xfe") is None
    assert decode_member(b'<?xml encoding="utf-8"?>\xff\xff') is None
    assert decode_member(b'<?xml encoding="cp949"?>\x81') is None

    tables = extract_tables("<table><tr><td> a </td><td>b</td></tr></table>")
    assert tables == [[["a", "b"]]]
    assert extract_tables("no tables here") == []


def test_shared_table_serializer_failure_is_unitless(monkeypatch) -> None:
    from xml.etree import ElementTree

    import src.integrations.dart.legacy_filing as legacy

    def _always_fail(*_args, **_kwargs):
        raise ValueError("hostile tree")

    monkeypatch.setattr(ElementTree, "tostring", _always_fail)

    records, diags = legacy._extract_records(
        ElementTree.fromstring(  # noqa: S314 - offline test fixture without external entities
            "<document><account><account_nm>매출액</account_nm><amount>100</amount></account></document>"
        ),
        "nothing to see here",
        company_id="001",
        filing_id="20240101000001",
        fiscal_period="2023Q4",
        document_hash="abc",
    )

    assert records == []
    assert diags == ["extraction_failed"]


def test_archive_member_priority_prefers_consolidated_statement() -> None:
    from src.integrations.dart.legacy_filing import _archive_member_priority

    assert _archive_member_priority("corp_00761.xml") == 3
    assert _archive_member_priority("corp_00760.xml") == 1
    assert _archive_member_priority("other.xml") == 2


def test_archive_with_directory_entry_skips_it() -> None:
    import io
    import zipfile

    from src.integrations.dart.legacy_filing import parse_legacy_filing_archive

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nested/", b"")
        zf.writestr("20150515001111.xml", legacy_statement_xml())
    parsed = parse_legacy_filing_archive(
        archive_bytes=buf.getvalue(), identity=identity("20150515001111"), document_hash="c" * 64
    )
    assert parsed.records != ()
    assert parsed.status != "extraction_failed"
