"""Section-anchored filing-document parser tests (offline fixtures only)."""
from __future__ import annotations

CFS = "D-0-3-2-0"
OFS = "D-0-3-4-0"


def make_archive(files: dict[str, str | bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content.encode("utf-8") if isinstance(content, str) else content)
    return buf.getvalue()


def table(rows: list[list[str]]) -> str:
    body = "".join("<TR>" + "".join(f"<TD>{cell}</TD>" for cell in row) + "</TR>" for row in rows)
    return f"<TABLE>{body}</TABLE>"


def para(text: str) -> str:
    return f"<P>{text}</P>"


def section(code: str, title: str, blocks: list[str]) -> str:
    return f'<TITLE ATOC="Y" AASSOCNOTE="{code}">{title}</TITLE>' + "".join(blocks)


def bs_rows(
    assets: str = "1,000",
    debt: str = "400",
    equity: str = "600",
    cash: str = "100",
    prior: tuple[str, str, str, str] = ("900", "350", "550", "90"),
) -> list[list[str]]:
    return [
        ["과목", "제 35 기", "제 34 기"],
        ["자산총계", assets, prior[0]],
        ["부채총계", debt, prior[1]],
        ["자본총계", equity, prior[2]],
        ["현금및현금성자산", cash, prior[3]],
        ["이익잉여금", "50", "40"],
    ]


def bs_section(
    *,
    code: str = OFS,
    title: str = "재무제표",
    heading: str = "재무상태표",
    unit: str = "(단위 : 원)",
    when: str = "2019년 12월 31일 현재",
    rows: list[list[str]] | None = None,
    extra: list[str] | None = None,
) -> str:
    blocks = [para("문서 앞부분 안내 문구"), para(heading), para(unit), para(when), table(rows or bs_rows())]
    return section(code, title, blocks + (extra or []))


def annual_archive() -> bytes:
    return make_archive({"20200101000001.xml": bs_section(code=CFS, title="연결재무제표") + bs_section()})


def flow_section(blocks: list[str]) -> str:
    return section(OFS, "재무제표", blocks)


def half_sections(*, cf_cash: str = "100") -> str:
    is_body = table(
        [
            ["과목", "제 35 기 반기", "제 35 기 반기", "제 34 기 반기", "제 34 기 반기"],
            ["구분", "3개월", "누적", "3개월", "누적"],
            ["매출액", "100", "250", "90", "200"],
            ["매출총이익", "60", "150", "50", "120"],
            ["영업이익", "30", "80", "25", "70"],
            ["당기순이익", "20", "50", "15", "40"],
        ]
    )
    cf_body = table(
        [
            ["과목", "제 35 기 반기", "제 34 기 반기"],
            ["구분", "누적", "누적"],
            ["영업활동으로인한현금흐름", "500", "400"],
            ["유형자산의취득", "(30)", "(20)"],
            ["기말현금및현금성자산", cf_cash, "90"],
            ["당기순이익", "20", "10"],
            ["감가상각비", "5", "4"],
        ]
    )
    return flow_section(
        [
            para("반기재무상태표"),
            table([["재무상태표", "단위 : 천원"], ["제 35 기 반기", "2019년 6월 30일 현재"]]),
            table(bs_rows()),
            para("반기손익계산서"),
            para("(단위 : 천원)"),
            is_body,
            para("반기현금흐름표"),
            para("(단위 : 천원)"),
            cf_body,
        ]
    )


def facts_by_name(result) -> dict[str, tuple[int, str]]:
    from src.integrations.dart.document_statements import StatementFact

    assert result.statements is not None
    out: dict[str, tuple[int, str]] = {}
    fact: StatementFact
    for fact in result.statements.facts:
        out[fact.fact] = (fact.value, fact.basis.value)
    return out


def test_balanced_sheet_verifies_with_no_marker_for_annual() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    result = parse_filing_document(annual_archive(), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert result.statements.consolidated is True
    assert result.statements.period_end.isoformat() == "2019-12-31"
    assert result.statements.report_kind == "annual"
    assert facts_by_name(result) == {
        "assets": (1000, "point_in_time"),
        "debt": (400, "point_in_time"),
        "equity": (600, "point_in_time"),
        "cash": (100, "point_in_time"),
    }
    assert result.statements.checks == ("bs_balance",)
    assert result.statements.unit_multipliers == {"BS": 1}


def test_broken_identity_falls_back_to_separate_basis() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    broken = bs_section(code=CFS, title="연결재무제표", rows=bs_rows(assets="1,001"))
    good = bs_section()
    result = parse_filing_document(make_archive({"F.xml": broken + good}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert result.statements.consolidated is False
    assert "identity_failed:bs_balance" in result.diagnostics


def test_unit_note_scales_values_exactly() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(code=CFS, title="연결재무제표", unit="(단위 : 백만원)")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert result.statements.unit_multipliers == {"BS": 1_000_000}
    assert facts_by_name(result)["assets"] == (1_000_000_000, "point_in_time")


def test_missing_unit_withholds_whole_document() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(unit="금액 표시 없음")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "missing_unit:BS" in result.diagnostics


def test_current_column_uses_highest_period_number() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    rows = [
        ["과목", "제 34 기", "제 35 기"],
        ["자산총계", "900", "1,000"],
        ["부채총계", "350", "400"],
        ["자본총계", "550", "600"],
        ["현금및현금성자산", "90", "100"],
        ["이익잉여금", "40", "50"],
    ]
    archive = make_archive({"F.xml": bs_section(code=CFS, title="연결재무제표", rows=rows)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_spanned_current_column_reads_single_amount() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    def spanned(cash_cells: tuple[str, str]) -> bytes:
        rows = [
            ["과목", "주석", "제71(당)기", "제71(당)기", "제70(전)기"],
            ["자산총계", "", "1,000", "", "900"],
            ["부채총계", "", "400", "", "350"],
            ["자본총계", "", "600", "", "550"],
            ["현금및현금성자산", "", cash_cells[0], cash_cells[1], "90"],
            ["이익잉여금", "", "50", "", "40"],
        ]
        return make_archive({"F.xml": bs_section(code=CFS, title="연결재무제표", rows=rows)})

    assert facts_by_name(parse_filing_document(spanned(("100", "")), reprt_code="11011", biz_year="2019"))["cash"] == (
        100,
        "point_in_time",
    )
    crowded = parse_filing_document(spanned(("100", "100")), reprt_code="11011", biz_year="2019")
    assert crowded.statements is not None
    assert "cash" not in facts_by_name(crowded)


def test_quarter_column_wins_for_q3_income_statement() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": half_sections()})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert result.statements is not None
    assert result.statements.report_kind == "half"
    assert result.statements.consolidated is False
    values = facts_by_name(result)
    assert values["sales"] == (100_000, "quarter")
    assert values["operating_profit"] == (30_000, "quarter")
    assert values["operating_cash_flow"] == (500_000, "cumulative")
    assert values["capex"] == (30_000, "cumulative")
    assert result.statements.unit_multipliers == {"BS": 1_000, "IS": 1_000, "CF": 1_000}
    assert result.statements.checks == ("bs_balance", "cf_cash_tieout")

def test_cumulative_only_interim_marks_flows_cumulative() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제 35 기 3분기", "제 34 기 3분기"],
            ["구분", "누적", "누적"],
            ["매출액", "250", "200"],
            ["매출총이익", "150", "120"],
            ["영업이익", "80", "70"],
            ["당기순이익", "50", "40"],
            ["기타", "1", "2"],
        ]
    )
    archive = make_archive(
        {
            "F.xml": bs_section(
                when="2019년 9월 30일 현재",
                heading="3분기재무상태표",
                extra=[para("3분기손익계산서"), para("(단위 : 원)"), body],
            )
        }
    )
    result = parse_filing_document(archive, reprt_code="11014", biz_year="2019")

    assert facts_by_name(result)["sales"] == (250, "cumulative")


def test_q1_single_column_is_emitted_as_quarter() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["구분", "누적", "누적"],
            ["매출액", "250", "200"],
            ["영업이익", "80", "70"],
            ["당기순이익", "50", "40"],
            ["매출총이익", "150", "120"],
            ["기타", "1", "2"],
        ]
    )
    archive = make_archive(
        {
            "F.xml": bs_section(
                when="2019년 3월 31일 현재",
                heading="1분기재무상태표",
                extra=[para("1분기손익계산서"), para("(단위 : 원)"), body],
            )
        }
    )
    result = parse_filing_document(archive, reprt_code="11013", biz_year="2019")

    assert result.statements is not None
    assert result.statements.report_kind == "q1"
    assert facts_by_name(result)["sales"] == (250, "quarter")


def test_period_mismatch_withholds_document() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2016.03.31 현재")})
    result = parse_filing_document(archive, reprt_code="11013", biz_year="2017")

    assert result.statements is None
    assert result.diagnostics == ("missing_section:D-0-3-2-0", "period_mismatch")


def test_non_december_fiscal_year_is_flagged() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2016년 2월 28일 현재")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2016")

    assert result.statements is None
    assert "non_december_fiscal_year" in result.diagnostics


def test_missing_balance_sheet_date_is_flagged() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="기말 현재")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "missing_period:BS" in result.diagnostics


def test_invalid_calendar_date_is_not_a_period() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2024.06.31 현재")})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2024")

    assert result.statements is None
    assert "missing_period:BS" in result.diagnostics


def test_unknown_report_code_or_year_withholds() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section()})
    assert parse_filing_document(archive, reprt_code="99999", biz_year="2019").diagnostics == (
        "missing_section:D-0-3-2-0",
        "period_mismatch",
    )
    assert "period_mismatch" in parse_filing_document(archive, reprt_code="11011", biz_year="201").diagnostics
    assert "period_mismatch" in parse_filing_document(archive, reprt_code="11011", biz_year="0000").diagnostics


def test_annual_report_with_quarter_marker_mismatches() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(heading="반기재무상태표")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "period_mismatch" in result.diagnostics


def test_interim_report_without_marker_mismatches() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2019년 6월 30일 현재")})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert result.statements is None
    assert "period_mismatch" in result.diagnostics


def test_generic_quarter_word_satisfies_interim_marker() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2019년 6월 30일 현재", heading="당분기재무상태표")})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert result.statements is not None
    assert result.statements.report_kind == "half"


def test_body_header_marker_satisfies_interim_when_heading_has_none() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    rows = [
        ["과목", "제35기 반기", "제34기 반기"],
        ["자산총계", "1,000", "900"],
        ["부채총계", "400", "350"],
        ["자본총계", "600", "550"],
        ["현금및현금성자산", "100", "90"],
        ["이익잉여금", "50", "40"],
    ]
    archive = make_archive({"F.xml": bs_section(when="2019년 6월 30일 현재", rows=rows)})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert result.statements is not None


def test_notes_section_is_never_read() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    notes = section("D-0-3-3-0", "주석", [para("재무상태표"), para("(단위 : 원)"), table(bs_rows())])
    result = parse_filing_document(make_archive({"F.xml": notes}), reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "missing_section:D-0-3-2-0" in result.diagnostics
    assert "missing_section:D-0-3-4-0" in result.diagnostics


def test_cash_tie_out_failure_drops_cash_flow_only() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": half_sections(cf_cash="101")})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert result.statements is not None
    assert result.statements.checks == ("bs_balance",)
    assert "identity_failed:cf_cash_tieout" in result.diagnostics
    values = facts_by_name(result)
    assert "operating_cash_flow" not in values
    assert values["sales"] == (100_000, "quarter")


def test_matching_cash_keeps_cash_flow_facts() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": half_sections()})
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert result.statements is not None
    assert result.statements.checks == ("bs_balance", "cf_cash_tieout")
    assert facts_by_name(result)["operating_cash_flow"] == (500_000, "cumulative")


def test_malformed_archive_never_raises() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    assert parse_filing_document(b"not a zip at all", reprt_code="11011", biz_year="2019").statements is None
    assert parse_filing_document(b"", reprt_code="11011", biz_year="2019").diagnostics == ("bad_zip",)


def test_attachment_only_archive_reports_missing_member() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"20200101000001_00001.xml": bs_section()})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert result.diagnostics == ("missing_main_member",)


def test_directory_entry_is_skipped_for_main_member() -> None:
    import io
    import zipfile

    from src.integrations.dart.document_statements import parse_filing_document

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nested/", b"")
        zf.writestr("20200101000001.xml", bs_section(code=CFS, title="연결재무제표"))
    result = parse_filing_document(buf.getvalue(), reprt_code="11011", biz_year="2019")

    assert result.statements is not None


def test_dangerous_declaration_is_rejected() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    for payload in ("<!DOCTYPE evil>", "<!ENTITY evil>"):
        archive = make_archive({"F.xml": payload + bs_section()})
        result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")
        assert result.statements is None
        assert result.diagnostics == ("unsafe_xml_declaration",)


def test_undecodable_member_is_reported() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": b"\xff\xfe\x00"})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert result.diagnostics == ("decode_failed",)


def test_unexpected_reader_failure_never_raises(monkeypatch) -> None:
    import src.integrations.dart.document_statements as doc

    def _boom(_markup: str) -> tuple:
        raise RuntimeError("torn reader")

    monkeypatch.setattr(doc, "read_blocks", _boom)
    result = doc.parse_filing_document(annual_archive(), reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert result.diagnostics == ("parse_failed",)


def test_archive_guards_reject_hostile_members() -> None:
    from zipfile import ZipInfo

    from src.integrations.dart.document_statements import _is_unsafe_name, _member_guard

    def info(name: str, *, size: int = 10, symlink: bool = False, encrypted: bool = False) -> ZipInfo:
        entry = ZipInfo(name)
        entry.file_size = size
        if symlink:
            entry.external_attr = 0o120000 << 16
        if encrypted:
            entry.flag_bits = 0x1
        return entry

    assert _member_guard([info(f"f{i}.xml") for i in range(33)]) == "too_many_members"
    assert _member_guard([info("a.xml"), info("a.xml")]) == "duplicate_member"
    assert _member_guard([info("../escape.xml")]) == "unsafe_member_path"
    assert _member_guard([info("link.xml", symlink=True)]) == "symlink_member"
    assert _member_guard([info("secret.xml", encrypted=True)]) == "encrypted_member"
    assert _member_guard([info("big.xml", size=17 * 1024 * 1024)]) == "member_too_large"
    assert _member_guard([info(f"f{i}.xml", size=14 * 1024 * 1024) for i in range(5)]) == "expanded_too_large"
    assert _member_guard([info("nested/"), info("20200101000001.xml")]) is None
    assert _is_unsafe_name("") is True
    assert _is_unsafe_name("/20200101000001.xml") is False


def test_amount_notations() -> None:
    from src.integrations.dart.document_statements import _parse_amount

    assert _parse_amount("(1,234)") == -1234
    assert _parse_amount("△1,234") == -1234
    assert _parse_amount("-1,234") == -1234
    assert _parse_amount("1,234") == 1234
    assert _parse_amount("-") is None
    assert _parse_amount("―") is None
    assert _parse_amount("－") is None  # noqa: RUF001
    assert _parse_amount("") is None
    assert _parse_amount("1,234.5") is None
    assert _parse_amount("12%") is None
    assert _parse_amount("1,234)") is None
    assert _parse_amount("(주석 1)") is None


def test_heading_classification_and_units() -> None:
    from src.integrations.dart.document_statements import _classify_heading, _unit_of

    assert _classify_heading("연결 재무상태표") == "BS"
    assert _classify_heading("대차대조표") == "BS"
    assert _classify_heading("현금흐름표") == "CF"
    assert _classify_heading("자본변동표") == "SCE"
    assert _classify_heading("포괄손익계산서") == "IS"
    assert _classify_heading("재무상태표에 대한 주석") is None
    assert _classify_heading("재무상태표 " + "가" * 60) is None
    assert _classify_heading("연결재무제표") is None
    assert _unit_of("(단위 : 백만원)") == 1_000_000
    assert _unit_of("단위:천원") == 1_000
    assert _unit_of("단위 : 십억원") == 1_000_000_000
    assert _unit_of("단위 : 원") == 1
    assert _unit_of("(백만원)") == 1_000_000
    assert _unit_of("금액 표시 없음") is None


def test_section_extraction_stops_at_next_title() -> None:
    from src.integrations.dart.document_statements import _extract_section

    markup = (
        '<TITLE ATOC="Y" AASSOCNOTE="D-0-3-1-0">개요</TITLE>preface'
        '<TITLE ATOC="Y" AASSOCNOTE="D-0-3-2-0">연결재무제표</TITLE>middle'
        '<TITLE ATOC="Y" AASSOCNOTE="D-0-3-4-0">재무제표</TITLE>tail'
    )
    assert _extract_section(markup, CFS) == "연결재무제표</TITLE>middle"
    assert _extract_section(markup, OFS) == "재무제표</TITLE>tail"
    assert _extract_section(markup, "D-0-3-9-0") is None


def test_duplicate_statement_body_withholds_kind() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    two = bs_section(extra=[para("재무상태표"), para("(단위 : 원)"), table(bs_rows())])
    result = parse_filing_document(make_archive({"F.xml": two}), reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "ambiguous_statement:BS" in result.diagnostics


def test_duplicate_income_body_drops_income_only() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
            ["매출총이익", "600", "550"],
            ["기타", "1", "2"],
        ]
    )
    is_blocks = [para("손익계산서"), para("(단위 : 원)"), body]
    archive = make_archive({"F.xml": bs_section(extra=is_blocks + is_blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "ambiguous_statement:IS" in result.diagnostics
    assert "sales" not in facts_by_name(result)


def test_capital_change_statement_is_skipped() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    sce = [para("자본변동표"), para("(단위 : 원)"), table(bs_rows())]
    archive = make_archive({"F.xml": bs_section(extra=sce)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_missing_statement_is_flagged() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
            ["매출총이익", "600", "550"],
            ["기타", "1", "2"],
        ]
    )
    archive = make_archive({"F.xml": section(OFS, "재무제표", [para("손익계산서"), para("(단위 : 원)"), body])})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "missing_statement:BS" in result.diagnostics


def test_missing_current_column_withholds_kind() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    rows = [
        ["과목", "전기", "전전기"],
        ["자산총계", "1,000", "900"],
        ["부채총계", "400", "350"],
        ["자본총계", "600", "550"],
        ["현금및현금성자산", "100", "90"],
        ["이익잉여금", "50", "40"],
    ]
    archive = make_archive({"F.xml": bs_section(rows=rows)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "missing_current_column:BS" in result.diagnostics


def test_missing_unit_on_income_drops_income_only() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
            ["매출총이익", "600", "550"],
            ["기타", "1", "2"],
        ]
    )
    is_blocks = [para("손익계산서"), para("금액 표시 없음"), body]
    archive = make_archive({"F.xml": bs_section(extra=is_blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "missing_unit:IS" in result.diagnostics
    assert "sales" not in facts_by_name(result)


def test_conflicting_duplicate_fact_is_withheld() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    rows = [
        ["과목", "제 35 기", "제 34 기"],
        ["매출액", "1,000", "900"],
        ["매출액", "1,001", "900"],
        ["매출액", "1,000", "900"],
        ["영업이익", "100", "90"],
        ["당기순이익", "80", "70"],
        ["매출총이익", "600", "550"],
    ]
    body = table(rows)
    is_blocks = [para("손익계산서"), para("(단위 : 원)"), body]
    archive = make_archive({"F.xml": bs_section(extra=is_blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "ambiguous_fact:sales" in result.diagnostics
    assert "sales" not in facts_by_name(result)
    assert facts_by_name(result)["operating_profit"] == (100, "annual")


def test_identical_duplicate_fact_is_kept_once() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    rows = [
        ["과목", "제 35 기", "제 34 기"],
        ["매출액", "1,000", "900"],
        ["매출액", "1,000", "900"],
        ["영업이익", "100", "90"],
        ["당기순이익", "80", "70"],
        ["매출총이익", "600", "550"],
        ["기타", "1", "2"],
    ]
    body = table(rows)
    is_blocks = [para("손익계산서"), para("(단위 : 원)"), body]
    archive = make_archive({"F.xml": bs_section(extra=is_blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["sales"] == (1000, "annual")


def test_new_layout_heading_table_is_read() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    heading = table([["재 무 상 태 표", ""], ["2019년 12월 31일 현재", "(단위 : 천원)"]])
    stray = table([["안내", "문구"]])
    notes = table([[f"r{i}", "x", "y"] for i in range(8)])
    archive = make_archive(
        {"F.xml": section(OFS, "재무제표", [stray, notes, heading, para("주식회사 테스트"), table(bs_rows())])}
    )
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert result.statements.unit_multipliers == {"BS": 1_000}
    assert facts_by_name(result)["assets"] == (1_000_000, "point_in_time")


def test_empty_and_ragged_rows_are_ignored() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    markup = (
        section(OFS, "재무제표", [para("재무상태표"), para("(단위 : 원)"), para("2019년 12월 31일 현재")])
        + "<TABLE><TR></TR>"
        + "".join(
            "<TR>" + "".join(f"<TD>{cell}</TD>" for cell in row) + "</TR>"
            for row in [
                ["과목", "제 35 기", "제 34 기"],
                ["자산총계"],
                ["자산총계", "1,000", "900"],
                ["부채총계", "400", "350"],
                ["자본총계", "600", "550"],
                ["현금및현금성자산", "100", "90"],
            ]
        )
        + "</TABLE>"
    )
    result = parse_filing_document(make_archive({"F.xml": markup}), reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_parenthesized_revenue_label_maps() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    rows = [
        ["과목", "제 35 기", "제 34 기"],
        ["수익(매출액)", "1,000", "900"],
        ["영업이익", "100", "90"],
        ["당기순이익", "80", "70"],
        ["매출총이익", "600", "550"],
        ["기타", "1", "2"],
    ]
    body = table(rows)
    is_blocks = [para("손익계산서"), para("(단위 : 원)"), body]
    archive = make_archive({"F.xml": bs_section(extra=is_blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["sales"] == (1000, "annual")


def test_committed_fixtures_match_expected_output() -> None:
    import glob
    import json
    import os

    from src.integrations.dart.document_statements import REVISION as DOCUMENT_STATEMENTS_REVISION
    from src.integrations.dart.document_statements import parse_filing_document

    paths = sorted(glob.glob("tests/fixtures/dart_documents/*.zip"))
    assert paths, "no committed fixtures"
    total = 0
    assert len(paths) <= 20
    for archive_path in paths:
        total += os.path.getsize(archive_path)
        with open(archive_path, "rb") as handle:
            archive_bytes = handle.read()
        with open(os.path.splitext(archive_path)[0] + ".expected.json", encoding="utf-8") as handle:
            expected = json.load(handle)
        identity = expected["identity"]
        result = parse_filing_document(
            archive_bytes, reprt_code=identity["reprt_code"], biz_year=identity["biz_year"]
        )
        if result.statements is None:
            actual = None
        else:
            statements = result.statements
            actual = {
                "consolidated": statements.consolidated,
                "period_end": statements.period_end.isoformat(),
                "report_kind": statements.report_kind,
                "unit_multipliers": dict(sorted(statements.unit_multipliers.items())),
                "facts": [
                    {"fact": fact.fact, "value": fact.value, "basis": fact.basis.value, "label": fact.label}
                    for fact in statements.facts
                ],
                "checks": list(statements.checks),
            }
        assert expected["parser_version"] == DOCUMENT_STATEMENTS_REVISION
        assert actual == expected["statements"], archive_path
        assert list(result.diagnostics) == expected["diagnostics"], archive_path
        assert expected["expected_values_source"], archive_path
    assert total <= 8 * 1024 * 1024


def test_second_quarter_marker_names() -> None:
    from src.integrations.dart.document_statements import _markers_in

    assert _markers_in("제35기 2분기말") == {"half"}
    assert _markers_in("제35기 1분기말") == {"q1"}


def test_interstitial_unit_table_belongs_to_heading() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    blocks = [
        para("재무상태표"),
        table([["2019년 12월 31일 현재", ""], ["(단위 : 원)", ""]]),
        table(bs_rows()),
    ]
    archive = make_archive({"F.xml": flow_section(blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def q3_bs_with_is(extra: list[str], *, heading: str = "3분기재무상태표", when: str = "2019년 9월 30일 현재") -> bytes:
    return make_archive({"F.xml": bs_section(when=when, heading=heading, extra=extra)})


def test_unmarked_sub_header_withholds_income_basis() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["구분", "-", "-"],
            ["매출액", "250", "200"],
            ["영업이익", "80", "70"],
            ["당기순이익", "50", "40"],
            ["매출총이익", "150", "120"],
        ]
    )
    archive = q3_bs_with_is([para("3분기손익계산서"), para("(단위 : 원)"), body])
    result = parse_filing_document(archive, reprt_code="11014", biz_year="2019")

    assert result.statements is not None
    assert "missing_period_basis:IS" in result.diagnostics
    assert "sales" not in facts_by_name(result)


def test_period_row_quarter_word_selects_quarter() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제35기 반기", "제34기 반기"],
            ["구분", "-", "-"],
            ["매출액", "250", "200"],
            ["영업이익", "80", "70"],
            ["당기순이익", "50", "40"],
            ["매출총이익", "150", "120"],
        ]
    )
    archive = q3_bs_with_is(
        [para("손익계산서"), para("(단위 : 원)"), body], heading="반기재무상태표", when="2019년 6월 30일 현재"
    )
    result = parse_filing_document(archive, reprt_code="11012", biz_year="2019")

    assert facts_by_name(result)["sales"] == (250, "quarter")


def test_period_row_cumulative_word_selects_cumulative() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "제35기 누적", "제34기 누적"],
            ["구분", "-", "-"],
            ["매출액", "250", "200"],
            ["영업이익", "80", "70"],
            ["당기순이익", "50", "40"],
            ["매출총이익", "150", "120"],
        ]
    )
    archive = q3_bs_with_is([para("손익계산서"), para("(단위 : 원)"), body])
    result = parse_filing_document(archive, reprt_code="11014", biz_year="2019")

    assert facts_by_name(result)["sales"] == (250, "cumulative")


def test_income_without_period_header_is_dropped_only() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = table(
        [
            ["과목", "전기", "전전기"],
            ["매출액", "1,000", "900"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
            ["매출총이익", "600", "550"],
            ["기타", "1", "2"],
        ]
    )
    archive = make_archive({"F.xml": bs_section(extra=[para("손익계산서"), para("(단위 : 원)"), body])})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "missing_current_column:IS" in result.diagnostics
    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_empty_row_never_breaks_fact_extraction() -> None:
    from src.integrations.dart.document_statements import _extract_kind_facts, _StatementBody

    body = _StatementBody(
        region="(단위 : 원)",
        rows=((), ("과목", "제 35 기"), ("자산총계", "1,000"), ("부채총계", "400"), ("자본총계", "600")),
    )
    facts, _, _, _ = _extract_kind_facts("BS", body, 1, "11011")

    assert facts["assets"][0] == 1000


def test_archive_level_guards_withhold() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    crowded = make_archive({f"f{i}.xml": "x" for i in range(33)})
    assert parse_filing_document(crowded, reprt_code="11011", biz_year="2019").diagnostics == ("too_many_members",)
    escaped = make_archive({"../escape.xml": bs_section()})
    assert parse_filing_document(escaped, reprt_code="11011", biz_year="2019").diagnostics == ("unsafe_member_path",)


def _five_row_caption(*, unit: str = "(단위 : 원)") -> str:
    return table(
        [
            ["제 35 기 결산공고", ""],
            ["2019년 12월 31일 현재", ""],
            ["주식회사 테스트", ""],
            ["제 35 기", "제 34 기"],
            [unit, ""],
        ]
    )


def _simple_is_body(*, sales: str = "1,000") -> str:
    return table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", sales, "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
            ["기타", "1", "2"],
        ]
    )


def _simple_cf_body() -> str:
    return table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["영업활동으로인한현금흐름", "500", "400"],
            ["유형자산의취득", "(30)", "(20)"],
            ["기말현금및현금성자산", "100", "90"],
            ["당기순이익", "20", "10"],
            ["감가상각비", "5", "4"],
        ]
    )


def test_caption_table_of_body_length_is_not_the_income_body() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    extra = [para("손익계산서"), _five_row_caption(), _simple_is_body()]
    archive = make_archive({"F.xml": bs_section(extra=extra)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["sales"] == (1000, "annual")
    assert "missing_unit:IS" not in result.diagnostics


def test_caption_table_does_not_hide_cash_flow_unit() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    extra = [para("현금흐름표"), _five_row_caption(), _simple_cf_body()]
    archive = make_archive({"F.xml": bs_section(extra=extra)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["operating_cash_flow"] == (500, "annual")
    assert "missing_unit:CF" not in result.diagnostics


def test_consolidated_section_verifies_instead_of_falling_back() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    cfs = section(
        CFS,
        "연결재무제표",
        [
            para("재무상태표"),
            _five_row_caption(),
            para("2019년 12월 31일 현재"),
            table(bs_rows()),
        ],
    )
    ofs = bs_section(
        rows=bs_rows(assets="2,000", debt="800", equity="1,200", cash="200", prior=("900", "350", "550", "90")),
    )
    result = parse_filing_document(make_archive({"F.xml": cfs + ofs}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert result.statements.consolidated is True
    assert result.statements.unit_multipliers.get("BS") == 1
    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_small_table_without_amounts_never_ends_the_heading() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    filler = table([["안내", "문구"], ["주식회사 테스트", "1건"]])
    blocks = [para("재무상태표"), para("(단위 : 원)"), para("2019년 12월 31일 현재"), filler, filler, table(bs_rows())]
    archive = make_archive({"F.xml": flow_section(blocks)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_prior_period_table_beside_current_one_is_not_ambiguous() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    prior = table(
        [
            ["과목", "34(전)기", "33(전)기"],
            ["매출액", "900", "800"],
            ["매출총이익", "550", "500"],
            ["영업이익", "90", "80"],
            ["당기순이익", "70", "60"],
            ["기타", "2", "1"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), _simple_is_body(), prior]
    archive = make_archive({"F.xml": bs_section(extra=extra)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["sales"] == (1000, "annual")
    assert "ambiguous_statement:IS" not in result.diagnostics


def test_period_parenthetical_with_quarter_digit_is_a_period() -> None:
    from src.integrations.dart.document_statements import _current_columns, parse_filing_document

    columns, _ = _current_columns((("과목", "제 68(당) 3분기", "제 67(전) 3분기"),))
    assert columns == [1]
    columns, _ = _current_columns((("과목", "제 68 (당)기 3 분기", "제 67 (전)기 3 분기"),))
    assert columns == [1]

    body = table(
        [
            ["과목", "제 68(당) 3분기", "제 67(전) 3분기"],
            ["구분", "3개월", "누적"],
            ["매출액", "250", "200"],
            ["매출총이익", "150", "120"],
            ["영업이익", "80", "70"],
            ["당기순이익", "50", "40"],
        ]
    )
    archive = q3_bs_with_is([para("3분기손익계산서"), para("(단위 : 원)"), body])
    result = parse_filing_document(archive, reprt_code="11014", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["sales"] == (250, "quarter")


def test_current_label_header_selects_current_columns() -> None:
    from src.integrations.dart.document_statements import _current_columns, parse_filing_document

    rows = [
        ["과목", "당분기말", "당분기말", "전기말"],
        ["자산총계", "1,000", "", "900"],
        ["부채총계", "400", "", "350"],
        ["자본총계", "600", "", "550"],
        ["현금및현금성자산", "100", "", "90"],
        ["이익잉여금", "50", "", "40"],
    ]
    columns, row_index = _current_columns(tuple(tuple(cell for cell in row) for row in rows))
    assert columns == [1, 2]
    assert row_index == 0
    archive = make_archive({"F.xml": bs_section(when="2019년 9월 30일 현재", heading="분기재무상태표", rows=rows)})
    result = parse_filing_document(archive, reprt_code="11014", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_year_label_header_selects_highest_year() -> None:
    from src.integrations.dart.document_statements import _current_columns, parse_filing_document

    rows = (
        ("과목", "2016회계연도", "2016회계연도", "2015회계연도"),
        ("자산총계", "1,000", "", "900"),
        ("부채총계", "400", "", "350"),
        ("자본총계", "600", "", "550"),
        ("현금및현금성자산", "100", "", "90"),
    )
    columns, _ = _current_columns(rows)
    assert columns == [1, 2]
    full_rows = [list(row) for row in rows] + [["이익잉여금", "50", "", "40"]]
    archive = make_archive({"F.xml": bs_section(rows=full_rows)})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["assets"] == (1000, "point_in_time")


def test_numbered_header_keeps_priority_over_current_label() -> None:
    from src.integrations.dart.document_statements import _current_columns

    columns, row_index = _current_columns(
        (
            ("과목", "제 35 기", "제 34 기"),
            ("당기", "1,000", "900"),
            ("부채총계", "400", "350"),
        )
    )
    assert columns == [1]
    assert row_index == 0


def test_headers_without_current_marker_still_withhold() -> None:
    from src.integrations.dart.document_statements import _current_columns, parse_filing_document

    columns, row_index = _current_columns((("과목", "당해", "전해"),))
    assert (columns, row_index) == ([], -1)
    rows = [
        ["과목", "당해", "전해"],
        ["자산총계", "1,000", "900"],
        ["부채총계", "400", "350"],
        ["자본총계", "600", "550"],
        ["현금및현금성자산", "100", "90"],
        ["이익잉여금", "50", "40"],
    ]
    result = parse_filing_document(make_archive({"F.xml": bs_section(rows=rows)}), reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "missing_current_column:BS" in result.diagnostics


def test_spaced_date_digits_are_recovered() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2016년 0 9월 30일 현재", heading="분기재무상태표")})
    result = parse_filing_document(archive, reprt_code="11014", biz_year="2016")

    assert result.statements is not None
    assert result.statements.period_end.isoformat() == "2016-09-30"


def test_ordinary_dates_are_still_found() -> None:
    from datetime import date

    from src.integrations.dart.document_statements import _candidate_dates

    assert date(2019, 12, 31) in _candidate_dates("2019.12.31 현재", (), [])
    assert date(2019, 12, 31) in _candidate_dates("2019년 12월 31일 현재", (), [])


def test_digits_without_separators_never_form_a_date() -> None:
    from src.integrations.dart.document_statements import _candidate_dates

    assert _candidate_dates("2016 09 30", (), []) == []
    assert _candidate_dates("20160930", (), []) == []


def test_fiscal_quarter_marker_is_fiscal_mismatch() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2017년 3월 31일 현재", heading="3분기말 재무상태표")})
    result = parse_filing_document(archive, reprt_code="11013", biz_year="2017")

    assert result.statements is None
    assert result.diagnostics == ("missing_section:D-0-3-2-0", "non_december_fiscal_year")


def test_off_quarter_fiscal_year_end_is_fiscal_mismatch() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2019년 3월 31일 현재")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "non_december_fiscal_year" in result.diagnostics


def test_wrong_year_filing_stays_period_mismatch() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    archive = make_archive({"F.xml": bs_section(when="2018년 12월 31일 현재")})
    result = parse_filing_document(archive, reprt_code="11011", biz_year="2019")

    assert result.statements is None
    assert "period_mismatch" in result.diagnostics
    assert "non_december_fiscal_year" not in result.diagnostics


def _is_body(rows: list[list[str]]) -> str:
    return table(rows)


def test_plain_and_comprehensive_income_tables_are_not_ambiguous() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    plain = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    twin = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), plain, para("포괄손익계산서"), para("(단위 : 원)"), twin]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["sales"] == (1000, "annual")
    assert "ambiguous_statement:IS" not in result.diagnostics


def test_comprehensive_only_filing_keeps_its_income_statement() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("포괄손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["sales"] == (1000, "annual")


def test_two_plain_income_tables_stay_ambiguous() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body, para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "ambiguous_statement:IS" in result.diagnostics
    assert "sales" not in facts_by_name(result)


def test_disagreeing_twin_drops_the_shared_fact() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    plain = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["영업이익", "100", "90"],
            ["당기순이익", "100", "70"],
            ["매출총이익", "600", "550"],
        ]
    )
    twin = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["영업이익", "100", "90"],
            ["당기순이익", "90", "70"],
            ["매출총이익", "600", "550"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), plain, para("포괄손익계산서"), para("(단위 : 원)"), twin]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "ambiguous_fact:net_income" in result.diagnostics
    assert "net_income" not in facts_by_name(result)
    assert facts_by_name(result)["sales"] == (1000, "annual")


def test_twin_only_facts_are_not_added() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    plain = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["당기순이익", "80", "70"],
        ]
    )
    twin = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), plain, para("포괄손익계산서"), para("(단위 : 원)"), twin]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "operating_profit" not in facts_by_name(result)
    assert facts_by_name(result)["sales"] == (1000, "annual")


def test_gross_profit_identity_passes_with_bracketed_cost() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출원가", "(700)", "(650)"],
            ["매출총이익", "300", "250"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["sales"] == (1000, "annual")
    assert facts_by_name(result)["gross_profit"] == (300, "annual")
    assert "gross_profit_identity" in result.statements.checks


def test_sign_flipped_gross_profit_withholds_revenue_and_gross_profit() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "8,686", "8,000"],
            ["매출원가", "8,064", "7,500"],
            ["매출총이익", "(622)", "500"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "sales" not in facts_by_name(result)
    assert "gross_profit" not in facts_by_name(result)
    assert "identity_failed:gross_profit" in result.diagnostics
    assert facts_by_name(result)["operating_profit"] == (100, "annual")


def test_missing_cost_of_sales_skips_the_identity() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["sales"] == (1000, "annual")
    assert facts_by_name(result)["gross_profit"] == (600, "annual")
    assert result.statements is not None
    assert "gross_profit_identity" not in result.statements.checks


def test_bracketed_operating_loss_is_read_as_printed() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업손실", "(6,148)", "(5,000)"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["operating_profit"] == (-6148, "annual")


def test_positive_operating_loss_needs_the_identity() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출원가", "700", "650"],
            ["매출총이익", "300", "250"],
            ["판매비와관리비", "800", "750"],
            ["영업손실", "500", "400"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert facts_by_name(result)["operating_profit"] == (-500, "annual")


def test_positive_operating_loss_without_confirmation_is_withheld() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["영업손실", "500", "400"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert "operating_profit" not in facts_by_name(result)
    assert "ambiguous_loss_sign:operating_profit" in result.diagnostics


def test_contradicting_identity_withholds_the_loss() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "2,000", "1,900"],
            ["매출원가", "700", "650"],
            ["매출총이익", "1,300", "1,250"],
            ["판매비와관리비", "800", "750"],
            ["영업손실", "500", "400"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert "operating_profit" not in facts_by_name(result)
    assert "ambiguous_loss_sign:operating_profit" in result.diagnostics


def test_positive_net_loss_is_always_withheld() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순손실", "900", "800"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert "net_income" not in facts_by_name(result)
    assert "ambiguous_loss_sign:net_income" in result.diagnostics

    bracketed = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순손실", "(900)", "(800)"],
        ]
    )
    extra_bracketed = [para("손익계산서"), para("(단위 : 원)"), bracketed]
    kept = parse_filing_document(
        make_archive({"F.xml": bs_section(extra=extra_bracketed)}), reprt_code="11011", biz_year="2019"
    )

    assert facts_by_name(kept)["net_income"] == (-900, "annual")


def test_loss_row_and_normal_row_conflict() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익(손실)", "100", "90"],
            ["영업손실", "(50)", "(40)"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert "operating_profit" not in facts_by_name(result)
    assert "ambiguous_fact:operating_profit" in result.diagnostics


def test_support_lines_outside_income_statement_are_ignored() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    is_body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["영업이익", "100", "90"],
            ["당기순이익", "80", "70"],
        ]
    )
    cf_body = table(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["영업활동으로인한현금흐름", "500", "400"],
            ["유형자산의취득", "(30)", "(20)"],
            ["기말현금및현금성자산", "100", "90"],
            ["매출원가", "700", "650"],
        ]
    )
    extra = [
        para("손익계산서"),
        para("(단위 : 원)"),
        is_body,
        para("현금흐름표"),
        para("(단위 : 원)"),
        cf_body,
    ]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "gross_profit_identity" not in result.statements.checks
    assert facts_by_name(result)["sales"] == (1000, "annual")


def test_loss_row_without_single_amount_is_skipped() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["당기순이익", "80", "70"],
            ["영업손실", "-", "-"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "operating_profit" not in facts_by_name(result)
    assert "ambiguous_loss_sign:operating_profit" not in result.diagnostics


def test_repeated_unconfirmed_loss_row_reports_once() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["당기순이익", "80", "70"],
            ["영업손실", "500", "400"],
            ["영업손실", "500", "400"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert "operating_profit" not in facts_by_name(result)
    assert list(result.diagnostics).count("ambiguous_loss_sign:operating_profit") == 1


def test_loss_only_primary_takes_basis_from_twin() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    primary = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["영업손실", "(100)", "(90)"],
            ["당기순손실", "(200)", "(180)"],
            ["매출원가", "700", "650"],
        ]
    )
    twin = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["매출액", "1,000", "900"],
            ["매출총이익", "600", "550"],
            ["당기순이익", "80", "70"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), primary, para("포괄손익계산서"), para("(단위 : 원)"), twin]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert facts_by_name(result)["operating_profit"] == (-100, "annual")


def test_loss_only_body_without_basis_withholds_silently() -> None:
    from src.integrations.dart.document_statements import parse_filing_document

    body = _is_body(
        [
            ["과목", "제 35 기", "제 34 기"],
            ["영업손실", "(100)", "(90)"],
            ["당기순손실", "(200)", "(180)"],
            ["매출원가", "700", "650"],
        ]
    )
    extra = [para("손익계산서"), para("(단위 : 원)"), body]
    result = parse_filing_document(make_archive({"F.xml": bs_section(extra=extra)}), reprt_code="11011", biz_year="2019")

    assert result.statements is not None
    assert "operating_profit" not in facts_by_name(result)
    assert "net_income" not in facts_by_name(result)
