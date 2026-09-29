"""Earnings-release parsing invariants: titles, tables, units, periods and lag gates."""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest

from src.integrations.dart.earnings_release import (
    EarningsReleaseKind,
    EarningsReleaseParseError,
    ReleaseBasis,
    ReleaseSpan,
    classify_earnings_release_title,
    parse_earnings_release,
)

RCEPT = "20190131800987"
CORP = "00126380"


def _zip(rows: list[list[str]]) -> bytes:
    body = "<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("document.xml", body)
    return output.getvalue()


def _preliminary_rows(
    *,
    caption: str = "1. 연결실적내용",
    unit: str = "단위 : 백만원, %",
    period: str = "('18.4Q)",
    sales_quarter: str = "321,223",
    sales_prior: str = "87,836",
    sales_cumulative: str = "1,365,439",
) -> list[list[str]]:
    return [
        ["※ 동 정보는 잠정치로서 향후 확정치와는 다를 수 있음."],
        [caption, unit],
        ["구분", "당기실적", "전기실적", "전기대비증감율(%)", "전년동기실적", "전년동기대비증감율(%)"],
        [period, "('18.3Q)", "('17.4Q)"],
        ["매출액", "당해실적", sales_quarter, "376,466", "-14.7", sales_prior, "265.7"],
        ["누계실적", sales_cumulative, "1,044,216", "-", sales_prior, "1,454.5"],
        ["영업이익", "당해실적", "-2,815", "30,962", "적자전환", "-9,181", "69.3"],
        ["누계실적", "45,325", "48,140", "-", "-9,181", "흑자전환"],
        ["2. 정보제공내역", "정보제공자", "기획부", "", "", ""],
    ]


def _parse(rows: list[list[str]], **kwargs) -> object:  # type: ignore[no-untyped-def]
    params = {
        "archive_bytes": _zip(rows),
        "rcept_no": RCEPT,
        "corp_code": CORP,
        "received_on": date(2019, 1, 31),
        "report_nm": "연결재무제표기준영업(잠정)실적(공정공시)",
        "max_filing_lag_days": 135,
    }
    params.update(kwargs)
    return parse_earnings_release(**params)  # type: ignore[arg-type]


def test_preliminary_consolidated_quarter_and_cumulative_in_krw() -> None:
    release = _parse(_preliminary_rows())

    assert (release.fiscal_year, release.fiscal_quarter) == (2018, 4)
    assert release.basis is ReleaseBasis.CONSOLIDATED
    assert release.kind is EarningsReleaseKind.PRELIMINARY
    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.QUARTER)].current_krw == 321223 * 1e6
    assert by_key[("sales", ReleaseSpan.QUARTER)].prior_year_krw == 87836 * 1e6
    assert by_key[("sales", ReleaseSpan.CUMULATIVE)].current_krw == 1365439 * 1e6


def test_negative_and_missing_amounts() -> None:
    release = _parse(
        _preliminary_rows(sales_quarter="-2,815", sales_prior="△1,200", sales_cumulative="-")
    )

    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.QUARTER)].current_krw == -2815 * 1e6
    assert by_key[("sales", ReleaseSpan.QUARTER)].prior_year_krw == -1200 * 1e6
    assert by_key[("sales", ReleaseSpan.CUMULATIVE)].current_krw is None


def test_period_label_variants_resolve() -> None:
    for label in ("(2022.4Q)", "(22년 4분기 )", "('18.4Q)"):
        release = _parse(
            _preliminary_rows(period=label),
            received_on=date(2023, 1, 31) if "2022" in label or "22년" in label else date(2019, 1, 31),
        )
        year = 2022 if label != "('18.4Q)" else 2018
        assert (release.fiscal_year, release.fiscal_quarter) == (year, 4)


def test_unrecognized_period_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(period="(당기)"))

    assert exc_info.value.reason == "unrecognized_period"


def test_unknown_unit_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(unit="단위 : 달러"))

    assert exc_info.value.reason == "unknown_unit"


def test_basis_conflict_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(caption="1. 실적내용"))

    assert exc_info.value.reason == "basis_conflict"


def test_correction_with_diff_only_body_withheld() -> None:
    rows = [
        ["1. 정정관련 공시서류", "연결재무제표기준영업(잠정)실적(공정공시)"],
        ["정정항목", "정정전", "정정후"],
        ["매출액(당해실적)", "240,204", "245,695"],
    ]
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(rows, report_nm="[기재정정]연결재무제표기준영업(잠정)실적(공정공시)")

    assert exc_info.value.reason == "no_result_table"


def test_correction_exempt_from_lag_bound() -> None:
    release = _parse(
        _preliminary_rows(),
        report_nm="[기재정정]연결재무제표기준영업(잠정)실적(공정공시)",
        received_on=date(2020, 2, 4),
    )

    assert release.is_correction is True
    assert (release.fiscal_year, release.fiscal_quarter) == (2018, 4)


def test_original_outside_lag_window_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), received_on=date(2019, 7, 20))

    assert exc_info.value.reason == "period_lag"

    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), received_on=date(2018, 12, 1))

    assert exc_info.value.reason == "period_lag"


def _profit_change_rows() -> list[list[str]]:
    return [
        ["1. 재무제표의 종류", "연결"],
        ["2. 매출액 또는 손익구조 변동내용(단위:천원)", "당해사업연도", "직전사업연도", "증감금액", "증감비율(%)"],
        ["- 매출액(재화의 판매 및 용역의 제공에 따른 수익액에 한함)", "1,435,754,699", "1,512,114,565", "-76,359,866", "-5.0%"],
        ["- 영업이익", "127,086,173", "126,277,061", "809,112", "0.6%"],
        ["- 법인세비용차감전계속사업이익", "102,501,357", "146,019,478", "-43,518,121", "-29.8%"],
        ["- 당기순이익", "77,889,420", "124,509,305", "-46,619,885", "-37.4%"],
        ["- 대규모법인여부", "해당"],
    ]


def test_profit_change_annual_parsed() -> None:
    release = _parse(
        _profit_change_rows(),
        report_nm="매출액또는손익구조30%(대규모법인은15%)이상변경",
        received_on=date(2021, 2, 1),
    )

    assert (release.fiscal_year, release.fiscal_quarter) == (2020, 4)
    assert release.basis is ReleaseBasis.CONSOLIDATED
    assert release.period_label == ""
    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("operating_profit", ReleaseSpan.ANNUAL)].current_krw == 127086173 * 1e3
    assert by_key[("operating_profit", ReleaseSpan.ANNUAL)].prior_year_krw == 126277061 * 1e3


def test_profit_change_off_season_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(
            _profit_change_rows(),
            report_nm="매출액또는손익구조30%(대규모법인은15%)이상변경",
            received_on=date(2021, 6, 15),
        )

    assert exc_info.value.reason == "off_season"


def test_title_classification() -> None:
    assert (
        classify_earnings_release_title("연결재무제표기준영업(잠정)실적(공정공시)")
        is EarningsReleaseKind.PRELIMINARY
    )
    assert (
        classify_earnings_release_title("[기재정정]영업(잠정)실적(공정공시)")
        is EarningsReleaseKind.PRELIMINARY
    )
    assert (
        classify_earnings_release_title("매출액또는손익구조30%(대규모법인15%)미만변경(자율공시)")
        is EarningsReleaseKind.PROFIT_CHANGE
    )
    assert classify_earnings_release_title("연결재무제표기준영업(잠정)실적(공정공시)(자회사의 주요경영사항)") is None
    assert classify_earnings_release_title("연결재무제표기준영업실적등에대한전망(공정공시)") is None
    assert classify_earnings_release_title("사업보고서 (2018.12)") is None


def test_empty_and_attachment_marked_titles() -> None:
    assert classify_earnings_release_title("") is None
    assert classify_earnings_release_title("   ") is None
    assert (
        classify_earnings_release_title("[첨부추가]연결재무제표기준영업(잠정)실적(공정공시)")
        is EarningsReleaseKind.PRELIMINARY
    )


def test_non_release_title_and_malformed_ids_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), report_nm="사업보고서 (2018.12)")
    assert exc_info.value.reason == "not_an_archive"

    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), rcept_no="123")
    assert exc_info.value.reason == "not_an_archive"

    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), corp_code="123")
    assert exc_info.value.reason == "not_an_archive"


def test_unreadable_archives_withheld() -> None:
    for bad in (b"", b"not a zip"):
        with pytest.raises(EarningsReleaseParseError) as exc_info:
            _parse(_preliminary_rows(), archive_bytes=bad)
        assert exc_info.value.reason == "not_an_archive"


def test_hostile_members_withheld() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../evil.xml", "<table></table>")
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), archive_bytes=output.getvalue())
    assert exc_info.value.reason == "not_an_archive"

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("document.xml", "plain text without markup")
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), archive_bytes=output.getvalue())
    assert exc_info.value.reason == "not_an_archive"

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("document.xml", '<!DOCTYPE x [<!ENTITY x "y">]><table></table>')
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), archive_bytes=output.getvalue())
    assert exc_info.value.reason == "not_an_archive"


def test_oversized_archive_withheld() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(33):
            archive.writestr(f"document{index}.xml", "<table></table>")
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), archive_bytes=output.getvalue())
    assert exc_info.value.reason == "not_an_archive"


def test_parenthesized_negative_and_bare_won_unit() -> None:
    rows = _preliminary_rows()
    rows[4] = ["매출액", "당해실적", "(1,200)", "376,466", "-14.7", "87,836", "265.7"]
    release = _parse(rows)

    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.QUARTER)].current_krw == -1200 * 1e6


def test_separate_preliminary_parsed_in_won() -> None:
    release = _parse(
        _preliminary_rows(caption="1. 실적내용", unit="단위 : 원"),
        report_nm="영업(잠정)실적(공정공시)",
    )

    assert release.basis is ReleaseBasis.SEPARATE
    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.QUARTER)].current_krw == 321223.0


def test_three_digit_year_label_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(period="(123.4Q)"))

    assert exc_info.value.reason == "unrecognized_period"


def test_preliminary_without_header_or_metrics_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse([["1. 연결실적내용", "단위 : 백만원, %"], ["매출액", "100"]])
    assert exc_info.value.reason == "no_result_table"

    unknown = [
        ["1. 연결실적내용", "단위 : 백만원, %"],
        ["구분", "당기실적", "전기실적", "전기대비증감율(%)", "전년동기실적", "전년동기대비증감율(%)"],
        ["('18.4Q)", "('18.3Q)", "('17.4Q)"],
        ["지급수수료", "당해실적", "100", "90", "11.1", "80", "25.0"],
        ["-"],
    ]
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(unknown)
    assert exc_info.value.reason == "no_metrics"


def test_stray_cumulative_row_inherits_nothing() -> None:
    rows = _preliminary_rows()
    rows.insert(4, ["누계실적", "9,999", "8,888", "-", "7,777", "28.6"])
    release = _parse(rows)

    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.CUMULATIVE)].current_krw == 1365439 * 1e6


def test_correction_before_quarter_end_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(
            _preliminary_rows(),
            report_nm="[첨부정정]연결재무제표기준영업(잠정)실적(공정공시)",
            received_on=date(2018, 12, 1),
        )

    assert exc_info.value.reason == "period_lag"


def test_profit_change_separate_basis_and_conflicts() -> None:
    rows = _profit_change_rows()
    rows[0] = ["1. 재무제표의 종류", "별도"]
    release = _parse(
        rows,
        report_nm="매출액또는손익구조30%(대규모법인은15%)이상변경",
        received_on=date(2021, 2, 1),
    )
    assert release.basis is ReleaseBasis.SEPARATE

    rows = _profit_change_rows()
    rows[0] = ["1. 재무제표의 종류", "기타"]
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(
            rows,
            report_nm="매출액또는손익구조30%(대규모법인은15%)이상변경",
            received_on=date(2021, 2, 1),
        )
    assert exc_info.value.reason == "basis_conflict"


def test_profit_change_without_table_or_metrics_or_unit_withheld() -> None:
    good = {
        "report_nm": "매출액또는손익구조30%(대규모법인은15%)이상변경",
        "received_on": date(2021, 2, 1),
    }
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse([["정정항목", "정정전", "정정후"]], **good)
    assert exc_info.value.reason == "no_result_table"

    rows = _profit_change_rows()
    rows[1] = ["2. 매출액 또는 손익구조 변동내용(단위:달러)", "당해사업연도", "직전사업연도", "증감금액", "증감비율(%)"]
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(rows, **good)
    assert exc_info.value.reason == "unknown_unit"

    rows = [
        ["1. 재무제표의 종류", "연결"],
        ["2. 매출액 또는 손익구조 변동내용(단위:천원)", "당해사업연도", "직전사업연도", "증감금액", "증감비율(%)"],
        ["- 대규모법인여부", "해당", "-", "-", "-"],
    ]
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(rows, **good)
    assert exc_info.value.reason == "no_metrics"


def test_free_text_and_empty_parens_amounts_are_none() -> None:
    release = _parse(_preliminary_rows(sales_quarter="(  )", sales_prior="적자전환"))

    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.QUARTER)].current_krw is None
    assert by_key[("sales", ReleaseSpan.QUARTER)].prior_year_krw is None


def test_directory_members_skipped() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("nested/", "")
        archive.writestr(
            "nested/document.xml",
            "<table><tr><td>1. 연결실적내용</td><td>단위 : 백만원, %</td></tr></table>",
        )
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), archive_bytes=output.getvalue())

    assert exc_info.value.reason == "no_result_table"


def test_non_date_received_on_withheld() -> None:
    with pytest.raises(EarningsReleaseParseError) as exc_info:
        _parse(_preliminary_rows(), received_on="2019-01-31")  # type: ignore[arg-type]

    assert exc_info.value.reason == "not_an_archive"


def test_profit_change_unit_fallback_row() -> None:
    rows = _profit_change_rows()
    rows[1] = ["2. 매출액 또는 손익구조 변동내용", "당해사업연도", "직전사업연도", "증감금액", "증감비율(%)"]
    rows.insert(2, ["(단위:천원)", "", "", "", ""])
    release = _parse(
        rows,
        report_nm="매출액또는손익구조30%(대규모법인은15%)이상변경",
        received_on=date(2021, 2, 1),
    )

    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("operating_profit", ReleaseSpan.ANNUAL)].current_krw == 127086173 * 1e3


def test_value_columns_follow_header_labels_not_fixed_positions() -> None:
    rows = [
        ["1. 연결실적내용", "단위 : 백만원, %"],
        ["구분", "당기실적", "전년동기실적", "전년동기대비증감율(%)"],
        ["('18.4Q)", "('17.4Q)"],
        ["매출액", "당해실적", "500", "400", "25.0"],
        ["누계실적", "2,000", "1,600", "25.0"],
    ]

    release = _parse(rows)

    by_key = {(item.metric, item.span): item for item in release.values}
    assert by_key[("sales", ReleaseSpan.QUARTER)].current_krw == 500 * 1e6
    assert by_key[("sales", ReleaseSpan.QUARTER)].prior_year_krw == 400 * 1e6
    assert by_key[("sales", ReleaseSpan.CUMULATIVE)].current_krw == 2000 * 1e6
    assert by_key[("sales", ReleaseSpan.CUMULATIVE)].prior_year_krw == 1600 * 1e6
