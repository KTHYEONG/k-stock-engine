"""Dividend decision title and archive parsing tests (offline fixtures only)."""

from __future__ import annotations

import io
import zipfile
from datetime import date


def make_archive(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()


def layout_2017_html() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><document>'
        "<table>"
        "<tr><td>구분</td><td>보통주</td><td>우선주</td></tr>"
        "<tr><td>주당 배당금(원)</td><td>300</td><td>350</td></tr>"
        "<tr><td>배당기준일</td><td>2017-12-31</td><td></td></tr>"
        "<tr><td>배당금지급 예정일자</td><td>2018-04-20</td><td></td></tr>"
        "</table></document>"
    )


def layout_2024_html() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><document>'
        "<table>"
        "<tr><th>항목</th><th>내용</th></tr>"
        "<tr><td>1주당 현금배당금(원) 보통주</td><td>1,000원</td></tr>"
        "<tr><td>배당기준일</td><td>2024.12.31</td></tr>"
        "<tr><td>배당금 지급예정일</td><td>2025.04.18</td></tr>"
        "</table></document>"
    )


def test_dividend_decision_title_matcher_flags_cash_only() -> None:
    from src.integrations.dart.dividend_decision import is_dividend_decision_title

    assert is_dividend_decision_title("현금ㆍ현물배당결정") is True
    assert is_dividend_decision_title("[기재정정]현금ㆍ현물배당결정") is True
    assert is_dividend_decision_title("현금배당결정") is True
    assert is_dividend_decision_title("주식배당결정") is False
    assert is_dividend_decision_title("") is False
    assert is_dividend_decision_title("현금·현물배당결정") is True


def test_parse_dividend_decision_reads_legacy_layout() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    archive = make_archive({"20180201001234.xml": layout_2017_html()})
    decision = parse_dividend_decision(
        archive_bytes=archive, rcept_no="20180201001234", corp_code="00126380", received_on=date(2018, 2, 1)
    )

    assert decision.record_date == date(2017, 12, 31)
    assert decision.pay_date == date(2018, 4, 20)
    assert decision.dps_common_krw == 300


def test_parse_dividend_decision_reads_current_layout() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    archive = make_archive({"20250203001234.xml": layout_2024_html()})
    decision = parse_dividend_decision(
        archive_bytes=archive, rcept_no="20250203001234", corp_code="00126380", received_on=date(2025, 2, 3)
    )

    assert decision.record_date == date(2024, 12, 31)
    assert decision.pay_date == date(2025, 4, 18)
    assert decision.dps_common_krw == 1000


def test_parse_dividend_decision_missing_dps_fails_closed() -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    html = (
        '<?xml version="1.0" encoding="utf-8"?><document><table>'
        "<tr><td>배당기준일</td><td>2024-12-31</td></tr>"
        "<tr><td>배당금 지급예정일</td><td>2025-04-18</td></tr>"
        "</table></document>"
    )
    with pytest.raises(PITDataError):
        parse_dividend_decision(
            archive_bytes=make_archive({"20250203001234.xml": html}),
            rcept_no="20250203001234",
            corp_code="00126380",
            received_on=date(2025, 2, 3),
        )


def test_parse_dividend_decision_reads_bom_and_cp949_members() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    bom_html = b"\xef\xbb\xbf" + layout_2017_html().encode("utf-8")
    decision = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": bom_html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )
    assert decision.dps_common_krw == 300

    cp949_html = layout_2017_html().replace("utf-8", "euc-kr").encode("cp949")
    decision_cp949 = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": cp949_html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )
    assert decision_cp949.record_date == date(2017, 12, 31)
    assert decision_cp949.dps_common_krw == 300


def test_parse_dividend_decision_reads_table_less_rows_and_undecided_pay() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    html = (
        '<?xml version="1.0" encoding="utf-8"?><document>'
        "<tr><td>보통주 300원</td></tr>"
        "<tr><td>배당기준일</td><td>2017-12-31</td></tr>"
        "<tr><td>배당금지급 예정일자</td><td>-</td></tr>"
        "</document>"
    )
    decision = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )

    assert decision.record_date == date(2017, 12, 31)
    assert decision.pay_date is None
    assert decision.dps_common_krw == 300
    assert decision.is_correction is False


def test_parse_dividend_decision_flags_correction_marker() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    html = layout_2017_html().replace("<document>", "<document><p>기재정정 사유</p>")
    decision = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )

    assert decision.is_correction is True


def test_parse_dividend_decision_rejects_unreadable_or_deless_archive() -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    with pytest.raises(PITDataError):
        parse_dividend_decision(
            archive_bytes=b"not a zip",
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError):
        parse_dividend_decision(
            archive_bytes=b"",
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError):
        parse_dividend_decision(
            archive_bytes=make_archive({"20180201001234.xml": "<document><table></table></document>"}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError):
        parse_dividend_decision(
            archive_bytes=make_archive({"20180201001234.xml": layout_2017_html()}),
            rcept_no="bad",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError, match="corp_code"):
        parse_dividend_decision(
            archive_bytes=make_archive({"20180201001234.xml": layout_2017_html()}),
            rcept_no="20180201001234",
            corp_code="bad",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError, match="received_on"):
        parse_dividend_decision(
            archive_bytes=make_archive({"20180201001234.xml": layout_2017_html()}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on="2018-02-01",  # type: ignore[arg-type]
        )


def test_parse_dividend_decision_rejects_unsafe_or_oversized_archives() -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    with pytest.raises(PITDataError, match="unsafe"):
        parse_dividend_decision(
            archive_bytes=make_archive({"../escape.xml": layout_2017_html()}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError, match="unsafe"):
        parse_dividend_decision(
            archive_bytes=make_archive({"doc.xml": "<!DOCTYPE foo>" + layout_2017_html()}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError, match="too many"):
        parse_dividend_decision(
            archive_bytes=make_archive({f"f{i}.xml": "<document/>" for i in range(33)}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError, match="unreadable"):
        parse_dividend_decision(
            archive_bytes=make_archive({"doc.xml": b"\x80\x81\xff\xfe"}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )
    with pytest.raises(PITDataError, match="unreadable"):
        parse_dividend_decision(
            archive_bytes=make_archive({"doc.xml": "plain words without markup"}),
            rcept_no="20180201001234",
            corp_code="00126380",
            received_on=date(2018, 2, 1),
        )


def test_parse_dividend_decision_skips_directory_members() -> None:
    import io as _io
    import zipfile as _zf

    from src.integrations.dart.dividend_decision import parse_dividend_decision

    raw = _io.BytesIO()
    with _zf.ZipFile(raw, "w", compression=_zf.ZIP_DEFLATED) as zf:
        zf.writestr("subdir/", "")
        zf.writestr("doc.xml", layout_2017_html())
    decision = parse_dividend_decision(
        archive_bytes=raw.getvalue(),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )

    assert decision.dps_common_krw == 300


def test_parse_dividend_decision_reads_single_cell_rows() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    html = (
        '<?xml version="1.0" encoding="utf-8"?><document><table>'
        "<tr><td>보통주 주당배당금, 300원</td></tr>"
        "<tr><td>배당기준일 2017-12-31</td></tr>"
        "<tr><td>배당금지급 예정일자 2018-04-20</td></tr>"
        "</table></document>"
    )
    decision = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )

    assert decision.record_date == date(2017, 12, 31)
    assert decision.pay_date == date(2018, 4, 20)
    assert decision.dps_common_krw == 300


def test_parse_dividend_decision_reads_record_date_outside_tables() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    html = (
        '<?xml version="1.0" encoding="utf-8"?><document>'
        "<p>배당기준일 2017-12-31</p><table>"
        "<tr><td>보통주 주당배당금</td><td>,</td><td>300</td></tr>"
        "<tr><td>배당금지급 예정일자</td><td>2018-04-20</td></tr>"
        "</table></document>"
    )
    decision = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )

    assert decision.record_date == date(2017, 12, 31)
    assert decision.dps_common_krw == 300


def test_parse_dividend_decision_absent_pay_row_means_undecided() -> None:
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    html = (
        '<?xml version="1.0" encoding="utf-8"?><document><table>'
        "<tr><td>보통주 주당배당금</td><td>300</td></tr>"
        "<tr><td>배당기준일</td><td>2017-12-31</td></tr>"
        "</table></document>"
    )
    decision = parse_dividend_decision(
        archive_bytes=make_archive({"20180201001234.xml": html}),
        rcept_no="20180201001234",
        corp_code="00126380",
        received_on=date(2018, 2, 1),
    )

    assert decision.pay_date is None


def test_parse_dividend_decision_rejects_malformed_dates() -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.integrations.dart.dividend_decision import parse_dividend_decision

    bad_record = (
        '<?xml version="1.0" encoding="utf-8"?><document><table>'
        "<tr><td>보통주 주당배당금</td><td>300</td></tr>"
        "<tr><td>배당기준일</td><td>2024-13-99</td></tr>"
        "<tr><td>배당금지급 예정일자</td><td>2025-04-18</td></tr>"
        "</table></document>"
    )
    with pytest.raises(PITDataError, match="record date"):
        parse_dividend_decision(
            archive_bytes=make_archive({"20250203001234.xml": bad_record}),
            rcept_no="20250203001234",
            corp_code="00126380",
            received_on=date(2025, 2, 3),
        )
    bad_pay = (
        '<?xml version="1.0" encoding="utf-8"?><document><table>'
        "<tr><td>보통주 주당배당금</td><td>300</td></tr>"
        "<tr><td>배당기준일</td><td>2024-12-31</td></tr>"
        "<tr><td>배당금지급 예정일자</td><td>추후 공지</td></tr>"
        "</table></document>"
    )
    with pytest.raises(PITDataError, match="pay date"):
        parse_dividend_decision(
            archive_bytes=make_archive({"20250203001234.xml": bad_pay}),
            rcept_no="20250203001234",
            corp_code="00126380",
            received_on=date(2025, 2, 3),
        )
