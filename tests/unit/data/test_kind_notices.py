"""KIND notice form parsing, iteration and coverage tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.kind_notices import (
    iter_kind_notices,
    kind_source_digest,
    parse_kind_administrative_form,
    parse_kind_delisting_form,
    read_kind_document,
    require_kind_coverage,
)
from tests.fixtures.kind_html import kind_fixture, kind_row_html, kind_search_html

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2024, 6, 1, tzinfo=UTC)

DELISTING_HTML = """<html><body><table>
<tr><td>단축코드</td><td>A005980</td></tr>
<tr><td>정리매매 허용기간 시작일</td><td>2018-09-19</td></tr>
<tr><td>종료일</td><td>2018-10-02</td></tr>
<tr><td>상장폐지일</td><td>2018-10-04</td></tr>
</table></body></html>"""

NO_LIQUIDATION_HTML = """<html><body><table>
<tr><td>단축코드</td><td>A005980</td></tr>
<tr><td>정리매매 허용기간 시작일</td><td>-</td></tr>
<tr><td>종료일</td><td>-</td></tr>
<tr><td>상장폐지일</td><td>2018-10-04</td></tr>
</table></body></html>"""

ADMIN_COMMON_HTML = """<html><body><table>
<tr><td>종목명</td><td>성지건설보통주</td></tr>
<tr><td>지정일</td><td>2018년 08월 16일</td></tr>
</table></body></html>"""

ADMIN_PREFERRED_HTML = """<html><body><table>
<tr><td>종목명</td><td>대상홀딩스1우선주</td></tr>
<tr><td>지정일</td><td>2018년 08월 16일</td></tr>
</table></body></html>"""


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _catalog(runtime):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")






def _publish_search(runtime, *, keyword: str, start: str, end: str, reported_total: int, pages: list[str]) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    body = json.dumps(
        {"keyword": keyword, "start": start, "end": end,
         "reported_total": reported_total, "pages": pages},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(body, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    _catalog(runtime).publish(
        [
            ReceiptIndexEntry(
                source="kind_notice_search", natural_key=f"{keyword}:{start}..{end}",
                as_of=date.fromisoformat(end), fiscal_period=None,
                status=EvidenceStatus.EMPTY if reported_total == 0 else EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="kind_notice_search", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def _publish_document(runtime, *, acptno: str, html: str, disclosed_at: str) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    body = json.dumps(
        {"acptno": acptno, "doc_no": "1234567", "template": "68051.htm",
         "html": html, "disclosed_at": disclosed_at},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(body, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    _catalog(runtime).publish(
        [
            ReceiptIndexEntry(
                source="kind_notice_documents", natural_key=acptno,
                as_of=date.fromisoformat(disclosed_at[:10]), fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="kind_notice_documents", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def test_parse_delisting_form_yields_window_and_date() -> None:
    form = parse_kind_delisting_form(DELISTING_HTML)

    assert form.tickers == ("005980",)
    assert form.liquidation_start == date(2018, 9, 19)
    assert form.liquidation_end == date(2018, 10, 2)
    assert form.delisting_date == date(2018, 10, 4)


def test_live_delisting_form_reads_liquidation_window_not_the_notice_period() -> None:
    # 실제 양식은 '상장폐지 예고기간'과 '정리매매 허용기간'에 같은 시작일/종료일 라벨을 쓴다.
    seongji = parse_kind_delisting_form(kind_fixture("body_delisting_seongji.html"))
    assert seongji.tickers == ("005980",)
    assert seongji.liquidation_start == date(2018, 9, 19)
    assert seongji.liquidation_end == date(2018, 10, 2)
    assert seongji.delisting_date == date(2018, 10, 4)

    ihq = parse_kind_delisting_form(kind_fixture("body_delisting_ihq.html"))
    assert ihq.tickers == ("003560",)
    assert ihq.liquidation_start == date(2026, 1, 6)
    assert ihq.liquidation_end == date(2026, 1, 14)
    assert ihq.delisting_date == date(2026, 1, 15)


def test_live_administrative_form_reads_issue_and_designation_date() -> None:
    form = parse_kind_administrative_form(kind_fixture("body_admin_seongji.html"))
    assert form.issue_name == "성지건설보통주"
    assert form.is_common is True
    assert form.effective_on == date(2018, 8, 16)


def test_live_kosdaq_administrative_form_uses_the_target_issue_label() -> None:
    # 코스닥 양식은 `1.대상종목 하이에이아이1호스팩 주권 보통주`로 라벨과 띄어쓰기가 다르다.
    form = parse_kind_administrative_form(kind_fixture("body_admin_kosdaq_spac.html"))
    assert form.issue_name == "하이에이아이1호스팩 주권 보통주"
    assert form.is_common is True
    assert form.effective_on == date(2018, 8, 27)


def test_administrative_form_keeps_spaced_names_whole() -> None:
    common = parse_kind_administrative_form("<p>1.종목명 KH 필룩스보통주 2.관리종목 지정일 2020-01-02</p>")
    preferred = parse_kind_administrative_form("<p>1.종목명 대상홀딩스1우선주 2.관리종목 지정일 2020-01-02</p>")
    assert common.issue_name == "KH 필룩스보통주"
    assert common.is_common is True
    assert preferred.is_common is False


def test_live_fund_delisting_form_resolves_the_short_code_from_its_isin() -> None:
    # ETF 양식은 단축코드 칸 없이 `종목코드 : KR7200050003`만 싣는다.
    form = parse_kind_delisting_form(kind_fixture("body_delisting_etf.html"))
    assert form.tickers == ("200050",)
    assert form.liquidation_start is None
    assert form.delisting_date == date(2018, 9, 27)


def test_delisting_form_accepts_alphanumeric_short_codes_only_inside_the_code_field() -> None:
    html = (
        "<html><body>2.상장폐지 주권 종류 및 주식수 주권종류 주식수 단축코드 보통주 1,000 A0001A0 "
        "3.상장폐지 사유 ADR ABCDEF1 해산 사유 발생 6.상장폐지일 2026-02-02</body></html>"
    )
    form = parse_kind_delisting_form(html)
    assert form.tickers == ("0001A0",)
    assert form.liquidation_start is None
    assert form.liquidation_end is None


def test_parse_delisting_form_without_liquidation_keeps_dates_null() -> None:
    form = parse_kind_delisting_form(NO_LIQUIDATION_HTML)

    assert form.liquidation_start is None
    assert form.liquidation_end is None
    assert form.delisting_date == date(2018, 10, 4)


def test_parse_delisting_form_without_short_code_fails_closed() -> None:
    with pytest.raises(PITDataError):
        parse_kind_delisting_form("<html><body>상장폐지일 2018-10-04</body></html>")


def test_parse_administrative_form_distinguishes_common_stock() -> None:
    common = parse_kind_administrative_form(ADMIN_COMMON_HTML)
    preferred = parse_kind_administrative_form(ADMIN_PREFERRED_HTML)

    assert common.is_common is True
    assert preferred.is_common is False
    assert common.effective_on == date(2018, 8, 16)
    assert preferred.effective_on == date(2018, 8, 16)


def test_iter_notices_collapses_across_keyword_windows(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240101000001", "2024-01-02 15:30", "상장폐지", "005930", "유가증권시장본부")
    _publish_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-01-31",
                    reported_total=1, pages=[kind_search_html([row], 1)])
    _publish_search(runtime, keyword="정리매매", start="2024-01-01", end="2024-01-31",
                    reported_total=1, pages=[kind_search_html([row], 1)])

    notices = list(iter_kind_notices(_catalog(runtime)))

    assert [notice.acptno for notice in notices] == ["20240101000001"]


def test_iter_notices_with_contradictory_duplicate_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = kind_row_html("20240101000001", "2024-01-02 15:30", "상장폐지", "005930", "유가증권시장본부")
    second = kind_row_html("20240101000001", "2024-01-02 15:30", "관리종목 지정", "005930", "유가증권시장본부")
    _publish_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-01-31",
                    reported_total=1, pages=[kind_search_html([first], 1)])
    _publish_search(runtime, keyword="관리종목", start="2024-01-01", end="2024-01-31",
                    reported_total=1, pages=[kind_search_html([second], 1)])

    with pytest.raises(PITDataError):
        list(iter_kind_notices(_catalog(runtime)))


def test_require_coverage_gap_fails_closed_naming_first_gap(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240101000001", "2016-02-02 15:30", "상장폐지", "005930", "유가증권시장본부")
    _publish_search(runtime, keyword="상장폐지", start="2016-01-01", end="2016-03-31",
                    reported_total=1, pages=[kind_search_html([row], 1)])

    with pytest.raises(PITDataError, match=r"상장폐지:2016-04-01\.\.2016-06-30"):
        require_kind_coverage(
            _catalog(runtime), keywords=("상장폐지",),
            through=date(2016, 7, 15), start=date(2016, 1, 1),
        )


def test_read_document_returns_none_without_receipt(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    assert read_kind_document(_catalog(runtime), "20240101000001") is None


def test_read_document_returns_body_and_digest_covers_receipts(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    row = kind_row_html("20240101000001", "2024-01-02 15:30", "상장폐지", "005930", "유가증권시장본부")
    _publish_search(runtime, keyword="상장폐지", start="2024-01-01", end="2024-01-31",
                    reported_total=1, pages=[kind_search_html([row], 1)])
    before = kind_source_digest(_catalog(runtime))
    _publish_document(runtime, acptno="20240101000001", html=DELISTING_HTML,
                      disclosed_at="2024-01-02T15:30:00+09:00")

    assert read_kind_document(_catalog(runtime), "20240101000001") == DELISTING_HTML
    assert kind_source_digest(_catalog(runtime)) != before


def test_parse_delisting_form_accepts_dotted_dates() -> None:
    html = """<html><body><table>
<tr><td>단축코드</td><td>A005980</td></tr>
<tr><td>정리매매 허용기간 시작일</td><td>2018.09.19</td></tr>
<tr><td>종료일</td><td>2018.10.02</td></tr>
<tr><td>상장폐지일</td><td>2018.10.04</td></tr>
</table></body></html>"""

    form = parse_kind_delisting_form(html)

    assert form.liquidation_start == date(2018, 9, 19)
    assert form.liquidation_end == date(2018, 10, 2)


def test_parse_delisting_form_with_one_sided_liquidation_fails_closed() -> None:
    html = """<html><body><table>
<tr><td>단축코드</td><td>A005980</td></tr>
<tr><td>정리매매 허용기간 시작일</td><td>2018-09-19</td></tr>
<tr><td>종료일</td><td>-</td></tr>
<tr><td>상장폐지일</td><td>2018-10-04</td></tr>
</table></body></html>"""

    with pytest.raises(PITDataError):
        parse_kind_delisting_form(html)


def test_parse_delisting_form_without_delisting_date_fails_closed() -> None:
    with pytest.raises(PITDataError):
        parse_kind_delisting_form("<html><body>단축코드 A005980</body></html>")


def test_parse_delisting_form_with_invalid_dates_fails_closed() -> None:
    bad_month = DELISTING_HTML.replace("2018-10-04", "2018-13-40")
    with pytest.raises(PITDataError):
        parse_kind_delisting_form(bad_month)
    bad_start = DELISTING_HTML.replace("2018-09-19", "2018-13-01")
    with pytest.raises(PITDataError):
        parse_kind_delisting_form(bad_start)
    bad_shape = DELISTING_HTML.replace("2018-10-04", "October 4th")
    with pytest.raises(PITDataError):
        parse_kind_delisting_form(bad_shape)


def test_parse_administrative_form_without_issue_name_fails_closed() -> None:
    with pytest.raises(PITDataError):
        parse_kind_administrative_form("<html><body>지정일 2018년 08월 16일</body></html>")


def test_parse_administrative_form_without_date_keeps_effective_null() -> None:
    form = parse_kind_administrative_form("<html><body>종목명 성지건설보통주</body></html>")

    assert form.is_common is True
    assert form.effective_on is None


def test_parse_kind_date_rejects_invalid_calendar_dates() -> None:
    from src.data.kind_notices import _parse_kind_date

    with pytest.raises(PITDataError):
        _parse_kind_date("2018-13-01")
    with pytest.raises(PITDataError):
        _parse_kind_date("next Tuesday")


def _publish_raw_search(runtime, *, key: str, raw: bytes) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(raw, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    _catalog(runtime).publish(
        [
            ReceiptIndexEntry(
                source="kind_notice_search", natural_key=key,
                as_of=date(2024, 1, 31), fiscal_period=None, status=EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="kind_notice_search", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def test_iter_notices_with_missing_payload_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = _catalog(runtime)
    _publish_raw_search(runtime, key="상장폐지:2024-01-01..2024-01-31", raw=b'{"pages": []}')
    entry = catalog.latest(
        source="kind_notice_search", natural_keys={"상장폐지:2024-01-01..2024-01-31"}
    )["상장폐지:2024-01-01..2024-01-31"]
    Path(entry.payload_path).unlink()

    with pytest.raises(PITDataError):
        list(iter_kind_notices(catalog))


def test_iter_notices_with_tampered_payload_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = _catalog(runtime)
    _publish_raw_search(runtime, key="상장폐지:2024-01-01..2024-01-31", raw=b'{"pages": []}')
    entry = catalog.latest(
        source="kind_notice_search", natural_keys={"상장폐지:2024-01-01..2024-01-31"}
    )["상장폐지:2024-01-01..2024-01-31"]
    Path(entry.payload_path).write_bytes(b"tampered")

    with pytest.raises(PITDataError):
        list(iter_kind_notices(catalog))


def test_iter_notices_with_unreadable_payload_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = _catalog(runtime)
    _publish_raw_search(runtime, key="상장폐지:2024-01-01..2024-01-31", raw=b"not json")

    with pytest.raises(PITDataError):
        list(iter_kind_notices(catalog))
    _publish_raw_search(runtime, key="관리종목:2024-01-01..2024-01-31", raw=b"[1, 2]")

    with pytest.raises(PITDataError):
        list(iter_kind_notices(catalog))


def test_iter_notices_with_pageless_payload_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_raw_search(
        runtime, key="상장폐지:2024-01-01..2024-01-31",
        raw=json.dumps({"reported_total": 0}).encode("utf-8"),
    )

    with pytest.raises(PITDataError, match="no pages"):
        list(iter_kind_notices(_catalog(runtime)))


def test_iter_notices_with_unparseable_page_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_raw_search(
        runtime, key="상장폐지:2024-01-01..2024-01-31",
        raw=json.dumps(
            {"reported_total": 1, "pages": ["<html><body>페이지 오류</body></html>"]}
        ).encode("utf-8"),
    )

    with pytest.raises(PITDataError, match="does not parse"):
        list(iter_kind_notices(_catalog(runtime)))


def test_require_coverage_with_malformed_total_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_raw_search(
        runtime, key="상장폐지:2024-01-02..2024-03-31",
        raw=json.dumps({"reported_total": "2", "pages": []}).encode("utf-8"),
    )

    with pytest.raises(PITDataError, match=r"상장폐지:2024-01-02\.\.2024-03-31"):
        require_kind_coverage(
            _catalog(runtime), keywords=("상장폐지",),
            through=date(2024, 4, 2), start=date(2024, 1, 2),
        )


def _publish_raw_document(runtime, *, acptno: str, raw: bytes) -> None:  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(raw, kind=EvidenceKind.DISCLOSURES, retrieved_at=RETRIEVED_AT, source_label="test")
    _catalog(runtime).publish(
        [
            ReceiptIndexEntry(
                source="kind_notice_documents", natural_key=acptno,
                as_of=date(2024, 1, 2), fiscal_period=None, status=EvidenceStatus.SUCCESS,
                content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES,
                source="kind_notice_documents", usable=True, unusable_reason=None,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        ],
    )


def test_read_document_with_corrupt_payloads_fails_closed(tmp_path: Path) -> None:
    from src.data.receipt_catalog import ReceiptCatalog as _Catalog

    runtime = _runtime(tmp_path)
    catalog = _catalog(runtime)
    _publish_document(runtime, acptno="20240101000001", html=DELISTING_HTML,
                      disclosed_at="2024-01-02T15:30:00+09:00")
    entry = catalog.latest(
        source="kind_notice_documents", natural_keys={"20240101000001"}
    )["20240101000001"]
    Path(entry.payload_path).write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        read_kind_document(_Catalog(runtime.workspace.bronze_root / "catalog"), "20240101000001")

    runtime2 = _runtime(tmp_path / "second")
    catalog2 = _catalog(runtime2)
    _publish_raw_document(runtime2, acptno="20240101000002", raw=b"not json")
    with pytest.raises(PITDataError, match="unreadable"):
        read_kind_document(catalog2, "20240101000002")

    _publish_raw_document(runtime2, acptno="20240101000003", raw=b"[1, 2]")
    with pytest.raises(PITDataError, match="unreadable"):
        read_kind_document(catalog2, "20240101000003")

    _publish_raw_document(
        runtime2, acptno="20240101000004",
        raw=json.dumps({"acptno": "20240101000004"}).encode("utf-8"),
    )
    with pytest.raises(PITDataError, match="unreadable"):
        read_kind_document(catalog2, "20240101000004")


def test_read_document_with_missing_payload_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = _catalog(runtime)
    _publish_document(runtime, acptno="20240101000001", html=DELISTING_HTML,
                      disclosed_at="2024-01-02T15:30:00+09:00")
    entry = catalog.latest(
        source="kind_notice_documents", natural_keys={"20240101000001"}
    )["20240101000001"]
    Path(entry.payload_path).unlink()

    with pytest.raises(PITDataError, match="unreadable"):
        read_kind_document(catalog, "20240101000001")


def test_require_coverage_with_unparseable_window_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _publish_raw_search(
        runtime, key="상장폐지:2024-01-02..2024-03-31",
        raw=json.dumps(
            {"reported_total": 1, "pages": ["<html><body>페이지 오류</body></html>"]}
        ).encode("utf-8"),
    )

    with pytest.raises(PITDataError, match=r"상장폐지:2024-01-02\.\.2024-03-31"):
        require_kind_coverage(
            _catalog(runtime), keywords=("상장폐지",),
            through=date(2024, 4, 2), start=date(2024, 1, 2),
        )
