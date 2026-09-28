"""KIND notice search parsing and viewer-chain invariants."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from tests.fixtures.kind_html import kind_fixture, kind_row_html, kind_search_html

_KST = ZoneInfo("Asia/Seoul")

SEARCH_HTML_TWO_ROWS = kind_search_html(
    [
        kind_row_html("20240102000001", "2024-01-02 15:30", "상장폐지", "00593", "유가증권시장본부"),
        kind_row_html("20240103000002", "2024-01-03 09:05", "관리종목지정", "00354", "유가증권시장본부"),
    ],
    2,
)

EMPTY_HTML = kind_fixture("search_empty.html")

ERROR_HTML = kind_fixture("search_error.html")

MARKET_WIDE_HTML = kind_search_html(
    [kind_row_html("20240104000003", "2024-01-04 10:00", "기타시장안내", None, "유가증권시장본부")], 1
)

VIEWER_HTML = kind_fixture("viewer_seongji.html")

CONTENTS_HTML = kind_fixture("contents_seongji.html")

BODY_HTML = "<html><head><meta charset=\"euc-kr\"></head><body>단축코드 A005980</body></html>"


def _fake_response(*, text: str = "", content: bytes | None = None, encoding: str | None = "utf-8"):
    raw = content if content is not None else text.encode("utf-8")
    return SimpleNamespace(status_code=200, text=text or raw.decode("utf-8", errors="replace"),
                           content=raw, encoding=encoding, apparent_encoding="utf-8", headers={})


def _client(transport):  # type: ignore[no-untyped-def]
    from src.integrations.krx.kind import KindClient

    return KindClient(transport=transport, min_interval_seconds=0.0)


def test_search_page_parses_rows_and_total() -> None:
    from src.integrations.krx.kind import parse_kind_search_page

    page = parse_kind_search_page(SEARCH_HTML_TWO_ROWS)

    assert page.reported_total == 2
    assert len(page.rows) == 2
    first, second = page.rows
    assert first.acptno == "20240102000001"
    assert first.disclosed_at == datetime(2024, 1, 2, 15, 30, tzinfo=_KST)
    assert first.company_code == "00593"
    assert first.title == "상장폐지"
    assert first.submitter == "유가증권시장본부"
    assert second.acptno == "20240103000002"
    assert second.title == "관리종목지정"
    assert second.company_code == "00354"


def test_live_search_page_parses_every_row_with_its_exchange_submitter() -> None:
    from src.integrations.krx.kind import parse_kind_search_page

    page = parse_kind_search_page(kind_fixture("search_admin_2018_08.html"))

    assert page.reported_total == 44
    assert len(page.rows) == 44
    assert all(row.submitter in {"유가증권시장본부", "코스닥시장본부"} for row in page.rows)
    seongji = next(row for row in page.rows if row.acptno == "20180814003152")
    assert seongji.disclosed_at == datetime(2018, 8, 14, 19, 8, tzinfo=_KST)
    assert seongji.company_code == "00598"
    assert seongji.company_name == "성지건설"
    assert seongji.title == "관리종목지정"
    assert seongji.submitter == "유가증권시장본부"


def test_row_without_submitter_cell_fails_closed() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.krx.kind import parse_kind_search_page

    row = (
        "<tr><td class=\"txc\">2024-01-02 15:30</td>"
        "<td><a href=\"#viewer\" onclick=\"openDisclsViewer('20240102000001','')\" title='상장폐지'>상장폐지</a></td></tr>"
    )
    with pytest.raises(PITDataError, match="submitter"):
        parse_kind_search_page(kind_search_html([row], 1))


def test_empty_result_page_has_zero_total() -> None:
    from src.integrations.krx.kind import parse_kind_search_page

    page = parse_kind_search_page(EMPTY_HTML)

    assert page.rows == ()
    assert page.reported_total == 0


def test_error_page_fails_closed() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.errors import ProviderRetryableError
    from src.integrations.krx.kind import parse_kind_search_page

    with pytest.raises(PITDataError):
        parse_kind_search_page(ERROR_HTML)

    class _ErrorTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            response = _fake_response(text=ERROR_HTML)
            if classify is not None:
                classify(response)
            return response

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("GET must not be used")

    with pytest.raises(ProviderRetryableError):
        _client(_ErrorTransport()).search_page("상장폐지", date(2024, 1, 1), date(2024, 1, 31), page_index=1)


def test_market_wide_notice_has_no_company_code() -> None:
    from src.integrations.krx.kind import parse_kind_search_page

    (row,) = parse_kind_search_page(MARKET_WIDE_HTML).rows

    assert row.company_code is None


def test_search_form_carries_the_full_field_set() -> None:
    seen: list[tuple] = []

    class _FakeTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            seen.append((endpoint, dict(form), dict(headers or {})))
            return _fake_response(text=EMPTY_HTML)

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("GET must not be used")

    _client(_FakeTransport()).search_page("상장폐지", date(2024, 1, 1), date(2024, 1, 31), page_index=2)

    (endpoint, form, headers) = seen[0]
    assert endpoint == "disclosure/details.do"
    assert form["method"] == "searchDetailsSub"
    assert form["pageIndex"] == "2"
    assert form["currentPageSize"] == "100"
    assert form["reportNm"] == "상장폐지"
    assert form["reportNmTemp"] == "상장폐지"
    assert form["fromDate"] == "2024-01-01"
    assert form["toDate"] == "2024-01-31"
    assert form["bfrDsclsType"] == "on"
    assert form["orderMode"] == "1"
    assert form["orderStat"] == "D"
    # KIND는 검색 폼 필드가 하나라도 빠지면 오류 페이지를 준다(실측).
    for field in (
        "disclosureType", "disTypevalue", "reportCd", "searchCodeType", "repIsuSrtCd", "allRepIsuSrtCd",
        "oldSearchCorpName", "searchCorpName", "business", "marketType", "settlementMonth", "securities",
        "submitOblgNm", "enterprise", "lastReport",
    ):
        assert form[field] == ""
    assert "searchCorpCode" not in form
    assert "searchType" not in form
    assert "Mozilla" in headers["User-Agent"]
    assert headers["X-Requested-With"] == "XMLHttpRequest"


def test_document_resolves_through_viewer_chain() -> None:
    calls: list[tuple] = []

    class _FakeTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("search must not be used")

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            calls.append((endpoint, dict(params or {})))
            if len(calls) == 1:
                return _fake_response(text=VIEWER_HTML)
            if len(calls) == 2:
                return _fake_response(text=CONTENTS_HTML)
            return _fake_response(text=BODY_HTML, content=BODY_HTML.encode("utf-8"))

    document = _client(_FakeTransport()).fetch_document("20180913000523")

    # 실제 뷰어는 `<option value='20180913001314|Y'>`처럼 작은따옴표를 쓴다.
    assert document.doc_no == "20180913001314"
    assert calls[1] == ("common/disclsviewer.do", {"method": "searchContents", "docNo": "20180913001314"})
    assert document.template == "68051.htm"
    assert document.html.strip()
    assert len(calls) == 3


def test_viewer_without_document_fails_closed() -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    class _FakeTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("search must not be used")

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            return _fake_response(text="<html><body><select></select></body></html>")

    with pytest.raises(ProviderTerminalError):
        _client(_FakeTransport()).fetch_document("20240102000001")


def test_search_page_rejects_bad_arguments() -> None:
    import pytest

    class _NoTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("no request must be sent")

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("no request must be sent")

    client = _client(_NoTransport())
    with pytest.raises(ValueError, match="keyword"):
        client.search_page("  ", date(2024, 1, 1), date(2024, 1, 31), page_index=1)
    with pytest.raises(ValueError, match="start must not be after end"):
        client.search_page("상장폐지", date(2024, 2, 1), date(2024, 1, 1), page_index=1)
    with pytest.raises(ValueError, match="page_index"):
        client.search_page("상장폐지", date(2024, 1, 1), date(2024, 1, 31), page_index=0)


def test_search_row_variants_fail_closed_or_fall_back() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.krx.kind import parse_kind_search_page

    def _page(row: str) -> str:
        return f"<html><body><div>전체 1 건</div><table><tbody><tr>{row}</tr></tbody></table></body></html>"

    viewer_no_title = (
        "<td>2024-01-02 15:30</td>"
        "<td><a href=\"#\">보기</a> <a href=\"#\" onclick=\"openDisclsViewer('20240102000001')\">상장폐지공고</a></td>"
        "<td>-</td><td>유가증권시장본부</td>"
    )
    (row,) = parse_kind_search_page(_page(viewer_no_title)).rows
    assert row.title == "상장폐지공고"

    missing_timestamp = (
        "<td><a href=\"#\" onclick=\"openDisclsViewer('20240102000001')\" title=\"상장폐지\">상장폐지</a></td>"
        "<td>-</td><td>유가증권시장본부</td>"
    )
    with pytest.raises(PITDataError, match="timestamp or acceptance"):
        parse_kind_search_page(_page(missing_timestamp))

    missing_acptno = "<td>2024-01-02 15:30</td><td>상장폐지</td><td>-</td><td>유가증권시장본부</td>"
    with pytest.raises(PITDataError, match="timestamp or acceptance"):
        parse_kind_search_page(_page(missing_acptno))

    bad_acptno = (
        "<td>2024-01-02 15:30</td>"
        "<td><a href=\"#\" onclick=\"openDisclsViewer('12345')\" title=\"상장폐지\">상장폐지</a></td>"
        "<td>-</td><td>유가증권시장본부</td>"
    )
    with pytest.raises(PITDataError, match="acceptance number"):
        parse_kind_search_page(_page(bad_acptno))

    bad_timestamp = (
        "<td>2024-13-45 99:99</td>"
        "<td><a href=\"#\" onclick=\"openDisclsViewer('20240102000001')\" title=\"상장폐지\">상장폐지</a></td>"
        "<td>-</td><td>유가증권시장본부</td>"
    )
    with pytest.raises(PITDataError, match="invalid timestamp"):
        parse_kind_search_page(_page(bad_timestamp))

    with pytest.raises(PITDataError, match="no result total"):
        parse_kind_search_page("<html><body><table><tbody></tbody></table></body></html>")


def test_fetch_document_chain_fails_closed() -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    def _transport_for(texts: list[str]):  # type: ignore[no-untyped-def]
        calls = {"n": 0}

        class _FakeTransport:
            def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
                raise AssertionError("search must not be used")

            def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
                calls["n"] += 1
                return _fake_response(text=texts[min(calls["n"] - 1, len(texts) - 1)])

        return _FakeTransport()

    client = _client(_transport_for([VIEWER_HTML]))
    with pytest.raises(ValueError, match="acptno"):
        client.fetch_document("123")

    with pytest.raises(ProviderTerminalError, match="no document"):
        _client(_transport_for(['<html><body><select><option value="|x">y</option></select></body></html>'])).fetch_document("20240102000001")

    with pytest.raises(ProviderTerminalError, match="no body path"):
        _client(_transport_for([VIEWER_HTML, "<html><body>no link</body></html>"])).fetch_document("20240102000001")

    with pytest.raises(ProviderTerminalError, match="empty"):
        _client(_transport_for([VIEWER_HTML, CONTENTS_HTML, "   "])).fetch_document("20240102000001")


def test_body_charset_detection_paths() -> None:
    body = "단축코드 A005980".encode("euc-kr")

    class _FakeTransport:
        def __init__(self, response):  # type: ignore[no-untyped-def]
            self._response = response

        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("search must not be used")

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            if (params or {}).get("method") == "search":
                return _fake_response(text=VIEWER_HTML)
            if (params or {}).get("method") == "searchContents":
                return _fake_response(text=CONTENTS_HTML)
            return self._response

    header_response = SimpleNamespace(
        status_code=200, content=body, encoding=None, apparent_encoding="utf-8",
        headers={"Content-Type": "text/html; charset=euc-kr"}, text="",
    )
    assert _client(_FakeTransport(header_response)).fetch_document("20240102000001").html == "단축코드 A005980"

    meta_response = SimpleNamespace(
        status_code=200,
        content=b'<meta charset="euc-kr">' + body,
        encoding=None, apparent_encoding="utf-8", headers={}, text="",
    )
    assert "단축코드" in _client(_FakeTransport(meta_response)).fetch_document("20240102000001").html

    fallback_response = SimpleNamespace(
        status_code=200, content=b"\xff\xfe invalid", encoding="bogus-charset",
        apparent_encoding="utf-8", headers={}, text="",
    )
    assert _client(_FakeTransport(fallback_response)).fetch_document("20240102000001").html.strip()

    apparent_response = SimpleNamespace(
        status_code=200, content="단축코드 A005980".encode("euc-kr"), encoding=None,
        apparent_encoding="euc-kr", headers={}, text="",
    )
    assert _client(_FakeTransport(apparent_response)).fetch_document("20240102000001").html == "단축코드 A005980"


def test_health_check_and_scoped_builder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.krx.kind import build_scoped_kind_client
    from src.integrations.quota import ProviderQuotaStateStore

    class _FakeTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            return _fake_response(text=SEARCH_HTML_TWO_ROWS)

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("GET must not be used")

    _client(_FakeTransport()).health_check()

    client = build_scoped_kind_client(
        policy=_provider().kind, quota_store=ProviderQuotaStateStore(tmp_path),
    )
    assert client.BASE_URL == "https://kind.krx.co.kr"


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def test_body_ignores_the_requests_latin1_default_and_uses_the_document_meta() -> None:
    # 실측: KIND 본문 헤더는 문자셋 없는 text/html이라 requests가 encoding을 ISO-8859-1로 채운다.
    live_body = kind_fixture("body_delisting_seongji.html").encode("utf-8")

    class _FakeTransport:
        def post_form(self, endpoint, form, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            raise AssertionError("search must not be used")

        def get(self, endpoint, params=None, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
            if (params or {}).get("method") == "search":
                return _fake_response(text=VIEWER_HTML)
            if (params or {}).get("method") == "searchContents":
                return _fake_response(text=CONTENTS_HTML)
            return SimpleNamespace(
                status_code=200, content=live_body, encoding="ISO-8859-1", apparent_encoding="utf-8",
                headers={"Content-Type": "text/html"}, text="",
            )

    html = _client(_FakeTransport()).fetch_document("20180913000523").html

    assert "단축코드" in html
    assert "A005980" in html
