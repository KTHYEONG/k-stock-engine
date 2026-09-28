"""KRX KIND transport-only client for exchange notices DART does not carry."""

from __future__ import annotations

import html as _html
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final
from zoneinfo import ZoneInfo

import requests

from src.config.providers import KindPolicy
from src.core.pit import PITDataError
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HttpTransport, RetryPolicy

__all__ = [
    "KindClient",
    "KindNoticeDocument",
    "KindNoticeRow",
    "KindSearchPage",
    "build_scoped_kind_client",
    "parse_kind_search_page",
]

_PROVIDER = "KIND"
_BASE_URL = "https://kind.krx.co.kr"
_KST = ZoneInfo("Asia/Seoul")
_TIMEOUT_SECONDS: Final = 30.0
_MAX_HTTP_ATTEMPTS: Final = 3
_PAGE_SIZE: Final = 100

_HEADERS: Final = {
    "User-Agent": "Mozilla/5.0 (compatible; KStockEngine/1.0; +https://github.com/KTHYEONG/k-stock-engine)",
    "Referer": "https://kind.krx.co.kr/disclosure/details.do?method=searchDetailsMain",
    "X-Requested-With": "XMLHttpRequest",
}

_EMPTY_FORM_FIELDS: Final = (
    "disclosureType",
    "disTypevalue",
    "reportCd",
    "searchCodeType",
    "repIsuSrtCd",
    "allRepIsuSrtCd",
    "oldSearchCorpName",
    "searchCorpName",
    "business",
    "marketType",
    "settlementMonth",
    "securities",
    "submitOblgNm",
    "enterprise",
    "lastReport",
)

_ERROR_MARKER = "페이지 오류"
_EMPTY_MARKER = "조회된 결과값이 없습니다"
# 하단 합계는 `전체 <em>44</em>건`처럼 숫자를 태그로 감싼다.
_TOTAL_PATTERN = re.compile(r"전체\s*(?:<[^>]+>\s*)*([\d,]+)\s*(?:<[^>]+>\s*)*건")
_ROW_BLOCK_PATTERN = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})")
_VIEWER_PATTERN = re.compile(r"openDisclsViewer\('(\d+)'")
_COMPANY_PATTERN = re.compile(r"companysummary_open\('([^']+)'")
_TITLE_ATTR_PATTERN = re.compile(r"title=(['\"])(.*?)\1", re.DOTALL)
_ANCHOR_PATTERN = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(r"<td\b[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_OPTION_PATTERN = re.compile(r"<option\b[^>]*value=(['\"])([^'\"]+)\1", re.IGNORECASE)
_BODY_PATH_PATTERN = re.compile(r"(/external/[^\"'\s<>]+\.htm)")
_META_CHARSET_PATTERN = re.compile(r"charset\s*=\s*[\"']?([\w\-]+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class KindNoticeRow:
    """One row of a KIND notice search result.

    Attributes:
        acptno: KIND acceptance number (14 digits); the notice's natural key.
        disclosed_at: Publication instant (KST, minute precision) shown by KIND.
        company_code: Five-character KIND company code (the first five characters of the
            company's common-stock short code), or None for market-wide notices.
        company_name: Display name as listed.
        title: Notice title exactly as listed (``title`` attribute of the viewer link).
        submitter: Filer column (``유가증권시장본부``, ``코스닥시장본부`` or a company name).
    """

    acptno: str
    disclosed_at: datetime
    company_code: str | None
    company_name: str
    title: str
    submitter: str


@dataclass(frozen=True, slots=True)
class KindSearchPage:
    """One parsed search page plus the total KIND declared for the query."""

    rows: tuple[KindNoticeRow, ...]
    reported_total: int


@dataclass(frozen=True, slots=True)
class KindNoticeDocument:
    """The raw body of one KIND notice.

    Attributes:
        acptno: Notice acceptance number.
        doc_no: KIND document number resolved from the viewer.
        template: Body file name (for example ``68051.htm``), kept for audit.
        html: Decoded body HTML.
    """

    acptno: str
    doc_no: str
    template: str
    html: str


def _strip_tags(text: str) -> str:
    return _TAG_PATTERN.sub("", text).strip()


def _viewer_anchor(block: str) -> tuple[str, str] | None:
    for match in _ANCHOR_PATTERN.finditer(block):
        attrs, inner = match.group(1), match.group(2)
        acptno = _VIEWER_PATTERN.search(attrs)
        if acptno is None:
            continue
        title = _TITLE_ATTR_PATTERN.search(attrs)
        return acptno.group(1), _html.unescape(title.group(2)) if title is not None else _strip_tags(inner)
    return None


def _parse_row(block: str) -> KindNoticeRow | None:
    anchor = _viewer_anchor(block)
    timestamp = _TIMESTAMP_PATTERN.search(block)
    if anchor is None and timestamp is None:
        return None
    if anchor is None or timestamp is None:
        raise PITDataError("KIND search row lacks its timestamp or acceptance number")
    acptno, title = anchor
    if len(acptno) != 14 or not acptno.isdigit():
        raise PITDataError(f"KIND search row has an invalid acceptance number {acptno!r}")
    try:
        naive = datetime.strptime(timestamp.group(1), "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise PITDataError(f"KIND search row has an invalid timestamp {timestamp.group(1)!r}") from exc
    company = _COMPANY_PATTERN.search(block)
    company_code = company.group(1)[:5] if company is not None else None
    company_name = ""
    if company is not None:
        for match in _ANCHOR_PATTERN.finditer(block):
            if "companysummary_open" in match.group(1):
                company_name = _strip_tags(match.group(2))
                break
    # 행 구성은 번호·시간·회사명·공시제목·제출인·차트이므로 제출인은 공시제목 바로 다음 칸이다.
    cells = _CELL_PATTERN.findall(block)
    title_index = next((index for index, cell in enumerate(cells) if _VIEWER_PATTERN.search(cell)), None)
    if title_index is None or title_index + 1 >= len(cells):
        raise PITDataError(f"KIND search row {acptno} has no submitter cell after its title")
    submitter = _html.unescape(_strip_tags(cells[title_index + 1]))
    return KindNoticeRow(
        acptno=acptno,
        disclosed_at=naive.replace(tzinfo=_KST),
        company_code=company_code,
        company_name=company_name,
        title=title,
        submitter=submitter,
    )


def parse_kind_search_page(html: str) -> KindSearchPage:
    """Parse one KIND ``searchDetailsSub`` result page.

    Raises:
        PITDataError: the page is the KIND error page, the declared total is missing, or a
            result row lacks its timestamp or acceptance number.
    """
    if _ERROR_MARKER in html:
        raise PITDataError("KIND returned its error page instead of a search result")
    if _EMPTY_MARKER in html:
        return KindSearchPage(rows=(), reported_total=0)
    total = _TOTAL_PATTERN.search(html)
    if total is None:
        raise PITDataError("KIND search page declares no result total")
    reported_total = int(total.group(1).replace(",", ""))
    rows: list[KindNoticeRow] = []
    for block in _ROW_BLOCK_PATTERN.findall(html):
        row = _parse_row(block)
        if row is not None:
            rows.append(row)
    return KindSearchPage(rows=tuple(rows), reported_total=reported_total)


def _decode_body(response: requests.Response) -> str:
    # KIND 본문은 `Content-Type: text/html`(문자셋 없음)로 오고 실제 문자셋은 문서 내 meta에 있다.
    # 이때 requests가 채우는 response.encoding은 HTTP 기본값 ISO-8859-1이라 한글이 오류 없이 깨지므로 쓰지 않는다.
    content = bytes(response.content)
    content_type = str((getattr(response, "headers", None) or {}).get("Content-Type", ""))
    header = _META_CHARSET_PATTERN.search(content_type)
    meta = _META_CHARSET_PATTERN.search(content[:4096].decode("ascii", errors="ignore"))
    if header is not None:
        charset = header.group(1)
    elif meta is not None:
        charset = meta.group(1)
    else:
        charset = response.apparent_encoding or "utf-8"
    try:
        return content.decode(charset, errors="strict")
    except (LookupError, ValueError, UnicodeDecodeError):
        return content.decode("utf-8", errors="replace")


class KindClient:
    """Transport-only KIND client. Pacing, retries and the daily ledger belong to ``HttpTransport``.

    KIND is a public web service without an API contract, so every page is validated
    structurally and anything unexpected fails closed instead of being guessed.
    """

    BASE_URL = _BASE_URL

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        quota_store: ProviderQuotaStateStore | None = None,
        daily_limit: int | None = None,
        min_interval_seconds: float,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        if transport is not None:
            self._transport = transport
            return
        gate: LedgerQuotaGate | None = None
        if quota_store is not None:
            gate = LedgerQuotaGate(quota_store, provider=_PROVIDER, daily_limit=daily_limit)
            gate.bind_now(self._now)
        self._transport = HttpTransport(
            provider=_PROVIDER,
            base_url=self.BASE_URL,
            min_interval_seconds=float(min_interval_seconds),
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=_TIMEOUT_SECONDS,
            session=requests.Session(),
            sleep=lambda seconds: time.sleep(seconds),
            monotonic=lambda: time.monotonic(),
        )

    def _search_form(self, keyword: str, start: date, end: date, *, page_index: int) -> dict[str, str]:
        return {
            "method": "searchDetailsSub",
            "forward": "details_sub",
            "currentPageSize": str(_PAGE_SIZE),
            "pageIndex": str(page_index),
            "orderMode": "1",
            "orderStat": "D",
            "reportNm": keyword,
            "reportNmTemp": keyword,
            "fromDate": start.strftime("%Y-%m-%d"),
            "toDate": end.strftime("%Y-%m-%d"),
            "bfrDsclsType": "on",
            # KIND는 검색 폼의 모든 필드가 있어야 결과를 준다. 하나라도 빠지면 HTTP 200 오류 페이지를 돌려준다.
            **dict.fromkeys(_EMPTY_FORM_FIELDS, ""),
        }

    def search_page(self, keyword: str, start: date, end: date, *, page_index: int) -> str:
        """Return the raw HTML of one search page (100 rows per page, newest first).

        Raises:
            ValueError: empty keyword, ``start > end`` or ``page_index < 1``.
            ProviderRetryableError: KIND returned its error page.
        """
        if not keyword.strip():
            raise ValueError("keyword must not be empty")
        if start > end:
            raise ValueError("start must not be after end")
        if isinstance(page_index, bool) or int(page_index) < 1:
            raise ValueError(f"invalid page_index {page_index!r}: must be a positive integer")

        def _reject_error_page(response: requests.Response) -> None:
            if _ERROR_MARKER in response.text:
                raise ProviderRetryableError(
                    "KIND returned its error page for a notice search",
                    provider=_PROVIDER,
                    endpoint="disclosure/details.do",
                )

        response = self._transport.post_form(
            "disclosure/details.do",
            self._search_form(keyword.strip(), start, end, page_index=int(page_index)),
            headers=dict(_HEADERS),
            classify=_reject_error_page,
        )
        return response.text

    def fetch_document(self, acptno: str) -> KindNoticeDocument:
        """Resolve and download the body of one notice (three requests).

        Raises:
            ProviderTerminalError: the viewer lists no document or no body path.
        """
        receipt = str(acptno or "").strip()
        if len(receipt) != 14 or not receipt.isdigit():
            raise ValueError("acptno must be a 14-digit acceptance number")
        viewer = self._transport.get(
            "common/disclsviewer.do",
            {"method": "search", "acptno": receipt},
            headers=dict(_HEADERS),
        )
        options = [value for _, value in _OPTION_PATTERN.findall(viewer.text)]
        if not options:
            raise ProviderTerminalError(
                f"KIND viewer lists no document for {receipt}",
                provider=_PROVIDER,
                endpoint="common/disclsviewer.do",
            )
        doc_no = options[0].split("|")[0].strip()
        if not doc_no:
            raise ProviderTerminalError(
                f"KIND viewer lists no document for {receipt}",
                provider=_PROVIDER,
                endpoint="common/disclsviewer.do",
            )
        contents = self._transport.get(
            "common/disclsviewer.do",
            {"method": "searchContents", "docNo": doc_no},
            headers=dict(_HEADERS),
        )
        paths = _BODY_PATH_PATTERN.findall(contents.text)
        if not paths:
            raise ProviderTerminalError(
                f"KIND contents list no body path for {receipt}",
                provider=_PROVIDER,
                endpoint="common/disclsviewer.do",
            )
        path = paths[0]
        body = self._transport.get(path.lstrip("/"), {}, headers=dict(_HEADERS))
        html = _decode_body(body)
        if not html.strip():
            raise ProviderTerminalError(
                f"KIND body is empty for {receipt}",
                provider=_PROVIDER,
                endpoint=path.lstrip("/"),
            )
        return KindNoticeDocument(
            acptno=receipt,
            doc_no=doc_no,
            template=path.rsplit("/", 1)[-1],
            html=html,
        )

    def health_check(self) -> None:
        """Issue one search for a fixed past day and require a parseable page."""
        html = self.search_page("상장폐지", date(2020, 6, 1), date(2020, 6, 1), page_index=1)
        parse_kind_search_page(html)


def build_scoped_kind_client(
    *, policy: KindPolicy, quota_store: ProviderQuotaStateStore, now: Callable[[], datetime] | None = None
) -> KindClient:
    """Build the KIND collector from its provider policy."""
    return KindClient(
        quota_store=quota_store,
        daily_limit=policy.daily_limit,
        min_interval_seconds=policy.min_interval_seconds,
        now=now,
    )
