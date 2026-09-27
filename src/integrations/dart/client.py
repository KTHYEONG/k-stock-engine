"""OpenDART transport-only client."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Final, Protocol
from xml.etree import ElementTree

import requests

from src.integrations.errors import (
    DartApiError,
    ProviderQuotaExhaustedError,
    ProviderRetryableError,
    ProviderTerminalError,
)
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HttpTransport, RetryPolicy

__all__ = [
    "DartApiError",
    "DartClientProtocol",
    "DartCorpCodeRecord",
    "DartCorporateActionPage",
    "DartDividendPage",
    "ProviderQuotaExhaustedError",
    "ProviderRetryableError",
    "ProviderTerminalError",
    "classify_dart_status",
    "dart_ledger_for_key",
    "dart_quota_provider",
]


@dataclass(frozen=True, slots=True)
class DartCorpCodeRecord:
    ticker: str
    corp_code: str
    corp_name: str


@dataclass(frozen=True, slots=True)
class DartCorporateActionPage:
    endpoint: str
    corp_code: str
    status: str
    records: tuple[dict[str, Any], ...]


_CORPORATE_ACTION_ENDPOINTS: tuple[str, ...] = (
    "fricDecsn.json",
    "crDecsn.json",
    "piicDecsn.json",
    "cmpDvDecsn.json",
    "cmpMgDecsn.json",
)


@dataclass(frozen=True, slots=True)
class DartDividendPage:
    corp_code: str
    bsns_year: str
    reprt_code: str
    status: str
    records: tuple[dict[str, Any], ...]


_DIVIDEND_ENDPOINT = "alotMatter.json"
_DIVIDEND_REPORT_CODES: tuple[str, ...] = ("11011", "11012", "11013", "11014")

JsonRequest = Callable[[str, dict[str, str]], dict[str, Any]]

OK_DART_STATUS = "000"
EMPTY_DART_STATUS = "013"
ABSENT_DART_STATUS = "014"
BLOCKED_DART_STATUS = "020"
EMPTY_DART_STATUSES = frozenset({EMPTY_DART_STATUS, ABSENT_DART_STATUS})
RETRYABLE_DART_STATUSES = frozenset({"800", "900"})

_PROVIDER = "OpenDART"
_PING_CORP_CODE = "00126380"
_MAX_HTTP_ATTEMPTS: Final = 3
_CONNECTION_FAILURE_COOLDOWN_SECONDS = 300.0
_SAFE_DAILY_REQUEST_LIMIT = 15_200


def dart_quota_provider(api_key: str | None, *, primary_api_key: str | None = None) -> str:
    """Return the quota-ledger provider name that meters one OpenDART key."""
    if not api_key or (primary_api_key is not None and api_key == primary_api_key):
        return _PROVIDER
    return f"{_PROVIDER}#{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:8]}"


def dart_ledger_for_key(*, key_env: str, primary_key_env: str, api_key: str) -> str:
    """Ledger name for one declared key without reading the environment."""
    if not api_key or key_env == primary_key_env:
        return _PROVIDER
    return f"{_PROVIDER}#{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:8]}"


def classify_dart_status(status: str, payload: Mapping[str, Any], endpoint: str) -> None:
    """Classify one DART status code into the shared error tree.

    ``000`` is success; ``013``/``014`` are empty/absent; ``020`` means quota
    exhausted; ``800``/``900`` are retryable; anything else is terminal.
    """
    if status == OK_DART_STATUS or status in EMPTY_DART_STATUSES:
        return
    if status == BLOCKED_DART_STATUS:
        raise ProviderQuotaExhaustedError(f"DART status {status}: {dict(payload)}")
    if status in RETRYABLE_DART_STATUSES:
        raise ProviderRetryableError(f"DART status {status}: {dict(payload)}")
    raise ProviderTerminalError(f"DART status {status}: {dict(payload)}")


class DartClientProtocol(Protocol):
    """Public DART surface used by collectors (no private members)."""

    def request_validated(self, endpoint: str, params: Mapping[str, str]) -> dict[str, Any]: ...
    def list_disclosures(
        self, start: date, end: date, *, corp_code: str | None = ..., detail_type: str | None = ..., page_count: int = ...
    ) -> list[dict[str, str]]: ...
    def fetch_document_archive(self, rcept_no: str) -> bytes: ...
    def ping(self) -> None: ...
    def load_corp_code_records(self) -> tuple[DartCorpCodeRecord, ...]: ...


class DartApiClient:
    BASE_URL = "https://opendart.fss.or.kr/api"
    DISCLOSURE_ENDPOINT = "list.json"

    def __init__(
        self,
        api_key: str,
        *,
        min_interval: float,
        request_json: JsonRequest | None = None,
        raw_request_json: JsonRequest | None = None,
        request_bytes: Callable[[str, dict[str, str]], bytes] | None = None,
        quota_store: ProviderQuotaStateStore | None = None,
        quota_provider: str | None = None,
        now: Callable[[], datetime] | None = None,
        daily_request_limit: int | None = None,
    ) -> None:
        self.api_key = api_key
        if not self.api_key and request_json is None and raw_request_json is None and request_bytes is None:
            raise ValueError("DART api_key is required")
        self._request_json = request_json
        self._raw_request_json = raw_request_json
        self._request_bytes = request_bytes
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=25, pool_maxsize=25)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; KStockEngine/1.0; +https://github.com/KTHYEONG/k-stock-engine)",
            "Accept": "application/json, text/plain, */*",
        })
        self._quota_store = quota_store
        self._provider = quota_provider or dart_quota_provider(self.api_key)
        if daily_request_limit is not None and (isinstance(daily_request_limit, bool) or int(daily_request_limit) < 1):
            raise ValueError("daily_request_limit must be a positive integer")
        self._daily_request_limit = int(daily_request_limit) if daily_request_limit is not None else _SAFE_DAILY_REQUEST_LIMIT
        self._now = now or (lambda: datetime.now(UTC))
        self._min_interval = float(min_interval)
        gate: LedgerQuotaGate | None = None
        if self._quota_store is not None:
            gate = LedgerQuotaGate(self._quota_store, provider=self._provider, daily_limit=self._daily_request_limit)
            gate.bind_now(self._now)
        self._quota_gate = gate
        self._transport = HttpTransport(
            provider=self._provider,
            base_url=self.BASE_URL,
            min_interval_seconds=self._min_interval,
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=30.0,
            session=self._session,
            sleep=lambda seconds: time.sleep(seconds),
            monotonic=lambda: time.monotonic(),
        )

    def _sync_session(self) -> None:
        self._transport._session = self._session

    def _pace(self) -> None:
        self._sync_session()
        self._transport._pace()

    def _with_key(self, params: Mapping[str, str]) -> dict[str, str]:
        query = dict(params)
        if self.api_key and "crtfc_key" not in query:
            query["crtfc_key"] = str(self.api_key)
        return query

    def _http_get(self, endpoint: str, params: Mapping[str, str]) -> requests.Response:
        """Issue one logical OpenDART GET with ledgered, paced, bounded retries."""
        self._sync_session()
        return self._transport.get(endpoint, self._with_key(params))

    def ping(self) -> None:
        """Send one ledgered ``company.json`` request."""
        self.request_validated("company.json", {"corp_code": _PING_CORP_CODE})

    def _seam_payload(self, endpoint: str, params: Mapping[str, str]) -> dict[str, Any] | None:
        if self._raw_request_json is not None:
            payload = self._raw_request_json(endpoint, dict(params))
            if not isinstance(payload, dict):
                raise ProviderTerminalError("DART response must be an object")
            return payload
        if self._request_json is not None:
            payload = self._request_json(endpoint, dict(params))
            if not isinstance(payload, dict):
                raise ProviderTerminalError("DART response must be an object")
            return payload
        return None

    def _fetch_json(self, endpoint: str, params: Mapping[str, str], *, validated: bool) -> dict[str, Any]:
        query = self._with_key(params)
        seam = self._seam_payload(endpoint, query)
        if seam is not None:
            if validated:
                try:
                    self._check_validated_status(endpoint, seam)
                except ProviderQuotaExhaustedError:
                    if self._quota_store is not None:
                        self._quota_store.record_rate_limit(
                            provider=self._provider, endpoint=endpoint, now=self._now(), retry_after=None
                        )
                    raise
            return seam
        parsed: list[dict[str, Any]] = []

        def _classify(response: requests.Response) -> None:
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderRetryableError(f"DART returned transient invalid JSON for {endpoint}") from exc
            if not isinstance(payload, dict):
                raise ProviderTerminalError(f"DART response must be an object for {endpoint}")
            if validated:
                self._check_validated_status(endpoint, payload)
            parsed.append(payload)

        self._sync_session()
        try:
            self._transport.get(endpoint, query, classify=_classify)
        except ProviderQuotaExhaustedError:
            raise
        except ProviderRetryableError as exc:
            cause = exc.__cause__
            if isinstance(cause, requests.exceptions.RequestException) and self._quota_store is not None:
                self._quota_store.record_rate_limit(
                    provider=self._provider,
                    endpoint=endpoint,
                    now=self._now(),
                    retry_after=_CONNECTION_FAILURE_COOLDOWN_SECONDS,
                )
            raise
        except ProviderTerminalError:
            raise
        return parsed[0]

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        return self._fetch_json(endpoint, params, validated=False)

    def _check_validated_status(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        status = str(payload.get("status") or "")
        if status == OK_DART_STATUS or status in EMPTY_DART_STATUSES:
            return payload
        if status == BLOCKED_DART_STATUS:
            raise ProviderQuotaExhaustedError(f"DART status {status}: {payload}")
        if status in RETRYABLE_DART_STATUSES:
            raise ProviderRetryableError(f"DART status {status}: {payload}")
        raise ProviderTerminalError(f"DART status {status}: {payload}")

    def request_validated(self, endpoint: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Fetch one validated DART payload; ``013``/``014`` return as empty pages."""
        return self._fetch_json(endpoint, dict(params), validated=True)

    def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        return self.request_validated(endpoint, params)

    def list_disclosures(
        self,
        start: date,
        end: date,
        *,
        corp_code: str | None = None,
        detail_type: str | None = None,
        page_count: int = 100,
    ) -> list[dict[str, str]]:
        if start > end:
            raise ValueError("start must not be after end")
        if not 1 <= page_count <= 100:
            raise ValueError("page_count must be within [1, 100]")
        by_receipt: dict[str, dict[str, str]] = {}
        expected_total: int | None = None
        page_no = 1
        while True:
            params: dict[str, str] = {
                "crtfc_key": str(self.api_key),
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_no": str(page_no),
                "page_count": str(page_count),
            }
            if corp_code:
                params["corp_code"] = corp_code
            if detail_type:
                params["pblntf_detail_ty"] = str(detail_type).strip()
            if self._request_json is not None and self._raw_request_json is None:
                payload = self._request_json(self.DISCLOSURE_ENDPOINT, params)
                if not isinstance(payload, dict):
                    raise ProviderTerminalError("DART response must be an object")
                status = payload.get("status")
                if status != OK_DART_STATUS:
                    raise DartApiError(f"DART status {status}: {payload}")
                raw = payload.get("list", [])
            else:
                payload = self.request_validated(self.DISCLOSURE_ENDPOINT, params)
                raw = payload.get("list", [])
            if not isinstance(raw, list):
                raise DartApiError("DART disclosure list must be a list")
            total_raw = payload.get("total_page", payload.get("totalPage"))
            if total_raw is None:
                total_page = 1
            else:
                try:
                    total_page = int(str(total_raw).strip())
                except (TypeError, ValueError) as exc:
                    raise DartApiError("DART disclosure pagination metadata is invalid") from exc
                if total_page < 1:
                    raise DartApiError("DART disclosure pagination metadata is invalid")
            if expected_total is not None and total_page != expected_total:
                raise DartApiError("DART disclosure pagination metadata is contradictory")
            expected_total = total_page
            if page_no > total_page:
                raise DartApiError("DART disclosure pagination metadata is contradictory")
            for item in raw:
                if not isinstance(item, dict):
                    continue
                rcept_no = str(item.get("rcept_no") or "").strip()
                rcept_dt = str(item.get("rcept_dt") or "").strip()
                if not rcept_no or not rcept_dt:
                    raise DartApiError("DART disclosure lacks receipt identity")
                candidate = {
                    "rcept_no": rcept_no,
                    "rcept_dt": rcept_dt,
                    "corp_code": str(item.get("corp_code") or "").strip(),
                    "corp_name": str(item.get("corp_name") or "").strip(),
                    "report_nm": str(item.get("report_nm") or "").strip(),
                    "rm": str(item.get("rm") or "").strip(),
                }
                previous = by_receipt.get(rcept_no)
                if previous is not None:
                    if previous != candidate:
                        raise DartApiError(f"DART disclosure receipt {rcept_no} has contradictory records")
                    continue
                by_receipt[rcept_no] = candidate
            if page_no >= total_page:
                break
            page_no += 1
            if page_no > 10000:
                raise DartApiError("DART disclosure pagination exceeded safe bounds")
        return sorted(by_receipt.values(), key=lambda x: (x["rcept_dt"], x["rcept_no"]))

    def fetch_multi_accounts(
        self, corp_codes: tuple[str, ...], *, biz_year: str, reprt_code: str
    ) -> list[dict[str, Any]]:
        """Fetch up to 100 companies' major accounts in one official request."""
        codes = tuple(dict.fromkeys(code.strip() for code in corp_codes if code.strip()))
        if not 1 <= len(codes) <= 100:
            raise ValueError("corp_codes must contain between 1 and 100 companies")
        if len(str(biz_year)) != 4 or str(reprt_code) not in {"11011", "11012", "11013", "11014"}:
            raise ValueError("invalid business year or report code")
        payload = self.request_validated(
            "fnlttMultiAcnt.json",
            {"corp_code": ",".join(codes), "bsns_year": str(biz_year), "reprt_code": str(reprt_code)},
        )
        records = payload.get("list", [])
        if not isinstance(records, list):
            raise DartApiError("DART multi-account list must be a list")
        return [dict(record) for record in records if isinstance(record, dict)]

    def fetch_document_archive(self, rcept_no: str) -> bytes:
        """Fetch document.xml ZIP archive for a 14-digit receipt number."""
        receipt = str(rcept_no or "").strip()
        if len(receipt) != 14 or not receipt.isdigit():
            raise ValueError("rcept_no must be a 14-digit receipt number")
        params = {"rcept_no": receipt}
        if self.api_key:
            params["crtfc_key"] = str(self.api_key)
        if self._request_bytes is not None:
            payload = self._request_bytes("document.xml", dict(params))
            if not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
                raise ProviderTerminalError("DART document archive is empty")
            raw = bytes(payload)
            if raw.lstrip()[:1] == b"{":
                raise ProviderTerminalError("DART document archive returned an error payload")
            return raw
        outcome: list[bytes] = []
        empty_hits = {"n": 0}

        def _classify(response: requests.Response) -> None:
            content = response.content
            if not content:
                empty_hits["n"] += 1
                if empty_hits["n"] >= _MAX_HTTP_ATTEMPTS:
                    raise ProviderTerminalError("DART document archive is empty")
                raise ProviderRetryableError("DART document archive is empty")
            if content.lstrip()[:1] == b"{":
                raise ProviderTerminalError("DART document archive returned an error payload")
            outcome.append(bytes(content))

        self._sync_session()
        self._transport.get("document.xml", params, classify=_classify)
        return outcome[0]

    def fetch_corporate_action_decisions(
        self, *, corp_codes: Sequence[str], start: date, end: date
    ) -> tuple[DartCorporateActionPage, ...]:
        if start > end:
            raise ValueError("start must not be after end")
        codes = tuple(str(code).strip() for code in corp_codes if str(code).strip())
        if not codes:
            raise ValueError("corp_codes must not be empty")
        pages: list[DartCorporateActionPage] = []
        for corp_code in codes:
            for endpoint in _CORPORATE_ACTION_ENDPOINTS:
                payload = self.request_validated(
                    endpoint, {"corp_code": corp_code, "bgn_de": start.strftime("%Y%m%d"), "end_de": end.strftime("%Y%m%d")}
                )
                status = str(payload.get("status") or "")
                raw = payload.get("list", [])
                records = tuple(dict(item) for item in raw if isinstance(item, dict)) if isinstance(raw, list) else ()
                pages.append(DartCorporateActionPage(endpoint=endpoint, corp_code=corp_code, status=status, records=records))
        return tuple(pages)

    def fetch_dividend_disclosures(
        self, *, corp_codes: Sequence[str], bsns_years: Sequence[str]
    ) -> tuple[DartDividendPage, ...]:
        codes = tuple(str(code).strip() for code in corp_codes if str(code).strip())
        if not codes:
            raise ValueError("corp_codes must not be empty")
        years = tuple(str(year).strip() for year in bsns_years if str(year).strip())
        if not years:
            raise ValueError("bsns_years must not be empty")
        pages: list[DartDividendPage] = []
        for corp_code in codes:
            for year in years:
                for reprt_code in _DIVIDEND_REPORT_CODES:
                    payload = self.request_validated(
                        _DIVIDEND_ENDPOINT, {"corp_code": corp_code, "bsns_year": year, "reprt_code": reprt_code}
                    )
                    status = str(payload.get("status") or "")
                    raw = payload.get("list", [])
                    records = tuple(dict(item) for item in raw if isinstance(item, dict)) if isinstance(raw, list) else ()
                    pages.append(
                        DartDividendPage(corp_code=corp_code, bsns_year=year, reprt_code=reprt_code, status=status, records=records)
                    )
        return tuple(pages)

    def load_corp_code_records(self) -> tuple[DartCorpCodeRecord, ...]:
        import re as _re

        raw: bytes
        if self._request_bytes is not None:
            payload = self._request_bytes("corpCode.xml", {"crtfc_key": str(self.api_key)})
            if not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
                raise DartApiError("DART corpCode.xml is empty")
            raw = bytes(payload)
        else:
            if not self.api_key:
                raise ValueError("api_key is required for corpCode")
            self._sync_session()
            response = self._transport.get("corpCode.xml", {"crtfc_key": str(self.api_key)})
            raw = response.content
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                xml_bytes = archive.read("CORPCODE.xml")
        except (zipfile.BadZipFile, OSError, KeyError) as exc:
            raise DartApiError("DART corpCode.xml could not be parsed") from exc
        if b"<!DOCTYPE" in xml_bytes or b"<!ENTITY" in xml_bytes:
            raise DartApiError("DART corpCode.xml contains unsafe XML declarations")
        try:
            root = ElementTree.fromstring(xml_bytes)  # noqa: S314
        except ElementTree.ParseError as exc:
            raise DartApiError("DART corpCode.xml could not be parsed") from exc
        ticker_re = _re.compile(r"^\d{6}$")
        seen: dict[str, DartCorpCodeRecord] = {}
        for item in root.findall("list"):
            ticker = (item.findtext("stock_code") or "").strip()
            corp_code = (item.findtext("corp_code") or "").strip()
            corp_name = (item.findtext("corp_name") or "").strip()
            if not ticker or not ticker_re.match(ticker):
                continue
            if not corp_code:
                continue
            prev = seen.get(ticker)
            if prev is not None and prev.corp_code != corp_code:
                raise DartApiError(f"DART corpCode.xml maps ticker {ticker} to multiple corp codes")
            if prev is None:
                seen[ticker] = DartCorpCodeRecord(ticker=ticker, corp_code=corp_code, corp_name=corp_name)
        if not seen:
            raise DartApiError("DART corpCode.xml contained no listed tickers")
        return tuple(seen[t] for t in sorted(seen))

    def load_corp_codes(self) -> dict[str, str]:
        return {rec.ticker: rec.corp_code for rec in self.load_corp_code_records()}
