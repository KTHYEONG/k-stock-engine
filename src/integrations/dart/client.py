"""OpenDART transport-only client."""
from __future__ import annotations

import io
import os
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

import requests


@dataclass(frozen=True, slots=True)
class DartCorpCodeRecord:
    ticker: str
    corp_code: str
    corp_name: str


class DartApiError(RuntimeError):
    """Base DART API failure."""


class DartRetryableError(DartApiError):
    """Transient DART failure."""


class DartTerminalError(DartApiError):
    """Permanent DART failure."""


JsonRequest = Callable[[str, dict[str, str]], dict[str, Any]]


OK_DART_STATUS = "000"
EMPTY_DART_STATUS = "013"
BLOCKED_DART_STATUS = "020"
RETRYABLE_DART_STATUSES = frozenset({"800", "900"})


class DartApiClient:
    BASE_URL = "https://opendart.fss.or.kr/api"
    DISCLOSURE_ENDPOINT = "list.json"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_json: JsonRequest | None = None,
        raw_request_json: JsonRequest | None = None,
        request_bytes: Callable[[str, dict[str, str]], bytes] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENDART_API_KEY")
        if not self.api_key and request_json is None and raw_request_json is None and request_bytes is None:
            raise ValueError("OPENDART_API_KEY not found in environment variables")
        self._request_json = request_json
        self._raw_request_json = raw_request_json
        self._request_bytes = request_bytes
        self._session = requests.Session()

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        request_params = dict(params)
        if self.api_key and "crtfc_key" not in request_params:
            request_params["crtfc_key"] = str(self.api_key)
        if self._raw_request_json is not None:
            payload = self._raw_request_json(endpoint, request_params)
            if not isinstance(payload, dict):
                raise DartTerminalError("DART response must be an object")
            return payload
        if self._request_json is not None:
            # When only validated request is supplied, use it directly for transport
            # but still allow status handling in caller.
            payload = self._request_json(endpoint, request_params)
            if not isinstance(payload, dict):
                raise DartTerminalError("DART response must be an object")
            return payload
        for attempt in range(3):
            response = self._session.get(f"{self.BASE_URL}/{endpoint}", params=request_params, timeout=30)
            if response.status_code != 200:
                if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                    if attempt < 2:
                        time.sleep(0.25 * (attempt + 1))
                        continue
                    raise DartRetryableError(f"DART HTTP {response.status_code} for {endpoint}")
                raise DartTerminalError(f"DART HTTP {response.status_code} for {endpoint}")
            try:
                payload = response.json()
            except ValueError:
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                # Some OpenDART edges return an empty 200 response on a reused
                # connection; retry once with a fresh connection before failing.
                try:
                    fresh = requests.get(
                        f"{self.BASE_URL}/{endpoint}", params=request_params, timeout=30
                    )
                    payload = fresh.json()
                except (requests.RequestException, ValueError) as fresh_exc:
                    raise DartRetryableError(
                        f"DART returned transient invalid JSON for {endpoint}"
                    ) from fresh_exc
                if not isinstance(payload, dict):
                    raise DartTerminalError(
                        f"DART response must be an object for {endpoint}"
                    ) from None
                return payload
            if not isinstance(payload, dict):
                raise DartTerminalError(f"DART response must be an object for {endpoint}")
            return payload
        raise DartRetryableError(f"DART request exhausted retries for {endpoint}")

    def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        payload = self._request(endpoint, params)
        status = payload.get("status")
        if status == OK_DART_STATUS:
            return payload
        if status == EMPTY_DART_STATUS:
            return payload
        if status == BLOCKED_DART_STATUS:
            raise DartApiError(f"DART status {status}: {payload}")
        if status in RETRYABLE_DART_STATUSES:
            raise DartRetryableError(f"DART status {status}: {payload}")
        # Any other non-000 is a terminal/api error, but contract expects DartApiError match
        raise DartApiError(f"DART status {status}: {payload}")

    def list_disclosures(
        self,
        start: date,
        end: date,
        *,
        corp_code: str | None = None,
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
            if self._request_json is not None and self._raw_request_json is None:
                payload = self._request_json(self.DISCLOSURE_ENDPOINT, params)
                if not isinstance(payload, dict):
                    raise DartTerminalError("DART response must be an object")
                status = payload.get("status")
                if status != OK_DART_STATUS:
                    raise DartApiError(f"DART status {status}: {payload}")
                raw = payload.get("list", [])
            else:
                payload = self._request_validated(self.DISCLOSURE_ENDPOINT, params)
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
                        raise DartApiError(
                            f"DART disclosure receipt {rcept_no} has contradictory records"
                        )
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
        payload = self._request_validated(
            "fnlttMultiAcnt.json",
            {
                "corp_code": ",".join(codes),
                "bsns_year": str(biz_year),
                "reprt_code": str(reprt_code),
            },
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
                raise DartTerminalError("DART document archive is empty")
            raw = bytes(payload)
            stripped = raw.lstrip()[:1]
            if stripped == b"{":
                raise DartTerminalError("DART document archive returned an error payload")
            return raw
        for attempt in range(3):
            response = self._session.get(f"{self.BASE_URL}/document.xml", params=params, timeout=30)
            if response.status_code != 200:
                if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                    if attempt < 2:
                        time.sleep(0.25 * (attempt + 1))
                        continue
                    raise DartRetryableError(f"DART HTTP {response.status_code} for document.xml")
                raise DartTerminalError(f"DART HTTP {response.status_code} for document.xml")
            content = response.content
            if not content:
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise DartTerminalError("DART document archive is empty")
            if content.lstrip()[:1] == b"{":
                raise DartTerminalError("DART document archive returned an error payload")
            return content
        raise DartRetryableError("DART request exhausted retries for document.xml")

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
            response = requests.get(
                f"{self.BASE_URL}/corpCode.xml",
                params={"crtfc_key": str(self.api_key)},
                timeout=30,
            )
            if response.status_code != 200:
                raise DartApiError(f"DART corpCode HTTP {response.status_code}")
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
