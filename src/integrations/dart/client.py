"""OpenDART transport-only client."""
from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Callable
from datetime import date
from typing import Any
from xml.etree import ElementTree

import requests


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
    ) -> None:
        self.api_key = api_key or os.getenv("OPENDART_API_KEY")
        if not self.api_key and request_json is None and raw_request_json is None:
            raise ValueError("OPENDART_API_KEY not found in environment variables")
        self._request_json = request_json
        self._raw_request_json = raw_request_json
        self._session = requests.Session()

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        if self._raw_request_json is not None:
            payload = self._raw_request_json(endpoint, params)
            if not isinstance(payload, dict):
                raise DartTerminalError("DART response must be an object")
            return payload
        if self._request_json is not None:
            # When only validated request is supplied, use it directly for transport
            # but still allow status handling in caller.
            payload = self._request_json(endpoint, params)
            if not isinstance(payload, dict):
                raise DartTerminalError("DART response must be an object")
            return payload
        response = self._session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=30)
        if response.status_code != 200:
            if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                raise DartRetryableError(f"DART HTTP {response.status_code} for {endpoint}")
            raise DartTerminalError(f"DART HTTP {response.status_code} for {endpoint}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise DartTerminalError(f"DART returned invalid JSON for {endpoint}") from exc
        if not isinstance(payload, dict):
            raise DartTerminalError(f"DART response must be an object for {endpoint}")
        return payload

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
        page_no: int = 1,
        page_count: int = 100,
    ) -> list[dict[str, str]]:
        if start > end:
            raise ValueError("start must not be after end")
        if not 1 <= page_count <= 100:
            raise ValueError("page_count must be within [1, 100]")
        params: dict[str, str] = {
            "crtfc_key": str(self.api_key),
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "page_no": str(page_no),
            "page_count": str(page_count),
        }
        if corp_code:
            params["corp_code"] = corp_code
        # Use validated request if injection was for validated path
        if self._request_json is not None and self._raw_request_json is None:
            # raw injection path: caller supplied request_json that returns raw payload
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
        records: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            rcept_no = str(item.get("rcept_no") or "").strip()
            rcept_dt = str(item.get("rcept_dt") or "").strip()
            if not rcept_no or not rcept_dt:
                raise DartApiError("DART disclosure lacks receipt identity")
            records.append(
                {
                    "rcept_no": rcept_no,
                    "rcept_dt": rcept_dt,
                    "corp_code": str(item.get("corp_code") or "").strip(),
                    "corp_name": str(item.get("corp_name") or "").strip(),
                    "report_nm": str(item.get("report_nm") or "").strip(),
                    "rm": str(item.get("rm") or "").strip(),
                }
            )
        return sorted(records, key=lambda x: (x["rcept_dt"], x["rcept_no"]))

    def load_corp_codes(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("api_key is required for corpCode")
        response = requests.get(
            f"{self.BASE_URL}/corpCode.xml",
            params={"crtfc_key": str(self.api_key)},
            timeout=30,
        )
        if response.status_code != 200:
            raise DartApiError(f"DART corpCode HTTP {response.status_code}")
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                xml_bytes = archive.read("CORPCODE.xml")
            if b"<!DOCTYPE" in xml_bytes or b"<!ENTITY" in xml_bytes:
                raise DartApiError("DART corpCode.xml contains unsafe XML declarations")
            root = ElementTree.fromstring(xml_bytes)  # noqa: S314
        except zipfile.BadZipFile as exc:
            raise DartApiError("DART corpCode.xml could not be parsed") from exc
        except (OSError, KeyError, ElementTree.ParseError) as exc:
            raise DartApiError("DART corpCode.xml could not be parsed") from exc
        mapping: dict[str, str] = {}
        for item in root.findall("list"):
            ticker = (item.findtext("stock_code") or "").strip()
            corp_code = (item.findtext("corp_code") or "").strip()
            if ticker and corp_code:
                mapping[ticker] = corp_code
        if not mapping:
            raise DartApiError("DART corpCode.xml contained no listed tickers")
        return mapping
