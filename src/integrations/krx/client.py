"""KRX transport-only client."""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import date
from enum import Enum
from typing import Any, ClassVar

import requests


class KrxApiError(RuntimeError):
    """KRX API failure."""


class KrxMarket(str, Enum):  # noqa: UP042
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    ALL = "ALL"


JsonRequest = Callable[[str, dict[str, str]], dict[str, Any]]


class KrxApiClient:
    BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
    ENDPOINTS: ClassVar[dict[str, str]] = {  # noqa: RUF012
        "KOSPI_INFO": "sto/stk_isu_base_info",
        "KOSDAQ_INFO": "sto/ksq_isu_base_info",
        "KOSPI_TRADE": "sto/stk_bydd_trd",
        "KOSDAQ_TRADE": "sto/ksq_bydd_trd",
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_json: JsonRequest | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("KRX_OPENAPI_KEY")
        if not self.api_key and request_json is None:
            raise ValueError("KRX_OPENAPI_KEY not found in environment variables")
        self._request_json = request_json or self._request
        self._session = requests.Session()
        self._last_request_time = 0.0
        self._request_count = 0

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        if self._request_count >= 10_000:
            raise KrxApiError("KRX daily request limit reached")
        last_error = "unknown KRX response error"
        for attempt in range(3):
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < 0.2:
                time.sleep(0.2 - elapsed)
            self._last_request_time = time.monotonic()
            self._request_count += 1
            response = self._session.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                headers={"AUTH_KEY": str(self.api_key)},
                timeout=30,
            )
            if response.status_code != 200:
                last_error = f"KRX HTTP {response.status_code} for {endpoint}"
            else:
                try:
                    payload = response.json()
                except ValueError:
                    last_error = f"KRX returned invalid JSON for {endpoint}"
                else:
                    if isinstance(payload, dict):
                        return payload
                    last_error = f"KRX response must be an object for {endpoint}"
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
        raise KrxApiError(last_error)

    def _records(self, endpoint: str, as_of: date) -> list[dict[str, Any]]:
        if not isinstance(as_of, date):
            raise ValueError("as_of must be a date")
        payload = self._request_json(endpoint, {"basDd": as_of.strftime("%Y%m%d")})
        if not isinstance(payload, dict):
            raise KrxApiError(f"KRX response must be an object for {endpoint}")
        records = payload.get("OutBlock_1", [])
        if not isinstance(records, list):
            raise KrxApiError(f"KRX records must be a list for {endpoint}")
        return [r for r in records if isinstance(r, dict)]

    def fetch_master_records(self, as_of: date, market: KrxMarket = KrxMarket.ALL) -> list[dict[str, Any]]:
        if not isinstance(market, KrxMarket):
            try:
                market = KrxMarket(market)
            except Exception as exc:
                raise ValueError(f"unknown KRX market {market!r}") from exc
        if market == KrxMarket.KOSPI:
            return self._records(self.ENDPOINTS["KOSPI_INFO"], as_of)
        if market == KrxMarket.KOSDAQ:
            return self._records(self.ENDPOINTS["KOSDAQ_INFO"], as_of)
        # ALL: both
        if market == KrxMarket.ALL:
            recs: list[dict[str, Any]] = []
            recs.extend(self._records(self.ENDPOINTS["KOSPI_INFO"], as_of))
            recs.extend(self._records(self.ENDPOINTS["KOSDAQ_INFO"], as_of))
            return recs
        raise ValueError(f"unknown KRX market {market!r}")

    def fetch_trade_records(self, as_of: date, market: KrxMarket = KrxMarket.ALL) -> list[dict[str, Any]]:
        if not isinstance(market, KrxMarket):
            try:
                market = KrxMarket(market)
            except Exception as exc:
                raise ValueError(f"unknown KRX market {market!r}") from exc
        recs: list[dict[str, Any]] = []
        if market in (KrxMarket.KOSPI, KrxMarket.ALL):
            recs.extend(self._records(self.ENDPOINTS["KOSPI_TRADE"], as_of))
        if market in (KrxMarket.KOSDAQ, KrxMarket.ALL):
            recs.extend(self._records(self.ENDPOINTS["KOSDAQ_TRADE"], as_of))
        return recs
