"""Official KRX historical collection with separated evidence streams."""
from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from src.data.schemas import PITDataError


class KrxHistoricalCollector:
    """Bounded historical KRX evidence; trade flow never maps to investor flow."""

    def __init__(self, api_key: str | None = None, *, request_json: Any | None = None) -> None:
        key = api_key or os.getenv("KRX_OPENAPI_KEY")
        if not key and request_json is None:
            raise ValueError("KRX_OPENAPI_KEY not found in environment variables")
        self._api_key = key
        self._request_json = request_json
        self._client: Any | None = None
        if request_json is None and key is not None:
            from src.integrations.krx.client import KrxApiClient

            self._client = KrxApiClient(api_key=key)

    def _check_range(self, start: date, end: date) -> None:
        if start > end:
            raise PITDataError("coverage_start must not be after coverage_end")

    def fetch_daily_market(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        if self._client is None:
            raise PITDataError("KRX daily-market endpoint is not configured")
        pages: list[dict[str, Any]] = []
        current = start
        while current <= end:
            records = self._client.fetch_trade_records(current)
            if not records:
                raise PITDataError(f"KRX daily market is empty for {current}; refusing to fabricate facts")
            pages.append({"records": records, "session": current.isoformat(), "retrieved_at": datetime.now().isoformat()})
            current = date.fromordinal(current.toordinal() + 1)
        return tuple(pages)

    def fetch_investor_flow(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        if self._request_json is None and self._client is None:
            raise PITDataError("KRX investor-flow endpoint is not configured")
        if self._client is not None:
            raise PITDataError("KRX investor-flow endpoint is not configured; trade records must not map to investor flow")
        request_json = self._request_json
        assert request_json is not None
        pages: list[dict[str, Any]] = []
        current = start
        while current <= end:
            payload = request_json("investor_flow", {"date": current.isoformat()})
            if not isinstance(payload, dict) or not payload:
                raise PITDataError(f"KRX investor flow is empty for {current}; certification blocked")
            pages.append(dict(payload))
            current = date.fromordinal(current.toordinal() + 1)
        if not pages:
            raise PITDataError("KRX investor-flow response is empty; certification blocked")
        return tuple(pages)

    def fetch_master_lineage(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        if self._client is None:
            raise PITDataError("KRX master-lineage endpoint is not configured")
        pages: list[dict[str, Any]] = []
        current = start
        while current <= end:
            records = self._client.fetch_master_records(current)
            if not records:
                raise PITDataError(f"KRX master lineage is empty for {current}; certification blocked")
            pages.append({"records": records, "session": current.isoformat()})
            current = date.fromordinal(current.toordinal() + 1)
        return tuple(pages)

    def fetch_status_and_actions(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        if self._client is None:
            raise PITDataError("KRX status-and-actions endpoint is not configured")
        pages: list[dict[str, Any]] = []
        current = start
        while current <= end:
            records = self._client.fetch_master_records(current)
            actions = [r for r in records if isinstance(r, dict) and (r.get("action_id") or r.get("status") or r.get("halt"))]
            if not actions:
                raise PITDataError(f"KRX status-and-actions response is empty for {current}; refusing to invent empty actions")
            pages.append({"records": actions, "session": current.isoformat()})
            current = date.fromordinal(current.toordinal() + 1)
        return tuple(pages)
