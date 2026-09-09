"""Official KRX historical collection with separated evidence streams."""
from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from src.core.pit import PITDataError
from src.integrations.quota import ProviderQuotaStateStore


def _validate_daily_market_records(records: list[dict[str, Any]], *, session: date) -> None:
    """Require KRX valuation fields before persisting a daily Bronze page."""
    missing: set[str] = set()
    for record in records:
        if not any(record.get(name) not in (None, "") for name in ("market_cap", "marcap", "MKTCAP")):
            missing.add("MKTCAP")
        if not any(record.get(name) not in (None, "") for name in ("shares_outstanding", "list_shrs", "LIST_SHRS")):
            missing.add("LIST_SHRS")
    if missing:
        raise PITDataError(  # pragma: no cover
            f"KRX daily market missing {' and '.join(name for name in ('MKTCAP', 'LIST_SHRS') if name in missing)} for {session}; certification blocked"
        )


class KrxHistoricalCollector:
    """Bounded historical KRX evidence; trade flow never maps to investor flow."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_json: Any | None = None,
        quota_store: ProviderQuotaStateStore | None = None,
        min_interval: float | None = None,
    ) -> None:
        raw_key = api_key or os.getenv("KRX_OPENAPI_KEY")
        key = raw_key.strip().strip("\"'") if raw_key else raw_key
        if not key and request_json is None:
            raise ValueError("KRX_OPENAPI_KEY not found in environment variables")
        self._api_key = key
        self._request_json = request_json
        self._client: Any | None = None
        if request_json is None and key is not None:
            from src.integrations.krx.client import KrxApiClient

            self._client = KrxApiClient(
                api_key=key,
                quota_store=quota_store,
                min_interval=min_interval,
            )

    def _check_range(self, start: date, end: date) -> None:
        if start > end:
            raise PITDataError("coverage_start must not be after coverage_end")

    def fetch_daily_market(self, start: date, end: date, *, sessions: Iterable[date] | None = None) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        if self._client is None:
            raise PITDataError("KRX daily-market endpoint is not configured")
        if sessions is not None:
            planned = tuple(sessions)
            for day in planned:
                if not isinstance(day, date):
                    raise PITDataError("sessions must contain dates only")
                if day < start or day > end:
                    raise PITDataError("planned session outside requested range")
            targets = tuple(sorted({day for day in planned if start <= day <= end}))
        else:
            current_all = start
            targets = ()
            while current_all <= end:
                targets += (current_all,)
                current_all = date.fromordinal(current_all.toordinal() + 1)
        for current in targets:
            records = self._client.fetch_trade_records(current)
            if not records:
                raise PITDataError(f"KRX daily market is empty for {current}; refusing to fabricate facts")
            _validate_daily_market_records(records, session=current)
            yield {"records": records, "session": current.isoformat(), "retrieved_at": datetime.now().isoformat()}

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

    def fetch_master_lineage(self, start: date, end: date, *, sessions: Iterable[date] | None = None) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        if self._client is None:
            raise PITDataError("KRX master-lineage endpoint is not configured")
        if sessions is not None:
            planned = tuple(sessions)
            for day in planned:
                if not isinstance(day, date):
                    raise PITDataError("sessions must contain dates only")
                if day < start or day > end:
                    raise PITDataError("planned session outside requested range")
            targets = tuple(sorted({day for day in planned if start <= day <= end}))
        else:
            current_all = start
            targets = ()
            while current_all <= end:
                targets += (current_all,)
                current_all = date.fromordinal(current_all.toordinal() + 1)
        pages: list[dict[str, Any]] = []
        for current in targets:
            records = self._client.fetch_master_records(current)
            if not records:
                raise PITDataError(f"KRX master lineage is empty for {current}; certification blocked")
            pages.append({"records": records, "session": current.isoformat()})
        return tuple(pages)

    def fetch_corporate_actions(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        self._check_range(start, end)
        raise PITDataError(
            "KRX corporate-action endpoint is not configured: unsupported legacy adapter; "  # pragma: no cover
            "use OpenDART structured decisions via collect_opendart_corporate_action_evidence"
        )
