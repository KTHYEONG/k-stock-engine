"""Official DART XBRL collection preserving filing identity."""
from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import date
from typing import Any

from src.data.schemas import PITDataError

_REQUIRED_FACTS: tuple[str, ...] = (
    "sales",
    "gross_profit",
    "operating_profit",
    "net_income",
    "assets",
    "equity",
    "cash",
    "debt",
    "operating_cash_flow",
    "capex",
)


class DartXbrlCollector:
    """Filing-identity plus XBRL-facts evidence; missing XBRL blocks certification."""

    def __init__(self, api_key: str | None = None, *, request_json: Any | None = None) -> None:
        key = api_key or os.getenv("OPENDART_API_KEY")
        if not key and request_json is None:
            raise ValueError("OPENDART_API_KEY not found in environment variables")
        self._api_key = key
        self._request_json = request_json
        self._client: Any | None = None
        if request_json is None and key is not None:
            from src.integrations.dart.client import DartApiClient

            self._client = DartApiClient(api_key=key)

    def fetch_disclosures(self, start: date, end: date) -> Iterable[dict[str, Any]]:
        if start > end:
            raise PITDataError("coverage_start must not be after coverage_end")
        if self._client is None:
            raise PITDataError("DART disclosures endpoint is not configured")
        records = self._client.list_disclosures(start, end)
        if not records:
            raise PITDataError("DART disclosures response is empty; certification blocked")
        return ({"records": records, "start": start.isoformat(), "end": end.isoformat()},)

    def fetch_xbrl_facts(self, filing_ids: tuple[Any, ...]) -> Iterable[dict[str, Any]]:
        if not filing_ids:
            raise PITDataError("DART XBRL facts require filing identities; certification blocked")
        identities: list[dict[str, str]] = []
        for item in filing_ids:
            if isinstance(item, str):
                if not item.strip():
                    raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
                raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
            if not isinstance(item, dict):
                raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
            corp_code = str(item.get("corp_code") or "").strip()
            filing_id = str(item.get("filing_id") or item.get("rcept_no") or "").strip()
            biz_year = str(item.get("biz_year") or item.get("bsns_year") or "").strip()
            reprt_code = str(item.get("reprt_code") or item.get("report_code") or "").strip()
            fs_div = str(item.get("fs_div") or "").strip()
            if not corp_code or not filing_id or not biz_year or not reprt_code or not fs_div:
                raise PITDataError("DART XBRL facts require filing identity with corp_code and filing_id")
            identities.append(
                {"corp_code": corp_code, "filing_id": filing_id, "biz_year": biz_year, "reprt_code": reprt_code, "fs_div": fs_div}
            )
        if not identities:
            raise PITDataError("DART XBRL facts require filing identities; certification blocked")
        if self._client is None and self._request_json is None:
            raise PITDataError("DART XBRL facts endpoint is not configured")
        pages: list[dict[str, Any]] = []
        for identity in identities:
            fid = identity["filing_id"]
            if self._request_json is not None:
                payload = self._request_json("fnlttSinglAcntAll", dict(identity))
                if not isinstance(payload, dict) or not payload:
                    raise PITDataError(f"missing XBRL facts for {fid}; certification failure")
                status = str(payload.get("status") or "").strip()
                if status == "014" or status == "013":
                    raise PITDataError(f"missing XBRL facts for {fid}; full-statement response is empty")
                facts = payload.get("list", payload.get("records", payload))
                if not facts:
                    raise PITDataError(f"missing XBRL facts for {fid}; certification failure")
                pages.append({"records": facts if isinstance(facts, list) else [facts], **identity, "raw_provenance": dict(payload)})
                continue
            assert self._client is not None
            try:
                raw = self._client._request_validated(
                    "fnlttSinglAcntAll.json",
                    {"corp_code": identity["corp_code"], "bsns_year": identity["biz_year"], "reprt_code": identity["reprt_code"], "fs_div": identity["fs_div"]},
                )
            except Exception as exc:
                raise PITDataError(f"missing XBRL facts for {fid}; certification failure") from exc
            status = str(raw.get("status") or "").strip()
            if status == "014":
                raise PITDataError(f"missing XBRL facts for {fid}; full-statement response is empty")
            facts = raw.get("list", raw)
            if not facts:
                raise PITDataError(f"missing XBRL facts for {fid}; certification failure")
            pages.append({"records": facts if isinstance(facts, list) else [facts], **identity, "raw_provenance": dict(raw)})
        if not pages:
            raise PITDataError("DART XBRL facts response is empty; certification blocked")
        return tuple(pages)
