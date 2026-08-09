"""External KRX and OpenDART collectors for stock evidence artifacts."""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from xml.etree import ElementTree

import requests

from src.stocks.data.quality import (
    InstrumentMasterRecord,
    InstrumentMasterSnapshot,
    KRXSessionCalendar,
)

JsonRequest = Callable[[str, dict[str, str]], dict[str, Any]]


class EvidenceCollectionError(RuntimeError):
    """Raised when an external evidence response cannot be trusted."""


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_date(value: object, field: str) -> date:
    text = str(value or "").strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise EvidenceCollectionError(f"invalid {field}: {value!r}")
    return datetime.strptime(text, "%Y%m%d").date()


def _text(record: dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip() not in ("", "-"):
            return str(value).strip()
    return ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class KRXEvidenceCollector:
    """Collect KRX security-master snapshots and session-calendar evidence."""

    BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
    ENDPOINTS: ClassVar[dict[str, str]] = {
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
        generated_time: datetime | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("KRX_OPENAPI_KEY")
        if not self.api_key and request_json is None:
            raise ValueError("KRX_OPENAPI_KEY not found in environment variables")
        self.generated_time = generated_time or _now_utc()
        self._request_json = request_json or self._request
        self._session = requests.Session()
        self._last_request_time = 0.0
        self._request_count = 0

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        if self._request_count >= 10_000:
            raise EvidenceCollectionError("KRX daily request limit reached")
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
            raise EvidenceCollectionError(f"KRX HTTP {response.status_code} for {endpoint}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise EvidenceCollectionError(f"KRX returned invalid JSON for {endpoint}") from exc
        if not isinstance(payload, dict):
            raise EvidenceCollectionError(f"KRX response must be an object for {endpoint}")
        return payload

    def _records(self, endpoint: str, as_of: date) -> list[dict[str, Any]]:
        payload = self._request_json(endpoint, {"basDd": as_of.strftime("%Y%m%d")})
        records = payload.get("OutBlock_1", [])
        if not isinstance(records, list):
            raise EvidenceCollectionError(f"KRX records must be a list for {endpoint}")
        return [record for record in records if isinstance(record, dict)]

    def collect_master_snapshot(self, as_of: date, market: str = "ALL") -> InstrumentMasterSnapshot:
        """Collect one dated KRX master snapshot without ticker-based inference."""
        raw: list[dict[str, Any]] = []
        if market in ("ALL", "KOSPI"):
            raw.extend(self._records(self.ENDPOINTS["KOSPI_INFO"], as_of))
        if market in ("ALL", "KOSDAQ"):
            raw.extend(self._records(self.ENDPOINTS["KOSDAQ_INFO"], as_of))
        if not raw:
            raise EvidenceCollectionError(f"KRX returned no master records for {as_of.isoformat()}")

        records: dict[str, InstrumentMasterRecord] = {}
        for item in raw:
            symbol = _text(item, "ISU_SRT_CD", "ISU_CD")
            full_code = _text(item, "ISU_CD")
            stock_type = _text(item, "KIND_STKCERT_TP_NM", "SECUGRP_NM")
            listing = _parse_date(_text(item, "LIST_DD"), "LIST_DD")
            if not symbol or not full_code or not stock_type:
                raise EvidenceCollectionError("KRX master record lacks identity or security type")
            is_common = stock_type == "보통주"
            asset_type = "common_stock" if is_common else f"krx:{stock_type}"
            records[symbol] = InstrumentMasterRecord(
                source_identifier=symbol,
                instrument_id=f"KRX:{symbol}",
                asset_type=asset_type,
                is_common_stock=is_common,
                listed_from=listing,
                available_time=self.generated_time,
            )
        return InstrumentMasterSnapshot(
            version=f"krx-master-{as_of.isoformat()}",
            records=tuple(records[key] for key in sorted(records)),
            generated_time=self.generated_time,
        )

    def collect_session_calendar(self, start: date, end: date) -> KRXSessionCalendar:
        """Discover sessions from both KRX daily-trade endpoints.

        An empty response is treated as a non-session; transport/API errors
        raise instead of silently creating a false holiday.
        """
        if start > end:
            raise ValueError("start must not be after end")
        sessions: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                kospi = self._records(self.ENDPOINTS["KOSPI_TRADE"], current)
                kosdaq = self._records(self.ENDPOINTS["KOSDAQ_TRADE"], current)
                if kospi or kosdaq:
                    sessions.append(current)
            current += timedelta(days=1)
        if not sessions:
            raise EvidenceCollectionError("KRX returned no sessions in requested range")
        return KRXSessionCalendar(
            version=f"krx-calendar-{start.isoformat()}-{end.isoformat()}",
            sessions=tuple(sessions),
            generated_time=self.generated_time,
        )

    def write_master_snapshot(self, path: Path, snapshot: InstrumentMasterSnapshot) -> None:
        """Persist a master snapshot in the artifact-loader JSON format."""
        _write_json(
            path,
            {
                "version": snapshot.version,
                "generated_time": snapshot.generated_time.isoformat(),
                "records": [
                    {
                        "source_identifier": record.source_identifier,
                        "instrument_id": record.instrument_id,
                        "asset_type": record.asset_type,
                        "is_common_stock": record.is_common_stock,
                        "listed_from": record.listed_from.isoformat(),
                        "delisted_on": record.delisted_on.isoformat() if record.delisted_on else None,
                        "tradable_from": record.tradable_from.isoformat() if record.tradable_from else None,
                        "tradable_to": record.tradable_to.isoformat() if record.tradable_to else None,
                        "available_time": record.available_time.isoformat() if record.available_time else None,
                    }
                    for record in snapshot.records
                ],
            },
        )

    def write_calendar(self, path: Path, calendar: KRXSessionCalendar) -> None:
        """Persist a KRX calendar in the artifact-loader JSON format."""
        _write_json(
            path,
            {
                "version": calendar.version,
                "generated_time": calendar.generated_time.isoformat(),
                "sessions": [session.isoformat() for session in calendar.sessions],
            },
        )


class OpenDartEvidenceCollector:
    """Collect immutable OpenDART disclosure and corporate-action candidates."""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_json: JsonRequest | None = None,
        generated_time: datetime | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENDART_API_KEY")
        if not self.api_key and request_json is None:
            raise ValueError("OPENDART_API_KEY not found in environment variables")
        self.generated_time = generated_time or _now_utc()
        self._request_json = request_json or self._request
        self._session = requests.Session()

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        response = self._session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=30)
        if response.status_code != 200:
            raise EvidenceCollectionError(f"DART HTTP {response.status_code} for {endpoint}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise EvidenceCollectionError(f"DART returned invalid JSON for {endpoint}") from exc
        if not isinstance(payload, dict) or payload.get("status") != "000":
            raise EvidenceCollectionError(f"DART rejected request for {endpoint}: {payload}")
        return payload

    def load_corp_codes(self) -> dict[str, str]:
        """Download the DART corporation-code map keyed by listed ticker."""
        response = requests.get(
            f"{self.BASE_URL}/corpCode.xml",
            params={"crtfc_key": self.api_key},
            timeout=30,
        )
        if response.status_code != 200:
            raise EvidenceCollectionError(f"DART corpCode HTTP {response.status_code}")
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                xml_bytes = archive.read("CORPCODE.xml")
            if b"<!DOCTYPE" in xml_bytes or b"<!ENTITY" in xml_bytes:
                raise EvidenceCollectionError("DART corpCode.xml contains unsafe XML declarations")
            root = ElementTree.fromstring(xml_bytes)  # noqa: S314
        except (OSError, KeyError, ElementTree.ParseError) as exc:
            raise EvidenceCollectionError("DART corpCode.xml could not be parsed") from exc
        mapping: dict[str, str] = {}
        for item in root.findall("list"):
            ticker = (item.findtext("stock_code") or "").strip()
            corp_code = (item.findtext("corp_code") or "").strip()
            if ticker and corp_code:
                mapping[ticker] = corp_code
        if not mapping:
            raise EvidenceCollectionError("DART corpCode.xml contained no listed tickers")
        return mapping

    def collect_disclosures(
        self,
        start: date,
        end: date,
        *,
        corp_code: str | None = None,
        page_count: int = 100,
    ) -> list[dict[str, str]]:
        """Collect DART filing records retaining receipt date and receipt number."""
        if start > end:
            raise ValueError("start must not be after end")
        if not 1 <= page_count <= 100:
            raise ValueError("page_count must be within [1, 100]")
        records: list[dict[str, str]] = []
        page = 1
        while True:
            params = {
                "crtfc_key": str(self.api_key),
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_no": str(page),
                "page_count": str(page_count),
            }
            if corp_code:
                params["corp_code"] = corp_code
            payload = self._request_json("list.json", params)
            raw = payload.get("list", [])
            if not isinstance(raw, list):
                raise EvidenceCollectionError("DART disclosure list must be a list")
            for item in raw:
                if not isinstance(item, dict):
                    continue
                receipt_number = _text(item, "rcept_no")
                receipt_date = _text(item, "rcept_dt")
                if not receipt_number or not receipt_date:
                    raise EvidenceCollectionError("DART disclosure lacks receipt identity")
                records.append(
                    {
                        "rcept_no": receipt_number,
                        "rcept_dt": receipt_date,
                        "corp_code": _text(item, "corp_code"),
                        "corp_name": _text(item, "corp_name"),
                        "report_nm": _text(item, "report_nm"),
                        "rm": _text(item, "rm"),
                    }
                )
            total_page = int(payload.get("total_page", page))
            if page >= total_page or not raw:
                break
            page += 1
        return sorted(records, key=lambda item: (item["rcept_dt"], item["rcept_no"]))

    def write_disclosure_artifact(self, path: Path, records: Iterable[dict[str, str]]) -> None:
        """Persist raw disclosure index records with retrieval provenance."""
        _write_json(
            path,
            {
                "version": f"dart-disclosures-{self.generated_time.date().isoformat()}",
                "generated_time": self.generated_time.isoformat(),
                "records": list(records),
            },
        )

    def collect_corporate_action_candidates(
        self, start: date, end: date, *, corp_code: str | None = None
    ) -> list[dict[str, str]]:
        """Collect disclosure candidates without inventing an adjustment factor."""
        keywords = ("분할", "합병", "배당", "증자", "감자", "권리", "주식병합", "액면")
        records = self.collect_disclosures(start, end, corp_code=corp_code)
        return [
            {
                **record,
                "candidate_kind": "corporate_action_disclosure",
                "verification_status": "candidate_only",
            }
            for record in records
            if any(keyword in record["report_nm"] for keyword in keywords)
        ]

    def write_corporate_action_candidates(
        self, path: Path, records: Iterable[dict[str, str]]
    ) -> None:
        """Persist unverified corporate-action candidates for reconciliation."""
        _write_json(
            path,
            {
                "version": f"dart-action-candidates-{self.generated_time.date().isoformat()}",
                "generated_time": self.generated_time.isoformat(),
                "records": list(records),
                "coverage_status": "candidate_only",
                "promotion_rule": "Requires verified effective date and adjustment factor from KRX or licensed corporate-action source.",
            },
        )
