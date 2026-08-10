"""External KRX and OpenDART collectors for stock evidence artifacts."""
from __future__ import annotations

import hashlib
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
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a file through a sibling temporary file followed by atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid monthly calendar artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"monthly calendar artifact must be a JSON object: {path}")
    return payload


def _iter_month_partitions(
    start: date, end: date
) -> Iterable[tuple[int, int, date, date]]:
    """Yield (year, month, clipped month start, clipped month end) ascending."""
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
        month_end = min(end, next_month - timedelta(days=1))
        month_start = max(start, cursor)
        yield cursor.year, cursor.month, month_start, month_end
        cursor = next_month


class KRXEvidenceCollector:
    """Collect KRX security-master snapshots and session-calendar evidence."""

    BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
    ENDPOINTS: ClassVar[dict[str, str]] = {
        "KOSPI_INFO": "sto/stk_isu_base_info",
        "KOSDAQ_INFO": "sto/ksq_isu_base_info",
        "KOSPI_TRADE": "sto/stk_bydd_trd",
        "KOSDAQ_TRADE": "sto/ksq_bydd_trd",
    }
    MANIFEST_SCHEMA_VERSION = "krx-calendar-manifest-1"

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
        raise EvidenceCollectionError(last_error)

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

    def _month_artifact_path(self, output_dir: Path, year: int, month: int) -> Path:
        return output_dir / "months" / f"{year}-{month:02d}.json"

    def _month_artifact_is_complete(
        self,
        output_dir: Path,
        year: int,
        month: int,
        month_start: date,
        month_end: date,
        manifest_entry: dict[str, Any] | None,
    ) -> str | None:
        """Return None when the month artifact satisfies the completion predicate, else a reason."""
        path = self._month_artifact_path(output_dir, year, month)
        if not path.is_file():
            return f"month artifact missing: {path.name}"
        try:
            payload = _read_json_object(path)
        except ValueError as exc:
            return str(exc)
        month_key = f"{year}-{month:02d}"
        if payload.get("version") != f"krx-calendar-month-{month_key}":
            return "month artifact version mismatch"
        try:
            generated = datetime.fromisoformat(str(payload["generated_time"]).replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError):
            return "month artifact generated_time invalid"
        if generated.tzinfo is None:
            return "month artifact generated_time must be timezone-aware"
        try:
            declared_start = date.fromisoformat(str(payload["range_start"]))
            declared_end = date.fromisoformat(str(payload["range_end"]))
        except (KeyError, ValueError, TypeError):
            return "month artifact range invalid"
        if declared_start != month_start or declared_end != month_end:
            return "month artifact range mismatch"
        raw_sessions = payload.get("sessions")
        if not isinstance(raw_sessions, list) or not raw_sessions:
            return "month artifact sessions must be a non-empty list"
        try:
            sessions = tuple(date.fromisoformat(str(value)) for value in raw_sessions)
        except (TypeError, ValueError):
            return "month artifact contains an invalid session date"
        if any(day < declared_start or day > declared_end for day in sessions):
            return "month artifact session outside declared range"
        if payload.get("source_endpoints") != [
            self.ENDPOINTS["KOSPI_TRADE"],
            self.ENDPOINTS["KOSDAQ_TRADE"],
        ]:
            return "month artifact source_endpoints mismatch"
        try:
            KRXSessionCalendar(
                version=str(payload["version"]),
                sessions=sessions,
                generated_time=generated,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return f"month artifact sessions invalid: {exc}"
        if manifest_entry is None or manifest_entry.get("status") != "complete":
            return "manifest entry not complete"
        if manifest_entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            return "month artifact digest mismatch"
        return None

    def _load_or_init_manifest(self, output_dir: Path, start: date, end: date) -> dict[str, Any]:
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            try:
                payload = _read_json_object(manifest_path)
            except ValueError as exc:
                raise EvidenceCollectionError(
                    f"resumable manifest unreadable: {manifest_path}"
                ) from exc
            if payload.get("schema_version") != self.MANIFEST_SCHEMA_VERSION:
                raise ValueError(f"incompatible resumable manifest schema in {output_dir}")
            if (
                payload.get("requested_start") != start.isoformat()
                or payload.get("requested_end") != end.isoformat()
            ):
                raise ValueError(f"resumable manifest declares a different range in {output_dir}")
            payload.setdefault("months", {})
            return payload
        return {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "partition": "month",
            "months": {},
        }

    def _write_month_artifact(
        self,
        output_dir: Path,
        year: int,
        month: int,
        calendar: KRXSessionCalendar,
        month_start: date,
        month_end: date,
    ) -> str:
        """Atomically write one monthly partition and return its content SHA-256."""
        month_key = f"{year}-{month:02d}"
        path = self._month_artifact_path(output_dir, year, month)
        payload = {
            "version": f"krx-calendar-month-{month_key}",
            "generated_time": calendar.generated_time.isoformat(),
            "range_start": month_start.isoformat(),
            "range_end": month_end.isoformat(),
            "sessions": [session.isoformat() for session in calendar.sessions],
            "source_endpoints": [
                self.ENDPOINTS["KOSPI_TRADE"],
                self.ENDPOINTS["KOSDAQ_TRADE"],
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        _atomic_write_text(path, text)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def collect_calendar_partitions(self, output_dir: Path, start: date, end: date) -> None:
        """Collect a requested inclusive range as validated monthly partitions.

        Completed months are validated before reuse and never re-requested.
        A failing month is marked incomplete and re-raised; prior months are preserved.
        """
        if start > end:
            raise ValueError("start must not be after end")
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"resumable output target must be a directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_or_init_manifest(output_dir, start, end)
        for year, month, month_start, month_end in _iter_month_partitions(start, end):
            month_key = f"{year}-{month:02d}"
            entry = manifest["months"].get(month_key)
            if (
                isinstance(entry, dict)
                and self._month_artifact_is_complete(
                    output_dir, year, month, month_start, month_end, entry
                )
                is None
            ):
                continue
            try:
                calendar = self.collect_session_calendar(month_start, month_end)
                digest = self._write_month_artifact(
                    output_dir, year, month, calendar, month_start, month_end
                )
                manifest["months"][month_key] = {
                    "range_start": month_start.isoformat(),
                    "range_end": month_end.isoformat(),
                    "status": "complete",
                    "path": f"months/{month_key}.json",
                    "session_count": len(calendar.sessions),
                    "sha256": digest,
                }
            except Exception as exc:
                manifest["months"][month_key] = {
                    "range_start": month_start.isoformat(),
                    "range_end": month_end.isoformat(),
                    "status": "incomplete",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_json(output_dir / "manifest.json", manifest)
                raise EvidenceCollectionError(
                    f"KRX calendar month {month_key} collection failed"
                ) from exc
            _write_json(output_dir / "manifest.json", manifest)

    def merge_calendar_partitions(
        self, input_dir: Path, start: date, end: date, output_path: Path
    ) -> None:
        """Merge validated monthly partitions into one final calendar artifact."""
        if start > end:
            raise ValueError("start must not be after end")
        if not input_dir.is_dir():
            raise ValueError(f"resumable input must be an existing directory: {input_dir}")
        manifest = self._load_or_init_manifest(input_dir, start, end)
        merged: list[date] = []
        for year, month, month_start, month_end in _iter_month_partitions(start, end):
            month_key = f"{year}-{month:02d}"
            entry = manifest["months"].get(month_key)
            error = self._month_artifact_is_complete(
                input_dir, year, month, month_start, month_end, entry
            )
            if error is not None:
                raise EvidenceCollectionError(
                    f"KRX calendar merge missing valid month {month_key}: {error}"
                )
            payload = _read_json_object(self._month_artifact_path(input_dir, year, month))
            merged.extend(date.fromisoformat(value) for value in payload["sessions"])
        calendar = KRXSessionCalendar(
            version=f"krx-calendar-{start.isoformat()}-{end.isoformat()}",
            sessions=tuple(merged),
            generated_time=self.generated_time,
        )
        self.write_calendar(output_path, calendar)


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
