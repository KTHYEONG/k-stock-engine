"""Deterministic readers for external stock-data evidence artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from itertools import pairwise
from pathlib import Path
from typing import Any

from legacy.stocks.data.quality import (
    CorporateActionInterval,
    CorporateActionSnapshot,
    FeatureAvailabilityRecord,
    InstrumentMasterRecord,
    InstrumentMasterSnapshot,
    KRXSessionCalendar,
)


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"evidence artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evidence artifact must be a JSON object: {path}")
    return payload


def _required(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"evidence artifact missing {key!r}")
    return value


def _aware_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def _date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO date") from exc


def _bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a JSON boolean")


def load_instrument_master_snapshot(path: Path) -> InstrumentMasterSnapshot:
    """Load and validate an effective-dated instrument master artifact."""
    payload = _read_object(path)
    raw_records = _required(payload, "records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("instrument master records must be a non-empty list")

    records: list[InstrumentMasterRecord] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("instrument master record must be an object")
        records.append(
            InstrumentMasterRecord(
                source_identifier=str(_required(raw, "source_identifier")),
                instrument_id=str(_required(raw, "instrument_id")),
                asset_type=str(_required(raw, "asset_type")),
                is_common_stock=_bool(_required(raw, "is_common_stock"), "is_common_stock"),
                listed_from=_date(_required(raw, "listed_from"), "listed_from"),
                delisted_on=(
                    _date(raw["delisted_on"], "delisted_on")
                    if raw.get("delisted_on")
                    else None
                ),
                tradable_from=(
                    _date(raw["tradable_from"], "tradable_from")
                    if raw.get("tradable_from")
                    else None
                ),
                tradable_to=(
                    _date(raw["tradable_to"], "tradable_to")
                    if raw.get("tradable_to")
                    else None
                ),
                available_time=(
                    _aware_datetime(raw["available_time"], "available_time")
                    if raw.get("available_time")
                    else None
                ),
            )
        )

    by_identifier: dict[str, list[InstrumentMasterRecord]] = {}
    for record in records:
        by_identifier.setdefault(record.source_identifier, []).append(record)
    for identifier, intervals in by_identifier.items():
        ordered = sorted(intervals, key=lambda item: item.listed_from)
        for previous, current in pairwise(ordered):
            previous_end = previous.delisted_on or date.max
            if current.listed_from <= previous_end:
                raise ValueError(f"overlapping master intervals for {identifier}")

    return InstrumentMasterSnapshot(
        version=str(_required(payload, "version")),
        records=tuple(records),
        generated_time=_aware_datetime(_required(payload, "generated_time"), "generated_time"),
    )


def load_krx_calendar_snapshot(path: Path) -> KRXSessionCalendar:
    """Load a sorted KRX session calendar artifact."""
    payload = _read_object(path)
    raw_sessions = _required(payload, "sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError("calendar sessions must be a list")
    sessions = tuple(_date(value, "session") for value in raw_sessions)
    return KRXSessionCalendar(
        version=str(_required(payload, "version")),
        sessions=sessions,
        generated_time=_aware_datetime(_required(payload, "generated_time"), "generated_time"),
    )


def load_corporate_action_snapshot(
    path: Path, calendar: KRXSessionCalendar
) -> CorporateActionSnapshot:
    """Load action/no-action intervals and validate their calendar membership."""
    payload = _read_object(path)
    raw_intervals = _required(payload, "intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError("corporate action intervals must be a non-empty list")
    intervals: list[CorporateActionInterval] = []
    for raw in raw_intervals:
        if not isinstance(raw, dict):
            raise ValueError("corporate action interval must be an object")
        interval = CorporateActionInterval(
            instrument_id=str(_required(raw, "instrument_id")),
            previous_session=_date(_required(raw, "previous_session"), "previous_session"),
            session=_date(_required(raw, "session"), "session"),
            action_code=str(_required(raw, "action_code")),
            adjustment_factor=float(raw.get("adjustment_factor", 1.0)),
        )
        if not calendar.is_session(interval.previous_session) or not calendar.is_session(interval.session):
            raise ValueError("corporate action interval references a non-calendar session")
        intervals.append(interval)
    return CorporateActionSnapshot(
        version=str(_required(payload, "version")),
        intervals=tuple(intervals),
        generated_time=_aware_datetime(_required(payload, "generated_time"), "generated_time"),
    )


@dataclass(frozen=True, slots=True)
class AvailabilityPolicy:
    """Conservative availability policy for date-only disclosure sources."""

    name: str = "date-only-next-krx-session"
    next_session_time: time = time(9, 0)
    calendar: KRXSessionCalendar | None = None

    def available_time(self, receipt_date: date, calendar: KRXSessionCalendar | None = None) -> datetime:
        """Return the first verified session after a date-only receipt."""
        calendar = calendar or self.calendar
        if calendar is None:
            raise ValueError("date-only availability policy requires a KRX calendar")
        future_sessions = [session for session in calendar.sessions if session > receipt_date]
        if not future_sessions:
            raise ValueError(f"no next KRX session after {receipt_date.isoformat()}")
        from zoneinfo import ZoneInfo

        return datetime.combine(
            future_sessions[0], self.next_session_time, tzinfo=ZoneInfo("Asia/Seoul")
        ).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DisclosureAvailabilityRecord:
    """One source disclosure record used to derive feature availability evidence."""

    feature_name: str
    source_field: str
    source_version: str
    source_hash: str
    receipt_date: date
    receipt_number: str
    null_rate: float = 0.0


def load_disclosure_availability_records(path: Path) -> tuple[DisclosureAvailabilityRecord, ...]:
    """Load date-only disclosure records used by the availability policy."""
    payload = _read_object(path)
    raw_records = _required(payload, "records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("disclosure records must be a non-empty list")
    records: list[DisclosureAvailabilityRecord] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("disclosure availability record must be an object")
        records.append(
            DisclosureAvailabilityRecord(
                feature_name=str(_required(raw, "feature_name")),
                source_field=str(_required(raw, "source_field")),
                source_version=str(_required(raw, "source_version")),
                source_hash=str(_required(raw, "source_hash")),
                receipt_date=_date(_required(raw, "receipt_date"), "receipt_date"),
                receipt_number=str(_required(raw, "receipt_number")),
                null_rate=float(raw.get("null_rate", 0.0)),
            )
        )
    return tuple(records)


def feature_availability_from_disclosures(
    records: tuple[DisclosureAvailabilityRecord, ...],
    decision_policy: AvailabilityPolicy,
) -> tuple[FeatureAvailabilityRecord, ...]:
    """Convert receipt-date records into deterministic field-level evidence."""
    if not records:
        raise ValueError("disclosure availability records must not be empty")
    by_feature: dict[str, DisclosureAvailabilityRecord] = {}
    for record in records:
        if record.feature_name in by_feature:
            raise ValueError(f"duplicate disclosure availability for {record.feature_name}")
        if not record.receipt_number:
            raise ValueError("disclosure receipt_number must be non-empty")
        by_feature[record.feature_name] = record
    return tuple(
        FeatureAvailabilityRecord(
            feature_name=record.feature_name,
            source_field=record.source_field,
            availability_rule=f"{decision_policy.name}:receipt={record.receipt_date.isoformat()}",
            source_version=record.source_version,
            source_hash=record.source_hash,
            null_rate=record.null_rate,
            use_class="research",
            available_time=decision_policy.available_time(record.receipt_date),
        )
        for record in sorted(by_feature.values(), key=lambda item: item.feature_name)
    )
