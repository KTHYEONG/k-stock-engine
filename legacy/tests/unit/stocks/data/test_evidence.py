"""Tests for immutable stock evidence artifact readers."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from legacy.stocks.data.evidence import (
    AvailabilityPolicy,
    DisclosureAvailabilityRecord,
    feature_availability_from_disclosures,
    load_corporate_action_snapshot,
    load_instrument_master_snapshot,
    load_krx_calendar_snapshot,
)


def write_json(tmp_path, name: str, payload: dict) -> None:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")


def base_payload() -> dict:
    return {
        "version": "master-v1",
        "generated_time": "2026-01-01T00:00:00+00:00",
        "records": [
            {
                "source_identifier": "000050",
                "instrument_id": "KRX:000050",
                "asset_type": "common_stock",
                "is_common_stock": True,
                "listed_from": "2024-01-01",
                "tradable_from": "2024-01-01",
                "available_time": "2026-01-01T00:00:00+00:00",
            }
        ],
    }


def calendar_payload() -> dict:
    return {
        "version": "calendar-v1",
        "generated_time": "2026-01-01T00:00:00+00:00",
        "sessions": ["2024-01-02", "2024-01-03", "2024-01-04"],
    }


def test_load_instrument_master_snapshot_rejects_overlapping_intervals(tmp_path) -> None:
    payload = base_payload()
    payload["records"].append(
        {
            "source_identifier": "000050",
            "instrument_id": "KRX:000050",
            "asset_type": "preferred_stock",
            "is_common_stock": False,
            "listed_from": "2024-01-02",
        }
    )
    write_json(tmp_path, "master.json", payload)

    with pytest.raises(ValueError, match="overlapping master intervals"):
        load_instrument_master_snapshot(tmp_path / "master.json")


def test_load_corporate_action_snapshot_requires_calendar_sessions(tmp_path) -> None:
    write_json(tmp_path, "calendar.json", calendar_payload())
    calendar = load_krx_calendar_snapshot(tmp_path / "calendar.json")
    write_json(
        tmp_path,
        "actions.json",
        {
            "version": "actions-v1",
            "generated_time": "2026-01-01T00:00:00+00:00",
            "intervals": [
                {
                    "instrument_id": "KRX:000050",
                    "previous_session": "2024-01-03",
                    "session": "2024-01-05",
                    "action_code": "no_action",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="non-calendar session"):
        load_corporate_action_snapshot(tmp_path / "actions.json", calendar)


def test_date_only_disclosure_uses_next_session_policy() -> None:
    from legacy.stocks.data.quality import KRXSessionCalendar

    calendar = KRXSessionCalendar(
        version="calendar-v1",
        sessions=(date(2024, 1, 2), date(2024, 1, 3)),
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    available = AvailabilityPolicy().available_time(date(2024, 1, 2), calendar)
    assert available.isoformat() == "2024-01-03T00:00:00+00:00"

    records = feature_availability_from_disclosures(
        (
            DisclosureAvailabilityRecord(
                feature_name="feature__revenue",
                source_field="revenue",
                source_version="dart-v1",
                source_hash="hash",
                receipt_date=date(2024, 1, 2),
                receipt_number="20240102000001",
            ),
        ),
        AvailabilityPolicy(calendar=calendar),
    )
    assert records[0].use_class == "research"
    assert records[0].available_time == available
