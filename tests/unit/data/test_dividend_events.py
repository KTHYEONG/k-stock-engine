"""Dividend-event v2 publication and source-integrity tests."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from src.core.time import KRX_TZ, SessionCalendar
from src.data.datasets import load_manifest, read_dataset, verify_dataset
from src.data.dividend_events import materialize_dividend_events


def _calendar() -> SessionCalendar:
    days = (
        datetime(2024, 1, 2, 9, tzinfo=KRX_TZ),
        datetime(2024, 1, 4, 9, tzinfo=KRX_TZ),
        datetime(2024, 1, 5, 9, tzinfo=KRX_TZ),
        datetime(2024, 6, 3, 9, tzinfo=KRX_TZ),
    )
    return SessionCalendar(days)


def _archive() -> bytes:
    html = """<table>
      <tr><td>배당기준일</td><td>2024-01-05</td></tr>
      <tr><td>배당금지급예정일</td><td>2024-06-03</td></tr>
      <tr><td>배당구분</td><td>결산배당</td></tr>
      <tr><td>보통주</td><td>100</td></tr>
    </table>"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("document.xml", html)
    return output.getvalue()


def _write_sources(bronze: Path) -> None:
    bridge = json.dumps([{"corp_code": "00126380", "ticker": "005930"}], ensure_ascii=False, sort_keys=True).encode()
    bridge_hash = hashlib.sha256(bridge).hexdigest()
    bridge_path = bronze / "dart_corp_codes" / bridge_hash / "payload.json"
    bridge_path.parent.mkdir(parents=True)
    bridge_path.write_bytes(bridge)

    envelope = {
        "corp_code": "00126380",
        "rcept_no": "20240102000001",
        "received_on": "2024-01-02",
        "archive_b64": base64.b64encode(_archive()).decode(),
    }
    raw = json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode()
    payload = bronze / "corporate_actions" / hashlib.sha256(raw).hexdigest() / "payload.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(raw)


def test_dividend_events_publish_v2_and_rebuild_noop(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    kwargs = {
        "bronze_root": bronze,
        "universe_root": tmp_path / "silver",
        "silver_root": tmp_path / "silver",
        "calendar": _calendar(),
    }
    first = materialize_dividend_events(**kwargs)
    before = (first / "manifest.json").read_bytes()
    second = materialize_dividend_events(**kwargs)

    assert first == second
    assert (first / "manifest.json").read_bytes() == before
    manifest = load_manifest(first)
    assert manifest.kind == "dividend_events"
    assert manifest.layer.value == "silver"
    assert verify_dataset(first, known_ids=lambda _dataset_id: True).passed
    frame = read_dataset(first).collect()
    assert frame.height == 1
    assert frame["dps_krw"].to_list() == [100]
    assert frame["instrument_id"].to_list() == ["KRX:005930"]


def test_dividend_events_reject_tampered_envelope(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError

    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    payload = next((bronze / "corporate_actions").glob("*/payload.json"))
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(PITDataError, match="hash mismatch"):
        materialize_dividend_events(
            bronze_root=bronze,
            universe_root=tmp_path / "silver",
            silver_root=tmp_path / "silver",
            calendar=_calendar(),
        )


def test_dividend_bridge_and_envelope_boundaries_fail_closed(tmp_path: Path, monkeypatch) -> None:
    import base64
    import hashlib
    import json

    import pytest

    from src.core.pit import PITDataError
    from src.data.dividend_events import (
        _decision_from_envelope,
        _iter_decision_envelopes,
        _load_corp_bridge,
    )

    original_read_bytes = Path.read_bytes
    bridge_root = tmp_path / "bridge"
    raw_bridge = json.dumps([{"corp_code": "1", "ticker": "005930"}]).encode()
    bridge_path = bridge_root / "dart_corp_codes" / ("a" * 64) / "payload.json"
    bridge_path.parent.mkdir(parents=True)
    bridge_path.write_bytes(raw_bridge)

    def fail_bridge_read(path: Path) -> bytes:
        if path == bridge_path:
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_bridge_read)
    with pytest.raises(PITDataError, match="bridge payload"):
        _load_corp_bridge(bridge_root)
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    mismatch_root = tmp_path / "mismatch"
    mismatch_path = mismatch_root / "dart_corp_codes" / ("b" * 64) / "payload.json"
    mismatch_path.parent.mkdir(parents=True)
    mismatch_path.write_bytes(raw_bridge)
    with pytest.raises(PITDataError, match="hash mismatch"):
        _load_corp_bridge(mismatch_root)

    invalid_root = tmp_path / "invalid-bridge"
    invalid_raw = b"not-json"
    invalid_path = invalid_root / "dart_corp_codes" / hashlib.sha256(invalid_raw).hexdigest() / "payload.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_bytes(invalid_raw)
    with pytest.raises(PITDataError, match="invalid dart corp-code"):
        _load_corp_bridge(invalid_root)

    object_root = tmp_path / "object-bridge"
    object_raw = b"{}"
    object_path = object_root / "dart_corp_codes" / hashlib.sha256(object_raw).hexdigest() / "payload.json"
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(object_raw)
    with pytest.raises(PITDataError, match="invalid dart corp-code"):
        _load_corp_bridge(object_root)

    corporate = tmp_path / "corporate"
    corporate_actions = corporate / "corporate_actions"
    invalid_b64 = {"rcept_no": "r1", "archive_b64": "%%%"}
    invalid_b64_raw = json.dumps(invalid_b64).encode()
    invalid_b64_path = corporate_actions / hashlib.sha256(invalid_b64_raw).hexdigest() / "payload.json"
    invalid_b64_path.parent.mkdir(parents=True)
    invalid_b64_path.write_bytes(invalid_b64_raw)
    assert _iter_decision_envelopes(corporate) == []

    status_payload = {
        "rcept_no": "r2",
        "archive_b64": base64.b64encode(b"<status>014</status>").decode(),
    }
    status_raw = json.dumps(status_payload).encode()
    status_path = corporate_actions / hashlib.sha256(status_raw).hexdigest() / "payload.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_bytes(status_raw)
    assert _iter_decision_envelopes(corporate) == []

    with pytest.raises(PITDataError, match="invalid dividend-decision"):
        _decision_from_envelope({"rcept_no": "r3", "archive_b64": "%%%"})

    unreadable_root = tmp_path / "unreadable"
    unreadable_actions = unreadable_root / "corporate_actions"
    unreadable_raw = b"{}"
    unreadable_path = unreadable_actions / hashlib.sha256(unreadable_raw).hexdigest() / "payload.json"
    unreadable_path.parent.mkdir(parents=True)
    unreadable_path.write_bytes(unreadable_raw)

    def fail_envelope_read(path: Path) -> bytes:
        if str(path).startswith(str(unreadable_root)):
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_envelope_read)
    with pytest.raises(PITDataError, match="Bronze payload is unreadable"):
        _iter_decision_envelopes(unreadable_root)


def test_dividend_undated_and_estimated_decisions_are_recorded(tmp_path: Path, monkeypatch) -> None:
    from datetime import date

    from src.data.datasets import load_manifest
    from src.data.dividend_events import materialize_dividend_events
    from src.integrations.dart.dividend_decision import DividendDecision, UndecidedRecordDateError

    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    import src.data.dividend_events as module

    def undated(_payload):
        raise UndecidedRecordDateError("undated")

    monkeypatch.setattr(module, "_decision_from_envelope", undated)
    path = materialize_dividend_events(
        bronze_root=bronze,
        universe_root=tmp_path / "silver",
        silver_root=tmp_path / "silver",
        calendar=_calendar(),
    )
    assert load_manifest(path).details["undated_record_rows"] == 1

    decision = DividendDecision(
        rcept_no="r-estimate",
        corp_code="00126380",
        received_on=date(2024, 1, 2),
        record_date=date(2024, 1, 5),
        pay_date=None,
        dps_common_krw=100,
        is_correction=False,
        dividend_kind="중간배당",
    )
    monkeypatch.setattr(module, "_decision_from_envelope", lambda _payload: decision)
    path = materialize_dividend_events(
        bronze_root=bronze,
        universe_root=tmp_path / "silver-estimate",
        silver_root=tmp_path / "silver-estimate",
        calendar=_calendar(),
    )
    assert load_manifest(path).details["estimated_pay_rows"] == 1
