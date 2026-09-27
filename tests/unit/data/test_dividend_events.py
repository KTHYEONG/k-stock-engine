"""Dividend-event v2 publication and source-integrity tests."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.pit import PITDataError
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


def _catalog_for(bronze: Path):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(bronze / "catalog")


def _publish_blob(bronze: Path, *, kind_dir: str, raw: bytes, kind, source: str, receipt_key: str | None):  # type: ignore[no-untyped-def]
    from datetime import date as _date

    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    store = BronzeStore(bronze)
    receipt = store.import_bytes(
        raw, kind=kind, retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), source_label="test",
    )
    catalog = _catalog_for(bronze)
    entries = []
    if receipt_key is not None:
        entries.append(
            ReceiptIndexEntry(
                source=source, natural_key=receipt_key, as_of=_date(2024, 1, 2),
                fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=receipt.content_hash,
                retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path,
            )
        )
    catalog.publish(
        entries,
        blobs=[
            BlobEntry(
                content_hash=receipt.content_hash, kind=kind, source=source,
                usable=True, unusable_reason=None, retrieved_at=receipt.retrieved_at,
                payload_path=receipt.payload_path,
            )
        ],
    )
    return receipt


def _write_sources(bronze: Path) -> None:
    from src.core.pit import EvidenceKind

    bridge = json.dumps([{"corp_code": "00126380", "ticker": "005930"}], ensure_ascii=False, sort_keys=True).encode()
    _publish_blob(
        bronze, kind_dir="dart_corp_codes", raw=bridge, kind=EvidenceKind.SECURITY_MASTER,
        source="dart_corp_codes", receipt_key="dart_corp_codes",
    )
    envelope = {
        "corp_code": "00126380",
        "rcept_no": "20240102000001",
        "received_on": "2024-01-02",
        "archive_b64": base64.b64encode(_archive()).decode(),
    }
    raw = json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode()
    _publish_blob(
        bronze, kind_dir="corporate_actions", raw=raw, kind=EvidenceKind.CORPORATE_ACTIONS,
        source="opendart:dividend_decision", receipt_key="20240102000001",
    )


def test_dividend_events_publish_v2_and_rebuild_noop(tmp_path: Path) -> None:
    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    catalog = _catalog_for(bronze)
    kwargs = {
        "catalog": catalog,
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
    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    payload = next((bronze / "corporate_actions").glob("*/payload.json"))
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(PITDataError, match="hash mismatch"):
        materialize_dividend_events(
            catalog=_catalog_for(bronze),
            silver_root=tmp_path / "silver",
            calendar=_calendar(),
        )


def test_dividend_events_only_catalogued_envelopes(tmp_path: Path) -> None:
    from src.data.dividend_events import _iter_decision_envelopes

    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    stray = {
        "corp_code": "00126380",
        "rcept_no": "99999999999999",
        "received_on": "2024-01-02",
        "archive_b64": base64.b64encode(_archive()).decode(),
    }
    raw = json.dumps(stray, sort_keys=True, ensure_ascii=False).encode()
    stray_path = bronze / "corporate_actions" / hashlib.sha256(raw).hexdigest() / "payload.json"
    stray_path.parent.mkdir(parents=True)
    stray_path.write_bytes(raw)

    assert [item["rcept_no"] for item in _iter_decision_envelopes(_catalog_for(bronze))] == ["20240102000001"]


def test_dividend_events_unusable_blob_skipped(tmp_path: Path) -> None:
    from src.data.dividend_events import _iter_decision_envelopes

    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    catalog = _catalog_for(bronze)
    blobs = list(catalog.blobs(source="opendart:dividend_decision", usable=True))
    assert len(blobs) == 1
    catalog.mark_unusable([blobs[0].content_hash], reason="quarantined")

    assert _iter_decision_envelopes(catalog) == []


def test_dividend_bridge_and_envelope_boundaries_fail_closed(tmp_path: Path, monkeypatch) -> None:
    from src.core.pit import EvidenceKind
    from src.data.dividend_events import (
        _decision_from_envelope,
        _iter_decision_envelopes,
        _load_corp_bridge,
    )

    original_read_bytes = Path.read_bytes
    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    catalog = _catalog_for(bronze)
    bridge_blobs = list(catalog.blobs(source="dart_corp_codes", usable=True))
    assert bridge_blobs
    bridge_path = Path(bridge_blobs[0].payload_path)

    def fail_bridge_read(path: Path) -> bytes:
        if Path(path) == bridge_path:
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_bridge_read)
    with pytest.raises(PITDataError, match="invalid dart corp-code bridge"):
        _load_corp_bridge(catalog)
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    Path(bridge_path).write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="invalid dart corp-code bridge"):
        _load_corp_bridge(catalog)

    invalid_root = tmp_path / "invalid-bridge"
    invalid_raw = b"not-json"
    _publish_blob(
        invalid_root / "bronze", kind_dir="dart_corp_codes", raw=invalid_raw,
        kind=EvidenceKind.SECURITY_MASTER, source="dart_corp_codes", receipt_key="dart_corp_codes",
    )
    with pytest.raises(PITDataError, match="invalid dart corp-code bridge"):
        _load_corp_bridge(_catalog_for(invalid_root / "bronze"))

    object_root = tmp_path / "object-bridge"
    _publish_blob(
        object_root / "bronze", kind_dir="dart_corp_codes", raw=b"{}",
        kind=EvidenceKind.SECURITY_MASTER, source="dart_corp_codes", receipt_key="dart_corp_codes",
    )
    with pytest.raises(PITDataError, match="invalid dart corp-code bridge"):
        _load_corp_bridge(_catalog_for(object_root / "bronze"))

    corporate = tmp_path / "corporate" / "bronze"
    invalid_b64 = {"rcept_no": "r1", "archive_b64": "%%%"}
    _publish_blob(
        corporate, kind_dir="corporate_actions", raw=json.dumps(invalid_b64).encode(),
        kind=EvidenceKind.CORPORATE_ACTIONS, source="opendart:dividend_decision", receipt_key="r1",
    )
    assert _iter_decision_envelopes(_catalog_for(corporate)) == []

    status_payload = {
        "rcept_no": "r2",
        "archive_b64": base64.b64encode(b"<status>014</status>").decode(),
    }
    _publish_blob(
        corporate, kind_dir="corporate_actions", raw=json.dumps(status_payload).encode(),
        kind=EvidenceKind.CORPORATE_ACTIONS, source="opendart:dividend_decision", receipt_key="r2",
    )
    assert _iter_decision_envelopes(_catalog_for(corporate)) == []

    with pytest.raises(PITDataError, match="invalid dividend-decision"):
        _decision_from_envelope({"rcept_no": "r3", "archive_b64": "%%%"})

    unreadable_bronze = tmp_path / "unreadable" / "bronze"
    _publish_blob(
        unreadable_bronze, kind_dir="corporate_actions", raw=b"{}",
        kind=EvidenceKind.CORPORATE_ACTIONS, source="opendart:dividend_decision", receipt_key="r9",
    )

    def fail_envelope_read(path: Path) -> bytes:
        if str(path).startswith(str(tmp_path / "unreadable")):
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_envelope_read)
    with pytest.raises(PITDataError, match="Bronze payload is unreadable"):
        _iter_decision_envelopes(_catalog_for(unreadable_bronze))


def test_dividend_undated_and_estimated_decisions_are_recorded(tmp_path: Path, monkeypatch) -> None:
    from datetime import date

    from src.data.datasets import load_manifest
    from src.integrations.dart.dividend_decision import DividendDecision, UndecidedRecordDateError

    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    import src.data.dividend_events as module

    catalog = _catalog_for(bronze)

    def undated(_payload):
        raise UndecidedRecordDateError("undated")

    monkeypatch.setattr(module, "_decision_from_envelope", undated)
    path = materialize_dividend_events(
        catalog=catalog,
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
        catalog=catalog,
        silver_root=tmp_path / "silver-estimate",
        calendar=_calendar(),
    )
    assert load_manifest(path).details["estimated_pay_rows"] == 1
