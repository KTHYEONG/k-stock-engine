"""Dividend-event v2 publication and source-integrity tests."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime
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
      <tr><td>보통주 1주당 배당금</td><td>100</td></tr>
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


def _decision(
    *,
    rcept_no: str,
    record: date,
    received: date,
    dps: int,
    correction: bool = False,
    kind: str | None = "결산배당",
    total: int | None = None,
    market_yield: object = None,
    pay: date | None = None,
    corp: str = "00126380",
) -> object:
    from datetime import date as _date
    from decimal import Decimal

    from src.integrations.dart.dividend_decision import DividendDecision

    return DividendDecision(
        rcept_no=rcept_no,
        corp_code=corp,
        received_on=received,
        record_date=record,
        pay_date=_date(2024, 6, 3) if pay is None else pay,
        dps_common_krw=dps,
        is_correction=correction,
        dividend_kind=kind,
        total_krw=total,
        market_yield_pct=None if market_yield is None else Decimal(str(market_yield)),
    )


def test_resolve_dividend_decisions_correction_moving_record_date_replaces_original() -> None:
    from datetime import date

    from src.data.dividend_events import resolve_dividend_decisions

    original = _decision(rcept_no="20240000000001", record=date(2024, 12, 31), received=date(2025, 1, 2), dps=500)
    correction = _decision(
        rcept_no="20250000000002", record=date(2025, 1, 1), received=date(2025, 1, 9), dps=500, correction=True,
    )

    kept, superseded = resolve_dividend_decisions(
        [correction, original], correction_window_days=45
    )

    assert [item.rcept_no for item in kept] == ["20250000000002"]
    assert [item.rcept_no for item in superseded] == ["20240000000001"]


def test_resolve_dividend_decisions_correction_outside_window_is_separate_dividend() -> None:
    from datetime import date

    from src.data.dividend_events import resolve_dividend_decisions

    original = _decision(rcept_no="20240000000001", record=date(2024, 1, 5), received=date(2024, 1, 6), dps=500)
    correction = _decision(
        rcept_no="20240000000002", record=date(2024, 4, 5), received=date(2024, 4, 6), dps=500, correction=True,
    )

    kept, superseded = resolve_dividend_decisions([original, correction], correction_window_days=45)

    assert [item.rcept_no for item in kept] == ["20240000000001", "20240000000002"]
    assert superseded == ()


def test_resolve_dividend_decisions_different_kinds_never_merge() -> None:
    from datetime import date

    from src.data.dividend_events import resolve_dividend_decisions

    quarterly = _decision(
        rcept_no="20240000000001", record=date(2024, 6, 30), received=date(2024, 7, 1), dps=200,
        kind="분기배당",
    )
    year_end = _decision(
        rcept_no="20240000000002", record=date(2024, 7, 20), received=date(2024, 7, 21), dps=200,
        kind="결산배당", correction=True,
    )

    kept, superseded = resolve_dividend_decisions([quarterly, year_end], correction_window_days=45)

    assert [item.rcept_no for item in kept] == ["20240000000001", "20240000000002"]
    assert superseded == ()


def test_resolve_dividend_decisions_correction_ignores_other_companies() -> None:
    from datetime import date

    from src.data.dividend_events import resolve_dividend_decisions

    other = _decision(
        rcept_no="20240000000009", record=date(2024, 12, 31), received=date(2025, 1, 2), dps=500,
        corp="00999999",
    )
    original = _decision(rcept_no="20240000000001", record=date(2024, 12, 31), received=date(2025, 1, 3), dps=500)
    correction = _decision(
        rcept_no="20250000000002", record=date(2025, 1, 1), received=date(2025, 1, 9), dps=500, correction=True,
    )

    kept, superseded = resolve_dividend_decisions(
        [correction, original, other], correction_window_days=45
    )

    assert [item.rcept_no for item in kept] == ["20250000000002", "20240000000009"]
    assert [item.rcept_no for item in superseded] == ["20240000000001"]


def test_resolve_dividend_decisions_correction_ignores_same_receipt_decision() -> None:
    from datetime import date

    from src.data.dividend_events import resolve_dividend_decisions

    first = _decision(rcept_no="20240000000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=100)
    correction = _decision(
        rcept_no="20240000000001", record=date(2024, 1, 6), received=date(2024, 1, 2), dps=100,
        correction=True,
    )

    kept, superseded = resolve_dividend_decisions([correction, first], correction_window_days=45)

    assert [item.rcept_no for item in kept] == ["20240000000001", "20240000000001"]
    assert [(item.record_date, item.is_correction) for item in kept] == [
        (date(2024, 1, 5), False),
        (date(2024, 1, 6), True),
    ]
    assert superseded == ()


def test_resolve_dividend_decisions_latest_receipt_wins_shared_record_date() -> None:
    from datetime import date

    from src.data.dividend_events import resolve_dividend_decisions

    early = _decision(rcept_no="20240000000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=100)
    late = _decision(rcept_no="20240000000002", record=date(2024, 1, 5), received=date(2024, 1, 3), dps=120)
    correction = _decision(
        rcept_no="20240000000003", record=date(2024, 1, 5), received=date(2024, 1, 4), dps=120,
        correction=True,
    )

    kept, superseded = resolve_dividend_decisions([late, early, correction], correction_window_days=45)

    assert [item.rcept_no for item in kept] == ["20240000000003"]
    assert [item.rcept_no for item in superseded] == ["20240000000001", "20240000000002"]


def test_check_dividend_plausibility_withholds_implausible_decisions() -> None:
    from datetime import date

    from src.config.providers import DividendPlausibilityPolicy
    from src.data.dividend_events import check_dividend_plausibility

    policy = DividendPlausibilityPolicy()

    total_as_dps = _decision(
        rcept_no="r1", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=38807917000,
    )
    assert (
        check_dividend_plausibility(total_as_dps, close_before_ex=4920, listed_shares=None, policy=policy)
        == "yield_above_ceiling"
    )

    mismatch = _decision(
        rcept_no="r2", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1000, market_yield="1.0",
    )
    assert (
        check_dividend_plausibility(mismatch, close_before_ex=10000, listed_shares=None, policy=policy)
        == "yield_mismatch"
    )

    # 총액이 상장주식의 절반도 지급하지 못하면 DPS를 잘못 읽은 것이다.
    exceeds = _decision(
        rcept_no="r3", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1000, total=1000 * 40,
    )
    assert (
        check_dividend_plausibility(exceeds, close_before_ex=50000, listed_shares=100, policy=policy)
        == "dps_exceeds_total"
    )
    # 자사주 몫만큼 총액이 작은 것은 정상이다.
    treasury = _decision(
        rcept_no="r3b", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1000, total=1000 * 80,
    )
    assert check_dividend_plausibility(treasury, close_before_ex=50000, listed_shares=100, policy=policy) is None
    # 공시 시가배당율이 DPS를 확인하면 차등배당으로 총액이 작아도 지급한다.
    differential = _decision(
        rcept_no="r3c", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1000, total=1000 * 10,
        market_yield="2.0",
    )
    assert (
        check_dividend_plausibility(differential, close_before_ex=50000, listed_shares=100, policy=policy)
        is None
    )
    # 실제 공시값과 몇 배 어긋나면 격리한다(011040: DPS 3원, 공시 4.9%).
    far_below = _decision(
        rcept_no="r3d", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=3, market_yield="4.9",
    )
    assert (
        check_dividend_plausibility(far_below, close_before_ex=6090, listed_shares=None, policy=policy)
        == "yield_mismatch"
    )

    untraded = _decision(rcept_no="r4", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=100)
    assert (
        check_dividend_plausibility(untraded, close_before_ex=None, listed_shares=None, policy=policy)
        == "no_reference_price"
    )
    assert (
        check_dividend_plausibility(untraded, close_before_ex=0, listed_shares=None, policy=policy)
        == "no_reference_price"
    )


def test_check_dividend_plausibility_pays_matching_decision() -> None:
    from datetime import date

    from src.config.providers import DividendPlausibilityPolicy
    from src.data.dividend_events import check_dividend_plausibility

    policy = DividendPlausibilityPolicy()
    plausible = _decision(
        rcept_no="r5", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1400,
        total=1400 * 6000000000, market_yield="2.8",
    )

    assert (
        check_dividend_plausibility(
            plausible, close_before_ex=50000, listed_shares=6000000000, policy=policy
        )
        is None
    )


def _materialize_with_market(
    tmp_path: Path, monkeypatch, decisions: list, market: dict, policy=None,
):  # type: ignore[no-untyped-def]
    import src.data.dividend_events as module

    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    monkeypatch.setattr(module, "_decision_from_envelope", lambda _payload: decisions.pop(0))
    monkeypatch.setattr(module, "_load_market_reference", lambda _path: market)
    silver = tmp_path / "silver-gated"
    path = module.materialize_dividend_events(
        catalog=_catalog_for(bronze),
        silver_root=silver,
        calendar=_calendar(),
        daily_market_path=tmp_path / "daily_market_fixture",
        policy=policy,
    )
    return path


def test_materialize_dividend_events_quarantines_total_as_dps(tmp_path: Path, monkeypatch) -> None:
    import json as _json
    from datetime import date

    from src.config.providers import DividendPlausibilityPolicy
    from src.data.datasets import load_manifest, read_dataset

    decision = _decision(
        rcept_no="20240102000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=38807917000,
    )
    path = _materialize_with_market(
        tmp_path, monkeypatch, decisions=[decision],
        market={("KRX:005930", date(2024, 1, 2)): (4920, None)},
        policy=DividendPlausibilityPolicy(),
    )

    assert read_dataset(path).collect().height == 0
    manifest = load_manifest(path)
    assert manifest.details["quarantined_rows"] == 1
    assert manifest.details["quarantined_by_reason"] == {"yield_above_ceiling": 1}
    quarantine = _json.loads((path / "quarantine.json").read_text(encoding="utf-8"))
    assert quarantine == [
        {
            "corp_code": "00126380",
            "rcept_no": "20240102000001",
            "reason": "yield_above_ceiling",
            "record_date": "2024-01-05",
        }
    ]


def test_materialize_dividend_events_pays_plausible_decision(tmp_path: Path, monkeypatch) -> None:
    import json as _json
    from datetime import date

    from src.config.providers import DividendPlausibilityPolicy
    from src.data.datasets import load_manifest, read_dataset

    decision = _decision(
        rcept_no="20240102000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1400,
        total=1400 * 6000000000, market_yield="2.8",
    )
    path = _materialize_with_market(
        tmp_path, monkeypatch, decisions=[decision],
        market={("KRX:005930", date(2024, 1, 2)): (50000, 6000000000)},
        policy=DividendPlausibilityPolicy(),
    )

    frame = read_dataset(path).collect()
    assert frame.height == 1
    assert frame["dps_krw"].to_list() == [1400]
    manifest = load_manifest(path)
    assert manifest.details["quarantined_rows"] == 0
    assert _json.loads((path / "quarantine.json").read_text(encoding="utf-8")) == []


def test_materialize_dividend_events_withholds_without_reference_price(tmp_path: Path, monkeypatch) -> None:
    from datetime import date

    from src.data.datasets import load_manifest, read_dataset

    decision = _decision(
        rcept_no="20240102000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=100,
    )
    path = _materialize_with_market(tmp_path, monkeypatch, decisions=[decision], market={})

    assert read_dataset(path).collect().height == 0
    assert load_manifest(path).details["quarantined_by_reason"] == {"no_reference_price": 1}


def test_materialize_dividend_events_withholds_when_ex_is_first_session(tmp_path: Path, monkeypatch) -> None:
    from datetime import date

    from src.data.datasets import load_manifest, read_dataset

    decision = _decision(
        rcept_no="20240102000001", record=date(2024, 1, 4), received=date(2024, 1, 2), dps=100,
    )
    path = _materialize_with_market(
        tmp_path, monkeypatch, decisions=[decision],
        market={("KRX:005930", date(2024, 1, 2)): (50000, 6000000000)},
    )

    assert read_dataset(path).collect().height == 0
    assert load_manifest(path).details["quarantined_by_reason"] == {"no_reference_price": 1}


def test_materialize_dividend_events_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    from datetime import date

    import src.data.dividend_events as module

    first_decision = _decision(
        rcept_no="20240102000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1400,
        total=1400 * 6000000000, market_yield="2.8",
    )
    second_decision = _decision(
        rcept_no="20240102000001", record=date(2024, 1, 5), received=date(2024, 1, 2), dps=1400,
        total=1400 * 6000000000, market_yield="2.8",
    )
    market = {("KRX:005930", date(2024, 1, 2)): (50000, 6000000000)}
    bronze = tmp_path / "bronze"
    _write_sources(bronze)
    monkeypatch.setattr(module, "_load_market_reference", lambda _path: market)
    monkeypatch.setattr(module, "_decision_from_envelope", lambda _payload: first_decision)
    first = module.materialize_dividend_events(
        catalog=_catalog_for(bronze),
        silver_root=tmp_path / "silver-first",
        calendar=_calendar(),
        daily_market_path=tmp_path / "daily_market_fixture",
    )
    monkeypatch.setattr(module, "_decision_from_envelope", lambda _payload: second_decision)
    second = module.materialize_dividend_events(
        catalog=_catalog_for(bronze),
        silver_root=tmp_path / "silver-second",
        calendar=_calendar(),
        daily_market_path=tmp_path / "daily_market_fixture",
    )

    assert first.name == second.name


def test_load_market_reference_reads_daily_market_dataset(tmp_path: Path) -> None:
    from datetime import date

    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.dividend_events import _load_market_reference

    assert _load_market_reference(None) is None

    frame = pl.DataFrame(
        {
            "session": [date(2024, 1, 2)],
            "instrument_id": ["KRX:005930"],
            "close": [50000],
            "listed_shares": [6000000000],
        },
        schema={"session": pl.Date, "instrument_id": pl.String, "close": pl.Int64, "listed_shares": pl.Int64},
    )
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity(
            kind="daily_market", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": frame},
    )

    assert _load_market_reference(published.path) == {("KRX:005930", date(2024, 1, 2)): (50000, 6000000000)}


def test_load_market_reference_rejects_bad_rows(tmp_path: Path) -> None:
    import polars as pl
    import pytest

    from src.core.pit import PITDataError
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.dividend_events import _load_market_reference

    frame = pl.DataFrame(
        {
            "session": ["not-a-date"],
            "instrument_id": ["KRX:005930"],
            "close": [50000],
            "listed_shares": [6000000000],
        },
        schema={
            "session": pl.String, "instrument_id": pl.String, "close": pl.Int64, "listed_shares": pl.Int64,
        },
    )
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity(
            kind="daily_market", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": frame},
    )

    with pytest.raises(PITDataError):
        _load_market_reference(published.path)


def test_load_market_reference_rejects_bad_amounts(tmp_path: Path) -> None:
    from datetime import date

    import polars as pl
    import pytest

    from src.core.pit import PITDataError
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.dividend_events import _load_market_reference

    frame = pl.DataFrame(
        {
            "session": [date(2024, 1, 2)],
            "instrument_id": ["KRX:005930"],
            "close": ["not-a-number"],
            "listed_shares": [6000000000],
        },
        schema={
            "session": pl.Date, "instrument_id": pl.String, "close": pl.String, "listed_shares": pl.Int64,
        },
    )
    published = publish_dataset(
        layer_root=tmp_path / "silver",
        identity=DatasetIdentity(
            kind="daily_market", layer=DatasetLayer.SILVER, policy_version="test-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": frame},
    )

    with pytest.raises(PITDataError):
        _load_market_reference(published.path)


def test_decision_from_envelope_passes_report_title(tmp_path: Path) -> None:
    import base64 as _base64
    import io as _io
    import zipfile as _zip

    from src.data.dividend_events import _decision_from_envelope

    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as archive:
        archive.writestr(
            "document.xml",
            "<table><tr><td>보통주 1주당 배당금</td><td>100</td></tr>"
            "<tr><td>배당기준일</td><td>2024-01-05</td></tr>"
            "<tr><td>배당금지급예정일</td><td>2024-06-03</td></tr></table>",
        )
    envelope = {
        "rcept_no": "20240102000001",
        "corp_code": "00126380",
        "received_on": "2024-01-02",
        "archive_b64": _base64.b64encode(buf.getvalue()).decode(),
        "report_nm": "[기재정정]현금ㆍ현물배당결정",
    }

    decision = _decision_from_envelope(envelope)

    assert decision.is_correction is True
    assert decision.dps_common_krw == 100
