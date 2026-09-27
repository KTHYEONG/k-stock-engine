"""Catalog-only disclosure readers: latest-wins, coverage keys, parity, conflicts."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.pit import PITDataError

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2024, 6, 1, tzinfo=UTC)


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _catalog(runtime):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def _publish(runtime, *, source: str, key: str, records: list, as_of: date, retrieved_at: datetime = RETRIEVED_AT):  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    if source == "dart_corp_disclosures":
        body = {"corp_code": key.split(":")[0], "start": key.split(":")[1].split("..")[0], "end": key.split(":")[1].split("..")[1], "records": records}
    else:
        detail, window = key.split(":", 1)
        start, end = window.split("..")
        body = {"detail_type": detail, "start": start, "end": end, "records": records}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    store = BronzeStore(runtime.workspace.bronze_root)
    receipt = store.import_bytes(raw, kind=EvidenceKind.DISCLOSURES, retrieved_at=retrieved_at, source_label="test")
    catalog = _catalog(runtime)
    catalog.publish(
        [ReceiptIndexEntry(source=source, natural_key=key, as_of=as_of, fiscal_period=None, status=EvidenceStatus.SUCCESS if records else EvidenceStatus.EMPTY, content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
        blobs=[BlobEntry(content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES, source=source, usable=True, unusable_reason=None, retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
    )
    return receipt


def _record(rcept_no: str, corp: str = "00126380", rcept_dt: str = "20160516", report_nm: str = "분기보고서 (2016.03)") -> dict:
    return {"corp_code": corp, "rcept_no": rcept_no, "rcept_dt": rcept_dt, "report_nm": report_nm, "rm": ""}


def test_uncatalogued_page_ignored(tmp_path: Path) -> None:
    import hashlib

    from src.data.dart_disclosures import iter_disclosure_records

    runtime = _runtime(tmp_path)
    _publish(runtime, source="dart_disclosure_windows", key="A:2016-01-01..2016-03-31", records=[_record("20160516000001")], as_of=date(2016, 3, 31))
    stray = {"detail_type": "A", "start": "2016-01-01", "end": "2016-03-31", "records": [_record("99999999999999")]}
    raw = json.dumps(stray, sort_keys=True, ensure_ascii=False).encode()
    stray_path = runtime.workspace.bronze_root / "disclosures" / hashlib.sha256(raw).hexdigest() / "payload.json"
    stray_path.parent.mkdir(parents=True, exist_ok=True)
    stray_path.write_bytes(raw)

    assert {row.rcept_no for row in iter_disclosure_records(_catalog(runtime))} == {"20160516000001"}


def test_superseded_window_ignored(tmp_path: Path) -> None:
    from src.data.dart_disclosures import iter_disclosure_records

    runtime = _runtime(tmp_path)
    _publish(runtime, source="dart_disclosure_windows", key="A:2016-01-01..2016-03-31", records=[_record("20160516000001")], as_of=date(2016, 3, 31), retrieved_at=datetime(2024, 6, 1, tzinfo=UTC))
    _publish(runtime, source="dart_disclosure_windows", key="A:2016-01-01..2016-03-31", records=[_record("20160516000001"), _record("20160516000002")], as_of=date(2016, 3, 31), retrieved_at=datetime(2024, 6, 2, tzinfo=UTC))

    assert sorted(row.rcept_no for row in iter_disclosure_records(_catalog(runtime))) == ["20160516000001", "20160516000002"]


def test_identity_selection_parity(tmp_path: Path) -> None:
    from src.data.dart_disclosures import iter_disclosure_records, periodic_filing_identities

    runtime = _runtime(tmp_path)
    window_rows = [
        {"corp_code": "00126380", "rcept_no": "20160516000001", "rcept_dt": "20160516", "report_nm": "분기보고서 (2016.03)", "rm": ""},
        {"corp_code": "00126380", "rcept_no": "20160816000002", "rcept_dt": "20160816", "report_nm": "반기보고서 (2016.06)", "rm": ""},
        {"corp_code": "00126380", "rcept_no": "bad-row", "rcept_dt": "baddate1", "report_nm": "분기보고서 (2016.03)", "rm": ""},
        {"corp_code": "", "rcept_no": "20160816000003", "rcept_dt": "20160816", "report_nm": "반기보고서 (2016.06)", "rm": ""},
    ]
    per_corp_rows = [
        {"corp_code": "00126380", "rcept_no": "20161115000003", "rcept_dt": "20161115", "report_nm": "분기보고서 (2016.09)", "rm": ""},
        {"corp_code": "00126380", "rcept_no": "20170331000004", "rcept_dt": "20170331", "report_nm": "사업보고서 (2016.12)", "rm": ""},
        {"corp_code": "00126380", "rcept_no": "20170331000005", "rcept_dt": "20170331", "report_nm": "감사보고서 (2016.12)", "rm": ""},
    ]
    _publish(runtime, source="dart_disclosure_windows", key="A:2016-01-01..2016-12-31", records=window_rows, as_of=date(2016, 12, 31))
    _publish(runtime, source="dart_corp_disclosures", key="00126380:2016-01-01..2016-12-31", records=per_corp_rows, as_of=date(2016, 12, 31))

    records = list(iter_disclosure_records(_catalog(runtime)))
    assert len(records) == 5
    identities = periodic_filing_identities(
        records, start=date(2016, 1, 1), end=date(2017, 12, 31),
        ticker_by_corp_code={"00126380": "005930"}, required_periods=None, corp_codes=None,
    )
    golden = {
        ("11013", "2016Q1", "20160516000001"),
        ("11012", "2016Q2", "20160816000002"),
        ("11014", "2016Q3", "20161115000003"),
        ("11011", "2016Q4", "20170331000004"),
    }
    assert {(item["reprt_code"], item["fiscal_period"], item["filing_id"]) for item in identities} == golden
    assert all(item["ticker"] == "005930" and item["fs_div"] == "CFS" for item in identities)


def test_conflicting_duplicate_fails(tmp_path: Path) -> None:
    from src.data.dart_disclosures import iter_disclosure_records

    runtime = _runtime(tmp_path)
    _publish(runtime, source="dart_disclosure_windows", key="A:2016-01-01..2016-03-31", records=[_record("20160516000001", report_nm="분기보고서 (2016.03)")], as_of=date(2016, 3, 31))
    _publish(runtime, source="dart_corp_disclosures", key="00126380:2016-01-01..2016-03-31", records=[_record("20160516000001", report_nm="반기보고서 (2016.06)")], as_of=date(2016, 3, 31))

    with pytest.raises(PITDataError, match="conflicting disclosure rows"):
        list(iter_disclosure_records(_catalog(runtime)))


def test_disclosure_payload_failures_raise_and_malformed_keys_skipped(tmp_path: Path) -> None:
    from src.core.pit import EvidenceKind
    from src.data.bronze import BronzeStore
    from src.data.dart_disclosures import iter_disclosure_records, per_corp_coverage
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptIndexEntry

    runtime = _runtime(tmp_path)
    catalog = _catalog(runtime)
    store = BronzeStore(runtime.workspace.bronze_root)
    moment = datetime(2024, 6, 1, tzinfo=UTC)

    assert list(iter_disclosure_records(catalog)) == []
    assert per_corp_coverage(catalog) == {}

    receipt = store.import_bytes(b'{"records": []}', kind=EvidenceKind.DISCLOSURES, retrieved_at=moment, source_label="test")
    catalog.publish(
        [ReceiptIndexEntry(source="dart_disclosure_windows", natural_key="A:2016-01-01..2016-03-31", as_of=date(2016, 3, 31), fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=receipt.content_hash, retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
        blobs=[BlobEntry(content_hash=receipt.content_hash, kind=EvidenceKind.DISCLOSURES, source="dart_disclosure_windows", usable=True, unusable_reason=None, retrieved_at=receipt.retrieved_at, payload_path=receipt.payload_path)],
    )
    (runtime.workspace.bronze_root / "disclosures").mkdir(parents=True, exist_ok=True)
    Path(receipt.payload_path).write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        list(iter_disclosure_records(catalog))

    catalog2_runtime = _runtime(tmp_path / "missing")
    catalog2 = _catalog(catalog2_runtime)
    store2 = BronzeStore(catalog2_runtime.workspace.bronze_root)
    receipt2 = store2.import_bytes(b'{"records": []}', kind=EvidenceKind.DISCLOSURES, retrieved_at=moment, source_label="test")
    catalog2.publish(
        [ReceiptIndexEntry(source="dart_disclosure_windows", natural_key="A:2016-01-01..2016-03-31", as_of=date(2016, 3, 31), fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=receipt2.content_hash, retrieved_at=receipt2.retrieved_at, payload_path=receipt2.payload_path)],
        blobs=[BlobEntry(content_hash=receipt2.content_hash, kind=EvidenceKind.DISCLOSURES, source="dart_disclosure_windows", usable=True, unusable_reason=None, retrieved_at=receipt2.retrieved_at, payload_path=receipt2.payload_path)],
    )
    Path(receipt2.payload_path).unlink()
    with pytest.raises(PITDataError, match="missing"):
        list(iter_disclosure_records(catalog2))

    catalog3_runtime = _runtime(tmp_path / "unreadable")
    catalog3 = _catalog(catalog3_runtime)
    store3 = BronzeStore(catalog3_runtime.workspace.bronze_root)
    receipt3 = store3.import_bytes(b"not-json", kind=EvidenceKind.DISCLOSURES, retrieved_at=moment, source_label="test")
    catalog3.publish(
        [ReceiptIndexEntry(source="dart_disclosure_windows", natural_key="A:2016-01-01..2016-03-31", as_of=date(2016, 3, 31), fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=receipt3.content_hash, retrieved_at=receipt3.retrieved_at, payload_path=receipt3.payload_path)],
        blobs=[BlobEntry(content_hash=receipt3.content_hash, kind=EvidenceKind.DISCLOSURES, source="dart_disclosure_windows", usable=True, unusable_reason=None, retrieved_at=receipt3.retrieved_at, payload_path=receipt3.payload_path)],
    )
    with pytest.raises(PITDataError, match="unreadable"):
        list(iter_disclosure_records(catalog3))

    catalog4_runtime = _runtime(tmp_path / "norows")
    catalog4 = _catalog(catalog4_runtime)
    store4 = BronzeStore(catalog4_runtime.workspace.bronze_root)
    receipt4 = store4.import_bytes(b'{"records": {}}', kind=EvidenceKind.DISCLOSURES, retrieved_at=moment, source_label="test")
    catalog4.publish(
        [ReceiptIndexEntry(source="dart_disclosure_windows", natural_key="A:2016-01-01..2016-03-31", as_of=date(2016, 3, 31), fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=receipt4.content_hash, retrieved_at=receipt4.retrieved_at, payload_path=receipt4.payload_path)],
        blobs=[BlobEntry(content_hash=receipt4.content_hash, kind=EvidenceKind.DISCLOSURES, source="dart_disclosure_windows", usable=True, unusable_reason=None, retrieved_at=receipt4.retrieved_at, payload_path=receipt4.payload_path)],
    )
    with pytest.raises(PITDataError, match="no records"):
        list(iter_disclosure_records(catalog4))

    assert per_corp_coverage(catalog4) == {}
    store5 = BronzeStore(catalog4_runtime.workspace.bronze_root)
    for bad_key in ("no-separator", "corp:not-a-date..2019-06-30", "corp:2019-06-30..2015-01-01", ":2015-01-01..2019-06-30"):
        raw = json.dumps({"corp_code": "x", "start": "2015-01-01", "end": "2019-06-30", "records": []}).encode()
        receipt5 = store5.import_bytes(raw, kind=EvidenceKind.DISCLOSURES, retrieved_at=moment, source_label="test")
        catalog4.publish(
            [ReceiptIndexEntry(source="dart_corp_disclosures", natural_key=bad_key, as_of=date(2019, 6, 30), fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=receipt5.content_hash, retrieved_at=receipt5.retrieved_at, payload_path=receipt5.payload_path)],
            blobs=[BlobEntry(content_hash=receipt5.content_hash, kind=EvidenceKind.DISCLOSURES, source="dart_corp_disclosures", usable=True, unusable_reason=None, retrieved_at=receipt5.retrieved_at, payload_path=receipt5.payload_path)],
        )
    assert per_corp_coverage(catalog4) == {}


def test_invalid_receipt_day_rows_skipped(tmp_path: Path) -> None:
    from src.data.dart_disclosures import iter_disclosure_records

    runtime = _runtime(tmp_path)
    _publish(
        runtime, source="dart_disclosure_windows", key="A:2016-01-01..2016-03-31",
        records=[
            {"corp_code": "00126380", "rcept_no": "bad1", "rcept_dt": "20161345", "report_nm": "분기보고서 (2016.03)"},
            {"corp_code": "00126380", "rcept_no": "bad2", "rcept_dt": "not-a-date", "report_nm": "분기보고서 (2016.03)"},
            "not-a-record",
        ],
        as_of=date(2016, 3, 31),
    )

    assert list(iter_disclosure_records(_catalog(runtime))) == []


def test_periodic_identity_selection_filters(tmp_path: Path) -> None:
    from src.data.dart_disclosures import DisclosureRecord, periodic_filing_identities

    _ = tmp_path
    records = [
        DisclosureRecord(corp_code="00126380", rcept_no="out-of-range", rcept_dt=date(2015, 12, 31), report_nm="분기보고서 (2015.03)"),
        DisclosureRecord(corp_code="00126380", rcept_no="no-period", rcept_dt=date(2016, 5, 16), report_nm="현금배당결정"),
        DisclosureRecord(corp_code="99999999", rcept_no="unmapped", rcept_dt=date(2016, 5, 16), report_nm="분기보고서 (2016.03)"),
        DisclosureRecord(corp_code="00126380", rcept_no="good", rcept_dt=date(2016, 5, 16), report_nm="분기보고서 (2016.03)"),
    ]
    identities = periodic_filing_identities(
        records, start=date(2016, 1, 1), end=date(2016, 12, 31),
        ticker_by_corp_code={"00126380": "005930"}, required_periods=None, corp_codes=frozenset({"00126380", "99999999"}),
    )
    assert [item["filing_id"] for item in identities] == ["good"]
