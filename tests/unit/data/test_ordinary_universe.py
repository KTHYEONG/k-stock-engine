from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.data.bronze import BronzeStore
from src.data.ordinary_universe import (
    build_ordinary_universe,
    dated_master_receipts,
    materialize_ordinary_universe_from_bronze,
    ordinary_universe_snapshot,
    write_ordinary_universe_silver,
)
from src.data.schemas import EvidenceKind, PITDataError
from src.data.streaming_normalization import _canonical_master_row


def _master(store: BronzeStore, day: str, records: list[dict[str, str]]):
    return store.import_bytes(
        json.dumps({"as_of": day, "records": records}, ensure_ascii=False).encode(),
        kind=EvidenceKind.SECURITY_MASTER,
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        source_label=f"KRX:historical-master:{day}",
    )


def _row(ticker: str, isin: str, **changes: str) -> dict[str, str]:
    return {
        "ISU_SRT_CD": ticker,
        "ISU_CD": isin,
        "KIND_STKCERT_TP_NM": "보통주",
        "SECUGRP_NM": "주권",
        "MKT_TP_NM": "KOSPI",
        "LIST_DD": "20100101",
        **changes,
    }


def test_dated_snapshot_excludes_non_ordinary_and_preserves_source(tmp_path) -> None:
    store = BronzeStore(tmp_path / "bronze")
    day = date(2018, 1, 2)
    rows = [
        _row("A", "KR0000000001"),
        _row("B", "KR0000000002", KIND_STKCERT_TP_NM="구형우선주"),
        _row("C", "KR0000000003", SECUGRP_NM="부동산투자회사"),
        _row("D", "KR0000000004", SECT_TP_NM="SPAC(소속부없음)"),
        _row("E", "KR0000000005", LIST_DD="20190101"),
        _row("F", "KR0000000006", KIND_STKCERT_TP_NM=""),
    ]
    receipt = _master(store, day.isoformat(), rows)
    snapshot = build_ordinary_universe((receipt,), sessions=(day,))[0]
    assert snapshot.eligible_tickers == {"A"}
    assert snapshot.rows["exclusion_reason"].to_list() == [
        "eligible", "non_ordinary_share", "non_equity_security_group", "spac", "not_listed_yet", "unknown_share_kind",
    ]
    assert snapshot.rows["source_hash"].unique().to_list() == [receipt.content_hash]
    assert snapshot.rows["available_at"].item(0).hour == 15
    out = write_ordinary_universe_silver((snapshot,), root=tmp_path / "silver")
    saved = pl.read_parquet(out / f"session={day.isoformat()}" / "part.parquet")
    assert saved.filter(pl.col("eligible"))["ticker"].to_list() == ["A"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["partitions"][0]["source_hash"] == receipt.content_hash
    with pytest.raises(PITDataError, match="already exists"):
        write_ordinary_universe_silver((snapshot,), root=tmp_path / "silver")


def test_missing_or_conflicting_daily_master_fails_closed(tmp_path) -> None:
    store = BronzeStore(tmp_path / "bronze")
    first = _master(store, "2018-01-02", [_row("A", "KR0000000001")])
    with pytest.raises(PITDataError, match="missing security_master snapshot"):
        build_ordinary_universe((first,), sessions=(date(2018, 1, 2), date(2018, 1, 3)))
    changed = _master(store, "2018-01-02", [_row("A", "KR0000000001", LIST_DD="20100102")])
    with pytest.raises(PITDataError, match="ambiguous security_master snapshot"):
        build_ordinary_universe((first, changed), sessions=(date(2018, 1, 2),))
    duplicate = _master(store, "2018-01-03", [_row("A", "KR0000000001"), _row("A", "KR0000000001")])
    with pytest.raises(PITDataError, match="duplicate security_master ticker"):
        ordinary_universe_snapshot(duplicate)
    first.payload_path.write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="hash mismatch"):
        ordinary_universe_snapshot(first)


def test_source_validation_and_metadata_selection(tmp_path) -> None:
    store = BronzeStore(tmp_path / "bronze")
    row = _row("A", "KR0000000001")
    receipt = _master(store, "2018-01-02", [row])
    selected = dated_master_receipts(tmp_path / "bronze", sessions=(date(2018, 1, 2),))
    assert selected == (receipt,)
    assert dated_master_receipts(tmp_path / "bronze", sessions=(date(2018, 1, 3),)) == ()

    for altered, expected in (
        (replace(receipt, kind=EvidenceKind.DAILY_MARKET), "expected security_master"),
        (replace(receipt, source_path="KRX:historical-master:2018-01-03"), "date conflicts"),
    ):
        with pytest.raises(PITDataError, match=expected):
            ordinary_universe_snapshot(altered)

    for payload, expected in (
        (b"{", "JSON"),
        (b"[]", "root"),
        (b'{"as_of":"bad","records":[]}', "invalid KRX master"),
        (b'{"as_of":"2018-01-02","session":"2018-01-03","records":[]}', "unambiguous"),
        (b'{"as_of":"2018-01-02","records":{}}', "records must be a list"),
        (b'{"as_of":"2018-01-02","records":[]}', "empty security_master"),
        (b'{"as_of":"2018-01-02","records":[null]}', "record must be an object"),
        (b'{"as_of":"2018-01-02","records":[{}]}', "lacks ticker or ISIN"),
        (json.dumps({"as_of": "2018-01-02", "records": [{**row, "LIST_DD": "bad"}]}).encode(), "invalid listing date"),
        (json.dumps({"as_of": "2018-01-02", "records": [row, _row("B", "KR0000000001")]}).encode(), "conflicting security_master ISIN"),
    ):
        bad = store.import_bytes(payload, kind=EvidenceKind.SECURITY_MASTER, retrieved_at=datetime(2026, 9, 20, tzinfo=UTC), source_label="invalid")
        with pytest.raises(PITDataError, match=expected):
            ordinary_universe_snapshot(bad)

    with pytest.raises(PITDataError, match="requires requested sessions"):
        build_ordinary_universe((receipt,), sessions=())
    outside = _master(store, "2018-01-03", [row])
    assert build_ordinary_universe((outside, receipt), sessions=(date(2018, 1, 2),))[0].session == date(2018, 1, 2)

    metadata = receipt.metadata_path
    original = metadata.read_text()
    metadata.write_text("{" , encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid security_master receipt"):
        dated_master_receipts(tmp_path / "bronze", sessions=(date(2018, 1, 2),))
    metadata.write_text(original, encoding="utf-8")
    for field, value, expected in (
        ("kind", "daily_market", "kind mismatch"),
        ("content_hash", "wrong", "path/hash mismatch"),
    ):
        changed = json.loads(original)
        changed[field] = value
        metadata.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(PITDataError, match=expected):
            dated_master_receipts(tmp_path / "bronze", sessions=(date(2018, 1, 2),))
    metadata.write_text(original, encoding="utf-8")


def test_unknown_classifications_remain_ineligible() -> None:
    from src.data.ordinary_universe import classify_krx_master_row

    base = _row("A", "KR0000000001")
    assert classify_krx_master_row({**base, "SECUGRP_NM": ""}) == (False, "unknown_security_group")
    assert classify_krx_master_row({**base, "MKT_TP_NM": ""}) == (False, "unknown_market")
    assert classify_krx_master_row({**base, "MKT_TP_NM": "KONEX"}) == (False, "excluded_market")


def test_silver_write_rejects_duplicate_sessions_and_removes_failed_staging(tmp_path, monkeypatch) -> None:
    store = BronzeStore(tmp_path / "bronze")
    snapshot = ordinary_universe_snapshot(_master(store, "2018-01-02", [_row("A", "KR0000000001")]))
    with pytest.raises(PITDataError, match="unique ascending dates"):
        write_ordinary_universe_silver((snapshot, snapshot), root=tmp_path / "silver")
    with pytest.raises(PITDataError, match="requires dated snapshots"):
        write_ordinary_universe_silver((), root=tmp_path / "silver")

    def fail_write(self, path):
        raise OSError("disk unavailable")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        write_ordinary_universe_silver((snapshot,), root=tmp_path / "silver")
    assert not list((tmp_path / "silver").glob(".ordinary-universe-*"))


def test_full_materializer_requires_exact_days_and_streams_partitions(tmp_path) -> None:
    store = BronzeStore(tmp_path / "bronze")
    first = date(2018, 1, 2)
    second = date(2018, 1, 3)
    _master(store, first.isoformat(), [_row("A", "KR0000000001")])
    with pytest.raises(PITDataError, match="requires requested sessions"):
        materialize_ordinary_universe_from_bronze(
            bronze_root=tmp_path / "bronze", sessions=(), silver_root=tmp_path / "silver"
        )
    with pytest.raises(PITDataError, match="missing security_master snapshot"):
        materialize_ordinary_universe_from_bronze(
            bronze_root=tmp_path / "bronze", sessions=(first, second), silver_root=tmp_path / "silver"
        )
    _master(store, second.isoformat(), [_row("B", "KR0000000002")])
    output = materialize_ordinary_universe_from_bronze(
        bronze_root=tmp_path / "bronze", sessions=(second, first), silver_root=tmp_path / "silver"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert [part["session"] for part in manifest["partitions"]] == [first.isoformat(), second.isoformat()]
    assert not list((tmp_path / "silver").glob(".ordinary-universe-*"))
    _master(store, second.isoformat(), [_row("C", "KR0000000003")])
    with pytest.raises(PITDataError, match="ambiguous security_master snapshot"):
        materialize_ordinary_universe_from_bronze(
            bronze_root=tmp_path / "bronze", sessions=(first, second), silver_root=tmp_path / "silver-extra"
        )


def test_streaming_silver_mapping_does_not_infer_common_from_missing_fields() -> None:
    moment = datetime(2018, 1, 2, tzinfo=UTC)
    base = {"ticker": "A", "market": "KOSPI", "share_class": "common"}
    row = _canonical_master_row(base, available_at=moment, source_hash="a" * 64, fallback_session=moment)
    assert row["share_class"] == "other"
    assert row["ordinary_equity_exclusion_reason"] == "unknown_share_kind"
    assert row["source_hash"] == "a" * 64
    approved = _canonical_master_row(
        {**base, **_row("A", "KR0000000001")}, available_at=moment,
        source_hash="b" * 64, fallback_session=moment,
    )
    assert approved["share_class"] == "common"
    assert approved["ordinary_equity_eligible"] is True
