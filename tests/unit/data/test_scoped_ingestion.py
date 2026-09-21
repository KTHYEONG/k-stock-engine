from __future__ import annotations

import base64
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import load_data_runtime
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2024, 6, 1, tzinfo=UTC)


def _writer(tmp_path: Path) -> tuple[ScopedBronzeWriter, ReceiptCatalog]:
    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return ScopedBronzeWriter(runtime=runtime, catalog=catalog), catalog


def _payload(
    *,
    kind: EvidenceKind = EvidenceKind.DAILY_MARKET,
    source: str = "krx_daily_market",
    natural_key: str = "2024-01-02",
    as_of: date | None = date(2024, 1, 2),
    fiscal_period: str | None = None,
    status: EvidenceStatus = EvidenceStatus.SUCCESS,
    body: bytes = b'{"records": [{"close": 1}]}',
    retrieved_at: datetime = RETRIEVED_AT,
) -> ScopedRawPayload:
    return ScopedRawPayload(
        kind=kind,
        source=source,
        natural_key=natural_key,
        as_of=as_of,
        fiscal_period=fiscal_period,
        status=status,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"{source}:{natural_key}",
    )


def test_pre_scope_price_evidence_fails(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    bronze_root = tmp_path / "data" / "bronze" / "kr_swing_2019_v1"

    with pytest.raises(PITDataError, match="evidence start"):
        writer.persist(_payload(as_of=date(2018, 12, 28), natural_key="2018-12-28"))

    assert not bronze_root.exists()
    assert not (bronze_root / "catalog").exists()
    assert catalog.successful_keys(source="krx_daily_market") == frozenset()


def test_pre_floor_dart_fact_fails(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)

    with pytest.raises(PITDataError, match="floor"):
        writer.persist(
            _payload(
                kind=EvidenceKind.FINANCIAL_FACTS,
                source="financial_facts",
                natural_key="00126380:2018:11011",
                as_of=date(2020, 4, 1),
                fiscal_period="2018Q4",
                body=b'{"records": [{"fact": 1}]}',
            )
        )


def test_current_corp_map_is_allowed(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)

    receipt = writer.persist(
        _payload(
            kind=EvidenceKind.SECURITY_MASTER,
            source="dart_corp_codes",
            natural_key="corp-map",
            as_of=None,
            body=b'[{"ticker": "005930"}]',
        )
    )

    assert receipt.bronze_receipt.payload_path.is_file()
    assert catalog.successful_keys(source="dart_corp_codes") == frozenset({"corp-map"})


def test_payload_before_catalog_visibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import scoped_ingestion

    writer, catalog = _writer(tmp_path)
    body = b'{"records": [{"close": 1}]}'

    def _forged_import_bytes(
        self: object, payload: bytes, *, kind: EvidenceKind, retrieved_at: datetime, source_label: str
    ) -> BronzeReceipt:
        assert payload == body
        target = tmp_path / "forged" / "payload.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"different-bytes")
        return BronzeReceipt(
            kind=kind,
            content_hash="0" * 64,
            source_path=source_label,
            retrieved_at=retrieved_at,
            ingested_at=retrieved_at,
            payload_path=target,
            metadata_path=target.parent / "receipt.json",
        )

    monkeypatch.setattr(scoped_ingestion.BronzeStore, "import_bytes", _forged_import_bytes)

    with pytest.raises(PITDataError, match="hash verification"):
        writer.persist(_payload(body=body))

    assert catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"]) == {}


def test_repeat_payload_is_idempotent(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    payload = _payload()

    first = writer.persist(payload)
    second = writer.persist(payload)

    assert first.bronze_receipt.content_hash == second.bronze_receipt.content_hash
    assert second.catalog_revision.row_count == 1
    assert catalog.successful_keys(source="krx_daily_market") == frozenset({"2024-01-02"})

    correction = _payload(body=b'{"records": [{"close": 2}]}', retrieved_at=datetime(2024, 6, 2, tzinfo=UTC))
    third = writer.persist(correction)
    assert third.bronze_receipt.content_hash != first.bronze_receipt.content_hash
    assert Path(first.bronze_receipt.payload_path).is_file()
    assert catalog.latest(source="krx_daily_market", natural_keys=["2024-01-02"])["2024-01-02"].content_hash == third.bronze_receipt.content_hash


def test_persist_rejects_malformed_payloads(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)

    with pytest.raises(PITDataError, match="timezone-aware"):
        writer.persist(_payload(retrieved_at=datetime(2024, 6, 1)))
    with pytest.raises(PITDataError, match="empty Bronze payload"):
        writer.persist(_payload(body=b""))
    with pytest.raises(PITDataError, match="adapter-supplied"):
        writer.persist(_payload(natural_key="  "))
    with pytest.raises(PITDataError, match="must declare as_of"):
        writer.persist(_payload(as_of=None))
    with pytest.raises(PITDataError, match="fiscal period"):
        writer.persist(
            _payload(
                kind=EvidenceKind.FINANCIAL_FACTS,
                source="financial_facts",
                natural_key="00126380:2019:11013",
                as_of=date(2019, 5, 16),
                fiscal_period="2019Q9",
            )
        )
    with pytest.raises(PITDataError, match="fiscal period"):
        writer.persist(
            _payload(
                kind=EvidenceKind.FINANCIAL_FACTS,
                source="financial_facts",
                natural_key="00126380:2019:11013",
                as_of=date(2019, 5, 16),
                fiscal_period=None,
            )
        )


def test_collection_fact_and_market_pages_persist_through_writer(tmp_path: Path) -> None:
    from src.data.collection import (
        collect_daily_market_sessions,
        collect_dart_financial_facts,
        dart_fact_scoped_payload,
        krx_market_scoped_payload,
        persist_scoped_payload,
        scoped_status_for_page,
    )

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)

    assert scoped_status_for_page({"status": "extraction_failed"}) == EvidenceStatus.EXTRACTION_FAILED
    assert scoped_status_for_page({"source_kind": "unavailable"}) == EvidenceStatus.PROVIDER_UNAVAILABLE
    assert scoped_status_for_page({"records": [{"x": 1}]}) == EvidenceStatus.SUCCESS
    assert scoped_status_for_page({"records": []}) == EvidenceStatus.EMPTY

    fact_page = {
        "identity": {
            "corp_code": "00126380",
            "biz_year": "2019",
            "reprt_code": "11013",
            "fiscal_period": "2019Q1",
            "published_at": "2019-05-15",
        },
        "records": [{"account": "revenue"}],
    }
    receipt = persist_scoped_payload(
        scoped_writer=writer, scoped_payload=dart_fact_scoped_payload(page=fact_page, retrieved_at=RETRIEVED_AT)
    )
    assert receipt.payload_path.is_file()

    with pytest.raises(PITDataError, match="adapter natural key"):
        dart_fact_scoped_payload(page={"records": []}, retrieved_at=RETRIEVED_AT)

    class _Dart:
        def fetch_financial_fact_sources(self, identities: object) -> list[dict[str, object]]:
            assert identities
            return [dict(fact_page)]

    collected = collect_dart_financial_facts(
        dart=_Dart(),
        identities=({"corp_code": "00126380", "biz_year": "2019", "reprt_code": "11013"},),
        bronze_root=tmp_path / "legacy-bronze",
        retrieved_at=RETRIEVED_AT,
        scoped_writer=writer,
    )
    assert collected.content_hash
    assert catalog.successful_keys(source="financial_facts", fiscal_start="2019Q1") == frozenset(
        {"00126380:2019:11013"}
    )

    class _Krx:
        def fetch_daily_market(self, start: object, end: object, **kwargs: object) -> list[dict[str, object]]:
            assert kwargs["sessions"]
            return [
                {
                    "session": "2024-01-02",
                    "records": [{"ISU_SRT_CD": "005930", "MKTCAP": 10, "LIST_SHRS": 5}],
                }
            ]

    scoped_direct = krx_market_scoped_payload(
        page={"session": "2024-01-03", "records": []}, session=date(2024, 1, 3), retrieved_at=RETRIEVED_AT
    )
    assert scoped_direct.natural_key == "2024-01-03"
    market = collect_daily_market_sessions(
        sessions=(date(2024, 1, 2),),
        krx=_Krx(),
        bronze_root=tmp_path / "legacy-bronze",
        retrieved_at=RETRIEVED_AT,
        scoped_writer=writer,
    )
    assert market.content_hash
    assert catalog.successful_keys(source="krx_daily_market") == frozenset({"2024-01-02"})


def test_collect_scoped_and_disclosures_commands_persist(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from src.data.cli import main

    payloads = [
        {
            "kind": "daily_market",
            "source": "krx_daily_market",
            "natural_key": "2024-01-02",
            "as_of": "2024-01-02",
            "fiscal_period": None,
            "status": "success",
            "payload_b64": base64.b64encode(b'{"records": [1]}').decode(),
            "retrieved_at": "2024-06-01T00:00:00+00:00",
            "source_label": "test:1",
        }
    ]
    payloads_path = tmp_path / "payloads.json"
    payloads_path.write_text(json.dumps(payloads), encoding="utf-8")
    assert main(["collect-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"), "--payloads", str(payloads_path)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["receipts"] == 1

    disclosures = [{"rcept_no": "20240101000001", "published_at": "2024-02-01", "corp_code": "00126380"}]
    disclosures_path = tmp_path / "disclosures.json"
    disclosures_path.write_text(json.dumps(disclosures), encoding="utf-8")
    assert main(["collect-dart-disclosures-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"), "--disclosures", str(disclosures_path), "--retrieved-at", "2024-06-01T00:00:00+00:00"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["receipts"] == 1

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"rcept_no": ""}]), encoding="utf-8")
    assert main(["collect-dart-disclosures-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"), "--disclosures", str(bad)]) == 1
