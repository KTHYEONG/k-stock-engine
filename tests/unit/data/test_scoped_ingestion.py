from __future__ import annotations

import base64
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.evidence_sources import KIS_FLOW_SOURCE, LS_FLOW_SOURCE, raw_rows_envelope, source_contract
from src.data.receipt_catalog import CoverageRange, EvidenceStatus, ReceiptCatalog
from src.data.runtime import load_data_runtime
from src.core.pit import BronzeReceipt, EvidenceKind, PITDataError
from src.data.scoped_ingestion import (
    ScopedBronzeWriter,
    ScopedRangePayload,
    ScopedRawPayload,
    flow_natural_key,
)

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


def _ls_row(day: date) -> dict[str, object]:
    row: dict[str, object] = {
        "date": day.strftime("%Y%m%d"),
        "close": 70100,
        "volume": 12_000_000,
        "value": 840_000_000_000,
    }
    row.update({f"tjj{index:04d}": index for index in range(12)})
    row.update({"tjj0016": 30, "tjj0017": 50, "tjj0018": -20})
    return row


def _kis_row(day: date) -> dict[str, object]:
    return {
        "stck_bsop_date": day.strftime("%Y%m%d"),
        "prsn_ntby_qty": "-120000",
        "frgn_ntby_qty": "30000",
        "orgn_ntby_qty": "-20000",
        "etc_ntby_qty": "110000",
    }


def _range_payload(
    *,
    source: str = LS_FLOW_SOURCE,
    window: tuple[date, ...],
    answered: tuple[date, ...] | None = None,
    rows: list[dict[str, object]] | None = None,
    retrieved_at: datetime = RETRIEVED_AT,
) -> ScopedRangePayload:
    contract = source_contract(source)
    if rows is None:
        maker = _ls_row if source == LS_FLOW_SOURCE else _kis_row
        rows = [maker(day) for day in (answered if answered is not None else window)]
    query: dict[str, str] = (
        {"symbol": "005930", "start": window[0].isoformat(), "end": window[-1].isoformat()}
        if source == LS_FLOW_SOURCE
        else {"symbol": "005930", "anchor": window[-1].isoformat()}
    )
    covered = set(answered if answered is not None else window)
    return ScopedRangePayload(
        source=source,
        payload=raw_rows_envelope(contract, query=query, rows=rows),
        ranges=tuple(
            CoverageRange(
                source=source,
                subject="005930",
                start=day,
                end=day,
                status=EvidenceStatus.SUCCESS if day in covered else EvidenceStatus.EMPTY,
                content_hash=None,
                retrieved_at=retrieved_at,
            )
            for day in window
        ),
        retrieved_at=retrieved_at,
        source_label=f"{source}:005930:{window[-1].isoformat()}",
    )


def _ls_range_payload(**kwargs: object) -> ScopedRangePayload:
    return _range_payload(**kwargs)  # type: ignore[arg-type]


def test_pre_scope_price_evidence_fails(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    bronze_root = tmp_path / "data" / "bronze" / "kr_swing_2019_v1"

    with pytest.raises(PITDataError, match="evidence start"):
        writer.persist(_payload(as_of=date(2015, 12, 30), natural_key="2015-12-30"))

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
                natural_key="00126380:2015:11011",
                as_of=date(2016, 4, 1),
                fiscal_period="2015Q4",
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


def test_empty_batch_does_not_create_bronze_state(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)

    assert writer.persist_many(()) == ()
    assert catalog.successful_keys(source="krx_daily_market") == frozenset()


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


def test_batched_payloads_publish_one_catalog_revision(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    receipts = writer.persist_many(
        (
            _payload(natural_key="2024-01-02", as_of=date(2024, 1, 2)),
            _payload(natural_key="2024-01-03", as_of=date(2024, 1, 3)),
        )
    )

    assert len(receipts) == 2
    assert receipts[0].catalog_revision == receipts[1].catalog_revision
    assert catalog.successful_keys(source="krx_daily_market") == frozenset(
        {"2024-01-02", "2024-01-03"}
    )


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


def test_collection_fact_pages_persist_through_writer(tmp_path: Path) -> None:
    from src.data.collection import dart_fact_scoped_payload, scoped_status_for_page

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
    stored = writer.persist(dart_fact_scoped_payload(page=fact_page, retrieved_at=RETRIEVED_AT))
    assert stored is not None
    assert stored.bronze_receipt.payload_path.is_file()
    assert [blob.content_hash for blob in catalog.blobs(source="financial_facts")] == [
        stored.bronze_receipt.content_hash
    ]

    with pytest.raises(PITDataError, match="adapter natural key"):
        dart_fact_scoped_payload(page={"records": []}, retrieved_at=RETRIEVED_AT)

    assert catalog.successful_keys(source="financial_facts", fiscal_start="2019Q1") == frozenset(
        {"00126380:2019:11013"}
    )


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


def _seed_dart_scope(tmp_path: Path):  # type: ignore[no-untyped-def]
    import json

    import polars as pl

    from src.data.dataset_registry import DatasetRegistry
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    bridge_raw = json.dumps(
        [{"ticker": "005930", "corp_code": "00126380", "corp_name": "Test Co"}],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    from tests.fixtures import seed_corp_code_bridge

    seed_corp_code_bridge(runtime.workspace.bronze_root, bridge_raw)
    published = publish_dataset(
        layer_root=runtime.workspace.silver_root,
        identity=DatasetIdentity(
            kind="ordinary_universe", layer=DatasetLayer.SILVER, policy_version="test-v1", inputs={}, params={}
        ),
        partitions={"part.parquet": pl.DataFrame({"ticker": ["005930"], "eligible": [True]})},
    )
    DatasetRegistry(runtime.workspace.state_root).register("ordinary_universe", published.dataset_id)
    return runtime


def test_collect_dart_job_commands_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from src.data.cli import main

    _seed_dart_scope(tmp_path)
    args = ["--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"), "--dry-run"]
    assert main(["collect-dart-disclosures", *args]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["job"] == "dart_disclosures"
    assert summary["status"] == "dry_run"
    assert summary["pending_left"] > 0

    assert main(["collect-dart-facts", *args]) == 2
    assert "periodic filing" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]

    assert main(["collect-dividend-decisions", *args]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["job"] == "dividend_decisions"
    assert summary["status"] == "dry_run"
    assert summary["pending_left"] == 0


def test_collect_disclosures_command_runs_with_fake_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from src.data.cli import main

    _seed_dart_scope(tmp_path)

    class _OfflineCollector:
        def health_check(self) -> None:
            return None

        def list_disclosure_window(self, start, end, *, disclosure_filter=None, detail_type=None):  # type: ignore[no-untyped-def]
            from src.integrations.dart.client import DisclosureListing

            rows = self.list_disclosures(start, end, disclosure_filter=disclosure_filter, detail_type=detail_type)
            return DisclosureListing(records=tuple(rows), reported_total=len(rows), raw_rows=len(rows))

        def list_disclosures(self, start, end, *, disclosure_filter=None, detail_type=None):  # type: ignore[no-untyped-def]
            filt = disclosure_filter if disclosure_filter is not None else detail_type
            code = getattr(filt, "code", filt)
            assert start <= end
            return [
                {
                    "rcept_no": "20160330001234", "rcept_dt": "20160330", "corp_code": "00126380",
                    "corp_name": "Test", "report_nm": "사업보고서 (2015.12)", "rm": "",
                }
            ]

    monkeypatch.setattr(
        "src.data.dart_backfill.build_scoped_dart_collector", lambda **_kwargs: _OfflineCollector()
    )
    args = ["--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data")]
    assert main(["collect-dart-disclosures", *args]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["job"] == "dart_disclosures"
    assert summary["status"] == "complete"
    assert summary["done"] > 0
    assert summary["pending_left"] == 0


def test_blocked_fact_page_is_retryable_not_empty() -> None:
    from src.data.collection import scoped_status_for_page

    blocked = {"source_kind": "blocked", "status": "020", "records": []}

    assert scoped_status_for_page(blocked) == EvidenceStatus.PROVIDER_UNAVAILABLE


def test_batch_publishes_one_catalog_revision_for_blobs_receipts_and_ranges(tmp_path: Path) -> None:
    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    sequences: list[int] = []
    real_publish = catalog.publish

    def _spy(entries, **kwargs):  # type: ignore[no-untyped-def]
        sequences.append(len(entries))
        return real_publish(entries, **kwargs)

    catalog.publish = _spy  # type: ignore[method-assign]

    writer.persist_many(
        (
            _payload(natural_key="2024-01-02", as_of=date(2024, 1, 2), body=b'{"records": [{"close": 1}]}'),
            _payload(natural_key="2024-01-03", as_of=date(2024, 1, 3), body=b'{"records": [{"close": 2}]}'),
            _ls_range_payload(window=(date(2024, 1, 2), date(2024, 1, 3))),
        )
    )

    assert sequences == [2]
    assert len(list(catalog.blobs(source="krx_daily_market"))) == 2
    assert len(list(catalog.blobs(source=LS_FLOW_SOURCE))) == 1
    assert len(list(catalog.ranges(source=LS_FLOW_SOURCE))) == 2


def test_fact_payload_derives_fiscal_period_when_collector_identity_drops_it() -> None:
    from src.data.collection import dart_fact_scoped_payload

    page = {
        "identity": {
            "corp_code": "00126380",
            "biz_year": "2016",
            "reprt_code": "11012",
            "published_at": "2016-08-12",
        },
        "records": [{"account": "revenue"}],
    }
    payload = dart_fact_scoped_payload(page=page, retrieved_at=RETRIEVED_AT)

    assert payload.fiscal_period == "2016Q2"
    assert payload.status == EvidenceStatus.SUCCESS


# --- contract gate in front of Bronze --------------------------------------


def test_contract_is_checked_before_any_byte_is_written(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    bronze_root = tmp_path / "data" / "bronze" / "kr_swing_2019_v1"
    priced_only = json.dumps(
        {
            "envelope": "raw-rows-v1",
            "provider": "KIS",
            "endpoint": "investor-trade-by-stock-daily",
            "query": {"symbol": "005930", "anchor": "20240102"},
            "rows": [{"stck_bsop_date": "20240102", "prsn_ntby_tr_pbmn": "840000000000"}],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = ScopedRangePayload(
        source=KIS_FLOW_SOURCE,
        payload=priced_only,
        ranges=(
            CoverageRange(
                source=KIS_FLOW_SOURCE,
                subject="005930",
                start=date(2024, 1, 2),
                end=date(2024, 1, 2),
                status=EvidenceStatus.SUCCESS,
                content_hash=None,
                retrieved_at=RETRIEVED_AT,
            ),
        ),
        retrieved_at=RETRIEVED_AT,
        source_label="KIS:investor-trade-by-stock-daily:005930:2024-01-02",
    )

    with pytest.raises(PITDataError, match="prsn_ntby_qty"):
        writer.persist(payload)

    assert not (bronze_root / "investor_flow").exists()
    assert list(catalog.blobs(source=KIS_FLOW_SOURCE)) == []
    assert list(catalog.ranges(source=KIS_FLOW_SOURCE)) == []


def test_keyed_payload_for_a_ranged_source_is_rejected(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)
    envelope = raw_rows_envelope(
        source_contract(LS_FLOW_SOURCE),
        query={"symbol": "005930", "start": "2024-01-02", "end": "2024-01-02"},
        rows=[_ls_row(date(2024, 1, 2))],
    )
    keyed = ScopedRawPayload(
        kind=EvidenceKind.INVESTOR_FLOW,
        source=LS_FLOW_SOURCE,
        natural_key=flow_natural_key(symbol="005930", session=date(2024, 1, 2)),
        as_of=date(2024, 1, 2),
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=envelope,
        retrieved_at=RETRIEVED_AT,
        source_label="LS:t1702:005930:2024-01-02",
    )

    with pytest.raises(PITDataError, match="ranged coverage"):
        writer.persist(keyed)


def test_range_payload_for_a_keyed_source_is_rejected(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)

    with pytest.raises(PITDataError, match="keyed coverage"):
        writer.persist(
            ScopedRangePayload(
                source="krx_daily_market",
                payload=b'{"session": "2024-01-02", "records": []}',
                ranges=(
                    CoverageRange(
                        source="krx_daily_market",
                        subject="005930",
                        start=date(2024, 1, 2),
                        end=date(2024, 1, 2),
                        status=EvidenceStatus.EMPTY,
                        content_hash=None,
                        retrieved_at=RETRIEVED_AT,
                    ),
                ),
                retrieved_at=RETRIEVED_AT,
                source_label="krx:005930",
            )
        )


def test_unregistered_source_is_rejected_by_the_writer(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)
    payload = _payload(source="investor_flow", natural_key="005930:2024-01-02")

    with pytest.raises(PITDataError, match="unregistered evidence source"):
        writer.persist(payload)


def test_blob_and_ranges_commit_in_one_revision(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)

    first = writer.persist_many((_ls_range_payload(window=(date(2024, 1, 2), date(2024, 1, 3))),))

    assert [receipt.catalog_revision.sequence for receipt in first] == [1]
    assert len(list(catalog.blobs(source=LS_FLOW_SOURCE))) == 1
    stored = list(catalog.ranges(source=LS_FLOW_SOURCE))
    assert len(stored) == 2
    assert {item.status for item in stored} == {EvidenceStatus.SUCCESS}
    assert {item.content_hash for item in stored} == {first[0].bronze_receipt.content_hash}


def test_answered_and_missing_sessions_are_recorded_as_one_range_set(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    window = (date(2024, 1, 2), date(2024, 1, 3))

    writer.persist(_range_payload(window=window, answered=(window[0],)))

    stored = list(catalog.ranges(source=LS_FLOW_SOURCE))
    assert [(item.start.isoformat(), item.status.value) for item in stored] == [
        ("2024-01-02", "success"),
        ("2024-01-03", "empty"),
    ]
    assert [item.content_hash for item in stored] == [stored[0].content_hash, None]


def test_range_without_bytes_stores_nothing(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    bronze_root = tmp_path / "data" / "bronze" / "kr_swing_2019_v1"

    assert writer.persist_many(
        (
            ScopedRangePayload(
                source=LS_FLOW_SOURCE,
                payload=None,
                ranges=(
                    CoverageRange(
                        source=LS_FLOW_SOURCE,
                        subject="005930",
                        start=date(2024, 1, 2),
                        end=date(2024, 1, 2),
                        status=EvidenceStatus.EMPTY,
                        content_hash=None,
                        retrieved_at=RETRIEVED_AT,
                    ),
                ),
                retrieved_at=RETRIEVED_AT,
                source_label="LS:t1702:005930:2024-01-02",
            ),
        )
    ) == ()
    assert next(iter(catalog.ranges(source=LS_FLOW_SOURCE))).status is EvidenceStatus.EMPTY
    assert not (bronze_root / "investor_flow").exists()


def test_range_scope_and_shape_checks_fail_closed(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)
    bronze_root = tmp_path / "data" / "bronze" / "kr_swing_2019_v1"
    base = _range_payload(window=(date(2024, 1, 2),)).ranges[0]

    with pytest.raises(PITDataError, match="precedes evidence start"):
        writer.persist(_range_payload(window=(date(2015, 12, 30),)))
    with pytest.raises(PITDataError, match="timezone-aware"):
        writer.persist(_replace_ranges(_ls_range_payload(window=(date(2024, 1, 2),)), retrieved_at=datetime(2024, 6, 1)))
    with pytest.raises(PITDataError, match="source and source label"):
        writer.persist(_replace_ranges(_ls_range_payload(window=(date(2024, 1, 2),)), source_label="  "))
    with pytest.raises(PITDataError, match="at least one answered range"):
        writer.persist(
            ScopedRangePayload(
                source=LS_FLOW_SOURCE,
                payload=None,
                ranges=(),
                retrieved_at=RETRIEVED_AT,
                source_label="x",
            )
        )
    with pytest.raises(PITDataError, match="does not match payload source"):
        writer.persist(_replace_ranges(_ls_range_payload(window=(date(2024, 1, 2),)), ranges=(replace(base, source=KIS_FLOW_SOURCE),)))
    with pytest.raises(PITDataError, match="ends before it starts"):
        writer.persist(
            _replace_ranges(_ls_range_payload(window=(date(2024, 1, 2),)), ranges=(replace(base, end=date(2024, 1, 1)),))
        )
    with pytest.raises(PITDataError, match="must not reference a blob"):
        writer.persist(
            _replace_ranges(
                _ls_range_payload(window=(date(2024, 1, 2),)),
                ranges=(replace(base, status=EvidenceStatus.EMPTY, content_hash="a" * 64),),
            )
        )
    with pytest.raises(PITDataError, match="cannot answer a successful range"):
        writer.persist(replace(_ls_range_payload(window=(date(2024, 1, 2),)), payload=None))
    with pytest.raises(PITDataError, match="needs an envelope that carries rows"):
        writer.persist(_empty_envelope_payload(base))
    assert not (bronze_root / "investor_flow").exists()


def _replace_ranges(payload: ScopedRangePayload, **overrides: object) -> ScopedRangePayload:
    return replace(payload, **overrides)  # type: ignore[arg-type]


def _empty_envelope_payload(base: CoverageRange) -> ScopedRangePayload:
    """A range claiming success while the provider answered with no rows at all."""
    return ScopedRangePayload(
        source=LS_FLOW_SOURCE,
        payload=raw_rows_envelope(
            source_contract(LS_FLOW_SOURCE),
            query={"symbol": "005930", "start": "2024-01-02", "end": "2024-01-02"},
            rows=[],
        ),
        ranges=(base,),
        retrieved_at=RETRIEVED_AT,
        source_label="LS:t1702:005930:2024-01-02",
    )


def test_non_canonical_raw_rows_payload_is_never_rewritten(tmp_path: Path) -> None:
    writer, catalog = _writer(tmp_path)
    canonical = raw_rows_envelope(
        source_contract(LS_FLOW_SOURCE),
        query={"symbol": "005930", "start": "2024-01-02", "end": "2024-01-02"},
        rows=[_ls_row(date(2024, 1, 2))],
    )
    spaced = json.dumps(json.loads(canonical), sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert spaced != canonical
    payload = _ls_range_payload(window=(date(2024, 1, 2),))
    spaced_payload = ScopedRangePayload(
        source=payload.source, payload=spaced, ranges=payload.ranges,
        retrieved_at=payload.retrieved_at, source_label=payload.source_label,
    )

    with pytest.raises(PITDataError, match="not canonical"):
        writer.persist(spaced_payload)

    assert list(catalog.ranges(source=LS_FLOW_SOURCE)) == []
    assert list(catalog.blobs(source=LS_FLOW_SOURCE)) == []


def test_keyed_payload_evidence_kind_must_match_its_source(tmp_path: Path) -> None:
    writer, _catalog = _writer(tmp_path)
    payload = _payload(kind=EvidenceKind.INVESTOR_FLOW, source="krx_daily_market")

    with pytest.raises(PITDataError, match="stores daily_market evidence"):
        writer.persist(payload)


def test_industry_pages_persist_through_the_writer(tmp_path: Path) -> None:
    from src.data.industry_collection import collect_classification_with_isolation

    writer, catalog = _writer(tmp_path)
    moment = "2024-06-01T09:00:00+09:00"

    class _Collector:
        def __init__(self, symbols: tuple[str, ...]) -> None:
            self.symbols = symbols

        def fetch_industry_classification(self, *, bronze_root: Path):  # type: ignore[no-untyped-def]
            for endpoint in ("inquire-price", "search-stock-info"):
                yield {
                    "endpoint": endpoint,
                    "collected_at": moment,
                    "output": {"price": 70100},
                    "records": [{"ticker": self.symbols[0], "industry": "전자"}],
                }

    report = collect_classification_with_isolation(
        stage="test-industry",
        collector_cls=_Collector,
        fetch_attr="fetch_industry_classification",
        bronze_root=tmp_path / "data" / "bronze" / "kr_swing_2019_v1",
        writer=writer,
        symbols=("005930",),
        pace_seconds=0.0,
    )

    assert report["pages_collected"] == 2
    assert catalog.successful_keys(source="kis_industry") == frozenset(
        {
            "inquire-price:005930:2024-06-01",
            "search-stock-info:005930:2024-06-01",
        }
    )
    assert len(list(catalog.blobs(source="kis_industry"))) == 2
    latest = catalog.latest(
        source="kis_industry", natural_keys=["inquire-price:005930:2024-06-01"]
    )
    assert latest["inquire-price:005930:2024-06-01"].as_of == date(2024, 6, 1)


def test_industry_page_without_an_endpoint_is_rejected(tmp_path: Path) -> None:
    from src.data.industry_collection import _industry_scoped_payload

    with pytest.raises(PITDataError, match="missing its endpoint"):
        _industry_scoped_payload(
            symbol="005930", page={"collected_at": "2024-06-01T09:00:00+09:00"}, moment=RETRIEVED_AT
        )


def test_only_the_writer_module_imports_bronze_store() -> None:
    """C1's Bronze import boundary: collection and industry collection no longer write Bronze."""
    import ast
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[3] / "src" / "data"
    importers: set[str] = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imported = (
                node.module == "src.data.bronze"
                if isinstance(node, ast.ImportFrom)
                else any(alias.name == "src.data.bronze" for alias in node.names)
                if isinstance(node, ast.Import)
                else False
            )
            if imported:
                importers.add(path.name)

    assert importers <= {"scoped_ingestion.py", "dart_documents.py"}
    assert "collection.py" not in importers
    assert "industry_collection.py" not in importers

