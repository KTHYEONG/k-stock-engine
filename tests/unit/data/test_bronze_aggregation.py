def test_discover_verified_bronze_receipts_keeps_all_pages_in_stable_order(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.bronze import BronzeStore
    from src.data.bronze_aggregation import discover_verified_bronze_receipts
    from src.data.schemas import EvidenceKind, PITDataError

    store = BronzeStore(tmp_path / "bronze")
    first = store.import_bytes(b'{"records":[{"filing_id":"A"}]}', kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=datetime(2020, 1, 1, tzinfo=UTC), source_label="fixture-a")
    second = store.import_bytes(b'{"records":[{"filing_id":"B"}]}', kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=datetime(2020, 1, 2, tzinfo=UTC), source_label="fixture-b")
    grouped = discover_verified_bronze_receipts(bronze_root=tmp_path / "bronze")
    assert [item.content_hash for item in grouped[EvidenceKind.FINANCIAL_FACTS]] == [first.content_hash, second.content_hash]
    second.payload_path.write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="hash"):
        discover_verified_bronze_receipts(bronze_root=tmp_path / "bronze")


def test_aggregate_small_bronze_pages_retains_standard_and_legacy_dart_records(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    from src.data.bronze import BronzeStore
    from src.data.bronze_aggregation import aggregate_small_bronze_pages
    from src.data.schemas import EvidenceKind

    store = BronzeStore(tmp_path / "bronze")
    standard = store.import_bytes(b'{"source_kind":"opendart_standard","records":[{"filing_id":"S"}]}', kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=datetime(2020, 1, 1, tzinfo=UTC), source_label="standard")
    legacy = store.import_bytes(b'{"source_kind":"legacy_document","records":[{"filing_id":"L"}]}', kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=datetime(2020, 1, 2, tzinfo=UTC), source_label="legacy")
    merged = aggregate_small_bronze_pages(kind=EvidenceKind.FINANCIAL_FACTS, receipts=(standard, legacy), store=store)
    payload = json.loads(merged.payload_path.read_text(encoding="utf-8"))
    assert [page["source_kind"] for page in payload["pages"]] == ["opendart_standard", "legacy_document"]
    assert [[row["filing_id"] for row in page["records"]] for page in payload["pages"]] == [["S"], ["L"]]
    assert payload["input_receipt_hashes"] == [standard.content_hash, legacy.content_hash]


def test_discover_verified_bronze_receipts_hashes_large_payload_without_read_bytes(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.bronze import BronzeStore
    from src.data.bronze_aggregation import discover_verified_bronze_receipts
    from src.data.schemas import EvidenceKind, PITDataError

    store = BronzeStore(tmp_path / 'bronze')
    receipt = store.import_bytes(b'x' * (2 * 1024 * 1024), kind=EvidenceKind.DAILY_MARKET, retrieved_at=datetime(2020, 1, 1, tzinfo=UTC), source_label='fixture')
    monkeypatch.setattr(type(receipt.payload_path), 'read_bytes', lambda _self: (_ for _ in ()).throw(AssertionError('read_bytes')))
    assert discover_verified_bronze_receipts(bronze_root=tmp_path / 'bronze')[EvidenceKind.DAILY_MARKET] == (receipt,)
    receipt.payload_path.write_bytes(b'tampered')
    with pytest.raises(PITDataError, match='hash mismatch'):
        discover_verified_bronze_receipts(bronze_root=tmp_path / 'bronze')


def test_select_streaming_receipts_excludes_derived_merged_payload(tmp_path) -> None:
    from datetime import UTC, datetime
    from src.data.bronze import BronzeStore
    from src.data.bronze_aggregation import select_streaming_receipts
    from src.data.schemas import EvidenceKind

    store = BronzeStore(tmp_path / 'bronze')
    original = store.import_bytes(b'{"records": []}', kind=EvidenceKind.SECURITY_MASTER, retrieved_at=datetime(2020, 1, 1, tzinfo=UTC), source_label='KRX:historical-master:2020-01-01')
    derived = store.import_bytes(b'{"derived": true, "records": []}', kind=EvidenceKind.SECURITY_MASTER, retrieved_at=datetime(2020, 1, 2, tzinfo=UTC), source_label='merged:security_master')
    assert select_streaming_receipts(kind=EvidenceKind.SECURITY_MASTER, receipts=(original, derived)) == (original,)
