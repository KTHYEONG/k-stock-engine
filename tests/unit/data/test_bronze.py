def test_bronze_import_preserves_bytes_and_is_idempotent(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from src.data.bronze import BronzeStore
    from src.data.schemas import EvidenceKind

    source = Path(tmp_path) / 'calendar.json'
    source.write_bytes(b'{"sessions":["2024-01-02"]}')
    store = BronzeStore(Path(tmp_path) / 'bronze')
    first = store.import_json(source, kind=EvidenceKind.CALENDAR, retrieved_at=datetime(2024, 1, 2, tzinfo=UTC))
    second = store.import_json(source, kind=EvidenceKind.CALENDAR, retrieved_at=datetime(2024, 1, 2, tzinfo=UTC))

    assert first.content_hash == second.content_hash
    assert first.payload_path.read_bytes() == source.read_bytes()
    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()


def test_retained_registry_rejects_missing_required_source(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    import pytest

    from src.data.bronze import BronzeStore, import_retained_stock_evidence
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='krx-bars'):
        import_retained_stock_evidence(Path(tmp_path), store=BronzeStore(Path(tmp_path) / 'bronze'), retrieved_at=datetime(2024, 1, 2, tzinfo=UTC))
