def test_migrate_retained_stock_evidence_records_bronze_receipts(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from src.data.bronze import migrate_retained_stock_evidence

    source = Path(tmp_path) / 'legacy'
    costs = source / 'costs'
    costs.mkdir(parents=True)
    payloads = {
        'calendar_20131213_20260311.json': '{"sessions": ["2024-01-02"]}',
        'master_20160104_20260310_historical_v1.json': '{"records": []}',
        'krx-bars-20160104-20260310_backfill_v1.json': '{"records": []}',
        'dart_disclosures_20160101_20260310_v1.json': '{"records": []}',
        'corporate_actions_20160104_20260310_v2.json': '{"intervals": []}',
    }
    for name, text in payloads.items():
        (source / name).write_text(text, encoding='utf-8')
    (costs / 'kis_lifetime_preferential_counterfactual_v1.json').write_text('{"commission": 0}', encoding='utf-8')

    artifact = migrate_retained_stock_evidence(source, Path(tmp_path) / 'data' / 'bronze' / 'stocks', retrieved_at=datetime(2024, 1, 3, tzinfo=UTC))

    assert len(artifact.receipts) == 6
    assert all(receipt.payload_path.exists() for receipt in artifact.receipts.values())
    assert artifact.content_hash


def test_purge_legacy_data_requires_verified_migration_and_confirmation(tmp_path) -> None:
    from pathlib import Path

    import pytest

    from src.data.legacy_inventory import MigrationArtifact, purge_legacy_data

    root = Path(tmp_path) / 'data'
    (root / 'canonical').mkdir(parents=True)
    (root / 'canonical' / 'old.parquet').write_bytes(b'legacy')
    artifact = MigrationArtifact.empty_verified(root)

    with pytest.raises(ValueError, match='confirm_purge'):
        purge_legacy_data(root, artifact, confirm_purge=False)

    removed = purge_legacy_data(root, artifact, confirm_purge=True)

    assert root / 'canonical' in removed
    assert not (root / 'canonical').exists()
