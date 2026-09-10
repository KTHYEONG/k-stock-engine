def test_inspect_legacy_data_classifies_reuse_and_removal(tmp_path) -> None:
    from pathlib import Path

    from src.data.legacy_inventory import LegacyDisposition, inspect_legacy_data

    root = Path(tmp_path) / 'data'
    evidence = root / 'evidence' / 'stocks'
    evidence.mkdir(parents=True)
    (evidence / 'calendar_20131213_20260311.json').write_text('{"sessions": []}', encoding='utf-8')
    (root / 'canonical').mkdir()
    (root / 'trading_state.db').write_bytes(b'legacy')

    inventory = inspect_legacy_data(root)

    by_path = {item.relative_path: item.disposition for item in inventory.entries}
    assert by_path['evidence/stocks/calendar_20131213_20260311.json'] is LegacyDisposition.REUSE_AS_BRONZE
    assert by_path['canonical'] is LegacyDisposition.REMOVE
    assert by_path['trading_state.db'] is LegacyDisposition.REMOVE


def test_inventory_allows_only_documented_raw_seeds(tmp_path) -> None:
    from src.data.legacy_inventory import LegacyDisposition, inspect_legacy_data

    root = tmp_path / 'data'
    stocks = root / 'evidence' / 'stocks'
    stocks.mkdir(parents=True)
    (stocks / 'calendar_20131213_20260311.json').write_text('{\"sessions\": []}', encoding='utf-8')
    (stocks / 'master_20260310.json').write_text('{\"records\": []}', encoding='utf-8')
    (root / 'derived' / 'stocks').mkdir(parents=True)

    items = {item.relative_path: item.disposition for item in inspect_legacy_data(root).entries}

    assert items['evidence/stocks/calendar_20131213_20260311.json'] is LegacyDisposition.REUSE_AS_BRONZE
    assert items['evidence/stocks/master_20260310.json'] is LegacyDisposition.REMOVE
    assert items['derived'] is LegacyDisposition.REMOVE


def test_plan_bronze_retention_reports_references_without_deletion(tmp_path) -> None:
    import json
    from src.data.legacy_inventory import plan_bronze_retention

    bronze = tmp_path / 'bronze'
    referenced = 'a' * 64
    unreferenced = 'b' * 64
    for content_hash in (referenced, unreferenced):
        receipt_dir = bronze / 'corporate_actions' / content_hash
        receipt_dir.mkdir(parents=True)
        (receipt_dir / 'payload.json').write_text('{}', encoding='utf-8')
        (receipt_dir / 'receipt.json').write_text(json.dumps({'content_hash': content_hash}), encoding='utf-8')
    silver = tmp_path / 'silver'
    silver.mkdir()
    (silver / 'dataset_manifest.json').write_text(json.dumps({'source_hashes': [referenced], 'non_hash_text': f'0{unreferenced}0'}), encoding='utf-8')

    plan = plan_bronze_retention(bronze_root=bronze, provenance_roots=(silver, tmp_path / 'missing'))

    assert plan.receipt_count == 2
    assert plan.referenced_hashes == (referenced,)
    assert plan.unreferenced_hashes == (unreferenced,)
    assert plan.deletion_eligible is False
    assert plan.unreferenced_payload_bytes == 2


def test_plan_bronze_retention_rejects_invalid_read_size(tmp_path) -> None:
    import pytest
    from src.data.legacy_inventory import plan_bronze_retention

    with pytest.raises(ValueError, match='read_size'):
        plan_bronze_retention(bronze_root=tmp_path / 'bronze', provenance_roots=(), read_size=0)
    with pytest.raises(ValueError, match='read_size'):
        plan_bronze_retention(bronze_root=tmp_path / 'bronze', provenance_roots=(), read_size=True)


def test_plan_bronze_retention_blocks_unreadable_and_malformed_provenance(tmp_path) -> None:
    import json
    from src.data.legacy_inventory import plan_bronze_retention

    content_hash = 'c' * 64
    receipt_dir = tmp_path / 'bronze' / 'corporate_actions' / content_hash
    receipt_dir.mkdir(parents=True)
    (receipt_dir / 'payload.json').write_text('{}', encoding='utf-8')
    (receipt_dir / 'receipt.json').write_text(json.dumps({'content_hash': content_hash}), encoding='utf-8')
    provenance = tmp_path / 'silver'
    provenance.mkdir()
    (provenance / 'unreadable.json').symlink_to(provenance / 'missing-target.json')
    (provenance / 'malformed.json').write_bytes(b'\xff')

    plan = plan_bronze_retention(bronze_root=tmp_path / 'bronze', provenance_roots=(provenance,), read_size=1)

    assert plan.referenced_hashes == ()
    assert plan.unreferenced_hashes == (content_hash,)
    assert plan.deletion_eligible is False
    assert sum(reason.startswith('unreadable_provenance:') for reason in plan.blocking_reasons) == 2
