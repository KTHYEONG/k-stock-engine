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
