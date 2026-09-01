def test_active_tree_excludes_archived_domains_and_old_import_paths() -> None:
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[3]
    src = root / 'src'
    legacy = root / 'legacy'

    assert (legacy / 'stocks').is_dir()
    assert (legacy / 'etfs').is_dir()
    assert (legacy / 'live_yeti_v1').is_dir()
    assert not (src / 'stocks').exists()
    assert not (src / 'etfs').exists()
    assert not (src / 'legacy').exists()
    retired = ('src.legacy', 'src.stocks', 'src.etfs', 'legacy.stocks', 'legacy.etfs')
    # Exclude boundary tests themselves which must reference retired prefixes to verify the boundary
    excluded = {"test_active_archive_boundary.py", "test_import_boundaries.py", "test_architecture_consolidation.py", "test_parquet_datasets.py"}
    active_files = [p for p in [*src.rglob('*.py'), *Path('tests').rglob('*.py')] if p.name not in excluded]
    pat = re.compile(r"(?:src\.legacy|src\.stocks|src\.etfs|legacy\.stocks|legacy\.etfs)")
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert not pat.search(text), f"{path} contains retired prefix"
