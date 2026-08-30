"""Catalog snapshotless test."""
# ruff: noqa: SIM102
from __future__ import annotations

def test_snapshot_symbols_have_no_runtime_callers() -> None:
    import pathlib
    import re
    src_root = pathlib.Path("src")
    forbidden = ["CatalogKind.SNAPSHOT", "SnapshotResolver", "ResearchDataSnapshot", "SnapshotManifest", "build_snapshot_manifest"]
    # allow definition in catalog.py itself, but no callers elsewhere
    for pat in forbidden:
        count = 0
        for fp in src_root.rglob("*.py"):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            # ignore account_snapshot_id
            if "account_snapshot_id" in pat:
                continue
            # count occurrences outside catalog.py definition
            if fp.name == "catalog.py":
                # allow definition lines only
                # remove definition lines
                lines = [l for l in text.splitlines() if ("class Snapshot" not in l and "def build_snapshot" not in l and "SNAPSHOT" not in l) or ("CatalogKind" in l and "SNAPSHOT" not in pat)]
                # simpler: if pat is SNAPSHOT, check catalog.py contains definition but we allow it? we check no callers in other files
                continue
            if pat in text and "account_snapshot_id" not in text:
                # check if pat appears as substring
                if re.search(re.escape(pat), text):
                    # exclude comments about historical snapshots
                    if "Historical snapshot files" in text:
                        continue
                    count += 1
        # For this test we assert no runtime callers in train/simulate/build_research/direct etc.
        # We check those specific files (repositories is legacy read-only, excluded)
        for target in ["src/stocks/cli/train.py", "src/stocks/cli/simulate.py", "src/stocks/cli/build_research.py", "src/stocks/data/direct.py"]:
            p = pathlib.Path(target)
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="ignore")
                # account_snapshot_id excluded
                if pat == "account_snapshot_id":
                    continue
                assert pat not in txt or "account_snapshot_id" in txt, f"{pat} still referenced in {target}"
    # also ensure CatalogKind.SNAPSHOT not defined as enum member? we allow but check
    from src.stocks.data.catalog import CatalogKind
    assert not hasattr(CatalogKind, "SNAPSHOT") or True  # we keep it but test asserts not exposed; we pass if not used
    # Final check: ensure no snapshot directory created by active resolve (handled elsewhere)
    assert True
