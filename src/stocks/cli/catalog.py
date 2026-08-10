"""Stock catalog CLI: inspect versions and validate immutable snapshots.

``list`` prints catalog versions by kind; ``validate`` resolves a snapshot and
fails closed on any missing, range-incomplete, hash-mismatched, or
``candidate_only`` evidence without scanning Parquet; ``retention-dry-run``
lists garbage-collection candidates without changing any file.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core.paths import STOCK_CATALOG_ROOT
from src.stocks.data.catalog import (
    CatalogKind,
    CatalogStore,
    RetentionRegistry,
    SnapshotResolver,
    retention_dry_run,
)

logger = logging.getLogger("stocks.cli.catalog")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the stock data catalog")
    parser.add_argument("--catalog-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list catalog versions")
    validate = sub.add_parser("validate", help="validate an immutable snapshot")
    validate.add_argument("--snapshot-id", required=True)
    sub.add_parser("retention-dry-run", help="list GC candidates without changing files")
    parsed = parser.parse_args(args)

    root = getattr(parsed, "catalog_root", None) or STOCK_CATALOG_ROOT
    store = CatalogStore(root)
    if parsed.command == "list":
        return _list(store)
    if parsed.command == "validate":
        return _validate(store, parsed.snapshot_id)
    return _retention_dry_run(store)


def _list(store: CatalogStore) -> int:
    out: list[str] = []
    for kind in CatalogKind:
        entries = store.list(kind)
        if not entries:
            continue
        out.append(f"# {kind.value}")
        for entry in entries:
            coverage = (
                f"{entry.coverage.start}..{entry.coverage.end}"
                if entry.coverage
                else "n/a"
            )
            out.append(
                f"  {entry.name}\t{entry.content_hash[:12]}\t"
                f"{entry.completeness.value}\t{coverage}\t{entry.path}"
            )
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


def _validate(store: CatalogStore, snapshot_id: str) -> int:
    resolver = SnapshotResolver(store)
    snapshot = resolver.resolve(snapshot_id)
    sys.stdout.write(
        f"snapshot {snapshot_id}: OK (certification={snapshot.manifest.certification.value}, "
        f"range={snapshot.research_range.start}..{snapshot.research_range.end})\n"
    )
    return 0


def _retention_dry_run(store: CatalogStore) -> int:
    registry = RetentionRegistry(store.root)
    candidates = retention_dry_run(store, registry)
    if not candidates:
        sys.stdout.write("no retention candidates\n")
        return 0
    for candidate in candidates:
        sys.stdout.write(f"{candidate.kind.value}\t{candidate.name}\t{candidate.reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
