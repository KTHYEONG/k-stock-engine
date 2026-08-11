"""Stock catalog CLI: inspect versions and validate immutable snapshots.

``list`` prints catalog versions by kind; ``validate`` resolves a snapshot and
fails closed on any missing, range-incomplete, hash-mismatched, or
``candidate_only`` evidence without scanning Parquet; ``validate-readiness``
rejects a model configuration that selects a missing, fully-null, or
non-finite feature column; ``retention-dry-run`` lists garbage-collection
candidates without changing any file.
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
from src.stocks.data.readiness import validate_selected_feature_readiness

logger = logging.getLogger("stocks.cli.catalog")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the stock data catalog")
    parser.add_argument("--catalog-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list catalog versions")
    validate = sub.add_parser("validate", help="validate an immutable snapshot")
    validate.add_argument("--snapshot-id", required=True)
    readiness = sub.add_parser(
        "validate-readiness", help="validate selected feature readiness against the content manifest"
    )
    readiness.add_argument("--dataset-dir", required=True, type=Path)
    readiness.add_argument("--feature", required=True, action="append")
    sub.add_parser("retention-dry-run", help="list GC candidates without changing files")
    parsed = parser.parse_args(args)

    root = getattr(parsed, "catalog_root", None) or STOCK_CATALOG_ROOT
    store = CatalogStore(root)
    if parsed.command == "list":
        return _list(store)
    if parsed.command == "validate":
        return _validate(store, parsed.snapshot_id)
    if parsed.command == "validate-readiness":
        return _validate_readiness(parsed.dataset_dir, tuple(parsed.feature))
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


def _validate_readiness(dataset_dir: Path, selected: tuple[str, ...]) -> int:
    try:
        report = validate_selected_feature_readiness(dataset_dir, selected)
    except ValueError as exc:
        sys.stdout.write(f"readiness failed: {exc}\n")
        return 1
    sys.stdout.write(f"dataset {report.dataset_dir} OK: total_rows={report.total_rows}\n")
    for feature in report.selected.values():
        sys.stdout.write(
            f"  {feature.name}\tnull={feature.null_count}\t"
            f"non_null={feature.non_null_count}\tnon_finite={feature.non_finite_count}\n"
        )
    if report.fully_null_stored_columns_not_selected:
        sys.stdout.write(
            "  fully-null stored columns (not selected): "
            + ", ".join(report.fully_null_stored_columns_not_selected)
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
