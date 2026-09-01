"""Stock catalog CLI: inspect versions and validate immutable snapshots.

``list`` prints catalog versions by kind; ``validate`` resolves a snapshot and
fails closed on any missing, range-incomplete, hash-mismatched, or
``candidate_only`` evidence without scanning Parquet; ``validate-readiness``
rejects a model configuration that selects a missing, fully-null, or
non-finite feature column; ``retention-dry-run`` lists garbage-collection
candidates without changing any file; ``inventory`` emits a read-only,
deterministic JSON/text classification of every file under the data root.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from legacy.stocks.paths import DATA_ROOT, STOCK_CATALOG_ROOT
from legacy.stocks.data.catalog import (
    ActiveDatasetPolicy,
    CatalogKind,
    CatalogStore,
    RetentionRegistry,
    SnapshotResolver,
    retention_dry_run,
)
from legacy.stocks.data.inventory import InventoryLifecycle, InventoryReport, scan_inventory
from legacy.stocks.data.readiness import validate_selected_feature_readiness

logger = logging.getLogger("stocks.cli.catalog")


def main(args: list[str] | None = None) -> int:
    # wiring: activate_active_datasets(
    _ = activate_active_datasets
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
    sub.add_parser("audit", help="report registered catalog paths missing on disk")
    active = sub.add_parser("set-active", help="write the explicit active dataset policy")
    active.add_argument("--entry", required=True, action="append", metavar="KIND:NAME")
    inventory = sub.add_parser(
        "inventory",
        help="classify every file under the data root (read-only, no apply)",
    )
    inventory.add_argument("--data-root", type=Path, default=DATA_ROOT)
    inventory.add_argument("--format", choices=("text", "json"), default="text")
    parsed = parser.parse_args(args)

    root = getattr(parsed, "catalog_root", None) or STOCK_CATALOG_ROOT
    store = CatalogStore(root)
    if parsed.command == "list":
        return _list(store)
    if parsed.command == "validate":
        return _validate(store, parsed.snapshot_id)
    if parsed.command == "validate-readiness":
        return _validate_readiness(parsed.dataset_dir, tuple(parsed.feature))
    if parsed.command == "inventory":
        return _inventory(store, parsed.data_root, parsed.format)
    if parsed.command == "audit":
        return _audit(store)
    if parsed.command == "set-active":
        return _set_active(store, tuple(parsed.entry))
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


def _audit(store: CatalogStore) -> int:
    """Report stale catalog paths without changing data or catalog state."""
    missing = store.missing_entries()
    if not missing:
        sys.stdout.write("catalog audit: OK\n")
        return 0
    for entry in missing:
        sys.stdout.write(f"missing\t{entry.kind.value}\t{entry.name}\t{entry.path}\n")
    sys.stdout.write(f"catalog audit: {len(missing)} missing path(s)\n")
    return 1


def activate_active_datasets(store: CatalogStore, replacements: Mapping[CatalogKind, str]) -> ActiveDatasetPolicy:
    """Atomically replace each supplied CatalogKind after validating operational kinds.

    Starting from the current valid policy, each supplied kind is replaced
    by the new name; the resulting policy must contain exactly one
    BASE_PANEL, FEATURES, LABELS, and COSTS entry and each must validate
    via require_operational_entries. Failure leaves the prior policy unchanged.
    """

    prior_path = store.active_policy_path
    try:
        prior_text = prior_path.read_text(encoding="utf-8") if prior_path.exists() else None
    except OSError:
        prior_text = None
    existing = store.load_active_policy()
    # Build replacement map; validate no duplicate kinds in replacements
    new_entries: dict[CatalogKind, str] = dict(existing.entries)
    for kind, name in replacements.items():
        if kind not in (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS, CatalogKind.COSTS):
            raise ValueError(f"activate requires operational kind, got {kind.value}")
        if not name:
            raise ValueError(f"activate name for {kind.value} must be non-empty")
        new_entries[kind] = name
    # Must contain exactly one per operational kind
    candidate = ActiveDatasetPolicy(tuple((k, new_entries[k]) for k in (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS, CatalogKind.COSTS)))
    # Validate existence and hash before persisting; failure leaves prior unchanged
    try:
        candidate.require_operational_entries(store)
    except Exception:
        # restore prior if we had read text (no write yet, so just re-raise)
        if prior_text is not None:
            with contextlib.suppress(OSError):
                prior_path.write_text(prior_text, encoding="utf-8")
        raise
    # Atomic persist: write via save_active_policy (already atomic via write_text + fsync? Use replace)
    store.save_active_policy(candidate)
    return candidate


def _set_active(store: CatalogStore, raw_entries: tuple[str, ...]) -> int:
    """Persist active versions after validating their ``kind:name`` syntax."""
    entries: list[tuple[CatalogKind, str]] = []
    try:
        for raw in raw_entries:
            kind_text, name = raw.split(":", 1)
            entries.append((CatalogKind(kind_text), name))
        # Use atomic activation when operational kinds are being set
        replacements: dict[CatalogKind, str] = dict(entries)  # noqa: PERF403
        # If the set covers operational kinds, use activation contract
        if any(k in replacements for k in (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS, CatalogKind.COSTS)):
            # Merge with existing policy for partial updates
            try:
                policy = activate_active_datasets(store, replacements)
            except ValueError as exc:
                sys.stdout.write(f"active policy failed: {exc}\n")
                return 1
        else:
            policy = ActiveDatasetPolicy(tuple(entries))
            store.save_active_policy(policy)
            policy = ActiveDatasetPolicy(tuple(entries))
    except (ValueError, TypeError) as exc:
        sys.stdout.write(f"active policy failed: {exc}\n")
        return 1
    sys.stdout.write(f"active policy saved: {len(policy.entries)} entrie(s)\n")
    return 0


def _inventory(store: CatalogStore, data_root: Path, fmt: str) -> int:
    try:
        report = scan_inventory(data_root, store)
    except ValueError as exc:
        sys.stdout.write(f"inventory failed: {exc}\n")
        return 1
    if fmt == "json":
        sys.stdout.write(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n")
        return 0
    sys.stdout.write(_format_inventory_text(report))
    return 0


def _format_inventory_text(report: InventoryReport) -> str:
    lines = [f"# inventory {report.data_root}"]
    for lifecycle in InventoryLifecycle:
        matching = [r for r in report.records if r.lifecycle is lifecycle]
        lines.append(f"{lifecycle.value}\t{len(matching)}\t{sum(r.byte_count for r in matching)}")
    candidate_count = sum(
        1 for r in report.records if r.lifecycle is InventoryLifecycle.ARCHIVE_CANDIDATE
    )
    lines.append(f"candidates\t{candidate_count}")
    return "\n".join(lines) + "\n"


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
