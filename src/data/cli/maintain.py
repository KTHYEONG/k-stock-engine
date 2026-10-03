"""Maintain-area commands: workspace setup, audits, verification and pruning."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from src.data.cli.common import (
    CommandFailed,
    add_scoped_args,
    emit,
    scoped_runtime,
)
from src.data.cli.registry import Command

__all__ = ["MAINTAIN_COMMANDS"]

_LOG = logging.getLogger(__name__)


def _add_audit(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)


def _add_verify(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--all", action="store_true", help="Verify every Silver and Gold dataset")


def _add_prune(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--apply", action="store_true", help="Delete the revalidated prune set")


def _add_index(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Classify without publishing anything")
    parser.add_argument("--batch-size", type=int, default=1000, help="Blobs per catalog revision")


def _add_migrate_snapshot(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--snapshot-root", type=Path, required=True, help="Legacy data/bronze/stocks directory")
    parser.add_argument(
        "--kinds", nargs="+", choices=("daily_market", "security_master"),
        default=["daily_market", "security_master"],
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without persisting anything")


def _run_migrate_snapshot(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.core.pit import EvidenceKind
    from src.data.snapshot_migration import migrate_snapshot_krx

    runtime = scoped_runtime(args)
    runtime.workspace.initialize()
    reports = migrate_snapshot_krx(
        runtime,
        snapshot_root=Path(args.snapshot_root),
        kinds=tuple(EvidenceKind(kind) for kind in args.kinds),
        dry_run=bool(args.dry_run),
        emit=lambda payload: emit(dict(payload)),
    )
    return {"type": "summary", "reports": [asdict(report) for report in reports]}


def _run_audit(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.dataset_registry import DatasetRegistry
    from src.data.ordinary_universe_price_audit import audit_ordinary_universe_price_availability

    runtime = scoped_runtime(args)
    DatasetRegistry(runtime.workspace.state_root).require("ordinary_universe")
    return asdict(
        audit_ordinary_universe_price_availability(
            universe_root=runtime.workspace.silver_root,
            bronze_root=runtime.workspace.bronze_root,
            artifact_root=runtime.workspace.state_root,
        )
    )


def _run_scope_info(args: argparse.Namespace) -> Mapping[str, object]:
    runtime = scoped_runtime(args)
    return {
        "scope_id": runtime.scope.scope_id,
        "content_hash": runtime.scope.content_hash,
        "bronze_root": str(runtime.workspace.bronze_root),
        "silver_root": str(runtime.workspace.silver_root),
        "gold_root": str(runtime.workspace.gold_root),
        "state_root": str(runtime.workspace.state_root),
        "runs_root": str(runtime.workspace.runs_root),
    }


def _run_init_workspace(args: argparse.Namespace) -> Mapping[str, object]:
    runtime = scoped_runtime(args)
    runtime.workspace.initialize()
    return {"scope_id": runtime.scope.scope_id, "data_root": str(runtime.workspace.root)}


def _run_index(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.bronze_index import index_bronze

    runtime = scoped_runtime(args)
    report = index_bronze(
        runtime,
        dry_run=bool(args.dry_run),
        batch_size=int(args.batch_size),
        emit=lambda payload: emit(dict(payload)),
    )
    return asdict(report)


def _run_verify(args: argparse.Namespace) -> Mapping[str, object]:
    from src.core.pit import PITDataError
    from src.data.dataset_maintenance import verify_datasets

    runtime = scoped_runtime(args)
    try:
        summary = verify_datasets(runtime, verify_all=bool(args.all), emit=lambda payload: emit(dict(payload)))
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=verify_datasets status=failed error=%s", exc)
        raise CommandFailed(
            1, {"type": "summary", "datasets": 0, "failed": 1, "stale": 0, "error": str(exc)}
        ) from exc
    if cast(int, summary.get("failed", 0)):
        raise CommandFailed(1, summary)
    return summary


def _run_prune(args: argparse.Namespace) -> Mapping[str, object]:
    from src.core.pit import PITDataError
    from src.data.dataset_maintenance import prune_datasets
    from src.data.dataset_registry import DatasetRegistry

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    try:
        summary = prune_datasets(runtime, registry, apply=bool(args.apply), emit=lambda payload: emit(dict(payload)))
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=prune_datasets status=failed error=%s", exc)
        raise CommandFailed(
            1,
            {"type": "summary", "candidates": 0, "unreadable": 1, "deleted": 0, "failed": 1, "error": str(exc)},
        ) from exc
    if cast(int, summary.get("unreadable", 0)) or cast(int, summary.get("failed", 0)):
        raise CommandFailed(1, summary)
    return summary


MAINTAIN_COMMANDS: tuple[Command, ...] = (
    Command("audit-ordinary-universe-prices", "Audit raw-price availability for the ordinary-share universe", _add_audit, _run_audit),
    Command("scope-info", "Show resolved scope and workspace roots", _add_audit, _run_scope_info),
    Command("init-workspace", "Create scope-namespaced workspace directories", _add_audit, _run_init_workspace),
    Command("index-bronze", "Register and classify every stored Bronze payload in the catalog", _add_index, _run_index),
    Command(
        "migrate-snapshot-krx", "Re-wrap legacy snapshot KRX session pages as scoped receipts",
        _add_migrate_snapshot, _run_migrate_snapshot,
    ),
    Command("verify-datasets", "Verify current or all scoped datasets", _add_verify, _run_verify),
    Command("prune-datasets", "Plan or apply removal of unreferenced datasets", _add_prune, _run_prune),
)
