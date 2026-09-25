"""PIT dataset foundation CLI."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.data.collection import collect_dart_financial_facts
from src.data.collection_plan import LS_MAX_SESSIONS_PER_REQUEST
from src.data.dataset_registry import DatasetRegistry
from src.data.datasets import dataset_kind_from_id, load_manifest, verify_dataset
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime, load_data_runtime
from src.data.schemas import EvidenceKind, PITDataError
from src.data.scope_coverage import CoverageRequirement
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

load_dotenv()

_LOG = logging.getLogger(__name__)


# 0.35초 간격에서는 전 종목 수집 시 약 10%가 KIS 호출 제한으로 실패했고, 1.0초에서는 전부 성공했다.
_KIS_CLASSIFICATION_PACE_SECONDS = 1.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIT dataset foundation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # normalize-dart-facts --bronze-root ... --silver-root ... --artifact-root ... --decision-time ... --batch-size ...
    # CLI registration: add_argument("normalize-dart-facts") subcommand (via add_parser) with those flags.
    p_dart_refresh = sub.add_parser("normalize-dart-facts", help="Incremental DART fact refresh")
    p_dart_refresh.add_argument("--bronze-root", type=Path, required=False, default=None)
    p_dart_refresh.add_argument("--silver-root", type=Path, required=False, default=None)
    p_dart_refresh.add_argument("--artifact-root", type=Path, required=False, default=None)
    p_dart_refresh.add_argument("--scope-config", type=Path, required=False, default=None)
    p_dart_refresh.add_argument("--data-root", type=Path, required=False, default=None)
    p_dart_refresh.add_argument("--decision-time", type=str, required=True)
    p_dart_refresh.add_argument("--batch-size", type=int, default=500)
    p_dart_refresh.add_argument(
        "--superseded-receipts", type=Path, default=None, help="JSON list of Bronze receipt hashes to exclude as superseded"
    )

    p_ordinary_price_audit = sub.add_parser(
        "audit-ordinary-universe-prices", help="Audit raw-price availability for the ordinary-share universe"
    )
    p_ordinary_price_audit.add_argument("--universe-root", type=Path, default=Path("data/silver/ordinary_universe"))
    p_ordinary_price_audit.add_argument("--bronze-root", type=Path, required=True)
    p_ordinary_price_audit.add_argument("--artifact-root", type=Path, required=True)

    p_scope_info = sub.add_parser("scope-info", help="Show resolved scope and workspace roots")
    _add_scoped_args(p_scope_info)

    p_init_ws = sub.add_parser("init-workspace", help="Create scope-namespaced workspace directories")
    _add_scoped_args(p_init_ws)

    p_collect_scoped = sub.add_parser("collect-scoped", help="Persist scoped raw payloads to Bronze and catalog")
    _add_scoped_args(p_collect_scoped)
    p_collect_scoped.add_argument("--payloads", type=Path, required=True)

    p_plan_scoped = sub.add_parser("plan-scoped", help="Build coverage-driven scoped flow plan under state")
    _add_scoped_args(p_plan_scoped)
    p_plan_scoped.add_argument("--requirements", type=Path, required=True)
    p_plan_scoped.add_argument("--max-sessions", type=int, default=LS_MAX_SESSIONS_PER_REQUEST)

    p_resume_scoped = sub.add_parser("resume-scoped", help="List pending scoped plan chunks")
    _add_scoped_args(p_resume_scoped)
    p_resume_scoped.add_argument("--plan-id", type=str, required=True)

    p_disc_scoped = sub.add_parser("collect-dart-disclosures-scoped", help="Persist scoped DART disclosure pages")
    _add_scoped_args(p_disc_scoped)
    p_disc_scoped.add_argument("--disclosures", type=Path, required=True)
    p_disc_scoped.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_facts_scoped = sub.add_parser("collect-dart-facts-scoped", help="Plan scoped DART fact batch from retained filings")
    _add_scoped_args(p_facts_scoped)
    p_facts_scoped.add_argument("--filings", type=Path, required=True)
    p_facts_scoped.add_argument("--offset", type=int, default=0)
    p_facts_scoped.add_argument("--limit", type=int, default=20)
    p_facts_scoped.add_argument("--execute", action="store_true")
    p_facts_scoped.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_missing_scoped = sub.add_parser(
        "collect-missing-dart-facts-scoped", help="Report scoped DART facts missing filing evidence"
    )
    _add_scoped_args(p_missing_scoped)
    p_missing_scoped.add_argument("--filings", type=Path, required=True)
    p_missing_scoped.add_argument("--offset", type=int, default=0)
    p_missing_scoped.add_argument("--limit", type=int, default=20)

    p_flow_silver = sub.add_parser("build-investor-flow-silver", help="Build Silver LS investor-flow dataset from raw rows")
    _add_scoped_args(p_flow_silver)
    p_flow_silver.add_argument("--universe-dataset-id", default=None)
    p_flow_silver.add_argument("--workers", type=int, default=4)

    p_daily_silver = sub.add_parser("build-daily-market-silver", help="Build Silver daily-market dataset with KRX base prices")
    _add_scoped_args(p_daily_silver)
    p_daily_silver.add_argument("--universe-dataset-id", default=None)

    p_ordinary = sub.add_parser("build-ordinary-universe", help="Build the point-in-time ordinary-share universe")
    _add_scoped_args(p_ordinary)
    p_ordinary.add_argument("--sessions", type=Path, default=None, help="Optional JSON list of requested session dates")

    p_panel = sub.add_parser("build-market-panel", help="Build decision-safe Gold market panel")
    _add_scoped_args(p_panel)
    p_panel.add_argument("--daily-market-dataset-id", default=None)
    p_panel.add_argument("--universe-dataset-id", default=None)
    p_panel.add_argument("--rules", type=Path, default=Path("config/market/krx_market_rules.toml"))
    p_panel.add_argument("--instrument-buckets", type=int, default=16)

    p_bench = sub.add_parser("build-reference-benchmarks", help="Build frictionless Gold reference benchmarks")
    _add_scoped_args(p_bench)
    p_bench.add_argument("--market-panel-dataset-id", default=None)
    p_bench.add_argument("--definitions", type=Path, default=Path("config/data/reference_benchmarks.toml"))

    p_verify_datasets = sub.add_parser("verify-datasets", help="Verify current or all scoped datasets")
    _add_scoped_args(p_verify_datasets)
    p_verify_datasets.add_argument("--all", action="store_true", help="Verify every Silver and Gold dataset")

    p_prune_datasets = sub.add_parser("prune-datasets", help="Plan or apply removal of unreferenced datasets")
    _add_scoped_args(p_prune_datasets)
    p_prune_datasets.add_argument("--apply", action="store_true", help="Delete the revalidated prune set")

    p_kis_backfill = sub.add_parser(
        "backfill-kis-investor-flow-gap", help="Backfill the LS investor-flow gap via KIS pages"
    )
    _add_scoped_args(p_kis_backfill)
    p_kis_backfill.add_argument("--market-panel-dataset-id", default=None)
    p_kis_backfill.add_argument("--ls-flow-dataset-id", default=None)
    p_kis_backfill.add_argument("--pace-seconds", type=float, default=0.35)

    p_kis_supplement = sub.add_parser(
        "build-investor-flow-kis-supplement", help="Build the provider-tagged KIS supplement for the LS gap"
    )
    _add_scoped_args(p_kis_supplement)
    p_kis_supplement.add_argument("--market-panel-dataset-id", default=None)
    p_kis_supplement.add_argument("--ls-flow-dataset-id", default=None)

    p_flow_union = sub.add_parser(
        "build-investor-flow-union", help="Union certified LS flow and its KIS supplement into one dataset"
    )
    _add_scoped_args(p_flow_union)
    p_flow_union.add_argument("--ls-flow-dataset-id", default=None)
    p_flow_union.add_argument("--kis-supplement-dataset-id", default=None)

    p_quality = sub.add_parser(
        "build-financial-quality", help="Build certified financial-quality evidence from facts and quarantine"
    )
    _add_scoped_args(p_quality)
    p_quality.add_argument("--facts-dataset-id", default=None)
    p_quality.add_argument("--quarantine-file", type=Path, default=None)
    p_quality.add_argument("--unresolved-events-file", type=Path, default=None)
    p_quality.add_argument("--decision-time", required=True)

    p_dividend = sub.add_parser(
        "build-dividend-events", help="Build Silver cash-dividend events from dated decision filings"
    )
    _add_scoped_args(p_dividend)

    p_industry = sub.add_parser(
        "collect-industry-classification", help="Collect current KIS industry classifications to Bronze"
    )
    _add_scoped_args(p_industry)
    p_industry.add_argument("--symbols-from", type=Path, required=False, default=None)
    p_industry.add_argument("--pace-seconds", type=float, default=_KIS_CLASSIFICATION_PACE_SECONDS)

    p_stock = sub.add_parser(
        "collect-stock-classification", help="Collect current KIS KSIC stock classifications to Bronze"
    )
    _add_scoped_args(p_stock)
    p_stock.add_argument("--symbols-from", type=Path, required=False, default=None)
    p_stock.add_argument("--pace-seconds", type=float, default=_KIS_CLASSIFICATION_PACE_SECONDS)

    p_industry_silver = sub.add_parser(
        "build-industry-classification-silver", help="Build the certified industry classification Silver snapshot"
    )
    _add_scoped_args(p_industry_silver)

    return parser.parse_args(argv)


def _add_scoped_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)


def _scoped_runtime(args: argparse.Namespace) -> DataRuntime:
    """Load the scoped runtime for scoped commands."""
    return load_data_runtime(scope_config=args.scope_config, data_root=args.data_root)


def _scoped_catalog(runtime: DataRuntime) -> ReceiptCatalog:
    """Open the scope-local receipt catalog under the workspace Bronze root."""
    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def _registered_id(registry: DatasetRegistry, kind: str, explicit: str | None) -> str:
    """Resolve an explicit dataset id or fail closed through the registry."""

    return str(explicit) if explicit is not None else registry.require(kind)


def _resolve_input_id(runtime: DataRuntime, registry: DatasetRegistry, kind: str, explicit: str | None) -> str:
    """Resolve one dataset input exclusively through an explicit id or registry."""

    _ = runtime
    return _registered_id(registry, kind, explicit)


def _register_dataset(runtime: DataRuntime, kind: str, dataset_id: str) -> None:
    DatasetRegistry(runtime.workspace.state_root).register(kind, dataset_id)


def _run_scoped(args: argparse.Namespace, func: Callable[[], dict[str, object]]) -> int:
    """Execute one scoped command body and emit its payload without traceback leaks."""
    try:
        payload = func()
    except (PITDataError, ValueError, OSError) as exc:
        _emit({"error": str(exc)})
        return 1
    _emit(payload)
    return 0


def _dataset_directories(runtime: DataRuntime) -> tuple[Path, ...]:
    """Return flat and one-level table dataset directories in a scope."""

    dataset_name = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*_[0-9a-f]{16}\Z")
    paths: list[Path] = []
    for root in (runtime.workspace.silver_root, runtime.workspace.gold_root):
        if not root.is_dir() or root.is_symlink():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
                continue
            if dataset_name.fullmatch(path.name) or (path / "manifest.json").is_file():
                paths.append(path)
                continue
            paths.extend(
                nested
                for nested in sorted(path.iterdir())
                if nested.is_dir()
                and not nested.is_symlink()
                and (
                    (nested / "manifest.json").is_file()
                    or (nested / "dataset_manifest.json").is_file()
                    or (nested / "content_manifest.json").is_file()
                )
            )
    return tuple(paths)


def _verify_datasets_command(args: argparse.Namespace) -> int:
    """Verify registered datasets, or every dataset with ``--all``."""

    try:
        runtime = _scoped_runtime(args)
        registry = DatasetRegistry(runtime.workspace.state_root)
        current = registry.snapshot()
        if not current:
            raise PITDataError("dataset registry has no current datasets")
        retired_ids = set(registry.retired())
        directories = _dataset_directories(runtime)
        known_ids = {path.name for path in directories} | retired_ids
        targets: list[tuple[str, Path | None]]
        if args.all:
            targets = [(path.name, path) for path in directories]
        else:
            targets = []
            for _kind, dataset_id in sorted(current.items()):
                matches = [path for path in directories if path.name == dataset_id]
                targets.append((dataset_id, matches[0] if len(matches) == 1 else None))
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=verify_datasets status=failed error=%s", exc)
        _emit({"type": "summary", "datasets": 0, "failed": 1, "stale": 0, "error": str(exc)})
        return 1

    registered_ids = set(current.values())
    failed = 0
    stale_count = 0
    for dataset_id, dataset_dir in targets:
        if dataset_dir is None:
            failures = [f"registered dataset directory is missing or ambiguous: {dataset_id}"]
            verification = None
            rows = 0
        else:
            verification = verify_dataset(dataset_dir, known_ids=known_ids.__contains__)
            failures = list(verification.failures)
            rows = verification.rows
        failed_checks = [] if verification is None else [check.name for check in verification.failed_checks]
        stale = _stale_inputs_for_directory(
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
            registered_ids=registered_ids,
            current=current,
            retired_ids=retired_ids,
        )
        stale_count += int(bool(stale))
        status = "failed" if failures else "ok"
        failed += int(bool(failures))
        _emit(
            {
                "type": "dataset",
                "dataset_id": dataset_id,
                "path": None if dataset_dir is None else str(dataset_dir),
                "status": status,
                "rows": rows,
                "failures": failures,
                "failed_checks": failed_checks,
                "stale_inputs": stale,
            }
        )
        _LOG.info(
            "[DATA] command=verify_datasets dataset=%s status=%s stale=%s",
            dataset_id,
            status,
            bool(stale),
        )

    _emit({"type": "summary", "datasets": len(targets), "failed": failed, "stale": stale_count})
    _LOG.info(
        "[DATA] command=verify_datasets action=summary datasets=%d failed=%d stale=%d",
        len(targets),
        failed,
        stale_count,
    )
    return 1 if failed else 0


def _stale_inputs_for_directory(
    *,
    dataset_dir: Path | None,
    dataset_id: str,
    registered_ids: set[str],
    current: Mapping[str, str],
    retired_ids: set[str],
) -> list[dict[str, object]]:
    """Return stale lineage warnings for one verified dataset directory."""

    if dataset_dir is None or dataset_id not in registered_ids:
        return []
    try:
        manifest = load_manifest(dataset_dir)
    except PITDataError:
        return []
    stale: list[dict[str, object]] = []
    for role, input_id in sorted(manifest.inputs.items()):
        if input_id in retired_ids:
            continue
        try:
            input_kind = dataset_kind_from_id(input_id)
        except PITDataError:
            continue
        registered_id = current.get(input_kind)
        if input_id != registered_id:
            stale.append(
                {
                    "role": role,
                    "input_kind": input_kind,
                    "input_id": input_id,
                    "registered_id": registered_id,
                }
            )
    return stale


def _legacy_prunable_manifest(path: Path) -> Mapping[str, object] | None:
    """Recognize a structurally valid pre-v2 dataset manifest for pruning."""

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = path / "dataset_manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if manifest_path.name == "dataset_manifest.json":
        content_path = path / "content_manifest.json"
        try:
            content = json.loads(content_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(content, dict):
            return None
        merged = dict(content)
        merged.update(raw)
        merged["dataset_id"] = path.name
        raw = merged
    if raw.get("dataset_id") != path.name:
        return None
    partitions = raw.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        return None
    if any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in partitions):
        return None
    return raw


def _legacy_lineage(manifest: Mapping[str, object]) -> set[str]:
    values: list[object] = []
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        values.extend(inputs.values())
    for name in (
        "daily_market_dataset_id",
        "universe_dataset_id",
        "ls_dataset_id",
        "kis_supplement_dataset_id",
        "market_panel_dataset_id",
        "source_financial_dataset_id",
    ):
        value = manifest.get(name)
        if isinstance(value, str):
            values.append(value)
    result: set[str] = set()
    for value in values:
        try:
            dataset_kind_from_id(str(value))
        except PITDataError:
            continue
        result.add(str(value))
    return result


def _validated_current_directories(
    runtime: DataRuntime, registry: DatasetRegistry
) -> tuple[dict[str, str], set[str]]:
    """Validate every current pointer before retention can classify anything."""

    current = registry.snapshot()
    if not current:
        raise PITDataError("dataset registry has no current datasets")
    directories = _dataset_directories(runtime)
    by_id: dict[str, list[Path]] = {}
    for directory in directories:
        by_id.setdefault(directory.name, []).append(directory)
    retired_ids = set(registry.retired())
    known_ids = set(by_id) | retired_ids
    for kind, dataset_id in sorted(current.items()):
        matches = by_id.get(dataset_id, [])
        if len(matches) != 1:
            raise PITDataError(f"current dataset is missing or ambiguous: {dataset_id}")
        dataset_dir = matches[0]
        try:
            manifest = load_manifest(dataset_dir)
        except PITDataError as exc:
            raise PITDataError(f"current dataset manifest is invalid: {dataset_id}") from exc
        if manifest.kind != kind:
            raise PITDataError(f"current dataset kind mismatch: {dataset_id}")
        verification = verify_dataset(dataset_dir, known_ids=known_ids.__contains__)
        if not verification.passed:
            raise PITDataError(f"current dataset failed verification: {dataset_id}: {verification.failures}")
    return dict(current), retired_ids


def _prune_plan(runtime: DataRuntime, registry: DatasetRegistry) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return readable orphan paths and unreadable manifests, fail closed."""

    current, _retired_ids = _validated_current_directories(runtime, registry)
    directories = _dataset_directories(runtime)
    graph: dict[str, set[str]] = {}
    unreadable: list[tuple[Path, str]] = []
    unreadable_ids: set[str] = set()
    for dataset_dir in directories:
        try:
            manifest = load_manifest(dataset_dir)
            upstream = graph.setdefault(manifest.dataset_id, set())
            for input_id in manifest.inputs.values():
                try:
                    dataset_kind_from_id(input_id)
                except PITDataError:
                    continue
                upstream.add(input_id)
        except PITDataError as exc:
            legacy = _legacy_prunable_manifest(dataset_dir)
            if legacy is None:
                unreadable.append((dataset_dir, str(exc)))
                unreadable_ids.add(dataset_dir.name)
                continue
            upstream = graph.setdefault(dataset_dir.name, set())
            upstream.update(_legacy_lineage(legacy))

    registered_ids = set(current.values())
    if registered_ids & unreadable_ids:
        return [], sorted(unreadable, key=lambda item: str(item[0]))
    reachable = set(registered_ids)
    pending = list(registered_ids)
    while pending:
        dataset_id = pending.pop()
        for input_id in graph.get(dataset_id, set()):
            if input_id not in reachable:
                reachable.add(input_id)
                pending.append(input_id)

    orphans = [
        dataset_dir
        for dataset_dir in directories
        if dataset_dir.name not in reachable and dataset_dir.name not in unreadable_ids
    ]
    return sorted(orphans, key=lambda path: str(path)), sorted(unreadable, key=lambda item: str(item[0]))


def _prune_datasets_command(args: argparse.Namespace) -> int:
    """List and optionally delete only revalidated unreferenced datasets."""

    try:
        runtime = _scoped_runtime(args)
        registry = DatasetRegistry(runtime.workspace.state_root)
        candidates, unreadable = _prune_plan(runtime, registry)
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=prune_datasets status=failed error=%s", exc)
        _emit({"type": "summary", "candidates": 0, "unreadable": 1, "deleted": 0, "failed": 1, "error": str(exc)})
        return 1

    for dataset_dir in candidates:
        _emit(
            {
                "type": "dataset",
                "dataset_id": dataset_dir.name,
                "path": str(dataset_dir),
                "status": "prunable",
                "action": "delete" if args.apply else "plan",
            }
        )
        _LOG.info("[DATA] command=prune_datasets dataset=%s action=prune", dataset_dir.name)
    for dataset_dir, error in unreadable:
        _emit(
            {
                "type": "dataset",
                "dataset_id": dataset_dir.name,
                "path": str(dataset_dir),
                "status": "unreadable_manifest",
                "action": "keep",
                "error": error,
            }
        )
        _LOG.warning("[DATA] command=prune_datasets dataset=%s action=keep reason=unreadable_manifest", dataset_dir.name)

    deleted = 0
    delete_failed = 0
    if args.apply:
        revalidated, revalidated_unreadable = _prune_plan(runtime, registry)
        unreadable.extend(item for item in revalidated_unreadable if item not in unreadable)
        revalidated_paths = set(revalidated)
        for dataset_dir in candidates:
            if dataset_dir not in revalidated_paths:
                continue
            if dataset_dir.is_symlink() or not dataset_dir.is_dir():
                continue
            try:
                try:
                    load_manifest(dataset_dir)
                except PITDataError:
                    if _legacy_prunable_manifest(dataset_dir) is None:
                        raise
                shutil.rmtree(dataset_dir)
            except (PITDataError, OSError) as exc:
                delete_failed += 1
                _emit(
                    {
                        "type": "dataset",
                        "dataset_id": dataset_dir.name,
                        "path": str(dataset_dir),
                        "status": "delete_failed",
                        "action": "keep",
                        "error": str(exc),
                    }
                )
                _LOG.error(
                    "[DATA] command=prune_datasets dataset=%s action=keep status=failed error=%s",
                    dataset_dir.name,
                    exc,
                )
                continue
            deleted += 1
            _LOG.info("[DATA] command=prune_datasets dataset=%s action=deleted", dataset_dir.name)

    _emit(
        {
            "type": "summary",
            "candidates": len(candidates),
            "unreadable": len(unreadable),
            "applied": bool(args.apply),
            "deleted": deleted,
            "failed": delete_failed,
        }
    )
    _LOG.info(
        "[DATA] command=prune_datasets action=summary candidates=%d deleted=%d failed=%d",
        len(candidates),
        deleted,
        delete_failed,
    )
    return 1 if unreadable or delete_failed else 0


def _eligible_universe_tickers(silver_root: Path) -> tuple[str, ...]:
    """Return eligible tickers from the scope's certified ordinary universe.

    The universe's eligibility classification already excludes preferred
    shares at collection time, so industry collection starting from this set
    never requests a preferred share in the first place.
    """
    import polars as pl

    datasets = sorted(
        path
        for path in Path(silver_root).glob("ordinary_universe_*")
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(datasets) != 1:
        raise PITDataError("ordinary universe requires exactly one published dataset")
    files = sorted(datasets[0].rglob("*.parquet"))
    if not files:
        raise PITDataError("ordinary universe has no published partitions")
    tickers = (
        pl.scan_parquet([str(path) for path in files])
        .filter(pl.col("eligible"))
        .select(pl.col("ticker").cast(pl.String))
        .unique()
        .collect()["ticker"]
        .sort()
        .to_list()
    )
    if not tickers:
        raise PITDataError("ordinary universe has no eligible tickers")
    return tuple(str(ticker) for ticker in tickers)


def _resolve_industry_symbols(runtime: DataRuntime, symbols_from: Path | str | None) -> tuple[str, ...]:
    if symbols_from is not None:
        text = Path(symbols_from).read_text(encoding="utf-8")
        cleaned = tuple(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
        if not cleaned:
            raise PITDataError(f"industry symbols file has no tickers: {symbols_from}")
        return cleaned
    return _eligible_universe_tickers(runtime.workspace.silver_root)


def _collect_classification_with_isolation(
    *,
    stage: str,
    collector_cls: Any,
    fetch_attr: str,
    bronze_root: Path,
    symbols: tuple[str, ...],
    pace_seconds: float,
) -> dict[str, object]:
    """Collect one symbol at a time, isolating per-ticker PIT failures."""
    total = len(symbols)
    pages = 0
    skipped: dict[str, str] = {}
    for index, symbol in enumerate(symbols, start=1):
        collector = collector_cls((symbol,))
        try:
            fetch = getattr(collector, fetch_attr)
            for _ in fetch(bronze_root=bronze_root):
                pages += 1
        except PITDataError as exc:
            skipped[symbol] = str(exc)
        if index % 100 == 0:
            _LOG.info(
                "[DATA] stage=%s done=%d/%d ok=%d skipped=%d",
                stage,
                index,
                total,
                pages,
                len(skipped),
            )
        time.sleep(pace_seconds)
    if pages == 0:
        raise PITDataError(f"{stage} collected nothing ({total} requested, {len(skipped)} skipped)")
    return {
        "symbols_requested": total,
        "pages_collected": pages,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }


def _read_json_list(path: Path, *, label: str) -> list[Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError(f"scoped {label} file is unreadable: {exc}") from exc
    if not isinstance(raw, list):
        raise PITDataError(f"scoped {label} file must hold a list")
    return raw


def _mapping_rows(raw: list[Any], *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise PITDataError(f"scoped {label} entry must be a mapping")
        rows.append(dict(entry))
    return rows


def _read_coverage_requirements(path: Path) -> tuple[CoverageRequirement, ...]:
    """Read coverage requirements from a JSON file."""
    rows = _mapping_rows(_read_json_list(path, label="requirements"), label="requirements")
    items: list[CoverageRequirement] = []
    for row in rows:
        as_of_raw = row.get("as_of")
        items.append(
            CoverageRequirement(
                source=str(row.get("source") or ""),
                natural_key=str(row.get("natural_key") or ""),
                as_of=date.fromisoformat(str(as_of_raw)) if as_of_raw else None,
                fiscal_period=str(row.get("fiscal_period") or "") or None,
                required=bool(row.get("required", True)),
            )
        )
    return tuple(items)


def _read_scoped_payloads(path: Path) -> tuple[ScopedRawPayload, ...]:
    """Read scoped raw payloads from a JSON file with base64-encoded bodies."""
    rows = _mapping_rows(_read_json_list(path, label="payloads"), label="payloads")
    items: list[ScopedRawPayload] = []
    for row in rows:
        as_of_raw = row.get("as_of")
        items.append(
            ScopedRawPayload(
                kind=EvidenceKind(str(row.get("kind") or "")),
                source=str(row.get("source") or ""),
                natural_key=str(row.get("natural_key") or ""),
                as_of=date.fromisoformat(str(as_of_raw)) if as_of_raw else None,
                fiscal_period=str(row.get("fiscal_period") or "") or None,
                status=EvidenceStatus(str(row.get("status") or "")),
                payload=base64.b64decode(str(row.get("payload_b64") or "")),
                retrieved_at=datetime.fromisoformat(str(row.get("retrieved_at") or "")),
                source_label=str(row.get("source_label") or ""),
            )
        )
    return tuple(items)


def _build_dart_fact_batch_artifact(args: argparse.Namespace) -> dict[str, object]:
    """Plan one quota-bounded DART fact batch and persist its artifact under state."""
    from src.data.collection_plan import scoped_plan_dir
    from src.data.dart_backfill import build_scoped_dart_collector, build_scoped_dart_fact_batch

    runtime = _scoped_runtime(args)
    rows = _mapping_rows(_read_json_list(Path(args.filings), label="filings"), label="filings")
    batch = build_scoped_dart_fact_batch(
        runtime=runtime,
        catalog=_scoped_catalog(runtime),
        filing_identities=[
            {str(key): str(value) for key, value in row.items() if value is not None} for row in rows
        ],
        offset=int(args.offset),
        limit=int(args.limit),
    )
    out_dir = scoped_plan_dir(runtime=runtime) / "dart-fact-batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{batch.plan_id}.json").write_text(
        json.dumps(
            {
                "plan_id": batch.plan_id,
                "scope_hash": batch.scope_hash,
                "identities": [dict(item) for item in batch.identities],
                "missing_without_filing": [
                    {
                        "source": item.source,
                        "natural_key": item.natural_key,
                        "fiscal_period": item.fiscal_period,
                    }
                    for item in batch.missing_without_filing
                ],
                "estimated_request_ceiling": batch.estimated_request_ceiling,
                "available_request_headroom": batch.available_request_headroom,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output: dict[str, object] = {
        "plan_id": batch.plan_id,
        "selected": len(batch.identities),
        "missing_without_filing": len(batch.missing_without_filing),
        "estimated_request_ceiling": batch.estimated_request_ceiling,
        "available_request_headroom": batch.available_request_headroom,
    }
    if bool(getattr(args, "execute", False)) and batch.identities:
        collection = collect_dart_financial_facts(
            dart=build_scoped_dart_collector(runtime=runtime),
            identities=tuple(dict(item) for item in batch.identities),
            bronze_root=runtime.workspace.bronze_root,
            retrieved_at=_parse_dt(args.retrieved_at),
        )
        output["content_hash"] = collection.content_hash
    return output


def _parse_dt(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _parse_decision_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    return parsed


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _parse_quality_decision_time(value: str) -> datetime:
    """Parse a scoped decision time without coercing naive inputs to UTC."""
    from src.data.schemas import PITDataError

    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise PITDataError(f"decision-time must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise PITDataError("decision-time must be timezone-aware")
    return parsed


def _read_quality_json_array(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Read a JSON array of mappings, failing closed on any shape violation."""
    from src.data.schemas import PITDataError

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError(f"scoped {label} file is unreadable: {exc}") from exc
    if not isinstance(raw, list):
        raise PITDataError(f"scoped {label} file must hold a list")
    rows: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise PITDataError(f"scoped {label} entry must be a mapping")
        rows.append(dict(entry))
    return rows


def _parse_quality_event_timestamp(value: Any, *, label: str) -> datetime:
    """Parse one evidenced event timestamp without defaulting naive inputs."""
    from src.data.schemas import PITDataError

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise PITDataError(f"scoped unresolved events entry has an invalid {label}: {value!r}") from exc
    else:
        raise PITDataError(f"scoped unresolved events entry lacks {label}")
    if parsed.tzinfo is None:
        raise PITDataError(f"scoped unresolved events entry has a naive {label}")
    return parsed


def _manual_quality_events(path: Path) -> list[Any]:
    """Load manually evidenced events; every entry must carry an explicit reason."""
    from src.data.financial_quality import FinancialQualityEvent
    from src.data.schemas import PITDataError

    events: list[Any] = []
    for index, entry in enumerate(_read_quality_json_array(path, label="unresolved events")):
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise PITDataError(f"scoped unresolved events entry {index} lacks reason")
        company_id = str(entry.get("company_id") or "").strip()
        fiscal_period = str(entry.get("fiscal_period") or "").strip()
        filing_id = str(entry.get("filing_id") or "").strip()
        if not company_id:
            raise PITDataError(f"scoped unresolved events entry {index} lacks company_id")
        if not filing_id:
            raise PITDataError(f"scoped unresolved events entry {index} lacks filing_id")
        try:
            events.append(
                FinancialQualityEvent(
                    company_id=company_id,
                    fiscal_period=fiscal_period,
                    filing_id=filing_id,
                    published_at=_parse_quality_event_timestamp(entry.get("published_at"), label="published_at"),
                    available_at=_parse_quality_event_timestamp(entry.get("available_at"), label="available_at"),
                    reason=reason.strip(),
                )
            )
        except ValueError as exc:
            raise PITDataError(str(exc)) from exc
    return events


def _build_financial_quality(args: argparse.Namespace) -> dict[str, object]:
    """Build one certified financial-quality dataset from facts and evidenced events."""
    import hashlib

    import polars as pl

    from src.data.datasets import dataset_digest, read_dataset
    from src.data.financial_quality import (
        build_financial_quality_events,
        materialize_financial_quality,
        quarantine_events,
    )

    runtime = _scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    facts_dataset_id = _registered_id(registry, "financial_facts", args.facts_dataset_id)
    decision_time = _parse_quality_decision_time(str(args.decision_time))
    facts_path = runtime.workspace.silver_root / facts_dataset_id
    try:
        facts = read_dataset(facts_path).collect()
    except PITDataError as exc:
        raise PITDataError(f"financial facts dataset is not a verified v2 dataset: {facts_path}") from exc
    quarantine_records = (
        _read_quality_json_array(Path(args.quarantine_file), label="quarantine")
        if args.quarantine_file is not None
        else []
    )
    quarantined = quarantine_events(quarantine_records)
    manual = (
        _manual_quality_events(Path(args.unresolved_events_file))
        if args.unresolved_events_file is not None
        else []
    )
    events = build_financial_quality_events(
        facts,
        unresolved_events=(*quarantined, *manual),
        decision_time=decision_time,
    )
    quarantine_digest = dataset_digest(
        [hashlib.sha256(Path(args.quarantine_file).read_bytes()).hexdigest()]
    ) if args.quarantine_file is not None else dataset_digest([])
    unresolved_digest = dataset_digest(
        [hashlib.sha256(Path(args.unresolved_events_file).read_bytes()).hexdigest()]
    ) if args.unresolved_events_file is not None else dataset_digest([])
    dataset_dir = materialize_financial_quality(
        events,
        layer_root=runtime.workspace.silver_root,
        decision_time=decision_time,
        facts_dataset_id=facts_dataset_id,
        quarantine_digest=quarantine_digest,
        unresolved_events_digest=unresolved_digest,
    )
    dataset_id = dataset_dir.name
    _register_dataset(runtime, "financial_quality", dataset_id)
    incomplete_periods = (
        events.filter(~pl.col("financial_complete"))
        .select("company_id", "fiscal_period")
        .unique()
        .height
    )
    return {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_dir),
        "rows": events.height,
        "quarantine_events": len(quarantined),
        "manual_events": len(manual),
        "incomplete_periods": incomplete_periods,
    }


def normalize_dart_facts(
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    batch_size: int = 500,
    superseded_receipts: Path | None = None,
    disclosures_dataset_id: str | None = None,
    financial_facts_dataset_id: str | None = None,
) -> dict[str, object]:
    """Incremental DART fact refresh entry point for the normalize-dart-facts command."""
    from src.core.krx_calendar import xkrx_session_calendar
    from src.data.incremental_normalization import refresh_dart_financial_facts

    artifact = refresh_dart_financial_facts(
        bronze_root=Path(bronze_root),
        silver_root=Path(silver_root),
        artifact_root=Path(artifact_root),
        decision_time=decision_time,
        calendar=xkrx_session_calendar(),
        batch_size=int(batch_size),
        superseded_receipt_hashes=frozenset(str(h) for h in json.loads(Path(superseded_receipts).read_text(encoding="utf-8")))
        if superseded_receipts is not None
        else frozenset(),
        disclosures_dataset_id=disclosures_dataset_id,
        financial_facts_dataset_id=financial_facts_dataset_id,
    )
    return {"output_hash": artifact.output_hash, "report_hash": artifact.report_hash, "row_count": artifact.row_count, "quarantined_filings": artifact.quarantined_filings, "quarantine_path": artifact.quarantine_path, "dataset_path": artifact.dataset_path}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "scope-info":
        from src.data.runtime import load_data_runtime

        runtime = load_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
        _emit(
            {
                "scope_id": runtime.scope.scope_id,
                "content_hash": runtime.scope.content_hash,
                "bronze_root": str(runtime.workspace.bronze_root),
                "silver_root": str(runtime.workspace.silver_root),
                "gold_root": str(runtime.workspace.gold_root),
                "state_root": str(runtime.workspace.state_root),
                "runs_root": str(runtime.workspace.runs_root),
            }
        )
        return 0
    if args.command == "init-workspace":
        from src.data.runtime import load_data_runtime

        runtime = load_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
        runtime.workspace.initialize()
        _emit({"scope_id": runtime.scope.scope_id, "data_root": str(runtime.workspace.root)})
        return 0
    if args.command == "collect-scoped":
        def _collect_scoped() -> dict[str, object]:
            runtime = _scoped_runtime(args)
            writer = ScopedBronzeWriter(runtime=runtime, catalog=_scoped_catalog(runtime))
            count = 0
            content_hash = ""
            for scoped_payload in _read_scoped_payloads(Path(args.payloads)):
                receipt = writer.persist(scoped_payload)
                count += 1
                content_hash = receipt.bronze_receipt.content_hash
            return {"scope_id": runtime.scope.scope_id, "receipts": count, "content_hash": content_hash}

        return _run_scoped(args, _collect_scoped)
    if args.command == "plan-scoped":
        def _plan_scoped() -> dict[str, object]:
            from src.data.collection_plan import build_scoped_flow_plan
            from src.data.scope_coverage import build_scope_coverage_report

            runtime = _scoped_runtime(args)
            report = build_scope_coverage_report(
                scope=runtime.scope,
                requirements=_read_coverage_requirements(Path(args.requirements)),
                catalog=_scoped_catalog(runtime),
            )
            plan = build_scoped_flow_plan(
                runtime=runtime, report=report, max_sessions_per_request=int(args.max_sessions)
            )
            return {"plan_id": plan.plan_id, "chunks": len(plan.chunks), "scope_hash": report.scope_hash}

        return _run_scoped(args, _plan_scoped)
    if args.command == "resume-scoped":
        def _resume_scoped() -> dict[str, object]:
            from src.data.collection_plan import (
                CollectionCheckpointStore,
                load_collection_plan,
                scoped_checkpoint_dir,
                scoped_plan_dir,
            )

            runtime = _scoped_runtime(args)
            plan = load_collection_plan(str(args.plan_id), artifact_root=scoped_plan_dir(runtime=runtime))
            store = CollectionCheckpointStore(scoped_checkpoint_dir(runtime=runtime))
            pending = [
                chunk.chunk_id
                for chunk in plan.chunks
                if not store.has_verified_receipt(
                    plan=plan, chunk=chunk, bronze_root=runtime.workspace.bronze_root
                )
            ]
            return {"plan_id": plan.plan_id, "pending": pending}

        return _run_scoped(args, _resume_scoped)
    if args.command == "collect-dart-disclosures-scoped":
        def _disclosures_scoped() -> dict[str, object]:
            from src.data.schemas import EvidenceKind as _EvidenceKind

            runtime = _scoped_runtime(args)
            writer = ScopedBronzeWriter(runtime=runtime, catalog=_scoped_catalog(runtime))
            rows = _mapping_rows(_read_json_list(Path(args.disclosures), label="disclosures"), label="disclosures")
            retrieved_at = _parse_dt(args.retrieved_at)
            count = 0
            for row in rows:
                key = str(row.get("rcept_no") or row.get("filing_id") or "").strip()
                if not key:
                    raise PITDataError("DART disclosure page is missing its adapter natural key")
                as_of_raw = str(row.get("published_at") or row.get("coverage_date") or "").strip()
                writer.persist(
                    ScopedRawPayload(
                        kind=_EvidenceKind.DISCLOSURES,
                        source="dart_disclosures",
                        natural_key=key,
                        as_of=date.fromisoformat(as_of_raw[:10]) if as_of_raw else None,
                        fiscal_period=None,
                        status=EvidenceStatus.SUCCESS,
                        payload=json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"),
                        retrieved_at=retrieved_at,
                        source_label=f"opendart:list:{key}",
                    )
                )
                count += 1
            return {"scope_id": runtime.scope.scope_id, "receipts": count}

        return _run_scoped(args, _disclosures_scoped)
    if args.command == "collect-dart-facts-scoped":
        def _facts_scoped() -> dict[str, object]:
            return _build_dart_fact_batch_artifact(args)

        return _run_scoped(args, _facts_scoped)
    if args.command == "collect-missing-dart-facts-scoped":
        def _missing_scoped() -> dict[str, object]:
            summary = _build_dart_fact_batch_artifact(args)
            return {
                "plan_id": summary["plan_id"],
                "missing_without_filing": summary["missing_without_filing"],
                "selected": summary["selected"],
            }

        return _run_scoped(args, _missing_scoped)
    if args.command == "build-ordinary-universe":
        def _build_ordinary() -> dict[str, object]:
            from src.data.ordinary_universe import (
                catalog_master_sessions,
                materialize_ordinary_universe_from_catalog,
            )

            runtime = _scoped_runtime(args)
            if args.sessions is None:
                sessions = catalog_master_sessions(_scoped_catalog(runtime))
            else:
                raw_sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
                if not isinstance(raw_sessions, list):
                    raise PITDataError("ordinary universe sessions must be a JSON list")
                sessions = tuple(date.fromisoformat(str(value)[:10]) for value in raw_sessions)
            path = materialize_ordinary_universe_from_catalog(
                catalog=_scoped_catalog(runtime),
                sessions=sessions,
                silver_root=runtime.workspace.silver_root,
            )
            _register_dataset(runtime, "ordinary_universe", path.name)
            return {"dataset_id": path.name, "dataset_path": str(path), "sessions": len(sessions)}

        return _run_scoped(args, _build_ordinary)
    if args.command == "build-investor-flow-silver":
        def _build_flow_silver() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.investor_flow_silver import materialize_investor_flow_silver

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            universe_id = _resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
            result = materialize_investor_flow_silver(
                bronze_root=runtime.workspace.bronze_root,
                universe_root=runtime.workspace.silver_root,
                silver_root=runtime.workspace.silver_root,
                workers=args.workers,
                universe_dataset_id=universe_id,
            )
            _register_dataset(runtime, "investor_flow_ls", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_flow_silver)
    if args.command == "build-daily-market-silver":
        def _build_daily_silver() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.daily_market_silver import materialize_daily_market_silver

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            universe_id = _resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
            result = materialize_daily_market_silver(
                catalog=_scoped_catalog(runtime),
                universe_root=runtime.workspace.silver_root,
                silver_root=runtime.workspace.silver_root,
                universe_dataset_id=universe_id,
            )
            _register_dataset(runtime, "daily_market", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_daily_silver)
    if args.command == "build-market-panel":
        def _build_market_panel() -> dict[str, object]:
            from dataclasses import asdict

            from src.core.market_rules import load_krx_market_rules
            from src.data.market_panel import materialize_market_panel

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            daily_id = _registered_id(registry, "daily_market", args.daily_market_dataset_id)
            universe_id = _resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
            result = materialize_market_panel(
                daily_market_path=runtime.workspace.silver_root / daily_id,
                universe_path=runtime.workspace.silver_root / universe_id,
                rules=load_krx_market_rules(args.rules),
                gold_root=runtime.workspace.gold_root,
                instrument_buckets=args.instrument_buckets,
            )
            _register_dataset(runtime, "market_panel", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path), "daily_market_dataset_id": daily_id, "universe_dataset_id": universe_id}

        return _run_scoped(args, _build_market_panel)
    if args.command == "build-reference-benchmarks":
        def _build_reference_benchmarks() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.reference_benchmarks import load_benchmark_definitions, materialize_reference_benchmarks

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            panel_id = _registered_id(registry, "market_panel", args.market_panel_dataset_id)
            version, definitions = load_benchmark_definitions(args.definitions)
            result = materialize_reference_benchmarks(
                market_panel_path=runtime.workspace.gold_root / panel_id,
                definitions=definitions,
                definitions_version=version,
                gold_root=runtime.workspace.gold_root,
            )
            _register_dataset(runtime, "reference_benchmarks", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path), "market_panel_dataset_id": panel_id}

        return _run_scoped(args, _build_reference_benchmarks)
    if args.command == "backfill-kis-investor-flow-gap":
        def _backfill_kis_gap() -> dict[str, object]:
            from src.data.investor_flow_gap import compute_missing_investor_flow_cells
            from src.integrations.kis.investor_flow import KisInvestorFlowCollector

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            panel_id = _registered_id(registry, "market_panel", args.market_panel_dataset_id)
            ls_id = _registered_id(registry, "investor_flow_ls", args.ls_flow_dataset_id)
            gap = compute_missing_investor_flow_cells(
                market_panel_path=runtime.workspace.gold_root / panel_id,
                ls_flow_silver_path=runtime.workspace.silver_root / ls_id,
            )
            if not gap.symbols:
                return {"target_cells": 0, "symbols_attempted": 0}
            collector = KisInvestorFlowCollector(tuple(s.ticker for s in gap.symbols))
            filled = attempted = 0
            for entry in gap.symbols:
                for _ in collector.fetch_investor_flow(
                    entry.sessions[0], entry.sessions[-1],
                    bronze_root=runtime.workspace.bronze_root,
                    symbols=(entry.ticker,),
                ):
                    filled += 1
                attempted += 1
                _LOG.info("[DATA] stage=kis_gap_backfill symbol=%s %d/%d", entry.ticker, attempted, len(gap.symbols))
                time.sleep(args.pace_seconds)
            return {"target_cells": gap.total_cells, "symbols_attempted": attempted}

        return _run_scoped(args, _backfill_kis_gap)
    if args.command == "build-investor-flow-kis-supplement":
        def _build_kis_supplement() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            panel_id = _registered_id(registry, "market_panel", args.market_panel_dataset_id)
            ls_id = _registered_id(registry, "investor_flow_ls", args.ls_flow_dataset_id)
            result = materialize_investor_flow_kis_supplement(
                bronze_root=runtime.workspace.bronze_root,
                market_panel_path=runtime.workspace.gold_root / panel_id,
                ls_flow_silver_path=runtime.workspace.silver_root / ls_id,
                silver_root=runtime.workspace.silver_root,
            )
            _register_dataset(runtime, "investor_flow_kis_supplement", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path), "market_panel_dataset_id": panel_id, "ls_dataset_id": ls_id}

        return _run_scoped(args, _build_kis_supplement)
    if args.command == "build-investor-flow-union":
        def _build_flow_union() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.investor_flow_union import materialize_investor_flow_union

            runtime = _scoped_runtime(args)
            registry = DatasetRegistry(runtime.workspace.state_root)
            ls_id = _registered_id(registry, "investor_flow_ls", args.ls_flow_dataset_id)
            kis_id = _registered_id(registry, "investor_flow_kis_supplement", args.kis_supplement_dataset_id)
            result = materialize_investor_flow_union(
                ls_flow_silver_path=runtime.workspace.silver_root / ls_id,
                kis_supplement_silver_path=runtime.workspace.silver_root / kis_id,
                silver_root=runtime.workspace.silver_root,
            )
            _register_dataset(runtime, "investor_flow", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_flow_union)
    if args.command == "build-financial-quality":
        def _build_quality() -> dict[str, object]:
            return _build_financial_quality(args)

        return _run_scoped(args, _build_quality)
    if args.command == "collect-industry-classification":
        def _collect_industry() -> dict[str, object]:
            from src.integrations.kis.industry import KisIndustryCollector

            runtime = _scoped_runtime(args)
            symbols = _resolve_industry_symbols(runtime, args.symbols_from)
            return _collect_classification_with_isolation(
                stage="collect-industry-classification",
                collector_cls=KisIndustryCollector,
                fetch_attr="fetch_industry_classification",
                bronze_root=runtime.workspace.bronze_root,
                symbols=symbols,
                pace_seconds=float(args.pace_seconds),
            )

        return _run_scoped(args, _collect_industry)
    if args.command == "collect-stock-classification":
        def _collect_stock() -> dict[str, object]:
            from src.integrations.kis.industry import KisStockClassificationCollector

            runtime = _scoped_runtime(args)
            symbols = _resolve_industry_symbols(runtime, args.symbols_from)
            return _collect_classification_with_isolation(
                stage="collect-stock-classification",
                collector_cls=KisStockClassificationCollector,
                fetch_attr="fetch_stock_classification",
                bronze_root=runtime.workspace.bronze_root,
                symbols=symbols,
                pace_seconds=float(args.pace_seconds),
            )

        return _run_scoped(args, _collect_stock)
    if args.command == "build-industry-classification-silver":
        def _build_industry_silver() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.industry_silver import materialize_industry_classification_silver

            runtime = _scoped_runtime(args)
            result = materialize_industry_classification_silver(
                bronze_root=runtime.workspace.bronze_root,
                silver_root=runtime.workspace.silver_root,
            )
            _register_dataset(runtime, "industry", result.dataset_id)
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_industry_silver)
    if args.command == "build-dividend-events":
        def _build_dividends() -> dict[str, object]:
            from src.core.krx_calendar import xkrx_session_calendar
            from src.data.dividend_events import materialize_dividend_events

            runtime = _scoped_runtime(args)
            path = materialize_dividend_events(
                bronze_root=runtime.workspace.bronze_root,
                universe_root=runtime.workspace.silver_root,
                silver_root=runtime.workspace.silver_root,
                calendar=xkrx_session_calendar(),
            )
            _register_dataset(runtime, "dividend_events", path.name)
            return {"dataset_id": path.name, "dataset_path": str(path)}

        return _run_scoped(args, _build_dividends)
    if args.command == "verify-datasets":
        return _verify_datasets_command(args)
    if args.command == "prune-datasets":
        return _prune_datasets_command(args)
    if args.command == "normalize-dart-facts":
        try:
            normalize_runtime: DataRuntime | None = None
            if args.scope_config is not None or args.data_root is not None:
                if args.scope_config is None or args.data_root is None:
                    raise PITDataError("normalize-dart-facts scope-config and data-root must be supplied together")
                normalize_runtime = _scoped_runtime(args)
            if normalize_runtime is None:
                if any(value is None for value in (args.bronze_root, args.silver_root, args.artifact_root)):
                    raise PITDataError("normalize-dart-facts requires explicit roots or scoped runtime arguments")
                bronze_root = Path(args.bronze_root)
                silver_root = Path(args.silver_root)
                artifact_root = Path(args.artifact_root)
            else:
                bronze_root = Path(args.bronze_root) if args.bronze_root is not None else normalize_runtime.workspace.bronze_root
                silver_root = Path(args.silver_root) if args.silver_root is not None else normalize_runtime.workspace.silver_root
                artifact_root = Path(args.artifact_root) if args.artifact_root is not None else normalize_runtime.workspace.root / "artifacts" / normalize_runtime.scope.scope_id
            normalize_registry = (
                DatasetRegistry(normalize_runtime.workspace.state_root)
                if normalize_runtime is not None
                else None
            )
            disclosures_dataset_id = (
                normalize_registry.current("disclosures")
                if normalize_registry is not None
                else None
            )
            financial_facts_dataset_id = (
                normalize_registry.current("financial_facts")
                if normalize_registry is not None
                else None
            )
            payload = normalize_dart_facts(
                bronze_root=bronze_root,
                silver_root=silver_root,
                artifact_root=artifact_root,
                decision_time=_parse_decision_time(args.decision_time),
                batch_size=int(getattr(args, "batch_size", 500)),
                superseded_receipts=getattr(args, "superseded_receipts", None),
                disclosures_dataset_id=disclosures_dataset_id,
                financial_facts_dataset_id=financial_facts_dataset_id,
            )
            dataset_id = Path(str(payload["dataset_path"])).name
            if normalize_runtime is not None:
                DatasetRegistry(normalize_runtime.workspace.state_root).register("financial_facts", dataset_id)
            else:
                resolved_silver = silver_root.resolve()
                if resolved_silver.parent.name == "silver":
                    data_root = resolved_silver.parent.parent
                    DatasetRegistry(data_root / "state" / resolved_silver.name).register(
                        "financial_facts", dataset_id
                    )
                else:
                    data_root = resolved_silver.parent
                    DatasetRegistry(
                        data_root / "state" / "unscoped",
                        data_root=data_root,
                        scope_id="",
                    ).register("financial_facts", dataset_id)
        except (PITDataError, ValueError, OSError) as exc:
            _emit({"error": str(exc)})
            return 1
        _emit({"output_hash": payload["output_hash"], "report_hash": payload["report_hash"], "row_count": payload["row_count"], "quarantined_filings": payload["quarantined_filings"], "quarantine_path": payload["quarantine_path"], "dataset_id": dataset_id, "dataset_path": payload["dataset_path"]})
        return 0
    if args.command == "audit-ordinary-universe-prices":
        try:
            from src.data.ordinary_universe_price_audit import audit_ordinary_universe_price_availability

            ordinary_price_audit = audit_ordinary_universe_price_availability(
                universe_root=Path(args.universe_root),
                bronze_root=Path(args.bronze_root),
                artifact_root=Path(args.artifact_root),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        from dataclasses import asdict

        _emit(asdict(ordinary_price_audit))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
