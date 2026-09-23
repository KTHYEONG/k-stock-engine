"""PIT dataset foundation CLI."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from src.core.time import SessionCalendar
from src.data.backtest_exclusions import resolve_backtest_exclusion_plan
from src.data.backtest_runner import run_champion_backtest
from src.data.backtest_sessions import BacktestMarketInputsPolicy, build_backtest_sessions
from src.data.bronze import BronzeStore, import_retained_stock_evidence, migrate_retained_stock_evidence
from src.data.collection import (
    InvestorFlowBatchProgress,
    collect_dart_disclosures,
    collect_dart_financial_facts,
    collect_planned_investor_flow,
    iter_planned_investor_flow_backfill,
)
from src.data.collection_plan import (
    LS_MAX_SESSIONS_PER_REQUEST,
    CollectionCheckpointStore,
    CollectionReadinessReport,
    build_historical_collection_plan_from_bronze,
    load_collection_plan,
    load_collection_plan_path,
)
from src.data.gold_informativeness import CHAMPION_SCORE_COVERAGE_FLOORS, certify_informative_gold
from src.data.gold_loader import (
    load_gold_window_inputs,
    parse_silver_dataset_bindings,
    resolve_gold_dataset_bindings,
    write_gold_input_binding_artifact,
)
from src.data.legacy_inventory import MigrationArtifactStore, inspect_legacy_data, plan_bronze_retention
from src.data.master_intervals import compact_security_master_intervals
from src.data.operations import execute_verified_legacy_purge
from src.data.pipeline import materialize_backtest_inputs
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime, load_data_runtime
from src.data.schemas import EvidenceKind, PITDataError, SilverTable
from src.data.scope_coverage import CoverageRequirement
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.data.silver import load_latest_silver_market_scan, load_latest_silver_table
from src.data.silver_schema import canonicalize_session_keys, observe_time_semantics
from src.data.storage_gc import plan_storage_root_retention
from src.data.streaming_normalization import refresh_corporate_action_silver
from src.integrations.investor_flow_router import resolve_investor_flow_collector
from src.integrations.ls.investor_flow import LsInvestorFlowCollector
from src.strategy.champion_strategy import ChampionStrategy
from src.strategy.compounding_strategy import CompoundingStrategy
from src.strategy.compounding_v2_strategy import (
    CompoundingV2Policy,
    CompoundingV2Strategy,
    summarize_compounding_v2_selection_shortfalls,
)
from src.strategy.core_strategy import CoreStrategy

load_dotenv()

_LOG = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIT dataset foundation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser("inventory", help="Classify legacy paths")
    p_inv.add_argument("--data-root", type=Path, default=Path("data"))

    p_mig = sub.add_parser("migrate-legacy", help="Migrate retained evidence to Bronze")
    p_mig.add_argument("--source-root", type=Path, default=Path("data/evidence/stocks"))
    p_mig.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_mig.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_mig.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_col = sub.add_parser("collect", help="Collect missing champion evidence")
    p_col.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_col.add_argument("--retrieved-at", type=str, required=False, default=None)
    p_col.add_argument("--plan-id", type=str, required=True)
    p_col.add_argument("--checkpoint-root", type=Path, default=Path("data/artifacts/collection-checkpoints"))
    p_col.add_argument("--investor-flow-provider", choices=("ls", "kiwoom"), default="ls")

    p_ls_backfill = sub.add_parser(
        "backfill-ls-investor-flow", help="Resumable single-session LS investor-flow backfill from a plan path"
    )
    p_ls_backfill.add_argument("--plan-path", type=Path, required=True)
    p_ls_backfill.add_argument("--bronze-root", type=Path, required=True)
    p_ls_backfill.add_argument("--checkpoint-root", type=Path, required=True)
    p_ls_backfill.add_argument("--chunk-batch-size", type=int, default=100)
    p_ls_backfill.add_argument("--retrieved-at", type=str, required=False, default=None)
    p_ls_backfill.add_argument("--max-batches", type=int, required=False, default=None)

    p_dart = sub.add_parser("collect-dart-facts", help="Collect periodic OpenDART full statements from retained disclosures")
    p_dart.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_dart.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_dart.add_argument("--coverage-start", type=str, required=True)
    p_dart.add_argument("--coverage-end", type=str, required=True)
    p_dart.add_argument("--offset", type=int, default=0)
    p_dart.add_argument("--limit", type=int, default=20)
    p_dart.add_argument("--corp-code", type=str, default=None)
    p_dart.add_argument("--filing-id", type=str, default=None)
    p_dart.add_argument("--biz-year", type=str, default=None)
    p_dart.add_argument("--report-code", type=str, default=None)
    p_dart.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_disc = sub.add_parser("collect-dart-disclosures", help="Collect DART disclosures to Bronze")
    p_disc.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_disc.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_disc.add_argument("--coverage-start", type=str, required=True)
    p_disc.add_argument("--coverage-end", type=str, required=True)
    p_disc.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_backfill = sub.add_parser("backfill-dart-facts", help="PIT-safe DART historical fact backfill")
    p_backfill.add_argument("--backfill-dart-facts", action="store_true", default=False)
    p_backfill.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_backfill.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_backfill.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_backfill.add_argument("--validation-start", type=str, required=True)
    p_backfill.add_argument("--validation-end", type=str, required=True)
    p_backfill.add_argument("--offset", type=int, default=0)
    p_backfill.add_argument("--limit", type=int, default=100)
    p_backfill.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_missing_dart = sub.add_parser("collect-missing-dart-facts", help="Collect retained DART filing identities missing fact evidence")
    p_missing_dart.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_missing_dart.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_missing_dart.add_argument("--backfill-artifact", type=Path, required=True)
    p_missing_dart.add_argument("--offset", type=int, default=0)
    p_missing_dart.add_argument("--limit", type=int, default=20)
    p_missing_dart.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_mat = sub.add_parser("materialize", help="Materialize backtest inputs")
    p_mat.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_mat.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_mat.add_argument("--gold-root", type=Path, default=Path("data/gold/stocks"))
    p_mat.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_mat.add_argument("--decision-time", type=str, required=True)

    p_norm = sub.add_parser("normalize", help="Validate/normalize Bronze evidence")
    p_norm.add_argument("--bronze-root", type=Path, required=True)
    p_norm.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_norm.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_norm.add_argument("--decision-time", type=str, required=True)
    p_norm.add_argument("--batch-size", type=int, default=50000)

    p_actions = sub.add_parser("refresh-corporate-actions", help="Refresh only corporate-action Silver evidence")
    p_actions.add_argument("--bronze-root", type=Path, required=True)
    p_actions.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_actions.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_actions.add_argument("--decision-time", type=str, required=True)

    # normalize-dart-facts --bronze-root ... --silver-root ... --artifact-root ... --decision-time ... --batch-size ...
    # CLI registration: add_argument("normalize-dart-facts") subcommand (via add_parser) with those flags.
    p_dart_refresh = sub.add_parser("normalize-dart-facts", help="Incremental DART fact refresh")
    p_dart_refresh.add_argument("--bronze-root", type=Path, required=True)
    p_dart_refresh.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_dart_refresh.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_dart_refresh.add_argument("--decision-time", type=str, required=True)
    p_dart_refresh.add_argument("--batch-size", type=int, default=500)

    p_retention = sub.add_parser("bronze-retention-plan", help="Audit Bronze retention without deletion")
    p_retention.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_retention.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_retention.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))

    p_sg_retention = sub.add_parser("silver-gold-retention-plan", help="Audit Silver/Gold storage-root retention without deletion")
    p_sg_retention.add_argument("--silver-base", type=Path, default=Path("data/silver"))
    p_sg_retention.add_argument("--gold-base", type=Path, default=Path("data/gold"))
    p_sg_retention.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))

    p_ordinary_price_audit = sub.add_parser(
        "audit-ordinary-universe-prices", help="Audit raw-price availability for the ordinary-share universe"
    )
    p_ordinary_price_audit.add_argument("--universe-root", type=Path, default=Path("data/silver/ordinary_universe"))
    p_ordinary_price_audit.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_ordinary_price_audit.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))

    p_purge = sub.add_parser("purge-legacy", help="Purge legacy outputs after verification")
    p_purge.add_argument("--data-root", type=Path, default=Path("data"))
    p_purge.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_purge.add_argument("--migration-artifact", type=Path, required=True)
    p_purge.add_argument("--silver-report", type=Path, required=True)
    p_purge.add_argument("--confirm-purge", action="store_true")

    p_import = sub.add_parser("import-retained", help="Bronze-only import")
    p_import.add_argument("--source-root", type=Path, required=True)
    p_import.add_argument("--bronze-root", type=Path, required=True)
    p_import.add_argument("--retrieved-at", type=str, required=False, default=None)

    p_run = sub.add_parser("run-backtest", help="Run Champion backtest from Gold")
    p_run.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_run.add_argument("--gold-root", type=Path, default=None)
    p_run.add_argument("--silver-root", type=Path, default=None)
    p_run.add_argument("--validation-start", type=str, default=None)
    p_run.add_argument("--validation-end", type=str, default=None)
    p_run.add_argument("--smoke-symbol", type=str, default=None)
    p_run.add_argument("--gold-dataset-id", type=str, default=None)
    p_run.add_argument("--backtest-run-manifest", type=Path, default=None)
    p_run.add_argument("--initial-cash", type=float, default=100000000.0)
    p_run.add_argument("--scenario", type=str, default="base")
    p_run.add_argument("--ledger-id", type=str, default="champion-2016")
    p_run.add_argument("--strategy-id", choices=("core-v1", "champion-v1", "compounding-v1", "compounding-v2"), default=None)

    p_rebuild = sub.add_parser("rebuild-data", help="Prepare verified rebuild before collection")
    # add_argument("rebuild-data", help="historical pipeline subcommand marker")
    p_rebuild.add_argument("--data-root", type=Path, default=Path("data"))
    p_rebuild.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_rebuild.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_rebuild.add_argument("--gold-root", type=Path, default=Path("data/gold/stocks"))
    p_rebuild.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_rebuild.add_argument("--validation-start", type=str, required=True)
    p_rebuild.add_argument("--validation-end", type=str, required=True)
    p_rebuild.add_argument("--certification-time", type=str, required=True)
    p_rebuild.add_argument("--resume", action="store_true", default=True)
    p_rebuild.add_argument("--no-resume", dest="resume", action="store_false")
    p_rebuild.add_argument("--investor-flow-provider", choices=("ls", "kiwoom"), default="ls")

    p_kis_probe = sub.add_parser("probe-kis-flow", help="Verify one historical KIS investor-flow session")
    p_kis_probe.add_argument("--symbol", type=str, required=True)
    p_kis_probe.add_argument("--session", type=str, required=True)

    p_plan = sub.add_parser("plan", help="Build immutable historical collection plan")
    p_plan.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_plan.add_argument("--artifact-root", type=Path, default=Path("data/artifacts/collection-plans"))
    p_plan.add_argument("--coverage-start", type=str, required=True)
    p_plan.add_argument("--coverage-end", type=str, required=True)
    p_plan.add_argument("--symbols", type=str, required=False, default=None)
    p_plan.add_argument("--sessions", type=str, required=False, default=None)
    p_plan.add_argument("--chunk-size", type=int, default=LS_MAX_SESSIONS_PER_REQUEST)

    p_resume = sub.add_parser("resume", help="Resume collection from checkpoints")
    p_resume.add_argument("--plan-id", type=str, required=True)
    p_resume.add_argument("--checkpoint-root", type=Path, default=Path("data/artifacts/collection-checkpoints"))
    p_resume.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))

    p_readiness = sub.add_parser("readiness", help="Check collection readiness for certification")
    p_readiness.add_argument("--plan-id", type=str, required=True)

    # CLI registration: add_argument("build-gold") subcommand (via add_parser) with those flags.
    p_gold = sub.add_parser("build-gold", help="Run Gold-layer pre-flight audit and write manifest artifact")
    p_gold.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_gold.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_gold.add_argument("--gold-root", type=Path, default=None)
    p_gold.add_argument("--decision-time", type=str, required=True)
    p_gold.add_argument("--validation-start", type=str, required=True)
    p_gold.add_argument("--validation-end", type=str, required=True)
    p_gold.add_argument("--silver-dataset-id", action="append", metavar="TABLE=DATASET_ID", required=True)

    p_bdm = sub.add_parser("backfill-daily-market", help="PIT-safe Silver daily-market coverage backfill")
    p_bdm.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_bdm.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_bdm.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_bdm.add_argument("--validation-start", type=str, required=True)
    p_bdm.add_argument("--validation-end", type=str, required=True)
    p_bdm.add_argument("--decision-time", type=str, required=True)

    p_audit = sub.add_parser("audit-provenance", help="Classify retry and production-provenance blockers")
    p_audit.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_audit.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_audit.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))

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

    p_rebase = sub.add_parser("rebase-2019", help="Rebase legacy raw receipts into scoped Bronze")
    _add_scoped_args(p_rebase)
    p_rebase.add_argument("--legacy-data-root", type=Path, required=False, default=None)
    p_rebase.add_argument("--dry-run", action="store_true")

    p_remove_legacy = sub.add_parser("remove-legacy-data", help="Verify and remove enumerated legacy roots")
    _add_scoped_args(p_remove_legacy)
    p_remove_legacy.add_argument("--rebase-report", type=Path, required=True)
    p_remove_legacy.add_argument("--apply", action="store_true")

    p_backtest = sub.add_parser("backtest", help="Run one scope-bound backtest segment")
    _add_scoped_args(p_backtest)
    p_backtest.add_argument("--segment", choices=("development", "validation", "holdout"), required=True)
    p_backtest.add_argument("--silver-dataset-id", dest="silver_dataset_id", action="append", default=None)
    p_backtest.add_argument("--gold-dataset-id", type=str, required=True)
    p_backtest.add_argument("--strategy-id", type=str, required=True)
    p_backtest.add_argument("--strategy-policy", type=Path, required=True)
    p_backtest.add_argument("--execution-policy", type=Path, required=True)
    p_backtest.add_argument("--universe-policy", type=Path, required=True)

    p_flow_silver = sub.add_parser("build-investor-flow-silver", help="Build Silver investor-flow dataset from raw LS rows")
    _add_scoped_args(p_flow_silver)
    p_flow_silver.add_argument("--workers", type=int, default=4)

    p_daily_silver = sub.add_parser("build-daily-market-silver", help="Build Silver daily-market dataset with KRX base prices")
    _add_scoped_args(p_daily_silver)

    p_panel = sub.add_parser("build-market-panel", help="Build decision-safe Gold market panel")
    _add_scoped_args(p_panel)
    p_panel.add_argument("--daily-market-dataset-id", required=True)
    p_panel.add_argument("--universe-dataset-id", required=True)
    p_panel.add_argument("--rules", type=Path, default=Path("config/market/krx_market_rules.toml"))
    p_panel.add_argument("--instrument-buckets", type=int, default=16)

    p_bench = sub.add_parser("build-reference-benchmarks", help="Build frictionless Gold reference benchmarks")
    _add_scoped_args(p_bench)
    p_bench.add_argument("--market-panel-dataset-id", required=True)
    p_bench.add_argument("--definitions", type=Path, default=Path("config/data/reference_benchmarks.toml"))

    p_compact = sub.add_parser(
        "compact-storage-generations", help="Plan/apply retention for superseded catalog revisions and Silver table generations"
    )
    _add_scoped_args(p_compact)
    p_compact.add_argument("--apply", action="store_true")

    p_kis_backfill = sub.add_parser(
        "backfill-kis-investor-flow-gap", help="Backfill the LS investor-flow gap via KIS pages"
    )
    _add_scoped_args(p_kis_backfill)
    p_kis_backfill.add_argument("--market-panel-dataset-id", required=True)
    p_kis_backfill.add_argument("--ls-flow-dataset-id", required=True)
    p_kis_backfill.add_argument("--pace-seconds", type=float, default=0.35)

    p_kis_supplement = sub.add_parser(
        "build-investor-flow-kis-supplement", help="Build the provider-tagged KIS supplement for the LS gap"
    )
    _add_scoped_args(p_kis_supplement)
    p_kis_supplement.add_argument("--market-panel-dataset-id", required=True)
    p_kis_supplement.add_argument("--ls-flow-dataset-id", required=True)

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


def _run_scoped(args: argparse.Namespace, func: Callable[[], dict[str, object]]) -> int:
    """Execute one scoped command body and emit its payload without traceback leaks."""
    try:
        payload = func()
    except (PITDataError, ValueError, OSError) as exc:
        _emit({"error": str(exc)})
        return 1
    _emit(payload)
    return 0


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


def _read_policy_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PITDataError(f"scoped {label} policy file must hold an object")
    return raw


def _hash_policy_document(policy: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(policy), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_scope_dataset_bindings(values: Sequence[str] | None) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for token in values or ():
        name, sep, ident = str(token).partition("=")
        if not sep or not name.strip() or not ident.strip():
            raise PITDataError(f"malformed silver dataset binding: {token!r}")
        table = name.strip()
        if table in bindings:
            raise PITDataError(f"duplicate silver dataset binding: {table!r}")
        try:
            SilverTable(table)
        except ValueError:
            raise PITDataError(f"unknown silver table: {table!r}") from None
        bindings[table] = ident.strip()
    return bindings


def _build_scope_strategy(strategy_id: str, policy: Mapping[str, Any]) -> Any:
    from src.data.backtest_runner import EqualWeightSwingStrategy

    if strategy_id == "equal-weight" and str(policy.get("kind") or "") == "equal_weight":
        return EqualWeightSwingStrategy(
            strategy_id=strategy_id,
            target_weight=float(policy.get("target_weight", 1.0)),
            max_positions=int(policy.get("max_positions", 5)),
        )
    raise PITDataError(f"unknown strategy {strategy_id!r}")


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


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def normalize_dart_facts(
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    batch_size: int = 500,
) -> dict[str, object]:
    """Incremental DART fact refresh entry point for the normalize-dart-facts command."""
    from src.data.incremental_normalization import refresh_dart_financial_facts

    artifact = refresh_dart_financial_facts(
        bronze_root=Path(bronze_root),
        silver_root=Path(silver_root),
        artifact_root=Path(artifact_root),
        decision_time=decision_time,
        batch_size=int(batch_size),
    )
    return {"output_hash": artifact.output_hash, "report_hash": artifact.report_hash, "row_count": artifact.row_count}


def _run_backtest_from_silver(
    *,
    silver_root: Path,
    strategy_id: str,
    validation_start: str,
    validation_end: str,
    artifact_root: Path,
    smoke_symbol: str | None = None,
) -> object:
    """Load certified Silver tables (including lifecycle) and run the backtest."""
    # Wiring: load lifecycle_events and pass lifecycle_events=lifecycle_events to build_backtest_sessions and evidence reporting
    from src.data.snapshot import PITSnapshotRepository

    calendar_frame = _load_silver_table(silver_root, SilverTable.CALENDAR)
    security_master = _load_silver_table(silver_root, SilverTable.SECURITY_MASTER)
    daily_market = _load_silver_table(silver_root, SilverTable.DAILY_MARKET)
    corporate_actions = _load_silver_table(silver_root, SilverTable.CORPORATE_ACTIONS)
    lifecycle_events = _load_silver_table(silver_root, SilverTable.LIFECYCLE_EVENTS)
    _ = (strategy_id, validation_start, validation_end, artifact_root, smoke_symbol)
    sessions_tuple = tuple(calendar_frame["session"].to_list()) if "session" in calendar_frame.columns else ()
    calendar = SessionCalendar(sessions_tuple if sessions_tuple else (datetime.now(UTC),))
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: daily_market}, root=silver_root)
    sessions = build_backtest_sessions(
        snapshot_repository=repository,
        calendar=calendar,
        start=calendar.sessions[0],
        end=calendar.sessions[0],
        decision_time_of=lambda session: session,
        security_master=security_master,
        corporate_actions=corporate_actions,
        lifecycle_events=lifecycle_events,
    )
    return _execute_backtest(sessions=sessions, strategy_id=strategy_id, artifact_root=artifact_root)


def _load_silver_table(silver_root: Path, table: SilverTable) -> Any:
    from src.data.silver import load_latest_silver_table

    return load_latest_silver_table(root=silver_root, table=table, decision_time=datetime.now(UTC))


def _load_manifest_silver_table(silver_root: Path, table: SilverTable, dataset_id: str | None = None) -> Any:
    """Load a manifest-bound Silver table without optional swallows."""
    from src.data.silver import load_silver_table_by_dataset_id

    if dataset_id is not None:
        return load_silver_table_by_dataset_id(
            root=Path(silver_root), table=table, dataset_id=str(dataset_id), decision_time=datetime.now(UTC)
        )
    return _load_silver_table(Path(silver_root), table)


def _filter_unresolved_lifecycle_events(frame: Any) -> tuple[Any, tuple[str, ...]]:
    """Keep only verified lifecycle evidence and report excluded instruments.

    An unresolved lifecycle receipt is evidence that must remain visible, but
    it cannot be converted into a ledger action.  Excluding that event from a
    run is therefore safer than aborting an otherwise usable backtest or
    inventing a settlement.
    """
    if "evidence_status" not in getattr(frame, "columns", ()):
        return frame, ()
    import polars as pl

    unresolved = frame.filter(pl.col("evidence_status") != "verified")
    excluded = tuple(sorted({str(value) for value in unresolved["instrument_id"].drop_nulls().to_list()}))
    return frame.filter(pl.col("evidence_status") == "verified"), excluded


def _load_silver_table_for_symbol(silver_root: Path, table: SilverTable, instrument_id: str) -> Any:
    """Load only rows for one instrument from a latest Silver dataset.

    Historical master/action partitions can be millions of rows and may have
    compatible-but-not-identical schemas across years.  Reading each parquet
    file independently keeps smoke/preflight runs bounded in memory while
    ``diagonal_relaxed`` preserves nullable legacy columns.
    """
    import polars as pl

    from src.data.silver import latest_silver_dataset_path

    dataset_root = latest_silver_dataset_path(
        root=silver_root, table=table, decision_time=datetime.now(UTC)
    )
    files = sorted(dataset_root.rglob("*.parquet"))
    matches: list[Any] = []
    required_columns = {
        SilverTable.SECURITY_MASTER: {"valid_from"},
        SilverTable.CORPORATE_ACTIONS: {"effective_date", "action_type"},
    }.get(table, set())
    for path in files:
        frame = pl.read_parquet(path)
        if required_columns and not required_columns.issubset(frame.columns):
            continue
        if "instrument_id" not in frame.columns:
            continue
        subset = frame.filter(pl.col("instrument_id") == instrument_id)
        if subset.height:
            matches.append(subset)
    if not matches:
        raise PITDataError(f"no {table.value} Silver rows for {instrument_id}")
    return pl.concat(matches, how="diagonal_relaxed")


def _champion_scores_by_session(scores_frame: Any) -> dict[date, tuple[Any, ...]]:
    """Build the in-memory session index consumed by ChampionStrategy."""
    from src.strategy.scoring import ChampionScoreReason, ChampionScoreRow

    grouped: dict[date, list[ChampionScoreRow]] = {}
    for row in scores_frame.to_dicts():
        raw_session = row["decision_session"]
        session_dt = raw_session if isinstance(raw_session, datetime) else datetime.fromisoformat(str(raw_session))
        if session_dt.tzinfo is None:
            session_dt = session_dt.replace(tzinfo=UTC)
        raw_reasons = row.get("exclusion_reasons")
        if isinstance(raw_reasons, str):
            parts = [p.strip() for p in raw_reasons.split(",")]
        else:
            parts = [str(p).strip() for p in (raw_reasons or ())]
        reasons = tuple(ChampionScoreReason(p) for p in parts if p)
        score_row = ChampionScoreRow(
            decision_session=session_dt,
            instrument_id=str(row["instrument_id"]),
            eligible=bool(row["eligible"]),
            champion_score=None if row.get("champion_score") is None else float(row["champion_score"]),
            rank=None if row.get("rank") is None else int(row["rank"]),
            exclusion_reasons=reasons,
            feature_policy_version=str(row["feature_policy_version"]),
            score_policy_version=str(row["score_policy_version"]),
        )
        grouped.setdefault(session_dt.date(), []).append(score_row)
    return {session: tuple(rows) for session, rows in grouped.items()}


def _eligible_universe_by_session(universe_frame: Any) -> dict[date, tuple[str, ...]]:
    seen: set[tuple[str, str]] = set()
    grouped: dict[date, list[str]] = {}
    for row in universe_frame.to_dicts():
        raw_session = row.get("decision_session", row.get("session"))
        session_dt = raw_session if isinstance(raw_session, datetime) else datetime.fromisoformat(str(raw_session))
        if session_dt.tzinfo is None:
            session_dt = session_dt.replace(tzinfo=UTC)
        if not row.get("eligible", False):
            continue
        iid = str(row["instrument_id"])
        key = (session_dt.date().isoformat(), iid)
        if key in seen:
            raise PITDataError(f"duplicate universe row for {key!r}")
        seen.add(key)
        grouped.setdefault(session_dt.date(), []).append(iid)
    return {session: tuple(sorted(ids)) for session, ids in grouped.items()}


def _dispatch_backtest(args: argparse.Namespace) -> int:
    """Execute Champion backtest or single-instrument smoke test with replayable artifacts."""
    from datetime import date

    import polars as pl

    from src.core.costs import (
        LiquiditySlippageModel,
        default_base_schedule,
        default_krx_tick_schedule,
    )
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.data.backtest_run_manifest import load_legacy_backtest_run_manifest
    from src.data.backtest_runner import run_managed_backtest
    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import latest_silver_dataset_path, load_silver_table_by_dataset_id, silver_dataset_path_by_id
    from src.data.snapshot import PITSnapshotRepository
    from src.engine.backtest import BacktestConfig
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    selected_silver_root = getattr(args, "silver_root", None)
    selected_gold_root = getattr(args, "gold_root", None)
    selected_validation_start = getattr(args, "validation_start", None)
    selected_validation_end = getattr(args, "validation_end", None)
    selected_strategy_id = getattr(args, "strategy_id", None)
    silver_root = Path(selected_silver_root or "data/silver/stocks")
    artifact_root = Path(getattr(args, "artifact_root", "data/artifacts"))
    gold_root = Path(selected_gold_root or "data/gold/stocks")
    val_start = date.fromisoformat(str(selected_validation_start or "2016-01-04"))
    val_end = date.fromisoformat(str(selected_validation_end or "2016-12-30"))
    smoke_symbol = getattr(args, "smoke_symbol", None)
    gold_dataset_id = getattr(args, "gold_dataset_id", None)
    manifest_arg = getattr(args, "backtest_run_manifest", None)
    scores_by_session: dict[date, tuple[Any, ...]] | None = None
    scores_frame: Any = None
    universe_frame: Any = None
    eligible_by_session: dict[date, tuple[str, ...]] | None = None
    strategy: Any = None
    run_manifest: Any = None
    bundle: Any = None

    strategy_id = str(selected_strategy_id or "core-v1")
    if not smoke_symbol:
        if manifest_arg is None:
            raise PITDataError("run-backtest requires --backtest-run-manifest for manifest-bound backtest")
        run_manifest = load_legacy_backtest_run_manifest(Path(manifest_arg))
        if selected_silver_root is not None and str(Path(selected_silver_root)) != run_manifest.silver_root:
            raise PITDataError("--silver-root conflicts with --backtest-run-manifest selection")
        if selected_gold_root is not None and str(Path(selected_gold_root)) != run_manifest.gold_root:
            raise PITDataError("--gold-root conflicts with --backtest-run-manifest selection")
        if gold_dataset_id is not None and str(gold_dataset_id) != run_manifest.gold_dataset_id:
            raise PITDataError("--gold-dataset-id conflicts with --backtest-run-manifest selection")
        if selected_validation_start is not None and val_start != run_manifest.validation_start:
            raise PITDataError("--validation-start conflicts with --backtest-run-manifest selection")
        if selected_validation_end is not None and val_end != run_manifest.validation_end:
            raise PITDataError("--validation-end conflicts with --backtest-run-manifest selection")
        if selected_strategy_id is not None and strategy_id != run_manifest.strategy_id:
            raise PITDataError("--strategy-id conflicts with --backtest-run-manifest selection")
        silver_root = Path(run_manifest.silver_root)
        gold_root = Path(run_manifest.gold_root)
        gold_dataset_id = run_manifest.gold_dataset_id
        val_start = run_manifest.validation_start
        val_end = run_manifest.validation_end
        strategy_id = run_manifest.strategy_id
        from src.data.gold_artifacts import (
            load_gold_artifact_frames,
            load_gold_universe_frame,
            resolve_gold_artifact_bundle,
        )

        gold_decision_time = datetime.now(UTC)
        if strategy_id == "compounding-v1":
            bundle = resolve_gold_artifact_bundle(
                gold_root=gold_root,
                dataset_id=str(gold_dataset_id),
                decision_time=gold_decision_time,
                required_kinds=("universe",),
            )
        elif strategy_id == "compounding-v2":
            bundle = resolve_gold_artifact_bundle(
                gold_root=gold_root,
                dataset_id=str(gold_dataset_id),
                decision_time=gold_decision_time,
                required_kinds=("universe", "qvef", "champion_scores"),
            )
        else:
            bundle = resolve_gold_artifact_bundle(
                gold_root=gold_root,
                dataset_id=str(gold_dataset_id),
                decision_time=gold_decision_time,
            )
        if strategy_id == "compounding-v1":
            universe_frame = load_gold_universe_frame(bundle=bundle, decision_time=gold_decision_time)
            scores_frame = None
        elif strategy_id == "compounding-v2":
            # qvef is loaded only so load_gold_artifact_frames can certify the
            # scores against the declared policy; it is not retained downstream.
            universe_frame, _qvef_frame, scores_frame = load_gold_artifact_frames(bundle=bundle, decision_time=gold_decision_time)
        else:
            universe_frame, _qvef_frame, scores_frame = load_gold_artifact_frames(
                bundle=bundle, decision_time=gold_decision_time
            )
    if not smoke_symbol and strategy_id not in ("core-v1", "compounding-v1") and scores_frame is None:
        _gid = str(gold_dataset_id) if gold_dataset_id is not None else ""
        if not _gid.strip() or "/" in _gid or "\\" in _gid or ".." in _gid:
            raise PITDataError("run-backtest requires resolved Gold artifact; missing --gold-dataset-id")
        from src.data.gold_artifacts import load_gold_artifact_frames, resolve_gold_artifact_bundle

        gold_decision_time = datetime.now(UTC)
        bundle = resolve_gold_artifact_bundle(
            gold_root=gold_root,
            dataset_id=_gid,
            decision_time=gold_decision_time,
        )
        scores_frame = load_gold_artifact_frames(bundle=bundle, decision_time=gold_decision_time)[2]
    if not smoke_symbol and run_manifest is None and (not gold_root.exists() or not (gold_root / "universe").exists()):
        raise PITDataError("run-backtest requires resolved Gold artifact, session repository, config, and strategy")

    initial_cash = float(getattr(args, "initial_cash", 100_000_000.0))
    scenario_str = str(getattr(args, "scenario", "base")).lower()
    scenario = ExecutionScenario.BASE if scenario_str == "base" else ExecutionScenario.IDEAL

    # 캘린더 세션키는 Silver 정규형(Asia/Seoul@09:00)으로 단일화한다.
    if run_manifest is not None:
        calendar_df = load_silver_table_by_dataset_id(
            root=silver_root,
            table=SilverTable.CALENDAR,
            dataset_id=str(run_manifest.silver_dataset_ids["calendar"]),
            decision_time=datetime.now(UTC),
        )
    else:
        calendar_df = _load_silver_table(silver_root, SilverTable.CALENDAR)
    # 캘린더 세션키는 Silver 정규형(Asia/Seoul@09:00)으로 단일화한다.
    cal_sessions = tuple(sorted(canonicalize_session_keys(calendar_df)["session"].to_list()))
    calendar = SessionCalendar(cal_sessions)

    if strategy_id == "compounding-v2" and scores_frame is not None:
        # v2 consumes scores only on its deterministic 20-session selection
        # cadence.  Predicate-filter before converting rows to Python objects;
        # the full PIT date range remains represented by the calendar and the
        # strategy's latest-prior-score lookup.
        selection_dates = tuple(
            session.date()
            for index, session in enumerate(cal_sessions)
            if index % 20 == 0
        )
        scores_frame = scores_frame.filter(
            pl.col("decision_session").dt.date().is_in(selection_dates)
        )

    if strategy is None and scores_frame is not None and strategy_id not in ("core-v1", "compounding-v1", "compounding-v2"):
        # 정보량 미달 Gold 가 전략 구성·원장 실행에 도달하지 못하게 먼저 차단한다.
        certify_informative_gold(frame=scores_frame, floors=CHAMPION_SCORE_COVERAGE_FLOORS, dataset_label=f"champion_scores/{gold_dataset_id}")
        scores_by_session = _champion_scores_by_session(scores_frame)
        strategy = ChampionStrategy(scores_by_session=scores_by_session, calendar=calendar)

    if strategy is None and scores_frame is not None and strategy_id == "compounding-v2":
        scores_by_session = _champion_scores_by_session(scores_frame)
        scores_frame = None
        strategy = CompoundingV2Strategy(scores_by_session=scores_by_session, calendar=calendar)

    val_sessions = [s for s in cal_sessions if val_start <= s.date() <= val_end]
    if not val_sessions:
        raise PITDataError(f"No calendar sessions between {val_start} and {val_end}")

    start_session = val_sessions[0]
    end_session = val_sessions[-1]
    end_idx = cal_sessions.index(end_session)
    if end_idx + 1 >= len(cal_sessions):
        raise PITDataError(f"Coverage exhausted: no session after {end_session}")
    next_session = cal_sessions[end_idx + 1]
    coverage_end = cal_sessions[end_idx + 2] if end_idx + 2 < len(cal_sessions) else next_session

    # Load market bars
    if run_manifest is not None:
        dm_root = silver_dataset_path_by_id(
            root=silver_root,
            table=SilverTable.DAILY_MARKET,
            dataset_id=str(run_manifest.silver_dataset_ids["daily_market"]),
            decision_time=datetime.now(UTC),
        )
    else:
        dm_root = latest_silver_dataset_path(
            root=silver_root,
            table=SilverTable.DAILY_MARKET,
            decision_time=datetime.now(UTC),
        )
    parquet_files = list(dm_root.rglob("*.parquet"))
    if not parquet_files:
        raise PITDataError("missing daily market parquet files")

    # Calendar-aligned warm-up window for rolling PIT inputs (ADTV20/vol60).
    start_idx = cal_sessions.index(start_session)
    warmup_sessions = 200 if strategy_id in ("compounding-v1", "compounding-v2") else 60
    warmup_start = cal_sessions[max(0, start_idx - warmup_sessions)]
    if strategy_id in ("compounding-v1", "compounding-v2") and start_idx < 200: raise PITDataError("compounding-v1 requires 200 pre-validation calendar sessions")  # noqa: E701
    market_columns = [
        "session",
        "instrument_id",
        "open",
        "close",
        "volume",
        "trading_value",
        "market_cap",
        "shares_outstanding",
        "available_at",
    ]
    scans = [
        pl.scan_parquet(path)
        .select([c for c in market_columns if c in pl.scan_parquet(path).collect_schema().names()])
        .filter((pl.col("session") >= warmup_start) & (pl.col("session") <= coverage_end))
        for path in parquet_files
    ]
    query = pl.concat(scans, how="vertical_relaxed")
    if smoke_symbol:
        query = query.filter(pl.col("instrument_id") == smoke_symbol)
    query = query.filter(
        (pl.col("open") > 0)
        & (pl.col("close") > 0)
    )
    daily_market = query.collect()
    if daily_market.height == 0:
        raise PITDataError("no daily market data in range")
    if "available_at" not in daily_market.columns:
        raise PITDataError("daily market missing certified available_at")
    if "market_cap" not in daily_market.columns:
        raise PITDataError("daily market missing market_cap")
    daily_market_pit = daily_market
    snapshot_repo = PITSnapshotRepository.from_frames(
        {SilverTable.DAILY_MARKET: daily_market_pit}, root=silver_root
    )

    if run_manifest is not None:
        manifest_time = datetime.now(UTC)
        security_master = load_silver_table_by_dataset_id(
            root=silver_root,
            table=SilverTable.SECURITY_MASTER,
            dataset_id=str(run_manifest.silver_dataset_ids["security_master"]),
            decision_time=manifest_time,
        )
        corporate_actions = load_silver_table_by_dataset_id(
            root=silver_root,
            table=SilverTable.CORPORATE_ACTIONS,
            dataset_id=str(run_manifest.silver_dataset_ids["corporate_actions"]),
            decision_time=manifest_time,
        )
    elif smoke_symbol:
        # Keep the production smoke path bounded even when the historical
        # Silver tables contain all instruments.  Unit fixtures often expose
        # only a daily-market directory, so retain the regular loader as a
        # compatibility fallback when no filtered dataset is available.
        try:
            security_master = _load_silver_table_for_symbol(
                silver_root, SilverTable.SECURITY_MASTER, smoke_symbol
            )
        except PITDataError:
            security_master = _load_silver_table(silver_root, SilverTable.SECURITY_MASTER)
        try:
            corporate_actions = _load_silver_table_for_symbol(
                silver_root, SilverTable.CORPORATE_ACTIONS, smoke_symbol
            )
        except PITDataError:
            corporate_actions = _load_silver_table(silver_root, SilverTable.CORPORATE_ACTIONS)
    else:
        security_master = _load_silver_table(silver_root, SilverTable.SECURITY_MASTER)
        corporate_actions = _load_silver_table(silver_root, SilverTable.CORPORATE_ACTIONS)

    # 일자 스냅샷 중복은 SCD2 구간 압축으로 제거한다 (unique 는 0행도 줄이지 못하는 순손실).
    security_master = compact_security_master_intervals(security_master, sessions=cal_sessions)

    if run_manifest is not None:
        lifecycle_events = _load_manifest_silver_table(
            silver_root, SilverTable.LIFECYCLE_EVENTS, dataset_id=str(run_manifest.silver_dataset_ids["lifecycle_events"])
        )
    else:
        lifecycle_events = _load_manifest_silver_table(silver_root, SilverTable.LIFECYCLE_EVENTS)
    lifecycle_events, excluded_lifecycle_instruments = _filter_unresolved_lifecycle_events(lifecycle_events)

    # 제외 결정은 1회 전처리로 확정하고 단일 빌드로 실행한다 (중복 재빌드 제거).
    coverage_daily_market = daily_market
    exclusion_plan = resolve_backtest_exclusion_plan(daily_market=daily_market, corporate_actions=corporate_actions, calendar=calendar, policy=BacktestMarketInputsPolicy(), terminal_session=next_session)
    excluded_unexplained_action_instruments: set[str] = set(exclusion_plan.corporate_action_instruments)
    excluded_missing_market_close_instruments: set[str] = set(exclusion_plan.missing_terminal_close_instruments)
    daily_market = exclusion_plan.eligible_daily_market
    # security_master·corporate_actions 에는 종가 결측 제외만 적용한다 (현행 산출물 등가).
    missing_close_list = sorted(excluded_missing_market_close_instruments)
    security_master = security_master.filter(~pl.col("instrument_id").is_in(missing_close_list))
    corporate_actions = corporate_actions.filter(~pl.col("instrument_id").is_in(missing_close_list))
    snapshot_repo = PITSnapshotRepository.from_frames(
        {SilverTable.DAILY_MARKET: daily_market}, root=silver_root
    )
    market_keys = (
        {
            (session.date(), iid)
            for session, iid in zip(
                daily_market["session"].to_list(),
                daily_market["instrument_id"].to_list(),
                strict=True,
            )
        }
        if strategy_id in ("compounding-v1", "compounding-v2")
        else set()
    )
    if strategy_id in ("compounding-v1", "compounding-v2") and universe_frame is not None:  # noqa: E701
        eligible_by_session = {day: tuple(iid for iid in ids if (day, iid) in market_keys) for day, ids in _eligible_universe_by_session(universe_frame).items()}
        universe_frame = None
    # 빌드 실패는 재시도 없이 전파한다 (메시지 정규식 제어 금지).
    sessions = build_backtest_sessions(snapshot_repository=snapshot_repo, calendar=calendar, start=start_session, end=next_session, decision_time_of=lambda s: s.replace(hour=15, minute=30, second=0), security_master=security_master, corporate_actions=corporate_actions, lifecycle_events=lifecycle_events, market_index_eligible_by_session=eligible_by_session)

    distinct_symbols = daily_market["instrument_id"].unique().to_list()
    instruments = {
        sym: Instrument(sym, AssetKind.STOCK, "KRX", sym.split(":")[-1], "KRW")
        for sym in distinct_symbols
    }

    costs = default_base_schedule()
    ticks = default_krx_tick_schedule()
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.1, ticks),
        scenario,
        target_participation_cap=0.0025,
        hard_participation_cap=0.005,
    )

    config = BacktestConfig(
        ledger_id=str(getattr(args, "ledger_id", "champion-2016")),
        initial_cash=initial_cash,
        instruments=instruments,
        scenario=scenario,
        cost_schedule=costs,
        calendar=calendar,
        fill_model=fill_model,
    )

    sessions_ordered = [s.session_open for s in sessions]
    if smoke_symbol:
        target_sym = smoke_symbol

        class SmokeStrategy:
            def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
                current_open = context.decision_time.replace(hour=9, minute=0, second=0)
                idx = sessions_ordered.index(current_open)
                if idx == 0:
                    return (
                        TradeIntent(
                            intent_id=f"smoke-buy-{target_sym}",
                            asset_kind=AssetKind.STOCK,
                            instrument_id=target_sym,
                            # Keep the smoke order below the historical hard
                            # participation cap even on thin-volume sessions.
                            target_value=initial_cash * 0.2,
                            decision_time=context.decision_time,
                            execution_time=sessions_ordered[1],
                            strategy_id="champion-v1",
                            reason="smoke_entry",
                            idempotency_key=f"smoke_entry_{target_sym}",
                            account_snapshot_id=context.portfolio.account_snapshot_id,
                        ),
                    )
                return ()

        strategy = SmokeStrategy()
    elif strategy is None and strategy_id == 'core-v1' and not smoke_symbol:
        from src.data.gold_artifacts import load_gold_artifact_frames, resolve_gold_artifact_bundle

        _ = (resolve_gold_artifact_bundle, load_gold_artifact_frames)
        if universe_frame is None:
            raise PITDataError("run-backtest requires selected Gold universe for core-v1")
        eligible_by_session = _eligible_universe_by_session(universe_frame)
        # The Gold universe is built before the final corporate-action and
        # PIT-master resolution.  Intersect it with the actually materialized
        # session bars so excluded/temporarily unavailable symbols cannot
        # reach CoreStrategy as missing snapshot inputs.
        session_symbols = {
            session.session_open.date(): frozenset(bar.instrument_id for bar in session.bars)
            for session in sessions
        }
        eligible_by_session = {
            day: tuple(iid for iid in ids if iid in session_symbols.get(day, frozenset()))
            for day, ids in eligible_by_session.items()
        }
        strategy = CoreStrategy(eligible_by_session=eligible_by_session, calendar=calendar)
    elif strategy is None and strategy_id == "compounding-v1" and eligible_by_session is not None and not smoke_symbol: strategy = CompoundingStrategy(eligible_by_session=eligible_by_session, calendar=calendar)  # noqa: E701
    elif strategy is None:
        raise PITDataError("run-backtest requires a resolved strategy for the selected bundle")

    from src.strategy.core_strategy import CoreStrategyPolicy

    _core_policy = CoreStrategyPolicy()
    # 관측된 시간 의미를 런 메타데이터에 기록한다 (드리프트 추적).
    silver_time_semantics: list[dict[str, object]] = []
    for _table_name, _table_frame in (
        ("calendar", calendar_df),
        ("daily_market", daily_market),
        ("security_master", security_master),
    ):
        for _observation in observe_time_semantics(_table_frame):
            silver_time_semantics.append(  # noqa: PERF401 -- 계약이 append 조립을 명시한다.
                {
                    "table": _table_name,
                    "column": _observation.column,
                    "time_zone": _observation.time_zone,
                    "hour_anchors": list(_observation.hour_anchors),
                    "canonical": _observation.canonical,
                }
            )
    metadata = {
        "validation_start": str(val_start),
        "validation_end": str(val_end),
        "instruments_tracked": len(instruments),
        "strategy_id": strategy_id,
        "score_policy_version": _core_policy.score_policy_version if strategy_id == "core-v1" else "champion-v1-scoring-v1",
        "selection_policy_version": CompoundingV2Policy().selection_policy_version if strategy_id == "compounding-v2" else ("compounding-v1-selection-v1" if strategy_id == "compounding-v1" else (_core_policy.selection_policy_version if strategy_id == "core-v1" else "champion-v1-selection-v1")),
        "portfolio_policy_version": "compounding-v2-portfolio-v1" if strategy_id == "compounding-v2" else ("compounding-v1-portfolio-v1" if strategy_id == "compounding-v1" else "champion-v1-portfolio-v1"),
        "universe_manifest_hash": getattr(bundle, "universe_manifest_hash", None) if bundle is not None else None,
        "champion_scores_manifest_hash": getattr(bundle, "champion_scores_manifest_hash", None) if bundle is not None else None,
        "market_input_policy_version": BacktestMarketInputsPolicy().version,
        "warmup_sessions": warmup_sessions,
        "data_action_certified": True,
        "provenance_mode": (
            "research_source_candidate_with_explicit_unavailable"
            if run_manifest is not None and "provenance" in str(silver_root)
            else "research_fixture_inputs"
        ),
        "investor_flow_policy": "optional_no_imputation",
        "unresolved_lifecycle_event_policy": "exclude_unresolved_keep_receipt",
        "excluded_unresolved_lifecycle_instruments": list(excluded_lifecycle_instruments),
        "excluded_missing_market_close_instruments": sorted(
            excluded_missing_market_close_instruments
        ),
        "excluded_unexplained_corporate_action_instruments": sorted(
            excluded_unexplained_action_instruments
        ),
        "exclusion_reason_counts": exclusion_plan.reason_counts(),
        "silver_time_semantics": silver_time_semantics,
    }
    if run_manifest is not None:
        _eligible_count = int(exclusion_plan.eligible_daily_market.height)
        _blocked_count = int(coverage_daily_market.height - _eligible_count)
        metadata = {
            **metadata,
            "run_manifest_hash": run_manifest.content_hash,
            "eligible_instrument_sessions": _eligible_count,
            "blocked_instrument_sessions": _blocked_count,
            "excluded_instruments": sorted(exclusion_plan.excluded_instruments),
            "exclusion_reason_counts": exclusion_plan.reason_counts(),
            "quarantined_instrument_sessions": sum(
                len(slots) for slots in exclusion_plan.quarantine_sessions_by_instrument.values()
            ),
        }
    if run_manifest is not None:
        _result, manifest = run_managed_backtest(
            sessions=sessions,
            config=config,
            strategy=strategy,
            artifact_root=artifact_root,
            dataset_hash=run_manifest.content_hash,
            manifest_hash=run_manifest.content_hash,
            smoke_symbol=smoke_symbol,
            extra_metadata=metadata,
        )
    else:
        _result, manifest = run_managed_backtest(
            sessions=sessions,
            config=config,
            strategy=strategy,
            artifact_root=artifact_root,
            dataset_hash=f"validation_{val_start}_{val_end}",
            smoke_symbol=smoke_symbol,
            extra_metadata=metadata,
        )

    _emit({
        "compounding_v2_selection": summarize_compounding_v2_selection_shortfalls(strategy.selection_diagnostics) if strategy_id == "compounding-v2" else {},
        "content_hash": manifest["content_hash"],
        "ledger_id": manifest["ledger_id"],
        "scenario": manifest["scenario"],
        "session_count": manifest["session_count"],
        "fill_count": manifest["fill_count"],
        "reject_count": manifest["reject_count"],
        "smoke_symbol": smoke_symbol,
        "performance": manifest["performance"],
        "research_segments": manifest["research_segments"],
        "accounting_reconciled": manifest["accounting_reconciled"],
        "artifact_path": str(artifact_root / "backtests" / manifest["content_hash"] / "result.json"),
    })
    return 0


def _execute_backtest(**kwargs: object) -> object:
    """Typed adapter used once CLI input loaders provide concrete objects."""
    return run_champion_backtest(**kwargs)  # type: ignore[arg-type]


def _row_counts_for_report(artifact_root: Path) -> dict[str, int]:
    summary_path = Path(artifact_root) / "streaming_report.json"
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = raw.get("row_counts", {})
        return {str(k): int(v) for k, v in dict(counts).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def normalize(
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    batch_size: int = 50000,
) -> dict[str, object]:
    """Streaming normalize entry point used by the CLI normalize command."""
    from src.data.streaming_normalization import stream_normalize_stock_evidence

    report = stream_normalize_stock_evidence(
        bronze_root=Path(bronze_root),
        silver_root=Path(silver_root),
        artifact_root=Path(artifact_root),
        decision_time=decision_time,
        batch_size=int(batch_size),
    )
    return {"report_hash": report.report_hash, "source_hashes": dict(report.source_hashes)}


def _build_sessions(**kwargs: object) -> object:
    return build_backtest_sessions(**kwargs)  # type: ignore[arg-type]


def _flow_backfill_progress_payload(progress: InvestorFlowBatchProgress) -> dict[str, object]:
    """Expose one completed LS backfill batch without provider secrets.

    Long-running historical collection must be resumable from Bronze and
    checkpoints, so each emitted payload contains only stable plan position and
    evidence-accounting totals. It never serializes collector configuration,
    access tokens, response bodies, or credentials.
    """
    artifact = progress.artifact
    plan_id = str(getattr(artifact, "plan_id", "") or "")
    plan_digest = ""
    report_path = str(getattr(artifact, "report_path", "") or "")
    if report_path:
        try:
            raw_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw_report = None
        if isinstance(raw_report, dict):
            plan_id = str(raw_report.get("plan_id") or plan_id)
            plan_digest = str(raw_report.get("plan_digest") or "")
    return {
        "plan_id": plan_id,
        "plan_digest": plan_digest,
        "content_hash": str(getattr(artifact, "content_hash", "") or ""),
        "batch_index": progress.batch_index,
        "chunk_offset": progress.chunk_offset,
        "planned_chunks": int(getattr(artifact, "planned_chunks", 0) or 0),
        "completed_chunks": int(getattr(artifact, "completed_chunks", 0) or 0),
        "previously_completed_chunks": int(getattr(artifact, "previously_completed_chunks", 0) or 0),
        "pending_chunks": int(getattr(artifact, "pending_chunks", 0) or 0),
        "provider_error_chunks": int(getattr(artifact, "provider_error_chunks", 0) or 0),
        "missing_session_chunks": int(getattr(artifact, "missing_session_chunks", 0) or 0),
        "receipt_count": int(getattr(artifact, "receipt_count", 0) or 0),
        "report_path": report_path,
    }


def _run_backfill_ls_investor_flow(args: argparse.Namespace) -> int:
    """Run the LS-only bounded backfill iterator from an explicit plan path."""
    batch_size = args.chunk_batch_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 500:
        _emit({"error": "chunk-batch-size must be an integer in 1..500"})
        return 1
    max_batches = args.max_batches
    if max_batches is not None and (
        isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1
    ):
        _emit({"error": "max-batches must be a positive integer"})
        return 1
    if args.retrieved_at is not None:
        try:
            fixed_moment = datetime.fromisoformat(str(args.retrieved_at))
        except ValueError:
            _emit({"error": "retrieved-at must be ISO-8601"})
            return 1
        if fixed_moment.tzinfo is None:
            _emit({"error": "retrieved-at must be timezone-aware"})
            return 1

        def _fixed_retrieved_at() -> datetime:
            return fixed_moment

        retrieved_at_factory: Callable[[], datetime] = _fixed_retrieved_at
    else:

        def _utc_now() -> datetime:
            return datetime.now(UTC)

        retrieved_at_factory = _utc_now
    try:
        plan = load_collection_plan_path(args.plan_path)
    except (PITDataError, ValueError, OSError) as exc:
        _emit({"error": str(exc)})
        return 1
    try:
        symbols = tuple(sorted({chunk.symbol for chunk in plan.chunks}))
        collector = LsInvestorFlowCollector(symbols)
    except (PITDataError, ValueError, OSError) as exc:
        _emit({"error": str(exc)})
        return 1
    try:
        iterator = iter_planned_investor_flow_backfill(
            plan=plan,
            provider="ls",
            collector=collector,
            bronze_root=Path(args.bronze_root),
            retrieved_at_factory=retrieved_at_factory,
            checkpoint_store=CollectionCheckpointStore(Path(args.checkpoint_root)),
            chunk_batch_size=batch_size,
        )
        try:
            for batch_number, progress in enumerate(iterator, start=1):
                _emit(_flow_backfill_progress_payload(progress))
                if max_batches is not None and batch_number >= max_batches:
                    break
        finally:
            closer = getattr(iterator, "close", None)
            if callable(closer):
                closer()
    except (PITDataError, ValueError, OSError) as exc:
        _emit({"error": str(exc)})
        return 1
    return 0


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
    if args.command == "rebase-2019":
        def _rebase_2019() -> dict[str, object]:
            from src.data.rebase import materialize_scoped_bronze

            runtime = _scoped_runtime(args)
            legacy_root = Path(args.legacy_data_root) if args.legacy_data_root else Path(args.data_root)
            report = materialize_scoped_bronze(
                runtime=runtime, legacy_data_root=legacy_root, dry_run=bool(args.dry_run)
            )
            return {
                "report_path": str(report.report_path),
                "retained": report.retained_payload_count,
                "rejected": report.rejected_payload_count,
                "dry_run": bool(args.dry_run),
            }

        return _run_scoped(args, _rebase_2019)
    if args.command == "remove-legacy-data":
        def _remove_legacy_data() -> dict[str, object]:
            from src.data.data_reset import (
                LEGACY_REMOVAL_TARGETS,
                remove_verified_legacy_data,
                verify_legacy_removal,
            )
            from src.data.rebase import RebaseReport, RetentionDecision

            runtime = _scoped_runtime(args)
            raw = json.loads(Path(args.rebase_report).read_text(encoding="utf-8"))
            decisions = tuple(
                RetentionDecision(
                    legacy_path=Path(str(item.get("legacy_path") or "")),
                    source=str(item.get("source") or ""),
                    natural_key=str(item["natural_key"]) if item.get("natural_key") not in (None, "") else None,
                    retained=bool(item.get("retained")),
                    reason=str(item.get("reason") or ""),
                )
                for item in raw.get("decisions", [])
            )
            report = RebaseReport(
                scope_hash=str(raw.get("scope_hash") or ""),
                content_hash=str(raw.get("content_hash") or ""),
                decisions=decisions,
                catalog_revision_hash=str(raw.get("catalog_revision_hash") or ""),
                retained_payload_count=int(raw.get("retained_payload_count", 0)),
                rejected_payload_count=int(raw.get("rejected_payload_count", 0)),
                report_path=Path(args.rebase_report),
            )
            verification = verify_legacy_removal(
                runtime=runtime, rebase_report=report, data_root=Path(args.data_root)
            )
            if bool(args.apply):
                removed = remove_verified_legacy_data(
                    verification=verification, data_root=Path(args.data_root), apply=True
                )
                return {
                    "removable": len(verification.verified_targets),
                    "removed": len(removed),
                    "absent": len(LEGACY_REMOVAL_TARGETS) - len(removed),
                }
            planned = remove_verified_legacy_data(
                verification=verification, data_root=Path(args.data_root), apply=False
            )
            return {
                "removable": len(planned),
                "removed": 0,
                "absent": len(LEGACY_REMOVAL_TARGETS) - len(planned),
            }

        return _run_scoped(args, _remove_legacy_data)
    if args.command == "backtest":
        def _scope_backtest() -> dict[str, object]:
            from src.data.backtest_run_manifest import BacktestSegment, build_backtest_run_manifest
            from src.data.backtest_runner import (
                NextSessionExecutionModel,
                run_scope_bound_backtest,
            )

            runtime = _scoped_runtime(args)
            strategy_policy = _read_policy_json(Path(args.strategy_policy), label="strategy")
            execution_policy = _read_policy_json(Path(args.execution_policy), label="execution")
            universe_policy = _read_policy_json(Path(args.universe_policy), label="universe")
            manifest = build_backtest_run_manifest(
                runtime=runtime,
                segment=cast(BacktestSegment, args.segment),
                silver_dataset_ids=_parse_scope_dataset_bindings(args.silver_dataset_id),
                gold_dataset_id=str(args.gold_dataset_id),
                strategy_id=str(args.strategy_id),
                strategy_policy_hash=_hash_policy_document(strategy_policy),
                execution_policy_hash=_hash_policy_document(execution_policy),
                universe_policy_hash=_hash_policy_document(universe_policy),
            )
            if manifest.period_end >= date(2026, 1, 1):
                raise PITDataError("2026 and later dates are not completed backtest segments")
            strategy = _build_scope_strategy(str(args.strategy_id), strategy_policy)
            execution = NextSessionExecutionModel(
                commission_rate=float(execution_policy.get("commission_rate", 0.0)),
                tax_rate=float(execution_policy.get("tax_rate", 0.0)),
            )
            result = run_scope_bound_backtest(
                runtime=runtime, manifest=manifest, strategy=strategy, execution_model=execution
            )
            return {
                "run_dir": str(result.result_path.parent),
                "manifest_hash": result.manifest_hash,
                "metrics": dict(result.metrics),
            }

        return _run_scoped(args, _scope_backtest)
    if args.command == "build-investor-flow-silver":
        def _build_flow_silver() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.investor_flow_silver import materialize_investor_flow_silver

            runtime = _scoped_runtime(args)
            result = materialize_investor_flow_silver(
                bronze_root=runtime.workspace.bronze_root,
                universe_root=runtime.workspace.silver_root,
                silver_root=runtime.workspace.silver_root,
                workers=args.workers,
            )
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_flow_silver)
    if args.command == "build-daily-market-silver":
        def _build_daily_silver() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.daily_market_silver import materialize_daily_market_silver

            runtime = _scoped_runtime(args)
            result = materialize_daily_market_silver(
                catalog=_scoped_catalog(runtime),
                universe_root=runtime.workspace.silver_root,
                silver_root=runtime.workspace.silver_root,
            )
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_daily_silver)
    if args.command == "build-market-panel":
        def _build_market_panel() -> dict[str, object]:
            from dataclasses import asdict

            from src.core.market_rules import load_krx_market_rules
            from src.data.market_panel import materialize_market_panel

            runtime = _scoped_runtime(args)
            result = materialize_market_panel(
                daily_market_path=runtime.workspace.silver_root / args.daily_market_dataset_id,
                universe_path=runtime.workspace.silver_root / args.universe_dataset_id,
                rules=load_krx_market_rules(args.rules),
                gold_root=runtime.workspace.gold_root,
                instrument_buckets=args.instrument_buckets,
            )
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_market_panel)
    if args.command == "build-reference-benchmarks":
        def _build_reference_benchmarks() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.reference_benchmarks import load_benchmark_definitions, materialize_reference_benchmarks

            runtime = _scoped_runtime(args)
            version, definitions = load_benchmark_definitions(args.definitions)
            result = materialize_reference_benchmarks(
                market_panel_path=runtime.workspace.gold_root / args.market_panel_dataset_id,
                definitions=definitions,
                definitions_version=version,
                gold_root=runtime.workspace.gold_root,
            )
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_reference_benchmarks)
    if args.command == "backfill-kis-investor-flow-gap":
        def _backfill_kis_gap() -> dict[str, object]:
            from src.data.investor_flow_gap import compute_missing_investor_flow_cells
            from src.integrations.kis.investor_flow import KisInvestorFlowCollector

            runtime = _scoped_runtime(args)
            gap = compute_missing_investor_flow_cells(
                market_panel_path=runtime.workspace.gold_root / args.market_panel_dataset_id,
                ls_flow_silver_path=runtime.workspace.silver_root / args.ls_flow_dataset_id,
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
            result = materialize_investor_flow_kis_supplement(
                bronze_root=runtime.workspace.bronze_root,
                market_panel_path=runtime.workspace.gold_root / args.market_panel_dataset_id,
                ls_flow_silver_path=runtime.workspace.silver_root / args.ls_flow_dataset_id,
                silver_root=runtime.workspace.silver_root,
            )
            return asdict(result) | {"dataset_path": str(result.dataset_path)}

        return _run_scoped(args, _build_kis_supplement)
    if args.command == "compact-storage-generations":
        def _compact_storage_generations() -> dict[str, object]:
            from dataclasses import asdict

            from src.data.storage_retention import (
                apply_catalog_revision_retention,
                apply_table_generation_retention,
                discover_generation_tables,
                plan_catalog_revision_retention,
                plan_table_generation_retention,
            )

            runtime = _scoped_runtime(args)
            catalog_root = runtime.workspace.bronze_root / "catalog"
            catalog_plan = plan_catalog_revision_retention(catalog_root)
            table_plans = [
                plan_table_generation_retention(table_root)
                for table_root in discover_generation_tables(runtime.workspace.silver_root)
            ]
            freed = 0
            if args.apply:
                freed = apply_catalog_revision_retention(catalog_root)
                freed += sum(
                    apply_table_generation_retention(table_root)
                    for table_root in discover_generation_tables(runtime.workspace.silver_root)
                )
            return {
                "catalog": asdict(catalog_plan) | {"catalog_root": str(catalog_plan.catalog_root)},
                "tables": [asdict(p) | {"table_root": str(p.table_root)} for p in table_plans],
                "applied": args.apply,
                "bytes_freed": freed,
            }

        return _run_scoped(args, _compact_storage_generations)
    if args.command == "audit-provenance":
        try:
            from src.data.provenance_audit import audit_production_provenance

            audit = audit_production_provenance(
                bronze_root=Path(args.bronze_root),
                silver_root=Path(args.silver_root),
                artifact_root=Path(args.artifact_root),
            )
        except (ValueError, OSError, json.JSONDecodeError):
            return 1
        _emit(
            {
                "artifact_path": str(audit.artifact_path),
                "unverified_empty_response": sum(item.state == "unverified_empty_response" for item in audit.investor_flow),
                "retry_required": sum(item.state == "retry_required" for item in audit.investor_flow),
                "fixture_tables": list(audit.fixture_tables),
            }
        )
        return 0
    if args.command == "inventory":
        try:
            inventory = inspect_legacy_data(Path(args.data_root))
        except (PITDataError, ValueError, OSError):
            return 1
        _emit(
            {
                "datasets": [
                    {"path": item.relative_path, "disposition": item.disposition.value}
                    for item in inventory.entries
                ]
            }
        )
        return 0
    if args.command == "migrate-legacy":
        try:
            artifact = migrate_retained_stock_evidence(
                Path(args.source_root),
                Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
            )
            artifact_path = MigrationArtifactStore(Path(args.artifact_root)).write(artifact)
        except (PITDataError, ValueError, OSError):
            return 1
        _emit(
            {
                "content_hash": artifact.content_hash,
                "receipts": len(artifact.receipts),
                "artifact_path": str(artifact_path),
            }
        )
        return 0
    if args.command == "backfill-ls-investor-flow":
        return _run_backfill_ls_investor_flow(args)
    if args.command == "collect":
        try:
            plan = load_collection_plan(str(args.plan_id))
            collect_symbols = tuple(sorted({chunk.symbol for chunk in plan.chunks}))
            collector = resolve_investor_flow_collector(str(args.investor_flow_provider), collect_symbols)
            collection = collect_planned_investor_flow(
                plan=plan,
                provider=str(args.investor_flow_provider),
                collector=collector,
                bronze_root=Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
                checkpoint_store=CollectionCheckpointStore(Path(args.checkpoint_root)),
                allow_source_unavailable=True,
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit(
            {
                "content_hash": collection.content_hash,
                "planned_chunks": getattr(collection, "planned_chunks", 0) or len(plan.chunks),
                "completed_chunks": getattr(collection, "completed_chunks", 0),
                "previously_completed_chunks": getattr(collection, "previously_completed_chunks", 0),
                "pending_chunks": getattr(collection, "pending_chunks", 0),
                "provider_error_chunks": getattr(collection, "provider_error_chunks", 0),
                "missing_session_chunks": getattr(collection, "missing_session_chunks", 0),
                "receipts": sorted(str(k.value) for k in collection.receipts),
                "report_path": str(getattr(collection, "report_path", "")),
            }
        )
        return 0
    if args.command == "collect-dart-disclosures":
        try:
            from src.integrations.dart.xbrl import DartXbrlCollector
            from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

            collection = collect_dart_disclosures(
                dart=DartXbrlCollector(quota_store=ProviderQuotaStateStore(Path(args.artifact_root) / "quota")),
                start=date.fromisoformat(str(args.coverage_start)),
                end=date.fromisoformat(str(args.coverage_end)),
                bronze_root=Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
            )
        except (PITDataError, ValueError, OSError, ProviderQuotaBlocked) as exc:
            _emit({"error": str(exc)})
            return 1
        _emit({"content_hash": collection.content_hash, "receipts": sorted(str(k.value) for k in collection.receipts)})
        return 0
    if args.command == "collect-dart-facts":
        try:
            from src.integrations.dart.xbrl import DartXbrlCollector
            from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

            identities: tuple[dict[str, str], ...]
            if any(getattr(args, name) for name in ("corp_code", "filing_id", "biz_year", "report_code")):
                if not all(getattr(args, name) for name in ("corp_code", "filing_id", "biz_year", "report_code")):
                    raise PITDataError("corp-code, filing-id, biz-year, and report-code must be supplied together")
                identities = (
                    {
                        "corp_code": str(args.corp_code),
                        "filing_id": str(args.filing_id),
                        "biz_year": str(args.biz_year),
                        "reprt_code": str(args.report_code),
                        "fs_div": "CFS",
                    },
                )
            else:
                identities = DartXbrlCollector.filing_identities_from_bronze(
                    Path(args.bronze_root),
                    start=date.fromisoformat(str(args.coverage_start)),
                    end=date.fromisoformat(str(args.coverage_end)),
                )
            if args.offset < 0 or args.limit < 1:
                raise PITDataError("offset must be nonnegative and limit must be positive")
            batch = identities[args.offset : args.offset + args.limit]
            if not batch:
                raise PITDataError("no periodic DART filings in requested batch")
            collection = collect_dart_financial_facts(dart=DartXbrlCollector(quota_store=ProviderQuotaStateStore(Path(args.artifact_root) / "quota")), identities=batch,
                bronze_root=Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
            )
        except (PITDataError, ValueError, OSError, ProviderQuotaBlocked) as exc:
            _emit({"error": str(exc)})
            return 1
        _emit({"filings": len(batch), "content_hash": collection.content_hash})
        return 0
    if args.command == "backfill-dart-facts":
        try:
            from src.data.dart_backfill import DartHistoricalBackfillRequest, run_dart_historical_backfill_batch
            from src.integrations.dart.xbrl import DartXbrlCollector
            from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

            backfill_request = DartHistoricalBackfillRequest(
                bronze_root=Path(args.bronze_root),
                artifact_root=Path(args.artifact_root),
                silver_root=Path(args.silver_root),
                validation_start=date.fromisoformat(str(args.validation_start)),
                validation_end=date.fromisoformat(str(args.validation_end)),
                retrieved_at=_parse_dt(args.retrieved_at),
                offset=int(args.offset),
                limit=int(args.limit),
            )
            backfill_plan = run_dart_historical_backfill_batch(
                request=backfill_request, dart=DartXbrlCollector(quota_store=ProviderQuotaStateStore(Path(args.artifact_root) / "quota"))
            )
        except (PITDataError, ValueError, OSError, ProviderQuotaBlocked) as exc:
            _emit({"error": str(exc)})
            return 1
        _emit({"plan_id": backfill_plan.plan_id, "required_periods": list(backfill_plan.required_periods)})
        return 0
    if args.command == "collect-missing-dart-facts":
        try:
            from src.data.dart_backfill import DartMissingFactsRequest, run_dart_missing_facts_batch
            from src.integrations.dart.xbrl import DartXbrlCollector
            from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

            missing_facts_plan = run_dart_missing_facts_batch(
                request=DartMissingFactsRequest(
                    bronze_root=Path(args.bronze_root), artifact_root=Path(args.artifact_root),
                    backfill_artifact=Path(args.backfill_artifact), retrieved_at=_parse_dt(args.retrieved_at),
                    offset=int(args.offset), limit=int(args.limit),
                ),
                dart=DartXbrlCollector(quota_store=ProviderQuotaStateStore(Path(args.artifact_root) / "quota")),
            )
        except (PITDataError, ValueError, OSError, ProviderQuotaBlocked) as exc:
            _emit({"error": str(exc)})
            return 1
        _emit({"plan_id": missing_facts_plan.plan_id, "candidate_count": missing_facts_plan.candidate_count, "selected_count": len(missing_facts_plan.selected_identities), "missing_without_filing_count": missing_facts_plan.missing_without_filing_count})
        return 0
    if args.command == "materialize":
        try:
            from src.core.datasets import DatasetCertification

            backtest_artifact = materialize_backtest_inputs(
                bronze_root=Path(args.bronze_root),
                silver_root=Path(args.silver_root),
                gold_root=Path(args.gold_root),
                artifact_root=Path(getattr(args, "artifact_root", Path("data/artifacts"))),
                decision_time=_parse_dt(args.decision_time),
                certification=DatasetCertification.RESEARCH,
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit(
            {
                "universe_hash": backtest_artifact.universe_hash,
                "qvef_hash": backtest_artifact.qvef_hash,
                "champion_scores_hash": backtest_artifact.champion_scores_hash,
                "benchmark_cap_hash": backtest_artifact.benchmark_cap_hash,
                "benchmark_equal_hash": backtest_artifact.benchmark_equal_hash,
                "content_hash": backtest_artifact.content_hash,
            }
        )
        return 0
    if args.command == "normalize":
        try:
            from src.data.streaming_normalization import stream_normalize_stock_evidence

            report = stream_normalize_stock_evidence(
                bronze_root=Path(args.bronze_root),
                silver_root=Path(getattr(args, "silver_root", Path("data/silver/stocks"))),
                artifact_root=Path(getattr(args, "artifact_root", Path("data/artifacts"))),
                decision_time=_parse_dt(args.decision_time),
                batch_size=int(getattr(args, "batch_size", 50000)),
            )
            row_counts = _row_counts_for_report(
                Path(getattr(args, "artifact_root", Path("data/artifacts"))),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"report_hash": report.report_hash, "row_counts": row_counts})
        return 0
    if args.command == "refresh-corporate-actions":
        try:
            decision_time = _parse_dt(args.decision_time)
            calendar_frame = load_latest_silver_table(root=Path(args.silver_root), table=SilverTable.CALENDAR, decision_time=decision_time)
            calendar = SessionCalendar(tuple(sorted(calendar_frame["session"].to_list())))
            daily_market = load_latest_silver_market_scan(root=Path(args.silver_root), decision_time=decision_time, columns=("session", "instrument_id", "close", "shares_outstanding", "market_cap"))
            report = refresh_corporate_action_silver(bronze_root=Path(args.bronze_root), silver_root=Path(args.silver_root), artifact_root=Path(args.artifact_root), decision_time=decision_time, daily_market=daily_market, calendar=calendar)
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"report_hash": report.report_hash})
        return 0
    if args.command == "normalize-dart-facts":
        try:
            payload = normalize_dart_facts(
                bronze_root=Path(args.bronze_root),
                silver_root=Path(getattr(args, "silver_root", Path("data/silver/stocks"))),
                artifact_root=Path(getattr(args, "artifact_root", Path("data/artifacts"))),
                decision_time=_parse_dt(args.decision_time),
                batch_size=int(getattr(args, "batch_size", 500)),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"output_hash": payload["output_hash"], "report_hash": payload["report_hash"], "row_count": payload["row_count"]})
        return 0
    if args.command == "bronze-retention-plan":
        retention_plan = plan_bronze_retention(
            bronze_root=Path(args.bronze_root),
            provenance_roots=(Path(args.silver_root), Path(args.artifact_root)),
        )
        _emit(
            {
                "receipt_count": retention_plan.receipt_count,
                "total_payload_bytes": retention_plan.total_payload_bytes,
                "referenced_payload_bytes": retention_plan.referenced_payload_bytes,
                "unreferenced_payload_bytes": retention_plan.unreferenced_payload_bytes,
                "referenced_hashes": list(retention_plan.referenced_hashes),
                "unreferenced_hashes": list(retention_plan.unreferenced_hashes),
                "deletion_eligible": retention_plan.deletion_eligible,
                "blocking_reasons": list(retention_plan.blocking_reasons),
            }
        )
        return 0
    if args.command == "silver-gold-retention-plan":
        sg_plan = plan_storage_root_retention(
            silver_base=Path(args.silver_base),
            gold_base=Path(args.gold_base),
            artifact_root=Path(args.artifact_root),
        )
        _emit(
            {
                "retained_silver_roots": list(sg_plan.retained_silver_roots),
                "reclaimable_silver_roots": list(sg_plan.reclaimable_silver_roots),
                "retained_gold_roots": list(sg_plan.retained_gold_roots),
                "reclaimable_gold_roots": list(sg_plan.reclaimable_gold_roots),
                "orphaned_staging_paths": list(sg_plan.orphaned_staging_paths),
                "blocking_reasons": list(sg_plan.blocking_reasons),
                "deletion_eligible": sg_plan.deletion_eligible,
            }
        )
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
    if args.command == "purge-legacy":
        # Purge consumes persisted proof only and requires --confirm-purge.
        try:
            from src.core.datasets import DatasetCertification
            from src.data.operations import RebuildPreparation
            from src.data.schemas import CertificationReport, EvidenceKind

            store_root = Path(args.migration_artifact).parent
            migration = MigrationArtifactStore(store_root).read_verified(Path(args.migration_artifact))
            report_raw = json.loads(Path(args.silver_report).read_text(encoding="utf-8"))
            source_hashes = {EvidenceKind(str(k)): str(v) for k, v in dict(report_raw["source_hashes"]).items()}
            report = CertificationReport(
                certification=DatasetCertification(str(report_raw["certification"])),
                report_hash=str(report_raw["report_hash"]),
                coverage_start=date.fromisoformat(str(report_raw["coverage_start"])),
                coverage_end=date.fromisoformat(str(report_raw["coverage_end"])),
                source_hashes=source_hashes,
            )
            from pathlib import Path as _Path

            gold_root = _Path(getattr(args, "gold_root", _Path("data/gold/stocks")))
            gold_artifact = None
            required_gold = ("universe", "qvef", "champion_scores", "benchmarks")
            if gold_root.exists() and all(list((gold_root / name).rglob("*.parquet")) for name in required_gold):
                gold_artifact = dict.fromkeys(required_gold, True)
            backtest_candidates = [
                p for p in _Path("data/artifacts").glob("backtests/*/result.json")
                if p.is_file()
            ] if _Path("data/artifacts").exists() else []
            backtest_path = backtest_candidates[0] if backtest_candidates else None
            preparation = RebuildPreparation(
                migration=migration,
                silver_report=report,
                gold_artifact=gold_artifact,
                backtest_artifact_path=backtest_path,
            )
            from src.data.operations import StockDataRebuildRequest as _RebuildRequest

            purge_request = _RebuildRequest(
                data_root=Path(args.data_root),
                bronze_root=Path(args.bronze_root),
                silver_root=Path("data/silver/stocks"),
                gold_root=gold_root,
                artifact_root=store_root,
                coverage_start=report.coverage_start,
                coverage_end=report.coverage_end,
                decision_time=_parse_dt(None),
            )
            execute_verified_legacy_purge(
                purge_request,
                preparation,
                confirm_purge=bool(getattr(args, "confirm_purge", False)),
            )
        except (ValueError, OSError):
            return 1
        except PITDataError:
            return 1
        return 0
    if args.command == "run-backtest":
        # A backtest requires an immutable Gold artifact, a resolved session
        # repository, an explicit engine config, and a strategy implementation.
        # Until those are supplied by the caller, refuse rather than fabricate them.
        return _dispatch_backtest(args)
    if args.command == "import-retained":
        retrieved_at = _parse_dt(args.retrieved_at)
        store = BronzeStore(Path(args.bronze_root))
        try:
            import_retained_stock_evidence(
                Path(args.source_root), store=store, retrieved_at=retrieved_at
            )
        except (PITDataError, ValueError, OSError):
            return 1
        return 0
    if args.command == "rebuild-data":
        try:
            from src.data.operations import (
                HistoricalDataPipelineRequest,
                run_historical_data_pipeline,
            )
            from src.integrations.dart.xbrl import DartXbrlCollector
            from src.integrations.kis.investor_flow import KisInvestorFlowCollector
            from src.integrations.krx.historical import KrxHistoricalCollector
            from src.integrations.quota import ProviderQuotaStateStore

            quota_store = ProviderQuotaStateStore(Path(args.artifact_root) / "quota")
            krx_collector = KrxHistoricalCollector(quota_store=quota_store)
            dart_collector = DartXbrlCollector()
            flow_symbols: tuple[str, ...]
            try:
                master = _load_silver_table(Path(args.silver_root), SilverTable.SECURITY_MASTER)
                flow_symbols = (
                    tuple(
                        sorted(
                            {
                                str(value).removeprefix("KRX:")
                                for value in master.get_column("instrument_id").to_list()
                                if str(value).strip()
                            }
                        )
                    )
                    if "instrument_id" in master.columns
                    else ()
                )
            except (FileNotFoundError, ValueError, PITDataError):
                flow_symbols = ()
            if not flow_symbols:
                raise PITDataError("rebuild-data requires a certified KRX security master for investor flow symbol planning")
            flow_collector = resolve_investor_flow_collector(str(args.investor_flow_provider), flow_symbols)
            pipeline_request = HistoricalDataPipelineRequest(
                data_root=Path(args.data_root),
                bronze_root=Path(args.bronze_root),
                silver_root=Path(args.silver_root),
                gold_root=Path(args.gold_root),
                artifact_root=Path(args.artifact_root),
                validation_start=date.fromisoformat(str(args.validation_start)),
                validation_end=date.fromisoformat(str(args.validation_end)),
                certification_time=_parse_dt(args.certification_time),
                resume=bool(args.resume),
                investor_flow_provider=str(args.investor_flow_provider),
            )
            pipeline_result = run_historical_data_pipeline(
                pipeline_request, krx=krx_collector, investor_flow=flow_collector, dart=dart_collector,
            )
            _emit({"plan_id": pipeline_result.plan_id, "certifiable": pipeline_result.certifiable, "result_path": str(pipeline_result.result_path)})
        except (PITDataError, ValueError, OSError):
            return 1
        return 0
    if args.command == "plan":
        try:
            symbols = (
                tuple(s.strip() for s in str(args.symbols).split(",") if s.strip())
                if args.symbols
                else None
            )
            if args.sessions:
                # Manual --sessions cannot bypass ordinary-share evidence. Route
                # the caller-supplied sessions through the same verified
                # master/market classifier; a preferred ticker is rejected here
                # rather than accepted through a caller-supplied boolean.
                if not symbols:
                    raise PITDataError("manual --sessions planning requires --symbols for verified classification")
                session_list = tuple(
                    date.fromisoformat(s.strip()) for s in str(args.sessions).split(",") if s.strip()
                )
                if not session_list:
                    raise PITDataError("manual --sessions list is empty")
                verified = build_historical_collection_plan_from_bronze(
                    bronze_root=Path(args.bronze_root),
                    start=date.fromisoformat(str(args.coverage_start)),
                    end=date.fromisoformat(str(args.coverage_end)),
                    chunk_size=int(args.chunk_size),
                    symbols=symbols,
                    artifact_root=Path(args.artifact_root),
                )
                wanted_cells = {(s, session) for s in symbols for session in session_list}
                covered = {(c.symbol, session) for c in verified.chunks for session in c.sessions}
                missing_cells = sorted(wanted_cells - covered)
                if missing_cells:
                    raise PITDataError(
                        f"manual sessions rejected by verified master classification: {missing_cells[0][0]}:{missing_cells[0][1].isoformat()}"
                    )
                plan = verified
            else:
                plan = build_historical_collection_plan_from_bronze(
                    bronze_root=Path(args.bronze_root),
                    start=date.fromisoformat(str(args.coverage_start)),
                    end=date.fromisoformat(str(args.coverage_end)),
                    chunk_size=int(args.chunk_size),
                    symbols=symbols,
                    artifact_root=Path(args.artifact_root),
                )
        except (PITDataError, ValueError, OSError):
            return 1
        try:
            receipt_raw = json.loads((Path(args.artifact_root) / f"{plan.plan_id}.json").read_text(encoding="utf-8"))
            requested_symbol_sessions = int(receipt_raw.get("requested_symbol_sessions", 0))
            non_trading = int(receipt_raw.get("non_trading_symbol_sessions", 0))
            plan_hash = str(receipt_raw.get("content_hash", plan.content_hash))
        except (OSError, ValueError, KeyError):
            requested_symbol_sessions = sum(len(c.sessions) for c in plan.chunks)
            non_trading = 0
            plan_hash = plan.content_hash
        _emit(
            {
                "plan_id": plan.plan_id,
                "chunks": len(plan.chunks),
                "requested_symbols": sorted({c.symbol for c in plan.chunks}),
                "requested_symbol_sessions": requested_symbol_sessions,
                "non_trading_symbol_sessions": non_trading,
                "plan_hash": plan_hash,
            }
        )
        return 0
    if args.command == "resume":
        try:
            resume_plan = load_collection_plan(str(args.plan_id))
            checkpoint_store = CollectionCheckpointStore(Path(args.checkpoint_root))
            pending = [
                chunk.chunk_id
                for chunk in resume_plan.chunks
                if not checkpoint_store.has_verified_receipt(
                    plan=resume_plan, chunk=chunk, bronze_root=Path(args.bronze_root)
                )
            ]
            _emit({"plan_id": resume_plan.plan_id, "pending": pending})
        except (PITDataError, ValueError, OSError):
            return 1
        return 0
    if args.command == "readiness":
        try:
            readiness_plan = load_collection_plan(str(args.plan_id))
            readiness_report = CollectionReadinessReport.incomplete(corporate_status_reason="unvalidated provider provenance")
            try:
                readiness_report.require_certifiable()
                certifiable = True
            except PITDataError:
                certifiable = False
            _emit({"plan_id": readiness_plan.plan_id, "certifiable": certifiable, "reasons": list(readiness_report.unresolved_reasons)})
        except (PITDataError, ValueError, OSError):
            return 1
        return 0
    if args.command == "probe-kis-flow":
        try:
            from src.integrations.kis.investor_flow import KisInvestorFlowCollector

            session = date.fromisoformat(str(args.session))
            page = KisInvestorFlowCollector((str(args.symbol),)).probe(str(args.symbol), session)
            records = page["records"]
            _emit(
                {
                    "provider": page["provider"],
                    "endpoint": page["endpoint"],
                    "symbol": str(args.symbol),
                    "requested_session": session.isoformat(),
                    "records": len(records) if isinstance(records, list) else 0,
                }
            )
        except (PITDataError, ValueError, OSError, RuntimeError):
            return 1
        return 0
    if args.command == "build-gold":
        try:
            from src.data.gold import materialize_gold_window
            from src.data.gold_loader import GoldWindowInputs

            decision_time = _parse_dt(args.decision_time)
            validation_start = date.fromisoformat(str(args.validation_start))
            validation_end = date.fromisoformat(str(args.validation_end))
            silver_root = Path(args.silver_root)
            dataset_ids = parse_silver_dataset_bindings(args.silver_dataset_id)
            dataset_ids = resolve_gold_dataset_bindings(silver_root=silver_root, requested=dataset_ids, decision_time=decision_time)

            inputs: GoldWindowInputs
            inputs = load_gold_window_inputs(silver_root=silver_root, validation_start=validation_start, validation_end=validation_end, decision_time=decision_time, silver_dataset_ids=dataset_ids)

            gold_target_root: Path | None = Path(args.gold_root) if args.gold_root else None
            from src.strategy.scoring import ChampionScorePolicy

            score_policy = ChampionScorePolicy()
            write_gold_input_binding_artifact(
                artifact_root=Path(args.artifact_root),
                dataset_ids=dataset_ids,
                decision_time=decision_time,
                validation_start=validation_start,
                validation_end=validation_end,
            )

            gold_report = materialize_gold_window(
                calendar=inputs.calendar, security_master=inputs.security_master, daily_market=inputs.daily_market, financial_facts=inputs.financial_facts, corporate_actions=inputs.corporate_actions, investor_flow=inputs.investor_flow,
                validation_start=validation_start,
                validation_end=validation_end,
                decision_time=decision_time,
                artifact_root=Path(args.artifact_root),
                gold_root=gold_target_root,
                silver_root=silver_root,
                score_policy=score_policy,
            )

            manifest = gold_report.manifest
            _emit(
                {
                    "manifest_hash": manifest.manifest_hash,
                    "warmup_ok": manifest.warmup.warmup_ok,
                    "warmup_sessions_found": manifest.warmup.warmup_sessions_found,
                    "bar_eligible": sum(1 for r in manifest.bar_audit if r.eligible),
                    "bar_ineligible": sum(1 for r in manifest.bar_audit if not r.eligible),
                    "dart_eligible": sum(1 for d in manifest.dart_eligibility if d.eligible),
                    "dart_ineligible": sum(1 for d in manifest.dart_eligibility if not d.eligible),
                    "ca_excluded": len(manifest.ca_excluded_instrument_ids),
                    "eligible_instruments": len(manifest.eligible_instrument_ids),
                    "universe_decisions_count": gold_report.universe_decisions_count,
                    "eligible_decisions_count": gold_report.eligible_decisions_count,
                    "feature_rows_count": gold_report.feature_rows_count,
                    "universe_path": gold_report.universe_path,
                    "features_path": gold_report.features_path,
                    "summary_artifact_path": gold_report.summary_artifact_path,
                }
            )
        except (PITDataError, ValueError, OSError) as exc:
            _emit({"error": str(exc)})
            return 1
        return 0
    if args.command == "backfill-daily-market":
        try:
            from src.data.operations import DailyMarketBackfillRequest, backfill_daily_market_coverage
            from src.integrations.krx.historical import KrxHistoricalCollector
            from src.integrations.quota import ProviderQuotaStateStore

            request = DailyMarketBackfillRequest(
                bronze_root=Path(args.bronze_root),
                silver_root=Path(args.silver_root),
                artifact_root=Path(args.artifact_root),
                validation_start=date.fromisoformat(str(args.validation_start)),
                validation_end=date.fromisoformat(str(args.validation_end)),
                decision_time=_parse_dt(args.decision_time),
            )
            quota_store = ProviderQuotaStateStore(Path(args.artifact_root) / "quota")
            result = backfill_daily_market_coverage(
                request,
                krx=KrxHistoricalCollector(quota_store=quota_store),
            )
        except (PITDataError, ValueError, OSError) as exc:
            _emit({"error": str(exc)})
            return 1
        _emit(
            {
                "history_start": result.history_start.isoformat(),
                "validation_start": result.validation_start.isoformat(),
                "validation_end": result.validation_end.isoformat(),
                "required_count": result.required_count,
                "covered_count": result.covered_count,
                "backfilled_count": len(result.backfilled_sessions),
                "missing_count": len(result.missing_sessions),
            }
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
