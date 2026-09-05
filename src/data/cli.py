"""PIT dataset foundation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.data.backtest_runner import run_champion_backtest
from src.data.backtest_sessions import build_backtest_sessions
from src.data.bronze import BronzeStore, import_retained_stock_evidence, migrate_retained_stock_evidence
from src.data.collection import collect_dart_disclosures, collect_dart_financial_facts, collect_planned_investor_flow
from src.data.collection_plan import (
    CollectionCheckpointStore,
    CollectionReadinessReport,
    build_historical_collection_plan,
    build_historical_collection_plan_from_bronze,
    load_collection_plan,
)
from src.data.legacy_inventory import MigrationArtifactStore, inspect_legacy_data
from src.data.operations import execute_verified_legacy_purge, prepare_stock_data_rebuild
from src.data.pipeline import materialize_backtest_inputs
from src.data.schemas import PITDataError, SilverTable


def _parse_args() -> argparse.Namespace:
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

    p_dart = sub.add_parser("collect-dart-facts", help="Collect periodic OpenDART full statements from retained disclosures")
    p_dart.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
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

    # normalize-dart-facts --bronze-root ... --silver-root ... --artifact-root ... --decision-time ... --batch-size ...
    # CLI registration: add_argument("normalize-dart-facts") subcommand (via add_parser) with those flags.
    p_dart_refresh = sub.add_parser("normalize-dart-facts", help="Incremental DART fact refresh")
    p_dart_refresh.add_argument("--bronze-root", type=Path, required=True)
    p_dart_refresh.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_dart_refresh.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_dart_refresh.add_argument("--decision-time", type=str, required=True)
    p_dart_refresh.add_argument("--batch-size", type=int, default=500)

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
    p_run.add_argument("--gold-root", type=Path, default=Path("data/gold/stocks"))
    p_run.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_run.add_argument("--validation-start", type=str, default="2016-01-04")
    p_run.add_argument("--validation-end", type=str, default="2016-12-30")
    p_run.add_argument("--smoke-symbol", type=str, default=None)
    p_run.add_argument("--initial-cash", type=float, default=100000000.0)
    p_run.add_argument("--scenario", type=str, default="base")
    p_run.add_argument("--ledger-id", type=str, default="champion-2016")

    p_rebuild = sub.add_parser("rebuild-data", help="Prepare verified rebuild before collection")
    p_rebuild.add_argument("--data-root", type=Path, default=Path("data"))
    p_rebuild.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_rebuild.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_rebuild.add_argument("--gold-root", type=Path, default=Path("data/gold/stocks"))
    p_rebuild.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_rebuild.add_argument("--coverage-start", type=str, required=True)
    p_rebuild.add_argument("--coverage-end", type=str, required=True)
    p_rebuild.add_argument("--decision-time", type=str, required=False, default=None)

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
    p_plan.add_argument("--chunk-size", type=int, default=20)

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

    return parser.parse_args()


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


def _load_silver_table(silver_root: Path, table: SilverTable) -> Any:
    from src.data.silver import load_latest_silver_table

    return load_latest_silver_table(root=silver_root, table=table, decision_time=datetime.now(UTC))


def _dispatch_backtest(args: argparse.Namespace) -> int:
    """Execute Champion backtest or single-instrument smoke test with replayable artifacts."""
    from datetime import date

    import polars as pl

    from src.core.costs import (
        LiquiditySlippageModel,
        TickSizeRule,
        TickSizeSchedule,
        default_base_schedule,
    )
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.data.backtest_runner import run_managed_backtest
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository
    from src.engine.backtest import BacktestConfig
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    silver_root = Path(getattr(args, "silver_root", "data/silver/stocks"))
    artifact_root = Path(getattr(args, "artifact_root", "data/artifacts"))
    gold_root = Path(getattr(args, "gold_root", "data/gold/stocks"))
    val_start = date.fromisoformat(str(getattr(args, "validation_start", "2016-01-04")))
    val_end = date.fromisoformat(str(getattr(args, "validation_end", "2016-12-30")))
    smoke_symbol = getattr(args, "smoke_symbol", None)

    if not smoke_symbol and (not gold_root.exists() or not (gold_root / "universe").exists()):
        raise PITDataError("run-backtest requires resolved Gold artifact, session repository, config, and strategy")

    initial_cash = float(getattr(args, "initial_cash", 100_000_000.0))
    scenario_str = str(getattr(args, "scenario", "base")).lower()
    scenario = ExecutionScenario.BASE if scenario_str == "base" else ExecutionScenario.IDEAL

    # Load calendar and normalize session times to 09:00:00 KST
    calendar_df = _load_silver_table(silver_root, SilverTable.CALENDAR)
    raw_sessions = tuple(sorted(calendar_df["session"].to_list()))
    cal_sessions = tuple(s.replace(hour=9, minute=0, second=0) for s in raw_sessions)
    calendar = SessionCalendar(cal_sessions)

    val_sessions = [s for s in cal_sessions if val_start <= s.date() <= val_end]
    if not val_sessions:
        raise PITDataError(f"No calendar sessions between {val_start} and {val_end}")

    start_session = val_sessions[0]
    end_session = val_sessions[-1]
    end_idx = cal_sessions.index(end_session)
    if end_idx + 1 >= len(cal_sessions):
        raise PITDataError(f"Coverage exhausted: no session after {end_session}")
    next_session = cal_sessions[end_idx + 1]

    # Load market bars
    dm_root = silver_root / SilverTable.DAILY_MARKET.value
    parquet_files = list(dm_root.rglob("*.parquet"))
    if not parquet_files:
        raise PITDataError("missing daily market parquet files")

    query = (
        pl.scan_parquet(parquet_files)
        .filter((pl.col("session") >= start_session) & (pl.col("session") <= next_session))
    )
    if smoke_symbol:
        query = query.filter(pl.col("instrument_id") == smoke_symbol)
    daily_market = query.collect()
    if daily_market.height == 0:
        raise PITDataError("no daily market data in range")

    # Available_at normalization to 15:30 KST
    daily_market_pit = daily_market.with_columns(
        pl.col("session").dt.replace(hour=15, minute=30, second=0).alias("available_at")
    )
    snapshot_repo = PITSnapshotRepository.from_frames(
        {SilverTable.DAILY_MARKET: daily_market_pit}, root=silver_root
    )

    sessions = build_backtest_sessions(
        snapshot_repository=snapshot_repo,
        calendar=calendar,
        start=start_session,
        end=end_session,
        decision_time_of=lambda s: s.replace(hour=15, minute=30, second=0),
    )

    distinct_symbols = daily_market["instrument_id"].unique().to_list()
    instruments = {
        sym: Instrument(sym, AssetKind.STOCK, "KRX", sym.split(":")[-1], "KRW")
        for sym in distinct_symbols
    }

    costs = default_base_schedule()
    ticks = TickSizeSchedule((
        TickSizeRule("all", datetime(2000, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1000.0),
    ))
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.1, ticks),
        scenario,
        target_participation_cap=0.1,
        hard_participation_cap=0.2,
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
                            target_value=initial_cash * 0.5,
                            decision_time=context.decision_time,
                            execution_time=sessions_ordered[1],
                            strategy_id="champion-v1",
                            reason="smoke_entry",
                            idempotency_key=f"smoke_entry_{target_sym}",
                            account_snapshot_id=context.portfolio.account_snapshot_id,
                        ),
                    )
                return ()

        strategy: Any = SmokeStrategy()
    else:
        eligible_set: set[str] = set()
        universe_root = gold_root / "universe"
        if universe_root.exists():
            u_files = list(universe_root.rglob("*.parquet"))
            if u_files:
                u_df = pl.scan_parquet(u_files).filter(pl.col("eligible")).collect()
                if u_df.height > 0:
                    eligible_set = set(u_df["instrument_id"].to_list())

        class UniverseStrategy:
            def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
                if not eligible_set:
                    return ()
                per_stock = (initial_cash * 0.8) / len(eligible_set)
                current_open = context.decision_time.replace(hour=9, minute=0, second=0)
                idx = sessions_ordered.index(current_open)
                if idx == 0 and idx + 1 < len(sessions_ordered):
                    return tuple(
                        TradeIntent(
                            intent_id=f"champion-buy-{sym}",
                            asset_kind=AssetKind.STOCK,
                            instrument_id=sym,
                            target_value=per_stock,
                            decision_time=context.decision_time,
                            execution_time=sessions_ordered[idx + 1],
                            strategy_id="champion-v1",
                            reason="universe_entry",
                            idempotency_key=f"universe_{sym}_{idx}",
                            account_snapshot_id=context.portfolio.account_snapshot_id,
                        )
                        for sym in sorted(eligible_set)
                    )
                return ()

        strategy = UniverseStrategy()

    _result, manifest = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=strategy,
        artifact_root=artifact_root,
        dataset_hash=f"validation_{val_start}_{val_end}",
        smoke_symbol=smoke_symbol,
        extra_metadata={
            "validation_start": str(val_start),
            "validation_end": str(val_end),
            "instruments_tracked": len(instruments),
        },
    )

    _emit({
        "content_hash": manifest["content_hash"],
        "ledger_id": manifest["ledger_id"],
        "scenario": manifest["scenario"],
        "session_count": manifest["session_count"],
        "fill_count": manifest["fill_count"],
        "reject_count": manifest["reject_count"],
        "smoke_symbol": smoke_symbol,
        "performance": manifest["performance"],
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


def main() -> int:
    args = _parse_args()
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
    if args.command == "collect":
        try:
            from src.integrations.kis.investor_flow import KisInvestorFlowCollector

            plan = load_collection_plan(str(args.plan_id))
            collector = KisInvestorFlowCollector(tuple(sorted({chunk.symbol for chunk in plan.chunks})))
            collection = collect_planned_investor_flow(
                plan=plan,
                kis=collector,
                bronze_root=Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
                checkpoint_store=CollectionCheckpointStore(Path(args.checkpoint_root)),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"receipts": sorted(str(k.value) for k in collection.receipts), "content_hash": collection.content_hash})
        return 0
    if args.command == "collect-dart-disclosures":
        try:
            from src.integrations.dart.xbrl import DartXbrlCollector

            collection = collect_dart_disclosures(
                dart=DartXbrlCollector(),
                start=date.fromisoformat(str(args.coverage_start)),
                end=date.fromisoformat(str(args.coverage_end)),
                bronze_root=Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"content_hash": collection.content_hash, "receipts": sorted(str(k.value) for k in collection.receipts)})
        return 0
    if args.command == "collect-dart-facts":
        try:
            from src.integrations.dart.xbrl import DartXbrlCollector

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
            collection = collect_dart_financial_facts(dart=DartXbrlCollector(), identities=batch,
                bronze_root=Path(args.bronze_root),
                retrieved_at=_parse_dt(args.retrieved_at),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"filings": len(batch), "content_hash": collection.content_hash})
        return 0
    if args.command == "backfill-dart-facts":
        try:
            from src.data.dart_backfill import DartHistoricalBackfillRequest, run_dart_historical_backfill_batch
            from src.integrations.dart.xbrl import DartXbrlCollector

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
                request=backfill_request, dart=DartXbrlCollector()
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"plan_id": backfill_plan.plan_id, "required_periods": list(backfill_plan.required_periods)})
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
            from src.data.operations import StockDataRebuildRequest
            from src.integrations.dart.xbrl import DartXbrlCollector
            from src.integrations.krx.historical import KrxHistoricalCollector

            krx_collector = KrxHistoricalCollector()
            dart_collector = DartXbrlCollector()
            rebuild_request = StockDataRebuildRequest(
                data_root=Path(args.data_root),
                bronze_root=Path(args.bronze_root),
                silver_root=Path(args.silver_root),
                gold_root=Path(args.gold_root),
                artifact_root=Path(args.artifact_root),
                coverage_start=date.fromisoformat(str(args.coverage_start)),
                coverage_end=date.fromisoformat(str(args.coverage_end)),
                decision_time=_parse_dt(args.decision_time),
            )
            preparation = prepare_stock_data_rebuild(rebuild_request, krx=krx_collector, dart=dart_collector)
            if len(preparation.migration.receipts) != 6:
                return 1
            _emit({"content_hash": preparation.migration.content_hash, "receipts": len(preparation.migration.receipts), "coverage_start": str(rebuild_request.coverage_start), "coverage_end": str(rebuild_request.coverage_end)})
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
                session_list = tuple(date.fromisoformat(s.strip()) for s in str(args.sessions).split(",") if s.strip())
                plan = build_historical_collection_plan(
                    sessions=session_list,
                    universe=tuple(
                        {"symbol": symbol, "is_common_stock": True, "tradable_from": None, "tradable_to": None}
                        for symbol in symbols or ()
                    ),
                    start=date.fromisoformat(str(args.coverage_start)),
                    end=date.fromisoformat(str(args.coverage_end)),
                    chunk_size=int(args.chunk_size),
                    artifact_root=Path(args.artifact_root),
                )
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
        _emit({"plan_id": plan.plan_id, "chunks": len(plan.chunks)})
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
            from src.data.gold_loader import GoldWindowInputs, load_gold_window_inputs

            decision_time = _parse_dt(args.decision_time)
            validation_start = date.fromisoformat(str(args.validation_start))
            validation_end = date.fromisoformat(str(args.validation_end))
            silver_root = Path(args.silver_root)

            inputs: GoldWindowInputs
            inputs = load_gold_window_inputs(silver_root=silver_root, validation_start=validation_start, validation_end=validation_end, decision_time=decision_time)

            gold_target_root: Path | None = Path(args.gold_root) if args.gold_root else None

            gold_report = materialize_gold_window(
                calendar=inputs.calendar, security_master=inputs.security_master, daily_market=inputs.daily_market, financial_facts=inputs.financial_facts, corporate_actions=inputs.corporate_actions, investor_flow=inputs.investor_flow,
                validation_start=validation_start,
                validation_end=validation_end,
                decision_time=decision_time,
                artifact_root=Path(args.artifact_root),
                gold_root=gold_target_root,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
