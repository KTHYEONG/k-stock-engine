"""PIT dataset foundation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from src.data.backtest_runner import run_champion_backtest
from src.data.backtest_sessions import build_backtest_sessions
from src.data.bronze import BronzeStore, import_retained_stock_evidence, migrate_retained_stock_evidence
from src.data.collection import ChampionCollectionRequest, CollectionArtifact, collect_champion_evidence
from src.data.collection_plan import (
    CollectionCheckpointStore,
    CollectionReadinessReport,
    build_historical_collection_plan,
    load_collection_plan,
)
from src.data.legacy_inventory import MigrationArtifactStore, inspect_legacy_data
from src.data.normalization import normalize_stock_evidence
from src.data.operations import execute_verified_legacy_purge, prepare_stock_data_rebuild
from src.data.pipeline import materialize_backtest_inputs
from src.data.schemas import PITDataError
from src.integrations.dart.client import DartApiClient
from src.integrations.krx.client import KrxApiClient


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
    p_col.add_argument("--coverage-start", type=str, required=True)
    p_col.add_argument("--coverage-end", type=str, required=True)
    p_col.add_argument("--retrieved-at", type=str, required=False, default=None)
    p_col.add_argument("--plan-id", type=str, required=False, default=None)

    p_mat = sub.add_parser("materialize", help="Materialize backtest inputs")
    p_mat.add_argument("--bronze-root", type=Path, default=Path("data/bronze/stocks"))
    p_mat.add_argument("--silver-root", type=Path, default=Path("data/silver/stocks"))
    p_mat.add_argument("--gold-root", type=Path, default=Path("data/gold/stocks"))
    p_mat.add_argument("--artifact-root", type=Path, default=Path("data/artifacts"))
    p_mat.add_argument("--decision-time", type=str, required=True)

    p_norm = sub.add_parser("normalize", help="Validate/normalize Bronze evidence")
    p_norm.add_argument("--bronze-root", type=Path, required=True)
    p_norm.add_argument("--decision-time", type=str, required=True)

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
    p_plan.add_argument("--coverage-start", type=str, required=True)
    p_plan.add_argument("--coverage-end", type=str, required=True)
    p_plan.add_argument("--symbols", type=str, required=True)
    p_plan.add_argument("--sessions", type=str, required=True)
    p_plan.add_argument("--chunk-size", type=int, default=20)

    p_resume = sub.add_parser("resume", help="Resume collection from checkpoints")
    p_resume.add_argument("--plan-id", type=str, required=True)
    p_resume.add_argument("--checkpoint-root", type=Path, default=Path("data/artifacts/collection-checkpoints"))

    p_readiness = sub.add_parser("readiness", help="Check collection readiness for certification")
    p_readiness.add_argument("--plan-id", type=str, required=True)

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


def _dispatch_backtest(args: argparse.Namespace) -> int:
    """Reserve the CLI boundary for fully resolved immutable run inputs."""
    if not Path(args.gold_root).exists():
        return 1
    # Config, strategy, and PIT repository are intentionally required inputs;
    # no synthetic objects are permitted at this boundary.
    raise PITDataError("run-backtest requires resolved Gold artifact, session repository, config, and strategy")


def _execute_backtest(**kwargs: object) -> object:
    """Typed adapter used once CLI input loaders provide concrete objects."""
    return run_champion_backtest(**kwargs)  # type: ignore[arg-type]


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
        if not getattr(args, "plan_id", None):
            return 1
        try:
            load_collection_plan(str(args.plan_id))
            try:
                krx_client = KrxApiClient()
                dart_client = DartApiClient()
            except ValueError:
                return 1

            class _KrxAdapter:
                def __init__(self, client: KrxApiClient) -> None:
                    self._client = client

                def fetch_market_snapshot(self, as_of: date) -> dict[str, object]:
                    return {"records": self._client.fetch_trade_records(as_of)}

                def fetch_flow_snapshot(self, as_of: date) -> dict[str, object]:
                    raise PITDataError("KRX investor-flow endpoint is not configured")

                def fetch_daily_market(self, start: date, end: date) -> tuple[dict[str, object], ...]:
                    return ({"records": self._client.fetch_trade_records(end)},)

                def fetch_investor_flow(self, start: date, end: date) -> tuple[dict[str, object], ...]:
                    raise PITDataError("KRX investor flow endpoint is not configured")

                def fetch_master_lineage(self, start: date, end: date) -> tuple[dict[str, object], ...]:
                    return ({"records": self._client.fetch_master_records(end)},)

                def fetch_status_and_actions(self, start: date, end: date) -> tuple[dict[str, object], ...]:
                    return ({"records": []},)

            class _DartAdapter:
                def __init__(self, client: DartApiClient) -> None:
                    self._client = client

                def fetch_fact_snapshot(self, start: date, end: date) -> dict[str, object]:
                    return {"records": self._client.list_disclosures(start, end)}

                def fetch_disclosures(self, start: date, end: date) -> tuple[dict[str, object], ...]:
                    return ({"records": self._client.list_disclosures(start, end)},)

                def fetch_xbrl_facts(self, filing_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
                    raise PITDataError("DART XBRL facts endpoint is not configured")

            request = ChampionCollectionRequest(
                bronze_root=Path(args.bronze_root),
                coverage_start=date.fromisoformat(str(args.coverage_start)),
                coverage_end=date.fromisoformat(str(args.coverage_end)),
                retrieved_at=_parse_dt(args.retrieved_at),
            )
            collection = collect_champion_evidence(
                request, krx=_KrxAdapter(krx_client), dart=_DartAdapter(dart_client)
            )
            if not isinstance(collection, CollectionArtifact):
                raise PITDataError("collection did not produce a Bronze artifact")
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"receipts": sorted(str(k.value) for k in collection.receipts), "content_hash": collection.content_hash})
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
            from src.data.pipeline import _load_bronze_receipts

            receipts = _load_bronze_receipts(Path(args.bronze_root))
            report_tables, report = normalize_stock_evidence(
                receipts, decision_time=_parse_dt(args.decision_time)
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"tables": sorted(str(k.value) for k in report_tables), "report_hash": report.report_hash})
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
            symbols = tuple(s.strip() for s in str(args.symbols).split(",") if s.strip())
            session_list = tuple(date.fromisoformat(s.strip()) for s in str(args.sessions).split(",") if s.strip())
            plan = build_historical_collection_plan(
                sessions=session_list,
                universe=tuple({"symbol": s, "is_common_stock": True, "tradable_from": None, "tradable_to": None} for s in symbols),
                start=date.fromisoformat(str(args.coverage_start)),
                end=date.fromisoformat(str(args.coverage_end)),
                chunk_size=int(args.chunk_size),
            )
        except (PITDataError, ValueError, OSError):
            return 1
        _emit({"plan_id": plan.plan_id, "chunks": len(plan.chunks)})
        return 0
    if args.command == "resume":
        try:
            resume_plan = load_collection_plan(str(args.plan_id))
            checkpoint_store = CollectionCheckpointStore(Path(args.checkpoint_root))
            pending = [c.chunk_id for c in resume_plan.chunks if checkpoint_store.is_pending(plan_id=resume_plan.plan_id, chunk_id=c.chunk_id, receipt_digest="")]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
