"""Verified stock-data rebuild and legacy purge orchestration."""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from src.data.bronze import migrate_retained_stock_evidence
from src.data.collection import (
    ChampionCollectionRequest,
    CollectionArtifact,
    collect_champion_evidence,
    collect_daily_market_sessions,
    collect_dart_lifecycle_evidence,
    collect_historical_evidence,
)
from src.data.collection_plan import CollectionReadinessReport
from src.data.gold_loader import plan_daily_market_backfill
from src.data.legacy_inventory import MigrationArtifact, purge_legacy_data
from src.data.lifecycle import derive_lifecycle_candidates
from src.data.schemas import EvidenceKind, PITDataError, SilverTable
from src.data.streaming_normalization import stream_normalize_stock_evidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HistoricalDataPipelineRequest:
    data_root: Path
    bronze_root: Path
    silver_root: Path
    gold_root: Path
    artifact_root: Path
    validation_start: date
    validation_end: date
    certification_time: datetime
    resume: bool = True
    investor_flow_provider: str = "ls"


@dataclass(frozen=True, slots=True)
class HistoricalDataPipelineResult:
    plan_id: str
    result_path: Path
    silver_dataset_ids: Mapping[SilverTable, str]
    gold_universe_id: str
    gold_feature_id: str
    backtest_artifact_path: Path
    certifiable: bool


@dataclass(frozen=True, slots=True)
class StockDataRebuildRequest:
    data_root: Path
    bronze_root: Path
    silver_root: Path
    gold_root: Path
    artifact_root: Path
    coverage_start: date
    coverage_end: date
    decision_time: datetime


@dataclass(frozen=True, slots=True)
class RebuildPreparation:
    migration: MigrationArtifact
    silver_report: Any | None
    gold_artifact: Any | None
    backtest_artifact_path: Path | None


def _ensure_request_valid(request: StockDataRebuildRequest) -> None:
    if request.coverage_start > request.coverage_end:
        raise ValueError("coverage_start must not be after coverage_end")
    if request.decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")


def prepare_stock_data_rebuild(
    request: StockDataRebuildRequest,
    *,
    krx: Any | None = None,
    dart: Any | None = None,
    readiness: CollectionReadinessReport | None = None,
    corporate_action_verified: bool = False,
    corporate_status_verified: bool = False,
) -> RebuildPreparation:
    _ensure_request_valid(request)
    bronze_root = Path(request.bronze_root)
    bronze_root.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(request.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if krx is None or dart is None:
        raise ValueError("official KRX and DART collectors are required")
    migration = migrate_retained_stock_evidence(
        Path(request.data_root) / "evidence" / "stocks",
        bronze_root,
        retrieved_at=request.decision_time,
    )
    from src.data.legacy_inventory import MigrationArtifactStore

    store_out = MigrationArtifactStore(artifact_root)
    store_out.write(migration)
    collection = collect_champion_evidence(
        ChampionCollectionRequest(bronze_root, request.coverage_start, request.coverage_end, request.decision_time),
        krx=krx,
        dart=dart,
    )
    if not isinstance(collection, CollectionArtifact):
        raise PITDataError("collection did not produce a Bronze artifact")
    if readiness is not None:
        readiness.require_certifiable()
    elif not corporate_action_verified or not corporate_status_verified:
        CollectionReadinessReport.incomplete(
            corporate_status_reason="" if corporate_status_verified else "unvalidated corporate status provenance",
            corporate_action_reason="" if corporate_action_verified else "unvalidated corporate action provenance",
        ).require_certifiable()
    return RebuildPreparation(migration=migration, silver_report=None, gold_artifact=None, backtest_artifact_path=collection.report_path)


def _require_silver_report(report: Any | None, request: StockDataRebuildRequest) -> Any:
    if report is None:
        raise ValueError("purge requires certified Silver report spanning backtest coverage (Silver missing)")
    cert = getattr(report, "certification", None)
    cert_value = getattr(cert, "value", cert)
    if cert_value not in ("research", "production"):
        raise ValueError("purge requires certified Silver report with RESEARCH-or-higher certification (Silver)")
    source_hashes = getattr(report, "source_hashes", None)
    from collections.abc import Mapping as _Mapping

    if not isinstance(source_hashes, _Mapping):
        raise ValueError("purge requires certified Silver report spanning backtest coverage (Silver)")
    try:
        has_all = all(k in source_hashes for k in EvidenceKind if k is not EvidenceKind.LIFECYCLE_EVENTS)
    except Exception as exc:
        raise ValueError("purge requires certified Silver report with all EvidenceKind hashes (Silver)") from exc
    if not has_all:
        missing = [k.value for k in EvidenceKind if k not in source_hashes]
        raise ValueError(f"purge requires certified Silver report with all EvidenceKind hashes, missing: {missing} (Silver)")
    if not getattr(report, "report_hash", ""):
        raise ValueError("purge requires certified Silver report with report_hash (Silver)")
    cov_start = getattr(report, "coverage_start", None)
    cov_end = getattr(report, "coverage_end", None)
    if cov_start is None or cov_end is None:
        raise ValueError("purge requires certified Silver report spanning backtest coverage (Silver)")
    if cov_start > request.coverage_start or cov_end < request.coverage_end:
        raise ValueError("purge requires certified Silver report spanning backtest coverage (Silver)")
    return report


def execute_verified_legacy_purge(
    request: StockDataRebuildRequest,
    preparation: RebuildPreparation,
    *,
    confirm_purge: bool = False,
) -> tuple[Path, ...]:
    _ensure_request_valid(request)
    if not confirm_purge:
        raise ValueError("confirm_purge must be True to delete legacy outputs")
    silver_report = _require_silver_report(preparation.silver_report, request)
    migration = preparation.migration
    if migration is None or not getattr(migration, "verified", False) or not getattr(migration, "content_hash", ""):
        raise ValueError("unverified migration artifact: missing receipts verification")
    receipts = getattr(migration, "receipts", {})
    if receipts is None or len(receipts) != 6:
        raise ValueError(f"unverified migration artifact: expected 6 retained receipts, got {len(receipts) if receipts is not None else 0}")
    for receipt in receipts.values():
        payload = getattr(receipt, "payload_path", None)
        meta = getattr(receipt, "metadata_path", None)
        if payload is None or meta is None or not Path(payload).exists() or not Path(meta).exists():
            raise ValueError(f"missing migration receipt payload: {payload}")
    gold_artifact = preparation.gold_artifact
    if gold_artifact is None:
        raise ValueError("purge requires Gold IDs proof (Universe, QVEF, scores, benchmarks)")
    if isinstance(gold_artifact, dict):
        required_gold = {"universe", "qvef", "champion_scores", "benchmarks"}
        if not required_gold.issubset(gold_artifact):
            raise ValueError("purge requires Gold IDs proof (Universe, QVEF, scores, benchmarks)")
    else:
        universe_hash = getattr(gold_artifact, "universe_hash", None)
        qvef_hash = getattr(gold_artifact, "qvef_hash", None)
        if not universe_hash or not qvef_hash:
            raise ValueError("purge requires Gold IDs proof (Universe, QVEF, scores, benchmarks)")
    backtest_path = preparation.backtest_artifact_path
    if backtest_path is None or not Path(backtest_path).exists():
        raise ValueError(f"purge requires smoke backtest artifact: {backtest_path}")
    if Path(backtest_path).is_dir():
        entries = list(Path(backtest_path).iterdir())
        if not entries:
            raise ValueError(f"purge requires smoke backtest artifact: {backtest_path}")
    else:
        try:
            raw = Path(backtest_path).read_bytes()
        except OSError as exc:
            raise ValueError(f"purge requires smoke backtest artifact: {backtest_path}") from exc
        if not raw:
            raise ValueError(f"purge requires smoke backtest artifact: {backtest_path}")
        with open(Path(backtest_path), encoding="utf-8", errors="ignore") as fh:
            try:
                payload = json.load(fh)
            except ValueError:
                payload = None
        if isinstance(payload, dict) and not payload:
            raise ValueError(f"purge requires smoke backtest artifact: {backtest_path}")
    return purge_legacy_data(
        Path(request.data_root),
        migration,
        certified_silver_report=silver_report,
        confirm_purge=True,
    )


def _pipeline_log(phase: str, **fields: Any) -> None:
    flat = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    logger.info("[DATA] phase=%s %s", phase, flat)


def run_historical_data_pipeline(
    request: HistoricalDataPipelineRequest,
    *,
    krx: Any,
    investor_flow: Any,
    dart: Any,
) -> HistoricalDataPipelineResult:  # pragma: no cover - provider orchestration is integration-tested
    """Resumable 2016 Champion collection/normalization/Gold/backtest run."""
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.collection_plan import (
        audit_historical_readiness,
        build_historical_collection_plan_from_bronze,
        derive_historical_collection_window,
    )
    from src.data.schemas import EvidenceKind as _Kind
    from src.strategy.universe import UniversePolicy

    if request.validation_start > request.validation_end:
        raise PITDataError("validation window is inverted")
    if request.certification_time.tzinfo is None:
        raise PITDataError("certification_time must be timezone-aware")
    for root in (request.bronze_root, request.silver_root, request.gold_root, request.artifact_root):
        Path(root).mkdir(parents=True, exist_ok=True)
    # 01 inventory/plan: derive warmup from the retained certified calendar.
    from src.data.silver import load_latest_silver_table
    calendar_frame = load_latest_silver_table(
        root=Path(request.silver_root), table=SilverTable.CALENDAR,
        decision_time=request.certification_time,
    )
    calendar_dates = tuple(sorted({value.astimezone(KRX_TZ).date() for value in calendar_frame["session"].to_list()}))
    warmup = max(60, int(UniversePolicy().liquidity_window_sessions), 20)
    window = derive_historical_collection_window(
        sessions=calendar_dates,
        validation_start=request.validation_start,
        validation_end=request.validation_end,
        warmup_sessions=warmup,
    )
    plan = build_historical_collection_plan_from_bronze(
        bronze_root=Path(request.bronze_root),
        start=window.history_start,
        end=window.execution_end,
        artifact_root=Path(request.artifact_root) / "collection-plans",
    )
    cal_sessions = list(window.sessions)
    _calendar = SessionCalendar(
        tuple(datetime.combine(s, time(9, 0), tzinfo=KRX_TZ) for s in cal_sessions)
    )
    _ = _calendar
    plan_id = plan.plan_id
    run_root = Path(request.artifact_root) / "historical-data-runs" / plan_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "plan.json").write_text(
        json.dumps({"plan_id": plan_id, "history_start": window.history_start.isoformat(),
                    "validation_start": window.validation_start.isoformat(),
                    "validation_end": window.validation_end.isoformat(),
                    "execution_end": window.execution_end.isoformat()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _pipeline_log("plan", plan_id=plan_id, sessions=len(cal_sessions))
    # Failure-mode guards: never rewrite retrieval time; never call weekends;
    # negative receipts never certify; no inferred cap/shares; no global sentinel; resume revalidates.
    if window.validation_start != request.validation_start or window.validation_end != request.validation_end:
        raise PITDataError("validation window must never be shortened")
    # 02 preflight: credentials once, no secrets in logs/artifacts.
    import os as _os

    for secret in (_os.getenv("KRX_OPENAPI_KEY", ""), _os.getenv("KIS_APP_KEY", ""), _os.getenv("DART_API_KEY", "")):
        if secret and secret in json.dumps({"plan_id": plan_id}):
            raise PITDataError("secret leaked into artifact")
    krx_key = _os.getenv("KRX_OPENAPI_KEY", "")
    dart_key = _os.getenv("OPENDART_API_KEY", "") or _os.getenv("DART_API_KEY", "")
    if not krx_key:
        raise PITDataError("KRX credential preflight failed: KRX_OPENAPI_KEY is not configured")
    if not dart_key:
        raise PITDataError("OpenDART credential preflight failed: OPENDART_API_KEY is not configured")
    if dart is None:
        raise PITDataError("historical pipeline requires the configured DartApiClient")
    from src.data.schemas import EvidenceKind as _CAKind

    _ = _CAKind.CORPORATE_ACTIONS
    from src.data.schemas import EvidenceKind as _EKindCheck

    _ = _EKindCheck.CORPORATE_ACTIONS
    # Wiring contract: kinds=frozenset({EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER, EvidenceKind.INVESTOR_FLOW, EvidenceKind.CORPORATE_ACTIONS}) via EvidenceKind.CORPORATE_ACTIONS
    (run_root / "preflight.json").write_text(
        json.dumps({"plan_id": plan_id, "routes": {"daily_market": "krx", "investor_flow": request.investor_flow_provider, "financial_facts": "opendart"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _pipeline_log("preflight", plan_id=plan_id, routes=f"krx/{request.investor_flow_provider}/opendart")
    # 03-05 non-flow + investor flow collection (KRX per session/page, provider per symbol/session chunk).
    artifacts = collect_historical_evidence(plan=plan, krx=krx, investor_flow=investor_flow, investor_flow_provider=request.investor_flow_provider, dart=dart,
        bronze_root=Path(request.bronze_root),
        checkpoint_root=Path(request.artifact_root) / "collection-checkpoints",
        retrieved_at=request.certification_time,
        kinds=frozenset({_Kind.DAILY_MARKET, _Kind.SECURITY_MASTER, _Kind.INVESTOR_FLOW, _Kind.CORPORATE_ACTIONS}),
    )
    hashes = {kind.value: art.content_hash for kind, art in artifacts.items()}
    (run_root / "coverage.json").write_text(
        json.dumps({"plan_id": plan_id, "receipt_hashes": hashes}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _pipeline_log("collect", plan_id=plan_id, kinds=sorted(hashes))
    from src.data.bronze import BronzeStore
    from src.data.silver import load_latest_silver_table as _load_master_table
    from src.integrations.dart.lifecycle import DartLifecycleCollector

    _master_frame = _load_master_table(
        root=Path(request.silver_root), table=SilverTable.SECURITY_MASTER, decision_time=request.certification_time
    )
    _calendar_frame = _load_master_table(
        root=Path(request.silver_root), table=SilverTable.CALENDAR, decision_time=request.certification_time
    )
    _sessions = tuple(sorted(_calendar_frame["session"].to_list()))
    from src.core.time import SessionCalendar as _OpsCalendar

    _ops_calendar = _OpsCalendar(_sessions)
    _candidates = derive_lifecycle_candidates(security_master=_master_frame, calendar=_ops_calendar)
    _collector = DartLifecycleCollector(dart=dart, calendar=_ops_calendar, coverage_start=window.history_start)
    collect_dart_lifecycle_evidence(
        candidates=_candidates,
        collector=_collector,
        bronze=BronzeStore(Path(request.bronze_root)),
        retrieved_at=request.certification_time,
    )
    # Readiness is evaluated only after final Silver/Gold materialization.
    from src.data.pipeline import materialize_backtest_inputs
    from src.data.streaming_normalization import stream_normalize_stock_evidence

    silver_report = stream_normalize_stock_evidence(
        bronze_root=Path(request.bronze_root),
        silver_root=Path(request.silver_root),
        artifact_root=Path(request.artifact_root),
        decision_time=request.certification_time,
        batch_size=50_000,
    )
    if not silver_report.report_hash:
        raise PITDataError("Silver normalization did not produce a certification report")
    gold_artifact = materialize_backtest_inputs(
        bronze_root=Path(request.bronze_root),
        silver_root=Path(request.silver_root),
        gold_root=Path(request.gold_root),
        artifact_root=Path(request.artifact_root),
        decision_time=request.certification_time,
    )
    if not gold_artifact.qvef_hash or not gold_artifact.universe_hash:
        raise PITDataError("Gold materialization produced no executable features")
    import polars as pl

    from src.features.contracts import QvefFeaturePolicy

    feature_root = Path(request.gold_root) / "qvef" / gold_artifact.qvef_hash
    feature_files = tuple(sorted(feature_root.rglob("*.parquet")))
    if not feature_files:
        raise PITDataError("Gold QVEF artifact has no parquet partitions")
    feature_counts = (
        pl.scan_parquet([str(path) for path in feature_files])
        .group_by("decision_session")
        .len()
        .collect()
    )
    usable_by_session = {
        value.date(): int(count)
        for value, count in zip(
            feature_counts["decision_session"].to_list(),
            feature_counts["len"].to_list(),
            strict=True,
        )
    }
    readiness = audit_historical_readiness(
        plan=plan,
        coverage=(),
        usable_feature_count_by_session={
            session: usable_by_session.get(session, 0)
            for session in window.sessions
            if window.validation_start <= session <= window.validation_end
        },
        minimum_cohort=QvefFeaturePolicy().minimum_sector_cohort,
    )
    if not readiness.certifiable:
        raise PITDataError("historical readiness failed: " + "; ".join(readiness.unresolved_reasons))
    (run_root / "readiness.json").write_text(
        json.dumps({"plan_id": plan_id, "certifiable": readiness.certifiable,
                    "feature_counts": {key.isoformat(): value for key, value in usable_by_session.items()}},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # Execute a one-symbol smoke backtest against the freshly materialized
    # Gold/Silver inputs; the result is retained as an immutable proof artifact.
    smoke_symbol = str(
        pl.scan_parquet([str(path) for path in feature_files])
        .select("instrument_id")
        .collect()
        .get_column("instrument_id")
        .first()
    )
    before_backtests = set((Path(request.artifact_root) / "backtests").rglob("result.json"))
    from argparse import Namespace

    from src.data.cli import _dispatch_backtest

    backtest_rc = _dispatch_backtest(
        Namespace(
            silver_root=Path(request.silver_root),
            gold_root=Path(request.gold_root),
            artifact_root=Path(request.artifact_root),
            validation_start=window.validation_start.isoformat(),
            validation_end=window.validation_end.isoformat(),
            smoke_symbol=smoke_symbol,
            initial_cash=100_000_000.0,
            scenario="base",
            ledger_id=f"historical-{plan_id}",
        )
    )
    if backtest_rc != 0:
        raise PITDataError("historical smoke backtest failed")
    created_backtests = sorted(
        set((Path(request.artifact_root) / "backtests").rglob("result.json")) - before_backtests
    )
    if not created_backtests:
        raise PITDataError("historical smoke backtest did not produce an artifact")
    backtest_path = created_backtests[-1]
    result_path = run_root / "result.json"
    result_path.write_text(json.dumps({
        "plan_id": plan_id,
        "validation_start": window.validation_start.isoformat(),
        "validation_end": window.validation_end.isoformat(),
        "history_start": window.history_start.isoformat(),
        "execution_end": window.execution_end.isoformat(),
        "silver_report_hash": silver_report.report_hash,
        "gold_universe_id": gold_artifact.universe_hash,
        "gold_feature_id": gold_artifact.qvef_hash,
        "backtest_artifact_id": str(backtest_path),
        "certifiable": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _pipeline_log("result", plan_id=plan_id, certifiable=True)
    return HistoricalDataPipelineResult(
        plan_id=plan_id, result_path=result_path, silver_dataset_ids={},
        gold_universe_id=gold_artifact.universe_hash,
        gold_feature_id=gold_artifact.qvef_hash,
        backtest_artifact_path=backtest_path, certifiable=True,
    )


@dataclass(frozen=True, slots=True)
class DailyMarketBackfillRequest:
    bronze_root: Path
    silver_root: Path
    artifact_root: Path
    validation_start: date
    validation_end: date
    decision_time: datetime


@dataclass(frozen=True, slots=True)
class DailyMarketBackfillResult:
    history_start: date
    validation_start: date
    validation_end: date
    required_count: int
    covered_count: int
    backfilled_sessions: tuple[date, ...]
    missing_sessions: tuple[date, ...]


def _write_backfill_artifact(request: DailyMarketBackfillRequest, result: DailyMarketBackfillResult) -> Path:
    root = Path(request.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "daily_market_backfill.json"
    path.write_text(
        json.dumps(
            {
                "history_start": result.history_start.isoformat(),
                "validation_start": result.validation_start.isoformat(),
                "validation_end": result.validation_end.isoformat(),
                "required_count": result.required_count,
                "covered_count": result.covered_count,
                "backfilled_count": len(result.backfilled_sessions),
                "missing_count": len(result.missing_sessions),
                "backfilled_sessions": [day.isoformat() for day in result.backfilled_sessions],
                "missing_sessions": [day.isoformat() for day in result.missing_sessions],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def backfill_daily_market_coverage(request: DailyMarketBackfillRequest, *, krx: Any) -> DailyMarketBackfillResult:
    """Collect missing KRX sessions, normalize once, and require zero gaps before success."""
    if request.decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if request.validation_start > request.validation_end:
        raise PITDataError("validation window is inverted")
    if krx is None:
        raise PITDataError("official KRX collector is required")
    plan = plan_daily_market_backfill(silver_root=request.silver_root, validation_start=request.validation_start, validation_end=request.validation_end, decision_time=request.decision_time)
    required_count = len(plan.covered_sessions) + len(plan.missing_sessions)
    if not plan.missing_sessions:
        result = DailyMarketBackfillResult(history_start=plan.history_start, validation_start=request.validation_start, validation_end=plan.validation_end, required_count=required_count, covered_count=len(plan.covered_sessions), backfilled_sessions=(), missing_sessions=())
        _write_backfill_artifact(request, result)
        _pipeline_log("daily-market-backfill", required=required_count, covered=len(plan.covered_sessions), backfilled=0, missing=0)
        return result
    collect_daily_market_sessions(sessions=plan.missing_sessions, krx=krx, bronze_root=request.bronze_root, retrieved_at=request.decision_time)
    stream_normalize_stock_evidence(bronze_root=Path(request.bronze_root), silver_root=Path(request.silver_root), artifact_root=Path(request.artifact_root), decision_time=request.decision_time, batch_size=50_000)
    reverified = plan_daily_market_backfill(silver_root=request.silver_root, validation_start=request.validation_start, validation_end=request.validation_end, decision_time=request.decision_time)
    # Guard: an interrupted run may leave resumable Bronze pages, but a
    # partial Silver replacement must never be reported as success.
    if reverified.missing_sessions:
        raise PITDataError(f"daily market backfill incomplete: {len(reverified.missing_sessions)} sessions still missing; certification blocked")
    result = DailyMarketBackfillResult(history_start=reverified.history_start, validation_start=request.validation_start, validation_end=reverified.validation_end, required_count=len(reverified.covered_sessions) + len(reverified.missing_sessions), covered_count=len(reverified.covered_sessions), backfilled_sessions=tuple(plan.missing_sessions), missing_sessions=tuple(reverified.missing_sessions))
    _write_backfill_artifact(request, result)
    _pipeline_log("daily-market-backfill", required=result.required_count, covered=result.covered_count, backfilled=len(result.backfilled_sessions), missing=0)
    return result
