"""Verified stock-data rebuild and legacy purge orchestration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.data.bronze import migrate_retained_stock_evidence
from src.data.collection import ChampionCollectionRequest, CollectionArtifact, collect_champion_evidence
from src.data.collection_plan import CollectionReadinessReport
from src.data.legacy_inventory import MigrationArtifact, purge_legacy_data
from src.data.schemas import EvidenceKind, PITDataError


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
        has_all = all(k in source_hashes for k in EvidenceKind)
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
