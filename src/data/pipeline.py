"""Backtest-ready Gold materialization from research-certified Silver."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.core.datasets import DatasetCertification
from src.core.time import SessionCalendar
from src.data.bronze import BronzeStore
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable
from src.data.silver import certify_silver, load_latest_silver_table
from src.features.contracts import QvefFeaturePolicy, QvefFeatureRow
from src.features.materialize import materialize_qvef_features
from src.features.qvef import build_qvef_features
from src.storage.parquet_datasets import canonical_content_hash
from src.strategy.scoring import ChampionScorePolicy, ChampionScoreRow, materialize_champion_scores, score_champion_rows
from src.strategy.universe import (
    UniverseDecision,
    UniversePolicy,
    build_historical_universe,
    materialize_historical_universe,
)


@dataclass(frozen=True, slots=True)
class BacktestDataArtifact:
    universe_hash: str
    qvef_hash: str
    champion_scores_hash: str
    benchmark_cap_hash: str
    benchmark_equal_hash: str
    silver_report_hash: str
    content_hash: str


def _require_certified_inputs(silver_root: Path, bronze_root: Path) -> None:
    missing: list[str] = []
    if not Path(bronze_root).exists():
        missing.extend(["investor_flow", "financial_facts"])
    if not Path(silver_root).exists():
        for name in ("investor_flow", "financial_facts"):
            if name not in missing:
                missing.append(name)
    else:
        for table in SilverTable:
            table_dir = Path(silver_root) / table.value
            if not table_dir.exists():
                missing.append(table.value)
    if missing:
        ordered = sorted(set(missing))
        raise PITDataError(
            f"missing required tables: {', '.join(ordered)} (investor_flow, financial_facts)"
        )


def _load_bronze_receipts(bronze_root: Path) -> dict[EvidenceKind, BronzeReceipt]:
    from src.data.bronze_aggregation import aggregate_small_bronze_pages, discover_verified_bronze_receipts

    receipts: dict[EvidenceKind, BronzeReceipt] = {}
    root = Path(bronze_root)
    if not root.exists():
        return receipts
    grouped = discover_verified_bronze_receipts(bronze_root=root)
    if not grouped:
        return receipts
    store = BronzeStore(root)
    for kind in EvidenceKind:
        found = grouped.get(kind)
        if not found:
            continue
        if len(found) == 1:
            receipts[kind] = found[0]
        elif kind in (EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER):
            manifest = {
                "kind": kind.value,
                "input_receipt_hashes": [item.content_hash for item in found],
                "manifest": [
                    {"content_hash": item.content_hash, "retrieved_at": item.retrieved_at.isoformat()}
                    for item in found
                ],
            }
            receipts[kind] = store.import_bytes(
                json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"),
                kind=kind,
                retrieved_at=max(item.retrieved_at for item in found),
                source_label=f"manifest:{kind.value}",
            )
        else:
            receipts[kind] = aggregate_small_bronze_pages(
                kind=kind, receipts=tuple(found), store=store
            )
    return receipts


def materialize_backtest_inputs(
    *,
    bronze_root: Path,
    silver_root: Path,
    gold_root: Path,
    artifact_root: Path | None = None,
    decision_time: datetime,
    certification: DatasetCertification = DatasetCertification.RESEARCH,
) -> BacktestDataArtifact:
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    effective_cert = certification
    if effective_cert not in (DatasetCertification.RESEARCH, DatasetCertification.PRODUCTION):
        raise PITDataError("materialization requires RESEARCH-or-higher certification")
    _require_certified_inputs(Path(silver_root), Path(bronze_root))
    bronze_receipts = _load_bronze_receipts(Path(bronze_root))
    if len(bronze_receipts) != len(EvidenceKind):
        raise PITDataError("missing required Bronze receipts for certified Silver")
    silver = _load_silver_tables(Path(silver_root), decision_time)
    sessions = tuple(sorted(silver[SilverTable.CALENDAR]["session"].to_list()))
    if not sessions:
        raise PITDataError("calendar has no sessions")
    calendar = SessionCalendar(sessions)
    report = certify_silver(silver, receipts=bronze_receipts, coverage_start=min(s.astimezone(UTC).date() for s in sessions), coverage_end=max(s.astimezone(UTC).date() for s in sessions), certification=effective_cert)
    qvef_policy = QvefFeaturePolicy()
    score_policy = ChampionScorePolicy()
    all_universe_decisions: list[UniverseDecision] = []
    all_feature_rows: list[QvefFeatureRow] = []
    all_scores: list[ChampionScoreRow] = []
    for session in sessions:
        if session > decision_time:
            continue
        universe = build_historical_universe(decision_session=session, decision_time=session, calendar=calendar, security_master=silver[SilverTable.SECURITY_MASTER], daily_market=silver[SilverTable.DAILY_MARKET], policy=UniversePolicy())
        all_universe_decisions.extend(universe)
        rows = build_qvef_features(decision_session=session, decision_time=session, calendar=calendar, universe=universe, security_master=silver[SilverTable.SECURITY_MASTER], daily_market=silver[SilverTable.DAILY_MARKET], investor_flow=silver[SilverTable.INVESTOR_FLOW], financial_facts=silver[SilverTable.FINANCIAL_FACTS], policy=qvef_policy)
        if not rows:
            continue
        all_feature_rows.extend(rows)
        all_scores.extend(score_champion_rows(rows, decision_time=session, policy=score_policy))
    if not all_universe_decisions or not all_feature_rows or not all_scores:
        raise PITDataError("no PIT-complete Champion features available")
    universe_decisions = tuple(all_universe_decisions)
    dataset_id = hashlib.sha256(f"historical:{report.report_hash}".encode()).hexdigest()
    materialize_historical_universe(universe_decisions, root=Path(gold_root) / "universe", dataset_id=dataset_id, decision_time=decision_time, policy=UniversePolicy(), provider_version="official-pit-v1", calendar_hash=report.source_hashes[EvidenceKind.CALENDAR], master_hash=report.source_hashes[EvidenceKind.SECURITY_MASTER], quality_report_hash=report.report_hash, certification=effective_cert)
    materialize_qvef_features(tuple(all_feature_rows), root=Path(gold_root) / "qvef", dataset_id=dataset_id, decision_time=decision_time, policy=qvef_policy, provider_version="official-pit-v1", calendar_hash=report.source_hashes[EvidenceKind.CALENDAR], master_hash=report.source_hashes[EvidenceKind.SECURITY_MASTER], quality_report_hash=report.report_hash, certification=effective_cert)
    materialize_champion_scores(tuple(all_scores), root=Path(gold_root) / "champion_scores", dataset_id=dataset_id, decision_time=decision_time, policy=score_policy, provider_version="official-pit-v1", calendar_hash=report.source_hashes[EvidenceKind.CALENDAR], master_hash=report.source_hashes[EvidenceKind.SECURITY_MASTER], quality_report_hash=report.report_hash, certification=effective_cert)
    hashes = {"universe": dataset_id, "qvef": dataset_id, "champion_scores": dataset_id}
    artifact_dir = Path(artifact_root or Path(gold_root).parent / "artifacts") / "collections"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath(f"{report.report_hash}.json").write_text(json.dumps({"report_hash": report.report_hash, "gold_hashes": hashes}, sort_keys=True), encoding="utf-8")
    content_hash = canonical_content_hash(pl.DataFrame({"kind": list(hashes), "hash": list(hashes.values())}), ["kind", "hash"])
    return BacktestDataArtifact(hashes["universe"], hashes["qvef"], hashes["champion_scores"], "", "", report.report_hash, content_hash)


def _load_silver_tables(root: Path, decision_time: datetime) -> dict[SilverTable, pl.DataFrame]:
    result: dict[SilverTable, pl.DataFrame] = {}
    for table in SilverTable:
        try:
            result[table] = load_latest_silver_table(root=root, table=table, decision_time=decision_time)
        except (FileNotFoundError, ValueError) as exc:
            raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
    return result
