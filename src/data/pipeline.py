"""Backtest-ready Gold materialization from research-certified Silver."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.core.datasets import DatasetCertification
from src.core.instruments import AssetKind
from src.core.time import SessionCalendar
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable
from src.data.silver import certify_silver
from src.features.contracts import QvefFeaturePolicy
from src.features.materialize import materialize_qvef_features
from src.features.qvef import build_qvef_features
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash
from src.strategy.scoring import ChampionScorePolicy, materialize_champion_scores, score_champion_rows
from src.strategy.universe import UniversePolicy, build_historical_universe


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
    receipts: dict[EvidenceKind, BronzeReceipt] = {}
    root = Path(bronze_root)
    if not root.exists():
        return receipts
    for kind in EvidenceKind:
        kind_dir = root / kind.value
        if not kind_dir.exists():
            continue
        for receipt_path in sorted(kind_dir.rglob("receipt.json")):
            try:
                meta = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            payload_path = receipt_path.parent / "payload.json"
            if not payload_path.exists():
                continue
            try:
                receipts[kind] = BronzeReceipt(
                    kind=kind,
                    content_hash=str(meta["content_hash"]),
                    source_path=str(meta.get("source_path", "")),
                    retrieved_at=datetime.fromisoformat(str(meta["retrieved_at"])),
                    ingested_at=datetime.fromisoformat(str(meta["ingested_at"])),
                    payload_path=payload_path,
                    metadata_path=receipt_path,
                )
            except (KeyError, ValueError):
                continue
            break
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
    hashes: dict[str, str] = {}
    for session in sessions:
        if session > decision_time:
            continue
        universe = build_historical_universe(decision_session=session, decision_time=decision_time, calendar=calendar, security_master=silver[SilverTable.SECURITY_MASTER], daily_market=silver[SilverTable.DAILY_MARKET], policy=UniversePolicy())
        rows = build_qvef_features(decision_session=session, decision_time=decision_time, calendar=calendar, universe=universe, security_master=silver[SilverTable.SECURITY_MASTER], daily_market=silver[SilverTable.DAILY_MARKET], investor_flow=silver[SilverTable.INVESTOR_FLOW], financial_facts=silver[SilverTable.FINANCIAL_FACTS], policy=qvef_policy)
        if not rows:
            continue
        dataset_id = hashlib.sha256(f"{session.isoformat()}:{report.report_hash}".encode()).hexdigest()
        materialize_qvef_features(rows, root=Path(gold_root) / "qvef", dataset_id=dataset_id, decision_time=decision_time, policy=qvef_policy, provider_version="official-pit-v1", calendar_hash=report.source_hashes[EvidenceKind.CALENDAR], master_hash=report.source_hashes[EvidenceKind.SECURITY_MASTER], quality_report_hash=report.report_hash, certification=effective_cert)
        scores = score_champion_rows(rows, decision_time=decision_time, policy=score_policy)
        materialize_champion_scores(scores, root=Path(gold_root) / "champion_scores", dataset_id=dataset_id, decision_time=decision_time, policy=score_policy, provider_version="official-pit-v1", calendar_hash=report.source_hashes[EvidenceKind.CALENDAR], master_hash=report.source_hashes[EvidenceKind.SECURITY_MASTER], quality_report_hash=report.report_hash, certification=effective_cert)
        hashes = {"qvef": dataset_id, "champion_scores": dataset_id}
        break
    if not hashes:
        raise PITDataError("no PIT-complete Champion features available")
    artifact_dir = Path(artifact_root or Path(gold_root).parent / "artifacts") / "collections"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath(f"{report.report_hash}.json").write_text(json.dumps({"report_hash": report.report_hash, "gold_hashes": hashes}, sort_keys=True), encoding="utf-8")
    content_hash = canonical_content_hash(pl.DataFrame({"kind": list(hashes), "hash": list(hashes.values())}), ["kind", "hash"])
    return BacktestDataArtifact("", hashes["qvef"], hashes["champion_scores"], "", "", report.report_hash, content_hash)


def _load_silver_tables(root: Path, decision_time: datetime) -> dict[SilverTable, pl.DataFrame]:
    result: dict[SilverTable, pl.DataFrame] = {}
    for table in SilverTable:
        table_root = root / table.value
        candidates = sorted((p for p in table_root.iterdir() if p.is_dir()), reverse=True) if table_root.exists() else []
        if not candidates:
            raise PITDataError(f"missing certified Silver table: {table.value}")
        try:
            result[table] = ParquetDatasetStore(table_root).read(candidates[0].name, AssetKind.STOCK, f"stock_pit_{table.value}_v1", decision_time)
        except (FileNotFoundError, ValueError) as exc:
            raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
    return result
