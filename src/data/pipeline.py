"""Backtest-ready Gold materialization from research-certified Silver."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.data.normalization import normalize_stock_evidence
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable


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
    decision_time: datetime,
) -> BacktestDataArtifact:
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    _require_certified_inputs(Path(silver_root), Path(bronze_root))
    bronze_receipts = _load_bronze_receipts(Path(bronze_root))
    if len(bronze_receipts) == len(EvidenceKind):
        normalize_stock_evidence(bronze_receipts, decision_time=decision_time)
    raise PITDataError(
        "Gold materialization requires persisted, provider-normalized Silver datasets; refusing fixture data"
    )
