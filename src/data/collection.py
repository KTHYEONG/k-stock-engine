"""Official KRX/DART/KIS collection persisted to Bronze before parsing."""
from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from src.data.bronze import BronzeStore
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError


@dataclass(frozen=True, slots=True)
class ChampionCollectionRequest:
    bronze_root: Path
    coverage_start: date
    coverage_end: date
    retrieved_at: datetime


class KrxDataPort(Protocol):
    def fetch_market_snapshot(self, as_of: date) -> dict[str, Any]: ...
    def fetch_flow_snapshot(self, as_of: date) -> dict[str, Any]: ...


class DartFactPort(Protocol):
    def fetch_fact_snapshot(self, start: date, end: date) -> dict[str, Any]: ...


def _persist_response(
    store: BronzeStore,
    payload: dict[str, Any],
    *,
    kind: EvidenceKind,
    retrieved_at: datetime,
) -> BronzeReceipt:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        return store.import_json(tmp_path, kind=kind, retrieved_at=retrieved_at)
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


def collect_missing_champion_evidence(
    request: ChampionCollectionRequest, *, krx: KrxDataPort, dart: DartFactPort
) -> Mapping[EvidenceKind, BronzeReceipt]:
    if request.coverage_start > request.coverage_end:
        raise PITDataError("coverage_start must not be after coverage_end")
    if request.retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    store = BronzeStore(Path(request.bronze_root))
    receipts: dict[EvidenceKind, BronzeReceipt] = {}
    try:
        market_payload = krx.fetch_market_snapshot(request.coverage_end)
        flow_payload = krx.fetch_flow_snapshot(request.coverage_end)
    except Exception as exc:
        raise PITDataError(f"KRX collection failed: {exc}") from exc
    try:
        fact_payload = dart.fetch_fact_snapshot(request.coverage_start, request.coverage_end)
    except Exception as exc:
        raise PITDataError(f"DART collection failed: {exc}") from exc
    if not isinstance(market_payload, dict) or not market_payload:
        raise PITDataError("KRX market response is empty; refusing to fabricate facts")
    if not isinstance(flow_payload, dict) or not flow_payload:
        raise PITDataError("KRX investor-flow response is empty; certification blocked")
    if not isinstance(fact_payload, dict) or not fact_payload:
        raise PITDataError("DART fact response is empty; certification blocked")
    receipts[EvidenceKind.DAILY_MARKET] = _persist_response(
        store, market_payload, kind=EvidenceKind.DAILY_MARKET, retrieved_at=request.retrieved_at
    )
    receipts[EvidenceKind.INVESTOR_FLOW] = _persist_response(
        store, flow_payload, kind=EvidenceKind.INVESTOR_FLOW, retrieved_at=request.retrieved_at
    )
    receipts[EvidenceKind.FINANCIAL_FACTS] = _persist_response(
        store, fact_payload, kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=request.retrieved_at
    )
    return dict(receipts)
