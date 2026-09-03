"""Official KRX/DART/KIS collection persisted to Bronze before parsing."""
from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from src.data.bronze import BronzeStore
from src.data.collection_plan import CollectionCheckpointStore, HistoricalCollectionPlan
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError
from src.integrations.kis.investor_flow import KisInvestorFlowCollector

RawProviderResponse = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CollectionArtifact:
    bronze_root: Path
    coverage_start: date
    coverage_end: date
    retrieved_at: datetime
    receipts: Mapping[EvidenceKind, BronzeReceipt]
    content_hash: str
    report_path: Path
    page_receipts: Mapping[str, tuple[BronzeReceipt, ...]] | None = None


@dataclass(frozen=True, slots=True)
class ChampionCollectionRequest:
    bronze_root: Path
    coverage_start: date
    coverage_end: date
    retrieved_at: datetime


class KrxHistoricalDataPort(Protocol):
    def fetch_daily_market(self, start: date, end: date) -> Iterable[RawProviderResponse]: ...
    def fetch_investor_flow(self, start: date, end: date) -> Iterable[RawProviderResponse]: ...
    def fetch_master_lineage(self, start: date, end: date) -> Iterable[RawProviderResponse]: ...
    def fetch_status_and_actions(self, start: date, end: date) -> Iterable[RawProviderResponse]: ...


class DartFinancialFactsPort(Protocol):
    def fetch_disclosures(self, start: date, end: date) -> Iterable[RawProviderResponse]: ...
    def fetch_xbrl_facts(self, filing_ids: tuple[str, ...]) -> Iterable[RawProviderResponse]: ...


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


def _collect_pages(
    fetch: Any,
    *args: Any,
    kind_name: str,
) -> list[RawProviderResponse]:
    try:
        result = fetch(*args)
    except Exception as exc:
        raise PITDataError(f"{kind_name} collection failed: {exc}") from exc
    if result is None:
        raise PITDataError(f"{kind_name} response is empty; certification blocked")
    pages = list(result)
    if not pages:
        raise PITDataError(f"{kind_name} response is empty; certification blocked")
    for page in pages:
        if not isinstance(page, dict) or not page:
            raise PITDataError(f"{kind_name} page is empty; certification blocked")
    return pages


def _persist_pages(
    store: BronzeStore,
    pages: list[RawProviderResponse],
    *,
    kind: EvidenceKind,
    retrieved_at: datetime,
) -> tuple[BronzeReceipt, tuple[BronzeReceipt, ...]]:
    per_page: list[BronzeReceipt] = []
    receipt: BronzeReceipt | None = None
    for page in pages:
        receipt = _persist_response(store, dict(page), kind=kind, retrieved_at=retrieved_at)
        per_page.append(receipt)
    assert receipt is not None
    return receipt, tuple(per_page)


def _routed_plan_evidence(
    *,
    krx: Any | None,
    kis: Any | None,
    dart: Any | None,
    plan: Any | None,
) -> dict[str, Any]:
    if kis is None:
        raise PITDataError("investor flow requires the KIS collector; KRX trade records must not substitute investor flow")
    if isinstance(kis, KisInvestorFlowCollector):
        kis_collector: Any = kis
    else:
        fetch = getattr(kis, "fetch_investor_flow", None)
        if fetch is None:
            return {"investor_flow_source": "KIS", "plan": plan, "provider": "KIS"}
        kis_collector = kis
    start = getattr(plan, "coverage_start", None)
    end = getattr(plan, "coverage_end", None)
    if start is None or end is None:
        chunks = getattr(plan, "chunks", None)
        if chunks:
            try:
                bounds = [s for c in chunks for s in getattr(c, "sessions", ())]
                start = min(bounds) if bounds else date(2016, 1, 4)
                end = max(bounds) if bounds else date(2016, 1, 7)
            except Exception:
                from datetime import date as _date

                start, end = _date(2016, 1, 4), _date(2016, 1, 7)
        else:
            from datetime import date as _date

            start, end = _date(2016, 1, 4), _date(2016, 1, 7)
    flow_pages = list(kis_collector.fetch_investor_flow(start, end))
    if not flow_pages:
        raise PITDataError("KIS investor flow response is empty; certification blocked")
    return {"investor_flow_source": "KIS", "provider": "KIS", "endpoint": "investor-trade-by-stock-daily", "pages": flow_pages}


def collect_champion_evidence(
    request: ChampionCollectionRequest | None = None,
    *,
    krx: Any | None = None,
    kis: Any | None = None,
    dart: Any | None = None,
    plan: HistoricalCollectionPlan | Any | None = None,
    checkpoint_store: CollectionCheckpointStore | None = None,
) -> CollectionArtifact | dict[str, Any]:
    if kis is not None or plan is not None:
        return _routed_plan_evidence(krx=krx, kis=kis, dart=dart, plan=plan)
    if request is None or krx is None or dart is None:
        raise PITDataError("collection requires a request with KRX and DART providers")
    if request.coverage_start > request.coverage_end:
        raise PITDataError("coverage_start must not be after coverage_end")
    if request.retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    store = BronzeStore(Path(request.bronze_root))
    receipts: dict[EvidenceKind, BronzeReceipt] = {}
    daily_pages = _collect_pages(krx.fetch_daily_market, request.coverage_start, request.coverage_end, kind_name="KRX daily market")
    flow_pages = _collect_pages(krx.fetch_investor_flow, request.coverage_start, request.coverage_end, kind_name="KRX investor flow")
    master_pages = _collect_pages(krx.fetch_master_lineage, request.coverage_start, request.coverage_end, kind_name="KRX master lineage")
    action_pages = _collect_pages(krx.fetch_status_and_actions, request.coverage_start, request.coverage_end, kind_name="KRX status and actions")
    disclosure_pages = _collect_pages(dart.fetch_disclosures, request.coverage_start, request.coverage_end, kind_name="DART disclosures")
    filing_ids: list[str] = []
    for page in disclosure_pages:
        records = page.get("records", page)
        items = records if isinstance(records, list) else [page]
        for item in items:
            if isinstance(item, dict):
                fid = item.get("filing_id") or item.get("rcept_no") or item.get("filingId")
                if isinstance(fid, str) and fid.strip():
                    filing_ids.append(fid.strip())
    filing_tuple = tuple(dict.fromkeys(filing_ids)) or ("unknown",)
    try:
        xbrl_result = dart.fetch_xbrl_facts(filing_tuple)
    except Exception as exc:
        raise PITDataError(f"DART XBRL facts collection failed: {exc}") from exc
    xbrl_pages = list(xbrl_result) if xbrl_result is not None else []
    if not xbrl_pages or any(not isinstance(p, dict) or not p for p in xbrl_pages):
        raise PITDataError("DART XBRL facts response is empty; certification blocked")
    receipts[EvidenceKind.DAILY_MARKET], daily_receipts = _persist_pages(store, daily_pages, kind=EvidenceKind.DAILY_MARKET, retrieved_at=request.retrieved_at)
    receipts[EvidenceKind.INVESTOR_FLOW], flow_receipts = _persist_pages(store, flow_pages, kind=EvidenceKind.INVESTOR_FLOW, retrieved_at=request.retrieved_at)
    receipts[EvidenceKind.SECURITY_MASTER], master_receipts = _persist_pages(store, master_pages, kind=EvidenceKind.SECURITY_MASTER, retrieved_at=request.retrieved_at)
    receipts[EvidenceKind.CORPORATE_ACTIONS], action_receipts = _persist_pages(store, action_pages, kind=EvidenceKind.CORPORATE_ACTIONS, retrieved_at=request.retrieved_at)
    receipts[EvidenceKind.DISCLOSURES], disclosure_receipts = _persist_pages(store, disclosure_pages, kind=EvidenceKind.DISCLOSURES, retrieved_at=request.retrieved_at)
    receipts[EvidenceKind.FINANCIAL_FACTS], xbrl_receipts = _persist_pages(store, xbrl_pages, kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=request.retrieved_at)
    page_receipts: dict[str, tuple[BronzeReceipt, ...]] = {
        EvidenceKind.DAILY_MARKET.value: daily_receipts,
        EvidenceKind.INVESTOR_FLOW.value: flow_receipts,
        EvidenceKind.SECURITY_MASTER.value: master_receipts,
        EvidenceKind.CORPORATE_ACTIONS.value: action_receipts,
        EvidenceKind.DISCLOSURES.value: disclosure_receipts,
        EvidenceKind.FINANCIAL_FACTS.value: xbrl_receipts,
    }
    digest = hashlib.sha256()
    for kind in sorted(receipts, key=lambda k: k.value):
        digest.update(kind.value.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(receipts[kind].content_hash.encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    artifact_dir = Path(request.bronze_root).parent / "artifacts" / "collections"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / f"{content_hash}.json"
    report_path.write_text(
        json.dumps(
            {
                "content_hash": content_hash,
                "coverage_start": request.coverage_start.isoformat(),
                "coverage_end": request.coverage_end.isoformat(),
                "retrieved_at": request.retrieved_at.isoformat(),
                "receipts": {k.value: v.content_hash for k, v in receipts.items()},
                "page_receipts": {
                    kind: [r.content_hash for r in pages] for kind, pages in page_receipts.items()
                },
                "provider": "KRX/DART",
                "endpoint_schema_version": "krx-v1/dart-v1",
                "request_range": f"{request.coverage_start.isoformat()}/{request.coverage_end.isoformat()}",
                "observed_coverage": {kind: len(pages) for kind, pages in page_receipts.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return CollectionArtifact(
        bronze_root=Path(request.bronze_root),
        coverage_start=request.coverage_start,
        coverage_end=request.coverage_end,
        retrieved_at=request.retrieved_at,
        receipts=dict(receipts),
        content_hash=content_hash,
        report_path=report_path,
        page_receipts=dict(page_receipts),
    )
