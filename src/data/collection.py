"""Official KRX/DART/KIS collection persisted to Bronze before parsing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from src.data.bronze import BronzeStore
from src.data.collection_plan import CollectionCheckpointStore, HistoricalCollectionPlan
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError
from src.integrations.dart.xbrl import DartXbrlCollector  # noqa: F401
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
    return store.import_bytes(
        text.encode("utf-8"),
        kind=kind,
        retrieved_at=retrieved_at,
        source_label=f"normalized-provider-page:{kind.value}",
    )


def collect_planned_investor_flow(
    *,
    plan: HistoricalCollectionPlan,
    kis: KisInvestorFlowCollector,
    bronze_root: Path,
    retrieved_at: datetime,
    checkpoint_store: CollectionCheckpointStore,
) -> CollectionArtifact:
    """Collect only verified KIS investor-flow chunks and make each completion resumable."""
    if retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    store = BronzeStore(bronze_root)
    receipts: list[BronzeReceipt] = []
    page_receipts: list[BronzeReceipt] = []
    for chunk in plan.chunks:
        if checkpoint_store.has_verified_receipt(plan=plan, chunk=chunk, bronze_root=bronze_root):
            continue
        try:
            pages = tuple(
                kis.fetch_investor_flow(
                    min(chunk.sessions),
                    max(chunk.sessions),
                    bronze_root=bronze_root,
                    retrieved_at=retrieved_at,
                    symbols=(chunk.symbol,),
                )
            )
        except PITDataError as exc:
            if "missing requested session" not in str(exc):
                raise
            negative = json.dumps(
                {
                    "provider": "KIS",
                    "endpoint": "investor-trade-by-stock-daily",
                    "symbol": chunk.symbol,
                    "sessions": [value.isoformat() for value in chunk.sessions],
                    "status": "source_unavailable",
                },
                sort_keys=True,
            ).encode("utf-8")
            receipt = store.import_bytes(
                negative,
                kind=EvidenceKind.INVESTOR_FLOW,
                retrieved_at=retrieved_at,
                source_label=f"KIS:source-unavailable:{chunk.symbol}:{chunk.chunk_id}",
            )
            checkpoint_store.mark_complete(
                plan_id=plan.plan_id,
                chunk_id=chunk.chunk_id,
                receipt_digest=receipt.content_hash,
                plan_digest=plan.content_hash,
                receipt_hashes=(receipt.content_hash,),
            )
            page_receipts.append(receipt)
            continue
        expected_sessions = {value.isoformat() for value in chunk.sessions}
        observed_sessions: set[str] = set()
        for page in pages:
            records = page.get("records", [])
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict) and str(record.get("ticker")) == chunk.symbol:
                    observed_sessions.add(str(record.get("session")))
        if not expected_sessions.issubset(observed_sessions):
            missing = ",".join(sorted(expected_sessions - observed_sessions))
            raise PITDataError(f"KIS investor flow missing requested sessions for {chunk.symbol}: {missing}")
        raw_receipts: list[BronzeReceipt] = []
        for page in pages:
            page_receipt = page.get("bronze_receipt")
            if isinstance(page_receipt, BronzeReceipt):
                raw_receipts.append(page_receipt)
        if not raw_receipts:
            raise PITDataError("KIS investor flow raw Bronze receipt is missing")
        digest = hashlib.sha256()
        for receipt in sorted(raw_receipts, key=lambda value: value.content_hash):
            digest.update(receipt.content_hash.encode("utf-8"))
            digest.update(b"\x00")
            page_receipts.append(receipt)
        receipt_digest = digest.hexdigest()
        checkpoint_store.mark_complete(
            plan_id=plan.plan_id,
            chunk_id=chunk.chunk_id,
            receipt_digest=receipt_digest,
            plan_digest=plan.content_hash,
            receipt_hashes=tuple(receipt.content_hash for receipt in raw_receipts),
        )
        receipts.extend(raw_receipts)
    digest = hashlib.sha256()
    for receipt in sorted(page_receipts, key=lambda value: value.content_hash):
        digest.update(receipt.content_hash.encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    artifact_dir = bronze_root.parent / "artifacts" / "collections"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / f"{content_hash}.json"
    report_path.write_text(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "content_hash": content_hash,
                "provider": "KIS",
                "endpoint": "investor-trade-by-stock-daily",
                "coverage_start": plan.coverage_start.isoformat(),
                "coverage_end": plan.coverage_end.isoformat(),
                "completed_chunks": len(plan.chunks),
                "page_receipts": [receipt.content_hash for receipt in page_receipts],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    latest = receipts[-1] if receipts else None
    return CollectionArtifact(
        bronze_root=bronze_root,
        coverage_start=plan.coverage_start,
        coverage_end=plan.coverage_end,
        retrieved_at=retrieved_at,
        receipts={EvidenceKind.INVESTOR_FLOW: latest} if latest is not None else {},
        content_hash=content_hash,
        report_path=report_path,
        page_receipts={EvidenceKind.INVESTOR_FLOW.value: tuple(page_receipts)},
    )


def collect_dart_disclosures(
    *,
    dart: Any,
    start: date,
    end: date,
    bronze_root: Path,
    retrieved_at: datetime,
    corp_codes: tuple[str, ...] | None = None,
) -> CollectionArtifact:
    """Persist DART disclosure records to Bronze disclosures before filing resolution."""
    if retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    try:
        if corp_codes is None:
            raw_pages = _collect_pages(dart.fetch_disclosures, start, end, kind_name="DART disclosures")
        else:
            raw_pages = _collect_pages(dart.fetch_disclosures, start, end, kind_name="DART disclosures", corp_codes=tuple(corp_codes))
    except TypeError:
        raw_pages = _collect_pages(dart.fetch_disclosures, start, end, kind_name="DART disclosures")
    store = BronzeStore(bronze_root)
    receipt, page_receipts = _persist_pages(
        store, raw_pages, kind=EvidenceKind.DISCLOSURES, retrieved_at=retrieved_at
    )
    digest = hashlib.sha256()
    for page_receipt in page_receipts:
        digest.update(page_receipt.content_hash.encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    artifact_dir = bronze_root.parent / "artifacts" / "collections"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / f"{content_hash}.json"
    report_path.write_text(
        json.dumps(
            {
                "content_hash": content_hash,
                "provider": "OpenDART",
                "endpoint": "list",
                "coverage_start": start.isoformat(),
                "coverage_end": end.isoformat(),
                "page_receipts": [item.content_hash for item in page_receipts],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return CollectionArtifact(
        bronze_root=bronze_root,
        coverage_start=start,
        coverage_end=end,
        retrieved_at=retrieved_at,
        receipts={EvidenceKind.DISCLOSURES: receipt},
        content_hash=content_hash,
        report_path=report_path,
        page_receipts={EvidenceKind.DISCLOSURES.value: page_receipts},
    )


def collect_dart_financial_facts(
    *,
    dart: Any,
    identities: tuple[dict[str, str], ...],
    bronze_root: Path,
    retrieved_at: datetime,
) -> CollectionArtifact:
    """Persist full-statement DART responses before downstream fact normalization."""
    from src.data.dart_documents import DartDocumentStore

    if retrieved_at.tzinfo is None:
        raise PITDataError("retrieved_at must be timezone-aware")
    if not identities:
        raise PITDataError("DART financial facts require filing identities")
    fetch = getattr(dart, "fetch_financial_fact_sources", None)
    if fetch is None:
        fetch = dart.fetch_xbrl_facts
        raw_pages = _collect_pages(fetch, identities, kind_name="DART full statements")
    else:
        try:
            result = dart.fetch_financial_fact_sources(identities)
        except Exception as exc:
            raise PITDataError(f"DART full statements collection failed: {exc}") from exc
        if result is None:
            raise PITDataError("DART full statements response is empty; certification blocked")
        raw_pages = list(result)
        if not raw_pages:
            raise PITDataError("DART full statements response is empty; certification blocked")
        for page in raw_pages:
            if not isinstance(page, dict) or not page:
                raise PITDataError("DART full statements page is empty; certification blocked")
    store = BronzeStore(bronze_root)
    document_store = DartDocumentStore(bronze_root)
    persisted: list[dict[str, Any]] = []
    for page in raw_pages:
        serializable = {k: v for k, v in dict(page).items() if k != "raw_archive"}
        archive = page.get("raw_archive")
        if isinstance(archive, (bytes, bytearray)) and len(archive) > 0:
            from collections.abc import Mapping as _Mapping

            raw_identity = page.get("identity")
            identity: _Mapping[str, Any] = raw_identity if isinstance(raw_identity, dict) else {}
            rcept_no = str(
                identity.get("rcept_no") or identity.get("filing_id") or page.get("rcept_no") or page.get("filing_id") or ""
            ).strip()
            if len(rcept_no) == 14 and rcept_no.isdigit():
                receipt_doc = document_store.store_archive(
                    bytes(archive), rcept_no=rcept_no, retrieved_at=retrieved_at
                )
                serializable["document_receipt"] = str(receipt_doc.metadata_path)
                if not serializable.get("raw_document_hash"):
                    serializable["raw_document_hash"] = receipt_doc.content_hash
        persisted.append(serializable)
    receipt, page_receipts = _persist_pages(
        store, persisted, kind=EvidenceKind.FINANCIAL_FACTS, retrieved_at=retrieved_at
    )
    standardized = sum(1 for p in persisted if p.get("source_kind") == "opendart_standard")
    legacy_document = sum(
        1
        for p in persisted
        if p.get("source_kind") == "legacy_document" and p.get("status") != "extraction_failed"
    )
    unavailable = sum(1 for p in persisted if p.get("source_kind") == "unavailable")
    extraction_failed = sum(1 for p in persisted if p.get("status") == "extraction_failed")
    filing_ids = [
        str(p.get("filing_id") or (p.get("identity") or {}).get("filing_id") or "").strip()
        for p in persisted
    ]
    digest = hashlib.sha256()
    for page_receipt in page_receipts:
        digest.update(page_receipt.content_hash.encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    artifact_dir = bronze_root.parent / "artifacts" / "collections"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / f"{content_hash}.json"
    report_path.write_text(
        json.dumps(
            {
                "content_hash": content_hash,
                "provider": "OpenDART",
                "endpoint": "fnlttSinglAcntAll",
                "filing_count": len(identities),
                "standardized": standardized,
                "legacy_document": legacy_document,
                "unavailable": unavailable,
                "extraction_failed": extraction_failed,
                "filing_ids": filing_ids,
                "page_receipts": [item.content_hash for item in page_receipts],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return CollectionArtifact(
        bronze_root=bronze_root,
        coverage_start=retrieved_at.date(),
        coverage_end=retrieved_at.date(),
        retrieved_at=retrieved_at,
        receipts={EvidenceKind.FINANCIAL_FACTS: receipt},
        content_hash=content_hash,
        report_path=report_path,
        page_receipts={EvidenceKind.FINANCIAL_FACTS.value: page_receipts},
    )


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
    **kwargs: Any,
) -> list[RawProviderResponse]:
    try:
        result = fetch(*args, **kwargs) if kwargs else fetch(*args)
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
    import re as _re

    _period = _re.compile(r"\((\d{4})\.(\d{2})\)")
    _code_by_month = {"03": "11013", "06": "11012", "09": "11014", "12": "11011"}
    identities: list[dict[str, str]] = []
    seen_fids: set[str] = set()
    fallback_fids: list[str] = []
    for page in disclosure_pages:
        records = page.get("records", page)
        items = records if isinstance(records, list) else [page]
        for item in items:
            if not isinstance(item, dict):
                continue
            fid = str(item.get("filing_id") or item.get("rcept_no") or item.get("filingId") or "").strip()
            if not fid or fid in seen_fids:
                continue
            seen_fids.add(fid)
            fallback_fids.append(fid)
            corp = str(item.get("corp_code") or item.get("company_id") or "").strip()
            report_nm = str(item.get("report_nm") or item.get("filing_type") or "")
            matched = _period.search(report_nm)
            biz_year = str(item.get("biz_year") or item.get("bsns_year") or "").strip()
            reprt_code = str(item.get("reprt_code") or item.get("report_code") or "").strip()
            if matched:
                year, month = matched.groups()
                if not biz_year:
                    biz_year = year
                if not reprt_code:
                    if "사업보고서" in report_nm:
                        reprt_code = "11011"
                    elif "반기보고서" in report_nm:
                        reprt_code = "11012"
                    elif "분기보고서" in report_nm:
                        reprt_code = _code_by_month.get(month, "")
                    else:
                        reprt_code = _code_by_month.get(month, "")
            if not corp or not biz_year or not reprt_code:
                continue
            identities.append(
                {
                    "corp_code": corp,
                    "filing_id": fid,
                    "biz_year": biz_year,
                    "reprt_code": reprt_code,
                    "fs_div": "CFS",
                }
            )
    if not fallback_fids:
        raise PITDataError("DART disclosures contain no filing identities")
    filing_tuple = tuple(identities) if identities else tuple(dict.fromkeys(fallback_fids))
    if not filing_tuple:
        raise PITDataError("DART disclosures contain no periodic filing identities")
    try:
        if identities and hasattr(dart, "fetch_financial_fact_sources"):
            xbrl_result = dart.fetch_financial_fact_sources(tuple(identities))
        elif hasattr(dart, "fetch_xbrl_facts"):
            xbrl_result = dart.fetch_xbrl_facts(
                tuple(d["filing_id"] if isinstance(d, dict) else d for d in filing_tuple)
            )
        elif identities:
            xbrl_result = dart.fetch_financial_fact_sources(tuple(identities))
        else:
            raise PITDataError("DART disclosures contain no periodic filing identities")
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
