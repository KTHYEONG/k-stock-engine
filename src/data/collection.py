"""Official KRX/DART/KIS collection persisted to Bronze before parsing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.data.bronze import BronzeStore
from src.data.receipt_catalog import EvidenceStatus
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError
from src.data.scoped_ingestion import FACT_SOURCE, ScopedBronzeWriter, ScopedRawPayload, dart_fact_natural_key
from src.integrations.dart.xbrl import REPRT_QUARTER, DartCircuitOpenError

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
    receipt_count: int = 0
    planned_chunks: int = 0
    completed_chunks: int = 0
    previously_completed_chunks: int = 0
    pending_chunks: int = 0
    provider_error_chunks: int = 0
    missing_session_chunks: int = 0


def scoped_status_for_page(page: RawProviderResponse) -> EvidenceStatus:
    """Map provider page markers to the retained evidence status."""
    if str(page.get("status") or "").strip() == "extraction_failed":
        return EvidenceStatus.EXTRACTION_FAILED
    if str(page.get("source_kind") or "").strip() in {"unavailable", "blocked"}:
        return EvidenceStatus.PROVIDER_UNAVAILABLE
    records = page.get("records")
    if isinstance(records, list) and records:
        return EvidenceStatus.SUCCESS
    return EvidenceStatus.EMPTY


def dart_fact_scoped_payload(*, page: RawProviderResponse, retrieved_at: datetime) -> ScopedRawPayload:
    """Convert one validated DART fact page to a scoped payload with its adapter natural key."""
    identity = page.get("identity")
    identity_map = identity if isinstance(identity, dict) else {}
    corp_code = str(identity_map.get("corp_code") or page.get("corp_code") or "").strip()
    biz_year = str(identity_map.get("biz_year") or page.get("biz_year") or "").strip()
    reprt_code = str(identity_map.get("reprt_code") or page.get("reprt_code") or "").strip()
    if not corp_code or not biz_year or not reprt_code:
        raise PITDataError("DART fact page is missing its adapter natural key")
    # 수집기는 식별자에서 fiscal_period를 떼어내므로 (사업연도, 보고서 코드)에서 결정적으로 복원한다.
    quarter = REPRT_QUARTER.get(reprt_code)
    fiscal_period = (
        str(identity_map.get("fiscal_period") or page.get("fiscal_period") or "").strip()
        or (f"{biz_year}{quarter}" if quarter else None)
    )
    published = str(identity_map.get("published_at") or page.get("published_at") or "").strip()
    as_of = date.fromisoformat(published[:10]) if published else retrieved_at.date()
    natural_key = dart_fact_natural_key(corp_code=corp_code, biz_year=biz_year, reprt_code=reprt_code)
    return ScopedRawPayload(
        kind=EvidenceKind.FINANCIAL_FACTS,
        source=FACT_SOURCE,
        natural_key=natural_key,
        as_of=as_of,
        fiscal_period=fiscal_period,
        status=scoped_status_for_page(page),
        payload=json.dumps(dict(page), sort_keys=True, ensure_ascii=False).encode("utf-8"),
        retrieved_at=retrieved_at,
        source_label=f"opendart:fnlttSinglAcntAll:{natural_key}",
    )


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


def _parse_flow_session(value: Any) -> date:
    text = str(value or "").strip().replace("/", "-")
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise PITDataError(f"investor flow has malformed session: {value!r}") from exc


def _find_uncached_corp_codes(
    bronze_root: Path, *, corp_codes: tuple[str, ...], start: date, end: date
) -> tuple[str, ...]:
    needed = {str(c).strip() for c in corp_codes if str(c).strip()}
    if not needed:
        return ()
    disclosures_dir = Path(bronze_root) / "disclosures"
    if not disclosures_dir.exists():
        return tuple(sorted(needed))
    for payload_path in disclosures_dir.glob("*/payload.json"):
        if not needed:
            break
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            stored_code = str((payload.get("corp_code") if isinstance(payload, dict) else None) or "").strip()
            if stored_code in needed:
                stored_start = date.fromisoformat(str((payload.get("start") if isinstance(payload, dict) else None) or "").strip())
                stored_end = date.fromisoformat(str((payload.get("end") if isinstance(payload, dict) else None) or "").strip())
                if stored_start <= start and stored_end >= end:
                    needed.remove(stored_code)
        except (OSError, ValueError):
            continue
    return tuple(sorted(needed))


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
    if corp_codes is not None:
        normalized = tuple(str(c).strip() for c in corp_codes if str(c).strip())
        to_fetch = _find_uncached_corp_codes(bronze_root, corp_codes=normalized, start=start, end=end)
        if not to_fetch:
            content_hash = hashlib.sha256().hexdigest()
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
                        "page_receipts": [],
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
                receipts={},
                content_hash=content_hash,
                report_path=report_path,
                page_receipts={EvidenceKind.DISCLOSURES.value: ()},
            )
    if corp_codes is None:
        raw_pages = _collect_pages(dart.fetch_disclosures, start, end, kind_name="DART disclosures")
    else:
        raw_pages = _collect_pages(dart.fetch_disclosures, start, end, kind_name="DART disclosures", corp_codes=to_fetch)
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
    scoped_writer: ScopedBronzeWriter | None = None,
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
            if getattr(dart, "aborted", False):
                raise DartCircuitOpenError("DART transport failures tripped the circuit breaker before any page was collected")
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
    if scoped_writer is not None:
        # 페이지마다 publish하면 카탈로그 전체 스냅샷(수십 MB)이 페이지 수만큼 생성된다.
        scoped_writer.persist_many(
            tuple(dart_fact_scoped_payload(page=scoped_page, retrieved_at=retrieved_at) for scoped_page in persisted)
        )
    standardized = sum(1 for p in persisted if p.get("source_kind") == "opendart_standard")
    legacy_document = sum(
        1
        for p in persisted
        if p.get("source_kind") == "legacy_document" and p.get("status") != "extraction_failed"
    )
    unavailable = sum(1 for p in persisted if p.get("source_kind") == "unavailable")
    blocked = sum(1 for p in persisted if p.get("source_kind") == "blocked")
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
                "blocked": blocked,
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


def _collect_pages(
    fetch: Any,
    *args: Any,
    kind_name: str,
    **kwargs: Any,
) -> list[RawProviderResponse]:
    try:
        result = fetch(*args, **kwargs) if kwargs else fetch(*args)
    except TypeError:
        raise
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
