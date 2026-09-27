"""DART page shapes adapted to the scoped Bronze writer."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from src.core.pit import EvidenceKind, PITDataError
from src.data.receipt_catalog import EvidenceStatus
from src.data.scoped_ingestion import FACT_SOURCE, ScopedRawPayload, dart_fact_natural_key
from src.integrations.dart.xbrl import REPRT_QUARTER

RawProviderResponse = dict[str, Any]


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
