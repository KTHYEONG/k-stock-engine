"""Persist in-scope raw evidence and publish its receipt index state."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from src.data.bronze import BronzeStore
from src.data.receipt_catalog import CatalogRevision, EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import DataRuntime
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

__all__ = [
    "CORP_CODE_SOURCE",
    "FACT_SOURCE",
    "FLOW_SOURCE",
    "ScopedBronzeWriter",
    "ScopedRawPayload",
    "ScopedReceipt",
    "dart_fact_natural_key",
    "flow_natural_key",
]

CORP_CODE_SOURCE = "dart_corp_codes"
FACT_SOURCE = EvidenceKind.FINANCIAL_FACTS.value
FLOW_SOURCE = EvidenceKind.INVESTOR_FLOW.value

_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


def dart_fact_natural_key(*, corp_code: str, biz_year: str, reprt_code: str) -> str:
    """Compose the adapter-owned natural key for one DART fact identity."""
    return f"{corp_code}:{biz_year}:{reprt_code}"


def flow_natural_key(*, symbol: str, session: date) -> str:
    """Compose the adapter-owned natural key for one investor-flow observation."""
    return f"{symbol}:{session.isoformat()}"


@dataclass(frozen=True, slots=True)
class ScopedRawPayload:
    """Raw provider payload and its declared temporal identity inside one research scope."""

    kind: EvidenceKind
    source: str
    natural_key: str
    as_of: date | None
    fiscal_period: str | None
    status: EvidenceStatus
    payload: bytes
    retrieved_at: datetime
    source_label: str


@dataclass(frozen=True, slots=True)
class ScopedReceipt:
    """Hash-verified Bronze receipt and the catalog revision that exposes it to planning."""

    bronze_receipt: BronzeReceipt
    catalog_revision: CatalogRevision


class ScopedBronzeWriter:
    """Persist in-scope raw evidence and publish its receipt index state atomically enough for safe resume."""

    def __init__(self, *, runtime: DataRuntime, catalog: ReceiptCatalog) -> None:
        self._runtime = runtime
        self._catalog = catalog

    def persist(self, payload: ScopedRawPayload) -> ScopedReceipt:
        scope = self._runtime.scope
        if payload.retrieved_at.tzinfo is None:
            raise PITDataError("retrieved_at must be timezone-aware")
        if not payload.payload:
            raise PITDataError("cannot persist empty Bronze payload")
        if not payload.source.strip() or not payload.natural_key.strip():
            raise PITDataError("scoped payload requires an adapter-supplied source and natural key")
        if payload.as_of is None and payload.source != CORP_CODE_SOURCE:
            raise PITDataError(f"source {payload.source!r} must declare as_of")
        if payload.as_of is not None and payload.as_of < scope.evidence_start:
            raise PITDataError(f"scoped payload as_of {payload.as_of.isoformat()} precedes evidence start")
        if payload.kind == EvidenceKind.FINANCIAL_FACTS and payload.fiscal_period is not None:
            if not _FISCAL_PATTERN.fullmatch(payload.fiscal_period):
                raise PITDataError(f"invalid fiscal period {payload.fiscal_period!r}")
            if _fiscal_key(payload.fiscal_period) < _fiscal_key(scope.features.fundamental_fiscal_start):
                raise PITDataError(f"scoped payload fiscal period {payload.fiscal_period!r} precedes scope floor")
        if payload.kind == EvidenceKind.FINANCIAL_FACTS and payload.status == EvidenceStatus.SUCCESS and not payload.fiscal_period:
            raise PITDataError("successful financial facts require a fiscal period")
        store = BronzeStore(self._runtime.workspace.bronze_root)
        receipt = store.import_bytes(
            payload.payload,
            kind=payload.kind,
            retrieved_at=payload.retrieved_at,
            source_label=payload.source_label,
        )
        if hashlib.sha256(Path(receipt.payload_path).read_bytes()).hexdigest() != receipt.content_hash:
            raise PITDataError("hash verification failed before catalog publication")
        revision = self._catalog.publish(
            (
                ReceiptIndexEntry(
                    source=payload.source,
                    natural_key=payload.natural_key,
                    as_of=payload.as_of,
                    fiscal_period=payload.fiscal_period,
                    status=payload.status,
                    content_hash=receipt.content_hash,
                    retrieved_at=receipt.retrieved_at,
                    payload_path=receipt.payload_path,
                ),
            )
        )
        return ScopedReceipt(bronze_receipt=receipt, catalog_revision=revision)
