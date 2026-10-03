"""The only Bronze writer: validate in-scope payloads against their source contract and publish them."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

from src.core.pit import BronzeReceipt, EvidenceKind, PITDataError
from src.data.bronze import BronzeStore
from src.data.evidence_sources import (
    DART_CORP_CODES_SOURCE,
    FINANCIAL_FACTS_SOURCE,
    CoverageShape,
    SourceContract,
    canonical_raw_rows_bytes,
    envelope_carries_rows,
    source_contract,
    validate_envelope,
)
from src.data.receipt_catalog import (
    BlobEntry,
    CatalogRevision,
    CoverageRange,
    EvidenceStatus,
    ReceiptCatalog,
    ReceiptIndexEntry,
)
from src.data.runtime import DataRuntime

__all__ = [
    "CORP_CODE_SOURCE",
    "FACT_SOURCE",
    "ScopedBronzeWriter",
    "ScopedRangePayload",
    "ScopedRawPayload",
    "ScopedReceipt",
    "dart_fact_natural_key",
    "flow_natural_key",
]

CORP_CODE_SOURCE = DART_CORP_CODES_SOURCE
FACT_SOURCE = FINANCIAL_FACTS_SOURCE

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
class ScopedRangePayload:
    """One answered range-shaped request: the raw envelope and the ranges it answers."""

    source: str
    payload: bytes | None  # None only when every range is ``empty``
    ranges: tuple[CoverageRange, ...]
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

    def _validate(self, payload: ScopedRawPayload) -> None:
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

    def _contract_for(self, source: str, coverage: CoverageShape, *, kind: EvidenceKind | None = None) -> SourceContract:
        contract = source_contract(source)
        if contract.coverage is not coverage:
            expected = "keyed" if coverage is CoverageShape.KEYED else "ranged"
            raise PITDataError(
                f"source {source!r} records {contract.coverage.value} coverage and does not accept a {expected} payload"
            )
        if kind is not None and kind is not contract.kind:
            raise PITDataError(f"source {source!r} stores {contract.kind.value} evidence, not {kind.value}")
        return contract

    def _store(
        self, payload: bytes, *, kind: EvidenceKind, retrieved_at: datetime, source_label: str
    ) -> BronzeReceipt:
        store = BronzeStore(self._runtime.workspace.bronze_root)
        receipt = store.import_bytes(payload, kind=kind, retrieved_at=retrieved_at, source_label=source_label)
        if hashlib.sha256(Path(receipt.payload_path).read_bytes()).hexdigest() != receipt.content_hash:
            raise PITDataError("hash verification failed before catalog publication")
        return receipt

    def _validate_range_payload(self, payload: ScopedRangePayload) -> SourceContract:
        if payload.retrieved_at.tzinfo is None:
            raise PITDataError("retrieved_at must be timezone-aware")
        if not payload.source.strip() or not payload.source_label.strip():
            raise PITDataError("scoped payload requires an adapter-supplied source and source label")
        if not payload.ranges:
            raise PITDataError("scoped range payload requires at least one answered range")
        contract = self._contract_for(payload.source, CoverageShape.RANGED)
        evidence_start = self._runtime.scope.evidence_start
        for item in payload.ranges:
            if item.source != payload.source:
                raise PITDataError(f"range source {item.source!r} does not match payload source {payload.source!r}")
            if item.end < item.start:
                raise PITDataError(f"range for {item.subject!r} ends before it starts")
            if item.start < evidence_start:
                raise PITDataError(f"range start {item.start.isoformat()} precedes evidence start")
            if item.status is not EvidenceStatus.SUCCESS and item.content_hash is not None:
                raise PITDataError(f"{item.status.value} range for {item.subject!r} must not reference a blob")
        if payload.payload is None:
            if any(item.status is not EvidenceStatus.EMPTY for item in payload.ranges):
                raise PITDataError("a range payload without bytes cannot answer a successful range")
        else:
            # 저장된 페이로드의 성격은 바이트가 정하므로, 호출자가 success를 주장해도 행이 없으면 성공이 아니다.
            status = (
                EvidenceStatus.SUCCESS if envelope_carries_rows(contract, payload.payload) else EvidenceStatus.EMPTY
            )
            if status is EvidenceStatus.EMPTY and any(
                item.status is not EvidenceStatus.EMPTY for item in payload.ranges
            ):
                raise PITDataError("a successful range needs an envelope that carries rows")
            validate_envelope(contract, payload.payload, status=status)
        return contract

    def validate(self, payload: ScopedRawPayload) -> SourceContract:
        """Check one keyed payload against the scope and its source contract without storing anything.

        Raises:
            PITDataError: the payload violates the scope window or its source contract.
        """
        self._validate(payload)
        contract = self._contract_for(payload.source, CoverageShape.KEYED, kind=payload.kind)
        validate_envelope(contract, payload.payload, status=payload.status)
        return contract

    def persist_many(
        self, payloads: Sequence[ScopedRawPayload | ScopedRangePayload]
    ) -> tuple[ScopedReceipt, ...]:
        """Validate, store and catalog one bounded batch in a single catalog revision.

        Every payload is checked against its ``SourceContract`` before any byte is
        written, so a malformed page cannot enter Bronze. Each stored payload is
        registered as a usable blob in the same revision that publishes its
        receipts or ranges. A range payload whose ranges are all ``empty``
        carries no bytes and therefore contributes no receipt.

        Raises:
            PITDataError: an unregistered source, a contract violation, a keyed
                payload for a ranged source (or the reverse), or any existing
                scope check (evidence start, fiscal floor, tz-aware time).
        """
        if not payloads:
            return ()
        entries: list[ReceiptIndexEntry] = []
        blobs: list[BlobEntry] = []
        ranges: list[CoverageRange] = []
        receipts: list[BronzeReceipt] = []
        for payload in payloads:
            if isinstance(payload, ScopedRangePayload):
                contract = self._validate_range_payload(payload)
                if payload.payload is None:
                    ranges.extend(payload.ranges)
                    continue
                receipt = self._store(
                    self._canonical_payload(contract, payload.payload),
                    kind=contract.kind,
                    retrieved_at=payload.retrieved_at,
                    source_label=payload.source_label,
                )
                receipts.append(receipt)
                blobs.append(self._blob(receipt, contract))
                ranges.extend(
                    replace(item, content_hash=receipt.content_hash)
                    if item.status is EvidenceStatus.SUCCESS
                    else item
                    for item in payload.ranges
                )
                continue
            contract = self.validate(payload)
            receipt = self._store(
                self._canonical_payload(contract, payload.payload),
                kind=contract.kind,
                retrieved_at=payload.retrieved_at,
                source_label=payload.source_label,
            )
            receipts.append(receipt)
            blobs.append(self._blob(receipt, contract))
            entries.append(
                ReceiptIndexEntry(
                    source=payload.source,
                    natural_key=payload.natural_key,
                    as_of=payload.as_of,
                    fiscal_period=payload.fiscal_period,
                    status=payload.status,
                    content_hash=receipt.content_hash,
                    retrieved_at=receipt.retrieved_at,
                    payload_path=receipt.payload_path,
                )
            )
        revision = self._catalog.publish(entries, blobs=blobs, ranges=ranges)
        return tuple(ScopedReceipt(bronze_receipt=receipt, catalog_revision=revision) for receipt in receipts)

    @staticmethod
    def _canonical_payload(contract: SourceContract, payload: bytes) -> bytes:
        """Return the bytes to store, refusing a non-canonical ``RAW_ROWS_V1`` payload.

        Raises:
            PITDataError: the payload is not already its own canonical form.
        """
        canonical = canonical_raw_rows_bytes(contract, payload)
        if canonical != payload:
            raise PITDataError(
                f"payload of {contract.source!r} is not canonical; build it with raw_rows_envelope instead of rewriting it"
            )
        return canonical

    @staticmethod
    def _blob(receipt: BronzeReceipt, contract: SourceContract) -> BlobEntry:
        return BlobEntry(
            content_hash=receipt.content_hash,
            kind=contract.kind,
            source=contract.source,
            usable=True,
            unusable_reason=None,
            retrieved_at=receipt.retrieved_at,
            payload_path=receipt.payload_path,
        )

    def persist(self, payload: ScopedRawPayload) -> ScopedReceipt:
        """Persist one keyed payload through the same batch contract.

        A keyed payload always stores bytes, so it always yields a receipt; a
        range payload that answered with no bytes goes through ``persist_many``.
        """
        return self.persist_many((payload,))[0]
