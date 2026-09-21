"""Deterministic comparison of required scope evidence against the receipt catalog."""
from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.research_scope import ResearchScope
from src.data.schemas import PITDataError

__all__ = [
    "CoverageRequirement",
    "ScopeCoverageReport",
    "build_scope_coverage_report",
]

_FISCAL_PATTERN = re.compile(r"\d{4}Q[1-4]")


def _fiscal_key(period: str) -> int:
    return int(period[:4]) * 4 + int(period[5])


@dataclass(frozen=True, slots=True)
class CoverageRequirement:
    """One source-natural-key observation required before a scoped release may claim coverage."""

    source: str
    natural_key: str
    as_of: date | None
    fiscal_period: str | None
    required: bool


@dataclass(frozen=True, slots=True)
class ScopeCoverageReport:
    """Deterministic comparison of required scope evidence against the receipt catalog."""

    scope_hash: str
    fulfilled: tuple[CoverageRequirement, ...]
    missing: tuple[CoverageRequirement, ...]
    unresolved: tuple[CoverageRequirement, ...]


def build_scope_coverage_report(
    *, scope: ResearchScope, requirements: Collection[CoverageRequirement], catalog: ReceiptCatalog
) -> ScopeCoverageReport:
    """Classify required evidence without treating missing or unsuccessful raw responses as usable data."""
    ordered = sorted(requirements, key=lambda item: (item.source, item.natural_key))
    for item in ordered:
        if item.as_of is not None and item.as_of < scope.evidence_start:
            raise PITDataError(f"coverage requirement {item.natural_key!r} precedes evidence start")
        if item.fiscal_period is not None:
            if not _FISCAL_PATTERN.fullmatch(item.fiscal_period):
                raise PITDataError(f"invalid fiscal period {item.fiscal_period!r}")
            if _fiscal_key(item.fiscal_period) < _fiscal_key(scope.features.fundamental_fiscal_start):
                raise PITDataError(f"coverage requirement {item.natural_key!r} precedes fiscal floor")
    by_source: dict[str, list[str]] = {}
    for item in ordered:
        by_source.setdefault(item.source, []).append(item.natural_key)
    indexed: dict[tuple[str, str], ReceiptIndexEntry] = {}
    for source, keys in by_source.items():
        for key, entry in catalog.latest(source=source, natural_keys=keys).items():
            indexed[(source, key)] = entry
    fulfilled: list[CoverageRequirement] = []
    missing: list[CoverageRequirement] = []
    unresolved: list[CoverageRequirement] = []
    for item in ordered:
        found = indexed.get((item.source, item.natural_key))
        if found is None:
            if item.required:
                missing.append(item)
        elif found.status == EvidenceStatus.SUCCESS:
            fulfilled.append(item)
        else:
            unresolved.append(item)
    return ScopeCoverageReport(
        scope_hash=scope.content_hash,
        fulfilled=tuple(fulfilled),
        missing=tuple(missing),
        unresolved=tuple(unresolved),
    )
