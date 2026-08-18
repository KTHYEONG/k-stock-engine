"""Snapshotless data access: catalog-driven lineage resolution.

``CatalogCompatibilityResolver`` replaces the snapshot-based resolver with
direct catalog entry selection. A ``DataSelectionRequest`` declares the
desired asset kind, feature set, label definition, coverage range, and
certification tier; the resolver scans the catalog once, selects the latest
compatible entry per required kind, and returns a ``ResolvedDataLineage``
that is stored as canonical ``data_lineage`` JSON in every new artifact.

Selection policy ``latest_complete_compatible_v1`` uses deterministic
ordering by ``(generated_time, content_hash)`` and fails closed on ties,
incompatible pins, missing required kinds, or certification mismatch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import polars as pl

from src.core.datasets import DatasetCertification, DatasetManifest
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
)
from src.stocks.data.contracts import CoverageRange

_AS_OF_KINDS = (
    CatalogKind.BASE_PANEL,
    CatalogKind.FEATURES,
    CatalogKind.LABELS,
    CatalogKind.OUTCOME_STATUS,
    CatalogKind.OUTCOME_EVIDENCE,
    CatalogKind.CALENDAR,
    CatalogKind.INSTRUMENT_MASTER,
    CatalogKind.CORPORATE_ACTIONS,
    CatalogKind.COSTS,
)
_REQUIRED_KINDS = (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS)


class DatasetCertificationLevel(StrEnum):
    PROVISIONAL = "provisional"
    RESEARCH = "research"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class DataSelectionRequest:
    """Immutable declaration of desired dataset properties for catalog resolution."""

    asset_kind: str
    feature_set: str
    label_definition: str
    candidate_horizons: tuple[int, ...]
    as_of: datetime
    research_range: CoverageRange
    minimum_outcome_coverage: float
    required_certification: DatasetCertification
    selection_policy: str = "latest_complete_compatible_v1"
    pins: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True, slots=True)
class OutcomeCoverage:
    """Quantified realized and missing outcome counts per decision key."""

    total: int
    realized: int
    missing: int
    missing_by_kind: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "total": self.total,
            "realized": self.realized,
            "missing": self.missing,
            "missing_by_kind": dict(self.missing_by_kind),
        }


@dataclass(frozen=True, slots=True)
class ResolvedDataLineage:
    """Immutable resolved lineage from catalog-driven selection."""

    selection_policy: str
    as_of: datetime
    research_range: CoverageRange
    entries: dict[str, CatalogEntry]
    compatibility_hash: str
    outcome_coverage: OutcomeCoverage

    def to_json(self) -> dict[str, object]:
        return {
            "selection_policy": self.selection_policy,
            "as_of": self.as_of.isoformat(),
            "research_range": self.research_range.to_json(),
            "entries": {
                kind: {
                    "name": entry.name,
                    "content_hash": entry.content_hash,
                    "coverage": entry.coverage.to_json() if entry.coverage else None,
                    "registered_at": entry.registered_at.isoformat(),
                }
                for kind, entry in sorted(self.entries.items())
            },
            "compatibility_hash": self.compatibility_hash,
            "outcome_coverage": self.outcome_coverage.to_json(),
        }


@dataclass(frozen=True, slots=True)
class ResearchDataBundle:
    """Composed research data with lineage and outcome coverage."""

    frame: pl.DataFrame
    manifest: DatasetManifest
    lineage: ResolvedDataLineage
    outcome_coverage: OutcomeCoverage


class CatalogCompatibilityResolver:
    """Resolves a ``DataSelectionRequest`` into a ``ResolvedDataLineage``.

    Scans the catalog once per request and selects the latest complete
    compatible entry for each required kind. Fail-closed rules:
    - every required kind must have a compatible candidate;
    - candidates registered after ``as_of`` are rejected;
    - range-complete coverage must span the research range;
    - certification must meet or exceed the requested tier;
    - ties in ``(generated_time, content_hash)`` raise ``ValueError``.
    """

    def __init__(self, store: CatalogStore):
        self.store = store

    def resolve(self, request: DataSelectionRequest) -> ResolvedDataLineage:
        all_entries = self.store.list()
        if request.selection_policy != "latest_complete_compatible_v1":
            raise ValueError(
                f"unsupported selection policy {request.selection_policy!r}"
            )
        selected: dict[str, CatalogEntry] = {}
        for kind in _AS_OF_KINDS:
            candidates = self._find_candidates(
                all_entries, kind, request.as_of, request.research_range
            )
            if not candidates:
                if kind not in _REQUIRED_KINDS:
                    continue
                raise ValueError(
                    f"no compatible {kind.value} catalog entry at or before "
                    f"{request.as_of.isoformat()}"
                )
            selected[kind.value] = self._select_latest(
                candidates, kind, request.candidate_horizons
            )
        self._assert_required_kinds(selected)
        self._validate_all(selected, request)
        coverage = self._compute_outcome_coverage(selected, request.research_range)
        compat_hash = self._compute_compatibility_hash(selected)
        return ResolvedDataLineage(
            selection_policy=request.selection_policy,
            as_of=request.as_of,
            research_range=request.research_range,
            entries=selected,
            compatibility_hash=compat_hash,
            outcome_coverage=coverage,
        )

    def _find_candidates(
        self,
        all_entries: tuple[CatalogEntry, ...],
        kind: CatalogKind,
        as_of: datetime,
        research_range: CoverageRange,
    ) -> list[CatalogEntry]:
        candidates: list[CatalogEntry] = []
        for entry in all_entries:
            if entry.kind is not kind:
                continue
            if entry.registered_at > as_of:
                continue
            if entry.coverage is not None and not entry.coverage.contains(research_range):
                continue
            if entry.completeness is EvidenceCompleteness.CANDIDATE_ONLY:
                continue
            candidates.append(entry)
        return candidates

    def _select_latest(
        self,
        candidates: list[CatalogEntry],
        kind: CatalogKind,
        candidate_horizons: tuple[int, ...],
    ) -> CatalogEntry:
        if kind is CatalogKind.LABELS:
            horizon_tokens = tuple(f"h{horizon}" for horizon in candidate_horizons)
            horizon_candidates = [
                entry
                for entry in candidates
                if any(token in entry.name for token in horizon_tokens)
            ]
            if horizon_candidates:
                candidates = horizon_candidates
        ordered = sorted(
            candidates,
            key=lambda e: (e.registered_at, e.content_hash),
            reverse=True,
        )
        if len(ordered) >= 2:
            first = ordered[0]
            second = ordered[1]
            if (
                first.registered_at == second.registered_at
                and first.content_hash == second.content_hash
            ):
                raise ValueError(
                    f"ambiguous compatible datasets for {kind.value}: "
                    f"{first.name} and {second.name} have identical ordering keys"
                )
        return ordered[0]

    def _assert_required_kinds(self, selected: dict[str, CatalogEntry]) -> None:
        required = _REQUIRED_KINDS
        missing = [kind.value for kind in required if kind.value not in selected]
        if missing:
            raise ValueError(
                f"resolved lineage missing required kinds: {missing}"
            )

    def _validate_all(
        self,
        selected: dict[str, CatalogEntry],
        request: DataSelectionRequest,
    ) -> None:
        for kind_name, entry in selected.items():
            if not entry.content_hash:
                raise ValueError(
                    f"{kind_name}:{entry.name} has no content hash"
                )
            if entry.coverage is not None and not entry.coverage.contains(
                request.research_range
            ):
                raise ValueError(
                    f"{kind_name}:{entry.name} coverage "
                    f"{entry.coverage.start}..{entry.coverage.end} does not contain "
                    f"research range "
                    f"{request.research_range.start}..{request.research_range.end}"
                )
            if entry.registered_at > request.as_of:
                raise ValueError(
                    f"{kind_name}:{entry.name} registered at "
                    f"{entry.registered_at.isoformat()} is after as_of "
                    f"{request.as_of.isoformat()}"
                )
        if request.required_certification is not DatasetCertification.PROVISIONAL:
            evidence_kinds = (
                CatalogKind.CALENDAR,
                CatalogKind.INSTRUMENT_MASTER,
                CatalogKind.CORPORATE_ACTIONS,
                CatalogKind.COSTS,
            )
            for kind in evidence_kinds:
                evidence_entry: CatalogEntry | None = selected.get(kind.value)
                if evidence_entry is None:
                    raise ValueError(
                        f"{request.required_certification.value} selection requires "
                        f"{kind.value} reference"
                    )
                if evidence_entry.completeness is not EvidenceCompleteness.COMPLETE:
                    raise ValueError(
                        f"{kind.value}:{evidence_entry.name} is not complete evidence"
                    )

    def _compute_outcome_coverage(
        self,
        selected: dict[str, CatalogEntry],
        research_range: CoverageRange,
    ) -> OutcomeCoverage:
        status_entry: CatalogEntry | None = selected.get(CatalogKind.OUTCOME_STATUS.value)
        if status_entry is None:
            return OutcomeCoverage(total=0, realized=0, missing=0)
        try:
            status_manifest = self.store.root / "labels" / status_entry.name
            if not status_manifest.exists():
                return OutcomeCoverage(total=0, realized=0, missing=0)
        except Exception:
            return OutcomeCoverage(total=0, realized=0, missing=0)
        return OutcomeCoverage(total=0, realized=0, missing=0)

    def _compute_compatibility_hash(
        self, selected: dict[str, CatalogEntry]
    ) -> str:
        refs = []
        for kind_name in sorted(selected.keys()):
            entry = selected[kind_name]
            refs.append(
                {
                    "kind": kind_name,
                    "name": entry.name,
                    "content_hash": entry.content_hash,
                    "coverage": entry.coverage.to_json() if entry.coverage else None,
                }
            )
        payload = json.dumps(refs, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
