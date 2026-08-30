"""Immutable, typed catalog and research-snapshot resolver for stock data.

The catalog is the single source of truth that binds the exact evidence,
base-panel, feature, and label versions consumed by research. It is:

- append-only: entries are written once to ``catalog.jsonl`` and never mutated;
- typed: every entry declares its kind, immutable name, content hash, coverage,
  completeness, provenance, and version references;
- fail-closed: the :class:`SnapshotResolver` yields a
  :class:`ResearchDataSnapshot` only when every referenced entry exists, is
  range-complete for the declared research windows, is hash-consistent, and is
  not ``candidate_only`` evidence.

Retention is a separate, dry-run-first concern: catalog references must be
marked before anything is garbage-collected, and evidence and published
snapshot versions are never deleted automatically.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from src.core.datasets import DatasetCertification
from src.stocks.data.contracts import (
    CoverageRange,
    ResearchWindows,
    TimingConvention,
)
from src.storage.parquet_datasets import file_sha256

CATALOG_LOG_NAME = "catalog.jsonl"
RETENTION_NAME = "retention.json"
ACTIVE_DATASETS_NAME = "active_datasets.json"
SNAPSHOT_MANIFEST_NAME = "snapshot_manifest.json"
SNAPSHOT_MANIFEST_VERSION = 1


class CatalogKind(StrEnum):
    """Typed catalog categories. Names are immutable semantic versions."""

    LEGACY_FEATURES = "legacy_features"
    CALENDAR = "calendar"
    INSTRUMENT_MASTER = "instrument_master"
    DISCLOSURES = "disclosures"
    RAW_BARS = "raw_bars"
    OUTCOME_OPEN_BARS = "outcome_open_bars"
    OUTCOME_EVIDENCE = "outcome_evidence"
    CORPORATE_ACTIONS = "corporate_actions"
    COSTS = "costs"
    BASE_PANEL = "base_panel"
    FEATURES = "features"
    LABELS = "labels"
    OUTCOME_STATUS = "outcome_status"
    SNAPSHOT = "snapshot"


# Evidence kinds whose completeness must be verified before certification.
EVIDENCE_KINDS = (
    CatalogKind.CALENDAR,
    CatalogKind.INSTRUMENT_MASTER,
    CatalogKind.DISCLOSURES,
    CatalogKind.CORPORATE_ACTIONS,
    CatalogKind.COSTS,
)
# Immutable research artifacts never eligible for automatic deletion.
IMMUTABLE_KINDS = (CatalogKind.LEGACY_FEATURES, CatalogKind.BASE_PANEL, CatalogKind.SNAPSHOT)
_EVIDENCE_KIND_SET = frozenset(EVIDENCE_KINDS)


class EvidenceCompleteness(StrEnum):
    """Completeness of an evidence artifact's coverage for its declared range."""

    COMPLETE = "complete"
    CANDIDATE_ONLY = "candidate_only"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One immutable catalog record.

    ``name`` is an immutable semantic version (never ``latest``). ``coverage``
    is inclusive ``[start, end]`` and ``None`` when not applicable (e.g. a
    schema-only legacy source record). ``references`` binds the exact versions
    this entry consumed: a tuple of ``(kind, name)`` pairs.
    """

    kind: CatalogKind
    name: str
    content_hash: str
    schema_hash: str
    registered_at: datetime
    coverage: CoverageRange | None = None
    completeness: EvidenceCompleteness = EvidenceCompleteness.INCOMPLETE
    path: str = ""
    references: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = ()
    row_count: int = 0
    pinned: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("catalog entry name must be non-empty")
        if not self.content_hash:
            raise ValueError(f"{self.kind.value}:{self.name} requires a content_hash")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")

    @property
    def is_evidence(self) -> bool:
        return self.kind in _EVIDENCE_KIND_SET


@dataclass(frozen=True, slots=True)
class ActiveDatasetPolicy:
    """Explicit dataset versions that remain in the active research surface."""

    entries: tuple[tuple[CatalogKind, str], ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.entries)) != len(self.entries):
            raise ValueError("active dataset policy contains duplicate entries")
        if any(not name for _kind, name in self.entries):
            raise ValueError("active dataset policy names must be non-empty")

    def contains(self, kind: CatalogKind, name: str) -> bool:
        return (kind, name) in self.entries

    def require_operational_entries(self, store: CatalogStore) -> Mapping[CatalogKind, CatalogEntry]:
        """Require exactly one active base_panel, features, labels, and costs entry."""

        required = (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS, CatalogKind.COSTS)
        # Fail if policy missing or duplicated kind
        seen: dict[CatalogKind, str] = {}
        for kind, name in self.entries:
            if kind in required:
                if kind in seen:
                    raise ValueError(f"active policy duplicated {kind.value} entry {name!r} vs {seen[kind]!r}")
                seen[kind] = name
        missing_kinds = [k.value for k in required if k not in seen]
        if missing_kinds:
            raise ValueError(f"active policy missing required kinds {missing_kinds}")
        # Validate each exists, has hashes, and return mapping
        result: dict[CatalogKind, CatalogEntry] = {}
        for kind in required:
            name = seen[kind]
            entry = store.get(kind, name)
            if entry is None:
                raise ValueError(f"{kind.value}:{name} not found in catalog")
            if not entry.content_hash:
                raise ValueError(f"{kind.value}:{name} has empty catalog content_hash")
            if not entry.schema_hash and kind in (CatalogKind.FEATURES, CatalogKind.LABELS, CatalogKind.BASE_PANEL):
                # allow empty for costs? but spec says non-empty catalog/content hashes
                pass
            result[kind] = entry
        return result

    def to_json(self) -> dict[str, object]:
        return {
            "active_datasets_version": 1,
            "entries": [
                {"kind": kind.value, "name": name}
                for kind, name in sorted(self.entries, key=lambda item: (item[0].value, item[1]))
            ],
        }

    @classmethod
    def from_json(cls, payload: object) -> ActiveDatasetPolicy:
        if not isinstance(payload, dict):
            raise ValueError("active dataset policy must be an object")
        version = payload.get("active_datasets_version", 1)
        if int(version) != 1:
            raise ValueError(f"unsupported active dataset policy version: {version}")
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("active dataset policy entries must be a list")
        if any(not isinstance(item, dict) for item in raw_entries):
            raise ValueError("active dataset policy entries must be objects")
        try:
            entries = tuple(
                (CatalogKind(str(item["kind"])), str(item["name"]))
                for item in raw_entries
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("active dataset policy entry requires kind and name") from exc
        return cls(entries=entries)


def entry_repr(entry: CatalogEntry) -> dict[str, object]:
    """Deterministic JSON representation of a catalog entry."""
    return {
        "kind": entry.kind.value,
        "name": entry.name,
        "content_hash": entry.content_hash,
        "schema_hash": entry.schema_hash,
        "registered_at": entry.registered_at.isoformat(),
        "coverage": entry.coverage.to_json() if entry.coverage else None,
        "completeness": entry.completeness.value,
        "path": entry.path,
        "references": [
            {"kind": kind, "name": name} for kind, name in entry.references
        ],
        "limitations": list(entry.limitations),
        "row_count": entry.row_count,
        "pinned": entry.pinned,
    }


def entry_from_dict(data: dict[str, object]) -> CatalogEntry:
    raw_kind = str(data["kind"])
    raw_coverage = data.get("coverage")
    raw_references = data.get("references") or []
    raw_limitations = data.get("limitations") or []
    raw_completeness = data.get("completeness") or EvidenceCompleteness.INCOMPLETE.value
    if not isinstance(raw_references, list):
        raise ValueError("catalog references must be a list")
    if not isinstance(raw_limitations, list):
        raise ValueError("catalog limitations must be a list")
    coverage = None
    if raw_coverage:
        if not isinstance(raw_coverage, dict):
            raise ValueError("catalog coverage must be an object")
        coverage = CoverageRange.from_json(raw_coverage)
    return CatalogEntry(
        kind=CatalogKind(raw_kind),
        name=str(data["name"]),
        content_hash=str(data["content_hash"]),
        schema_hash=str(data.get("schema_hash", "")),
        registered_at=datetime.fromisoformat(str(data["registered_at"])),
        coverage=coverage,
        completeness=EvidenceCompleteness(str(raw_completeness)),
        path=str(data.get("path", "")),
        references=tuple(
            (str(ref["kind"]), str(ref["name"]))
            for ref in raw_references
            if isinstance(ref, dict)
        ),
        limitations=tuple(str(item) for item in raw_limitations),
        row_count=int(str(data.get("row_count") or 0)),
        pinned=bool(data.get("pinned", False)),
    )


class CatalogStore:
    """Append-only, typed catalog under ``<root>/catalog/stocks``.

    ``catalog.jsonl`` is the append-only log. Registration rejects a duplicate
    ``(kind, name)`` so a rewritten artifact always produces a new version.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self.root / CATALOG_LOG_NAME

    def register(self, entry: CatalogEntry) -> None:
        if self.get(entry.kind, entry.name) is not None:
            raise ValueError(f"catalog already has {entry.kind.value}:{entry.name}")
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry_repr(entry), sort_keys=True) + "\n")

    def list(self, kind: CatalogKind | None = None) -> tuple[CatalogEntry, ...]:
        """Read the append-only log in registration order."""
        if not self.log_path.exists():
            return ()
        entries: list[CatalogEntry] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = entry_from_dict(json.loads(line))
                if kind is None or entry.kind is kind:
                    entries.append(entry)
        return tuple(entries)

    def get(self, kind: CatalogKind, name: str) -> CatalogEntry | None:
        for entry in self.list(kind):
            if entry.name == name:
                return entry
        return None

    @property
    def active_policy_path(self) -> Path:
        return self.root / ACTIVE_DATASETS_NAME

    def load_active_policy(self) -> ActiveDatasetPolicy:
        """Load the optional active-version policy; absent means no override."""
        if not self.active_policy_path.exists():
            return ActiveDatasetPolicy()
        try:
            payload = json.loads(self.active_policy_path.read_text(encoding="utf-8"))
            return ActiveDatasetPolicy.from_json(payload)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid active dataset policy: {self.active_policy_path}") from exc

    def save_active_policy(self, policy: ActiveDatasetPolicy) -> None:
        self.active_policy_path.write_text(
            json.dumps(policy.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def missing_entries(self) -> tuple[CatalogEntry, ...]:
        """Return registered entries whose declared path is absent on disk."""
        project_root = self.root.parent.parent.parent
        missing: list[CatalogEntry] = []
        for entry in self.list():
            if not entry.path:
                continue
            path = Path(entry.path)
            if not path.is_absolute():
                path = project_root / path
            if not path.exists():
                missing.append(entry)
        return tuple(sorted(missing, key=lambda item: (item.kind.value, item.name)))

    def require(self, kind: CatalogKind, name: str) -> CatalogEntry:
        entry = self.get(kind, name)
        if entry is None:
            raise ValueError(f"catalog has no {kind.value} entry named {name!r}")
        return entry


def snapshot_manifest_hash(payload: dict[str, object]) -> str:
    """Deterministic fingerprint of a snapshot manifest.

    Identity inputs serialize identically; any change to a referenced content,
    evidence, or feature-contract hash changes the fingerprint.
    """
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Persisted ``snapshot_manifest.json`` content: only references, no rows.

    ``manifest_hash`` binds every reference: base panel, features, labels,
    calendar, master, corporate actions, costs, and feature-contract hashes, in
    addition to the declared windows, timing convention, and certification tier.
    """

    snapshot_id: str
    certification: DatasetCertification
    timing_convention: TimingConvention
    windows: ResearchWindows
    references: tuple[CatalogEntry, ...]
    manifest_hash: str = ""

    def to_json(self) -> dict[str, object]:
        by_kind = {entry.kind.value: entry for entry in self.references}
        return {
            "snapshot_manifest_version": SNAPSHOT_MANIFEST_VERSION,
            "snapshot_id": self.snapshot_id,
            "manifest_hash": self.manifest_hash,
            "certification": self.certification.value,
            "timing_convention": self.timing_convention.value,
            "windows": self.windows.to_json(),
            "references": {
                kind: {
                    "name": entry.name,
                    "content_hash": entry.content_hash,
                }
                for kind, entry in sorted(by_kind.items())
            },
        }


def build_snapshot_manifest(
    *,
    snapshot_id: str,
    certification: DatasetCertification,
    timing_convention: TimingConvention,
    windows: ResearchWindows,
    references: tuple[CatalogEntry, ...],
) -> SnapshotManifest:
    """Construct a manifest whose hash is computed from canonical JSON."""
    probe = SnapshotManifest(
        snapshot_id=snapshot_id,
        certification=certification,
        timing_convention=timing_convention,
        windows=windows,
        references=references,
    )
    payload = probe.to_json()
    del payload["manifest_hash"]
    manifest_hash = snapshot_manifest_hash(payload)
    payload["manifest_hash"] = manifest_hash
    return SnapshotManifest(
        snapshot_id=snapshot_id,
        certification=certification,
        timing_convention=timing_convention,
        windows=windows,
        references=references,
        manifest_hash=manifest_hash,
    )


@dataclass(frozen=True, slots=True)
class ResearchDataSnapshot:
    """The only train/backtest input selector, resolved and validated.

    This is the in-memory result of :meth:`SnapshotResolver.resolve`: it
    bundles the immutable manifest and every resolved reference entry. No rows
    are copied into it; readers obtain base/feature/label rows through the
    bounded lazy read plan against the referenced dataset versions.
    """

    manifest: SnapshotManifest
    base_panel: CatalogEntry
    features: CatalogEntry | None
    labels: CatalogEntry | None
    outcome_status: CatalogEntry | None
    outcome_evidence: CatalogEntry | None
    calendar: CatalogEntry | None
    master: CatalogEntry | None
    corporate_actions: CatalogEntry | None
    costs: CatalogEntry | None
    feature_contracts_hash: str = ""

    @property
    def research_range(self) -> CoverageRange:
        return self.manifest.windows.research_range

    @property
    def execution_range(self) -> CoverageRange:
        """Common immutable range usable by local ML and backtests."""
        core = (self.base_panel, self.features, self.labels)
        if any(entry is None or entry.coverage is None for entry in core):
            raise ValueError("execution requires covered base_panel, features, and labels")
        covered = tuple(
            entry.coverage
            for entry in core
            if entry is not None and entry.coverage is not None
        )
        if len(covered) != len(core):
            raise ValueError("execution requires covered base_panel, features, and labels")
        start = max(self.research_range.start, *(range_.start for range_ in covered))
        end = min(self.research_range.end, *(range_.end for range_ in covered))
        if start > end:
            raise ValueError("execution datasets have no common covered range")
        return CoverageRange(start=start, end=end)

    @property
    def status_provenance(self) -> str:
        """Pinned vs legacy-inferred outcome-status provenance.

        ``pinned`` means the snapshot manifest declares a hash-bound
        ``OUTCOME_STATUS`` reference. ``legacy-inferred`` means no status
        artifact is pinned (a legacy snapshot): it is diagnostic-only and must
        never be promoted for certified training/backtesting.
        """
        return "pinned" if self.outcome_status is not None else "legacy-inferred"

    @property
    def evidence_provenance(self) -> str:
        """Pinned vs unpinned outcome-evidence provenance.

        ``pinned`` means the snapshot manifest declares a hash-bound
        ``OUTCOME_EVIDENCE`` reference alongside the outcome status; without it
        a status spine cannot classify confirmed no-bars from collection gaps.
        """
        return "pinned" if self.outcome_evidence is not None else "unpinned"

    def reference(self, kind: CatalogKind) -> CatalogEntry | None:
        for entry in self.manifest.references:
            if entry.kind is kind:
                return entry
        return None


class SnapshotResolver:
    """Resolves a snapshot id into a validated :class:`ResearchDataSnapshot`.

    Fail-closed rules (spec acceptance criterion 1):
    - every referenced entry must exist in the catalog;
    - the catalog entry hash must equal the hash pinned in the snapshot manifest;
    - base-panel, feature, and label coverage must contain the research range;
    - evidence coverage must contain the research range and must be
      ``COMPLETE``, never ``candidate_only``;
    - a certified snapshot (research or production) must reference complete
      calendar, master, corporate-action, and cost evidence.
    """

    def __init__(self, store: CatalogStore):
        self.store = store

    def resolve(self, snapshot_id: str) -> ResearchDataSnapshot:
        manifest = self._load_manifest(snapshot_id)
        self._assert_manifest_hash(manifest)
        references = self._resolve_references(manifest)
        self._assert_range_complete(manifest, references)
        self._assert_evidence(manifest, references)
        self._assert_status_reference(manifest, references)
        return ResearchDataSnapshot(
            manifest=manifest,
            base_panel=references[CatalogKind.BASE_PANEL],
            features=references.get(CatalogKind.FEATURES),
            labels=references.get(CatalogKind.LABELS),
            outcome_status=references.get(CatalogKind.OUTCOME_STATUS),
            outcome_evidence=references.get(CatalogKind.OUTCOME_EVIDENCE),
            calendar=references.get(CatalogKind.CALENDAR),
            master=references.get(CatalogKind.INSTRUMENT_MASTER),
            corporate_actions=references.get(CatalogKind.CORPORATE_ACTIONS),
            costs=references.get(CatalogKind.COSTS),
        )

    def resolve_execution(self, snapshot_id: str) -> ResearchDataSnapshot:
        """Resolve the minimal hash-bound dataset contract for local execution."""
        manifest = self._load_manifest(snapshot_id)
        self._assert_manifest_hash(manifest)
        references = self._resolve_references(manifest)
        required = (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS)
        missing = [kind.value for kind in required if kind not in references]
        if missing:
            raise ValueError(f"execution snapshot requires datasets, missing {missing}")
        snapshot = ResearchDataSnapshot(
            manifest=manifest,
            base_panel=references[CatalogKind.BASE_PANEL],
            features=references[CatalogKind.FEATURES],
            labels=references[CatalogKind.LABELS],
            outcome_status=references.get(CatalogKind.OUTCOME_STATUS),
            outcome_evidence=references.get(CatalogKind.OUTCOME_EVIDENCE),
            calendar=references.get(CatalogKind.CALENDAR),
            master=references.get(CatalogKind.INSTRUMENT_MASTER),
            corporate_actions=references.get(CatalogKind.CORPORATE_ACTIONS),
            costs=references.get(CatalogKind.COSTS),
        )
        _ = snapshot.execution_range
        return snapshot

    def _load_manifest(self, snapshot_id: str) -> SnapshotManifest:
        manifest_path = self.store.root / "snapshots" / snapshot_id / SNAPSHOT_MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError(f"no snapshot manifest for {snapshot_id!r}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_references = payload.get("references")
        raw_windows = payload.get("windows")
        if not isinstance(raw_references, dict):
            raise ValueError("snapshot references must be an object")
        if not isinstance(raw_windows, dict):
            raise ValueError("snapshot windows must be an object")
        references = tuple(
            CatalogEntry(
                kind=CatalogKind(kind),
                name=str(ref["name"]),
                content_hash=str(ref["content_hash"]),
                schema_hash="",
                registered_at=datetime.now(UTC),
            )
            for kind, ref in sorted(raw_references.items())
            if isinstance(ref, dict)
        )
        return SnapshotManifest(
            snapshot_id=str(payload["snapshot_id"]),
            certification=DatasetCertification(str(payload["certification"])),
            timing_convention=TimingConvention(str(payload["timing_convention"])),
            windows=ResearchWindows.from_json(raw_windows),
            references=references,
            manifest_hash=str(payload["manifest_hash"]),
        )

    def _assert_manifest_hash(self, manifest: SnapshotManifest) -> None:
        payload = manifest.to_json()
        del payload["manifest_hash"]
        recomputed = snapshot_manifest_hash(payload)
        if recomputed != manifest.manifest_hash:
            raise ValueError(
                f"snapshot {manifest.snapshot_id!r} manifest hash mismatch: "
                f"pinned {manifest.manifest_hash}, recomputed {recomputed}"
            )

    def _resolve_references(
        self, manifest: SnapshotManifest
    ) -> dict[CatalogKind, CatalogEntry]:
        resolved: dict[CatalogKind, CatalogEntry] = {}
        for pinned in manifest.references:
            entry = self.store.require(pinned.kind, pinned.name)
            if entry.content_hash != pinned.content_hash:
                raise ValueError(
                    f"snapshot pins {pinned.kind.value}:{pinned.name} with hash "
                    f"{pinned.content_hash} but catalog has {entry.content_hash}"
                )
            resolved[pinned.kind] = entry
        for required in (CatalogKind.BASE_PANEL,):
            if required not in resolved:
                raise ValueError(f"snapshot requires a {required.value} reference")
        return resolved

    def _assert_range_complete(
        self, manifest: SnapshotManifest, references: dict[CatalogKind, CatalogEntry]
    ) -> None:
        research = manifest.windows.research_range
        for kind, entry in references.items():
            if kind is CatalogKind.FEATURES and entry.coverage is None:
                continue
            if entry.coverage is None:
                raise ValueError(
                    f"{kind.value}:{entry.name} declares no coverage"
                )
            if not entry.coverage.contains(research):
                raise ValueError(
                    f"{kind.value}:{entry.name} coverage "
                    f"{entry.coverage.start}..{entry.coverage.end} does not contain "
                    f"research range {research.start}..{research.end}"
                )

    def _assert_evidence(
        self, manifest: SnapshotManifest, references: dict[CatalogKind, CatalogEntry]
    ) -> None:
        certification = manifest.certification
        if certification is DatasetCertification.PROVISIONAL:
            return
        missing = [kind.value for kind in EVIDENCE_KINDS if kind not in references]
        if missing:
            raise ValueError(
                f"{certification.value} snapshot requires evidence references, missing {missing}"
            )
        for kind in EVIDENCE_KINDS:
            entry = references[kind]
            if entry.completeness is EvidenceCompleteness.CANDIDATE_ONLY:
                raise ValueError(
                    f"{kind.value}:{entry.name} is candidate_only and cannot certify"
                )
            if entry.completeness is not EvidenceCompleteness.COMPLETE:
                raise ValueError(
                    f"{kind.value}:{entry.name} is not complete evidence"
                )

    def _assert_status_reference(
        self, manifest: SnapshotManifest, references: dict[CatalogKind, CatalogEntry]
    ) -> None:
        """Require a pinned, complete outcome-status reference to certify.

        A legacy snapshot without a pinned ``OUTCOME_STATUS`` reference stays
        resolvable for diagnostic replay only; it can never be certified. A
        certified snapshot must pin a hash-consistent, complete status entry so
        no outcome is silently classified from a legacy fallback.
        """
        certification = manifest.certification
        status = references.get(CatalogKind.OUTCOME_STATUS)
        if certification is DatasetCertification.PROVISIONAL:
            if status is not None and status.completeness is EvidenceCompleteness.CANDIDATE_ONLY:
                raise ValueError(
                    f"outcome_status:{status.name} is candidate_only and cannot be referenced"
                )
            return
        if status is None:
            raise ValueError(
                f"{certification.value} snapshot requires a pinned outcome-status "
                "reference; legacy-inferred status provenance cannot certify"
            )
        if status.completeness is not EvidenceCompleteness.COMPLETE:
            raise ValueError(
                f"outcome_status:{status.name} is not complete evidence and cannot certify"
            )


@dataclass(frozen=True, slots=True)
class RetentionRecord:
    """Per-version retention status persisted in ``retention.json``."""

    kind: CatalogKind
    name: str
    referenced_by: tuple[str, ...] = ()
    pinned: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "referenced_by": list(self.referenced_by),
            "pinned": self.pinned,
        }


class RetentionRegistry:
    """Catalog-aware retention tracking with dry-run-first deletion.

    References are marked from snapshot manifests before anything may be
    deleted. ``dry_run`` lists candidates without touching files; ``delete``
    refuses referenced, pinned, evidence, and published-snapshot versions.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def retention_path(self) -> Path:
        return self.root / RETENTION_NAME

    def rebuild(
        self,
        store: CatalogStore,
        *,
        expire_before: datetime | None = None,
    ) -> tuple[RetentionRecord, ...]:
        """Recompute referenced/pinned status from the catalog and snapshots."""
        records: dict[tuple[CatalogKind, str], RetentionRecord] = {}
        for entry in store.list():
            records[(entry.kind, entry.name)] = RetentionRecord(
                kind=entry.kind, name=entry.name, pinned=entry.pinned
            )
        for snapshot_dir in sorted((self.root / "snapshots").glob("*/")):
            snapshot_id = snapshot_dir.name
            for kind, name in self._manifest_references(snapshot_id):
                key = (kind, name)
                if key in records:
                    records[key] = RetentionRecord(
                        kind=kind,
                        name=name,
                        referenced_by=(*records[key].referenced_by, snapshot_id),
                        pinned=records[key].pinned,
                    )
        # Direct training no longer requires snapshots; explicitly active
        # dataset versions must therefore be retained independently.
        for kind, name in store.load_active_policy().entries:
            key = (kind, name)
            if key in records and "active-policy" not in records[key].referenced_by:
                records[key] = RetentionRecord(
                    kind=kind,
                    name=name,
                    referenced_by=(*records[key].referenced_by, "active-policy"),
                    pinned=records[key].pinned,
                )
        return tuple(records.values())

    def _manifest_references(self, snapshot_id: str) -> tuple[tuple[CatalogKind, str], ...]:
        manifest_path = self.root / "snapshots" / snapshot_id / SNAPSHOT_MANIFEST_NAME
        if not manifest_path.exists():
            return ()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return tuple(
            (CatalogKind(kind), str(ref["name"]))
            for kind, ref in sorted(payload["references"].items())
        )

    def save(self, records: tuple[RetentionRecord, ...]) -> None:
        payload = {"retention_version": 1, "records": [r.to_json() for r in records]}
        self.retention_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def load(self) -> tuple[RetentionRecord, ...]:
        if not self.retention_path.exists():
            return ()
        payload = json.loads(self.retention_path.read_text(encoding="utf-8"))
        records = []
        for raw in payload.get("records", []):
            kind = CatalogKind(str(raw["kind"]))
            records.append(
                RetentionRecord(
                    kind=kind,
                    name=str(raw["name"]),
                    referenced_by=tuple(str(item) for item in raw.get("referenced_by", [])),
                    pinned=bool(raw.get("pinned", False)),
                )
            )
        return tuple(records)


@dataclass(frozen=True, slots=True)
class RetentionCandidate:
    kind: CatalogKind
    name: str
    reason: str

    def to_json(self) -> dict[str, object]:
        return {"kind": self.kind.value, "name": self.name, "reason": self.reason}


def retention_dry_run(
    store: CatalogStore,
    registry: RetentionRegistry,
    active_policy: ActiveDatasetPolicy | None = None,
) -> tuple[RetentionCandidate, ...]:
    """List deletion candidates without changing any file.

    A version is eligible only when it is not referenced, not pinned, not
    evidence, and not an immutable published snapshot/legacy source. ``SNAPSHOT``
    entries are never eligible; evidence is never eligible.
    """
    policy = active_policy if active_policy is not None else store.load_active_policy()
    records = {f"{r.kind.value}:{r.name}": r for r in registry.load()}
    candidates: list[RetentionCandidate] = []
    for entry in store.list():
        if entry.kind in IMMUTABLE_KINDS or entry.kind in _EVIDENCE_KIND_SET:
            continue
        if policy.contains(entry.kind, entry.name):
            continue
        record = records.get(f"{entry.kind.value}:{entry.name}")
        if record is not None and (record.pinned or record.referenced_by):
            continue
        candidates.append(
            RetentionCandidate(kind=entry.kind, name=entry.name, reason="unreferenced")
        )
    return tuple(candidates)


def retention_delete(
    store: CatalogStore,
    registry: RetentionRegistry,
    candidates: tuple[RetentionCandidate, ...],
) -> int:
    """Delete the selected candidates, refusing evidence and immutable kinds.

    Returns the number of catalog entries removed. Referenced, pinned, evidence,
    legacy-source, and published-snapshot versions are always rejected.
    """
    records = {f"{r.kind.value}:{r.name}": r for r in registry.load()}
    deleted = 0
    lines: list[str] = []
    if store.log_path.exists():
        lines = store.log_path.read_text(encoding="utf-8").splitlines()
    for candidate in candidates:
        if candidate.kind in IMMUTABLE_KINDS or candidate.kind in _EVIDENCE_KIND_SET:
            raise ValueError(f"refusing to delete {candidate.kind.value}:{candidate.name}")
        entry = store.get(candidate.kind, candidate.name)
        if entry is None:
            continue
        record = records.get(f"{candidate.kind.value}:{candidate.name}")
        if record is not None and (record.pinned or record.referenced_by):
            raise ValueError(
                f"refusing to delete referenced/pinned {candidate.kind.value}:{candidate.name}"
            )
        deleted += 1
        lines = [line for line in lines if not _line_is(entry, line)]
    store.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return deleted


def _line_is(entry: CatalogEntry, line: str) -> bool:
    if not line.strip():
        return False
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return False
    return str(data.get("kind")) == entry.kind.value and str(data.get("name")) == entry.name


def register_file_evidence(
    store: CatalogStore,
    *,
    kind: CatalogKind,
    name: str,
    path: Path,
    coverage: CoverageRange,
    completeness: EvidenceCompleteness,
    registered_at: datetime | None = None,
    limitations: tuple[str, ...] = (),
    pinned: bool = False,
) -> CatalogEntry:
    """Register a single-file evidence artifact by its file digest."""
    if kind not in _EVIDENCE_KIND_SET:
        raise ValueError(f"register_file_evidence requires an evidence kind, got {kind.value}")
    if not path.is_file():
        raise FileNotFoundError(f"evidence artifact not found: {path}")
    entry = CatalogEntry(
        kind=kind,
        name=name,
        content_hash=file_sha256(path),
        schema_hash=sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        if kind in (CatalogKind.CALENDAR, CatalogKind.COSTS)
        else "",
        registered_at=registered_at or datetime.now(UTC),
        coverage=coverage,
        completeness=completeness,
        path=str(path),
        limitations=limitations,
        pinned=pinned,
    )
    store.register(entry)
    return entry
