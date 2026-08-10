"""Catalog: typed entries, fail-closed snapshot resolution, retention safety."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.datasets import DatasetCertification
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
    ResearchDataSnapshot,
    RetentionCandidate,
    RetentionRegistry,
    SnapshotResolver,
    build_snapshot_manifest,
    retention_delete,
    retention_dry_run,
    snapshot_manifest_hash,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows, TimingConvention

RANGE = CoverageRange(start=date(2024, 1, 1), end=date(2024, 3, 31))
WINDOWS = ResearchWindows(
    train=CoverageRange(start=date(2024, 1, 1), end=date(2024, 1, 31)),
    validation=CoverageRange(start=date(2024, 2, 1), end=date(2024, 2, 15)),
    test=CoverageRange(start=date(2024, 2, 16), end=date(2024, 3, 31)),
)
REGISTERED = datetime(2026, 1, 1, tzinfo=UTC)


def entry(
    kind: CatalogKind,
    name: str,
    *,
    content_hash: str = "abc",
    coverage: CoverageRange | None = RANGE,
    completeness: EvidenceCompleteness = EvidenceCompleteness.COMPLETE,
) -> CatalogEntry:
    return CatalogEntry(
        kind=kind,
        name=name,
        content_hash=content_hash,
        schema_hash="schema",
        registered_at=REGISTERED,
        coverage=coverage,
        completeness=completeness,
        path=f"data/{name}",
    )


def complete_evidence(store: CatalogStore) -> dict[CatalogKind, CatalogEntry]:
    refs = {
        CatalogKind.CALENDAR: entry(CatalogKind.CALENDAR, "calendar_v1"),
        CatalogKind.INSTRUMENT_MASTER: entry(CatalogKind.INSTRUMENT_MASTER, "master_v1"),
        CatalogKind.DISCLOSURES: entry(CatalogKind.DISCLOSURES, "disclosures_v1"),
        CatalogKind.CORPORATE_ACTIONS: entry(
            CatalogKind.CORPORATE_ACTIONS, "actions_v1"
        ),
        CatalogKind.COSTS: entry(CatalogKind.COSTS, "costs_v1"),
    }
    for ref in refs.values():
        store.register(ref)
    return refs


def register_dataset_refs(store: CatalogStore) -> dict[CatalogKind, CatalogEntry]:
    base = entry(CatalogKind.BASE_PANEL, "base_panel_v1", content_hash="base")
    features = entry(CatalogKind.FEATURES, "features_v1", content_hash="feat")
    labels = entry(CatalogKind.LABELS, "labels_v1", content_hash="label")
    for ref in (base, features, labels):
        store.register(ref)
    return {CatalogKind.BASE_PANEL: base, CatalogKind.FEATURES: features, CatalogKind.LABELS: labels}


def write_snapshot(
    store: CatalogStore,
    snapshot_id: str,
    refs: list[CatalogEntry],
    *,
    certification: DatasetCertification = DatasetCertification.RESEARCH,
    windows: ResearchWindows = WINDOWS,
) -> dict[str, object]:
    manifest = build_snapshot_manifest(
        snapshot_id=snapshot_id,
        certification=certification,
        timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
        windows=windows,
        references=tuple(refs),
    )
    payload = manifest.to_json()
    path = store.root / "snapshots" / snapshot_id / "snapshot_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload


class TestCatalogStore:
    def test_register_and_round_trip(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.CALENDAR, "calendar_v1"))
        loaded = store.get(CatalogKind.CALENDAR, "calendar_v1")
        assert loaded is not None
        assert loaded.kind is CatalogKind.CALENDAR
        assert loaded.name == "calendar_v1"
        assert loaded.coverage == RANGE
        assert loaded.completeness is EvidenceCompleteness.COMPLETE

    def test_duplicate_registration_is_rejected(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.CALENDAR, "calendar_v1"))
        with pytest.raises(ValueError, match="already has"):
            store.register(entry(CatalogKind.CALENDAR, "calendar_v1"))


class TestSnapshotManifestDeterminism:
    def refs(self) -> list[CatalogEntry]:
        return [
            entry(CatalogKind.BASE_PANEL, "base_panel_v1", content_hash="base"),
            entry(CatalogKind.FEATURES, "features_v1", content_hash="feat"),
            entry(CatalogKind.CALENDAR, "calendar_v1", content_hash="cal"),
        ]

    def test_identical_inputs_yield_identical_hash(self) -> None:
        first = build_snapshot_manifest(
            snapshot_id="s1",
            certification=DatasetCertification.RESEARCH,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=WINDOWS,
            references=tuple(self.refs()),
        )
        second = build_snapshot_manifest(
            snapshot_id="s1",
            certification=DatasetCertification.RESEARCH,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=WINDOWS,
            references=tuple(self.refs()),
        )
        assert first.manifest_hash == second.manifest_hash

    def test_content_hash_change_changes_manifest_hash(self) -> None:
        changed = [
            entry(CatalogKind.BASE_PANEL, "base_panel_v1", content_hash="base"),
            entry(CatalogKind.FEATURES, "features_v1", content_hash="OTHER"),
            entry(CatalogKind.CALENDAR, "calendar_v1", content_hash="cal"),
        ]
        base = build_snapshot_manifest(
            snapshot_id="s1",
            certification=DatasetCertification.RESEARCH,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=WINDOWS,
            references=tuple(self.refs()),
        )
        altered = build_snapshot_manifest(
            snapshot_id="s1",
            certification=DatasetCertification.RESEARCH,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=WINDOWS,
            references=tuple(changed),
        )
        assert base.manifest_hash != altered.manifest_hash


class TestSnapshotResolver:
    def test_resolves_complete_research_snapshot(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        write_snapshot(store, "research_1", list(refs.values()))
        snapshot = SnapshotResolver(store).resolve("research_1")
        assert isinstance(snapshot, ResearchDataSnapshot)
        assert snapshot.base_panel.name == "base_panel_v1"
        assert snapshot.calendar is not None

    def test_missing_reference_is_rejected(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        del refs[CatalogKind.CALENDAR]
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match="missing"):
            SnapshotResolver(store).resolve("research_1")

    def test_range_incomplete_evidence_is_rejected(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        short = entry(
            CatalogKind.CALENDAR,
            "calendar_v1",
            coverage=CoverageRange(start=date(2024, 1, 1), end=date(2024, 2, 1)),
        )
        evidence = {
            CatalogKind.CALENDAR: short,
            CatalogKind.INSTRUMENT_MASTER: entry(CatalogKind.INSTRUMENT_MASTER, "m1"),
            CatalogKind.CORPORATE_ACTIONS: entry(CatalogKind.CORPORATE_ACTIONS, "a1"),
            CatalogKind.COSTS: entry(CatalogKind.COSTS, "c1"),
        }
        for ref in evidence.values():
            store.register(ref)
        refs = {**evidence, **register_dataset_refs(store)}
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match="does not contain"):
            SnapshotResolver(store).resolve("research_1")

    def test_candidate_only_evidence_is_rejected(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        evidence = {
            CatalogKind.CALENDAR: entry(CatalogKind.CALENDAR, "cal1"),
            CatalogKind.INSTRUMENT_MASTER: entry(CatalogKind.INSTRUMENT_MASTER, "m1"),
            CatalogKind.DISCLOSURES: entry(CatalogKind.DISCLOSURES, "d1"),
            CatalogKind.CORPORATE_ACTIONS: entry(
                CatalogKind.CORPORATE_ACTIONS,
                "actions_candidate",
                completeness=EvidenceCompleteness.CANDIDATE_ONLY,
            ),
            CatalogKind.COSTS: entry(CatalogKind.COSTS, "c1"),
        }
        for ref in evidence.values():
            store.register(ref)
        refs = {**evidence, **register_dataset_refs(store)}
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match="candidate_only"):
            SnapshotResolver(store).resolve("research_1")

    def test_incomplete_cost_evidence_rejects_certification(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        evidence = {
            CatalogKind.CALENDAR: entry(CatalogKind.CALENDAR, "cal1"),
            CatalogKind.INSTRUMENT_MASTER: entry(CatalogKind.INSTRUMENT_MASTER, "m1"),
            CatalogKind.DISCLOSURES: entry(CatalogKind.DISCLOSURES, "d1"),
            CatalogKind.CORPORATE_ACTIONS: entry(CatalogKind.CORPORATE_ACTIONS, "a1"),
            CatalogKind.COSTS: entry(
                CatalogKind.COSTS,
                "costs_incomplete",
                completeness=EvidenceCompleteness.INCOMPLETE,
            ),
        }
        for ref in evidence.values():
            store.register(ref)
        refs = {**evidence, **register_dataset_refs(store)}
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match="not complete evidence"):
            SnapshotResolver(store).resolve("research_1")

    def test_hash_mismatch_is_rejected(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        complete_evidence(store)
        refs = register_dataset_refs(store)
        # The snapshot pins a content hash that differs from the catalog entry.
        tampered_base = entry(
            CatalogKind.BASE_PANEL, "base_panel_v1", content_hash="tampered"
        )
        refs[CatalogKind.BASE_PANEL] = tampered_base
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match="pins.*base_panel"):
            SnapshotResolver(store).resolve("research_1")


class TestRetention:
    def _store_with_datasets(self, tmp_path) -> CatalogStore:
        store = CatalogStore(tmp_path)
        features = entry(
            CatalogKind.FEATURES, "features_v1", content_hash="feat", completeness=EvidenceCompleteness.COMPLETE
        )
        store.register(features)
        return store

    def test_dry_run_lists_unreferenced_candidates_and_changes_no_files(self, tmp_path) -> None:
        store = self._store_with_datasets(tmp_path)
        registry = RetentionRegistry(store.root)
        registry.rebuild(store)
        registry.save(registry.rebuild(store))
        before = store.log_path.read_text(encoding="utf-8")

        candidates = retention_dry_run(store, registry)
        assert [c.name for c in candidates] == ["features_v1"]
        assert store.log_path.read_text(encoding="utf-8") == before

    def test_delete_rejects_evidence_and_immutable_kinds(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.CALENDAR, "cal1"))
        store.register(entry(CatalogKind.BASE_PANEL, "base_panel_v1"))
        registry = RetentionRegistry(store.root)

        evidence_candidate = RetentionCandidate(
            kind=CatalogKind.CALENDAR, name="cal1", reason="unreferenced"
        )
        base_candidate = RetentionCandidate(
            kind=CatalogKind.BASE_PANEL, name="base_panel_v1", reason="unreferenced"
        )
        with pytest.raises(ValueError, match="refusing to delete calendar:cal1"):
            retention_delete(store, registry, (evidence_candidate,))
        with pytest.raises(ValueError, match="refusing to delete base_panel:base_panel_v1"):
            retention_delete(store, registry, (base_candidate,))

    def test_referenced_versions_are_not_candidates(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        write_snapshot(store, "research_1", list(refs.values()))
        registry = RetentionRegistry(store.root)
        registry.save(registry.rebuild(store))

        candidates = retention_dry_run(store, registry)
        names = {c.name for c in candidates}
        assert "base_panel_v1" not in names
        assert "features_v1" not in names
        assert "labels_v1" not in names
        assert "calendar_v1" not in names

    def test_delete_rejects_referenced_versions(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        write_snapshot(store, "research_1", list(refs.values()))
        registry = RetentionRegistry(store.root)
        registry.save(registry.rebuild(store))

        referenced = RetentionCandidate(
            kind=CatalogKind.FEATURES, name="features_v1", reason="unreferenced"
        )
        with pytest.raises(ValueError, match="referenced/pinned"):
            retention_delete(store, registry, (referenced,))

    def test_delete_removes_unreferenced_features_entry(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.FEATURES, "features_v1"))
        registry = RetentionRegistry(store.root)
        registry.save(registry.rebuild(store))

        candidate = RetentionCandidate(
            kind=CatalogKind.FEATURES, name="features_v1", reason="unreferenced"
        )
        assert retention_delete(store, registry, (candidate,)) == 1
        assert store.get(CatalogKind.FEATURES, "features_v1") is None
