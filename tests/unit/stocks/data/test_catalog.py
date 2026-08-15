"""Catalog: typed entries, fail-closed snapshot resolution, retention safety."""
from __future__ import annotations

import calendar
import hashlib
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
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows, TimingConvention
from src.stocks.data.evidence_collectors import (
    EvidenceCollectionError,
    OpenDartEvidenceCollector,
)
from src.stocks.data.inventory import InventoryLifecycle, Recommendation, scan_inventory

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
    status = entry(CatalogKind.OUTCOME_STATUS, "labels_v1_outcome_status", content_hash="status")
    for ref in (base, features, labels, status):
        store.register(ref)
    return {
        CatalogKind.BASE_PANEL: base,
        CatalogKind.FEATURES: features,
        CatalogKind.LABELS: labels,
        CatalogKind.OUTCOME_STATUS: status,
    }


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
        with pytest.raises(ValueError, match=r"pins.*base_panel"):
            SnapshotResolver(store).resolve("research_1")

    def test_absent_outcome_status_reference_rejects_certification(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        del refs[CatalogKind.OUTCOME_STATUS]
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match="pinned outcome-status"):
            SnapshotResolver(store).resolve("research_1")

    def test_legacy_inferred_status_provenance_cannot_certify(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        del refs[CatalogKind.OUTCOME_STATUS]
        write_snapshot(store, "legacy_1", list(refs.values()))
        with pytest.raises(ValueError, match="legacy-inferred status provenance"):
            SnapshotResolver(store).resolve("legacy_1")

    def test_hash_mismatched_outcome_status_is_rejected(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = {**complete_evidence(store), **register_dataset_refs(store)}
        tampered_status = entry(
            CatalogKind.OUTCOME_STATUS,
            "labels_v1_outcome_status",
            content_hash="tampered",
        )
        refs[CatalogKind.OUTCOME_STATUS] = tampered_status
        write_snapshot(store, "research_1", list(refs.values()))
        with pytest.raises(ValueError, match=r"pins.*outcome_status"):
            SnapshotResolver(store).resolve("research_1")

    def test_provisional_snapshot_allows_legacy_inferred_provenance(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        refs = register_dataset_refs(store)
        del refs[CatalogKind.OUTCOME_STATUS]
        write_snapshot(
            store,
            "provisional_1",
            list(refs.values()),
            certification=DatasetCertification.PROVISIONAL,
        )
        snapshot = SnapshotResolver(store).resolve("provisional_1")
        assert snapshot.status_provenance == "legacy-inferred"

    def test_provisional_snapshot_with_pinned_status_exposes_pinned_provenance(
        self, tmp_path
    ) -> None:
        store = CatalogStore(tmp_path)
        refs = register_dataset_refs(store)
        write_snapshot(
            store,
            "provisional_1",
            list(refs.values()),
            certification=DatasetCertification.PROVISIONAL,
        )
        snapshot = SnapshotResolver(store).resolve("provisional_1")
        assert snapshot.outcome_status is not None
        assert snapshot.status_provenance == "pinned"


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


def _disclosure_record(page: int, receipt_date: str = "20240102") -> dict[str, str]:
    return {
        "rcept_no": f"2024010200000{page}",
        "rcept_dt": receipt_date,
        "corp_code": "00126380",
        "corp_name": "테스트",
        "report_nm": "현금배당결정",
        "rm": "",
    }


def _disclosure_month_text(year: int, month: int, records: list[dict[str, str]]) -> str:
    month_end = date(year, month, calendar.monthrange(year, month)[1])
    payload = {
        "version": f"dart-disclosures-month-{year:04d}-{month:02d}",
        "generated_time": "2026-01-01T00:00:00+00:00",
        "range_start": date(year, month, 1).isoformat(),
        "range_end": month_end.isoformat(),
        "record_count": len(records),
        "records": records,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _disclosure_parts(tmp_path: Path) -> Path:
    """Build a hash-validated two-month disclosure partition set for 2024-01..2024-02."""
    parts = tmp_path / "parts"
    (parts / "months").mkdir(parents=True)
    month_texts: dict[str, str] = {}
    for month, page in ((1, 1), (2, 2)):
        month_key = f"2024-{month:02d}"
        month_texts[month_key] = _disclosure_month_text(
            2024, month, [_disclosure_record(page, receipt_date=f"20240{month}02")]
        )
        (parts / "months" / f"{month_key}.json").write_text(
            month_texts[month_key], encoding="utf-8"
        )
    manifest = {
        "schema_version": "dart-disclosures-manifest-1",
        "requested_start": "2024-01-01",
        "requested_end": "2024-02-29",
        "months": {
            key: {
                "status": "complete",
                "path": f"months/{key}.json",
                "record_count": 1,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            for key, text in month_texts.items()
        },
    }
    (parts / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return parts


def _publish_collector() -> OpenDartEvidenceCollector:
    return OpenDartEvidenceCollector(
        api_key="fixture-key", generated_time=datetime(2026, 1, 1, tzinfo=UTC)
    )


class TestDisclosurePublication:
    NAME = "dart_disclosures_20240101_20240229_v1"

    def test_register_and_round_trip(self, tmp_path) -> None:
        parts = _disclosure_parts(tmp_path)
        output = tmp_path / f"{self.NAME}.json"
        catalog_root = tmp_path / "catalog"
        entry = _publish_collector().publish_disclosure_dataset(
            parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
        )
        assert entry.kind is CatalogKind.DISCLOSURES
        assert entry.completeness is EvidenceCompleteness.COMPLETE
        assert entry.coverage == CoverageRange(start=date(2024, 1, 1), end=date(2024, 2, 29))
        assert entry.row_count == 2
        assert entry.path == str(output)

        store = CatalogStore(catalog_root)
        assert store.list(CatalogKind.DISCLOSURES) == (entry,)

        again = _publish_collector().publish_disclosure_dataset(
            parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
        )
        assert again == entry
        assert len(store.log_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_rerun_with_fresh_collector_is_recoverable(self, tmp_path) -> None:
        parts = _disclosure_parts(tmp_path)
        output = tmp_path / f"{self.NAME}.json"
        catalog_root = tmp_path / "catalog"
        first = _publish_collector().publish_disclosure_dataset(
            parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
        )
        output_before = output.read_bytes()
        fresh = OpenDartEvidenceCollector(
            api_key="fixture-key", generated_time=datetime(2027, 5, 5, tzinfo=UTC)
        )
        again = fresh.publish_disclosure_dataset(
            parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
        )
        assert again == first
        assert output.read_bytes() == output_before
        store = CatalogStore(catalog_root)
        assert len(store.log_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_corrupt_month_fails_publication_without_output(self, tmp_path) -> None:
        parts = _disclosure_parts(tmp_path)
        month_path = parts / "months" / "2024-02.json"
        month_path.write_text(
            month_path.read_text(encoding="utf-8").replace(
                "20240102000002", "20240102999999"
            ),
            encoding="utf-8",
        )
        output = tmp_path / f"{self.NAME}.json"
        with pytest.raises(EvidenceCollectionError, match="2024-02"):
            _publish_collector().publish_disclosure_dataset(
                parts, date(2024, 1, 1), date(2024, 2, 29), output,
                tmp_path / "catalog", self.NAME,
            )
        assert not output.exists()

    def test_divergent_catalog_record_fails_without_mutation(self, tmp_path) -> None:
        parts = _disclosure_parts(tmp_path)
        output = tmp_path / f"{self.NAME}.json"
        catalog_root = tmp_path / "catalog"
        entry = _publish_collector().publish_disclosure_dataset(
            parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
        )
        store = CatalogStore(catalog_root)
        tampered = store.log_path.read_text(encoding="utf-8").replace(
            entry.content_hash, "0" * 64
        )
        store.log_path.write_text(tampered, encoding="utf-8")

        with pytest.raises(ValueError, match="different immutable fields"):
            _publish_collector().publish_disclosure_dataset(
                parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
            )
        assert store.list(CatalogKind.DISCLOSURES)[0].content_hash == "0" * 64

    def test_disclosure_publication_is_never_corporate_actions(self, tmp_path) -> None:
        parts = _disclosure_parts(tmp_path)
        output = tmp_path / f"{self.NAME}.json"
        catalog_root = tmp_path / "catalog"
        entry = _publish_collector().publish_disclosure_dataset(
            parts, date(2024, 1, 1), date(2024, 2, 29), output, catalog_root, self.NAME
        )
        assert entry.kind is CatalogKind.DISCLOSURES
        store = CatalogStore(catalog_root)
        assert store.list(CatalogKind.CORPORATE_ACTIONS) == ()
        assert store.get(CatalogKind.CORPORATE_ACTIONS, self.NAME) is None


class TestInventoryArchiveCandidates:
    def test_pinned_or_snapshot_referenced_never_archive_candidates(self, tmp_path) -> None:
        data_root = tmp_path / "data"
        evidence_dir = data_root / "evidence" / "stocks"
        evidence_dir.mkdir(parents=True)

        pinned_payload = b"shared calendar payload"
        pinned_file = evidence_dir / "calendar_v1.json"
        pinned_file.write_bytes(pinned_payload)
        duplicate = evidence_dir / "calendar_unregistered_duplicate.json"
        duplicate.write_bytes(pinned_payload)

        store = CatalogStore(tmp_path / "catalog")
        pinned_entry = CatalogEntry(
            kind=CatalogKind.CALENDAR,
            name="calendar_v1",
            content_hash=hashlib.sha256(pinned_payload).hexdigest(),
            schema_hash="schema",
            registered_at=REGISTERED,
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(pinned_file),
            pinned=True,
        )
        store.register(pinned_entry)

        master_payload = b"master payload"
        master_file = evidence_dir / "master_v1.json"
        master_file.write_bytes(master_payload)
        master_entry = CatalogEntry(
            kind=CatalogKind.INSTRUMENT_MASTER,
            name="master_v1",
            content_hash=hashlib.sha256(master_payload).hexdigest(),
            schema_hash="schema",
            registered_at=REGISTERED,
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(master_file),
        )
        store.register(master_entry)
        write_snapshot(store, "research_1", [pinned_entry, master_entry])

        report = scan_inventory(data_root, store)
        records = {record.path: record for record in report.records}

        calendar_record = records["evidence/stocks/calendar_v1.json"]
        assert calendar_record.lifecycle is InventoryLifecycle.EVIDENCE
        assert calendar_record.recommendation is Recommendation.RETAIN
        assert "calendar_v1" in calendar_record.catalog_reference
        assert "research_1" in calendar_record.snapshot_reference

        master_record = records["evidence/stocks/master_v1.json"]
        assert master_record.recommendation is Recommendation.RETAIN
        assert "research_1" in master_record.snapshot_reference

        duplicate_record = records["evidence/stocks/calendar_unregistered_duplicate.json"]
        assert duplicate_record.lifecycle is InventoryLifecycle.ARCHIVE_CANDIDATE
        assert duplicate_record.recommendation is Recommendation.CANDIDATE

        candidate_paths = {
            record.path
            for record in report.records
            if record.lifecycle is InventoryLifecycle.ARCHIVE_CANDIDATE
        }
        assert candidate_paths == {"evidence/stocks/calendar_unregistered_duplicate.json"}
