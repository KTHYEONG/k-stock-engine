"""Snapshotless data access: catalog-driven lineage resolution tests."""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.core.datasets import DatasetCertification
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows
from src.stocks.data.lineage import (
    CatalogCompatibilityResolver,
    DataSelectionRequest,
    ResolvedDataLineage,
)

RANGE = CoverageRange(start=date(2024, 1, 1), end=date(2024, 3, 31))
WINDOWS = ResearchWindows(
    train=CoverageRange(start=date(2024, 1, 1), end=date(2024, 1, 31)),
    validation=CoverageRange(start=date(2024, 2, 1), end=date(2024, 2, 15)),
    test=CoverageRange(start=date(2024, 2, 16), end=date(2024, 3, 31)),
)
REGISTERED = datetime(2026, 1, 1, tzinfo=UTC)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


def entry(
    kind: CatalogKind,
    name: str,
    *,
    content_hash: str = "abc",
    coverage: CoverageRange | None = RANGE,
    completeness: EvidenceCompleteness = EvidenceCompleteness.COMPLETE,
    registered_at: datetime = REGISTERED,
) -> CatalogEntry:
    return CatalogEntry(
        kind=kind,
        name=name,
        content_hash=content_hash,
        schema_hash="schema",
        registered_at=registered_at,
        coverage=coverage,
        completeness=completeness,
        path=f"data/{name}",
    )


def register_all_entries(store: CatalogStore) -> None:
    entries = [
        entry(CatalogKind.BASE_PANEL, "base_v1", content_hash="base_hash"),
        entry(CatalogKind.FEATURES, "features_v1", content_hash="feat_hash"),
        entry(CatalogKind.LABELS, "labels_v1", content_hash="label_hash"),
        entry(CatalogKind.OUTCOME_STATUS, "status_v1", content_hash="status_hash"),
        entry(CatalogKind.OUTCOME_EVIDENCE, "evidence_v1", content_hash="evidence_hash"),
        entry(CatalogKind.CALENDAR, "calendar_v1", content_hash="cal_hash"),
        entry(CatalogKind.INSTRUMENT_MASTER, "master_v1", content_hash="master_hash"),
        entry(CatalogKind.CORPORATE_ACTIONS, "actions_v1", content_hash="actions_hash"),
        entry(CatalogKind.COSTS, "costs_v1", content_hash="costs_hash"),
    ]
    for e in entries:
        store.register(e)


def _make_request(**overrides) -> DataSelectionRequest:
    defaults: dict[str, object] = {
        "asset_kind": "stock",
        "feature_set": "stock_net_alpha_v1",
        "label_definition": "net_alpha_o2o",
        "candidate_horizons": (5, 10, 15),
        "as_of": AS_OF,
        "research_range": RANGE,
        "minimum_outcome_coverage": 0.0,
        "required_certification": DatasetCertification.RESEARCH,
    }
    defaults.update(overrides)
    return DataSelectionRequest(**defaults)


class TestDirectSelectionHashEquivalence:
    """SDA-01: direct selection resolves one compatible entry per kind."""

    def test_resolves_all_required_kinds(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        register_all_entries(store)
        resolver = CatalogCompatibilityResolver(store)
        lineage = resolver.resolve(_make_request())

        assert isinstance(lineage, ResolvedDataLineage)
        assert lineage.selection_policy == "latest_complete_compatible_v1"
        assert lineage.as_of == AS_OF
        assert lineage.research_range == RANGE
        assert "base_panel" in lineage.entries
        assert "features" in lineage.entries
        assert "labels" in lineage.entries
        assert lineage.entries["base_panel"].content_hash == "base_hash"
        assert lineage.entries["features"].content_hash == "feat_hash"
        assert lineage.entries["labels"].content_hash == "label_hash"

    def test_lineage_to_json_round_trip(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        register_all_entries(store)
        resolver = CatalogCompatibilityResolver(store)
        lineage = resolver.resolve(_make_request())

        payload = lineage.to_json()
        assert payload["selection_policy"] == "latest_complete_compatible_v1"
        assert payload["as_of"] == AS_OF.isoformat()
        assert "base_panel" in payload["entries"]
        assert "compatibility_hash" in payload
        assert isinstance(payload["outcome_coverage"], dict)

    def test_compatibility_hash_is_deterministic(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        register_all_entries(store)
        resolver = CatalogCompatibilityResolver(store)
        first = resolver.resolve(_make_request())
        second = resolver.resolve(_make_request())
        assert first.compatibility_hash == second.compatibility_hash

    def test_content_hash_change_alters_compatibility_hash(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        register_all_entries(store)
        resolver = CatalogCompatibilityResolver(store)
        first = resolver.resolve(_make_request())

        store2 = CatalogStore(tmp_path / "v2")
        register_all_entries(store2)
        tampered = entry(
            CatalogKind.BASE_PANEL, "base_v2", content_hash="different_hash"
        )
        store2.register(tampered)
        second_resolver = CatalogCompatibilityResolver(store2)
        second = second_resolver.resolve(_make_request())
        assert first.compatibility_hash != second.compatibility_hash


class TestDeterministicSelectionAndAmbiguity:
    """SDA-02: deterministic ordering and tie-breaking."""

    def test_selects_latest_by_generated_time(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        early = datetime(2025, 1, 1, tzinfo=UTC)
        late = datetime(2025, 6, 1, tzinfo=UTC)
        store.register(
            entry(CatalogKind.BASE_PANEL, "base_old", content_hash="old", registered_at=early)
        )
        store.register(
            entry(CatalogKind.BASE_PANEL, "base_new", content_hash="new", registered_at=late)
        )
        for kind in (
            CatalogKind.FEATURES,
            CatalogKind.LABELS,
            CatalogKind.OUTCOME_STATUS,
            CatalogKind.OUTCOME_EVIDENCE,
            CatalogKind.CALENDAR,
            CatalogKind.INSTRUMENT_MASTER,
            CatalogKind.CORPORATE_ACTIONS,
            CatalogKind.COSTS,
        ):
            store.register(entry(kind, f"{kind.value}_v1"))
        resolver = CatalogCompatibilityResolver(store)
        lineage = resolver.resolve(_make_request())
        assert lineage.entries["base_panel"].name == "base_new"

    def test_raises_on_equal_ordering_tie(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        same_time = datetime(2025, 1, 1, tzinfo=UTC)
        store.register(
            entry(
                CatalogKind.BASE_PANEL,
                "base_a",
                content_hash="same_hash",
                registered_at=same_time,
            )
        )
        store.register(
            entry(
                CatalogKind.BASE_PANEL,
                "base_b",
                content_hash="same_hash",
                registered_at=same_time,
            )
        )
        for kind in (
            CatalogKind.FEATURES,
            CatalogKind.LABELS,
            CatalogKind.OUTCOME_STATUS,
            CatalogKind.OUTCOME_EVIDENCE,
            CatalogKind.CALENDAR,
            CatalogKind.INSTRUMENT_MASTER,
            CatalogKind.CORPORATE_ACTIONS,
            CatalogKind.COSTS,
        ):
            store.register(entry(kind, f"{kind.value}_v1"))
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="ambiguous compatible datasets"):
            resolver.resolve(_make_request())

    def test_rejects_future_entries(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        future = datetime(2099, 1, 1, tzinfo=UTC)
        store.register(
            entry(CatalogKind.BASE_PANEL, "base_future", registered_at=future)
        )
        for kind in (
            CatalogKind.FEATURES,
            CatalogKind.LABELS,
            CatalogKind.OUTCOME_STATUS,
            CatalogKind.OUTCOME_EVIDENCE,
            CatalogKind.CALENDAR,
            CatalogKind.INSTRUMENT_MASTER,
            CatalogKind.CORPORATE_ACTIONS,
            CatalogKind.COSTS,
        ):
            store.register(entry(kind, f"{kind.value}_v1"))
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="no compatible base_panel"):
            resolver.resolve(_make_request())


class TestIncompatibleOrFutureDatasetFailsClosed:
    """SDA-07: incompatible/future datasets fail closed before any read."""

    def test_rejects_incomplete_evidence_for_research_certification(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.BASE_PANEL, "base_v1"))
        store.register(entry(CatalogKind.FEATURES, "features_v1"))
        store.register(entry(CatalogKind.LABELS, "labels_v1"))
        store.register(entry(CatalogKind.OUTCOME_STATUS, "status_v1"))
        store.register(entry(CatalogKind.OUTCOME_EVIDENCE, "evidence_v1"))
        store.register(entry(CatalogKind.CALENDAR, "calendar_v1"))
        store.register(entry(CatalogKind.INSTRUMENT_MASTER, "master_v1"))
        store.register(
            entry(
                CatalogKind.CORPORATE_ACTIONS,
                "actions_v1",
                completeness=EvidenceCompleteness.INCOMPLETE,
            )
        )
        store.register(entry(CatalogKind.COSTS, "costs_v1"))
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="not complete evidence"):
            resolver.resolve(_make_request())

    def test_rejects_candidate_only_entry(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(
            entry(
                CatalogKind.BASE_PANEL,
                "base_candidate",
                completeness=EvidenceCompleteness.CANDIDATE_ONLY,
            )
        )
        for kind in (
            CatalogKind.FEATURES,
            CatalogKind.LABELS,
            CatalogKind.OUTCOME_STATUS,
            CatalogKind.OUTCOME_EVIDENCE,
            CatalogKind.CALENDAR,
            CatalogKind.INSTRUMENT_MASTER,
            CatalogKind.CORPORATE_ACTIONS,
            CatalogKind.COSTS,
        ):
            store.register(entry(kind, f"{kind.value}_v1"))
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="no compatible base_panel"):
            resolver.resolve(_make_request())

    def test_rejects_range_incomplete_entry(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        short_range = CoverageRange(start=date(2024, 1, 1), end=date(2024, 2, 1))
        store.register(
            entry(
                CatalogKind.BASE_PANEL,
                "base_short",
                coverage=short_range,
            )
        )
        for kind in (
            CatalogKind.FEATURES,
            CatalogKind.LABELS,
            CatalogKind.OUTCOME_STATUS,
            CatalogKind.OUTCOME_EVIDENCE,
            CatalogKind.CALENDAR,
            CatalogKind.INSTRUMENT_MASTER,
            CatalogKind.CORPORATE_ACTIONS,
            CatalogKind.COSTS,
        ):
            store.register(entry(kind, f"{kind.value}_v1"))
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="no compatible base_panel"):
            resolver.resolve(_make_request())

    def test_provisional_allows_incomplete_evidence(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.BASE_PANEL, "base_v1"))
        store.register(entry(CatalogKind.FEATURES, "features_v1"))
        store.register(entry(CatalogKind.LABELS, "labels_v1"))
        store.register(entry(CatalogKind.OUTCOME_STATUS, "status_v1"))
        store.register(entry(CatalogKind.OUTCOME_EVIDENCE, "evidence_v1"))
        store.register(entry(CatalogKind.CALENDAR, "calendar_v1"))
        store.register(entry(CatalogKind.INSTRUMENT_MASTER, "master_v1"))
        store.register(
            entry(
                CatalogKind.CORPORATE_ACTIONS,
                "actions_incomplete",
                completeness=EvidenceCompleteness.INCOMPLETE,
            )
        )
        store.register(entry(CatalogKind.COSTS, "costs_v1"))
        resolver = CatalogCompatibilityResolver(store)
        lineage = resolver.resolve(
            _make_request(required_certification=DatasetCertification.PROVISIONAL)
        )
        assert lineage.entries["corporate_actions"].name == "actions_incomplete"

    def test_missing_required_kind_raises(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        store.register(entry(CatalogKind.BASE_PANEL, "base_v1"))
        store.register(entry(CatalogKind.FEATURES, "features_v1"))
        for kind in (
            CatalogKind.OUTCOME_STATUS,
            CatalogKind.OUTCOME_EVIDENCE,
            CatalogKind.CALENDAR,
            CatalogKind.INSTRUMENT_MASTER,
            CatalogKind.CORPORATE_ACTIONS,
            CatalogKind.COSTS,
        ):
            store.register(entry(kind, f"{kind.value}_v1"))
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="no compatible labels"):
            resolver.resolve(_make_request())

    def test_empty_catalog_raises(self, tmp_path) -> None:
        store = CatalogStore(tmp_path)
        resolver = CatalogCompatibilityResolver(store)
        with pytest.raises(ValueError, match="no compatible base_panel"):
            resolver.resolve(_make_request())
