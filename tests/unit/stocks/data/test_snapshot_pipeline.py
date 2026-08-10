"""End-to-end: base panel -> feature panel -> labels -> snapshot -> resolve -> compose."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import DatasetCertification
from src.core.instruments import AssetKind
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
    build_snapshot_manifest,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows, TimingConvention
from src.stocks.data.curation import (
    BasePanelRequest,
    FeaturePanelRequest,
    build_base_panel,
    build_feature_panel,
)
from src.stocks.data.labels import LABEL_FEATURE_SET, build_label_dataset, publish_label_dataset
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.research.labels import LabelDefinition
from src.storage.parquet_datasets import ParquetDatasetStore

DATES = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]
GENERATED = datetime(2026, 1, 1, tzinfo=UTC)
CALENDAR = KRXSessionCalendar(
    version="fixture-calendar",
    sessions=tuple(DATES),
    generated_time=GENERATED,
)
DEFINITION = LabelDefinition(
    name="fwd_ret_2d", entry_field="open", exit_field="close", horizon_sessions=2
)
WINDOWS = ResearchWindows(
    train=CoverageRange(start=date(2024, 1, 2), end=date(2024, 1, 4)),
    validation=CoverageRange(start=date(2024, 1, 5), end=date(2024, 1, 5)),
    test=CoverageRange(start=date(2024, 1, 8), end=date(2024, 1, 8)),
)
EVIDENCE_RANGE = CoverageRange(start=date(2024, 1, 1), end=date(2024, 1, 31))


def legacy_row(day_index: int, ticker: str = "000050") -> dict[str, object]:
    return {
        "date": DATES[day_index],
        "ticker": ticker,
        "open": 100.0 + day_index,
        "high": 110.0 + day_index,
        "low": 90.0 + day_index,
        "close": 105.0 + day_index,
        "volume": 1_000_000.0,
        "trading_value": 1.05e8,
        "market_cap": 1e12,
        "sector": "S1",
        "log_return_5d": 0.1 + 0.01 * day_index,
        "volatility_20d": 0.2,
        "total_assets": 1e12,
        "total_assets_right": 1e12,
        "target_return_5d": 0.05,
    }


def fixture_source(root: Path) -> Path:
    for i, day in enumerate(DATES):
        year_dir = root / f"year={day.year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([legacy_row(i)]).write_parquet(year_dir / f"{day.isoformat()}_feat.parquet")
    return root


def register_evidence(store: CatalogStore) -> None:
    for kind in (
        CatalogKind.CALENDAR,
        CatalogKind.INSTRUMENT_MASTER,
        CatalogKind.DISCLOSURES,
        CatalogKind.CORPORATE_ACTIONS,
        CatalogKind.COSTS,
    ):
        store.register(
            CatalogEntry(
                kind=kind,
                name=f"{kind.value}_v1",
                content_hash=f"evidence-{kind.value}",
                schema_hash="schema",
                registered_at=GENERATED,
                coverage=EVIDENCE_RANGE,
                completeness=EvidenceCompleteness.COMPLETE,
                path=f"data/evidence/{kind.value}_v1",
            )
        )


def build_pipeline(tmp_path: Path) -> dict[str, object]:
    source = fixture_source(tmp_path / "source")
    canonical = tmp_path / "canonical"
    derived = tmp_path / "derived"
    catalog_root = tmp_path / "catalog"
    store = CatalogStore(catalog_root)

    base = build_base_panel(
        source,
        canonical,
        BasePanelRequest(
            dataset_id="base_panel_v1",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            calendar=CALENDAR,
            generated_time=GENERATED,
        ),
    )

    features = build_feature_panel(
        canonical,
        derived,
        FeaturePanelRequest(
            dataset_id="features_v1",
            base_panel_id="base_panel_v1",
            feature_set="stock_alpha_v1",
            generated_time=GENERATED,
        ),
    )

    base_frame = ParquetDatasetStore(canonical).read(
        "base_panel_v1", AssetKind.STOCK, "base_panel", GENERATED
    )
    labels = build_label_dataset(base_frame, CALENDAR, DEFINITION)
    published = publish_label_dataset(
        labels,
        destination_root=canonical,
        dataset_id="labels_v1",
        base_panel_hash=base.manifest.content_hash,
        calendar_hash=CALENDAR.content_hash,
        definition=DEFINITION,
        generated_time=GENERATED,
    )

    register_evidence(store)
    base_entry = CatalogEntry(
        kind=CatalogKind.BASE_PANEL,
        name="base_panel_v1",
        content_hash=base.manifest.content_hash,
        schema_hash=base.manifest.schema_hash,
        registered_at=GENERATED,
        coverage=CoverageRange(start=DATES[0], end=DATES[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(canonical / "base_panel_v1"),
    )
    feature_entry = CatalogEntry(
        kind=CatalogKind.FEATURES,
        name="features_v1",
        content_hash=features.manifest.content_hash,
        schema_hash=features.manifest.schema_hash,
        registered_at=GENERATED,
        coverage=CoverageRange(start=DATES[0], end=DATES[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(derived / "features_v1"),
    )
    label_entry = CatalogEntry(
        kind=CatalogKind.LABELS,
        name="labels_v1",
        content_hash=published.manifest.content_hash,
        schema_hash=published.manifest.schema_hash,
        registered_at=GENERATED,
        coverage=CoverageRange(start=DATES[0], end=DATES[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(canonical / "labels_v1"),
    )
    for entry_ in (base_entry, feature_entry, label_entry):
        store.register(entry_)

    snapshot = build_snapshot_manifest(
        snapshot_id="research_snap_1",
        certification=DatasetCertification.RESEARCH,
        timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
        windows=WINDOWS,
        references=(
            base_entry,
            feature_entry,
            label_entry,
            store.require(CatalogKind.CALENDAR, "calendar_v1"),
            store.require(CatalogKind.INSTRUMENT_MASTER, "instrument_master_v1"),
            store.require(CatalogKind.DISCLOSURES, "disclosures_v1"),
            store.require(CatalogKind.CORPORATE_ACTIONS, "corporate_actions_v1"),
            store.require(CatalogKind.COSTS, "costs_v1"),
        ),
    )
    manifest_path = catalog_root / "snapshots" / "research_snap_1" / "snapshot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(snapshot.to_json(), sort_keys=True, indent=2), encoding="utf-8"
    )
    return {
        "store": store,
        "canonical": canonical,
        "derived": derived,
        "base": base,
        "features": features,
        "labels": published,
    }


class TestSnapshotPipeline:
    def test_base_feature_label_datasets_are_immutable_v2(self, tmp_path) -> None:
        pipeline = build_pipeline(tmp_path)
        base = pipeline["base"]
        features = pipeline["features"]
        labels = pipeline["labels"]
        assert base.manifest.schema_version == "v2"
        assert features.manifest.schema_version == "v2"
        assert labels.manifest.schema_version == "v2"
        assert base.manifest.content_hash
        assert features.manifest.content_hash
        assert labels.manifest.content_hash
        assert features.manifest.feature_set == "stock_alpha_v1"
        assert labels.manifest.feature_set == LABEL_FEATURE_SET

    def test_feature_panel_has_no_target_or_label_columns(self, tmp_path) -> None:
        pipeline = build_pipeline(tmp_path)
        store = ParquetDatasetStore(pipeline["derived"])
        frame = store.read("features_v1", AssetKind.STOCK, "stock_alpha_v1", GENERATED)
        assert not any(c.startswith(("target_", "label_")) for c in frame.columns)
        assert "feature__log_return_5d" in frame.columns
        assert "feature__total_assets" in frame.columns
        assert "feature__total_assets_right" not in frame.columns

    def test_snapshot_resolves_and_composes(self, tmp_path) -> None:
        pipeline = build_pipeline(tmp_path)
        from src.stocks.data.catalog import SnapshotResolver

        snapshot = SnapshotResolver(pipeline["store"]).resolve("research_snap_1")
        assert snapshot.manifest.snapshot_id == "research_snap_1"
        assert snapshot.manifest.manifest_hash

        repository = ResearchDataRepository(
            base_root=pipeline["canonical"],
            feature_root=pipeline["derived"],
            label_root=pipeline["canonical"],
        )
        composed = repository.compose_training_snapshot(
            snapshot,
            feature_set="stock_alpha_v1",
            decision_time=GENERATED,
        )
        assert composed.frame.height == len(DATES)
        assert {"close", "open", "feature__log_return_5d"}.issubset(composed.frame.columns)
        assert not any(
            c.startswith(("target_", "label_")) for c in composed.frame.columns
        )

    def test_provisional_snapshot_rejected_for_paper_live(self, tmp_path) -> None:
        pipeline = build_pipeline(tmp_path)
        from src.stocks.data.catalog import SnapshotResolver

        store = pipeline["store"]
        entries = store.list()
        provisional = build_snapshot_manifest(
            snapshot_id="provisional_snap_1",
            certification=DatasetCertification.PROVISIONAL,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=WINDOWS,
            references=tuple(
                store.require(entry.kind, entry.name)
                for entry in entries
                if entry.kind is not CatalogKind.SNAPSHOT
            ),
        )
        path = store.root / "snapshots" / "provisional_snap_1" / "snapshot_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provisional.to_json(), sort_keys=True, indent=2), encoding="utf-8")

        snapshot = SnapshotResolver(store).resolve("provisional_snap_1")
        assert snapshot.manifest.certification is DatasetCertification.PROVISIONAL
        with pytest.raises(ValueError, match="provisional"):
            resolve_snapshot_for_mode(tmp_path / "catalog", "provisional_snap_1", mode="paper")

    def test_catalog_cli_validates_snapshot(self, tmp_path, capsys) -> None:
        from src.stocks.cli import catalog as catalog_cli

        pipeline = build_pipeline(tmp_path)
        root = str(pipeline["store"].root)
        assert catalog_cli.main(["--catalog-root", root, "validate", "--snapshot-id", "research_snap_1"]) == 0
        captured = capsys.readouterr().out
        assert "research_snap_1: OK" in captured
