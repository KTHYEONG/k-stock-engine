"""Integration: base panel -> v2 features -> residual labels -> v2 snapshot -> compose -> train CLI."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
from src.core.instruments import AssetKind
from src.stocks.cli import build_research_v2 as build_research_v2_cli
from src.stocks.cli import train as train_cli
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
    SnapshotResolver,
    build_snapshot_manifest,
    register_file_evidence,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows, TimingConvention
from src.stocks.data.curation import (
    FeaturePanelRequest,
    build_feature_panel,
)
from src.stocks.data.labels import (
    build_label_dataset,
    publish_label_dataset,
)
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.data.research_v2 import (
    STOCK_ALPHA_V2_FEATURE_SET,
    StockAlphaV2MaterializationRequest,
    materialize_stock_alpha_v2_snapshot,
)
from src.stocks.research.features import stock_alpha_v2_allowlist
from src.stocks.research.labels import LabelDefinition
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

ALLOWLIST = stock_alpha_v2_allowlist()
GENERATED = datetime(2026, 1, 1, tzinfo=UTC)
TICKERS = tuple(f"KRX:{i:06d}" for i in range(1, 26))
CALENDAR_VERSION = "fixture-calendar"


def weekdays(n: int = 30) -> list[date]:
    days: list[date] = []
    cursor = date(2024, 1, 1)
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


SESSIONS = weekdays(30)
WINDOWS = ResearchWindows(
    train=CoverageRange(SESSIONS[2], SESSIONS[12]),
    validation=CoverageRange(SESSIONS[13], SESSIONS[17]),
    test=CoverageRange(SESSIONS[18], SESSIONS[22]),
)


def session_dt(session: date) -> datetime:
    return datetime.combine(session, datetime.min.time(), tzinfo=UTC)


def base_panel_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for t, ticker in enumerate(TICKERS):
        for s, session in enumerate(SESSIONS):
            open_price = 100.0 + float(s) + float(t % 7)
            row: dict[str, object] = {
                "instrument_id": ticker,
                "session": session_dt(session),
                "observation_time": session_dt(session),
                "available_time": session_dt(session),
                "open": open_price,
                "high": open_price + 1.0,
                "low": open_price - 1.0,
                "close": open_price + 0.5,
                "volume": 1_000_000.0,
                "trading_value": 1.05e8,
                "market_cap": 1e12,
                "sector": "S1",
                "action_interval_covered": True,
                "data_quality_status": "eligible",
                "data_quality_reason": None,
            }
            for j, name in enumerate(ALLOWLIST):
                row[f"raw__{name}"] = float((t * 31 + s * 7 + j) % 50) / 10.0
            rows.append(row)
    return pl.DataFrame(rows)


def write_base_panel(root: Path) -> Path:
    frame = base_panel_frame()
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="base_panel",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=session_dt(SESSIONS[0]),
        time_end=session_dt(SESSIONS[-1]),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        generated_time=GENERATED,
        schema_version="v2",
        content_hash=canonical_content_hash(frame, frame.columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    return ParquetDatasetStore(root).write_partitioned(
        frame,
        dataset_id="base_v1",
        manifest=manifest,
        expected_feature_set="base_panel",
        decision_time=GENERATED,
        content_manifest={"fixture": True},
    )


def write_calendar_evidence(root: Path) -> Path:
    path = root / "calendar.json"
    path.write_text(
        json.dumps(
            {
                "version": CALENDAR_VERSION,
                "sessions": [d.isoformat() for d in SESSIONS],
                "generated_time": GENERATED.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


def setup_catalog(
    catalog_root: Path, base_root: Path, calendar_path: Path
) -> CatalogStore:
    store = CatalogStore(catalog_root)
    base_entry = CatalogEntry(
        kind=CatalogKind.BASE_PANEL,
        name="base_v1",
        content_hash=ParquetDatasetStore(base_root).read_manifest("base_v1").content_hash,
        schema_hash="schema",
        registered_at=GENERATED,
        coverage=CoverageRange(SESSIONS[0], SESSIONS[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(base_root / "base_v1"),
    )
    store.register(base_entry)
    register_file_evidence(
        store,
        kind=CatalogKind.CALENDAR,
        name="calendar_v1",
        path=calendar_path,
        coverage=CoverageRange(SESSIONS[0], SESSIONS[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        registered_at=GENERATED,
    )
    return store


def build_source_snapshot(
    catalog_root: Path, store: CatalogStore, *, certification: DatasetCertification
) -> None:
    references = [
        store.require(CatalogKind.BASE_PANEL, "base_v1"),
        store.require(CatalogKind.CALENDAR, "calendar_v1"),
    ]
    manifest = build_snapshot_manifest(
        snapshot_id="source_snap_v1",
        certification=certification,
        timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
        windows=WINDOWS,
        references=tuple(references),
    )
    path = catalog_root / "snapshots" / "source_snap_v1" / "snapshot_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json(), sort_keys=True, indent=2), encoding="utf-8")


def v2_request(
    catalog_root: Path, base_root: Path, feature_root: Path, label_root: Path,
    *, calendar_path: Path,
) -> StockAlphaV2MaterializationRequest:
    return StockAlphaV2MaterializationRequest(
        source_snapshot_id="source_snap_v1",
        feature_dataset_id="features_v2",
        label_dataset_id="labels_v2",
        snapshot_id="snap_v2",
        catalog_root=catalog_root,
        base_root=base_root,
        feature_root=feature_root,
        label_root=label_root,
        generated_time=GENERATED,
        windows=WINDOWS,
        certification=DatasetCertification.PROVISIONAL,
        calendar_path=calendar_path,
    )


@pytest.fixture
def v2_pipeline(tmp_path) -> dict[str, Path]:
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    catalog_root = tmp_path / "catalog"
    write_base_panel(base_root)
    calendar_path = write_calendar_evidence(tmp_path)
    store = setup_catalog(catalog_root, base_root, calendar_path)
    build_source_snapshot(
        catalog_root, store, certification=DatasetCertification.PROVISIONAL
    )
    result = materialize_stock_alpha_v2_snapshot(
        v2_request(catalog_root, base_root, feature_root, label_root, calendar_path=calendar_path)
    )
    assert result.snapshot_id == "snap_v2"
    return {
        "base_root": base_root,
        "feature_root": feature_root,
        "label_root": label_root,
        "catalog_root": catalog_root,
        "calendar_path": calendar_path,
    }


class TestStockAlphaV2SnapshotIntegration:
    def test_pipeline_composes_labeled_training_snapshot(self, v2_pipeline) -> None:
        repository = ResearchDataRepository(
            base_root=v2_pipeline["base_root"],
            feature_root=v2_pipeline["feature_root"],
            label_root=v2_pipeline["label_root"],
        )
        store = CatalogStore(v2_pipeline["catalog_root"])
        snapshot = SnapshotResolver(store).resolve("snap_v2")
        composed = repository.compose_labeled_training_snapshot(
            snapshot,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            decision_time=GENERATED,
        )
        assert composed.frame.height > 0
        feature_columns = [
            c for c in composed.frame.columns if c.startswith("feature__")
        ]
        assert feature_columns == [f"feature__{name}" for name in ALLOWLIST]
        assert "residual_o2o_5d" in composed.frame.columns
        assert "relevance" in composed.frame.columns
        assert "label_available_time" in composed.frame.columns

    def test_v1_snapshot_fails_v2_composition_and_is_unchanged(
        self, v2_pipeline, tmp_path
    ) -> None:
        base_root = v2_pipeline["base_root"]
        feature_root = v2_pipeline["feature_root"]
        label_root = v2_pipeline["label_root"]
        catalog_root = v2_pipeline["catalog_root"]
        store = CatalogStore(catalog_root)

        feature_result = build_feature_panel(
            base_root,
            feature_root,
            FeaturePanelRequest(
                dataset_id="features_v1",
                base_panel_id="base_v1",
                feature_set="stock_alpha_v1",
                generated_time=GENERATED,
            ),
        )
        feature_entry = CatalogEntry(
            kind=CatalogKind.FEATURES,
            name="features_v1",
            content_hash=feature_result.manifest.content_hash,
            schema_hash=feature_result.manifest.schema_hash,
            registered_at=GENERATED,
            coverage=CoverageRange(SESSIONS[0], SESSIONS[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(feature_root / "features_v1"),
        )
        store.register(feature_entry)

        base_frame = ParquetDatasetStore(base_root).read(
            "base_v1", AssetKind.STOCK, "base_panel", GENERATED
        )
        calendar = KRXSessionCalendar(
            version=CALENDAR_VERSION,
            sessions=tuple(SESSIONS),
            generated_time=GENERATED,
        )
        labels = build_label_dataset(
            base_frame,
            calendar,
            LabelDefinition(
                name="fwd_ret_5d", entry_field="open", exit_field="close", horizon_sessions=5
            ),
        )
        published = publish_label_dataset(
            labels,
            destination_root=label_root,
            dataset_id="labels_v1",
            base_panel_hash=ParquetDatasetStore(base_root)
            .read_manifest("base_v1")
            .content_hash,
            calendar_hash=calendar.content_hash,
            definition=LabelDefinition(
                name="fwd_ret_5d", entry_field="open", exit_field="close", horizon_sessions=5
            ),
            generated_time=GENERATED,
        )
        label_entry = CatalogEntry(
            kind=CatalogKind.LABELS,
            name="labels_v1",
            content_hash=published.manifest.content_hash,
            schema_hash=published.manifest.schema_hash,
            registered_at=GENERATED,
            coverage=CoverageRange(SESSIONS[0], SESSIONS[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(label_root / "labels_v1"),
        )
        store.register(label_entry)

        v1_snapshot = build_snapshot_manifest(
            snapshot_id="v1_snap",
            certification=DatasetCertification.PROVISIONAL,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=WINDOWS,
            references=(
                store.require(CatalogKind.BASE_PANEL, "base_v1"),
                store.require(CatalogKind.CALENDAR, "calendar_v1"),
                feature_entry,
                label_entry,
            ),
        )
        v1_path = catalog_root / "snapshots" / "v1_snap" / "snapshot_manifest.json"
        v1_path.parent.mkdir(parents=True, exist_ok=True)
        v1_path.write_text(
            json.dumps(v1_snapshot.to_json(), sort_keys=True, indent=2), encoding="utf-8"
        )

        repository = ResearchDataRepository(
            base_root=base_root,
            feature_root=feature_root,
            label_root=label_root,
        )
        snapshot = SnapshotResolver(store).resolve("v1_snap")
        with pytest.raises(ValueError, match="feature_set mismatch"):
            repository.compose_labeled_training_snapshot(
                snapshot,
                feature_set=STOCK_ALPHA_V2_FEATURE_SET,
                decision_time=GENERATED,
            )

        assert v1_path.read_bytes() == v1_path.read_bytes()
        assert (feature_root / "features_v1" / "dataset_manifest.json").exists()
        assert (label_root / "labels_v1" / "dataset_manifest.json").exists()

    def test_train_cli_reaches_training_without_contract_error(
        self, v2_pipeline, tmp_path, monkeypatch, capsys
    ) -> None:
        artifact_root = tmp_path / "artifacts"
        exit_code = train_cli.main(
            [
                "--artifact-id",
                "stock_alpha_v2_integration",
                "--snapshot-id",
                "snap_v2",
                "--catalog-root",
                str(v2_pipeline["catalog_root"]),
                "--base-root",
                str(v2_pipeline["base_root"]),
                "--feature-root",
                str(v2_pipeline["feature_root"]),
                "--label-root",
                str(v2_pipeline["label_root"]),
                "--registry",
                str(artifact_root),
                "--mode",
                "research",
                "--decision-time",
                GENERATED.isoformat(),
            ]
        )
        assert exit_code == 0
        metrics_path = artifact_root / "stock_alpha_v2_integration" / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["no_trade"] is True

    def test_build_research_v2_cli_prints_ids_and_hashes(self, v2_pipeline, capsys) -> None:
        root = v2_pipeline["catalog_root"]
        exit_code = build_research_v2_cli.main(
            [
                "--source-snapshot-id",
                "source_snap_v1",
                "--feature-dataset-id",
                "features_v2_cli",
                "--label-dataset-id",
                "labels_v2_cli",
                "--snapshot-id",
                "snap_v2_cli",
                "--catalog-root",
                str(root),
                "--base-root",
                str(v2_pipeline["base_root"]),
                "--feature-root",
                str(v2_pipeline["feature_root"]),
                "--label-root",
                str(v2_pipeline["label_root"]),
                "--calendar-path",
                str(v2_pipeline["calendar_path"]),
                "--train-start",
                WINDOWS.train.start.isoformat(),
                "--train-end",
                WINDOWS.train.end.isoformat(),
                "--validation-start",
                WINDOWS.validation.start.isoformat(),
                "--validation-end",
                WINDOWS.validation.end.isoformat(),
                "--test-start",
                WINDOWS.test.start.isoformat(),
                "--test-end",
                WINDOWS.test.end.isoformat(),
                "--certification",
                "provisional",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "label_horizon_mode=five_day" in out
        assert "feature_dataset_id=features_v2_cli" in out
        assert "label_dataset_id=labels_v2_cli" in out
        assert "snapshot_id=snap_v2_cli" in out
        assert "feature_content_hash=" in out
        assert "label_content_hash=" in out
        assert "certification=provisional" in out

        store = CatalogStore(root)
        snapshot = SnapshotResolver(store).resolve("snap_v2_cli")
        repository = ResearchDataRepository(
            base_root=v2_pipeline["base_root"],
            feature_root=v2_pipeline["feature_root"],
            label_root=v2_pipeline["label_root"],
        )
        composed = repository.compose_labeled_training_snapshot(
            snapshot,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            decision_time=GENERATED,
        )
        assert "residual_o2o_5d" in composed.frame.columns
        assert "relevance" in composed.frame.columns
        assert "label_available_time" in composed.frame.columns
        assert not any(
            c.startswith("residual_o2o_") and c != "residual_o2o_5d"
            for c in composed.frame.columns
        )

    def test_build_research_v2_cli_multi_horizon_publishes_route_columns(
        self, v2_pipeline, capsys
    ) -> None:
        root = v2_pipeline["catalog_root"]
        exit_code = build_research_v2_cli.main(
            [
                "--source-snapshot-id",
                "source_snap_v1",
                "--feature-dataset-id",
                "features_v3_cli",
                "--label-dataset-id",
                "labels_v3_cli",
                "--snapshot-id",
                "snap_v3_cli",
                "--label-horizon-mode",
                "multi_horizon",
                "--catalog-root",
                str(root),
                "--base-root",
                str(v2_pipeline["base_root"]),
                "--feature-root",
                str(v2_pipeline["feature_root"]),
                "--label-root",
                str(v2_pipeline["label_root"]),
                "--calendar-path",
                str(v2_pipeline["calendar_path"]),
                "--train-start",
                SESSIONS[2].isoformat(),
                "--train-end",
                SESSIONS[8].isoformat(),
                "--validation-start",
                SESSIONS[9].isoformat(),
                "--validation-end",
                SESSIONS[10].isoformat(),
                "--test-start",
                SESSIONS[11].isoformat(),
                "--test-end",
                SESSIONS[13].isoformat(),
                "--certification",
                "provisional",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "label_horizon_mode=multi_horizon" in out
        assert "feature_dataset_id=features_v3_cli" in out
        assert "label_dataset_id=labels_v3_cli" in out
        assert "snapshot_id=snap_v3_cli" in out

        store = CatalogStore(root)
        snapshot = SnapshotResolver(store).resolve("snap_v3_cli")
        repository = ResearchDataRepository(
            base_root=v2_pipeline["base_root"],
            feature_root=v2_pipeline["feature_root"],
            label_root=v2_pipeline["label_root"],
        )
        composed = repository.compose_labeled_training_snapshot(
            snapshot,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            decision_time=GENERATED,
        )
        expected_columns = ["instrument_id", "session"]
        for h in (5, 10, 15):
            expected_columns += [
                f"residual_o2o_{h}d",
                f"relevance_{h}d",
                f"label_available_time_{h}d",
            ]
        assert all(c in composed.frame.columns for c in expected_columns)
        assert "relevance" not in composed.frame.columns

    def test_provisional_snapshot_rejected_for_paper_mode(self, v2_pipeline) -> None:
        with pytest.raises(ValueError, match="provisional"):
            resolve_snapshot_for_mode(
                v2_pipeline["catalog_root"], "snap_v2", mode="paper"
            )
