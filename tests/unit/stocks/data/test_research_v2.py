"""Unit tests for stock_alpha_v2 feature panel, residual labels, and snapshot materialization."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
from src.core.instruments import AssetKind
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
    build_snapshot_manifest,
    register_file_evidence,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows, TimingConvention
from src.stocks.data.curation import (
    FeaturePanelRequest,
    V2_READINESS_NAME,
    build_stock_alpha_v2_feature_panel,
)
from src.stocks.data.labels import (
    LABEL_AVAILABLE_COLUMN,
    build_residual_o2o_label_dataset,
)
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.data.repositories import ResearchDataRepository
from src.stocks.data.research_v2 import (
    STOCK_ALPHA_V2_FEATURE_SET,
    StockAlphaV2MaterializationRequest,
    materialize_stock_alpha_v2_snapshot,
)
from src.stocks.research.features import stock_alpha_v2_allowlist
from src.storage.parquet_datasets import (
    ParquetDatasetStore,
    canonical_content_hash,
)

ALLOWLIST = stock_alpha_v2_allowlist()
GENERATED = datetime(2026, 1, 1, tzinfo=UTC)
TICKERS = tuple(f"KRX:{i:06d}" for i in range(1, 26))


def weekdays(n: int = 20) -> list[date]:
    days: list[date] = []
    cursor = date(2024, 1, 1)
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


SESSIONS = weekdays(20)


def test_materialization_uses_projection() -> None:
    """The request contract makes raw and compact outcome sources exclusive."""
    # materialization_uses_projection
    from src.stocks.data.research_v2 import NetAlphaMaterializationRequest

    windows = ResearchWindows(
        train=CoverageRange(start=date(2020, 1, 1), end=date(2020, 1, 2)),
        validation=CoverageRange(start=date(2020, 1, 3), end=date(2020, 1, 4)),
        test=CoverageRange(start=date(2020, 1, 5), end=date(2020, 1, 6)),
    )
    with pytest.raises(ValueError, match="either raw_bar_dataset_id or outcome_open_bar_dataset_id"):
        NetAlphaMaterializationRequest(
            source_snapshot_id="source", feature_dataset_id="features", label_dataset_id="labels",
            snapshot_id="snapshot", catalog_root=Path("catalog"), base_root=Path("base"),
            feature_root=Path("features"), label_root=Path("labels"),
            generated_time=datetime(2024, 1, 10, tzinfo=UTC), windows=windows,
            certification=DatasetCertification.PROVISIONAL, raw_bar_dataset_id="raw",
            outcome_open_bar_dataset_id="projection",
        )


def session_dt(session: date) -> datetime:
    return datetime.combine(session, datetime.min.time(), tzinfo=UTC)


def calendar(sessions: list[date] | None = None) -> KRXSessionCalendar:
    return KRXSessionCalendar(
        version="fixture-calendar",
        sessions=tuple(sessions or SESSIONS),
        generated_time=GENERATED,
    )


def base_panel_frame(
    sessions: list[date] | None = None,
    tickers: tuple[str, ...] = TICKERS,
    opens: dict[tuple[str, date], float] | None = None,
    include_features: bool = True,
) -> pl.DataFrame:
    sessions = sessions or SESSIONS
    rows: list[dict[str, object]] = []
    for t, ticker in enumerate(tickers):
        for s, session in enumerate(sessions):
            open_price = (
                opens.get((ticker, session), 100.0 + float(s))
                if opens is not None
                else 100.0 + float(s) + float(t % 7)
            )
            row: dict[str, object] = {
                "instrument_id": ticker,
                "session": session_dt(session),
                "observation_time": datetime.combine(
                    session, datetime.min.time(), tzinfo=UTC
                ),
                "available_time": datetime.combine(
                    session, datetime.min.time(), tzinfo=UTC
                ),
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
            if include_features:
                for j, name in enumerate(ALLOWLIST):
                    row[f"raw__{name}"] = float((t * 31 + s * 7 + j) % 50) / 10.0
            rows.append(row)
    return pl.DataFrame(rows)


def write_base_panel(
    root: Path,
    dataset_id: str = "base_v1",
    sessions: list[date] | None = None,
    tickers: tuple[str, ...] = TICKERS,
    opens: dict[tuple[str, date], float] | None = None,
    include_features: bool = True,
) -> Path:
    sessions = sessions or SESSIONS
    frame = base_panel_frame(sessions, tickers, opens, include_features)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="base_panel",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=session_dt(sessions[0]),
        time_end=session_dt(sessions[-1]),
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
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set="base_panel",
        decision_time=GENERATED,
        content_manifest={"fixture": True},
    )


def v2_feature_request(dataset_id: str = "features_v2") -> FeaturePanelRequest:
    return FeaturePanelRequest(
        dataset_id=dataset_id,
        base_panel_id="base_v1",
        feature_set=STOCK_ALPHA_V2_FEATURE_SET,
        generated_time=GENERATED,
    )


WINDOWS = ResearchWindows(
    train=CoverageRange(SESSIONS[2], SESSIONS[7]),
    validation=CoverageRange(SESSIONS[8], SESSIONS[10]),
    test=CoverageRange(SESSIONS[11], SESSIONS[13]),
)


def write_calendar_evidence(root: Path) -> Path:
    path = root / "calendar.json"
    path.write_text(
        json.dumps(
            {
                "version": "fixture-calendar",
                "sessions": [d.isoformat() for d in SESSIONS],
                "generated_time": GENERATED.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


def setup_source_snapshot(
    catalog_root: Path,
    base_root: Path,
    *,
    base_id: str = "base_v1",
    calendar_path: Path | None = None,
    include_calendar: bool = True,
    windows: ResearchWindows = WINDOWS,
) -> CatalogStore:
    store = CatalogStore(catalog_root)
    write_base_panel(base_root, base_id)
    base_entry = CatalogEntry(
        kind=CatalogKind.BASE_PANEL,
        name=base_id,
        content_hash=ParquetDatasetStore(base_root).read_manifest(base_id).content_hash,
        schema_hash="schema",
        registered_at=GENERATED,
        coverage=CoverageRange(SESSIONS[0], SESSIONS[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(base_root / base_id),
    )
    store.register(base_entry)
    references: list[CatalogEntry] = [base_entry]
    if include_calendar:
        if calendar_path is None:
            calendar_path = write_calendar_evidence(catalog_root)
        calendar_entry = register_file_evidence(
            store,
            kind=CatalogKind.CALENDAR,
            name="calendar_v1",
            path=calendar_path,
            coverage=CoverageRange(SESSIONS[0], SESSIONS[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            registered_at=GENERATED,
        )
        references.append(calendar_entry)
    manifest = build_snapshot_manifest(
        snapshot_id="source_snap_v1",
        certification=DatasetCertification.PROVISIONAL,
        timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
        windows=windows,
        references=tuple(references),
    )
    manifest_path = (
        catalog_root / "snapshots" / "source_snap_v1" / "snapshot_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_json(), sort_keys=True, indent=2), encoding="utf-8"
    )
    return store


class TestStockAlphaV2FeaturePanel:
    def test_projects_exactly_34_ordered_columns(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        write_base_panel(base_root)
        result = build_stock_alpha_v2_feature_panel(
            base_root, feature_root, v2_feature_request()
        )
        frame = ParquetDatasetStore(feature_root).read(
            result.dataset_id, AssetKind.STOCK, STOCK_ALPHA_V2_FEATURE_SET, GENERATED
        )
        assert frame.columns == [
            "instrument_id",
            "session",
            *(f"feature__{name}" for name in ALLOWLIST),
        ]
        assert result.manifest.feature_set == STOCK_ALPHA_V2_FEATURE_SET
        readiness = json.loads(
            (result.contract_path.parent / V2_READINESS_NAME).read_text()
        )
        assert readiness["allowlist_hash"]
        assert readiness["base_panel_id"] == "base_v1"
        assert readiness["min_coverage"] == 0.75
        assert len(readiness["features"]) == len(ALLOWLIST)

    def test_rejects_missing_source_column(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        missing = base_panel_frame().drop(f"raw__{ALLOWLIST[0]}")
        _write_raw_frame(base_root, missing)
        with pytest.raises(ValueError, match="missing v2 raw source columns"):
            build_stock_alpha_v2_feature_panel(
                base_root, feature_root, v2_feature_request()
            )

    def test_rejects_all_null_feature(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        frame = base_panel_frame().with_columns(
            pl.lit(None, dtype=pl.Float64).alias(f"raw__{ALLOWLIST[0]}")
        )
        _write_raw_frame(base_root, frame)
        with pytest.raises(ValueError, match="fully null"):
            build_stock_alpha_v2_feature_panel(
                base_root, feature_root, v2_feature_request()
            )

    def test_rejects_low_coverage(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        frame = base_panel_frame().with_row_index("_row")
        frame = frame.with_columns(
            pl.when((pl.col("_row") % 2) == 0)
            .then(None)
            .otherwise(pl.col(f"raw__{ALLOWLIST[0]}"))
            .alias(f"raw__{ALLOWLIST[0]}")
        ).drop("_row")
        _write_raw_frame(base_root, frame)
        with pytest.raises(ValueError, match="below"):
            build_stock_alpha_v2_feature_panel(
                base_root, feature_root, v2_feature_request()
            )

    def test_rejects_nan_feature(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        frame = base_panel_frame().with_columns(
            pl.when(pl.col("session").first().is_not_null())
            .then(float("nan"))
            .otherwise(pl.col(f"raw__{ALLOWLIST[0]}"))
            .alias(f"raw__{ALLOWLIST[0]}")
        )
        _write_raw_frame(base_root, frame)
        with pytest.raises(ValueError, match="NaN/Infinity"):
            build_stock_alpha_v2_feature_panel(
                base_root, feature_root, v2_feature_request()
            )

    def test_rejects_target_source(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        frame = base_panel_frame().with_columns(
            pl.lit(0.05).alias("raw__target_return_5d")
        )
        _write_raw_frame(base_root, frame)
        with pytest.raises(ValueError, match=r"target|label"):
            build_stock_alpha_v2_feature_panel(
                base_root, feature_root, v2_feature_request()
            )


class TestResidualLabels:
    def test_calendar_gap_suspends_label(self) -> None:
        sessions = weekdays(15)
        cal = calendar(sessions)
        frame = base_panel_frame(
            sessions, opens=None, include_features=False
        )
        # Remove instrument KRX:000001 at a mid session -> it is T+1 for the
        # previous decision and T+6 for the decision six sessions earlier.
        gap = sessions[8]
        frame = frame.filter(
            ~((pl.col("instrument_id") == "KRX:000001") & (pl.col("session") == session_dt(gap)))
        )
        out = build_residual_o2o_label_dataset(frame, cal, horizon_sessions=5)
        gap_rows = out.filter(
            (pl.col("instrument_id") == "KRX:000001")
            & (pl.col("session") == session_dt(sessions[2]))
        )
        assert gap_rows.is_empty()
        gap2_rows = out.filter(
            (pl.col("instrument_id") == "KRX:000001")
            & (pl.col("session") == session_dt(sessions[7]))
        )
        assert gap2_rows.is_empty()

    def test_labels_require_20_names_and_relevance_range(self) -> None:
        sessions = weekdays(15)
        cal = calendar(sessions)
        frame = base_panel_frame(sessions, include_features=False)
        out = build_residual_o2o_label_dataset(frame, cal, horizon_sessions=5)
        assert out.columns == [
            "instrument_id",
            "session",
            "residual_o2o_5d",
            "relevance",
            LABEL_AVAILABLE_COLUMN,
        ]
        assert out.height > 0
        relevance = out["relevance"]
        assert relevance.null_count() == 0
        assert min(int(v) for v in relevance.to_list()) >= 0
        assert max(int(v) for v in relevance.to_list()) <= 4

    def test_label_availability_no_earlier_than_exit_open(self) -> None:
        sessions = weekdays(15)
        cal = calendar(sessions)
        frame = base_panel_frame(sessions, include_features=False)
        out = build_residual_o2o_label_dataset(frame, cal, horizon_sessions=5)
        by_pos = {session: p for p, session in enumerate(sessions)}
        for row in out.iter_rows(named=True):
            decision = row["session"].date()
            p = by_pos[decision]
            exit_session = sessions[p + 1 + 5]
            exit_open = datetime.combine(exit_session, datetime.min.time(), tzinfo=UTC)
            assert row[LABEL_AVAILABLE_COLUMN] >= exit_open

    def test_too_few_names_drops_session(self) -> None:
        sessions = weekdays(15)
        cal = calendar(sessions)
        frame = base_panel_frame(
            sessions, tickers=TICKERS[:15], include_features=False
        )
        out = build_residual_o2o_label_dataset(frame, cal, horizon_sessions=5)
        assert out.is_empty()


class TestMaterializeSnapshot:
    def test_materializes_and_composes(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = write_calendar_evidence(tmp_path)
        setup_source_snapshot(catalog_root, base_root, calendar_path=calendar_path)

        result = materialize_stock_alpha_v2_snapshot(
            StockAlphaV2MaterializationRequest(
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
        )
        assert result.snapshot_id == "snap_v2"
        assert result.feature_content_hash
        assert result.label_content_hash
        assert result.min_coverage == 0.75

        store = CatalogStore(catalog_root)
        assert store.get(CatalogKind.FEATURES, "features_v2") is not None
        assert store.get(CatalogKind.LABELS, "labels_v2") is not None
        assert store.get(CatalogKind.SNAPSHOT, "snap_v2") is None

        repository = ResearchDataRepository(
            base_root=base_root,
            feature_root=feature_root,
            label_root=label_root,
        )
        from src.stocks.data.catalog import SnapshotResolver

        snapshot = SnapshotResolver(store).resolve("snap_v2")
        composed = repository.compose_labeled_training_snapshot(
            snapshot,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            decision_time=GENERATED,
        )
        assert composed.frame.height > 0
        assert {
            "residual_o2o_5d",
            "relevance",
            LABEL_AVAILABLE_COLUMN,
        }.issubset(composed.frame.columns)

    def test_materializes_and_composes_multi_horizon_v3(self, tmp_path) -> None:
        from src.stocks.data.labels import (
            MULTI_HORIZON_RESIDUAL_DEFINITION,
        )
        from src.stocks.data.research_v2 import (
            materialize_stock_alpha_v3_snapshot,
        )

        sessions = weekdays(40)
        windows = ResearchWindows(
            train=CoverageRange(sessions[2], sessions[15]),
            validation=CoverageRange(sessions[16], sessions[18]),
            test=CoverageRange(sessions[19], sessions[21]),
        )
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = tmp_path / "calendar.json"
        calendar_path.write_text(
            json.dumps(
                {
                    "version": "fixture-calendar",
                    "sessions": [d.isoformat() for d in sessions],
                    "generated_time": GENERATED.isoformat(),
                }
            ),
            encoding="utf-8",
        )
        write_base_panel(base_root, "base_v1", sessions=sessions)
        store = CatalogStore(catalog_root)
        base_entry = CatalogEntry(
            kind=CatalogKind.BASE_PANEL,
            name="base_v1",
            content_hash=ParquetDatasetStore(base_root)
            .read_manifest("base_v1")
            .content_hash,
            schema_hash="schema",
            registered_at=GENERATED,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(base_root / "base_v1"),
        )
        store.register(base_entry)
        calendar_entry = register_file_evidence(
            store,
            kind=CatalogKind.CALENDAR,
            name="calendar_v1",
            path=calendar_path,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            registered_at=GENERATED,
        )
        manifest = build_snapshot_manifest(
            snapshot_id="source_snap_v3",
            certification=DatasetCertification.PROVISIONAL,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=windows,
            references=(base_entry, calendar_entry),
        )
        manifest_path = (
            catalog_root / "snapshots" / "source_snap_v3" / "snapshot_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest.to_json(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

        result = materialize_stock_alpha_v3_snapshot(
            StockAlphaV2MaterializationRequest(
                source_snapshot_id="source_snap_v3",
                feature_dataset_id="features_v3",
                label_dataset_id="labels_v3",
                snapshot_id="snap_v3",
                catalog_root=catalog_root,
                base_root=base_root,
                feature_root=feature_root,
                label_root=label_root,
                generated_time=GENERATED,
                windows=windows,
                certification=DatasetCertification.PROVISIONAL,
                calendar_path=calendar_path,
            )
        )
        assert result.snapshot_id == "snap_v3"
        assert result.feature_content_hash
        assert result.label_content_hash

        store = CatalogStore(catalog_root)
        assert store.get(CatalogKind.FEATURES, "features_v3") is not None
        assert store.get(CatalogKind.LABELS, "labels_v3") is not None

        from src.stocks.data.catalog import SnapshotResolver

        repository = ResearchDataRepository(
            base_root=base_root,
            feature_root=feature_root,
            label_root=label_root,
        )
        snapshot = SnapshotResolver(store).resolve("snap_v3")
        composed = repository.compose_labeled_training_snapshot(
            snapshot,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            decision_time=GENERATED,
        )
        assert composed.frame.height > 0
        expected_columns = ["instrument_id", "session"]
        for h in (5, 10, 15):
            expected_columns += [
                f"residual_o2o_{h}d",
                f"relevance_{h}d",
                f"label_available_time_{h}d",
            ]
        assert all(c in composed.frame.columns for c in expected_columns)
        assert composed.manifest.label_definition == MULTI_HORIZON_RESIDUAL_DEFINITION
        assert composed.manifest.label_horizon_sessions == 5

    def test_net_alpha_materialization_registers_status_artifact(self, tmp_path) -> None:
        from src.stocks.data.catalog import SnapshotResolver
        from src.stocks.data.research_v2 import (
            NetAlphaMaterializationRequest,
            materialize_net_alpha_snapshot,
        )

        sessions = weekdays(60)
        windows = ResearchWindows(
            train=CoverageRange(sessions[2], sessions[15]),
            validation=CoverageRange(sessions[16], sessions[20]),
            test=CoverageRange(sessions[21], sessions[25]),
        )
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = tmp_path / "calendar.json"
        calendar_path.write_text(
            json.dumps(
                {
                    "version": "fixture-calendar",
                    "sessions": [d.isoformat() for d in sessions],
                    "generated_time": GENERATED.isoformat(),
                }
            ),
            encoding="utf-8",
        )
        rows: list[dict[str, object]] = []
        for t, ticker in enumerate(tuple(f"KRX:{i:06d}" for i in range(1, 61))):
            for s, session in enumerate(sessions):
                session_dt_value = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
                open_price = 100.0 + float(s) + float(t % 7)
                row: dict[str, object] = {
                    "instrument_id": ticker,
                    "session": session_dt_value,
                    "observation_time": datetime.combine(session, time(15, 30), tzinfo=UTC),
                    "available_time": datetime.combine(session, time(15, 31), tzinfo=UTC),
                    "open": open_price,
                    "high": open_price + 1.0,
                    "low": max(1.0, open_price - 1.0),
                    "close": open_price + 0.5,
                    "volume": 1_000_000.0,
                    "trading_value": open_price * 1_000_000.0,
                    "market_cap": 1e12 + float(t) * 1e9,
                    "sector": f"S{t % 4}",
                    "action_interval_covered": True,
                    "data_quality_status": "eligible",
                    "data_quality_reason": None,
                    "raw__adtv_20d": 1.0e8 + float(s) * 1.0e6,
                    "raw__volatility_20d": 0.02 + float(t % 5) * 0.005,
                }
                for j, name in enumerate(stock_alpha_v2_allowlist()):
                    if name not in ("adtv_20d", "volatility_20d"):
                        row[f"raw__{name}"] = float((t * 31 + s * 7 + j) % 50) / 10.0
                rows.append(row)
        base_frame = pl.DataFrame(rows)
        base_manifest = make_manifest(
            asset_kind=AssetKind.STOCK,
            columns=base_frame.columns,
            feature_set="base_panel",
            label_definition="none",
            label_horizon_sessions=1,
            time_start=session_dt(sessions[0]),
            time_end=session_dt(sessions[-1]),
            provider_version="fixture",
            universe_policy_version="fixture",
            row_count=base_frame.height,
            generated_time=GENERATED,
            schema_version="v2",
            content_hash=canonical_content_hash(base_frame, base_frame.columns),
            storage_layout=HIVE_PARTITION_LAYOUT,
        )
        ParquetDatasetStore(base_root).write_partitioned(
            base_frame,
            dataset_id="base_v1",
            manifest=base_manifest,
            expected_feature_set="base_panel",
            decision_time=GENERATED,
            content_manifest={"fixture": True},
        )
        store = CatalogStore(catalog_root)
        base_entry = CatalogEntry(
            kind=CatalogKind.BASE_PANEL,
            name="base_v1",
            content_hash=ParquetDatasetStore(base_root)
            .read_manifest("base_v1")
            .content_hash,
            schema_hash="schema",
            registered_at=GENERATED,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(base_root / "base_v1"),
        )
        store.register(base_entry)
        calendar_entry = register_file_evidence(
            store,
            kind=CatalogKind.CALENDAR,
            name="calendar_v1",
            path=calendar_path,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            registered_at=GENERATED,
        )
        cost_path = tmp_path / "costs.json"
        cost_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "coverage": {"start": sessions[0].isoformat(), "end": sessions[-1].isoformat()},
                    "assumption_id": "test_kis_v1",
                    "sources": [
                        {"uri": "https://law.go.kr/fixture", "retrieved_at": GENERATED.isoformat(), "content_hash": "h" * 64}
                    ],
                    "commission": [
                        {"effective_from": sessions[0].isoformat(), "buy_rate": 0.000036396, "sell_rate": 0.000036396}
                    ],
                    "sell_taxes": [
                        {"effective_from": sessions[0].isoformat(), "market": "KOSPI", "securities_transaction_tax_rate": 0.0003, "rural_special_tax_rate": 0.0015, "sell_tax_rate": 0.0018, "source_uri": "https://law.go.kr/fixture", "source_hash": "h" * 64},
                        {"effective_from": sessions[0].isoformat(), "market": "KOSDAQ", "securities_transaction_tax_rate": 0.0018, "rural_special_tax_rate": 0.0, "sell_tax_rate": 0.0018, "source_uri": "https://law.go.kr/fixture", "source_hash": "h" * 64},
                    ],
                    "tick_size_rules": [
                        {"rule_id": f"krx_{i}", "effective_from": sessions[0].isoformat(), "lower_inclusive": lo, "upper_exclusive": hi, "tick": tick}
                        for i, (lo, hi, tick) in enumerate(
                            ((0.0, 1000.0, 1.0), (1000.0, 5000.0, 5.0), (5000.0, 10000.0, 10.0), (10000.0, 50000.0, 50.0), (50000.0, 100000.0, 100.0), (100000.0, 500000.0, 500.0), (500000.0, None, 1000.0))
                        )
                    ],
                    "liquidity_model": {"model_id": "sqrt_impact_v1", "impact_coefficient": 0.1, "stress_multiplier": 1.5},
                    "settlement_days": 2,
                }
            ),
            encoding="utf-8",
        )
        cost_entry = register_file_evidence(
            store,
            kind=CatalogKind.COSTS,
            name="costs_v1",
            path=cost_path,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            registered_at=GENERATED,
        )
        master_path = tmp_path / "master.json"
        master_path.write_text(json.dumps({"version": "fixture"}), encoding="utf-8")
        master_entry = register_file_evidence(
            store,
            kind=CatalogKind.INSTRUMENT_MASTER,
            name="master_v1",
            path=master_path,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            registered_at=GENERATED,
        )
        action_path = tmp_path / "actions.json"
        action_path.write_text(json.dumps({"version": "fixture"}), encoding="utf-8")
        action_entry = register_file_evidence(
            store,
            kind=CatalogKind.CORPORATE_ACTIONS,
            name="actions_v1",
            path=action_path,
            coverage=CoverageRange(sessions[0], sessions[-1]),
            completeness=EvidenceCompleteness.COMPLETE,
            registered_at=GENERATED,
        )
        manifest = build_snapshot_manifest(
            snapshot_id="source_snap_na",
            certification=DatasetCertification.PROVISIONAL,
            timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
            windows=windows,
            references=(
                base_entry,
                calendar_entry,
                cost_entry,
                master_entry,
                action_entry,
            ),
        )
        manifest_path = (
            catalog_root / "snapshots" / "source_snap_na" / "snapshot_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest.to_json(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

        result = materialize_net_alpha_snapshot(
            NetAlphaMaterializationRequest(
                source_snapshot_id="source_snap_na",
                feature_dataset_id="features_na",
                label_dataset_id="labels_na",
                snapshot_id="snap_na",
                catalog_root=catalog_root,
                base_root=base_root,
                feature_root=feature_root,
                label_root=label_root,
                generated_time=GENERATED,
                windows=windows,
                certification=DatasetCertification.PROVISIONAL,
                calendar_path=calendar_path,
                candidate_horizon_sessions=(3, 5),
                reference_notional=1.0e6,
            )
        )
        assert result.snapshot_id == "snap_na"

        status_entry = store.get(CatalogKind.OUTCOME_STATUS, "labels_na_outcome_status")
        assert status_entry is not None
        assert status_entry.content_hash
        assert status_entry.schema_hash
        assert status_entry.row_count > 0
        status_refs = dict(status_entry.references)
        assert status_refs[CatalogKind.BASE_PANEL.value] == "base_v1"
        assert status_refs[CatalogKind.CALENDAR.value] == "calendar_v1"
        assert status_refs[CatalogKind.LABELS.value] == "labels_na"
        assert status_refs[CatalogKind.INSTRUMENT_MASTER.value] == "master_v1"
        assert status_refs[CatalogKind.CORPORATE_ACTIONS.value] == "actions_v1"
        assert status_refs[CatalogKind.COSTS.value] == "costs_v1"

        resolved = SnapshotResolver(store).resolve("snap_na")
        assert resolved.outcome_status is not None
        assert resolved.outcome_status.name == "labels_na_outcome_status"
        assert resolved.status_provenance == "pinned"
        assert any(
            entry.kind is CatalogKind.OUTCOME_EVIDENCE
            for entry in resolved.manifest.references
        ), "SCENARIO_SNAPSHOT_PINS_OUTCOME_EVIDENCE"

        repository = ResearchDataRepository(
            base_root=base_root,
            feature_root=feature_root,
            label_root=label_root,
        )
        composed = repository.compose_labeled_training_snapshot(
            resolved,
            feature_set="stock_net_alpha_v1",
            decision_time=GENERATED,
        )
        assert composed.frame.height > 0
        assert "outcome_status" in composed.frame.columns

    def test_existing_id_creates_no_manifest_or_catalog_append(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = write_calendar_evidence(tmp_path)
        store = setup_source_snapshot(
            catalog_root, base_root, calendar_path=calendar_path
        )
        store.register(
            CatalogEntry(
                kind=CatalogKind.FEATURES,
                name="features_v2",
                content_hash="existing",
                schema_hash="schema",
                registered_at=GENERATED,
            )
        )
        with pytest.raises(ValueError, match="already has FEATURES"):
            materialize_stock_alpha_v2_snapshot(
                StockAlphaV2MaterializationRequest(
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
                )
            )
        assert not (catalog_root / "snapshots" / "snap_v2").exists()

    def test_incompatible_windows_rejected(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = write_calendar_evidence(tmp_path)
        setup_source_snapshot(catalog_root, base_root, calendar_path=calendar_path)
        wide = ResearchWindows(
            train=CoverageRange(date(2020, 1, 1), date(2020, 12, 31)),
            validation=CoverageRange(date(2021, 1, 1), date(2021, 6, 30)),
            test=CoverageRange(date(2021, 7, 1), date(2021, 12, 31)),
        )
        with pytest.raises(ValueError, match="coverage"):
            materialize_stock_alpha_v2_snapshot(
                StockAlphaV2MaterializationRequest(
                    source_snapshot_id="source_snap_v1",
                    feature_dataset_id="features_v2",
                    label_dataset_id="labels_v2",
                    snapshot_id="snap_v2",
                    catalog_root=catalog_root,
                    base_root=base_root,
                    feature_root=feature_root,
                    label_root=label_root,
                    generated_time=GENERATED,
                    windows=wide,
                )
            )

    def test_missing_calendar_rejected(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        setup_source_snapshot(catalog_root, base_root, include_calendar=False)
        with pytest.raises((ValueError, FileNotFoundError)):
            materialize_stock_alpha_v2_snapshot(
                StockAlphaV2MaterializationRequest(
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
                )
            )
        assert not (catalog_root / "snapshots" / "snap_v2").exists()
        store = CatalogStore(catalog_root)
        assert store.get(CatalogKind.FEATURES, "features_v2") is None
        assert store.get(CatalogKind.LABELS, "labels_v2") is None

    def test_failed_reread_creates_no_manifest_or_catalog_append(
        self, tmp_path, monkeypatch
    ) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = write_calendar_evidence(tmp_path)
        setup_source_snapshot(catalog_root, base_root, calendar_path=calendar_path)

        calls = {"count": 0}
        original = ParquetDatasetStore.read_bounded

        def flaky(self, dataset_id, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("simulated re-read failure")
            return original(self, dataset_id, *args, **kwargs)

        monkeypatch.setattr(ParquetDatasetStore, "read_bounded", flaky)
        with pytest.raises(ValueError, match="simulated re-read failure"):
            materialize_stock_alpha_v2_snapshot(
                StockAlphaV2MaterializationRequest(
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
                    calendar_path=calendar_path,
                )
            )
        store = CatalogStore(catalog_root)
        assert store.get(CatalogKind.FEATURES, "features_v2") is None
        assert store.get(CatalogKind.LABELS, "labels_v2") is None
        assert not (catalog_root / "snapshots" / "snap_v2").exists()

    def test_tampered_partition_prevents_registration(self, tmp_path) -> None:
        base_root = tmp_path / "base"
        feature_root = tmp_path / "features"
        label_root = tmp_path / "labels"
        catalog_root = tmp_path / "catalog"
        calendar_path = write_calendar_evidence(tmp_path)
        setup_source_snapshot(catalog_root, base_root, calendar_path=calendar_path)

        base_dir = base_root / "base_v1"
        partition = next((base_dir / "partitions").rglob("*.parquet"))
        partition.write_bytes(partition.read_bytes() + b"tampered")

        with pytest.raises(ValueError, match="tampered partition"):
            materialize_stock_alpha_v2_snapshot(
                StockAlphaV2MaterializationRequest(
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
                    calendar_path=calendar_path,
                )
            )
        store = CatalogStore(catalog_root)
        assert store.get(CatalogKind.FEATURES, "features_v2") is None
        assert store.get(CatalogKind.LABELS, "labels_v2") is None
        assert not (catalog_root / "snapshots" / "snap_v2").exists()


def _write_raw_frame(root: Path, frame: pl.DataFrame, dataset_id: str = "base_v1") -> None:
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="base_panel",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=frame["session"].min(),
        time_end=frame["session"].max(),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        generated_time=GENERATED,
        schema_version="v2",
        content_hash=canonical_content_hash(frame, frame.columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    ParquetDatasetStore(root).write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set="base_panel",
        decision_time=GENERATED,
        content_manifest={"fixture": True},
    )


def test_net_alpha_materialization_rejects_non_ready_research_snapshot(tmp_path) -> None:
    """A RESEARCH-certified snapshot with a historical unresolved outcome fails."""
    from src.stocks.data.catalog import SnapshotResolver
    from src.stocks.data.research_v2 import (
        NetAlphaMaterializationRequest,
        materialize_net_alpha_snapshot,
    )

    sessions = weekdays(30)
    windows = ResearchWindows(
        train=CoverageRange(sessions[2], sessions[15]),
        validation=CoverageRange(sessions[16], sessions[20]),
        test=CoverageRange(sessions[21], sessions[25]),
    )
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    catalog_root = tmp_path / "catalog"
    calendar_path = tmp_path / "calendar.json"
    calendar_path.write_text(
        json.dumps(
            {
                "version": "fixture-calendar",
                "sessions": [d.isoformat() for d in sessions],
                "generated_time": GENERATED.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    tickers = tuple(f"KRX:{i:06d}" for i in range(1, 7))
    rows: list[dict[str, object]] = []
    for t, ticker in enumerate(tickers):
        for s, session in enumerate(sessions):
            session_dt_value = session_dt(session)
            open_price = 100.0 + float(s) + float(t % 7)
            row: dict[str, object] = {
                "instrument_id": ticker,
                "session": session_dt_value,
                "observation_time": datetime.combine(session, time(15, 30), tzinfo=UTC),
                "available_time": datetime.combine(session, time(15, 31), tzinfo=UTC),
                "open": open_price,
                "high": open_price + 1.0,
                "low": max(1.0, open_price - 1.0),
                "close": open_price + 0.5,
                "volume": 1_000_000.0,
                "trading_value": open_price * 1_000_000.0,
                "market_cap": 1e12 + float(t) * 1e9,
                "sector": f"S{t % 4}",
                "action_interval_covered": True,
                "data_quality_status": "eligible",
                "data_quality_reason": None,
                "raw__adtv_20d": 1.0e8 + float(s) * 1.0e6,
                "raw__volatility_20d": 0.02 + float(t % 5) * 0.005,
            }
            for j, name in enumerate(stock_alpha_v2_allowlist()):
                if name not in ("adtv_20d", "volatility_20d"):
                    row[f"raw__{name}"] = float((t * 31 + s * 7 + j) % 50) / 10.0
            rows.append(row)
    rows[0]["open"] = None
    base_frame = pl.DataFrame(rows)
    base_manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=base_frame.columns,
        feature_set="base_panel",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=session_dt(sessions[0]),
        time_end=session_dt(sessions[-1]),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=base_frame.height,
        generated_time=GENERATED,
        schema_version="v2",
        content_hash=canonical_content_hash(base_frame, base_frame.columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
        calendar_hash="calendar-hash",
        cost_source_hash="cost-hash",
        master_hash="master-hash",
        quality_report_hash="quality-hash",
    )
    ParquetDatasetStore(base_root).write_partitioned(
        base_frame,
        dataset_id="base_na_r",
        manifest=base_manifest,
        expected_feature_set="base_panel",
        decision_time=GENERATED,
        content_manifest={"fixture": True},
    )
    store = CatalogStore(catalog_root)
    base_entry = CatalogEntry(
        kind=CatalogKind.BASE_PANEL,
        name="base_na_r",
        content_hash=ParquetDatasetStore(base_root)
        .read_manifest("base_na_r")
        .content_hash,
        schema_hash="schema",
        registered_at=GENERATED,
        coverage=CoverageRange(sessions[0], sessions[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(base_root / "base_na_r"),
    )
    store.register(base_entry)
    calendar_entry = register_file_evidence(
        store,
        kind=CatalogKind.CALENDAR,
        name="calendar_na_r",
        path=calendar_path,
        coverage=CoverageRange(sessions[0], sessions[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        registered_at=GENERATED,
    )
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coverage": {"start": sessions[0].isoformat(), "end": sessions[-1].isoformat()},
                "assumption_id": "test_kis_v1",
                "sources": [
                    {"uri": "https://law.go.kr/fixture", "retrieved_at": GENERATED.isoformat(), "content_hash": "h" * 64}
                ],
                "commission": [
                    {"effective_from": sessions[0].isoformat(), "buy_rate": 0.000036396, "sell_rate": 0.000036396}
                ],
                "sell_taxes": [
                    {"effective_from": sessions[0].isoformat(), "market": "KOSPI", "securities_transaction_tax_rate": 0.0003, "rural_special_tax_rate": 0.0015, "sell_tax_rate": 0.0018, "source_uri": "https://law.go.kr/fixture", "source_hash": "h" * 64},
                    {"effective_from": sessions[0].isoformat(), "market": "KOSDAQ", "securities_transaction_tax_rate": 0.0018, "rural_special_tax_rate": 0.0, "sell_tax_rate": 0.0018, "source_uri": "https://law.go.kr/fixture", "source_hash": "h" * 64},
                ],
                "tick_size_rules": [
                    {"rule_id": f"krx_{i}", "effective_from": sessions[0].isoformat(), "lower_inclusive": lo, "upper_exclusive": hi, "tick": tick}
                    for i, (lo, hi, tick) in enumerate(
                        ((0.0, 1000.0, 1.0), (1000.0, 5000.0, 5.0), (5000.0, 10000.0, 10.0), (10000.0, 50000.0, 50.0), (50000.0, 100000.0, 100.0), (100000.0, 500000.0, 500.0), (500000.0, None, 1000.0))
                    )
                ],
                "liquidity_model": {"model_id": "sqrt_impact_v1", "impact_coefficient": 0.1, "stress_multiplier": 1.5},
                "settlement_days": 2,
            }
        ),
        encoding="utf-8",
    )
    cost_entry = register_file_evidence(
        store,
        kind=CatalogKind.COSTS,
        name="costs_na_r",
        path=cost_path,
        coverage=CoverageRange(sessions[0], sessions[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        registered_at=GENERATED,
    )
    master_path = tmp_path / "master.json"
    master_path.write_text(json.dumps({"version": "fixture"}), encoding="utf-8")
    master_entry = register_file_evidence(
        store,
        kind=CatalogKind.INSTRUMENT_MASTER,
        name="master_na_r",
        path=master_path,
        coverage=CoverageRange(sessions[0], sessions[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        registered_at=GENERATED,
    )
    action_path = tmp_path / "actions.json"
    action_path.write_text(json.dumps({"version": "fixture"}), encoding="utf-8")
    action_entry = register_file_evidence(
        store,
        kind=CatalogKind.CORPORATE_ACTIONS,
        name="actions_na_r",
        path=action_path,
        coverage=CoverageRange(sessions[0], sessions[-1]),
        completeness=EvidenceCompleteness.COMPLETE,
        registered_at=GENERATED,
    )
    manifest = build_snapshot_manifest(
        snapshot_id="source_snap_na_r",
        certification=DatasetCertification.PROVISIONAL,
        timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
        windows=windows,
        references=(base_entry, calendar_entry, cost_entry, master_entry, action_entry),
    )
    manifest_path = (
        catalog_root / "snapshots" / "source_snap_na_r" / "snapshot_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_json(), sort_keys=True, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outcome readiness failed"):
        materialize_net_alpha_snapshot(
            NetAlphaMaterializationRequest(
                source_snapshot_id="source_snap_na_r",
                feature_dataset_id="features_na_r",
                label_dataset_id="labels_na_r",
                snapshot_id="snap_na_r",
                catalog_root=catalog_root,
                base_root=base_root,
                feature_root=feature_root,
                label_root=label_root,
                generated_time=GENERATED,
                windows=windows,
                certification=DatasetCertification.RESEARCH,
                calendar_path=calendar_path,
                candidate_horizon_sessions=(3, 5),
                reference_notional=1.0e6,
            )
        )
    assert not (catalog_root / "snapshots" / "snap_na_r" / "snapshot_manifest.json").exists()
    resolved = SnapshotResolver(store).resolve("source_snap_na_r")
    assert resolved is not None
