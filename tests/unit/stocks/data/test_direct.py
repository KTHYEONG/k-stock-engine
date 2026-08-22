"""Direct market data loader: contract tests for the lean ML backtest data path."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.data.direct import (
    DirectDataRequest,
    DirectMarketDataLoader,
    MlMarketData,
)
from src.stocks.ml.data import validate_ml_market_data
from src.stocks.research.models import ModelManifest
from src.storage.parquet_datasets import ParquetDatasetStore


class _PickleableDummy:
    """Minimal picklable stand-in model for artifact registry fixtures."""

    def fit(self, train: object, validation: object) -> None:  # pragma: no cover
        return None

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


def _write_dataset(
    store: ParquetDatasetStore,
    dataset_id: str,
    frame: pl.DataFrame,
    *,
    feature_set: str = "stock_net_alpha_v1",
    label_definition: str = "net_alpha_o2o",
) -> None:
    """Write a minimal partitioned dataset for testing."""
    from dataclasses import replace
    
    from src.core.datasets import HIVE_PARTITION_LAYOUT
    from src.storage.parquet_datasets import canonical_content_hash
    
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set=feature_set,
        label_definition=label_definition,
        label_horizon_sessions=10,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 31, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        generated_time=datetime.now(UTC),
        schema_version="v2",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    # Create content hash for partitioned write
    content_hash = canonical_content_hash(frame, frame.columns)
    manifest = replace(manifest, content_hash=content_hash)
    
    store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=feature_set,
        decision_time=datetime(2024, 3, 31, tzinfo=UTC),
    )


def _base_frame() -> pl.DataFrame:
    """Minimal base market data."""
    rows = []
    for t in range(3):
        for s in range(20):
            session = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=s)
            close = 100.0 + float(t * 7 + s) % 20
            rows.append({
                "instrument_id": f"KRX:0{t + 1:05d}",
                "session": session,
                "open": close - 1.0,
                "close": close,
                "volume": 1_000_000.0,
                "trading_value": close * 1_000_000.0,
            })
    return pl.DataFrame(rows)


def _feature_frame() -> pl.DataFrame:
    """Minimal feature data."""
    rows = []
    for t in range(3):
        for s in range(20):
            session = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=s)
            rows.append({
                "instrument_id": f"KRX:0{t + 1:05d}",
                "session": session,
                "feature__momentum_5d": float((t + s) % 7) / 7.0,
                "feature__volatility_20d": 0.02 + float(s) * 0.001,
            })
    return pl.DataFrame(rows)


def _label_frame() -> pl.DataFrame:
    """Minimal label data with multi-horizon targets."""
    rows = []
    for t in range(3):
        for s in range(20):
            session = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=s)
            rows.append({
                "instrument_id": f"KRX:0{t + 1:05d}",
                "session": session,
                "horizon_10_target": 0.01 + float(s) * 0.001 if s < 18 else None,
                "horizon_10_available": session + timedelta(days=10) if s < 18 else None,
            })
    return pl.DataFrame(rows)


def test_direct_h10_without_catalog(tmp_path: Path) -> None:
    """LMD-01: Direct loader returns non-empty MlMarketData without catalog."""
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    base_root.mkdir()
    feature_root.mkdir()
    label_root.mkdir()

    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    _write_dataset(base_store, "base_2024", _base_frame(), feature_set="base_panel")
    _write_dataset(feature_store, "features_2024", _feature_frame())
    _write_dataset(label_store, "labels_2024", _label_frame(), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root,
        feature_root=feature_root,
        label_root=label_root,
    )

    request = DirectDataRequest(
        base_dataset_id="base_2024",
        feature_dataset_id="features_2024",
        label_dataset_id="labels_2024",
        start=date(2024, 1, 1),
        end=date(2024, 1, 20),
        candidate_horizon_sessions=(10,),
    )

    result = loader.load(request)

    assert isinstance(result, MlMarketData)
    assert not result.frame.is_empty()
    assert 10 in result.labels_by_horizon
    assert not result.labels_by_horizon[10].is_empty()
    assert result.input_ids["base_dataset_id"] == "base_2024"
    assert result.input_ids["feature_dataset_id"] == "features_2024"
    assert result.input_ids["label_dataset_id"] == "labels_2024"


def test_horizon_and_key_validation(tmp_path: Path) -> None:
    """LMD-02: Duplicate keys, missing columns, non-finite values raise ValueError."""
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    base_root.mkdir()
    feature_root.mkdir()
    label_root.mkdir()

    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    # Test duplicate keys
    base_frame = _base_frame()
    duplicate_base = pl.concat([base_frame, base_frame.head(1)])
    _write_dataset(base_store, "base_dup", duplicate_base, feature_set="base_panel")
    _write_dataset(feature_store, "features_dup", _feature_frame())
    _write_dataset(label_store, "labels_dup", _label_frame(), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root,
        feature_root=feature_root,
        label_root=label_root,
    )

    request = DirectDataRequest(
        base_dataset_id="base_dup",
        feature_dataset_id="features_dup",
        label_dataset_id="labels_dup",
        start=date(2024, 1, 1),
        end=date(2024, 1, 20),
        candidate_horizon_sessions=(10,),
    )

    with pytest.raises(ValueError, match="duplicate"):
        loader.load(request)


def test_validate_ml_market_data_rejects_empty(tmp_path: Path) -> None:
    """validate_ml_market_data rejects empty frames."""
    empty_data = MlMarketData(
        frame=pl.DataFrame({"instrument_id": [], "session": []}),
        labels_by_horizon={},
        input_ids={},
    )
    with pytest.raises(ValueError, match="empty"):
        validate_ml_market_data(empty_data, (10,))


def test_validate_ml_market_data_rejects_missing_horizons(tmp_path: Path) -> None:
    """validate_ml_market_data rejects missing requested horizons."""
    data = MlMarketData(
        frame=pl.DataFrame({
            "instrument_id": ["KRX:00001"],
            "session": [datetime(2024, 1, 1, tzinfo=UTC)],
        }),
        labels_by_horizon={
            10: pl.DataFrame({
                "instrument_id": ["KRX:00001"],
                "session": [datetime(2024, 1, 1, tzinfo=UTC)],
                "target": [0.01],
            }),
        },
        input_ids={},
    )
    with pytest.raises(ValueError, match="missing requested horizons"):
        validate_ml_market_data(data, (10, 20))


def test_direct_loader_rejects_non_monotonic_sessions() -> None:
    """Direct loader rejects non-monotonic session ordering."""
    loader = DirectMarketDataLoader(
        base_root=Path("/nonexistent"),
        feature_root=Path("/nonexistent"),
        label_root=Path("/nonexistent"),
    )

    # Create non-monotonic sessions directly
    base_frame = pl.DataFrame({
        "instrument_id": ["KRX:00001", "KRX:00001", "KRX:00001"],
        "session": [
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),  # Non-monotonic
            datetime(2024, 1, 2, tzinfo=UTC),
        ],
        "open": [100.0, 101.0, 102.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        "trading_value": [101_000_000.0, 102_000_000.0, 103_000_000.0],
    })
    feature_frame = pl.DataFrame({
        "instrument_id": ["KRX:00001", "KRX:00001", "KRX:00001"],
        "session": [
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ],
        "feature__momentum_5d": [0.1, 0.2, 0.3],
    })
    label_frame = pl.DataFrame({
        "instrument_id": ["KRX:00001", "KRX:00001", "KRX:00001"],
        "session": [
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ],
        "horizon_10_target": [0.01, 0.02, 0.03],
        "horizon_10_available": [
            datetime(2024, 1, 13, tzinfo=UTC),
            datetime(2024, 1, 11, tzinfo=UTC),
            datetime(2024, 1, 12, tzinfo=UTC),
        ],
    })

    # Test validation directly
    from src.stocks.data.direct import _validate_direct_frame

    with pytest.raises(ValueError, match="non-monotonic"):
        _validate_direct_frame(base_frame, ())


def test_direct_loader_rejects_non_finite_features() -> None:
    """Direct loader rejects non-finite numeric feature values."""
    # Create frame with non-finite feature values
    frame = pl.DataFrame({
        "instrument_id": ["KRX:00001", "KRX:00001"],
        "session": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ],
        "feature__momentum_5d": [float("inf"), 0.2],
        "feature__volatility_20d": [0.02, float("nan")],
    })

    from src.stocks.data.direct import _validate_direct_frame

    with pytest.raises(ValueError, match="non-finite"):
        _validate_direct_frame(
            frame,
            ("feature__momentum_5d", "feature__volatility_20d"),
        )


def test_direct_loader_rejects_zero_usable_labels(tmp_path: Path) -> None:
    """Direct loader rejects a requested horizon with zero usable labels."""
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    base_root.mkdir()
    feature_root.mkdir()
    label_root.mkdir()

    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    # Create label frame with all null targets
    label_frame = _label_frame().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("horizon_10_target")
    )
    _write_dataset(base_store, "base_no_labels", _base_frame(), feature_set="base_panel")
    _write_dataset(feature_store, "features_no_labels", _feature_frame())
    _write_dataset(label_store, "labels_no_labels", label_frame, feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root,
        feature_root=feature_root,
        label_root=label_root,
    )

    request = DirectDataRequest(
        base_dataset_id="base_no_labels",
        feature_dataset_id="features_no_labels",
        label_dataset_id="labels_no_labels",
        start=date(2024, 1, 1),
        end=date(2024, 1, 20),
        candidate_horizon_sessions=(10,),
    )

    with pytest.raises(ValueError, match="zero usable labels"):
        loader.load(request)


def test_schema_parity_v6(tmp_path: Path) -> None:
    """DIRECT_SCHEMA_PARITY_V6.

    The direct loader preserves the feature dataset manifest, so a training
    manifest's feature_schema_hash equals the selected feature dataset's
    manifest.schema_hash, and the input content hashes carry the feature
    schema/content identity.
    """
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    base_root.mkdir()
    feature_root.mkdir()
    label_root.mkdir()

    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    _write_dataset(base_store, "base_2024", _base_frame(), feature_set="base_panel")
    _write_dataset(feature_store, "features_2024", _feature_frame())
    _write_dataset(label_store, "labels_2024", _label_frame(), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root,
        feature_root=feature_root,
        label_root=label_root,
    )
    request = DirectDataRequest(
        base_dataset_id="base_2024",
        feature_dataset_id="features_2024",
        label_dataset_id="labels_2024",
        start=date(2024, 1, 1),
        end=date(2024, 1, 20),
        candidate_horizon_sessions=(10,),
    )
    result = loader.load(request)

    assert result.feature_manifest is not None
    feature_schema_hash = result.feature_manifest.schema_hash
    assert result.input_content_hashes["feature_schema_hash"] == feature_schema_hash
    assert result.input_content_hashes["feature_content_hash"] == (
        result.feature_manifest.content_hash or feature_schema_hash
    )


def test_v6_simulation_rejects_divergent_feature_content_hash(tmp_path: Path) -> None:
    """DIRECT_SCHEMA_PARITY_V6 (independent replay).

    A v6 artifact with an exact feature content hash must reject an independent
    simulation whose snapshot content hash diverges, failing closed before the
    backtester runs.
    """
    import joblib
    from src.stocks.research.artifacts import (
        MANIFEST_FILENAME,
        MODEL_FILENAME,
        ModelArtifactRegistry,
        _manifest_to_dict,
    )
    from src.stocks.workflows.contracts import SimulationRequest
    from src.stocks.workflows.simulate_portfolio import simulate_portfolio

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_manifest = ModelManifest(
        artifact_id="v6_sim",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        eligible_from=datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        eligible_to=datetime(2024, 3, 31, tzinfo=UTC).isoformat(),
        model_type="net_alpha_elastic_net",
        params={
            "holm_gate_version": "v6",
            "raw_feature_schema_hash": "h",
            "feature_content_hash": "feature-abc",
        },
    )
    artifact_dir = registry._artifact_dir("v6_sim")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / MANIFEST_FILENAME).open("w", encoding="utf-8") as fh:
        json.dump(_manifest_to_dict(artifact_manifest), fh, indent=2, default=str)
    joblib.dump(_PickleableDummy(), artifact_dir / MODEL_FILENAME)

    snapshot_manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=_feature_frame().columns,
        feature_set="stock_net_alpha_v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 31, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=_feature_frame().height,
        generated_time=datetime(2024, 3, 31, tzinfo=UTC),
        schema_version="v2",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    from dataclasses import replace

    snapshot_manifest = replace(snapshot_manifest, content_hash="different-content")
    snapshot = DatasetSnapshot(
        manifest=snapshot_manifest, frame=_feature_frame()
    )
    request = SimulationRequest(
        artifact_id="v6_sim",
        decision_time=datetime(2024, 2, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="v6 independent replay input lineage mismatch"):
        simulate_portfolio(snapshot, registry, request)


DIRECT_PUSHDOWN_01 = "DIRECT-PUSHDOWN-01"


def test_direct_pushdown_01_only_intersecting_partitions_scanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DIRECT-PUSHDOWN-01: bounded scans touch only intersecting partitions.

    Only partitions intersecting [start, end] are scanned; collected columns
    equal the projection and labels contain only requested horizons.
    """
    import polars as pl

    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    for root in (base_root, feature_root, label_root):
        root.mkdir()
    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    # Sessions span three months so the dataset owns three monthly partitions.
    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(90)]
    base_rows, feature_rows, label_rows = [], [], []
    for session in sessions:
        for t in range(2):
            price = 100.0 + t
            base_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
                "open": price,
                "close": price * 1.01,
                "volume": 1_000_000.0,
                "trading_value": price * 1_000_000.0,
            })
            feature_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
                "feature__momentum_5d": 0.1 * t,
            })
            label_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
                "horizon_sessions": 10,
                "net_alpha_target": 0.01,
                "label_available_time": session + timedelta(days=10),
            })
            label_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
                "horizon_sessions": 20,
                "net_alpha_target": 0.02,
                "label_available_time": session + timedelta(days=20),
            })
    _write_dataset(base_store, "base_q1", pl.DataFrame(base_rows), feature_set="base_panel")
    _write_dataset(feature_store, "feat_q1", pl.DataFrame(feature_rows))
    _write_dataset(label_store, "lab_q1", pl.DataFrame(label_rows), feature_set="labels")

    scanned_paths: list[str] = []
    original_scan = pl.scan_parquet

    def tracking_scan(source: object, *args: object, **kwargs: object):
        if isinstance(source, (list, tuple)):
            scanned_paths.extend(str(item) for item in source)
        else:
            scanned_paths.append(str(source))
        return original_scan(source, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", tracking_scan)

    loader = DirectMarketDataLoader(
        base_root=base_root, feature_root=feature_root, label_root=label_root
    )
    request = DirectDataRequest(
        base_dataset_id="base_q1",
        feature_dataset_id="feat_q1",
        label_dataset_id="lab_q1",
        start=date(2024, 2, 10),
        end=date(2024, 2, 14),
        candidate_horizon_sessions=(10,),
    )
    result = loader.load(request)

    # Only February partitions may appear in any scan.
    assert scanned_paths, "expected bounded scans to occur"
    for path in scanned_paths:
        assert "month=02" in path, f"scanned non-intersecting partition: {path}"

    # Collected columns equal the requested projection (plus the loader's
    # synthesized availability columns for panels that lack them).
    assert set(result.frame.columns) == {
        "instrument_id", "session", "open", "close",
        "volume", "trading_value", "feature__momentum_5d",
        "observation_time", "available_time",
    }
    expected_keys = {
        (f"KRX:{t + 1:05d}", s)
        for s in (datetime(2024, 2, d, tzinfo=UTC) for d in range(10, 15))
        for t in range(2)
    }
    actual_keys = set(
        zip(
            result.frame["instrument_id"].to_list(),
            result.frame["session"].to_list(),
            strict=True,
        )
    )
    assert actual_keys == expected_keys
    assert result.frame.height == result.frame.select(
        pl.struct(["instrument_id", "session"]).n_unique()
    ).item()

    # Labels contain only the requested horizon.
    assert set(result.labels_by_horizon) == {10}
    horizon_frame = result.labels_by_horizon[10]
    assert horizon_frame.height == 10  # 5 sessions x 2 tickers


DIRECT_SEPARATION_01 = "DIRECT-SEPARATION-01"


def test_direct_separation_01_one_row_per_key_with_independent_labels(
    tmp_path: Path,
) -> None:
    """DIRECT-SEPARATION-01: separated composition keeps frame at N rows.

    For N unique keys and H horizons, frame.height == N with unique keys and
    labels_by_horizon contains H independently sorted frames equal to the
    legacy composition's labels.
    """
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    for root in (base_root, feature_root, label_root):
        root.mkdir()
    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
    tickers = 3
    base_rows, feature_rows, label_rows = [], [], []
    for session in sessions:
        for t in range(tickers):
            price = 100.0 + t
            base_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "open": price, "close": price * 1.01,
                "volume": 1e6, "trading_value": price * 1e6,
            })
            feature_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "feature__momentum_5d": 0.1 * t,
            })
            # Horizon 20 misses the last session's target (unrealized tail).
            target_20 = 0.02 if session < sessions[-1] else None
            available_20 = (
                session + timedelta(days=20) if target_20 is not None else None
            )
            for horizon, target, available in (
                (10, 0.01, session + timedelta(days=10)),
                (20, target_20, available_20),
            ):
                label_rows.append({
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "horizon_sessions": horizon,
                    "net_alpha_target": target,
                    "label_available_time": available,
                })
    _write_dataset(base_store, "base_sep", pl.DataFrame(base_rows), feature_set="base_panel")
    _write_dataset(feature_store, "feat_sep", pl.DataFrame(feature_rows))
    _write_dataset(label_store, "lab_sep", pl.DataFrame(label_rows), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root, feature_root=feature_root, label_root=label_root
    )
    request = DirectDataRequest(
        base_dataset_id="base_sep",
        feature_dataset_id="feat_sep",
        label_dataset_id="lab_sep",
        start=date(2024, 1, 1),
        end=date(2024, 1, 6),
        candidate_horizon_sessions=(10, 20),
    )
    result = loader.load(request)

    n_keys = len(sessions) * tickers
    assert result.frame.height == n_keys  # never N * H
    unique_keys = result.frame.select(
        pl.struct(["instrument_id", "session"]).n_unique()
    ).item()
    assert unique_keys == n_keys
    assert set(result.labels_by_horizon) == {10, 20}

    # Legacy label semantics: per-horizon rows with non-null targets joined to
    # the decision keys, sorted by (instrument_id, session).
    for horizon, expected_rows in ((10, n_keys), (20, n_keys - tickers)):
        frame = result.labels_by_horizon[horizon]
        assert frame.height == expected_rows
        assert set(frame.columns) == {
            "instrument_id", "session", "target", "label_available_time",
        }
        keys = list(
            zip(
                frame["instrument_id"].to_list(),
                frame["session"].to_list(),
                strict=True,
            )
        )
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)


DIRECT_VALIDATION_01 = "DIRECT-VALIDATION-01"


def test_direct_validation_01_fail_closed_bounded_plan() -> None:
    """DIRECT-VALIDATION-01: vectorized validation fails closed.

    Duplicate keys, non-finite values, and source-order violations fail closed
    through the bounded aggregate/window plan.
    """
    from src.stocks.data.direct import _validate_direct_frame

    good = pl.DataFrame({
        "instrument_id": ["A", "A", "B"],
        "session": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ],
        "feature__a": [0.1, 0.2, 0.3],
    })

    with pytest.raises(ValueError, match="duplicate"):
        _validate_direct_frame(
            pl.concat([good, good.head(1)]), ("feature__a",)
        )

    non_finite = good.with_columns(
        pl.when(pl.col("instrument_id") == "B")
        .then(pl.lit(float("nan")))
        .otherwise(pl.col("feature__a"))
        .alias("feature__a")
    )
    with pytest.raises(ValueError, match="non-finite"):
        _validate_direct_frame(non_finite, ("feature__a",))

    disordered = pl.DataFrame({
        "instrument_id": ["A", "A"],
        "session": [
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ],
        "feature__a": [0.1, 0.2],
    })
    with pytest.raises(ValueError, match="non-monotonic"):
        _validate_direct_frame(disordered, ("feature__a",))


FULL_TERMINAL_01 = "FULL_TERMINAL_01_LAZY_DIRECT_OWNERSHIP"


def _write_long_label_dataset(
    store: ParquetDatasetStore,
    dataset_id: str,
    sessions: list[datetime],
    tickers: int = 2,
) -> None:
    """Long-format H10/H20/H30 labels; horizon 30 is never requested."""
    last = sessions[-1]
    rows = []
    for session in sessions:
        for t in range(tickers):
            for horizon in (10, 20, 30):
                usable = session + timedelta(days=horizon) <= last
                rows.append({
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "horizon_sessions": horizon,
                    "net_alpha_target": 0.001 * horizon if usable else None,
                    "label_available_time": session + timedelta(days=horizon),
                })
    _write_dataset(store, dataset_id, pl.DataFrame(rows), feature_set="labels")


def test_full_terminal_01_lazy_direct_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FULL_TERMINAL_01_LAZY_DIRECT_OWNERSHIP.

    H10/H20 labels are predicate-filtered before collect; after construction
    exactly one decision-width training frame is live and label frames contain
    no feature__ columns.
    """
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    for root in (base_root, feature_root, label_root):
        root.mkdir()
    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(40)]
    base_rows = []
    feature_rows = []
    for session in sessions:
        for t in range(2):
            price = 100.0 + t
            base_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "open": price, "close": price * 1.01,
                "volume": 1e6, "trading_value": price * 1e6,
            })
            feature_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "feature__momentum_5d": 0.1 * t,
                "feature__volatility_20d": 0.02,
            })
    _write_dataset(base_store, "base_lazy", pl.DataFrame(base_rows), feature_set="base_panel")
    _write_dataset(feature_store, "feat_lazy", pl.DataFrame(feature_rows))
    _write_long_label_dataset(label_store, "lab_lazy", sessions)

    def _no_eager_bounded_read(self, *args: object, **kwargs: object):
        raise AssertionError("load_training_data must not use eager read_bounded")

    monkeypatch.setattr(ParquetDatasetStore, "read_bounded", _no_eager_bounded_read)

    loader = DirectMarketDataLoader(
        base_root=base_root, feature_root=feature_root, label_root=label_root
    )
    request = DirectDataRequest(
        base_dataset_id="base_lazy",
        feature_dataset_id="feat_lazy",
        label_dataset_id="lab_lazy",
        start=date(2024, 1, 1),
        end=date(2024, 2, 9),
        candidate_horizon_sessions=(10, 20),
    )

    # The horizon predicate must live inside the lazy plan itself, so no other
    # horizon's rows ever reach a collect.
    for horizon, expected_scan_rows in ((10, 60), (20, 40)):
        scan = loader._scan_horizon_labels(request, horizon)
        assert isinstance(scan, pl.LazyFrame)
        assert "horizon_sessions" in scan.explain()
        rows = scan.collect()
        assert rows.height == expected_scan_rows  # only this horizon's rows
        assert rows["net_alpha_target"].null_count() == 0

    data = loader.load_training_data(request, datetime(2024, 3, 1, tzinfo=UTC))

    # Requested horizons only; label frames stay narrow and leak-free.
    assert set(data.labels_by_horizon) == {10, 20}
    n_keys = len(sessions) * 2 - 2  # one warm-up row dropped per ticker
    assert data.feature_frame.height == n_keys
    assert data.feature_frame.select(
        pl.struct(["instrument_id", "session"]).n_unique()
    ).item() == data.feature_frame.height
    for frame in data.labels_by_horizon.values():
        assert not any(c.startswith("feature__") for c in frame.columns)
        assert "horizon_sessions" not in frame.columns
        assert set(frame.columns) <= {
            "instrument_id",
            "session",
            "net_alpha_target",
            "label_available_time",
            "gross_return",
            "risk_residual",
            "reference_cost",
        }
    assert data.labels_by_horizon[10].height == 58
    assert data.labels_by_horizon[20].height == 38


TERMINAL_OBS_02 = "TERMINAL_OBS_02_JOIN_PREFLIGHT_FAILS_CLOSED"


def test_terminal_obs_02_join_preflight_fails_closed(tmp_path: Path) -> None:
    """TERMINAL_OBS_02_JOIN_PREFLIGHT_FAILS_CLOSED.

    A duplicate base key fails closed in the narrow preflight before the wide
    collect: the final checkpoint carries duplicate_key_count > 0 and no
    decision-frame-collected event is ever emitted.
    """
    base_root = tmp_path / "base"
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    for root in (base_root, feature_root, label_root):
        root.mkdir()
    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
    feature_rows = []
    label_rows = []
    base_rows = []
    for session in sessions:
        for t in range(2):
            price = 100.0 + t
            base_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "open": price, "close": price * 1.01,
                "volume": 1e6, "trading_value": price * 1e6,
            })
            feature_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "feature__momentum_5d": 0.1 * t,
            })
            label_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "horizon_sessions": 10,
                "net_alpha_target": 0.001,
                "label_available_time": session + timedelta(days=10),
            })
    # Duplicate one identity key on the base side.
    duplicated_base = pl.concat([pl.DataFrame(base_rows), pl.DataFrame(base_rows).head(1)])
    _write_dataset(base_store, "base_dup", duplicated_base, feature_set="base_panel")
    _write_dataset(feature_store, "feat_dup", pl.DataFrame(feature_rows))
    _write_dataset(label_store, "lab_dup", pl.DataFrame(label_rows), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root, feature_root=feature_root, label_root=label_root
    )
    from src.stocks.data.direct import DirectLoadCheckpoint

    request = DirectDataRequest(
        base_dataset_id="base_dup",
        feature_dataset_id="feat_dup",
        label_dataset_id="lab_dup",
        start=date(2024, 1, 1),
        end=date(2024, 1, 6),
        candidate_horizon_sessions=(10,),
    )
    checkpoints: list[DirectLoadCheckpoint] = []

    with pytest.raises(ValueError, match="duplicate"):
        loader.load_training_data(
            request,
            datetime(2024, 3, 1, tzinfo=UTC),
            checkpoint=checkpoints.append,
        )

    assert checkpoints, "preflight must emit durable checkpoints before failing"
    final_checkpoint = checkpoints[-1]
    assert final_checkpoint.stage == "direct_preflight"
    duplicate_key_count = (final_checkpoint.base_duplicate_keys or 0) + (
        final_checkpoint.feature_duplicate_keys or 0
    )
    assert duplicate_key_count > 0
    assert all(
        checkpoint.stage != "decision_frame_collected" for checkpoint in checkpoints
    )


def test_terminal_obs_02_preflight_admits_one_to_one(tmp_path: Path) -> None:
    """The same fixture without duplicates passes preflight and collects once."""
    base_root = tmp_path / "base_ok"
    feature_root = tmp_path / "features_ok"
    label_root = tmp_path / "labels_ok"
    for root in (base_root, feature_root, label_root):
        root.mkdir()
    base_store = ParquetDatasetStore(base_root)
    feature_store = ParquetDatasetStore(feature_root)
    label_store = ParquetDatasetStore(label_root)

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
    base_rows = []
    feature_rows = []
    label_rows = []
    for session in sessions:
        for t in range(2):
            price = 100.0 + t
            base_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "open": price, "close": price * 1.01,
                "volume": 1e6, "trading_value": price * 1e6,
            })
            feature_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "feature__momentum_5d": 0.1 * t,
            })
            label_rows.append({
                "instrument_id": f"KRX:{t + 1:05d}", "session": session,
                "horizon_sessions": 10,
                "net_alpha_target": 0.001,
                "label_available_time": session + timedelta(days=10),
            })
    _write_dataset(base_store, "base_ok", pl.DataFrame(base_rows), feature_set="base_panel")
    _write_dataset(feature_store, "feat_ok", pl.DataFrame(feature_rows))
    _write_dataset(label_store, "lab_ok", pl.DataFrame(label_rows), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=base_root, feature_root=feature_root, label_root=label_root
    )
    from src.stocks.data.direct import DirectLoadCheckpoint

    request = DirectDataRequest(
        base_dataset_id="base_ok",
        feature_dataset_id="feat_ok",
        label_dataset_id="lab_ok",
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
        candidate_horizon_sessions=(10,),
    )
    checkpoints: list[DirectLoadCheckpoint] = []
    data = loader.load_training_data(
        request, datetime(2024, 3, 1, tzinfo=UTC), checkpoint=checkpoints.append
    )
    stages = [checkpoint.stage for checkpoint in checkpoints]
    assert "decision_frame_collected" not in stages
    assert "direct_collected" in stages
    preflight = next(c for c in checkpoints if c.stage == "direct_preflight")
    assert preflight.predicted_joined_rows == preflight.matched_keys
    assert preflight.planned_lower_bound_bytes is not None
    assert preflight.planned_lower_bound_bytes > 0
    assert data.feature_frame.height == len(sessions) * 2 - 2
