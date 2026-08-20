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

    _write_dataset(base_store, "base_2024", _base_frame())
    _write_dataset(feature_store, "features_2024", _feature_frame())
    _write_dataset(label_store, "labels_2024", _label_frame())

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
    _write_dataset(base_store, "base_dup", duplicate_base)
    _write_dataset(feature_store, "features_dup", _feature_frame())
    _write_dataset(label_store, "labels_dup", _label_frame())

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
    with pytest.raises(ValueError, match="non-monotonic"):
        loader._validate_monotonic_sessions(base_frame)


def test_direct_loader_rejects_non_finite_features() -> None:
    """Direct loader rejects non-finite numeric feature values."""
    loader = DirectMarketDataLoader(
        base_root=Path("/nonexistent"),
        feature_root=Path("/nonexistent"),
        label_root=Path("/nonexistent"),
    )

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

    with pytest.raises(ValueError, match="non-finite"):
        loader._validate_numeric_finiteness(frame)


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
    _write_dataset(base_store, "base_no_labels", _base_frame())
    _write_dataset(feature_store, "features_no_labels", _feature_frame())
    _write_dataset(label_store, "labels_no_labels", label_frame)

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

    _write_dataset(base_store, "base_2024", _base_frame())
    _write_dataset(feature_store, "features_2024", _feature_frame())
    _write_dataset(label_store, "labels_2024", _label_frame())

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
