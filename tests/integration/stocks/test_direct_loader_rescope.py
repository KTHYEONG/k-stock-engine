"""Direct-loader universe rescope integration: SCENARIO_RESCOPE_LOADER_INTEGRATION."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.core.datasets import HIVE_PARTITION_LAYOUT, make_manifest
from src.core.instruments import AssetKind
from src.stocks.data.direct import (
    DirectDataRequest,
    DirectLoadCheckpoint,
    DirectMarketDataLoader,
)
from src.stocks.ml.contracts import UniverseRescopeSettings
from src.storage.parquet_datasets import ParquetDatasetStore

SESSIONS = 40
TICKERS = 4


def _write_dataset(
    store: ParquetDatasetStore,
    dataset_id: str,
    frame: pl.DataFrame,
    *,
    feature_set: str,
) -> None:
    from src.storage.parquet_datasets import canonical_content_hash

    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set=feature_set,
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 2, 9, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        generated_time=datetime.now(UTC),
        schema_version="v2",
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    manifest = replace(manifest, content_hash=canonical_content_hash(frame, frame.columns))
    store.write_partitioned(
        frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=feature_set,
        decision_time=datetime(2024, 3, 31, tzinfo=UTC),
    )


def _base_frame() -> pl.DataFrame:
    rows = [
        {
            "instrument_id": f"KRX:{t + 1:05d}",
            "session": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
            "open": 100.0 + t,
            "close": (100.0 + t) * 1.01,
            "volume": 1e6,
            "trading_value": (100.0 + t) * 1e6,
            "market_cap": float((t + 1) * 1_000_000_000_000),
        }
        for i in range(SESSIONS)
        for t in range(TICKERS)
    ]
    return pl.DataFrame(rows)


def _feature_frame() -> pl.DataFrame:
    rows = [
        {
            "instrument_id": f"KRX:{t + 1:05d}",
            "session": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
            "feature__momentum_5d": 0.1 * (t + i % 3),
            "feature__volatility_20d": 0.02,
        }
        for i in range(SESSIONS)
        for t in range(TICKERS)
    ]
    return pl.DataFrame(rows)


def _label_frame() -> pl.DataFrame:
    last = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=SESSIONS - 1)
    rows = [
        {
            "instrument_id": f"KRX:{t + 1:05d}",
            "session": session,
            "horizon_sessions": 10,
            "net_alpha_target": 0.001 if session + timedelta(days=10) <= last else None,
            "label_available_time": session + timedelta(days=10),
        }
        for i in range(SESSIONS)
        for t in range(TICKERS)
        for session in [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)]
    ]
    return pl.DataFrame(rows)


def _request() -> DirectDataRequest:
    return DirectDataRequest(
        base_dataset_id="base_rescope",
        feature_dataset_id="feat_rescope",
        label_dataset_id="lab_rescope",
        start=date(2024, 1, 1),
        end=date(2024, 2, 9),
        candidate_horizon_sessions=(10,),
    )


def test_direct_loader_rescope_integration(tmp_path: Path) -> None:
    """SCENARIO_RESCOPE_LOADER_INTEGRATION."""
    roots = (tmp_path / "base", tmp_path / "features", tmp_path / "labels")
    for root in roots:
        root.mkdir()
    base_store = ParquetDatasetStore(roots[0])
    feature_store = ParquetDatasetStore(roots[1])
    label_store = ParquetDatasetStore(roots[2])
    _write_dataset(base_store, "base_rescope", _base_frame(), feature_set="base_panel")
    _write_dataset(feature_store, "feat_rescope", _feature_frame(), feature_set="stock_net_alpha_v1")
    _write_dataset(label_store, "lab_rescope", _label_frame(), feature_set="labels")

    loader = DirectMarketDataLoader(
        base_root=roots[0], feature_root=roots[1], label_root=roots[2]
    )
    checkpoints: list[tuple[str, dict[str, object]]] = []

    def capture(checkpoint: DirectLoadCheckpoint) -> None:
        checkpoints.append((checkpoint.stage, checkpoint.journal_payload()))

    scoped = loader.load_training_data(
        _request(),
        datetime(2024, 3, 1, tzinfo=UTC),
        checkpoint=capture,
        rescope=UniverseRescopeSettings(market_cap_quantile_lo=0.5),
    )

    # Top half by trailing market cap survives: tickers 3 and 4.
    retained = set(scoped.feature_frame["instrument_id"].unique().to_list())
    assert retained == {"KRX:00003", "KRX:00004"}
    per_session = (
        scoped.feature_frame.group_by("session").len().sort("session")["len"].to_list()
    )
    assert set(per_session) == {2}

    # Labels stay a subset of the retained decision keys (usable-label join),
    # so no scoped-out key leaks into the horizon frames.
    labels = scoped.labels_by_horizon[10]
    decision_keys = scoped.feature_frame.select("instrument_id", "session")
    leaked = labels.join(decision_keys, on=["instrument_id", "session"], how="anti")
    assert leaked.is_empty()
    assert set(labels["instrument_id"].unique().to_list()) <= retained

    # Rescope diagnostics surface through the direct-load journal.
    collected = dict(checkpoints)["direct_collected"]
    diag = collected["universe_rescope"]
    assert isinstance(diag, str)
    assert "'kept_row_count': 80" in diag
    assert "'kept_row_fraction': 0.5" in diag

    # Flag-off parity: no rescope reproduces the unscoped frame exactly.
    plain = loader.load_training_data(_request(), datetime(2024, 3, 1, tzinfo=UTC))
    assert set(plain.feature_frame["instrument_id"].unique().to_list()) == {
        f"KRX:{t + 1:05d}" for t in range(TICKERS)
    }
    unscoped_keys = plain.feature_frame.select("instrument_id", "session")
    scoped_keys = scoped.feature_frame.select(unscoped_keys.columns)
    assert scoped_keys.equals(
        unscoped_keys.filter(
            pl.col("instrument_id").is_in(["KRX:00003", "KRX:00004"])
        ).sort(["instrument_id", "session"])
    )
    # The enabled run carries the extra passthrough column; the flag-off run does not.
    assert "market_cap" not in plain.feature_frame.columns
    assert "market_cap" in scoped.feature_frame.columns
