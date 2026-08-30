from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.ml.fitting import OofCache, OofFitRequest, fit_horizon_oof
from src.stocks.ml.preparation import prepare_folds, prepare_horizon_labels, prepare_matrix_from_frame
from src.stocks.research.folds import Fold


def _small_unhedged_prepared_fixture():
    from datetime import UTC as _UTC, datetime as _dt, timedelta

    rng = np.random.default_rng(123)
    n_sessions = 60
    per_session = 10
    start = _dt(2024, 1, 1, tzinfo=_UTC)
    rows = []  # noqa: PERF401
    for s in range(n_sessions):
        session = start + timedelta(days=s)
        for t in range(per_session):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t+1:05d}",
                    "session": session,
                    "feature__a": float(rng.normal()),
                    "feature__b": float(rng.normal()),
                    "open": 10000.0,
                    "adtv_20d": 1e9,
                    "volatility_20d": 0.02,
                }
            )
    frame = pl.DataFrame(rows).sort(["session", "instrument_id"])
    frame = frame.with_columns(pl.col("session").rank("dense").cast(pl.Int64).alias("session_index"))
    # build labels with gross_return etc. correlated with feature__a
    label_rows = [  # noqa: PERF401
        {
            "instrument_id": row["instrument_id"],
            "session": row["session"],
            "net_alpha_target": float(row["feature__a"] * 0.5 + rng.normal(scale=0.1)),
            "label_available_time": row["session"] + timedelta(days=10),
            "gross_return": float(row["feature__a"] * 0.01 + 0.005 + rng.normal(scale=0.002)),
            "risk_residual": float(row["feature__a"] * 0.005 + rng.normal(scale=0.002)),
            "reference_cost": 0.001,
        }
        for row in frame.iter_rows(named=True)
    ]
    labels = pl.DataFrame(label_rows)
    matrix = prepare_matrix_from_frame(frame, ("feature__a", "feature__b"))
    # simple folds: expanding
    folds_raw = []
    for i in range(3):
        val_start = 30 + i * 5
        train_mask = [int(idx) for idx in range(frame.height) if frame["session_index"][idx] < val_start - 2]
        val_mask = [int(idx) for idx in range(frame.height) if val_start <= frame["session_index"][idx] < val_start + 5]
        folds_raw.append(Fold(train_mask=train_mask, validation_mask=val_mask, train_label_end=val_start - 2, validation_decision_start=val_start, segment_id=i))
    folds = prepare_folds(folds_raw)
    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    from src.stocks.ml.contracts import NetAlphaResearchData

    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=start,
        time_end=start + timedelta(days=n_sessions),
        generated_time=start,
        row_count=frame.height,
    )
    data = NetAlphaResearchData(feature_frame=frame, labels_by_horizon={10: labels}, manifest=manifest)
    horizon = prepare_horizon_labels(matrix, data, 10, route_objective=SimpleNamespace(kind="unhedged_absolute"))
    request = NetAlphaTrainingRequest(artifact_id="test_fit", candidate_horizon_sessions=(10,), route_objective=SimpleNamespace(kind="unhedged_absolute"))
    # attach manifest etc for fitting
    return matrix, horizon, folds, request


def _get_route_calibration_ledger():
    try:
        from src.stocks.ml.model_selection import route_calibration_ledger as _rcl

        return _rcl
    except Exception:
        from src.stocks.ml.training import route_calibration_ledger as _rcl2

        return _rcl2

route_calibration_ledger = _get_route_calibration_ledger()


def test_oof_cache_closes_without_error(tmp_path) -> None:
    cache = OofCache(tmp_path)
    assert cache.root.exists()
    cache.close()


def test_fit_horizon_oof_unhedged_outputs_gross_for_route_calibration() -> None:
    # Given
    matrix, horizon, folds, request = _small_unhedged_prepared_fixture()
    # When
    result = fit_horizon_oof(matrix, horizon, folds, OofFitRequest(request=request))
    # Then
    assert "gross_return" in result.labeled.columns
    assert result.labeled["gross_return"].null_count() == 0
    ledger = route_calibration_ledger(result.labeled, request)
    assert "gross_return" in ledger.columns
    assert ledger.height == result.labeled.height
