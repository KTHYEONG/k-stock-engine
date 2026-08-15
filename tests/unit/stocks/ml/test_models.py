"""Net-alpha models: decimal OOF calibration serialization and champion wrapper."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from src.core.instruments import AssetKind
from src.stocks.ml.models import (
    SCORE_COLUMN,
    CalibrationState,
    CalibratedNetAlphaModel,
    ElasticNetNetAlpha,
    NetAlphaCalibrator,
    NetAlphaModelConfig,
)
from src.stocks.research.models import ModelManifest


def _manifest(artifact_id: str = "na_cal") -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="net-alpha-v1",
        universe_policy_hash="net-alpha-v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
    )


def _oof_panel(n_rows: int = 120, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    scores = np.sort(rng.uniform(-1.0, 1.0, n_rows))
    realized = 0.02 * scores + rng.normal(0.0, 0.001, n_rows)
    sessions = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 10)]
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i % 8 + 1:05d}" for i in range(n_rows)],
            "session": [sessions[i % len(sessions)] for i in range(n_rows)],
            SCORE_COLUMN: scores,
            "realized_net_return": realized,
        }
    )


def test_calibration_state_json_round_trip() -> None:
    oof = _oof_panel()
    calibrator = NetAlphaCalibrator(
        bucket_count=10,
        seed=42,
        n_bootstrap=50,
        bootstrap_alpha=0.05,
        block_length=5,
        label_column="realized_net_return",
    )
    state = calibrator.fit(oof)
    restored = CalibrationState.from_json(state.to_json())
    assert restored == state
    assert restored.bucket_count == state.bucket_count


def test_calibration_fits_decimal_realized_returns_and_applies() -> None:
    oof = _oof_panel()
    calibrator = NetAlphaCalibrator(
        bucket_count=10,
        seed=42,
        n_bootstrap=50,
        bootstrap_alpha=0.05,
        block_length=5,
        label_column="realized_net_return",
    )
    state = calibrator.fit(oof)
    assert state.buckets
    calibrated = calibrator.apply(oof)
    assert "net_alpha_lower_bound" in calibrated.columns
    assert "expected_net_alpha" in calibrated.columns
    assert calibrated["net_alpha_lower_bound"].min() >= 0.0


def test_calibrated_model_predict_applies_lower_bound() -> None:
    manifest = _manifest()
    oof = _oof_panel().with_columns(
        pl.col("realized_net_return").alias("net_alpha_target"),
        pl.col(SCORE_COLUMN).alias("feature_momentum_5d"),
    )
    base = ElasticNetNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=NetAlphaModelConfig(seed=42),
    )
    base.fit(oof, oof.head(0))
    calibrator = NetAlphaCalibrator(
        bucket_count=5, seed=42, n_bootstrap=50, block_length=3,
        label_column="realized_net_return",
    )
    calibrator.fit(oof)
    wrapped = CalibratedNetAlphaModel(base, calibrator)
    prediction = wrapped.predict(oof.select("instrument_id", "session", "feature_momentum_5d"))
    assert SCORE_COLUMN in prediction.columns
    assert "net_alpha_lower_bound" in prediction.columns
    assert "calibration_state" in (wrapped.manifest().params or {})
