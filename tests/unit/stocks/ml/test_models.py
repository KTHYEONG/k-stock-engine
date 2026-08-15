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


def _lightgbm_frame(n_rows: int = 400, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 12)]
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i % 8 + 1:05d}" for i in range(n_rows)],
            "session": [sessions[i % len(sessions)] for i in range(n_rows)],
            "feature_momentum_5d": rng.normal(0.0, 1.0, n_rows),
            "net_alpha_target": rng.normal(0.0, 1.0, n_rows),
        }
    )


def test_lightgbm_target_free_fit_is_deterministic_and_no_early_stopping(monkeypatch) -> None:
    """A target-free validation frame trains the fixed budget without early stopping."""
    from src.stocks.ml import models as models_module
    from src.stocks.ml.models import LightGbmNetAlpha

    manifest = _manifest()
    train = _lightgbm_frame()
    empty = train.head(0)
    captured: dict[str, object] = {}
    real_train = models_module.lgb.train

    def spy(*args, **kwargs):
        captured["valid_sets"] = kwargs.get("valid_sets")
        captured["has_early_stopping"] = any(
            (
                "early" in type(cb).__name__.lower()
                and "stopping" in type(cb).__name__.lower()
            )
            for cb in (kwargs.get("callbacks") or [])
        )
        return real_train(*args, **kwargs)

    monkeypatch.setattr(models_module.lgb, "train", spy)
    config = NetAlphaModelConfig(seed=42, n_estimators=20, min_child_samples=10)
    model = LightGbmNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=config, num_threads=1,
    )
    model.fit(train, empty)
    assert captured["valid_sets"] is None
    assert captured["has_early_stopping"] is False
    assert model._best_iteration is None

    predictions = model.predict(
        train.select("instrument_id", "session", "feature_momentum_5d")
    )
    assert predictions[SCORE_COLUMN].n_unique() > 1

    second = LightGbmNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=config, num_threads=1,
    )
    second.fit(train, empty)
    assert np.array_equal(
        predictions[SCORE_COLUMN].to_numpy(),
        second.predict(train.select("instrument_id", "session", "feature_momentum_5d"))[
            SCORE_COLUMN
        ].to_numpy(),
    )


def test_lightgbm_labeled_validation_requests_early_stopping(monkeypatch) -> None:
    """A finite labeled validation set keeps the early-stopping path."""
    from src.stocks.ml import models as models_module
    from src.stocks.ml.models import LightGbmNetAlpha

    manifest = _manifest()
    train = _lightgbm_frame()
    valid = _lightgbm_frame(seed=8)
    captured: dict[str, object] = {}
    real_train = models_module.lgb.train

    def spy(*args, **kwargs):
        captured["valid_sets"] = kwargs.get("valid_sets")
        captured["has_early_stopping"] = any(
            (
                "early" in type(cb).__name__.lower()
                and "stopping" in type(cb).__name__.lower()
            )
            for cb in (kwargs.get("callbacks") or [])
        )
        return real_train(*args, **kwargs)

    monkeypatch.setattr(models_module.lgb, "train", spy)
    model = LightGbmNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=NetAlphaModelConfig(seed=42, n_estimators=20, min_child_samples=10),
        num_threads=1,
    )
    model.fit(train, valid)
    assert captured["valid_sets"] is not None
    assert captured["has_early_stopping"] is True
    assert model._best_iteration is not None


def test_causal_calibration_adapter_preserves_public_contract() -> None:
    """The causal adapter preserves predicted_net_alpha and appends decimal columns."""
    from src.stocks.research.economic_alpha import CausalAlphaCalibrator

    from src.stocks.ml.models import CausalCalibrationAdapter

    scored = _oof_panel().select("instrument_id", "session", SCORE_COLUMN)
    state: dict[str, object] = {
        "bucket_count": 3,
        "history_sessions": 10,
        "round_trip_cost": 0.001,
        "exit_cost_rate": 0.0005,
        "buckets": [],
    }
    adapter = CausalCalibrationAdapter(
        CausalAlphaCalibrator(bucket_count=3, min_calibration_sessions=1, seed=1), state
    )
    augmented = adapter.apply(scored)
    assert SCORE_COLUMN in augmented.columns
    assert "expected_net_alpha" in augmented.columns
    assert "net_alpha_lower_bound" in augmented.columns
    assert "calibration_state" in CalibratedNetAlphaModel(
        _DummyModel(_manifest()), adapter
    ).manifest().params


class _DummyModel:
    def __init__(self, manifest: ModelManifest):
        self._manifest = manifest

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        del train, validation

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(SCORE_COLUMN))

    def manifest(self) -> ModelManifest:
        return self._manifest
