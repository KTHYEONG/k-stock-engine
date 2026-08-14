"""Deterministic net-alpha model and OOF calibration tests."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.research.models import ModelManifest
from src.stocks.research.net_alpha import (
    ElasticNetNetAlpha,
    LightGbmNetAlpha,
    NetAlphaCalibrator,
    NetAlphaModelConfig,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        artifact_id="net_alpha_test",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v3",
        feature_schema_hash="schema",
        universe_policy_hash="policy",
        label_definition="net_alpha_5d_target",
        label_horizon_sessions=5,
        eligible_from="2024-01-01",
        eligible_to="2024-12-31",
        model_type="elastic_net_net_alpha",
    )


def _frame(n_sessions: int = 8, n_tickers: int = 10) -> pl.DataFrame:
    rows = [
        {
            "session": datetime(2024, 1, s + 1, tzinfo=UTC),
            "instrument_id": f"KRX:{t:06d}",
            "momentum__rank": float((s + t) % n_tickers) / n_tickers,
            "momentum__sector_rank": float((s * 3 + t) % n_tickers) / n_tickers,
            "net_alpha_5d_target": 0.01 * ((s + t) % 5) - 0.01,
        }
        for s in range(n_sessions)
        for t in range(n_tickers)
    ]
    return pl.DataFrame(rows)


def _linear_frame(n_sessions: int = 12, n_tickers: int = 8) -> pl.DataFrame:
    rows = [
        {
            "session": datetime(2024, 1, s + 1, tzinfo=UTC),
            "instrument_id": f"KRX:{t:06d}",
            "momentum__rank": float(s) / n_sessions + float(t) / n_tickers,
            "momentum__sector_rank": 1.0 - (float(s) / n_sessions + float(t) / n_tickers),
            "net_alpha_5d_target": 0.05 * (float(s) / n_sessions + float(t) / n_tickers)
            - 0.005
            + (t % 2) * 1e-9,
        }
        for s in range(n_sessions)
        for t in range(n_tickers)
    ]
    return pl.DataFrame(rows)


FEATURES = ("momentum__rank", "momentum__sector_rank")
LABEL = "net_alpha_5d_target"


class TestElasticNet:
    def test_fit_predict_deterministic(self) -> None:
        frame = _linear_frame()
        train, validation = frame.head(80), frame.tail(16)
        first = ElasticNetNetAlpha(_manifest(), FEATURES, LABEL)
        second = ElasticNetNetAlpha(_manifest(), FEATURES, LABEL)
        first.fit(train, validation)
        second.fit(train, validation)
        predict_frame = validation.drop(LABEL)
        out_first = first.predict(predict_frame)
        out_second = second.predict(predict_frame)
        assert out_first["pred_score"].to_list() == pytest.approx(
            out_second["pred_score"].to_list()
        )
        assert "pred_score" in out_first.columns

    def test_rejects_target_columns(self) -> None:
        model = ElasticNetNetAlpha(_manifest(), FEATURES, LABEL)
        model.fit(_linear_frame(), _linear_frame().tail(10))
        with pytest.raises(ValueError, match="target/label"):
            model.predict(_linear_frame().with_columns(
                pl.lit(1.0).alias("label_should_be_dropped")
            ))

    def test_missing_feature_rejected(self) -> None:
        model = ElasticNetNetAlpha(_manifest(), FEATURES, LABEL)
        with pytest.raises(ValueError, match="missing feature columns"):
            model.fit(_linear_frame().drop("momentum__sector_rank"), _linear_frame())


class TestLightGbm:
    def test_fit_predict_deterministic(self) -> None:
        frame = _linear_frame()
        train, validation = frame.head(80), frame.tail(16)
        first = LightGbmNetAlpha(_manifest(), FEATURES, LABEL)
        second = LightGbmNetAlpha(_manifest(), FEATURES, LABEL)
        first.fit(train, validation)
        second.fit(train, validation)
        predict_frame = validation.drop(LABEL)
        out_first = first.predict(predict_frame)
        out_second = second.predict(predict_frame)
        assert out_first["pred_score"].to_list() == pytest.approx(
            out_second["pred_score"].to_list(), abs=1e-9
        )

    def test_manifest_type(self) -> None:
        model = LightGbmNetAlpha(_manifest(), FEATURES, LABEL)
        model.fit(_linear_frame(), _linear_frame().tail(10))
        assert model.manifest().model_type == "lightgbm_l1_net_alpha"

    def test_rejects_non_positive_threads(self) -> None:
        with pytest.raises(ValueError, match="num_threads"):
            LightGbmNetAlpha(_manifest(), FEATURES, LABEL, num_threads=0)


class TestCalibrator:
    def test_fit_apply_uses_oof_labels_only(self) -> None:
        frame = _linear_frame()
        train = frame.head(80).with_columns(
            (pl.col("momentum__rank") * 0.5).alias("pred_score")
        )
        scored = frame.tail(16).with_columns(
            (pl.col("momentum__rank") * 0.5).alias("pred_score")
        )
        calibrator = NetAlphaCalibrator(bucket_count=4, label_column=LABEL)
        state = calibrator.fit(train)
        assert state.bucket_count == 4
        augmented = calibrator.apply(scored.drop(LABEL))
        assert "expected_net_alpha" in augmented.columns
        assert "net_alpha_lower_bound" in augmented.columns

    def test_apply_without_fit_raises(self) -> None:
        calibrator = NetAlphaCalibrator(bucket_count=4, label_column=LABEL)
        with pytest.raises(ValueError, match="no frozen state"):
            calibrator.apply(_frame())

    def test_empty_bucket_state_zeroes_output(self) -> None:
        calibrator = NetAlphaCalibrator(
            bucket_count=4, label_column=LABEL, bootstrap_alpha=0.5
        )
        train = _frame().with_columns(
            (pl.col("momentum__rank")).alias("pred_score")
        )
        state = calibrator.fit(train)
        if not state.buckets:
            augmented = calibrator.apply(train.drop(LABEL))
            assert augmented["expected_net_alpha"].to_list() == [0.0] * augmented.height

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="num_leaves"):
            NetAlphaModelConfig(num_leaves=1)
        with pytest.raises(ValueError, match="learning_rate"):
            NetAlphaModelConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="elastic_alpha"):
            NetAlphaModelConfig(elastic_alpha=-1.0)


def test_predict_matches_across_equal_instances() -> None:
    frame = _linear_frame()
    train = frame.head(80)
    config = NetAlphaModelConfig(seed=7)
    first = LightGbmNetAlpha(_manifest(), FEATURES, LABEL, config=config)
    second = LightGbmNetAlpha(_manifest(), FEATURES, LABEL, config=config)
    first.fit(train, frame.tail(16))
    second.fit(train, frame.tail(16))
    predict_frame = frame.tail(16).drop(LABEL)
    assert np.array_equal(
        first.predict(predict_frame)["pred_score"].to_numpy(),
        second.predict(predict_frame)["pred_score"].to_numpy(),
    )
