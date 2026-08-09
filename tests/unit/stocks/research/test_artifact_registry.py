"""PLAN-04-ARTIFACT-COMPATIBILITY: Artifact compatibility fails closed."""
from __future__ import annotations

from datetime import datetime, UTC

import pytest

from src.core.instruments import AssetKind
from src.stocks.ml.artifacts import (
    ModelArtifactRegistry,
    PredictionRequest,
)
from src.stocks.ml.dataset import schema_hash
from src.stocks.ml.models import DeterministicBaseline, ModelManifest

FEATURES = ["session_index", "instrument_id", "feature_momentum_5d", "date"]


def make_manifest(
    artifact_id: str = "stock_alpha_v1_20240101",
    asset_kind: AssetKind = AssetKind.STOCK,
    feature_set: str = "stock_alpha_v1",
    schema: list[str] | None = None,
) -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind=asset_kind,
        feature_set=feature_set,
        feature_schema_hash=schema_hash(schema or FEATURES),
        universe_policy_hash="universe-hash",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="baseline",
    )


def request(decision: datetime | None = None, **overrides: object) -> PredictionRequest:
    return PredictionRequest(
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v1",
        feature_schema_hash=schema_hash(FEATURES),
        decision_time=decision or datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
    )


class TestArtifactRegistry:
    def test_loading_matching_artifact_returns_stock_kind(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        loaded = registry.load("stock_alpha_v1_20240101", request())
        assert loaded.manifest.asset_kind is AssetKind.STOCK

    def test_loading_with_different_asset_kind_fails_closed(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest(asset_kind=AssetKind.ETF))
        registry.publish(model, model.manifest())
        with pytest.raises(ValueError, match="asset kind"):
            registry.load("stock_alpha_v1_20240101", request())

    def test_loading_with_different_feature_schema_fails_closed(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        mismatch = PredictionRequest(
            asset_kind=AssetKind.STOCK,
            feature_set="stock_alpha_v1",
            feature_schema_hash=schema_hash(["different", "columns"]),
            decision_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="schema hash"):
            registry.load("stock_alpha_v1_20240101", mismatch)

    def test_loading_with_different_feature_set_fails_closed(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        wrong_set = PredictionRequest(
            asset_kind=AssetKind.STOCK,
            feature_set="etf_alpha_v2",
            feature_schema_hash=schema_hash(FEATURES),
            decision_time=datetime(2024, 6, 1, 8, 0, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="feature-set"):
            registry.load("stock_alpha_v1_20240101", wrong_set)

    def test_no_implicit_latest_fallback(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        with pytest.raises(FileNotFoundError):
            registry.load("latest", request())
        with pytest.raises(FileNotFoundError):
            registry.load("does_not_exist", request())

    def test_out_of_eligibility_window_fails_closed(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        too_late = request(decision=datetime(2030, 1, 1, tzinfo=UTC))
        with pytest.raises(ValueError, match="not eligible"):
            registry.load("stock_alpha_v1_20240101", too_late)

    def test_duplicate_publish_is_rejected(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        with pytest.raises(ValueError, match="already exists"):
            registry.publish(model, model.manifest())
