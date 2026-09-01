"""Artifact manifest inspection: read_manifest without model load."""
from __future__ import annotations

import pytest

from src.core.instruments import AssetKind
from legacy.stocks.research.artifacts import ModelArtifactRegistry
from legacy.stocks.research.datasets import schema_hash
from legacy.stocks.research.models import DeterministicBaseline, ModelManifest

FEATURES = ["session_index", "instrument_id", "feature_momentum_5d"]


def make_manifest(artifact_id: str = "a001") -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v1",
        feature_schema_hash=schema_hash(FEATURES),
        universe_policy_hash="universe-hash",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="baseline",
    )


class TestReadManifest:
    def test_read_manifest_returns_frozen_metadata_without_loading_model(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        model = DeterministicBaseline(manifest=make_manifest())
        registry.publish(model, model.manifest())
        manifest = registry.read_manifest("a001")
        assert manifest.artifact_id == "a001"
        assert manifest.eligible_from == "2024-01-01T00:00:00+00:00"
        assert manifest.eligible_to == "2024-12-31T00:00:00+00:00"

    def test_read_manifest_rejects_missing_artifact(self, tmp_path) -> None:
        # SDA-06: artifact replay must pin immutable lineage before loading data.
        registry = ModelArtifactRegistry(tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.read_manifest("missing")

    def test_read_manifest_rejects_invalid_artifact_id(self, tmp_path) -> None:
        registry = ModelArtifactRegistry(tmp_path)
        with pytest.raises(ValueError, match="invalid artifact_id"):
            registry.read_manifest("no spaces here")


def test_in_memory_registry_keeps_artifacts_off_disk(tmp_path) -> None:
    root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry.in_memory()
    model = DeterministicBaseline(manifest=make_manifest())

    registry.publish(model, model.manifest())
    registry.write_metrics("a001", {"promoted": False})

    assert registry.list_artifact_ids() == ("a001",)
    assert registry.read_metrics("a001") == {"promoted": False}
    assert not root.exists()
