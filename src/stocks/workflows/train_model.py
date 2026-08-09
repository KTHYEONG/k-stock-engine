"""Stock model-training workflow: snapshot -> folds -> model -> artifact."""
from __future__ import annotations

from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.folds import PurgedWalkForward
from src.stocks.research.labels import LabelDefinition
from src.stocks.research.models import DeterministicBaseline, ModelManifest
from src.stocks.workflows.contracts import TrainingRequest


def train_model(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: TrainingRequest,
) -> ModelManifest:
    """Train a deterministic baseline and publish an immutable artifact.

    The workflow receives validated dataset metadata; CLI/adapters, not this
    workflow, read Parquet or manufacture manifests.
    """
    manifest = snapshot.manifest
    if manifest.asset_kind is not AssetKind.STOCK:
        raise ValueError(
            f"train_model only accepts stock datasets, got {manifest.asset_kind.value}"
        )

    label = LabelDefinition(
        name=manifest.label_definition,
        entry_field="close",
        exit_field="close",
        horizon_sessions=manifest.label_horizon_sessions,
    )
    label_frame = label.apply(snapshot.frame)
    if manifest.label_definition not in label_frame.columns:
        raise ValueError("label computation failed")

    splitter = PurgedWalkForward(
        n_folds=request.n_folds,
        label_horizon_sessions=manifest.label_horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
    )
    folds = splitter.split(label_frame)
    if not folds:
        raise ValueError("no folds available for training")

    fold = folds[0]
    train_frame = label_frame[fold.train_mask]
    validation_frame = label_frame[fold.validation_mask]

    model = DeterministicBaseline(
        manifest=ModelManifest(
            artifact_id=request.artifact_id,
            asset_kind=AssetKind.STOCK,
            feature_set=manifest.feature_set,
            feature_schema_hash=manifest.schema_hash,
            universe_policy_hash=manifest.universe_policy_hash,
            label_definition=manifest.label_definition,
            label_horizon_sessions=manifest.label_horizon_sessions,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2025-12-31T00:00:00+00:00",
            model_type="baseline",
        )
    )
    model.fit(train_frame, validation_frame)
    registry.publish(model, model.manifest())
    return model.manifest()
