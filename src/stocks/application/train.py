"""Stock training application: dataset -> folds -> model -> artifact."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.core.instruments import AssetKind
from src.stocks.labels.definitions import LabelDefinition
from src.stocks.ml.artifacts import ModelArtifactRegistry
from src.stocks.ml.dataset import DatasetManifest
from src.stocks.ml.models import DeterministicBaseline, ModelManifest
from src.stocks.ml.splits import PurgedWalkForward

STOCK_FEATURE_SET = "stock_alpha_v1"
LABEL_NAME = "fwd_ret_5d"


def run_training(
    dataset: pl.DataFrame,
    manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    artifact_id: str,
    n_folds: int = 3,
) -> str:
    """Train a deterministic baseline and publish an immutable artifact.

    The ``migration_wiring`` contract requires this application to publish a
    trained model via ``registry.publish``.
    """
    if manifest.asset_kind is not AssetKind.STOCK:
        raise ValueError(f"train.py only accepts stock datasets, got {manifest.asset_kind.value}")

    label = LabelDefinition(
        name=LABEL_NAME,
        entry_field="close",
        exit_field="close",
        horizon_sessions=manifest.label_horizon_sessions,
    )
    label_frame = label.apply(dataset)
    if LABEL_NAME not in label_frame.columns:
        raise ValueError("label computation failed")

    splitter = PurgedWalkForward(
        n_folds=n_folds,
        label_horizon_sessions=manifest.label_horizon_sessions,
        embargo_sessions=max(0, manifest.label_horizon_sessions),
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
            artifact_id=artifact_id,
            asset_kind=AssetKind.STOCK,
            feature_set=STOCK_FEATURE_SET,
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
    return artifact_id


def main(args: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train a stock baseline model artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parsed = parser.parse_args(args)

    dataset = pl.read_parquet(parsed.dataset)
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="fixture",
        provider_version="fixture",
        universe_policy_version="v1",
        universe_policy_hash="universe-v1",
        feature_set=STOCK_FEATURE_SET,
        feature_set_hash="features-v1",
        label_definition=LABEL_NAME,
        label_horizon_sessions=5,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 12, 31, tzinfo=UTC),
        generated_time=datetime.now(UTC),
        row_count=dataset.height,
    )
    registry = ModelArtifactRegistry(parsed.registry)
    run_training(dataset, manifest, registry, parsed.artifact_id)
    return 0
