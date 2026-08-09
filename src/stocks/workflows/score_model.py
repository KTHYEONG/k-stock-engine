"""Stock model-scoring workflow: artifact -> prediction scores."""
from __future__ import annotations

import polars as pl

from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry, PredictionRequest
from src.stocks.workflows.contracts import ScoringRequest


def score_model(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: ScoringRequest,
) -> pl.DataFrame:
    """Load a stock artifact and score ``snapshot.frame`` with it."""
    manifest = snapshot.manifest
    prediction = PredictionRequest(
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        decision_time=request.decision_time,
    )
    loaded = registry.load(request.artifact_id, prediction)
    scored = loaded.model.predict(snapshot.frame)
    if scored.is_empty():
        raise ValueError("no rows scored")
    return scored
