"""Stock model-scoring workflow: artifact -> point-in-time features -> scores."""
from __future__ import annotations

import polars as pl

from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import CANONICAL_FEATURE_SET
from src.stocks.ml.features import (
    build_model_features,
    stock_net_alpha_v1_roles,
)
from src.stocks.research.artifacts import ModelArtifactRegistry, PredictionRequest
from src.stocks.research.datasets import research_eligible_frame
from src.stocks.research.features import build_features, phase1_allowlist
from src.stocks.workflows.contracts import ScoringRequest


def score_model(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: ScoringRequest,
) -> pl.DataFrame:
    """Load a stock artifact, build its feature panel, and score it."""
    manifest = snapshot.manifest
    prediction = PredictionRequest(
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        decision_time=request.decision_time,
    )
    loaded = registry.load(request.artifact_id, prediction)
    gated = _drop_label_columns(research_eligible_frame(snapshot.frame))
    if manifest.feature_set == CANONICAL_FEATURE_SET:
        feature_frame, _model_columns = build_model_features(
            gated, stock_net_alpha_v1_roles()
        )
    elif manifest.feature_set == "stock_alpha_v2":
        feature_frame = gated
    else:
        feature_frame = build_features(gated, phase1_allowlist())
    scored = loaded.model.predict(feature_frame)
    if scored.is_empty():
        raise ValueError("no rows scored")
    return scored


def _drop_label_columns(frame: pl.DataFrame) -> pl.DataFrame:
    from src.stocks.research.labels import (
        LABEL_AVAILABLE_COLUMN,
        RELEVANCE_COLUMN,
        RESIDUAL_O2O_LABEL,
    )

    drops = [
        c
        for c in frame.columns
        if c.startswith(("target_", "label_", "residual_", "relevance_"))
        or c in (LABEL_AVAILABLE_COLUMN, RELEVANCE_COLUMN, RESIDUAL_O2O_LABEL, "fwd_ret_5d")
    ]
    return frame.drop(drops)
