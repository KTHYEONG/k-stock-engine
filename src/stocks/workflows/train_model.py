"""Stock model-training workflow adapter.

``train_model`` is a thin public adapter over the net-alpha mainline: it
composes ``NetAlphaResearchData`` from the immutable snapshot and immediately
delegates to ``train_net_alpha_model``. There is no v2 feature-set dispatch, no
LambdaRank fallback, and no Optuna search.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import polars as pl

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from src.stocks.ml.data import compose_net_alpha_training_data
from src.stocks.ml.training import train_net_alpha_model
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest

logger = logging.getLogger("stocks.workflows.train_model")


def train_model(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
) -> ModelManifest:
    """Train the net-alpha mainline and publish a champion or ``NO_TRADE`` artifact."""
    decision_time = _decision_time(snapshot.frame)
    data = compose_net_alpha_training_data(
        snapshot,
        decision_time,
        candidate_horizon_sessions=request.candidate_horizon_sessions,
    )
    manifest = train_net_alpha_model(data, registry, request)
    logger.info("published artifact %s (%s)", manifest.artifact_id, manifest.model_type)
    return manifest


def _decision_time(frame: pl.DataFrame) -> datetime:
    value = frame["available_time"].max() if "available_time" in frame.columns else None
    if value is None:
        raise ValueError("composed snapshot must carry an available_time column")
    if not isinstance(value, datetime):
        raise ValueError("available_time must be datetime")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
