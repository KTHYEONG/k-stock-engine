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

from legacy.stocks.data.contracts import DatasetSnapshot
from legacy.stocks.ml.contracts import NetAlphaTrainingRequest
from legacy.stocks.ml.data import compose_net_alpha_training_data
from legacy.stocks.ml.result_ledger import (
    CostRunContext,
    MlRunContext,
    ResultLedgerObserver,
)
from legacy.stocks.ml.training import train_net_alpha_model
from legacy.stocks.observability.contracts import RunDiagnostics
from legacy.stocks.research.artifacts import ModelArtifactRegistry
from legacy.stocks.research.models import ModelManifest

logger = logging.getLogger("stocks.workflows.train_model")


def train_model(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
    *,
    observer: ResultLedgerObserver | None = None,
    diagnostics: RunDiagnostics | None = None,
) -> ModelManifest:
    """Train the net-alpha mainline and publish a champion or ``NO_TRADE`` artifact.

    ``train_model`` is side-effect free by default: it never writes the result
    ledger. An optional ``observer`` (the CLI supplies the real
    ``MlResultLedger``) records the terminal outcome; ledger write failures are
    logged and never change the published artifact.
    """
    decision_time = _decision_time(snapshot.frame)
    data = compose_net_alpha_training_data(
        snapshot,
        decision_time,
        candidate_horizon_sessions=request.candidate_horizon_sessions,
    )
    context: MlRunContext | None = None
    if observer is not None:
        cost_context = CostRunContext(
            cost_schedule_kind="request",
            has_liquidity_model=request.liquidity_model is not None,
        )
        context = MlRunContext.from_cli(
            request=request,
            snapshot_id=snapshot.manifest.content_hash or "n/a",
            data=data,
            cost_context=cost_context,
            started_at=_utc_now(),
        )
    try:
        manifest = train_net_alpha_model(data, registry, request, diagnostics=diagnostics)
    except Exception as exc:
        if observer is not None and context is not None:
            try:
                observer.record_failed(context, "train_net_alpha_model", exc)
            except Exception:
                logger.warning("result ledger failed to record failure", exc_info=True)
        raise
    if observer is not None and context is not None:
        try:
            observer.record_completed(context, manifest, registry)
        except Exception:
            logger.warning("result ledger failed to record completion", exc_info=True)
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
