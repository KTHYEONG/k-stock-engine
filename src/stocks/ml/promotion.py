"""Training promotion and publication extracted from training.py.

``publish_training_outcome`` delegates to the existing promotion logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.stocks.observability.contracts import RunDiagnostics

if TYPE_CHECKING:
    from src.stocks.research.models import ModelManifest


def publish_training_outcome(
    context: object,
    diagnostics: RunDiagnostics,
) -> ModelManifest:
    """Publish the training outcome (champion or NO_TRADE).

    This is a thin facade that delegates to the existing promotion logic
    in ``training.py``.

    Parameters
    ----------
    context:
        The promotion context carrying all necessary training state.
    diagnostics:
        Diagnostic sink for checkpoint emission.

    Returns
    -------
    ModelManifest
        The published model manifest.
    """
    from src.stocks.ml.training import _run_discovery_and_publish

    return _run_discovery_and_publish(
        registry=context.registry,  # type: ignore[attr-defined]
        data=context.data,  # type: ignore[attr-defined]
        request=context.request,  # type: ignore[attr-defined]
        frame=context.frame,  # type: ignore[attr-defined]
        pre_holdout=context.pre_holdout,  # type: ignore[attr-defined]
        holdout=context.holdout,  # type: ignore[attr-defined]
        folds=context.folds,  # type: ignore[attr-defined]
        learner_columns=context.learner_columns,  # type: ignore[attr-defined]
        schema=context.schema,  # type: ignore[attr-defined]
        telemetry=context.telemetry,  # type: ignore[attr-defined]
        schema_hash=context.schema_hash,  # type: ignore[attr-defined]
        universe_policy_hash=context.universe_policy_hash,  # type: ignore[attr-defined]
        oof_cache=context.oof_cache,  # type: ignore[attr-defined]
    )
