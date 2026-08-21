"""Horizon discovery and economic selection extracted from training.py.

``HorizonDiscovery`` is the immutable outcome of per-horizon OOF discovery.
``discover_horizons`` is the public entry point for the horizon discovery funnel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.stocks.observability.contracts import RunDiagnostics

if TYPE_CHECKING:
    from src.stocks.ml.contracts import (
        HorizonOOFDiagnostic,
    )
    from src.stocks.ml.data import HorizonOutcomeCoverage
    from src.stocks.ml.execution_replay import ExecutionReplayEvidence
    from src.stocks.ml.fitting import OofCache
    from src.stocks.ml.horizons import HorizonOOFEvidence


@dataclass(frozen=True, slots=True)
class HorizonDiscovery:
    """Immutable outcome of per-horizon OOF discovery.

    ``evidence`` are the ``(horizon, profile)`` candidates that cleared the
    fold-coverage, cohort, missing-realized, and Rank-IC pre-gates;
    ``diagnostics`` retain the typed per-horizon OOF diagnostics for every
    candidate horizon.
    """

    evidence: tuple[HorizonOOFEvidence, ...]
    diagnostics: tuple[HorizonOOFDiagnostic, ...]
    oof_by_horizon: dict[int, tuple[Path, Path, list[float]]]
    dropout_reasons: dict[tuple[int, str], str] = field(default_factory=dict)
    execution_evidence_by_candidate: dict[
        tuple[int, str], ExecutionReplayEvidence
    ] = field(default_factory=dict)
    coverage_by_horizon: dict[int, HorizonOutcomeCoverage] = field(
        default_factory=dict
    )
    horizon_memory: dict[int, dict[str, object]] = field(default_factory=dict)
    oof_cache: OofCache | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    path_evaluation_count: int = 0
    path_evaluation_bound: int = 0


def discover_horizons(
    discovery_context: object,
    diagnostics: RunDiagnostics | None,
) -> HorizonDiscovery:
    """Discover optimal horizons through OOF evaluation.

    This is a thin wrapper that delegates to the existing discovery logic
    in ``training.py``.  The full implementation requires the complete
    training context and is not duplicated here.

    Parameters
    ----------
    discovery_context:
        The discovery context carrying folds, data, request, and cache.
    diagnostics:
        Diagnostic sink for checkpoint emission.

    Returns
    -------
    HorizonDiscovery
        The immutable discovery outcome.
    """
    context = cast(Any, discovery_context)
    if any(
        getattr(context, name, None) is None
        for name in ("pre_holdout", "folds", "learner_columns", "oof_cache")
    ):
        if hasattr(context, "fit_context"):
            from src.stocks.ml.fitting import fit_horizon_oof

            return cast(HorizonDiscovery, fit_horizon_oof(context.fit_context))
        return HorizonDiscovery((), (), {})

    from src.stocks.ml.training import _build_horizon_evidence

    return cast(
        HorizonDiscovery,
        _build_horizon_evidence(
        pre_holdout=context.pre_holdout,
        folds=context.folds,
        data=context.data,
        request=context.request,
        learner_columns=context.learner_columns,
        oof_cache=context.oof_cache,
        ),
    )
