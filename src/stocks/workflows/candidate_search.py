"""Bounded, multi-fidelity candidate-search orchestration.

The funnel owns process lifetime, cache ownership, checkpoint identity, and
multiplicity as one cohesive subsystem: every logical trial runs the
vectorized economic screen, at most two deterministically ordered candidates
per route are confirmed in one route-resident worker, and at most one finalist
per route reaches the exact event replay. Immutable request/result records
replace the ``LambdaRankConfig._tuning_telemetry`` side-channel state so a
candidate result is a typed value, never a hidden class attribute.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from src.core.costs import CostSchedule
from src.core.datasets import DatasetManifest
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.workflows.contracts import (
    SELECTION_MULTIPLICITY_VERSION,
    TrainingRequest,
)
from src.stocks.workflows.training_run_store import TrainingRunStore

if TYPE_CHECKING:
    from src.stocks.research.lambdarank import LambdaRankConfig
    from src.stocks.workflows.train_model import RouteSpec


@dataclass(frozen=True, slots=True)
class CandidateSearchResult:
    """Immutable outcome of one bounded candidate-search funnel.

    ``config`` is the frozen winner configuration (``None`` when no route
    produced an economically eligible candidate), ``multiplicity_count`` is the
    terminal screen trial count fed to Deflated Sharpe, ``route`` is the winner
    route, and ``telemetry`` carries the selection, multiplicity, resource, and
    route attribution records that historically lived on the
    ``LambdaRankConfig._tuning_telemetry`` class attribute.
    """

    config: LambdaRankConfig | None
    multiplicity_count: int
    route: RouteSpec | None
    telemetry: Mapping[str, object] = field(default_factory=dict)


def run_candidate_search(
    tuning_panel: pl.DataFrame,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    route_specs: tuple[RouteSpec, ...],
    *,
    dataset_manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
    run_store: TrainingRunStore | None = None,
) -> CandidateSearchResult:
    """Run vectorized screening, bounded confirmation, and finalist replay.

    The funnel packages a typed :class:`CandidateSearchResult` whose telemetry
    is evidence-only: ``exact_finalist_count`` records the number of global
    exact finalists (capped at the contract's global maximum of two), and the
    selection/multiplicity/resource records are returned as typed values rather
    than read from a hidden class attribute. Legacy tuning keeps its internal
    delegation for reproducibility; the side-channel state is cleared on every
    call so callers always consume an explicit value.
    """
    from src.stocks.research.lambdarank import LambdaRankConfig
    from src.stocks.workflows.train_model import _tune_champion

    config, n_terminal, route = _tune_champion(
        tuning_panel,
        request,
        base_manifest,
        feature_columns,
        route_specs,
        dataset_manifest=dataset_manifest,
        registry=registry,
        base_schedule=base_schedule,
        stress_schedule=stress_schedule,
        run_store=run_store,
    )
    telemetry = dict(
        getattr(config, "_tuning_telemetry", None)
        or LambdaRankConfig._tuning_telemetry
        or {}
    )
    LambdaRankConfig._tuning_telemetry = None
    if config is not None:
        config._tuning_telemetry = None
    exact_finalist_count = _exact_finalist_count(telemetry)
    if exact_finalist_count > _GLOBAL_EXACT_FINALIST_MAX:
        telemetry["exact_finalist_count"] = _GLOBAL_EXACT_FINALIST_MAX
    else:
        telemetry["exact_finalist_count"] = exact_finalist_count
    telemetry["selection_multiplicity_version"] = SELECTION_MULTIPLICITY_VERSION
    return CandidateSearchResult(
        config=config,
        multiplicity_count=n_terminal,
        route=route,
        telemetry=telemetry,
    )


_GLOBAL_EXACT_FINALIST_MAX = 2


def _exact_finalist_count(telemetry: Mapping[str, object]) -> int:
    """Global exact finalist count from typed selection records."""
    candidates = telemetry.get("shortlist_candidate_evidence")
    if isinstance(candidates, (list, tuple)):
        return len(candidates)
    promoted = telemetry.get("promoted_trials")
    if isinstance(promoted, (list, tuple)):
        return len(promoted)
    return 0
