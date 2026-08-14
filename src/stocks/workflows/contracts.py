"""Stock workflow input contracts.

Workflows receive typed requests and validated dataset snapshots. CLI modules
construct these contracts from a validated repository result; they never
manufacture a fake manifest or choose fixture dates. Cost schedules and risk
budgets are explicit inputs, never hard-coded strategy constants. The training
contract is the canonical ``NetAlphaTrainingRequest`` from ``src.stocks.ml``;
there is no legacy LambdaRank/Optuna training request.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.costs import CostSchedule

COMPUTE_PLAN_VERSION = "sub10-refit-v1"

SELECTION_MULTIPLICITY_VERSION = "selection-multiplicity-global-count-v1"


@dataclass(frozen=True, slots=True)
class ScoringRequest:
    """Input contract for the model-scoring workflow."""

    artifact_id: str
    decision_time: datetime


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """Input contract for the portfolio-simulation workflow."""

    artifact_id: str
    decision_time: datetime
    top_k: int = 5
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    participation_limit: float = 0.01
    portfolio_value: float = 100_000_000.0
    initial_cash: float = 100_000_000.0
    cost_schedule: CostSchedule | None = None
    stress_cost_schedule: CostSchedule | None = None
