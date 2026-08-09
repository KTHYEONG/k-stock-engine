"""Stock workflow input contracts.

Workflows receive typed requests and validated dataset snapshots. CLI modules
construct these contracts from a validated repository result; they never
manufacture a fake manifest or choose fixture dates. Cost schedules and risk
budgets are explicit inputs, never hard-coded strategy constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.costs import CostSchedule


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Input contract for the model-training workflow."""

    artifact_id: str
    n_folds: int = 3
    embargo_sessions: int = 5
    holdout_sessions: int = 0
    seed: int = 42
    n_bootstrap: int = 200
    bootstrap_alpha: float = 0.05
    top_k: int = 5
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    participation_limit: float = 0.01
    portfolio_value: float = 100_000_000.0
    initial_cash: float = 100_000_000.0
    base_cost_schedule: CostSchedule | None = None
    stress_cost_schedule: CostSchedule | None = None


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
