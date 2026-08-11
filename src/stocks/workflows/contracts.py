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
    """Input contract for the model-training workflow.

    ``n_folds`` and ``embargo_sessions`` are honored by the outer walk-forward;
    ``holdout_sessions`` is the forward-holdout block size (0 defers to the
    fixed 252-session contract). ``n_bootstrap`` bounds the moving-block
    bootstrap resamples and ``seed`` fixes every stochastic step.
    """

    artifact_id: str
    n_folds: int = 3
    embargo_sessions: int = 5
    holdout_sessions: int = 0
    seed: int = 42
    n_bootstrap: int = 200
    bootstrap_alpha: float = 0.05
    optuna_trials: int = 80
    max_rss_mib: int | None = None
    top_k: int = 5
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    participation_limit: float = 0.01
    portfolio_value: float = 100_000_000.0
    initial_cash: float = 100_000_000.0
    base_cost_schedule: CostSchedule | None = None
    stress_cost_schedule: CostSchedule | None = None

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be positive")
        if self.embargo_sessions < 0:
            raise ValueError("embargo_sessions must be non-negative")
        if self.holdout_sessions < 0:
            raise ValueError("holdout_sessions must be non-negative")
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        if not 0.0 < self.bootstrap_alpha < 1.0:
            raise ValueError("bootstrap_alpha must be in (0, 1)")
        if self.optuna_trials < 1:
            raise ValueError("optuna_trials must be positive")
        if self.max_rss_mib is not None and self.max_rss_mib <= 0:
            raise ValueError("max_rss_mib must be positive when supplied")


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
