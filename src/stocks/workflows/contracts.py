"""Stock workflow input contracts.

Workflows receive typed requests and validated dataset snapshots. CLI modules
construct these contracts from a validated repository result; they never
manufacture a fake manifest or choose fixture dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Input contract for the model-training workflow."""

    artifact_id: str
    n_folds: int = 3
    embargo_sessions: int = 5


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
