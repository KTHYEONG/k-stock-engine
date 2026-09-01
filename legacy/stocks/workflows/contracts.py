"""Stock workflow input contracts.

Workflows receive typed requests and validated dataset snapshots. CLI modules
construct these contracts from a validated repository result; they never
manufacture a fake manifest or choose fixture dates. Cost schedules and risk
budgets are explicit inputs, never hard-coded strategy constants. The training
contract is the canonical ``NetAlphaTrainingRequest`` from ``legacy.stocks.ml``;
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
    """Input contract for the portfolio-simulation workflow.

    ``policy_profile_id`` and ``no_trade_band_bps`` are explicit policy-profile
    confirmations: when supplied they must exactly match the immutable
    artifact's selected ``policy_profile`` or ``simulate_portfolio`` raises
    ``ValueError``. The portfolio caps (``top_k``, ``max_single_weight``,
    ``max_exposure``, ``participation_limit``) are validated against the
    artifact's stored portfolio fingerprint, so the independent backtester can
    never silently use a different top-k/exposure/band than the OOF that
    selected the policy.
    """

    artifact_id: str
    decision_time: datetime
    top_k: int = 5
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    participation_limit: float = 0.01
    portfolio_value: float = 100_000_000.0
    initial_cash: float = 100_000_000.0
    policy_profile_id: str | None = None
    no_trade_band_bps: float | None = None
    cost_schedule: CostSchedule | None = None
    stress_cost_schedule: CostSchedule | None = None
