"""Stock portfolio-simulation workflow: artifact -> allocations -> ledger simulator."""
from __future__ import annotations

from src.core.costs import default_base_schedule
from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.allocation_policy import AllocationPolicy
from src.stocks.trading.simulator import SimResult, StockSimulator
from src.stocks.workflows.contracts import ScoringRequest, SimulationRequest
from src.stocks.workflows.score_model import score_model


def simulate_portfolio(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: SimulationRequest,
) -> SimResult:
    """Score the snapshot, apply the constrained policy, and run the ledger."""
    scored = score_model(
        snapshot,
        registry,
        ScoringRequest(artifact_id=request.artifact_id, decision_time=request.decision_time),
    )
    policy = AllocationPolicy(
        top_k=request.top_k,
        max_single_weight=request.max_single_weight,
        max_exposure=request.max_exposure,
        participation_limit=request.participation_limit,
        portfolio_value=request.portfolio_value,
    )
    simulator = StockSimulator(
        cost_schedule=request.cost_schedule or default_base_schedule(),
        initial_cash=request.initial_cash,
        adtv_participation_limit=request.participation_limit,
        stress_schedule=request.stress_cost_schedule,
    )
    return simulator.simulate(scored, policy, AssetKind.STOCK)
