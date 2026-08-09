"""Stock portfolio-simulation workflow: artifact -> allocations -> simulator."""
from __future__ import annotations

from src.core.costs import CostModel
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
    """Score the snapshot, apply the allocation policy, and simulate fills."""
    scored = score_model(
        snapshot,
        registry,
        ScoringRequest(artifact_id=request.artifact_id, decision_time=request.decision_time),
    )
    policy = AllocationPolicy(
        top_k=request.top_k,
        max_single_weight=request.max_single_weight,
        max_exposure=request.max_exposure,
    )
    simulator = StockSimulator(CostModel(commission_rate=0.00015, tax_rate=0.0023))
    return simulator.simulate(scored, policy, AssetKind.STOCK, price_frame=scored)
