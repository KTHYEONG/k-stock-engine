"""ETF optimization workflow: dataset -> optimization report."""
from __future__ import annotations

from src.etfs.data.contracts import EtfDataset
from src.etfs.domain.universe import EtfUniverse
from src.etfs.research.optimization_runner import (
    OptimizationReport,
    OptimizationRequest,
    run_optimization,
)


def optimize(
    dataset: EtfDataset,
    universe: EtfUniverse,
    request: OptimizationRequest,
) -> OptimizationReport:
    """Search strategy parameters on the same folds used for evaluation."""
    return run_optimization(dataset, universe, request)
