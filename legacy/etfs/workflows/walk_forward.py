"""ETF walk-forward workflow: dataset -> stability assessment."""
from __future__ import annotations

from legacy.etfs.backtesting.engine import EtfSimulationConfig
from legacy.etfs.data.contracts import EtfDataset
from legacy.etfs.domain.universe import EtfUniverse
from legacy.etfs.research.analysis import StabilityReport, assess_stability
from legacy.etfs.research.search_space import WalkForwardFold
from legacy.etfs.research.walk_forward import (
    WalkForwardReport,
    build_walk_forward_folds,
    run_walk_forward,
)
from legacy.etfs.strategies.index_switch_v1 import IndexSwitchParams


def run_walk_forward_workflow(
    dataset: EtfDataset,
    universe: EtfUniverse,
    params: IndexSwitchParams,
    *,
    n_folds: int = 3,
    config: EtfSimulationConfig | None = None,
    target_market: str = "KOSPI",
) -> tuple[WalkForwardReport, StabilityReport]:
    """Run walk-forward evaluation and assess strategy stability."""
    folds = build_walk_forward_folds(dataset, n_folds=n_folds)
    report = run_walk_forward(
        dataset,
        universe,
        params,
        folds,
        config=config,
        target_market=target_market,
    )
    return report, assess_stability(report)


__all__ = ["WalkForwardFold", "build_walk_forward_folds", "run_walk_forward_workflow"]
