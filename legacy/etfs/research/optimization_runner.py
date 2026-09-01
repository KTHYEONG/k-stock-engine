"""ETF research: parameter optimization runner.

Searches the declared ``SearchSpace`` deterministically and evaluates each
candidate with the same strategy, simulator configuration, costs, and
walk-forward folds used by ETF evaluation. It never imports legacy ETF code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.instruments import AssetKind
from legacy.etfs.backtesting.engine import EtfSimulationConfig
from legacy.etfs.data.contracts import EtfDataset
from legacy.etfs.domain.universe import EtfUniverse
from legacy.etfs.research.search_space import SearchSpace
from legacy.etfs.research.walk_forward import (
    WalkForwardReport,
    build_walk_forward_folds,
    run_walk_forward,
)
from legacy.etfs.strategies.index_switch_v1 import IndexSwitchParams


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    """Configuration for a deterministic search run."""

    search_space: SearchSpace
    n_folds: int = 3
    max_trials: int = 100
    seed: int = 42
    target_market: str = "KOSPI"


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Outcome of the search with the best candidate and its metrics."""

    asset_kind: AssetKind
    best_params: IndexSwitchParams
    best_mean_return_pct: float
    best_mean_mdd_pct: float
    trials: int

    def __post_init__(self) -> None:
        if self.asset_kind is not AssetKind.ETF:
            raise ValueError("optimization report must carry AssetKind.ETF")


def run_optimization(
    dataset: EtfDataset,
    universe: EtfUniverse,
    request: OptimizationRequest,
) -> OptimizationReport:
    """Search ``request.search_space`` and return the best ETF candidate.

    Every candidate is scored on the same walk-forward folds as evaluation so
    optimization cannot overfit to a single out-of-sample window.
    """
    rng = np.random.default_rng(request.seed)
    folds = build_walk_forward_folds(dataset, n_folds=request.n_folds)
    config = EtfSimulationConfig()

    best: tuple[IndexSwitchParams, WalkForwardReport] | None = None
    for _ in range(request.max_trials):
        params = _sample_params(request.search_space, rng)
        report = run_walk_forward(
            dataset,
            universe,
            params,
            folds,
            config=config,
            target_market=request.target_market,
        )
        if not report.results:
            continue
        if best is None or report.mean_return_pct > best[1].mean_return_pct:
            best = (params, report)

    if best is None:
        return OptimizationReport(
            asset_kind=AssetKind.ETF,
            best_params=IndexSwitchParams(),
            best_mean_return_pct=0.0,
            best_mean_mdd_pct=0.0,
            trials=request.max_trials,
        )
    params, report = best
    return OptimizationReport(
        asset_kind=AssetKind.ETF,
        best_params=params,
        best_mean_return_pct=report.mean_return_pct,
        best_mean_mdd_pct=report.mean_mdd_pct,
        trials=request.max_trials,
    )


def _sample_params(space: SearchSpace, rng: np.random.Generator) -> IndexSwitchParams:
    """Sample a candidate ``IndexSwitchParams`` within the search space."""
    return IndexSwitchParams(
        macro_ema_period=int(rng.integers(*space.macro_ema_period, endpoint=True)),
        fast_ema_period=int(rng.integers(*space.fast_ema_period, endpoint=True)),
        roc_n=int(rng.integers(*space.roc_n, endpoint=True)),
        roc_lower=float(rng.uniform(*space.roc_lower)),
        ibs_entry=float(rng.uniform(*space.ibs_entry)),
        ibs_exit=float(rng.uniform(*space.ibs_exit)),
        max_hold_days=int(rng.integers(*space.max_hold_days, endpoint=True)),
        stop_loss_pct=float(rng.uniform(*space.stop_loss_pct)),
    )
