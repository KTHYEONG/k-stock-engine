"""ETF research: walk-forward evaluation runner.

Runs the declared ETF strategy over contiguous time folds of the dataset so
optimization and evaluation share identical folds, simulator configuration, and
cost assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from legacy.etfs.backtesting.engine import EtfBacktester, EtfSimulationConfig
from legacy.etfs.backtesting.results import EtfBacktestResult
from legacy.etfs.data.contracts import EtfDataset
from legacy.etfs.domain.universe import EtfUniverse
from legacy.etfs.research.search_space import WalkForwardFold
from legacy.etfs.strategies.index_switch_v1 import IndexSwitchParams


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Per-fold outcome plus the ordered folds used for evaluation."""

    fold: WalkForwardFold
    result: EtfBacktestResult


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    """Aggregated walk-forward evaluation over all folds."""

    folds: list[WalkForwardFold]
    results: list[WalkForwardResult]
    mean_return_pct: float
    mean_mdd_pct: float


def build_walk_forward_folds(
    dataset: EtfDataset,
    *,
    n_folds: int,
    date_column: str = "date",
) -> list[WalkForwardFold]:
    """Split the index frame into ``n_folds`` contiguous evaluation windows."""
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    dates = sorted(dataset.index_frame[date_column].unique().to_list())
    if len(dates) < n_folds:
        raise ValueError(
            f"cannot create {n_folds} folds from {len(dates)} sessions"
        )
    window = max(1, len(dates) // n_folds)
    folds: list[WalkForwardFold] = []
    for k in range(n_folds):
        start = k * window
        end = len(dates) if k == n_folds - 1 else min(len(dates), start + window)
        train_end = dates[start] if start > 0 else None
        folds.append(
            WalkForwardFold(
                train_end=train_end,
                validation_start=dates[start],
                validation_end=dates[end - 1],
                eval_year=dates[start].year,
            )
        )
    return folds


def run_walk_forward(
    dataset: EtfDataset,
    universe: EtfUniverse,
    params: IndexSwitchParams,
    folds: list[WalkForwardFold],
    config: EtfSimulationConfig | None = None,
    target_market: str = "KOSPI",
) -> WalkForwardReport:
    """Evaluate the strategy on each fold's validation window."""
    config = config or EtfSimulationConfig()
    results: list[WalkForwardResult] = []
    for fold in folds:
        window = dataset.index_frame.filter(
            pl.col("date") >= fold.validation_start
        ).filter(pl.col("date") <= fold.validation_end)
        window_etf = _matching_etf_rows(dataset.etf_frame, window)
        if window.is_empty() or window_etf.is_empty():
            continue
        backtester = EtfBacktester(window, window_etf, config=config, params=params)
        fold_results = backtester.run(universe, target_market=target_market)
        if not fold_results:
            continue
        results.append(WalkForwardResult(fold=fold, result=fold_results[0]))

    if not results:
        return WalkForwardReport(folds=list(folds), results=[], mean_return_pct=0.0, mean_mdd_pct=0.0)
    mean_return = sum(r.result.total_return_pct for r in results) / len(results)
    mean_mdd = sum(r.result.mdd_pct for r in results) / len(results)
    return WalkForwardReport(
        folds=list(folds),
        results=results,
        mean_return_pct=mean_return,
        mean_mdd_pct=mean_mdd,
    )


def _matching_etf_rows(etf_frame: pl.DataFrame, window: pl.DataFrame) -> pl.DataFrame:
    if "date" not in etf_frame.columns or "date" not in window.columns:
        return etf_frame
    lo, hi = window["date"].min(), window["date"].max()
    return etf_frame.filter(pl.col("date") >= lo).filter(pl.col("date") <= hi)
