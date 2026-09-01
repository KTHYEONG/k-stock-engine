"""ETF optimize CLI: parse args, load dataset, run optimization."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from legacy.etfs.data.repositories import EtfDatasetRepository
from legacy.etfs.domain.universe import KOSDAQ_ETF_UNIVERSE, KOSPI_ETF_UNIVERSE
from legacy.etfs.research.optimization_runner import OptimizationRequest
from legacy.etfs.research.search_space import SearchSpace
from legacy.etfs.workflows.optimize import optimize
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("etfs.cli.optimize")

MARKETS = {"KOSPI": KOSPI_ETF_UNIVERSE, "KOSDAQ": KOSDAQ_ETF_UNIVERSE}


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ETF IndexSwitchV1 optimization")
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--index-dataset", required=True)
    parser.add_argument("--etf-dataset", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--feature-set", default="etf_switch_v1")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=3)
    parsed = parser.parse_args(args)

    from datetime import UTC, datetime

    repository = EtfDatasetRepository(ParquetDatasetStore(parsed.dataset_root))
    dataset = repository.read(
        parsed.index_dataset, parsed.etf_dataset, parsed.feature_set, datetime.now(UTC)
    )
    request = OptimizationRequest(
        search_space=SearchSpace(),
        n_folds=parsed.n_folds,
        max_trials=parsed.n_trials,
        target_market=parsed.market,
    )
    report = optimize(dataset, MARKETS[parsed.market], request)
    best = report.best_params
    logger.info(
        "market=%s trials=%d best_ret=%.2f%% best_mdd=%.2f%% params=%s",
        parsed.market,
        report.trials,
        report.best_mean_return_pct,
        report.best_mean_mdd_pct,
        {
            "macro_ema_period": best.macro_ema_period,
            "fast_ema_period": best.fast_ema_period,
            "ibs_entry": best.ibs_entry,
            "ibs_exit": best.ibs_exit,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
