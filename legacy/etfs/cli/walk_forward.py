"""ETF walk-forward CLI: parse args, load dataset, assess stability."""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from legacy.etfs.data.repositories import EtfDatasetRepository
from legacy.etfs.domain.universe import KOSDAQ_ETF_UNIVERSE, KOSPI_ETF_UNIVERSE
from legacy.etfs.strategies.index_switch_v1 import IndexSwitchParams
from legacy.etfs.workflows.walk_forward import run_walk_forward_workflow
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("etfs.cli.walk_forward")

MARKETS = {"KOSPI": KOSPI_ETF_UNIVERSE, "KOSDAQ": KOSDAQ_ETF_UNIVERSE}


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ETF IndexSwitchV1 walk-forward")
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--index-dataset", required=True)
    parser.add_argument("--etf-dataset", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--feature-set", default="etf_switch_v1")
    parser.add_argument("--n-folds", type=int, default=3)
    parsed = parser.parse_args(args)

    repository = EtfDatasetRepository(ParquetDatasetStore(parsed.dataset_root))
    dataset = repository.read(
        parsed.index_dataset, parsed.etf_dataset, parsed.feature_set, datetime.now(UTC)
    )
    report, stability = run_walk_forward_workflow(
        dataset,
        MARKETS[parsed.market],
        IndexSwitchParams(),
        n_folds=parsed.n_folds,
        target_market=parsed.market,
    )
    logger.info(
        "market=%s folds=%d mean_ret=%.2f%% mean_mdd=%.2f%% passed=%s",
        parsed.market,
        len(report.results),
        report.mean_return_pct,
        report.mean_mdd_pct,
        stability.passed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
