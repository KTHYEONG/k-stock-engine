"""ETF backtest CLI: parse args, load frames, render workflow results."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from legacy.etfs.backtesting.engine import EtfSimulationConfig
from legacy.etfs.domain.universe import KOSDAQ_ETF_UNIVERSE, KOSPI_ETF_UNIVERSE
from legacy.etfs.settings import DEFAULT_ETF
from legacy.etfs.workflows.backtest import run_backtest

logger = logging.getLogger("etfs.cli.backtest")

MARKETS = {"KOSPI": KOSPI_ETF_UNIVERSE, "KOSDAQ": KOSDAQ_ETF_UNIVERSE}


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ETF IndexSwitchV1 backtest")
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--etf", required=True, type=Path)
    parsed = parser.parse_args(args)

    index_df = pl.read_parquet(parsed.index)
    etf_df = pl.read_parquet(parsed.etf)
    settings = DEFAULT_ETF
    config = EtfSimulationConfig(
        initial_balance=settings.initial_balance,
        fee_rate=settings.fee_rate,
        capital_use=settings.capital_use,
    )
    results = run_backtest(
        index_df, etf_df, MARKETS[parsed.market], target_market=parsed.market, config=config
    )
    for result in results:
        logger.info(
            "market=%s ret=%.2f%% mdd=%.2f%% trades=%d pf=%.3f",
            result.market,
            result.total_return_pct,
            result.mdd_pct,
            result.total_trades,
            result.profit_factor,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
