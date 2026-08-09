"""ETF backtest application command."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.etfs.domain.universe import KOSDAQ_ETF_UNIVERSE, KOSPI_ETF_UNIVERSE
from src.etfs.simulation.runner import EtfBacktester

MARKETS = {"KOSPI": KOSPI_ETF_UNIVERSE, "KOSDAQ": KOSDAQ_ETF_UNIVERSE}


def run_backtest(index_df: pl.DataFrame, etf_df: pl.DataFrame, market: str = "KOSPI") -> list[list[dict[str, float | int]]]:
    backtester = EtfBacktester(index_df, etf_df)
    universe = MARKETS[market]
    results = backtester.run(universe, target_market=market)
    return [res.trades for res in results]


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ETF IndexSwitchV1 backtest")
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--etf", required=True, type=Path)
    parsed = parser.parse_args(args)

    index_df = pl.read_parquet(parsed.index)
    etf_df = pl.read_parquet(parsed.etf)
    run_backtest(index_df, etf_df, market=parsed.market)
    return 0
