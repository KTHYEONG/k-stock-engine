"""ETF backtest workflow: frames + universe -> ``EtfBacktestResult``."""
from __future__ import annotations

import polars as pl

from src.etfs.backtesting.engine import EtfBacktester, EtfSimulationConfig
from src.etfs.backtesting.results import EtfBacktestResult
from src.etfs.domain.universe import EtfUniverse


def run_backtest(
    index_df: pl.DataFrame,
    etf_df: pl.DataFrame,
    universe: EtfUniverse,
    target_market: str = "KOSPI",
    config: EtfSimulationConfig | None = None,
) -> list[EtfBacktestResult]:
    """Run the ETF switching backtest and return typed results.

    Rendering/output decisions belong to the CLI, not this workflow.
    """
    backtester = EtfBacktester(index_df, etf_df, config=config)
    return backtester.run(universe, target_market=target_market)
