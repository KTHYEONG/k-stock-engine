"""IndexSwitchV1: the versioned ETF switching rule strategy.

This is a faithful port of the legacy ETFStrategyEngine (IBS & Price Action)
signal logic. Its index inputs, warm-up requirement, and entry/exit convention
are declared as a configuration artifact.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class IndexSwitchParams:
    """Configuration artifact for the versioned switch strategy."""

    macro_ema_period: int = 120
    fast_ema_period: int = 20
    roc_n: int = 2
    roc_lower: float = -0.02
    ibs_entry: float = 0.15
    ibs_exit: float = 0.80
    max_hold_days: int = 3
    stop_loss_pct: float = 0.10

    @property
    def required_warmup(self) -> int:
        return self.macro_ema_period + 20


class IndexSwitchV1:
    """Deterministic IBS & price-action switch signal generator."""

    name = "IndexSwitchV1"

    def __init__(self, params: IndexSwitchParams | None = None):
        self.params = params or IndexSwitchParams()

    def generate_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """Vectorized signal generation; ``signal_trigger`` is 1/0/-1.

        Mirrors the legacy ``ETFStrategyEngine.generate_signal`` exactly so
        fixture parity can be asserted before legacy removal.
        """
        p = self.params
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        )
        df = df.with_columns(
            [
                ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8)).alias("ibs"),
                pl.col("close").ewm_mean(span=p.macro_ema_period, adjust=False).alias("macro_ema"),
                pl.col("close").ewm_mean(span=p.fast_ema_period, adjust=False).alias("fast_ema"),
                ((pl.col("close") - pl.col("close").shift(p.roc_n)) / pl.col("close").shift(p.roc_n)).alias("roc"),
            ]
        )
        bull_alignment = (pl.col("fast_ema") > pl.col("macro_ema")) & (pl.col("close") > pl.col("macro_ema"))
        bull_signal = bull_alignment & (pl.col("roc") < p.roc_lower) & (pl.col("ibs") < p.ibs_entry)
        bear_alignment = (pl.col("fast_ema") < pl.col("macro_ema")) & (pl.col("close") < pl.col("macro_ema"))
        bear_signal = bear_alignment & (pl.col("roc") > -p.roc_lower) & (pl.col("ibs") > (1.0 - p.ibs_entry))
        return df.with_columns(
            pl.when(bull_signal)
            .then(1)
            .when(bear_signal)
            .then(-1)
            .otherwise(0)
            .alias("signal_trigger")
        )
