"""ETF fill assumptions and backtest engine.

This module ports the legacy Numba engine to deterministic NumPy with identical
signal, fill, and ledger semantics. Fixture parity with the legacy engine is
asserted before the legacy ETF code may be removed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.etfs.backtesting.results import EtfBacktestResult
from src.etfs.domain.universe import EtfUniverse
from src.etfs.strategies.index_switch_v1 import IndexSwitchParams, IndexSwitchV1


@dataclass(frozen=True, slots=True)
class EtfSimulationConfig:
    """Declared fee/slippage and capital assumptions (configuration artifact)."""

    initial_balance: float = 10_000_000.0
    fee_rate: float = 0.00015
    capital_use: float = 0.99


def _backtest_engine(
    arr: np.ndarray,
    ibs_exit: float,
    max_hold_days: int,
    stop_loss_pct: float,
    initial_balance: float,
    fee_rate: float,
    capital_use: float,
) -> tuple[list[tuple[int, int, int, float, float, float, float, float]], float, np.ndarray]:
    """Deterministic NumPy port of the legacy Numba ETF engine.

    Column layout is identical to the legacy engine:
      0: sig (T-1), 1: idx_ibs (T), 2: idx_close (T),
      3-6: b1 o/h/l/c, 7-10: i1 o/h/l/c.
    """
    n = len(arr)
    balance = initial_balance
    equity_curve = np.zeros(n)
    in_position = False
    asset_type = 0
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0
    trades: list[tuple[int, int, int, float, float, float, float, float]] = []

    for i in range(n):
        c_sig = arr[i, 0]
        c_idx_ibs = arr[i, 1]

        b1_o, b1_h, b1_l, b1_c = arr[i, 3], arr[i, 4], arr[i, 5], arr[i, 6]
        i1_o, i1_h, i1_l, i1_c = arr[i, 7], arr[i, 8], arr[i, 9], arr[i, 10]

        bar_processed = False

        if in_position:
            c_o, _c_h, c_l, c_c = (b1_o, b1_h, b1_l, b1_c) if asset_type == 1 else (i1_o, i1_h, i1_l, i1_c)
            hold_days = i - entry_idx
            exit_triggered = False
            exit_price = 0.0
            stop_price = entry_price * (1.0 - stop_loss_pct)
            if c_o <= stop_price:
                exit_price = c_o
                exit_triggered = True
            elif c_l <= stop_price:
                exit_price = stop_price
                exit_triggered = True
            elif hold_days >= max_hold_days or (
                asset_type == 1 and c_idx_ibs >= ibs_exit
            ) or (asset_type == -1 and c_idx_ibs <= (1.0 - ibs_exit)):
                exit_price = c_c
                exit_triggered = True

            if exit_triggered:
                pnl = (exit_price - entry_price) * amount
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                balance += (amount * entry_price) + pnl
                trades.append((entry_idx, i, asset_type, entry_price, exit_price, pnl, amount, entry_fee_stored))
                in_position = False
                bar_processed = True

        if not in_position and not bar_processed and c_sig != 0:
            target_capital = balance * capital_use
            if c_sig == 1 and b1_o > 0:
                fill_price = b1_o
                amount = target_capital / fill_price
                entry_fee = target_capital * fee_rate
                balance -= target_capital + entry_fee
                entry_fee_stored = entry_fee
                in_position = True
                asset_type = 1
                entry_price = fill_price
                entry_idx = i
            elif c_sig == -1 and i1_o > 0:
                fill_price = i1_o
                amount = target_capital / fill_price
                entry_fee = target_capital * fee_rate
                balance -= target_capital + entry_fee
                entry_fee_stored = entry_fee
                in_position = True
                asset_type = -1
                entry_price = fill_price
                entry_idx = i

        if in_position:
            c_c = b1_c if asset_type == 1 else i1_c
            unrealized = (c_c - entry_price) * amount
            equity_curve[i] = balance + (amount * entry_price) + unrealized
        else:
            equity_curve[i] = balance

    if in_position and n > 0:
        last_idx = n - 1
        c_c = arr[last_idx, 6] if asset_type == 1 else arr[last_idx, 10]
        exit_price = c_c
        pnl = (exit_price - entry_price) * amount
        exit_fee = amount * exit_price * fee_rate
        pnl -= exit_fee
        balance += (amount * entry_price) + pnl
        trades.append((entry_idx, last_idx, asset_type, entry_price, exit_price, pnl, amount, entry_fee_stored))

    return trades, balance, equity_curve


class EtfBacktester:
    """Stateful ETF switching backtester over index and ETF price frames."""

    def __init__(
        self,
        index_df: pl.DataFrame,
        etf_df: pl.DataFrame,
        config: EtfSimulationConfig | None = None,
        params: IndexSwitchParams | None = None,
    ):
        self.index_df = index_df.sort("date")
        self.etf_df = etf_df.sort("date")
        self.config = config or EtfSimulationConfig()
        self.params = params or IndexSwitchParams()

    def run(self, universe: EtfUniverse, target_market: str = "KOSPI") -> list[EtfBacktestResult]:
        engine = IndexSwitchV1(self.params)
        results: list[EtfBacktestResult] = []

        mkt_idx = self.index_df.filter(pl.col("ticker") == target_market)
        if mkt_idx.is_empty():
            return results
        mkt_idx = mkt_idx.select(
            [
                pl.col("date"),
                pl.col("OPNPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("open"),
                pl.col("HGPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("high"),
                pl.col("LWPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("low"),
                pl.col("CLSPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("close"),
            ]
        ).filter(pl.col("close").is_not_null())
        if mkt_idx.is_empty():
            return results

        sig_df = engine.generate_signal(mkt_idx)

        def get_ohlc(dframe: pl.DataFrame, prefix: str) -> pl.DataFrame:
            exprs = [
                pl.col(c).cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias(f"{prefix}_{c}")
                for c in ["open", "high", "low", "close"]
            ]
            return dframe.select([pl.col("date"), *exprs])

        df_b1 = self.etf_df.filter(pl.col("ticker") == universe.bull_1x)
        df_i1 = self.etf_df.filter(pl.col("ticker") == universe.bear_1x)
        if df_b1.is_empty() or df_i1.is_empty():
            return results

        sim_df = (
            sig_df.join(get_ohlc(df_b1, "b1"), on="date", how="inner")
            .join(get_ohlc(df_i1, "i1"), on="date", how="inner")
        )

        input_df = sim_df.select(
            [
                pl.col("signal_trigger").shift(1).fill_null(0).alias("sig"),
                pl.col("ibs").fill_null(0.5).alias("idx_ibs"),
                pl.col("close").fill_null(0.0).alias("idx_close"),
                pl.col("b1_open").fill_null(0.0),
                pl.col("b1_high").fill_null(0.0),
                pl.col("b1_low").fill_null(0.0),
                pl.col("b1_close").fill_null(0.0),
                pl.col("i1_open").fill_null(0.0),
                pl.col("i1_high").fill_null(0.0),
                pl.col("i1_low").fill_null(0.0),
                pl.col("i1_close").fill_null(0.0),
            ]
        )
        arr = input_df.to_numpy().astype(np.float64)
        if len(arr) < 2:
            return results

        p = self.params
        trades, final_balance, equity_curve = _backtest_engine(
            arr, p.ibs_exit, p.max_hold_days, p.stop_loss_pct,
            self.config.initial_balance, self.config.fee_rate, self.config.capital_use,
        )

        n_trades = len(trades)
        wins = sum(1 for t in trades if t[5] > 0)
        win_rate = wins / n_trades * 100.0 if n_trades > 0 else 0.0
        gross_profit = sum(t[5] for t in trades if t[5] > 0)
        gross_loss = abs(sum(t[5] for t in trades if t[5] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else (5.0 if gross_profit > 0 else 1.0)

        eq = np.asarray(equity_curve)
        peaks = np.maximum.accumulate(eq)
        drawdowns = (peaks - eq) / peaks
        mdd_pct = float(np.nanmax(drawdowns) * 100.0) if len(drawdowns) > 0 and not np.isnan(drawdowns).all() else 0.0
        tot_ret = (final_balance - self.config.initial_balance) / self.config.initial_balance * 100.0

        results.append(
            EtfBacktestResult(
                market=target_market,
                total_return_pct=float(tot_ret),
                mdd_pct=mdd_pct,
                total_trades=n_trades,
                win_rate=win_rate,
                profit_factor=pf,
                final_balance=float(final_balance),
                equity_curve=eq.tolist(),
                trades=[
                    {"entry_idx": t[0], "exit_idx": t[1], "asset": t[2], "entry_price": t[3],
                     "exit_price": t[4], "pnl": t[5], "amount": t[6], "entry_fee": t[7]}
                    for t in trades
                ],
            )
        )
        return results
