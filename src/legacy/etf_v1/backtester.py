import polars as pl
import numpy as np
from typing import Dict, Any, List
import logging
from .strategy_engine import ETFStrategyEngine
from .etf_config import ETFConfig
from .backtest_engine_numba import backtest_etf_numba

logger = logging.getLogger("etf.backtester")

class ETFBacktester:
    """
    Stateful Backtester for ETF Switching Strategy using Numba.
    Supports Long (1X) / Short (Inverse 1X) switching based on IBS & Price Action logic.
    """
    
    def __init__(self, index_df: pl.DataFrame, etf_df: pl.DataFrame):
        self.index_df = index_df.sort("date")
        self.etf_df = etf_df.sort("date")
        self.universe = ETFConfig.UNIVERSE
        
    def run(self, params: Dict[str, Any], target_market: str = "KOSPI") -> List[Dict[str, Any]]:
        engine = ETFStrategyEngine(params)
        results = []
        
        target_universe = {target_market: self.universe[target_market]} if target_market in self.universe else self.universe
        
        for mkt_name, tickers in target_universe.items():
            index_ticker = tickers["index_ticker"]
            
            # --- 1. Filter Index Data ---
            mkt_idx = self.index_df.filter(pl.col("ticker") == mkt_name)
                
            if mkt_idx.is_empty():
                logger.error(f"Index Data for {mkt_name} is empty.")
                continue
                
            # Prepare Index Data (OHLCV)
            try:
                mkt_idx = mkt_idx.select([
                    pl.col("date"),
                    pl.col("OPNPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("open"),
                    pl.col("HGPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("high"),
                    pl.col("LWPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("low"),
                    pl.col("CLSPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("close"),
                ]).filter(pl.col("close").is_not_null())
            except Exception as e:
                logger.error(f"Failed to process Index Data for {index_ticker}: {e}")
                continue
                
            # Generate Signals
            sig_df = engine.generate_signal(mkt_idx)
            
            # --- 2. Load Assets Data ---
            b1 = tickers["bull_1x"]
            i1 = tickers["bear_1x"]
            
            df_b1 = self.etf_df.filter(pl.col("ticker") == b1)
            df_i1 = self.etf_df.filter(pl.col("ticker") == i1)
            
            if df_b1.is_empty() or df_i1.is_empty():
                 continue
                 
            def get_ohlc(dframe, prefix):
                cols = ["open", "high", "low", "close"]
                exprs = []
                for c in cols:
                    exprs.append(pl.col(c).cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias(f"{prefix}_{c}"))
                return dframe.select([pl.col("date")] + exprs)

            r_b1 = get_ohlc(df_b1, "b1")
            r_i1 = get_ohlc(df_i1, "i1")
            
            # --- 3. Join All ---
            sim_df = sig_df.join(r_b1, on="date", how="inner")\
                           .join(r_i1, on="date", how="inner")
            
            # --- 4. Run Simulation ---
            res = self._simulate_market(sim_df, params)
            if res:
                res["market"] = mkt_name
                results.append(res)
                
        return results

    def _simulate_market(self, df: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        input_df = df.select([
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
        ])
        
        arr = input_df.to_numpy().astype(np.float64)
        if len(arr) < 2: return {}
        
        ibs_exit = float(params.get("IBS_EXIT", 0.8))
        max_hold_days = int(params.get("MAX_HOLD_DAYS", 3))
        stop_loss_pct = float(params.get("STOP_LOSS_PCT", 0.1))
        
        try:
            trades, final_balance, equity_curve = backtest_etf_numba(
                arr, ibs_exit, max_hold_days, stop_loss_pct, 10000000.0, 0.00015
            )
        except Exception as e:
            logger.error(f"Numba Error: {e}")
            return {}
            
        import pandas as pd
        trades_df = pd.DataFrame(trades, columns=["entry_idx", "exit_idx", "asset", "entry_price", "exit_price", "pnl", "amount", "entry_fee"])
        
        n_trades = len(trades_df)
        wins = len(trades_df[trades_df["pnl"] > 0])
        win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0
        
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else (5.0 if gross_profit > 0 else 1.0)
        
        eq_arr = equity_curve
        peaks = np.maximum.accumulate(eq_arr)
        drawdowns = (peaks - eq_arr) / peaks
        # safe check to avoid NaN
        mdd_pct = float(np.nanmax(drawdowns) * 100.0) if len(drawdowns) > 0 and not np.isnan(drawdowns).all() else 0.0
        
        tot_ret = (final_balance - 10000000.0) / 10000000.0 * 100.0
        
        return {
            "total_return_pct": float(tot_ret),
            "mdd_pct": float(mdd_pct),
            "total_trades": int(n_trades),
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "final_balance": float(final_balance),
            "equity_curve": eq_arr.tolist(),
            "trades_df": trades_df
        }