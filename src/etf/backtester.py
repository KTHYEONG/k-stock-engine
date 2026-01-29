
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
    Stateful Backtester for ETF Switching Strategy.
    Simulates:
    - Long (2X) / Short (Inverse 2X) switching
    - Stop Loss / Take Profit (ATR Based) - Intraday High/Low Check
    - Trailing Stop
    """
    
    def __init__(self, index_df: pl.DataFrame, etf_df: pl.DataFrame):
        self.index_df = index_df.sort("date")
        self.etf_df = etf_df.sort("date")
        self.universe = ETFConfig.UNIVERSE
        self._warning_logged = set()
        
    def run(self, params: Dict[str, Any], target_market: str = None) -> List[Dict[str, Any]]:
        """
        Run backtest for defined markets.
        """
        engine = ETFStrategyEngine(params)
        results = []
        
        target_universe = {target_market: self.universe[target_market]} if target_market and target_market in self.universe else self.universe
        
        for mkt_name, tickers in target_universe.items():
            index_ticker = tickers["index_ticker"]
            
            # --- 1. Filter Index Data ---
            if "IDX_NM" in self.index_df.columns:
                mkt_idx = self.index_df.filter(pl.col("IDX_NM") == index_ticker)
                if mkt_idx.is_empty():
                    continue
            else:
                mkt_idx = self.index_df.filter(pl.col("ticker") == index_ticker)
                if mkt_idx.is_empty():
                    continue
                
            # Prepare Index Data (OHLCV)
            try:
                mkt_idx = mkt_idx.select([
                    pl.col("date"),
                    pl.col("OPNPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("open"),
                    pl.col("HGPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("high"),
                    pl.col("LWPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("low"),
                    pl.col("CLSPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("close"),
                    pl.col("ACC_TRDVOL").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("volume"),
                    pl.col("ticker")
                ]).filter(pl.col("close").is_not_null())
            except Exception as e:
                logger.error(f"❌ Failed to process Index Data for {index_ticker}: {e}")
                continue
                
            # Generate Signals
            sig_df = engine.generate_signal(mkt_idx)
            
            # --- 2. Load 4 Assets Data ---
            b1 = tickers["bull_1x"]
            i1 = tickers["bear_1x"]
            b2 = tickers["bull_2x"]
            i2 = tickers["bear_2x"]
            
            df_b1 = self.etf_df.filter(pl.col("ticker") == b1)
            df_i1 = self.etf_df.filter(pl.col("ticker") == i1)
            df_b2 = self.etf_df.filter(pl.col("ticker") == b2)
            df_i2 = self.etf_df.filter(pl.col("ticker") == i2)
            
            if df_b1.is_empty() or df_i1.is_empty() or df_b2.is_empty() or df_i2.is_empty():
                 continue
                 
            # Helper to calculate 4 returns: Gap, O-H, O-L, O-C
            def get_detailed_ret(dframe, prefix):
                # Clean Str -> Float
                cols = ["open", "high", "low", "close"]
                exprs = []
                for c in cols:
                    # Try cast if string, else keep
                    exprs.append(pl.col(c).cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False))
                
                try:
                    dframe = dframe.with_columns(exprs)
                except: pass
                
                # Metrics
                # Gap: (Open - PrevClose) / PrevClose
                # O-H: (High - Open) / Open
                # O-L: (Low - Open) / Open
                # O-C: (Close - Open) / Open
                
                return dframe.select([
                    pl.col("date"),
                    ((pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1)).fill_null(0.0).alias(f"{prefix}_gap"),
                    ((pl.col("high") - pl.col("open")) / pl.col("open")).fill_null(0.0).alias(f"{prefix}_oh"),
                    ((pl.col("low") - pl.col("open")) / pl.col("open")).fill_null(0.0).alias(f"{prefix}_ol"),
                    ((pl.col("close") - pl.col("open")) / pl.col("open")).fill_null(0.0).alias(f"{prefix}_oc")
                ])

            r_b1 = get_detailed_ret(df_b1, "b1")
            r_i1 = get_detailed_ret(df_i1, "i1")
            r_b2 = get_detailed_ret(df_b2, "b2")
            r_i2 = get_detailed_ret(df_i2, "i2")
            
            # --- 3. Join All ---
            sim_df = sig_df.join(r_b1, on="date", how="left")\
                           .join(r_i1, on="date", how="left")\
                           .join(r_b2, on="date", how="left")\
                           .join(r_i2, on="date", how="left")
            
            # --- 4. Run Simulation ---
            res = self._simulate_market(sim_df, params)
            if res:
                res["market"] = f"{mkt_name}_HYBRID"
                results.append(res)
                
        return results

    def _simulate_market(self, df: pl.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Numba Accelerating Wrapper (REALITY CHECK MODE).
        Constructs the 20-column array.
        """
        # 0: Signal(T-1)
        # 1: Risk(T-1)
        # 2: NATR(T-1)
        # 3: Hurst(T-1)
        # Assets (Gap, OH, OL, OC) -> B1, I1, B2, I2
        
        input_df = df.select([
            pl.col("signal_trigger").shift(1).fill_null(0).alias("sig"),
            pl.col("risk_mult").shift(1).fill_null(1.0).alias("risk"),
            pl.col("natr").shift(1).fill_null(0.0).alias("natr"),
            pl.col("hurst").shift(1).fill_null(0.5).alias("hurst"),
            
            # B1 (Fill nulls with 0)
            pl.col("b1_gap").fill_null(0.0),
            pl.col("b1_oh").fill_null(0.0),
            pl.col("b1_ol").fill_null(0.0),
            pl.col("b1_oc").fill_null(0.0),
            
            # I1
            pl.col("i1_gap").fill_null(0.0),
            pl.col("i1_oh").fill_null(0.0),
            pl.col("i1_ol").fill_null(0.0),
            pl.col("i1_oc").fill_null(0.0),
            
            # B2
            pl.col("b2_gap").fill_null(0.0),
            pl.col("b2_oh").fill_null(0.0),
            pl.col("b2_ol").fill_null(0.0),
            pl.col("b2_oc").fill_null(0.0),
            
            # I2
            pl.col("i2_gap").fill_null(0.0),
            pl.col("i2_oh").fill_null(0.0),
            pl.col("i2_ol").fill_null(0.0),
            pl.col("i2_oc").fill_null(0.0),
        ])
        
        data_arr = input_df.to_numpy()
        
        if data_arr.dtype != np.float64:
            data_arr = data_arr.astype(np.float64)
            
        if len(data_arr) < 2: return {}
        
        # Params
        sl_atr = float(params.get('STOP_LOSS_ATR', 3.0))
        tp_atr = float(params.get('TAKE_PROFIT_ATR', 5.0))
        lev_hurst = float(params.get('LEV_HURST', 0.55))
        lev_natr = float(params.get('LEV_NATR', 2.5))
        ts_trigger = float(params.get('TS_TRIGGER_ATR', 5.0))
        ts_dist = float(params.get('TS_DIST_ATR', 3.0))
        
        # Run Numba
        try:
            cagr, mdd, tot_ret, n_trades, n_wins, win_rate, pf, trade_rets_arr, daily_rets_arr = backtest_etf_numba(
                data_arr, sl_atr, tp_atr, lev_hurst, lev_natr, ts_trigger, ts_dist
            )
        except Exception as e:
            logger.error(f"Numba Error: {e}")
            return {}
            
        return {
            "cagr": float(cagr),
            "mdd": float(mdd),
            "total_return": float(tot_ret),
            "trades": int(n_trades),
            "wins": int(n_wins),
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "equity": float(1.0 + tot_ret),
            "trade_list": trade_rets_arr.flatten().tolist(),
            "daily_returns": daily_rets_arr.tolist()
        }
