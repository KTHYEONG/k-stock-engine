
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any, Optional
from catboost import CatBoostRanker
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.data_loader import YetiRankDataLoader
from src.utils.logger import setup_logger

# Mute verbose info logs from data_loader
logging.getLogger("training.data_loader").setLevel(logging.WARNING)
logger = setup_logger("evaluation.backtester")

class YetiRankBacktester:
    """
    YetiRank 모델의 실전 성과 검증 (Backtesting)
    - Full technical indicator suite implementation
    - Look-ahead bias prevention
    - Robust numerical stability
    """
    
    def __init__(self, start_date: str = "20240101", end_date: str = "20251231", model_year: Optional[int] = None):
        self.loader = YetiRankDataLoader(start_date="20160401") 
        self.model_dir = PROJECT_ROOT / "models" / "yetirank"
        self.output_dir = PROJECT_ROOT / "results" / "backtest"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_date = start_date
        self.end_date = end_date
        self.model_year = model_year # [ADD] 특정 모델 고정 사용 기능
        
    def load_model(self, year: int) -> CatBoostRanker:
        model_path = self.model_dir / f"yetirank_{year}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = CatBoostRanker()
        model.load_model(str(model_path))
        return model

    def _calculate_supertrend_np(self, df_pl: pl.DataFrame, multiplier=3.0) -> np.ndarray:
        """Numpy-based SuperTrend calculation with Ticker Boundary Protection"""
        high = df_pl["high"].to_numpy()
        low = df_pl["low"].to_numpy()
        close = df_pl["close"].to_numpy()
        atr = df_pl["atr_14"].to_numpy() 
        tickers = df_pl["ticker"].to_numpy()
        
        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        final_upper = np.zeros_like(basic_upper)
        final_lower = np.zeros_like(basic_lower)
        trend = np.ones_like(close, dtype=int) 
        
        final_upper[0] = basic_upper[0]
        final_lower[0] = basic_lower[0]
        
        for i in range(1, len(close)):
            if tickers[i] != tickers[i-1]:
                final_upper[i] = basic_upper[i]
                final_lower[i] = basic_lower[i]
                trend[i] = 1
                continue
                
            final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
            final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
            
            if trend[i-1] == 1:
                trend[i] = -1 if close[i] < final_lower[i] else 1
            else:
                trend[i] = 1 if close[i] > final_upper[i] else -1
                    
        return trend

    def generate_predictions(self):
        """
        [Ultra-Stable] 모델 예측값 + 모든 확장 지표 생성
        """
        logger.info("⚡ Generating Safe & Stable Predictions/Indicators...")
        
        if not hasattr(self, "_cached_full_df"):
             self._cached_full_df = self.loader.load_full_data(end_date=self.end_date, sample_ratio=1.0).sort(["ticker", "date"])
             self._cached_feature_names = self.loader.get_feature_names(self._cached_full_df)
        
        full_df = self._cached_full_df
        feature_names = self._cached_feature_names
        
        # --- [1단계] 기초 데이터 ---
        price_col = pl.col("close") if "close" in full_df.columns else (1 + pl.col("log_return_1d")).cumprod().over("ticker")
        vol_col = pl.col("volume") if "volume" in full_df.columns else pl.col("trading_volume") if "trading_volume" in full_df.columns else pl.lit(1.0)
        high_col = pl.col("high") if "high" in full_df.columns else price_col
        low_col = pl.col("low") if "low" in full_df.columns else price_col
        
        idf = full_df.with_columns([
            price_col.alias("close"),
            high_col.alias("high"),
            low_col.alias("low"),
            vol_col.alias("volume"),
            price_col.shift(1).over("ticker").alias("prev_close"),
            high_col.shift(1).over("ticker").alias("prev_high"),
            low_col.shift(1).over("ticker").alias("prev_low"),
            price_col.diff().over("ticker").alias("diff")
        ])

        # --- [2단계] 의존성 고려한 지표 계산 파이프라인 ---
        
        # 1. TR, DM (for ADX)
        idf = idf.with_columns([
            pl.max_horizontal([
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("prev_close")).abs(),
                (pl.col("low") - pl.col("prev_close")).abs()
            ]).alias("tr"),
            pl.when((pl.col("high") - pl.col("prev_high")) > (pl.col("prev_low") - pl.col("low")))
              .then((pl.col("high") - pl.col("prev_high")).clip(lower_bound=0))
              .otherwise(0.0).alias("_pdm"),
            pl.when((pl.col("prev_low") - pl.col("low")) > (pl.col("high") - pl.col("prev_high")))
              .then((pl.col("prev_low") - pl.col("low")).clip(lower_bound=0))
              .otherwise(0.0).alias("_ndm"),
            pl.col("diff").clip(lower_bound=0).alias("_gain"),
            (-pl.col("diff").clip(upper_bound=0)).alias("_loss"),
            (((pl.col("high")+pl.col("low")+pl.col("close"))/3)).alias("tp"),
            pl.when(pl.col("high") > pl.col("low"))
              .then(((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low")))
              .otherwise(0.0).alias("_mf_mult")
        ])

        # 2. Rolling Sums / Means
        idf = idf.with_columns([
            pl.col("tr").rolling_mean(14).over("ticker").fill_null(0).alias("atr_raw"),
            pl.col("_pdm").rolling_mean(14).over("ticker").alias("_pdm_14"),
            pl.col("_ndm").rolling_mean(14).over("ticker").alias("_ndm_14"),
            pl.col("_gain").rolling_mean(14).over("ticker").alias("_avg_gain"),
            pl.col("_loss").rolling_mean(14).over("ticker").alias("_avg_loss"),
            (pl.col("tp") * pl.col("volume")).alias("_flow_raw"),
            pl.col("close").ewm_mean(span=12, adjust=False).over("ticker").alias("ema_12"),
            pl.col("close").ewm_mean(span=26, adjust=False).over("ticker").alias("ema_26"),
            (pl.col("diff").sign().fill_null(0) * pl.col("volume")).cum_sum().over("ticker").alias("obv"),
            pl.col("close").rolling_mean(20).over("ticker").alias("bb_mid"),
            pl.col("close").rolling_std(20).over("ticker").alias("bb_std"),
            # Dynamic MA supports
            pl.col("close").rolling_mean(60).over("ticker").alias("ma_60"),
            pl.col("close").rolling_mean(120).over("ticker").alias("ma_120"),
            (pl.col("_mf_mult") * pl.col("volume")).rolling_sum(20).over("ticker").alias("_mfv_sum"),
            pl.col("volume").rolling_sum(20).over("ticker").alias("_vol_sum")
        ])

        # 3. Oscillators (RSI, DMI, MACD Line)
        idf = idf.with_columns([
            (100 - (100 / (1 + (pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-9))))).alias("rsi_raw"),
            (100 * pl.col("_pdm_14") / (pl.col("atr_raw") + 1e-9)).alias("_pdi"),
            (100 * pl.col("_ndm_14") / (pl.col("atr_raw") + 1e-9)).alias("_ndi"),
            pl.when(pl.col("close") > pl.col("prev_close")).then(pl.col("_flow_raw")).otherwise(0).rolling_sum(14).over("ticker").alias("_pos_flow"),
            pl.when(pl.col("close") < pl.col("prev_close")).then(pl.col("_flow_raw")).otherwise(0).rolling_sum(14).over("ticker").alias("_neg_flow"),
            (pl.col("ema_12") - pl.col("ema_26")).alias("macd_line"),
            pl.when(pl.col("_vol_sum") > 0).then(pl.col("_mfv_sum") / pl.col("_vol_sum")).otherwise(0.0).alias("cmf_raw")
        ])

        # 4. Final Complex Indicators (ADX, MACD Hist, etc)
        idf = idf.with_columns([
            (100 * (pl.col("_pdi") - pl.col("_ndi")).abs() / (pl.col("_pdi") + pl.col("_ndi") + 1e-9)).alias("_dx"),
            (100 - (100 / (1 + (pl.col("_pos_flow") / (pl.col("_neg_flow") + 1e-9))))).alias("mfi_raw"),
            (pl.col("macd_line") - pl.col("macd_line").ewm_mean(span=9, adjust=False).over("ticker")).alias("macd_hist_raw"),
            pl.when((pl.col("rsi_raw").rolling_max(14).over("ticker") - pl.col("rsi_raw").rolling_min(14).over("ticker")) > 0)
              .then((pl.col("rsi_raw") - pl.col("rsi_raw").rolling_min(14).over("ticker")) / (pl.col("rsi_raw").rolling_max(14).over("ticker") - pl.col("rsi_raw").rolling_min(14).over("ticker")))
              .otherwise(0.5).alias("stoch_rsi_raw"),
            pl.when(pl.col("bb_std") > 0).then((pl.col("close") - pl.col("bb_mid")) / (2 * pl.col("bb_std"))).otherwise(0.5).alias("bb_pos_raw"),
            ((pl.col("high").rolling_max(52).over("ticker") + pl.col("low").rolling_min(52).over("ticker")) / 2).alias("cloud_raw"),
            pl.when(pl.col("close") > 0).then(pl.col("atr_raw") / pl.col("close") * 100).otherwise(0.0).alias("natr_raw"),
            # CCI Calculation
            ((pl.col("tp") - pl.col("tp").rolling_mean(20).over("ticker")) / (0.015 * pl.col("tp").rolling_std(20).over("ticker") + 1e-9)).alias("cci_raw"),
            # VHF Calculation
            ((pl.col("close").rolling_max(28).over("ticker") - pl.col("close").rolling_min(28).over("ticker")) / (pl.col("diff").abs().rolling_sum(28).over("ticker") + 1e-9)).alias("vhf_raw")
        ])
        
        idf = idf.with_columns([
            pl.col("_dx").rolling_mean(14).over("ticker").fill_null(0).alias("adx_raw"),
            pl.col("cmf_raw").clip(lower_bound=-1.0, upper_bound=1.0).alias("cmf_raw")
        ])

        # SuperTrend
        # [FIX] _calculate_supertrend_np expects 'atr_14' column
        idf = idf.with_columns(pl.col("atr_raw").alias("atr_14"))
        st_trend = self._calculate_supertrend_np(idf)
        idf = idf.with_columns(pl.Series("st_trend_raw", st_trend))
        
        # --- [3단계] Look-ahead Bias Prevention (Shift All Decision Indicators) ---
        cols_to_shift = {
            "rsi_raw": "rsi_14", "mfi_raw": "mfi_14", "natr_raw": "natr_14", 
            "adx_raw": "adx_14", "macd_hist_raw": "macd_hist", "stoch_rsi_raw": "stoch_rsi", 
            "bb_pos_raw": "bb_position", "cloud_raw": "cloud_top_approx", 
            "st_trend_raw": "supertrend_direction", "atr_raw": "atr_14",
            "ma_60": "ma_60", "ma_120": "ma_120", "cmf_raw": "cmf", "obv": "obv",
            "cci_raw": "cci", "vhf_raw": "vhf"
        }
        
        # Add extra MAs and Breakout channels for optimization support
        extra_ma_exprs = [
            pl.col("close").rolling_mean(p).over("ticker").shift(1).fill_null(0).alias(f"ma_{p}")
            for p in [5, 10, 20, 200]
        ]
        extra_high_exprs = [
            pl.col("high").rolling_max(p).over("ticker").shift(1).fill_null(0).alias(f"high_{p}")
            for p in [5, 10, 20, 200]
        ]

        indicator_df = idf.with_columns([
            pl.col(old).shift(1).over("ticker").fill_null(0).alias(new) for old, new in cols_to_shift.items()
        ]).with_columns(
            extra_ma_exprs + extra_high_exprs
        ).select(["date", "ticker", "volume"] + list(cols_to_shift.values()) + [e.meta.output_name() for e in extra_ma_exprs + extra_high_exprs])
        
        # --- [4단계] Merge with Model Predictions ---
        test_df = full_df.filter(
            (pl.col("date") >= pl.lit(self.start_date).str.to_date("%Y%m%d")) & 
            (pl.col("date") <= pl.lit(self.end_date).str.to_date("%Y%m%d"))
        ).sort("date")
        
        predictions = []
        for year in sorted(test_df.select(pl.col("date").dt.year()).unique().to_series().to_list()):
            year_data = test_df.filter(pl.col("date").dt.year() == year)
            if year_data.is_empty(): continue
            try:
                # [MODIFIED] model_year가 지정되어 있으면 해당 모델 상시 사용, 아니면 데이터 연도에 맞춤
                target_model_year = self.model_year if self.model_year is not None else year
                model = self.load_model(target_model_year)
                X = year_data.select(feature_names).to_pandas()
                scores = model.predict(X)
                pred_df = year_data.select(["date", "ticker", "log_return_1d", "close"]).with_columns(pl.Series("pred_score", scores))
                pred_df = pred_df.join(indicator_df, on=["date", "ticker"], how="left")
                pred_df = pred_df.with_columns(pl.col("log_return_1d").shift(-1).over("ticker").alias("next_day_ret"))
                predictions.append(pred_df)
            except Exception as e: logger.error(f"Prediction failed for year {year}: {e}")

        self._cached_predictions = pl.concat(predictions).sort(["date", "pred_score"], descending=[False, True]) if predictions else pl.DataFrame()
        logger.info(f"✅ Safe Predictions cached! Rows: {len(self._cached_predictions)}")

    def run_backtest(self, top_k: int = 20, rebalance_period: int = 5, **kwargs):
        """[Comprehensive] Backtesting Engine with Dynamic Logic"""
        if not hasattr(self, "_cached_predictions"): self.generate_predictions()
        df = self._cached_predictions
        if df.is_empty(): return {}

        # 1. Configurable Parameters
        ratio = kwargs.get('filter_candidates_ratio', 2.0)
        mom_filter = kwargs.get('momentum_filter', 'None')
        trend_filter = kwargs.get('trend_filter', 'None')
        vol_filter = kwargs.get('volatility_filter', 'None')
        volume_filter = kwargs.get('volume_filter', 'None')
        
        # 2. Vectorized Filtering (Causal T-1)
        cond = pl.lit(True)
        # Momentum Filters
        if mom_filter == 'RSI': cond = cond & (pl.col("rsi_14") < kwargs.get('rsi_max', 80))
        elif mom_filter == 'MFI': cond = cond & (pl.col("mfi_14") < kwargs.get('mfi_max', 80))
        elif mom_filter == 'STOCH_RSI': cond = cond & (pl.col("stoch_rsi") * 100 < kwargs.get('stoch_rsi_overbought', 90))
        elif mom_filter == 'CCI': cond = cond & (pl.col("cci") < kwargs.get('cci_threshold', 100))
        
        # Trend Filters
        if trend_filter == 'ADX': cond = cond & (pl.col("adx_14") > kwargs.get('adx_min', 25))
        elif trend_filter == 'MACD': cond = cond & (pl.col("macd_hist") > 0)
        elif trend_filter == 'SUPERTREND': cond = cond & (pl.col("supertrend_direction") == 1)
        elif trend_filter == 'ICHIMOKU': cond = cond & (pl.col("close") > pl.col("cloud_top_approx"))
        elif trend_filter == 'VHF': cond = cond & (pl.col("vhf") > kwargs.get('vhf_threshold', 0.3))
        elif trend_filter == 'MA':
            # Use closest pre-calculated MA
            ma_p = kwargs.get('ma_period', 60)
            available_mas = [5, 10, 20, 60, 120, 200]
            closest_ma = min(available_mas, key=lambda x: abs(x - ma_p))
            cond = cond & (pl.col("close") > pl.col(f"ma_{closest_ma}"))
            
        # Volatility & Volume Filters
        if vol_filter == 'Bollinger': cond = cond & (pl.col("bb_position") < kwargs.get('bb_position_max', 1.0))
        elif vol_filter == 'NATR': cond = cond & (pl.col("natr_14") < kwargs.get('panic_regime_natr', 10.0))
        # Keltner (Approximated using ATR)
        elif vol_filter == 'Keltner': 
            cond = cond & (pl.col("close") > (pl.col("close").rolling_mean(20).over("ticker") - kwargs.get('keltner_atr_mult', 1.5) * pl.col("atr_14")))
            
        if volume_filter == 'CMF': cond = cond & (pl.col("cmf") > kwargs.get('cmf_threshold', 0.0))
        elif volume_filter == 'Volume': cond = cond & (pl.col("volume") > pl.col("volume").rolling_mean(20).over("ticker") * kwargs.get('min_volume_ratio', 1.0))
        
        # Entry Lookback (Channel Breakout Style)
        entry_p = kwargs.get('entry_lookback', 20)
        available_highs = [5, 10, 20, 200]
        closest_high = min(available_highs, key=lambda x: abs(x - entry_p))
        if closest_high != 200: # Use as a secondary filter if not default/long
             cond = cond & (pl.col("close") >= pl.col(f"high_{closest_high}"))
            
        filtered_df = df.with_columns(cond.alias("is_buyable"))
            
        # 3. Simulation Loop
        sl_avg = kwargs.get('stop_loss_k', 0.05) 
        tp_avg = kwargs.get('take_profit_k', 0.15)
        max_h = kwargs.get('max_hold_days', 20)
        fee = 1.9e-3 # Typical fee + slippage
        
        holdings = {} # ticker -> {entry, days, max}
        results = []
        dates = filtered_df["date"].unique().sort()
        total_trades = 0 # [ADD] 누적 거래 횟수 추적
        
        trade_records = [] # [ADD] 개별 매매 수익률 기록

        for idx, date in enumerate(dates[:-1]):
            day_df = filtered_df.filter(pl.col("date") == date)
            row_map = {row["ticker"]: row for row in day_df.iter_rows(named=True)}
            is_rebal = (idx % rebalance_period == 0)
            
            # Exits (Daily Check)
            survivors = {}
            for t, info in holdings.items():
                if t not in row_map: # Data gap?
                    survivors[t] = info; continue
                row = row_map[t]
                curr_p, atr = row["close"], row["atr_14"]
                info["max_p"] = max(info.get("max_p", 0), curr_p)
                
                # Hybrid Exit: Fixed % if k < 1 else ATR-based
                sl = info["entry"]*(1-sl_avg) if sl_avg < 1 else info["entry"]-(sl_avg*atr)
                tp = info["entry"]*(1+tp_avg) if tp_avg < 1 else info["entry"]+(tp_avg*atr)
                
                if curr_p < sl or curr_p > tp or info["days"] > max_h: 
                    # Trade Closed
                    trade_ret = (curr_p - info["entry"]) / info["entry"]
                    trade_records.append(trade_ret * 100) # 퍼센트로 저장
                    continue # Exit
                
                # Trailing Stop (Dynamic)
                if kwargs.get('use_trailing_stop', False):
                    act = kwargs.get('trailing_activation_atr', 3.0)
                    if (info["max_p"] - info["entry"]) > (act * atr):
                         if curr_p < (info["max_p"] - 2.0*atr): 
                             # Trailing Stop Hit
                             trade_ret = (curr_p - info["entry"]) / info["entry"]
                             trade_records.append(trade_ret * 100)
                             continue
                
                info["days"] += 1
                survivors[t] = info

            # Smart Rebalancing Logic (Daily Check)
            # 1. Check if we hold invalid stocks (Rank dropped or filtered out)
            current_holdings = list(holdings.keys())
            
            # Get today's top candidates (AI Score + Filter)
            # We allow a slightly wider buffer for holding than for entry to prevent flickering
            # Entry: Top K, Hold: Top K * 1.5
            target_k = int(kwargs.get('top_k', top_k))
            hold_rank_limit = int(target_k * 1.5) 
            
            # Score descending sort logic for rank check
            sorted_day_df = day_df.sort("pred_score", descending=True)
            valid_tickers_for_holding = set(sorted_day_df.head(hold_rank_limit)["ticker"].to_list())
            
            # Sell Logic: Sell if Rank Dropped AND not profitable enough to ignore rank
            for t in current_holdings:
                if t not in survivors: continue # Already sold by SL/TP
                
                # If stock is no longer in Top K * 1.5, we consider selling
                if t not in valid_tickers_for_holding:
                    # Smart Hold: If trend is super strong (e.g. > Moving Average), maybe keep it?
                    # For now, strict Rank adherence often works best for AI models.
                    # Sell!
                    entry_p = survivors[t]["entry"]
                    curr_p = row_map.get(t, {}).get("close", entry_p)
                    trade_ret = (curr_p - entry_p) / entry_p
                    trade_records.append(trade_ret * 100)
                    del survivors[t]
            
            # Buy Logic: Fill empty slots
            needed = target_k - len(survivors)
            if needed > 0:
                # Entry Candidates: Must be 'buyable' filter passed AND Top K score
                candidates = sorted_day_df.filter(pl.col("is_buyable")).filter(~pl.col("ticker").is_in(list(survivors.keys())))
                
                # Take top available
                for row in candidates.head(needed).iter_rows(named=True):
                    survivors[row["ticker"]] = {"entry": row["close"], "days": 1, "max_p": row["close"]}
                    total_trades += 1
            
            turnover = (len(holdings) - len(set(holdings.keys()) & set(survivors.keys()))) / (len(holdings) if holdings else 1)

            holdings = survivors
            if not holdings:
                 net_ret = 0.0
            else:
                 port = day_df.filter(pl.col("ticker").is_in(holdings.keys()))
                 avg_ret = port["next_day_ret"].mean() or 0.0
                 timing = 1.0 if (day_df.filter(pl.col("log_return_1d") > 0).height / day_df.height) > kwargs.get('market_timing_threshold', 0.0) else 0.0
                 net_ret = (np.exp(avg_ret)-1)*timing - (turnover*fee*timing)
            results.append({"date": date, "net_return": net_ret, "turnover": turnover})

        out_df = pl.DataFrame(results)
        metrics = self.calculate_metrics(out_df, total_trades)
        
        # [MODIFIED] Return more details for validation
        if kwargs.get('return_details', False):
            return metrics, out_df, trade_records
            
        if kwargs.get('save_plot', False): self.save_results(out_df, metrics, int(kwargs.get('top_k', 20)))
        return metrics

    def calculate_metrics(self, df: pl.DataFrame, total_trades: int = 0) -> Dict[str, Any]:
        rets = df["net_return"].to_numpy()
        cum = (1 + rets).cumprod() 
        days = len(df); total = cum[-1]-1 if days>0 else 0
        cagr = (1+total)**(252/days)-1 if days>0 else 0
        vol = np.std(rets)*np.sqrt(252); sharpe = (cagr/vol) if vol>0 else 0
        peak = np.maximum.accumulate(cum); mdd = np.min((cum-peak)/peak) if days>0 else 0
        return {
            "Total Return": f"{total*100:.2f}%", "CAGR": f"{cagr*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.4f}", "MDD": f"{mdd*100:.2f}%",
            "Win Rate": f"{(np.sum(rets>0)/len(rets))*100:.2f}%", "Avg Turnover": f"{df['turnover'].mean()*100:.2f}%",
            "Total Trades": f"{total_trades}회" # [ADD] 결과에 추가
        }

    def save_results(self, perf_df: pl.DataFrame, metrics: Dict[str, Any], top_k: int):
        perf_df.write_csv(self.output_dir / f"backtest_top{top_k}_daily.csv")
        with open(self.output_dir / f"backtest_top{top_k}_summary.json", "w", encoding="utf-8") as f:
            import json; json.dump(metrics, f, indent=4, ensure_ascii=False)
        try:
            plt.figure(figsize=(10, 6)); plt.plot(perf_df["date"], (1+perf_df["net_return"]).cumprod()-1)
            plt.title(f"Cumulative Return - Top {top_k}"); plt.grid(True); plt.savefig(self.output_dir / f"backtest_top{top_k}_plot.png"); plt.close()
        except: pass
        print("\n" + "="*40 + f"\n📊 Final Backtest Summary (Top-{top_k})\n" + "="*40)
        for k, v in metrics.items(): print(f"{k:<20}: {v}")
        print("="*40 + "\n")

if __name__ == "__main__":
    YetiRankBacktester(start_date="20240101", end_date="20251231").run_backtest(top_k=5)
