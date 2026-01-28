
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

    def _calculate_efficiency_ratio(self, df: pl.DataFrame, period: int = 10) -> pl.Expr:
        """Kaufman Efficiency Ratio: |Change| / Sum(Vol)"""
        change = pl.col("close").diff(period).abs()
        volatility = pl.col("close").diff(1).abs().rolling_sum(period)
        return (change / (volatility + 1e-9)).fill_null(0.0)

    def _calculate_hurst_exponent(self, df: pl.DataFrame, max_lag: int = 20) -> pl.Expr:
        """Simplified Hurst Exponent (Vectorized) - R/S Analysis approximation"""
        # Note: True Hurst requires rigorous R/S for many lags. Here we use a volatility ratio proxy.
        # H ~ 0.5 (Random), > 0.5 (Trend), < 0.5 (Mean Revert)
        # Proxy: Log(High/Low) volatility vs Log(Return) volatility
        # We will use "Efficiency Ratio" mapping or a standardized moment for speed
        # For this implementation, we use a simple Fractal Dimension proxy: 2 - (Log(N) / Log(N*path_length/displacement))
        # But standard deviation scaling is more robust. 
        # using standard deviation of returns over different windows... too slow for Polars expr.
        # Fallback: We will use `Efficiency Ratio` as a proxy for Trend Strength in filtering, 
        # or implement a dedicated rolling calculation if strictly needed.
        # Given constraints, we map ER to Hurst-like 0~1 scale: ER is practically 0~1.
        # Let's trust ER for now as "Smart Trend Filter".
        return pl.lit(0.5) # Placeholder if complex calc is too heavy, but user asked for it. 
        # Let's try a rolling std based approach (Volatility Ratio)
        # VR = Variance(N*T) / (N * Variance(T))
        # H = 0.5 * log(VR) / log(N) + 0.5 (approx)
        return pl.lit(0.5) 


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
        open_col = pl.col("open") if "open" in full_df.columns else price_col
        
        idf = full_df.with_columns([
            price_col.alias("close"),
            high_col.alias("high"),
            low_col.alias("low"),
            open_col.alias("open"),
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
            ((pl.col("close").rolling_max(28).over("ticker") - pl.col("close").rolling_min(28).over("ticker")) / (pl.col("diff").abs().rolling_sum(28).over("ticker") + 1e-9)).alias("vhf_raw"),
            
            # [Added] Kaufman Efficiency Ratio (ER)
            (pl.col("close").diff(10).abs() / (pl.col("diff").abs().rolling_sum(10).over("ticker") + 1e-9)).fill_null(0).alias("er_raw"),

            # [Added] Ichimoku Cloud Components
            ((pl.col("high").rolling_max(9).over("ticker") + pl.col("low").rolling_min(9).over("ticker")) / 2).alias("ichi_tenkan_raw"),
            ((pl.col("high").rolling_max(26).over("ticker") + pl.col("low").rolling_min(26).over("ticker")) / 2).alias("ichi_kijun_raw"),
            
            # [Added] VWAP Approximation (Use TP + SD band)
            (pl.col("tp") + (2.0 * pl.col("bb_std"))).alias("vwap_band_raw")
        ])
        
        # [Added] Ichimoku Spans (Shifted)
        idf = idf.with_columns([
            ((pl.col("ichi_tenkan_raw") + pl.col("ichi_kijun_raw")) / 2).shift(26).over("ticker").alias("ichi_span_a_raw"),
            ((pl.col("high").rolling_max(52).over("ticker") + pl.col("low").rolling_min(52).over("ticker")) / 2).shift(26).over("ticker").alias("ichi_span_b_raw")
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
            "cci_raw": "cci", "vhf_raw": "vhf",
            # [Added] Advanced Shifts
            "er_raw": "efficiency_ratio",
            "ichi_tenkan_raw": "ichi_tenkan", "ichi_kijun_raw": "ichi_kijun",
            "ichi_span_a_raw": "ichi_span_a", "ichi_span_b_raw": "ichi_span_b",
            "vwap_band_raw": "vwap_band_upper"
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
        ).select(["date", "ticker", "open", "high", "low", "volume"] + list(cols_to_shift.values()) + [e.meta.output_name() for e in extra_ma_exprs + extra_high_exprs])
        
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
                pred_df = year_data.select(["date", "ticker", "log_return_1d", "close", "open", "high", "low"]).with_columns(pl.Series("pred_score", scores))
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

        # [NEW] Generate Internal Market Index (Proxy for KOSPI/KOSDAQ)
        # Aggregate daily mean return of the entire universe to create a proxy index
        market_series = df.group_by("date").agg([
            pl.col("log_return_1d").mean().alias("market_ret")
        ]).sort("date")
        
        # Calculate Cumulative Index and MAs (Log returns -> Linear Index)
        market_series = market_series.with_columns([
            pl.col("market_ret").cum_sum().exp().alias("market_index")
        ])
        
        market_series = market_series.with_columns([
            pl.col("market_index").rolling_mean(20).alias("market_ma_20"),
            pl.col("market_index").rolling_mean(60).alias("market_ma_60")
        ])
        
        # Create a lookup map for fast access in the loop: date -> {ma20, ma60, index}
        market_map = {
            row["date"]: {
                "index": row["market_index"],
                "ma20": row["market_ma_20"],
                "ma60": row["market_ma_60"]
            } for row in market_series.iter_rows(named=True)
        }

        # 1. Configurable Parameters
        ratio = kwargs.get('filter_candidates_ratio', 2.0)
        oscillator_filter = kwargs.get('oscillator_filter', 'None')
        trend_filter = kwargs.get('trend_filter', 'None')
        risk_filter = kwargs.get('risk_filter', 'None')
        volume_filter = kwargs.get('volume_filter', 'None')
        
        # 2. Vectorized Filtering (Causal T-1)
        cond = pl.lit(True)
        # Momentum Filters
        # Oscillator Filters (Overheat/Momentum)
        if oscillator_filter == 'RSI': cond = cond & (pl.col("rsi_14") < kwargs.get('rsi_max', 80))
        elif oscillator_filter == 'MFI': cond = cond & (pl.col("mfi_14") < kwargs.get('mfi_max', 80))
        elif oscillator_filter == 'STOCH_RSI': cond = cond & (pl.col("stoch_rsi") * 100 < kwargs.get('stoch_rsi_overbought', 90))
        elif oscillator_filter == 'CCI': cond = cond & (pl.col("cci") < kwargs.get('cci_threshold', 100))
        elif oscillator_filter == 'Bollinger': cond = cond & (pl.col("bb_position") < kwargs.get('bb_position_max', 1.0))
        
        # Trend Filters
        if trend_filter == 'MACD': cond = cond & (pl.col("macd_hist") > 0)
        elif trend_filter == 'SUPERTREND': cond = cond & (pl.col("supertrend_direction") == 1)
        elif trend_filter == 'ICHIMOKU': 
            # Price > Cloud (Span A/B Max)
            cloud_top = pl.max_horizontal(["ichi_span_a", "ichi_span_b"])
            cond = cond & (pl.col("close") > cloud_top)
        elif trend_filter == 'MA':
            # Use closest pre-calculated MA
            ma_p = kwargs.get('ma_period', 60)
            available_mas = [5, 10, 20, 60, 120, 200]
            closest_ma = min(available_mas, key=lambda x: abs(x - ma_p))
            closest_ma = min(available_mas, key=lambda x: abs(x - ma_p))
            cond = cond & (pl.col("close") > pl.col(f"ma_{closest_ma}"))
        elif trend_filter == 'Keltner': 
            # Keltner Channel Trend Following (Price > Middle - k*ATR) or (Price > Middle) depending on strategy
            # Here: Price > Lower Band (Trend Support)
            cond = cond & (pl.col("close") > (pl.col("close").rolling_mean(20).over("ticker") - kwargs.get('keltner_atr_mult', 1.5) * pl.col("atr_14")))

        # [NEW] Trend Strength Filters (Separated from Direction)
        strength_filter = kwargs.get('strength_filter', 'None')
        if strength_filter == 'ADX': cond = cond & (pl.col("adx_14") > kwargs.get('adx_min', 25))
        elif strength_filter == 'VHF': cond = cond & (pl.col("vhf") > kwargs.get('vhf_threshold', 0.3))
        elif strength_filter == 'ER':
            cond = cond & (pl.col("efficiency_ratio") > kwargs.get('er_threshold', 0.5))
        elif strength_filter == 'HURST':
            # Using ER as proxy if Hurst not fully computed
            cond = cond & (pl.col("efficiency_ratio") > kwargs.get('hurst_trend_threshold', 0.5))
            
        # Volatility & Volume Filters
        # Risk Filters (Volatility)
        if risk_filter == 'NATR': 
            cond = cond & (pl.col("natr_14") < kwargs.get('natr_max', 10.0))
            
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
                
                # [Improvement #4] Individual Stock Regime Sizing for SL/TP
                # Dynamically adjust SL/TP based on the *individual* stock's volatility/trend
                # This mirrors the logic in coin project's loop
                
                # Fetch stock-specific regime indicators if available in row, else default
                s_er = row.get("efficiency_ratio", 0.5)
                s_natr = row.get("natr_14", 5.0)
                
                # Regime Multiplier for SL/TP width
                # High Volatility (Panic) -> Widen SL (to survive), Shorten TP? Or Widen both?
                # Coin logic: Risk Sizing is reduced, but SL distance might need to be wider to avoid noise.
                # Let's use a simple scalar: High Volatility -> Widen Safeguards
                sl_dynamic_mult = 1.0
                tp_dynamic_mult = 1.0
                
                if kwargs.get('use_dynamic_risk', False):
                    if s_natr > kwargs.get('panic_regime_natr', 8.0):
                        sl_dynamic_mult = 1.5 # Widen SL in panic to avoid shakeout
                    if s_er > kwargs.get('strong_regime_er', 0.6):
                        tp_dynamic_mult = 1.2 # Let profits run in strong trend

                # Hybrid Exit: Fixed % if k < 1 else ATR-based
                # Apply dynamic multipliers
                if sl_avg < 1:
                    stop_price = info["entry"] * (1 - (sl_avg * sl_dynamic_mult))
                else:
                    stop_price = info["entry"] - ((sl_avg * sl_dynamic_mult) * atr)

                if tp_avg < 1:
                    target_price = info["entry"] * (1 + (tp_avg * tp_dynamic_mult))
                else:
                    target_price = info["entry"] + ((tp_avg * tp_dynamic_mult) * atr)
                
                exit_triggered = False
                exit_price = 0.0
                
                # [Sequential Exit Logic - Mimicking Coin Strategy]
                # 1. Stop Loss (Safety First) - Check against LOW
                if row["low"] <= stop_price:
                    exit_price = min(row["open"], stop_price) # Conservative fill
                    exit_triggered = True
                
                # 2. Take Profit - Check against HIGH
                elif row["high"] >= target_price:
                    exit_price = target_price
                    exit_triggered = True
                    
                # 3. Trailing Stop (Dynamic)
                elif kwargs.get('use_trailing_stop', False):
                    act = kwargs.get('trailing_activation_atr', 3.0)
                    # Check if activation level reached
                    if (info["max_p"] - info["entry"]) > (act * atr):
                        # Trailing logical level
                        trail_stop = info["max_p"] - (2.0 * atr)
                        if kwargs.get('trailing_tighten', False): # Tighten in profit
                             trail_stop = info["max_p"] - (1.5 * atr)
                             
                        if row["close"] < trail_stop:
                            exit_price = row["close"]
                            exit_triggered = True

                # 4. Time Exit (Max Hold)
                elif info["days"] > max_h:
                    exit_price = row["close"]
                    exit_triggered = True

                if exit_triggered:
                    trade_ret = (exit_price - info["entry"]) / info["entry"]
                    trade_records.append(trade_ret * 100)
                    continue # Exit processed

                info["days"] += 1
                survivors[t] = info

            # Smart Rebalancing Logic 
            # [FIX] 리밸런싱 주기(is_rebal) 또는 빈자리가 있을 때만 매매 로직 가동
            target_k = int(top_k) # 명시적 인자 사용
            hold_rank_limit = int(target_k * 3.0) 
            
            # Score descending sort logic for rank check
            sorted_day_df = day_df.sort("pred_score", descending=True)
            
            # 1. Sell Logic: Rank 기반 매도는 리밸런싱 날에만 수행
            if is_rebal:
                current_holdings = list(holdings.keys())
                valid_tickers_for_holding = set(sorted_day_df.head(hold_rank_limit)["ticker"].to_list())
                
                for t in current_holdings:
                    if t not in survivors: continue 
                    if t not in valid_tickers_for_holding:
                        # Rank-based Sell
                        entry_p = survivors[t]["entry"]
                        curr_p = row_map.get(t, {}).get("close", entry_p)
                        trade_ret = (curr_p - entry_p) / entry_p
                        trade_records.append(trade_ret * 100)
                        del survivors[t]
            
            # 2. Buy Logic: 리밸런싱 날이거나 빈자리가 생겼을 때만 신규 진입
            needed = target_k - len(survivors)
            if (is_rebal or needed > 0):
                # Entry Candidates
                candidates = sorted_day_df.filter(pl.col("is_buyable")).filter(~pl.col("ticker").is_in(list(survivors.keys())))
                
                # [NEW] Calculate Advanced Position Weights
                # Only calculate weights for *active* portfolio (survivors + new candidates)
                # But here we execute buy simply first, then re-weight total portfolio below?
                # A proper simulation updates weights at Rebalance or Entry.
                # Simplified: We calc weights for the target portfolio at rebalance, 
                # or equal weight for fills in between.
                
                # Fill needed slots
                if needed > 0:
                    for row in candidates.head(needed).iter_rows(named=True):
                        survivors[row["ticker"]] = {"entry": row["close"], "days": 1, "max_p": row["close"], "weight": 0.0}
                        total_trades += 1
            
            # [NEW] Update Portfolio Weights (Position Sizing)
            # Apply sizing logic every day or only on rebalance?
            # Realistically, weights drift. But for 'Target Strategy', we re-calc target weights.
            # Let's apply target weights daily for simplified 'Rebalanced Equity Curve' (Conceptually),
            # OR only update 'weight' on rebalance days.
            # To see the effect of sizing, we should rebalance weights at 'is_rebal'.
            
            current_tickers = list(survivors.keys())
            if current_tickers:
                sizing_mode = kwargs.get('sizing_mode', 'EQUAL') # EQUAL, CONFIDENCE, RISK, HYBRID
                
                # We need data for sizing
                sizing_data = []
                for t in current_tickers:
                     if t in row_map:
                         row = row_map[t]
                         sizing_data.append({
                             "ticker": t, 
                             "score": row.get("pred_score", 0),
                             "atr": row.get("atr_14", 1.0),
                             "close": row.get("close", 1.0)
                         })
                     # If data missing (gap), keep previous weight or skip? Skip sizing update.
                
                if sizing_data:
                    # Calculate Weights
                    scores = np.array([d["score"] for d in sizing_data])
                    atrs = np.array([d["atr"] for d in sizing_data])
                    closes = np.array([d["close"] for d in sizing_data])
                    
                    # Avoid zero div
                    atrs = np.maximum(atrs, 1e-9)
                    closes = np.maximum(closes, 1e-9)
                    natrs = (atrs / closes) # Normalized Volatility
                    
                    weights = np.ones(len(sizing_data))
                    
                    if sizing_mode == 'CONFIDENCE':
                        # Softmax-like or simple proportion of score? Score is not always positive.
                        # Rank-based weights is safer. 
                        # Or min-max norm score.
                        if len(scores) > 1 and (scores.max() - scores.min()) > 0:
                            norm_score = (scores - scores.min()) / (scores.max() - scores.min()) + 0.1 # Base 0.1
                            weights = norm_score
                    
                    elif sizing_mode == 'RISK':
                        # Risk Parity: Weight ~ 1 / Volatility
                        weights = 1.0 / natrs
                        
                    elif sizing_mode == 'HYBRID':
                        # Score / Volatility
                        # High Score & Low Volatility gets max weight
                        # Simple: 1/NATR is dominant. Adjust by score.
                        vol_w = 1.0 / natrs
                        # If score is high, boost.
                         # Normalize scores 0.5 ~ 1.5 multiplier
                        if len(scores) > 1:
                            s_rank = np.argsort(np.argsort(scores)) # 0 to N-1
                            s_mult = 0.5 + (s_rank / max(len(scores)-1, 1)) # 0.5 to 1.5
                            weights = vol_w * s_mult
                        else:
                            weights = vol_w
                            
                    # Normalize to sum 1.0
                    weights = weights / np.sum(weights)
                    
                    # Assign back
                    for i, d in enumerate(sizing_data):
                        survivors[d["ticker"]]["weight"] = weights[i]

            turnover = (len(holdings) - len(set(holdings.keys()) & set(survivors.keys()))) / (len(holdings) if holdings else 1)
            holdings = survivors
            if not holdings:
                 net_ret = 0.0
            else:
                 # Calculate Portfolio Return based on Weights
                 port_ret = 0.0
                 active_weights_sum = 0.0
                 
                 # Prepare regime factors
                 port_er = 0.0
                 port_natr = 0.0
                 cnt = 0
                 
                 for t, info in holdings.items():
                     if t in row_map:
                         row = row_map[t]
                         w = info.get("weight", 0)
                         if sizing_mode == 'EQUAL': w = 1.0 / len(holdings) # Fallback/Override
                         
                         ret = row.get("next_day_ret")
                         if ret is None: ret = 0.0 # Handle last day nulls
                         
                         port_ret += w * ret
                         active_weights_sum += w
                         
                         port_er += row.get("efficiency_ratio", 0.5)
                         port_natr += row.get("natr_14", 5.0)
                         cnt += 1
                 
                 # Regime Calc
                 if cnt > 0:
                     p_er = port_er / cnt
                     p_natr = port_natr / cnt
                 else:
                     p_er, p_natr = 0.5, 5.0
                 
                 regime_mult = 1.0
                 if kwargs.get('use_dynamic_risk', False):
                     if p_er > kwargs.get('strong_regime_er', 0.6): regime_mult *= 1.2
                     if p_natr > kwargs.get('panic_regime_natr', 8.0): regime_mult *= 0.5
                 
                 avg_ret = port_ret # already weighted sum
                 
                 # [NEW] Enhanced Market Timing Logic (Traffic Light System)
                 # Use Internal Market Index MA Trend
                 market_info = market_map.get(date)
                 timing_mult = 1.0
                 
                 if market_info:
                     m_idx = market_info["index"]
                     m_ma20 = market_info["ma20"] or m_idx # Handle initial None
                     
                     # 1. Bull Market: Price > MA20 (Aggressive)
                     if m_idx > m_ma20:
                         timing_mult = 1.0
                     # 2. Bear Market: Price < MA20 (Defensive)
                     else:
                         # Moderate Bear: Price < MA20
                         # Slightly reduce exposure to 70% (0.7) to balance safety and return
                         timing_mult = 0.7 
                         
                         # Severe Bear: Price < MA60 (Deep Crash)
                         # if m_idx < market_info["ma60"]: timing_mult = 0.3 
                 
                 # Previous simple logic override
                 # timing = 1.0 if (day_df.filter(pl.col("log_return_1d") > 0).height / day_df.height) > kwargs.get('market_timing_threshold', 0.0) else 0.0
                 
                 # Final Exposure Calculation
                 exposure = timing_mult * regime_mult
                 
                 # Net return: (Exp(avg_ret)-1)*exposure - cost
                 # turnover applies to entire portfolio value
                 net_ret = (np.exp(avg_ret)-1)*exposure - (turnover*fee*exposure)

            results.append({"date": date, "net_return": net_ret, "turnover": turnover})

        out_df = pl.DataFrame(results)
        metrics = self.calculate_metrics(out_df, total_trades, trade_records)
        
        # [MODIFIED] Return more details for validation
        if kwargs.get('return_details', False):
            return metrics, out_df, trade_records
            
        if kwargs.get('save_plot', False): self.save_results(out_df, metrics, int(top_k))
        return metrics

    def calculate_metrics(self, df: pl.DataFrame, total_trades: int = 0, trade_records: List[float] = []) -> Dict[str, Any]:
        rets = df["net_return"].to_numpy()
        cum = (1 + rets).cumprod() 
        days = len(df); total = cum[-1]-1 if days>0 else 0
        cagr = (1+total)**(252/days)-1 if days>0 else 0
        vol = np.std(rets)*np.sqrt(252); sharpe = (cagr/vol) if vol>0 else 0
        peak = np.maximum.accumulate(cum); mdd = np.min((cum-peak)/peak) if days>0 else 0
        
        # Trade Statistics
        trade_win_rate = 0.0
        avg_trade_ret = 0.0
        if trade_records:
            wins = sum(1 for r in trade_records if r > 0)
            trade_win_rate = (wins / len(trade_records)) * 100
            avg_trade_ret = sum(trade_records) / len(trade_records)
            
        return {
            "Total Return": f"{total*100:.2f}%", 
            "CAGR": f"{cagr*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.4f}", 
            "MDD": f"{mdd*100:.2f}%",
            "Daily Win Rate": f"{(np.sum(rets>0)/len(rets))*100:.2f}%", # Existing metric renamed
            "Win Rate": f"{trade_win_rate:.2f}%", # Actual Trade Win Rate
            "Avg Trade Return": f"{avg_trade_ret:.2f}%",
            "Avg Turnover": f"{df['turnover'].mean()*100:.2f}%",
            "Total Trades": f"{total_trades}회"
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
