import polars as pl
import numpy as np
from src.data.preprocessors.base_processor import BaseProcessor

class TechProcessor(BaseProcessor):
    """
    기술적 지표 (Technical Indicators) 전처리기
    
    Features:
    1. Log Returns: ln(Close / Close_shift)
    2. Volatility: Rolling Std of Log Returns
    3. Price Disparity: Close / Moving Average (Stationary Trend)
    4. Volume Ratio: Volume / Moving Average Volume
    5. Intraday Volatility: (High - Low) / Open
    6. Amihud Illiquidity: Mean(|Return| / Trading Value) - Liquidity Risk
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Pre-check: Ensure sorted by date for rolling calc
        df = df.sort(["ticker", "date"])
        
        # 1. Log Returns (1, 5, 20, 60, 120 days)
        windows = [1, 5, 20, 60, 120]
        exprs = []
        for w in windows:
            exprs.append(
                (pl.col("close") / pl.col("close").shift(w).over("ticker")).log().alias(f"log_return_{w}d")
            )
        
        df = df.with_columns(exprs)
        
        # 2. Volatility (20, 60 days)
        vol_windows = [20, 60]
        exprs = []
        for w in vol_windows:
            exprs.append(
                pl.col("log_return_1d").rolling_std(window_size=w).over("ticker").alias(f"volatility_{w}d")
            )
        
        df = df.with_columns(exprs)
        
        # 3. Price Disparity (이격도)
        ma_windows = [5, 20, 60, 120]
        exprs = []
        for w in ma_windows:
            ma_col = pl.col("close").rolling_mean(window_size=w).over("ticker")
            exprs.append(
                (pl.col("close") / ma_col.replace(0, None)).alias(f"disparity_{w}d")
            )
            
        df = df.with_columns(exprs)
        
        # 4. Volume Ratio
        vol_ratio_windows = [5, 20, 60]
        exprs = []
        for w in vol_ratio_windows:
            vol_ma = pl.col("volume").rolling_mean(window_size=w).over("ticker")
            exprs.append(
                (pl.col("volume") / vol_ma.replace(0, None)).alias(f"volume_ratio_{w}d")
            )
            
        df = df.with_columns(exprs)
        
        # 5. Intraday Volatility (일중 변동성)
        # (High - Low) / Open (0/NaN 방어)
        df = df.with_columns([
            ((pl.col("high") - pl.col("low")) / pl.col("open").replace(0, None)).fill_null(0)
            .alias("intraday_vol")
        ])
        
        # 6. Amihud Illiquidity (보정된 유동성 지표)
        # trading_value가 없으면 계산해서 생성 (Close * Volume)
        if "trading_value" not in df.collect_schema().names():
             df = df.with_columns(
                 (pl.col("close") * pl.col("volume")).alias("trading_value")
             )
             
        # 1e6을 곱하여 수치 가독성 확보 및 0/inf 방어
        abs_ret = pl.col("log_return_1d").abs()
        amihud_daily = (abs_ret / pl.col("trading_value").replace(0, None)).fill_null(0) * 1e6
        
        df = df.with_columns([
            amihud_daily.rolling_mean(window_size=20).over("ticker").alias("amihud_20d")
        ])
        
        return df
