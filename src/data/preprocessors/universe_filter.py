import polars as pl
import logging
from src.data.preprocessors.base_processor import BaseProcessor

logger = logging.getLogger("preprocessors.filter")

class UniverseFilter(BaseProcessor):
    """
    유니버스 필터링 (Universe Construction) 전처리기
    
    Layers:
    1. Liquidity: Penny Stocks, Min Trading Value, Sector Component Count
    2. Quality (QMJ): Deficit Filter with Turnaround Rescue logic
    3. Risk: Debt Ratio, Capital Erosion, High Volatility
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Pre-check: Ensure required columns exist
        cols = df.collect_schema().names()
        
        # 1. Liquidity & Microstructure
        df = df.sort(["ticker", "date"])
        df = df.with_columns([
            pl.col("trading_value").rolling_mean(window_size=5).over("ticker").alias("avg_trading_value_5d")
        ])
        
        liq_filter = (pl.col("close") >= 1000) & (pl.col("avg_trading_value_5d") >= 1e9)
        
        # 2. Sector Bias
        df = df.with_columns([
            pl.col("ticker").count().over(["date", "sector"]).alias("sector_count")
        ])
        sector_filter = (pl.col("sector_count") >= 5)
        
        # 3. Quality Filter (Simplified)
        hard_cut = (pl.col("debt_ratio") > 3.0) 
        if "capital_erosion_rate" in cols:
             hard_cut = hard_cut | (pl.col("capital_erosion_rate") > 50)
             
        if "relative_trend_score" in cols:
            rescue_condition = (pl.col("operating_income") > 0) | (pl.col("market_cap") > 5e11) | (pl.col("relative_trend_score") > 1.0)
        else:
            rescue_condition = (pl.col("operating_income") > 0) | (pl.col("market_cap") > 5e11)

        quality_pass = rescue_condition & (~hard_cut)
        
        # 4. Low Volatility Anomaly: 변동성 상위 10% 제거
        df = df.with_columns([
            pl.col("volatility_60d").rank("average", descending=True).over("date").alias("vol_rank"),
            pl.col("ticker").count().over("date").alias("daily_ticker_count")
        ])
        df = df.with_columns([
            (pl.col("vol_rank") / pl.col("daily_ticker_count")).alias("vol_percentile")
        ])
        vol_filter = (pl.col("vol_percentile") > 0.1) # 상위 10% 제거
        
        # 통합 필터 적용
        df = df.filter(liq_filter & sector_filter & quality_pass & vol_filter).drop(["sector_count", "daily_ticker_count", "vol_rank", "vol_percentile"])
        
        return df
