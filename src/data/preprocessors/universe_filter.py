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
    
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        initial_size = len(df)
        
        # 0. 필수 피처 확인 (부족하면 필터링 불가하므로 필터링 bypass 또는 보수적 처리)
        # 여기서는 이미 앞단에서 피처가 생성되었다고 가정.
        
        # 1. Liquidity & Microstructure
        # - Penny Stocks (< 1,000 KRW)
        # - Min Trading Value (5-day average >= 1B KRW)
        df = df.sort(["ticker", "date"])
        df = df.with_columns([
            pl.col("trading_value").rolling_mean(window_size=5).over("ticker").alias("avg_trading_value_5d")
        ])
        
        liq_filter = (pl.col("close") >= 1000) & (pl.col("avg_trading_value_5d") >= 1e9)
        
        # 2. Sector Bias: 섹터 내 종목 수 5개 미만 필터링
        df = df.with_columns([
            pl.col("ticker").count().over(["date", "sector"]).alias("sector_count")
        ])
        sector_filter = (pl.col("sector_count") >= 5)
        
        # 3. Quality Filter with Turnaround Rescue
        # - 기본: 2년 연속 영업이익 적자 제거 (fiscal_year_count 등으로 판별해야 하나, 
        #   여기서는 단순화를 위해 현재 시점에 기록된 2년 적자 여부 필터 사용)
        #   만약 데이터에 'consecutive_deficit_years' 같은 컬럼이 없다면 로직 구현 필요.
        #   (현 단계에서는 financials.parquet에 과거 이력이 있다고 가정)
        
        # Rescue Logic:
        # 1) 최근 분기 흑자 (operating_income > 0)
        # 2) 대형주 (market_cap > 500B)
        # 3) 상대적 강세 (relative_trend_score > 1.0)
        
        # Hard Cut: Debt Ratio > 300% or Capital Erosion
        hard_cut = (pl.col("debt_ratio") > 3.0) | (pl.col("capital_erosion_rate") > 50)
        
        # Quality Filter (Simplified: assuming operating_income is from the latest fiscal/quarterly report)
        # 턴어라운드 구제 포함:
        # (적자가 아니거나) OR (구제조건 만족) AND (하드컷 미해당)
        if "relative_trend_score" in df.columns:
            rescue_condition = (pl.col("operating_income") > 0) | (pl.col("market_cap") > 5e11) | (pl.col("relative_trend_score") > 1.0)
        else:
            rescue_condition = (pl.col("operating_income") > 0) | (pl.col("market_cap") > 5e11)

        quality_pass = rescue_condition & (~hard_cut)
        
        # 4. Low Volatility Anomaly: 변동성 상위 10% 제거
        # 날짜별 변동성 순위 계산
        df = df.with_columns([
            pl.col("volatility_60d").rank("average", descending=True).over("date").alias("vol_rank")
        ])
        # 전체 종목 수 대비 상위 10% (rank가 작을수록 변동성 큼)
        df = df.with_columns([
            (pl.col("vol_rank") / pl.col("ticker").count().over("date")).alias("vol_percentile")
        ])
        vol_filter = (pl.col("vol_percentile") > 0.1) # 상위 10% 제거
        
        # 통합 필터 적용
        df = df.filter(liq_filter & sector_filter & quality_pass & vol_filter)
        
        final_size = len(df)
        logger.info(f"Universe Filter Applied: {initial_size} -> {final_size} (Dropped {initial_size - final_size})")
        
        return df
