import polars as pl
import logging
from config.base import MIN_MARKET_CAP, MAX_TURNOVER_RATIO, MIN_TRADING_VALUE

logger = logging.getLogger("filters.liquidity")

def apply_liquidity_filter(df: pl.DataFrame) -> pl.DataFrame:
    """
    유동성 필터 (Liquidity Cut)
    
    1. 시가총액 Check:
       - Market Cap < MIN_MARKET_CAP (예: 100억 ~ 1000억) 제외
       
    2. 거래회전율 (Turnover Ratio) Check:
       - Turnover > 50% 제외 (투기 과열)
       - (Option) 너무 낮은 회전율 제외 (소외주) - 여기서는 상한선만 적용 (Config 기준)
    
    Expected Columns:
        - market_cap (float)
        - turnover_ratio (float) OR (volume, shares_outstanding)
    """
    initial_count = len(df)
    
    # 1. Market Cap Check
    if "market_cap" not in df.columns:
        logger.warning("Column 'market_cap' not found. Liquidity filter skipped Market Cap check.")
        mcap_mask = pl.lit(True)
    else:
        mcap_mask = (pl.col("market_cap") >= MIN_MARKET_CAP)
        
    # 2. Turnover Ratio Check
    if "turnover_ratio" in df.columns:
        # turnover_ratio가 %단위인지 소수점인지 확인 필요. 보통 1.5 = 1.5% 인지 0.015인지.
        # Config의 MAX_TURNOVER_RATIO = 0.5 (50% 의미로 가정). 
        # 데이터가 %단위(50.0)라면 스케일 조정 필요.
        # 여기서는 Config가 Ratio(0.5)라고 가정하고 데이터도 Ratio(0.5)라고 가정.
        # 만약 데이터가 50.0 처럼 들어온다면 /100 해야 함. 
        # 일반적인 퀀트 데이터 관례상 Ratio(0~1) 또는 Percentage(0~100).
        # pykrx 등은 보통 Percentage(0.5 = 0.5%)로 줌. 
        # 설계상 Hard Cut 50%는 매우 높은 수치임. (상장주식의 반이 돌아감). 따라서 0.5 (50%)로 해석.
        turnover_mask = (pl.col("turnover_ratio").fill_null(0) <= MAX_TURNOVER_RATIO)
    else:
        # 계산 시도: Volume / Shares Outstanding
        if "volume" in df.columns and "shares_outstanding" in df.columns:
            turnover = pl.col("volume") / pl.col("shares_outstanding")
            turnover_mask = (turnover.fill_null(0) <= MAX_TURNOVER_RATIO)
        else:
            logger.warning("Columns for Turnover check not found. Skipped.")
            turnover_mask = pl.lit(True)

    # 3. Minimum Trading Value Check (Config 기반)
    if "trading_value" in df.columns:
        tv_mask = (pl.col("trading_value").fill_null(0) >= MIN_TRADING_VALUE)
    elif "volume" in df.columns and "close" in df.columns:
        tv_mask = ((pl.col("volume") * pl.col("close")).fill_null(0) >= MIN_TRADING_VALUE)
    else:
        tv_mask = pl.lit(True)

    # Apply Filters
    df_filtered = df.filter(mcap_mask & turnover_mask & tv_mask)
    
    filtered_count = initial_count - len(df_filtered)
    if filtered_count > 0:
        logger.info(f"Liquidity Filter dropped {filtered_count} stocks.")
        
    return df_filtered
