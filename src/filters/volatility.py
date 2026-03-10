import polars as pl
import logging

logger = logging.getLogger("filters.volatility")

def apply_volatility_filter(df: pl.DataFrame) -> pl.DataFrame:
    """
    최상위 퀀트 투자자 기준 거래 정지 및 상태 필터
    
    1. Zero Volume Check:
       - 최근 5거래일 내 거래량이 0인 날이 하루라도 있으면 제외 (min_vol_5d > 0)
    """
    initial_count = len(df)
    
    # 1. Zero Volume Check (Halt proxy)
    # TechProcessor에서 계산된 min_vol_5d 사용
    if "min_vol_5d" in df.columns:
        vol_mask = (pl.col("min_vol_5d").fill_null(0) > 0)
    else:
        # min_vol_5d가 없으면 당일 거래량이라도 체크
        if "volume" in df.columns:
             vol_mask = (pl.col("volume").fill_null(0) > 0)
        else:
            logger.warning("Volume info not found. Skipped.")
            vol_mask = pl.lit(True)
        
    # Apply Filters
    df_filtered = df.filter(vol_mask)
    
    filtered_count = initial_count - len(df_filtered)
    if filtered_count > 0:
        logger.info(f"Volatility/Halt Filter dropped {filtered_count} stocks.")
        
    return df_filtered
