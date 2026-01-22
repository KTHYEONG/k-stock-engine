import polars as pl
import logging

logger = logging.getLogger("filters.volatility")

def apply_volatility_filter(df: pl.DataFrame) -> pl.DataFrame:
    """
    변동성 및 상태 필터 (Volatility & Status Cut)
    
    1. 거래 정지 점검 (Trading Halt Proxy):
       - Volume == 0 인 종목 제외
       
    2. (Optional) 관리종목/환기종목 Flag:
       - 데이터에 'is_managed' 등의 플래그가 있다면 제외
    
    Expected Columns:
        - volume (int)
    """
    initial_count = len(df)
    
    # 1. Trading Halt Check (Volume = 0)
    if "volume" in df.columns:
        # 거래량이 0이 아니고 Null이 아닌 것만 남김
        vol_mask = (pl.col("volume").is_not_null()) & (pl.col("volume") > 0)
    else:
        logger.warning("Column 'volume' not found. Volatility filter skipped Halt check.")
        vol_mask = pl.lit(True)
        
    # 2. Admin Issue Check (Optional)
    # 데이터셋에 'admin_issue'(관리종목여부)가 있다고 가정
    if "admin_issue" in df.columns:
        # 1/True/Risk라면 제외
        issue_mask = (pl.col("admin_issue").fill_null(0) == 0)
    else:
        issue_mask = pl.lit(True)
        
    # Apply Filters
    df_filtered = df.filter(vol_mask & issue_mask)
    
    filtered_count = initial_count - len(df_filtered)
    if filtered_count > 0:
        logger.info(f"Volatility Filter dropped {filtered_count} stocks.")
        
    return df_filtered
