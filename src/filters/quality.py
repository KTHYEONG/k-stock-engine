import polars as pl
import logging

logger = logging.getLogger("filters.quality")

def apply_quality_filter(df: pl.DataFrame) -> pl.DataFrame:
    """
    최상위 퀀트 투자자 기준 재무 건전성 필터 (Quality Hard Cut)
    
    1. 자본잠식 Check:
       - 자본잠식률 <= 0 (자본총계 > 자본금)
    2. 수익성 Check:
       - 영업이익 > 0 (흑자 기업)
    """
    initial_count = len(df)
    
    # 1. 자본잠식 Check (capital_erosion_rate <= 0)
    if "capital_erosion_rate" in df.columns:
        erosion_mask = (pl.col("capital_erosion_rate").fill_null(100) <= 0)
    else:
        logger.warning("Column 'capital_erosion_rate' not found. Skipped.")
        erosion_mask = pl.lit(True)
        
    # 2. 영업이익 Check (operating_income > 0)
    if "operating_income" in df.columns:
        profit_mask = (pl.col("operating_income").fill_null(-1) > 0)
    else:
        logger.warning("Column 'operating_income' not found. Skipped.")
        profit_mask = pl.lit(True)
        
    # Apply Filters (Rescue 로직 전면 폐기)
    df_filtered = df.filter(erosion_mask & profit_mask)
    
    filtered_count = initial_count - len(df_filtered)
    if filtered_count > 0:
        logger.info(f"Quality Filter dropped {filtered_count} stocks.")
        
    return df_filtered
