import polars as pl
import logging

logger = logging.getLogger("filters.liquidity")

def apply_liquidity_filter(df: pl.DataFrame) -> pl.DataFrame:
    """
    최상위 퀀트 투자자 기준 유동성 및 가격 필터
    
    1. ADTV_20d > 50억 (평균 거래대금)
    2. Price >= 1,000 (동전주 제외)
    3. Common Stock Only (티커 끝이 0인 6자리 종목)
    """
    initial_count = len(df)
    
    # 1. Ticker Check (우선주/스팩 제외 - KRX)
    if "ticker" in df.columns:
        ticker_mask = (pl.col("ticker").str.len_chars() == 6) & (pl.col("ticker").str.ends_with("0"))
    else:
        logger.warning("Column 'ticker' not found. Skipped.")
        ticker_mask = pl.lit(True)
        
    # 2. ADTV_20d Check (TechProcessor에서 미리 계산되어야 함)
    if "adtv_20d" in df.columns:
        adtv_mask = (pl.col("adtv_20d").fill_null(0) >= 5e9)
    else:
        # adtv_20d가 없으면 trading_value로라도 체크 시도 (임시)
        if "trading_value" in df.columns:
             adtv_mask = (pl.col("trading_value").fill_null(0) >= 5e9)
        else:
            logger.warning("ADTV info not found. Skipped.")
            adtv_mask = pl.lit(True)

    # 3. Price Check
    if "close" in df.columns:
        price_mask = (pl.col("close") >= 1000)
    else:
        logger.warning("Column 'close' not found. Skipped.")
        price_mask = pl.lit(True)

    # Apply Filters
    df_filtered = df.filter(ticker_mask & adtv_mask & price_mask)
    
    filtered_count = initial_count - len(df_filtered)
    if filtered_count > 0:
        logger.info(f"Liquidity Filter dropped {filtered_count} stocks.")
        
    return df_filtered
