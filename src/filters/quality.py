import polars as pl
import logging
from config.base import PB_HARD_CUT, CAPITAL_EROSION_HARD_CUT

logger = logging.getLogger("filters.quality")

def apply_quality_filter(df: pl.DataFrame) -> pl.DataFrame:
    """
    재무 건전성 필터 (Quality Cut)
    
    1. P/B 무결성 Check:
       - PBR이 0이거나 NaN인 경우 제외 (데이터 누락)
       - PBR < 0.1 제외 (비정상적으로 낮은 밸류에이션 = 리스크)
       
    2. 자본잠식 Check:
       - 자본잠식률 > 50% 제외 (관리종목 지정 위험)
       - 자본잠식률 = (자본금 - 자본총계) / 자본금 * 100
    
    Expected Columns:
        - pbr (float)
        - capital_erosion_rate (float) OR (capital, total_equity)
    """
    initial_count = len(df)
    
    # 1. PBR Check
    # PBR 컬럼이 없으면 계산 시도 또는 경고
    if "pbr" not in df.columns:
        logger.warning("Column 'pbr' not found. Quality filter skipped PBR check.")
        pbr_mask = pl.lit(True)
    else:
        # PBR > 0 (0은 결측 취급) AND PBR >= Hard Cut
        # null 값은 drop
        pbr_mask = (pl.col("pbr").is_not_null()) & \
                   (pl.col("pbr") > 0) & \
                   (pl.col("pbr") >= PB_HARD_CUT)

    # 2. Capital Erosion Check
    # 이미 계산된 'capital_erosion_rate'가 있다고 가정하거나 계산
    if "capital_erosion_rate" in df.columns:
        erosion_mask = (pl.col("capital_erosion_rate").fill_null(0) <= CAPITAL_EROSION_HARD_CUT)
    elif "capital" in df.columns and "total_equity" in df.columns:
        # 계산: (Capital - Total Equity) / Capital * 100
        # Total Equity가 null이면 위험한 것으로 간주할 수 있으나, 데이터 누락일 수 있음. 일단 보수적으로 Pass 시키거나 Drop.
        # 여기서는 Drop.
        erosion_rate = (pl.col("capital") - pl.col("total_equity")) / pl.col("capital") * 100
        erosion_mask = (erosion_rate.fill_null(0) <= CAPITAL_EROSION_HARD_CUT)
    else:
        logger.warning("Columns for Capital Erosion check not found. Skipped.")
        erosion_mask = pl.lit(True)
        
    # 3. [Korean Market Special] Momentum Rescue Logic (재무 불량주 구제)
    # 한국장 특성상 재무가 나빠도(자본잠식, 적자) 강력한 모멘텀/테마로 오르는 종목이 많음.
    # 따라서 "재무 탈락"이더라도 "모멘텀 최상위"라면 살려줌.
    
    rescue_mask = pl.lit(False)
    
    # 구조 요청 1: 강력한 상대 추세 (relative_trend_score > 0.8)
    if "relative_trend_score" in df.columns:
        rescue_mask = rescue_mask | (pl.col("relative_trend_score") > 0.8)
        
    # 구조 요청 2: 폭발적인 거래량 (volume_ratio >= 3.0)
    if "volume_ratio_20d" in df.columns:
        rescue_mask = rescue_mask | (pl.col("volume_ratio_20d") >= 3.0)
    
    # 재무 필터를 통과했거나(AND) 혹은 구조대(Rescue)가 왔거나(OR)
    final_mask = (pbr_mask & erosion_mask) | rescue_mask
        
    # Apply Filters
    df_filtered = df.filter(final_mask)
    
    filtered_count = initial_count - len(df_filtered)
    if filtered_count > 0:
        logger.info(f"Quality Filter dropped {filtered_count} stocks. (Rescued high momentum stocks)")
        
    return df_filtered
