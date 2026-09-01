import polars as pl
import logging
from typing import Dict, Any

from legacy.stock_yetirank_v1.filters.quality import apply_quality_filter
from legacy.stock_yetirank_v1.filters.liquidity import apply_liquidity_filter
from legacy.stock_yetirank_v1.filters.volatility import apply_volatility_filter

logger = logging.getLogger("filters.pipeline")

class UniverseFilter:
    """
    Layer 1: Universe Filter 통합 파이프라인
    
    Quality -> Liquidity -> Volatility 순으로 필터를 적용하여
    투자 부적격 종목을 사전에 제거합니다.
    """
    
    def __init__(self):
        pass
        
    def apply_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        모든 필터를 순차적으로 적용합니다.
        
        Args:
            df (pl.DataFrame): 일별 종목 데이터 (Snapshot)
            
        Returns:
            pl.DataFrame: 필터링된 유니버스
        """
        initial_count = len(df)
        logger.info(f"Starting Universe Filter. Initial stocks: {initial_count}")
        
        # 1. Volatility/Status Filter (거래정지 등 명백한 불가 종목 먼저 제거)
        df = apply_volatility_filter(df)
        
        # 2. Quality Filter (재무 건전성)
        df = apply_quality_filter(df)
        
        # 3. Liquidity Filter (유동성)
        df = apply_liquidity_filter(df)
        
        final_count = len(df)
        dropped_count = initial_count - final_count
        drop_rate = (dropped_count / initial_count * 100) if initial_count > 0 else 0
        
        logger.info(f"Universe Filter Completed. Final stocks: {final_count} (Dropped {dropped_count}, {drop_rate:.1f}%)")
        
        return df
        
    def analyze_filtering(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        필터링 효과를 분석하기 위해 각 단계별 통계를 반환합니다. (디버깅용)
        """
        stats = {"initial": len(df)}
        
        df_vol = apply_volatility_filter(df)
        stats["after_volatility"] = len(df_vol)
        
        df_qual = apply_quality_filter(df_vol)
        stats["after_quality"] = len(df_qual)
        
        df_liq = apply_liquidity_filter(df_qual)
        stats["after_liquidity"] = len(df_liq)
        
        return stats
