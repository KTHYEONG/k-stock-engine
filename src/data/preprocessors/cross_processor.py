import polars as pl
from src.data.preprocessors.base_processor import BaseProcessor

class CrossSectionalProcessor(BaseProcessor):
    """
    횡단면 연산 (Cross-Sectional & Sector Relative) 전처리기
    
    주요 기능:
    1. Daily Percentile Rank: 시장 전체에서의 상대적 위치 (0~1)
    2. Sector Neutrality: 섹터 내 중앙값 대비 차이 (Rel-Value)
    """
    
    def __init__(self):
        # 랭킹을 적용할 피처 목록
        self.rank_features = [
            "volatility_20d", "volatility_60d",
            "volume_ratio_5d", "volume_ratio_20d", "volume_ratio_60d",
            "log_return_5d", "log_return_20d", "log_return_60d", "log_return_120d",
            "amihud_20d", "turnover_ratio", "relative_trend_score"
        ]
        
        # 섹터 상대화를 적용할 피처 목록
        self.sector_rel_features = [
            "ep_ratio", "bp_ratio", "sp_ratio", "op_ratio", "roe"
        ]

    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # 1. Daily Percentile Rank (0~1)
        # Polars의 over("date")와 rank()를 사용하여 효율적으로 계산
        rank_exprs = []
        for feat in self.rank_features:
            # 존재 여부 확인 후 랭킹 계산 (rank / count)
            # null은 최하위 혹은 무시되도록 처리
            rank_exprs.append(
                (pl.col(feat).rank(descending=False).over("date") / 
                 pl.col(feat).count().over("date")).alias(f"rank_{feat}")
            )
            
        df = df.with_columns(rank_exprs)
        
        # 2. Sector Relative (Stock_Value - Sector_Median)
        # 섹터 정보가 있고('sector' != 'Unknown'), 섹터당 종목 수가 일정 이상일 때 유의미함
        # 여기서는 단순 Median 차감을 적용
        sector_exprs = []
        for feat in self.sector_rel_features:
            sector_median = pl.col(feat).median().over(["date", "sector"])
            sector_exprs.append(
                (pl.col(feat) - sector_median).alias(f"rel_{feat}_sector")
            )
            
        df = df.with_columns(sector_exprs)
        
        return df
