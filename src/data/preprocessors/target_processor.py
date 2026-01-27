import polars as pl
from src.data.preprocessors.base_processor import BaseProcessor

class TargetProcessor(BaseProcessor):
    """
    타겟(Label) 생성 전처리기
    
    Target: 
    1. future_return_5d: ln(Close_t+5 / Close_t)
    2. target_rank: Percentile rank of future_return_5d in cross-section
    """
    
    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # 1. 미래 수익률 및 위험 조정 수익률 계산
        # 단순히 많이 오르는 것보다, 변동성 대비 안정적으로 오르는 종목을 상위로 둡니다.
        # volatility_20d는 TechProcessor에서 이미 계산되었다고 가정합니다.
        
        df = df.sort(["ticker", "date"]).with_columns([
            (pl.col("close").shift(-self.horizon).over("ticker") / pl.col("close").replace(0, None))
            .log()
            .alias(f"raw_target_return")
        ])
        
        # [MODIFIED] 소액 국내 투자자 맞춤형 타겟 (수익률 - 0.5 * 변동성)
        # 변동성에 너무 민감하게 반응하여 수익 기회를 놓치는 것을 방지하되,
        # 최소한의 리스크 관리(극심한 변동성 제거)는 유지합니다.
        df = df.with_columns([
            (pl.col("raw_target_return") - (0.5 * pl.col("volatility_20d").fill_null(0.0))).alias(f"target_return_{self.horizon}d")
        ])
        
        # 2. 크로스섹션(날짜별) 순위 변환 (Percentile)
        df = df.with_columns([
            ((pl.col(f"target_return_{self.horizon}d").rank("average") - 1) / 
             (pl.col("ticker").count() - 1).replace(0, None))
            .over("date")
            .alias("target_rank")
        ])
        
        # 중간 계산 컬럼 제거
        df = df.drop("raw_target_return")
        
        return df
