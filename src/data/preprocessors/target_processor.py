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

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
            
        # 1. 미래 수익률 계산 (t + n 일 수익률)
        # ticker별로 5일 뒤 종가를 현재로 당겨옴 (Look-forward)
        df = df.sort(["ticker", "date"]).with_columns([
            (pl.col("close").shift(-self.horizon).over("ticker") / pl.col("close"))
            .log()
            .alias(f"target_return_{self.horizon}d")
        ])
        
        # 2. 크로스섹션(날짜별) 순위 변환 (Percentile)
        # YetiRank는 '순위'를 학습하는 모델이므로 정규화된 순위가 label로 적합
        df = df.with_columns([
            ((pl.col(f"target_return_{self.horizon}d").rank("average") - 1) / (pl.col("ticker").count() - 1))
            .over("date")
            .alias("target_rank")
        ])
        
        # 타겟이 없는 데이터(최근 5일치)는 학습에 쓸 수 없으므로 제거하지 않고 로드 시 필터링하도록 둠.
        # (만약 여기서 제거하면 실시간 예측 시 데이터가 부족해짐)
        
        return df
