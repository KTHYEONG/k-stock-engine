import polars as pl
from src.legacy.stock_yetirank_v1.data.preprocessors.base_processor import BaseProcessor

class TargetProcessor(BaseProcessor):
    """
    타겟(Label) 생성 전처리기
    
    Target:
    1. future_return_5d: ln(Open_t+1+h / Open_t+1)  (T signal -> T+1 open execution)
    2. target_rank: Percentile rank of future_return_5d in cross-section
    """
    
    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # 1. 미래 수익률 및 위험 조정 수익률 계산
        # [CRITICAL UPDATE] 백테스트와 동일한 실전 수익률 타겟 설정
        # Signal(T) -> Entry(T+1 Open) -> Exit(T+5 Close) 
        # 타겟 = ln(Close_t+5 / Open_t+1)
        schema_cols = df.collect_schema().names()
        if "open" not in schema_cols or "close" not in schema_cols:
            raise ValueError(
                "TargetProcessor requires 'open' and 'close' columns. "
                "Regenerate source data with valid OHLC fields."
            )

        # T+5일 종가 / T+1일 시가 (5거래일 보유 수익률)
        df = df.sort(["ticker", "date"]).with_columns([
            (pl.col("close").shift(-self.horizon).over("ticker") / pl.col("open").shift(-1).over("ticker").replace(0, None))
            .log()
            .alias(f"target_return_{self.horizon}d")
        ])
        
        # 2. 크로스섹션(날짜별) 순위 변환 (Percentile)
        df = df.with_columns([
            ((pl.col(f"target_return_{self.horizon}d").rank("average") - 1) / 
             (pl.col("ticker").count() - 1).replace(0, None))
            .over("date")
            .alias("target_rank")
        ])
        
        return df
