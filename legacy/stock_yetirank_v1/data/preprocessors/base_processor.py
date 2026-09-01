import polars as pl
from abc import ABC, abstractmethod
from typing import List

class BaseProcessor(ABC):
    """피처 전처리기의 기저 추상 클래스"""
    
    @abstractmethod
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """데이터프레임을 입력받아 피처가 추가된 데이터프레임 반환"""
        pass

    def calculate_rolling_stats(self, df: pl.DataFrame, col: str, windows: List[int], metrics: List[str] = ["mean"]) -> pl.DataFrame:
        """주어진 컬럼에 대해 여러 윈도우 크기의 이동 평균/표준편차 등을 계산"""
        expressions = []
        for w in windows:
            for m in metrics:
                if m == "mean":
                    expressions.append(
                        pl.col(col).rolling_mean(window_size=w).over("ticker").alias(f"{col}_ma{w}")
                    )
                elif m == "std":
                    expressions.append(
                        pl.col(col).rolling_std(window_size=w).over("ticker").alias(f"{col}_std{w}")
                    )
        return df.with_columns(expressions)

    def apply_rank_transform(self, df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
        """날짜별로 주어진 컬럼들에 대해 백분위 순위 변환(0~1) 적용"""
        expressions = []
        for col in cols:
            # 날짜별 그룹 내에서 순위 계산 (uniform: 0 to 1)
            expressions.append(
                ((pl.col(col).rank("average") - 1) / (pl.col(col).count() - 1))
                .over("date")
                .alias(f"{col}_rank")
            )
        return df.with_columns(expressions)
