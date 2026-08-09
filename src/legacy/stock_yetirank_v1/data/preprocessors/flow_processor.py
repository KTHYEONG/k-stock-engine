import polars as pl
from src.data.preprocessors.base_processor import BaseProcessor

class FlowProcessor(BaseProcessor):
    """
    수급 지표 (Investor Flow) 전처리기
    
    11대 핵심 피처 중 수급 관련 피처 생성:
    1. flow_intensity_20d: (최근 20일 외국인+기관 순매수 대금) / 시가총액
    2. flow_consensus_base: 외국인/기관 동시 순매수 강도 파악을 위한 베이스 피처
                            (CrossSectionalProcessor에서 rank 합산으로 'flow_consensus' 생성)
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.sort(["ticker", "date"])
        
        # Pre-check: Ensure required columns exist
        cols = df.collect_schema().names()
        required_cols = ["foreign_net_buy", "institution_net_buy", "market_cap"]
        
        for col in required_cols:
            if col not in cols:
                if col in ["foreign_net_buy", "institution_net_buy"]:
                    df = df.with_columns(pl.lit(0.0).alias(col))
                elif col == "market_cap":
                    # market_cap이 없으면 close로 임시 사용 (0 방지)
                    df = df.with_columns(pl.col("close").alias("market_cap"))

        # 1. Total Net Purchase (Ensure nulls are filled)
        df = df.with_columns([
            pl.col("foreign_net_buy").fill_null(0).alias("foreign_net_buy"),
            pl.col("institution_net_buy").fill_null(0).alias("institution_net_buy")
        ])

        df = df.with_columns([
            (pl.col("foreign_net_buy") + pl.col("institution_net_buy")).alias("net_purchase_total")
        ])

        # 2. flow_intensity_20d: 최근 20일 누적 메이저 순매수 / 시가총액
        np_cum_20d = pl.col("net_purchase_total").rolling_sum(window_size=20).over("ticker")
        df = df.with_columns([
            (np_cum_20d / pl.col("market_cap").replace(0, None)).alias("flow_intensity_20d")
        ])

        # flow_consensus 생성을 위해 Base 값 유지 (foreign_net_buy, institution_net_buy는 이미 존재)
        # CrossSectionalProcessor에서 이 두 값을 랭크화하고 합칠 예정.
        
        return df
