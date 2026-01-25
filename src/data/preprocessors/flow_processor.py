import polars as pl
from src.data.preprocessors.base_processor import BaseProcessor

class FlowProcessor(BaseProcessor):
    """
    수급 지표 (Investor Flow) 전처리기
    
    Features:
    1. NP_mkt_cap: (Foreign + Inst Net Purchase) / Market Cap
    2. NP_vol: (Foreign + Inst Net Purchase) / Trading Value
    3. Z_flow: Z-Score of 60-day cumulative Net Purchase
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Pre-check: Ensure required columns exist (Lazy friendly)
        cols = df.collect_schema().names()
        required_cols = ["foreign_net_buy", "institution_net_buy", "market_cap", "trading_value"]
        
        for col in required_cols:
            if col not in cols:
                if col in ["foreign_net_buy", "institution_net_buy"]:
                    df = df.with_columns(pl.lit(0.0).alias(col))
                elif col == "market_cap":
                    # market_cap이 없으면 close를 임시로 사용 (0 방지)
                    df = df.with_columns(pl.col("close").alias("market_cap"))
                elif col == "trading_value":
                    # 거래대금이 없으면 가격 * 거래량으로 계산
                    df = df.with_columns((pl.col("close") * pl.col("volume")).alias("trading_value"))

        # 1. Total Net Purchase
        df = df.with_columns([
            (pl.col("foreign_net_buy") + pl.col("institution_net_buy")).alias("net_purchase_total")
        ])

        # 2. NP_mkt_cap (영향력)
        # Net Purchase / Market Cap
        df = df.with_columns([
            (pl.col("net_purchase_total") / pl.col("market_cap").replace(0, None)).alias("np_mkt_cap")
        ])

        # 3. NP_vol (긴급성)
        # Net Purchase / Trading Value
        df = df.with_columns([
            (pl.col("net_purchase_total") / pl.col("trading_value").replace(0, None)).alias("np_vol")
        ])

        # 4. Z_flow (매집 강도)
        # 누적 수급 (60일 합계)의 Z-Score
        # Step A: 60일 누적 합계 계산
        df = df.with_columns([
            pl.col("net_purchase_total").rolling_sum(window_size=60).over("ticker").alias("np_cum_60d")
        ])
        
        # Step B: 누적 합계의 Z-Score (과거 데이터 대비 현재 누적액의 희소성)
        # Z = (x - mean) / std
        df = df.with_columns([
            ((pl.col("np_cum_60d") - pl.col("np_cum_60d").rolling_mean(window_size=120).over("ticker")) / 
             pl.col("np_cum_60d").rolling_std(window_size=120).over("ticker").replace(0, None))
            .alias("z_flow")
        ])

        return df
