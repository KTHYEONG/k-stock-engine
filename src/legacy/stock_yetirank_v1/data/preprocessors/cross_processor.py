import polars as pl
from src.legacy.stock_yetirank_v1.data.preprocessors.base_processor import BaseProcessor

class CrossSectionalProcessor(BaseProcessor):
    """
    횡단면 연산 (Cross-Sectional) 전처리기
    
    13대 핵심 피처 중 횡단면 랭크 및 섹터 모멘텀 피처 생성:
    1. trend_120d_rank: 120일 이격도(disparity_120d)의 Daily Rank (0~1)
    2. vol_20d_rank: 20일 변동성(volatility_20d)의 Daily Rank (0~1)
    3. flow_consensus: 외국인 순매수 Rank + 기관 순매수 Rank 의 합산 (Daily 0~2)
    4. mcap_rank: 시가총액(market_cap)의 Daily Rank (0~1) - Size Factor
    5. sector_ret_5d: 해당 종목이 속한 섹터의 5일 수익률 평균 - Sector Momentum
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        
        # --- 1. 단일 피처 Daily Rank 계산 (0~1) ---
        rank_exprs = []
        
        # trend_120d_rank
        if "disparity_120d" in df.collect_schema().names():
            rank_exprs.append(
                (pl.col("disparity_120d").rank(descending=False).over("date") / 
                 pl.col("disparity_120d").count().over("date")).alias("trend_120d_rank")
            )
            
        # vol_20d_rank
        if "volatility_20d" in df.collect_schema().names():
            rank_exprs.append(
                (pl.col("volatility_20d").rank(descending=False).over("date") / 
                 pl.col("volatility_20d").count().over("date")).alias("vol_20d_rank")
            )
            
        # [NEW] mcap_rank (Size Factor)
        if "market_cap" in df.collect_schema().names():
            rank_exprs.append(
                (pl.col("market_cap").rank(descending=False).over("date") / 
                 pl.col("market_cap").count().over("date")).alias("mcap_rank")
            )
            
        # flow_consensus 부품: 외국인, 기관 순매수 Rank
        if "foreign_net_buy" in df.collect_schema().names():
            rank_exprs.append(
                (pl.col("foreign_net_buy").rank(descending=False).over("date") / 
                 pl.col("foreign_net_buy").count().over("date")).alias("foreign_net_buy_rank")
            )
            
        if "institution_net_buy" in df.collect_schema().names():
            rank_exprs.append(
                (pl.col("institution_net_buy").rank(descending=False).over("date") / 
                 pl.col("institution_net_buy").count().over("date")).alias("inst_net_buy_rank")
            )
            
        df = df.with_columns(rank_exprs)
        
        # --- 2. [NEW] 횡단면 Z-Score 정규화 (Cross-sectional Normalization) ---
        # 연속형 피처들의 절대적 수치를 당일 시장 평균 대비 상대적 수치(Standard Deviation)로 변환
        # [FIX] 매크로 지표(vix_close)는 시장 전체 공통값이므로 Z-Score에서 제외 (적용 시 0.0으로 파괴됨)
        norm_cols = [
            "overnight_ret", "intraday_ret", "ret_2_5d", "ret_6_20d", "ret_21_60d",
            "vol_regime", "volume_shock", "flow_intensity_20d", "vol_asymmetry_20d",
            "close_high_ratio_10d"
        ]
        
        actual_cols = [c for c in norm_cols if c in df.collect_schema().names()]
        if actual_cols:
            df = df.with_columns([
                ((pl.col(c) - pl.col(c).mean().over("date")) / (pl.col(c).std().over("date") + 1e-8)).alias(c)
                for c in actual_cols
            ])

        # --- 3. 파생 피처 최종 계산 ---
        # flow_consensus (Rank의 합, 0~2 범위)
        if "foreign_net_buy_rank" in df.collect_schema().names() and "inst_net_buy_rank" in df.collect_schema().names():
            df = df.with_columns([
                (pl.col("foreign_net_buy_rank").fill_null(0) + pl.col("inst_net_buy_rank").fill_null(0)).alias("flow_consensus")
            ])
            df = df.drop(["foreign_net_buy_rank", "inst_net_buy_rank"])
            
        # [NEW] sector_ret_5d (Sector Momentum)
        if "ret_2_5d" in df.collect_schema().names() and "sector" in df.collect_schema().names():
            df = df.with_columns([
                pl.col("ret_2_5d").mean().over(["date", "sector"]).alias("sector_ret_5d")
            ])
            
        return df
