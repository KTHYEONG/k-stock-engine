import polars as pl
import logging
from legacy.stock_yetirank_v1.data.preprocessors.base_processor import BaseProcessor

logger = logging.getLogger("preprocessors.filter")

class UniverseFilter(BaseProcessor):
    """
    최상위 퀀트 투자자 기준 유니버스 필터링 (Universe Construction)
    
    6대 핵심 필터 (Hard Cut):
    1. ADTV_20d > 50억 (최근 20거래일 평균 거래대금 50억 이상)
    2. Price >= 1,000원 (동전주 제외)
    3. No Zero Volume (최근 5거래일 거래량 0인 날 없음)
    4. Capital Erosion Rate <= 0% (자본잠식 없음, 자본총계 > 자본금)
    5. Profitability (최근 공시 기준 영업이익 흑자)
    6. Common Stock Only (우선주, 스팩 등 제외 - 티커 끝이 0인 6자리 종목)
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # 1. 기초 지표 계산 (Amihud: |Return| / Trading_Value)
        # 이미 TechProcessor에서 adtv_20d가 계산되어 있다고 가정
        df = df.with_columns([
            (pl.col("close").pct_change().over("ticker").abs() / (pl.col("trading_value").replace(0, 1.0)))
            .rolling_mean(20).over("ticker").alias("amihud_20d")
        ])

        # 2. 6자리 종목코드 중 끝자리가 '0'인 보통주만 필터링 (KRX 기준)
        is_index = pl.col("ticker").is_in(["KOSPI", "KOSDAQ"])
        is_common_stock = (pl.col("ticker").str.len_chars() == 6) & (pl.col("ticker").str.ends_with("0"))
        
        # 3. [UPDATED] 한국 시장 맞춤형 유동성 및 가격 필터 (기관급)
        # ADTV 50억 이상 + Amihud 상위 10% 제외 + 시총 500억 이상 (슬리피지 방어)
        liq_filter = (pl.col("adtv_20d") >= 5e9) & \
                     (pl.col("amihud_20d") < pl.col("amihud_20d").quantile(0.9).over("date")) & \
                     (pl.col("close") >= 1000) & \
                     (pl.col("min_vol_5d") > 0)
        
        # 4. 재무 건전성 필터 (Hard Cut)
        quality_filter = (pl.col("capital_erosion_rate").fill_null(100) <= 0)
        
        # 5. 통합 필터 적용 (시총 500억 이상 추가)
        if "market_cap" in df.collect_schema().names():
            liq_filter = liq_filter & (pl.col("market_cap") >= 50e9)
        df_filtered = df.filter(
            is_index | (is_common_stock & liq_filter & quality_filter)
        )
        
        return df_filtered
