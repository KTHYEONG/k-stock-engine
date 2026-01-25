import polars as pl
import numpy as np
from src.data.preprocessors.base_processor import BaseProcessor

class TechProcessor(BaseProcessor):
    """
    기술적 지표 (Technical Indicators) 전처리기
    
    Features:
    1. Log Returns: ln(Close / Close_shift)
    2. Volatility: Rolling Std of Log Returns
    3. Price Disparity: Close / Moving Average (Stationary Trend)
    4. Volume Ratio: Volume / Moving Average Volume
    5. Intraday Volatility: (High - Low) / Open
    6. Amihud Illiquidity: Mean(|Return| / Trading Value) - Liquidity Risk
    """
    
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # Pre-check: Ensure sorted by date for rolling calc
        df = df.sort(["ticker", "date"])
        
        # 1. Log Returns (1, 5, 20, 60, 120 days)
        # 로그 수익률은 정규분포에 가까워 ML 모델이 학습하기 좋음
        windows = [1, 5, 20, 60, 120]
        exprs = []
        for w in windows:
            exprs.append(
                (pl.col("close") / pl.col("close").shift(w)).log().over("ticker").alias(f"log_return_{w}d")
            )
        
        df = df.with_columns(exprs)
        
        # 2. Volatility (20, 60 days) - Standard Deviation of Daily Log Return
        # 일별 로그 수익률(1d)을 기준으로 변동성 계산
        vol_windows = [20, 60]
        exprs = []
        for w in vol_windows:
            exprs.append(
                pl.col("log_return_1d").rolling_std(window_size=w).over("ticker").alias(f"volatility_{w}d")
            )
        
        df = df.with_columns(exprs)
        
        # 3. Price Disparity (이격도) = Close / MA(n)
        # 이동평균선 대비 현재 주가 위치 (Stationarity 확보)
        ma_windows = [5, 20, 60, 120]
        exprs = []
        for w in ma_windows:
            ma_col = pl.col("close").rolling_mean(window_size=w).over("ticker")
            exprs.append(
                (pl.col("close") / ma_col).alias(f"disparity_{w}d")
            )
            
        df = df.with_columns(exprs)
        
        # 4. Volume Ratio (거래량 이격도)
        # 거래량이 평소 대비 얼마나 터졌는지 확인 (수급의 강도)
        vol_ratio_windows = [5, 20, 60]
        exprs = []
        for w in vol_ratio_windows:
            vol_ma = pl.col("volume").rolling_mean(window_size=w).over("ticker")
            # 0으로 나누기 방지
            exprs.append(
                (pl.col("volume") / vol_ma.fill_null(1).replace(0, 1)).alias(f"volume_ratio_{w}d")
            )
            
        df = df.with_columns(exprs)
        
        # 5. Intraday Volatility (일중 변동성)
        # (High - Low) / Open
        df = df.with_columns([
            ((pl.col("high") - pl.col("low")) / pl.col("open").replace(0, np.nan).fill_null(pl.col("close")))
            .alias("intraday_vol")
        ])
        
        # 6. Amihud Illiquidity (유동성 충격 지표)
        # Mean(|Return| / Trading Value) over 20 days
        # 값이 클수록 적은 거래대금으로 가격이 크게 변함 (Liquidity Risk)
        # Trading Value 단위 보정 (보통 원화 그대로 쓰면 값이 너무 작아짐) -> 여기서는 비율이므로 상관 없으나 1e9 등으로 나누기도 함.
        # 여기서는 Raw Value 사용하되, Z-Score나 Rank로 추후 변환됨.
        
        # Abs Return
        abs_ret = pl.col("log_return_1d").abs()
        # Trading Value (0 방지)
        tv = pl.col("trading_value").fill_null(0).replace(0, np.inf) # 0이면 0으로 수렴하게, 분모니까 inf로? 
        # 분모가 0인 경우 Amihud는 정의되지 않음(거래 없음). 0으로 처리하는게 안전.
        # Trading Value가 0이면 Illiquidity는 0으로 가정(거래가 없어서 충격도 없음? 아니면 무한대? 보통 거래정지 종목이므로 0 처리 후 필터링됨)
        
        amihud_daily = (abs_ret / pl.col("trading_value")).fill_nan(0).fill_null(0).replace(float('inf'), 0)
        
        # Rolling Mean 20d
        # trading_value가 너무 크므로 결과값이 매우 작을 수 있음 (e.g. 1e-10). 
        # 가독성을 위해 1e9(10억) 곱해서 저장할 수도 있으나, Rank 변환 예정이라 그대로 둠.
        df = df.with_columns([
            amihud_daily.rolling_mean(window_size=20).over("ticker").alias("amihud_20d")
        ])
        
        return df
