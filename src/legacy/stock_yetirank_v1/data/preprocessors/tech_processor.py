import polars as pl
from src.legacy.stock_yetirank_v1.data.preprocessors.base_processor import BaseProcessor

class TechProcessor(BaseProcessor):
    """
    기술적 지표 (Technical Indicators) 전처리기
    
    11대 핵심 피처 중 가격/거래량 관련 피처 생성:
    1. overnight_ret: (당일 시가 / 전일 종가) - 1
    2. intraday_ret: (당일 종가 / 당일 시가) - 1
    3. ret_2_5d: (전일 종가 / 5일 전 종가) - 1
    4. ret_6_20d: (5일 전 종가 / 20일 전 종가) - 1
    5. ret_21_60d: (20일 전 종가 / 60일 전 종가) - 1
    6. disparity_120d: (종가 / 120일 이동평균) (CrossSectional에서 랭크화)
    7. volatility_20d: 20일 일간 수익률 표준편차 (CrossSectional에서 랭크화)
    8. vol_regime: 20일 변동성 / 60일 변동성
    9. volume_shock: 당일 거래량 / 20일 평균 거래량
    
    기타 유지 필터 지표:
    - adtv_20d: 20일 평균 거래대금
    - min_vol_5d: 5일 최소 거래량
    """
    
    def process(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.sort(["ticker", "date"])
        
        exprs = []
        
        # --- 1. 가격 모멘텀 피처 (Orthogonalized Returns) ---
        # 1. overnight_ret: ln(open / close_shift1)
        exprs.append(
            (pl.col("open") / pl.col("close").shift(1).over("ticker")).log().alias("overnight_ret")
        )
        
        # 2. intraday_ret: ln(close / open)
        # open이 0인 경우 방어
        exprs.append(
            (pl.col("close") / pl.col("open").replace(0, None)).log().alias("intraday_ret")
        )
        
        # 3. ret_2_5d: ln(close_shift1 / close_shift5)
        exprs.append(
            (pl.col("close").shift(1).over("ticker") / pl.col("close").shift(5).over("ticker")).log().alias("ret_2_5d")
        )
        
        # 4. ret_6_20d: ln(close_shift5 / close_shift20)
        exprs.append(
            (pl.col("close").shift(5).over("ticker") / pl.col("close").shift(20).over("ticker")).log().alias("ret_6_20d")
        )
        
        # 5. ret_21_60d: ln(close_shift20 / close_shift60)
        exprs.append(
            (pl.col("close").shift(20).over("ticker") / pl.col("close").shift(60).over("ticker")).log().alias("ret_21_60d")
        )
        
        # 6. disparity_120d (추후 랭크용)
        ma120 = pl.col("close").rolling_mean(window_size=120).over("ticker")
        exprs.append(
            (pl.col("close") / ma120.replace(0, None)).alias("disparity_120d")
        )
        
        # --- 2. 변동성 (Volatility) 피처 ---
        # 기준이 되는 일일 수익률 계산 (기존 필터 등에 쓰일 수 있으므로 유지 또는 즉시 계산)
        log_ret_1d = (pl.col("close") / pl.col("close").shift(1).over("ticker")).log()
        
        vol_20d = log_ret_1d.rolling_std(window_size=20).over("ticker")
        vol_60d = log_ret_1d.rolling_std(window_size=60).over("ticker")
        
        exprs.append(vol_20d.alias("volatility_20d"))
        exprs.append(vol_60d.alias("volatility_60d"))
        
        # 7. vol_regime: vol_20d / vol_60d
        exprs.append(
            (vol_20d / vol_60d.replace(0, None)).alias("vol_regime")
        )
        
        # --- 3. 거래량 (Volume) & 필터용 피처 ---
        # 8. volume_shock: 당일 거래량 / 20일 평균 거래량
        vol_ma20 = pl.col("volume").rolling_mean(window_size=20).over("ticker")
        exprs.append(
            (pl.col("volume") / vol_ma20.replace(0, None)).alias("volume_shock")
        )
        
        # Filter용: adtv_20d, min_vol_5d
        if "trading_value" not in df.collect_schema().names():
            df = df.with_columns((pl.col("close") * pl.col("volume")).alias("trading_value"))
            
        exprs.append(
            pl.col("trading_value").rolling_mean(window_size=20).over("ticker").alias("adtv_20d")
        )
        exprs.append(
            pl.col("volume").rolling_min(window_size=5).over("ticker").alias("min_vol_5d")
        )
        
        df = df.with_columns(exprs)
        # [Microstructure Alpha] 주포 매집 및 모멘텀 품질 지표
        df = df.with_columns([
            # 1. 일중 변동성 비대칭 (Up_Vol/Down_Vol): 양봉 변동성 합 / 음봉 변동성 합
            # 매집주 포착: 오를 때 힘차게 오르고 내릴 때 마르는 특성 정량화 (when/then으로 길이 유지)
            (
                pl.when(pl.col("close") >= pl.col("open"))
                .then(pl.col("high") - pl.col("open"))
                .otherwise(0.0)
                .rolling_sum(20).over("ticker")
                /
                pl.when(pl.col("close") < pl.col("open"))
                .then(pl.col("open") - pl.col("low"))
                .otherwise(0.0)
                .rolling_sum(20).over("ticker")
                .replace(0, 1.0)
            ).alias("vol_asymmetry_20d"),
            
            # 2. 시가/종가 위치 강도 (Close-to-High Ratio): 윗꼬리 억제력
            # 진성 모멘텀: 장 막판까지 매수세가 지배하여 종가를 고가에 붙이는 힘
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 0.0001))
            .rolling_mean(10).over("ticker")
            .alias("close_high_ratio_10d"),
            
            # 3. [NEW] Information Ratio (Path Dependency): 모멘텀의 매끄러움 (Sharpe of returns)
            # 노이즈가 적고 꾸준히 우상향하는 진짜 모멘텀을 필터링 (기관 선호 피처)
            (pl.col("close").pct_change().rolling_mean(20).over("ticker") / 
             (pl.col("close").pct_change().rolling_std(20).over("ticker") + 1e-8)
            ).alias("info_ratio_20d"),
            
            # 4. [NEW] Volume Price Trend (VPT): 거래량 동반 추세 강도
            # 단순 수익률이 아닌 거래량이 실린 진짜 수익률 누적
            (pl.col("volume") * pl.col("close").pct_change().fill_null(0)).rolling_sum(20).over("ticker").alias("vpt_20d")
        ])

        return df
