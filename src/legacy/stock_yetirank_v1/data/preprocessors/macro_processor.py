import polars as pl
from pathlib import Path
import logging
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.legacy.stock_yetirank_v1.data.preprocessors.base_processor import BaseProcessor
from src.legacy.stock_yetirank_v1.data.collectors.vix_collector import VixCollector

logger = logging.getLogger("preprocessors.macro")

class MacroProcessor(BaseProcessor):
    """
    글로벌 매크로 지표 처리기 (VIX 등)
    - T-1일 미국 VIX 종가를 T일 한국 시장 데이터에 Join
    - 미래 참조(Look-ahead Bias) 방지를 위해 match_date 기준 Join 수행
    """
    
    def __init__(self):
        super().__init__()
        self.vix_path = PROJECT_ROOT / "data" / "market_index" / "vix_daily.parquet"
        self.collector = VixCollector()

    def process(self, ldf: pl.LazyFrame) -> pl.LazyFrame:
        """
        VIX 데이터를 로드하여 시점 일치(join_asof) 반영 (강화 버전)
        """
        if not self.vix_path.exists():
            logger.info("VIX data not found. Collecting historical data...")
            self.collector.collect_vix(start_date="2016-01-01")
            
        if not self.vix_path.exists():
            logger.warning("VIX data missing. Skipping MacroProcessor.")
            return ldf

        # 1. VIX 데이터 준비: 타입 강제 및 정렬
        vix_ldf = (
            pl.scan_parquet(self.vix_path)
            .select([
                pl.col("match_date").dt.date().alias("date"), # 반드시 Date 타입으로 통일
                pl.col("vix_close")
            ])
            .drop_nulls()
            .sort("date")
            .unique(subset=["date"], keep="last") # 날짜당 하나의 값만 보장
        )

        # 2. 한국 종목 데이터 준비: 날짜 타입 확인 및 전역 정렬 (asof join 필수)
        # asof join은 'on' 컬럼 기준으로 전체가 정렬되어 있어야 함
        ldf = ldf.with_columns(pl.col("date").dt.date()) # 타입 통일
        
        # 기존에 잘못 저장된 vix_close 컬럼이 있다면 제거하여 _right 컬럼 생성 방지
        if "vix_close" in ldf.collect_schema().names():
            ldf = ldf.drop("vix_close")
        
        # 3. join_asof 수행
        # tolerance를 5일로 늘려 명절 연휴 등 긴 휴장기에도 대응
        ldf = ldf.sort("date").join_asof(
            vix_ldf,
            on="date",
            strategy="backward"
        )

        # [NEW] 횡단면 변별력을 위한 VIX Z-Score 변환 (20일 롤링)
        ldf = ldf.with_columns([
            ((pl.col("vix_close") - pl.col("vix_close").rolling_mean(20).over("ticker")) / 
             (pl.col("vix_close").rolling_std(20).over("ticker") + 1e-8)).alias("vix_zscore_20d")
        ]).drop("vix_close")

        # 4. 결측치 처리: 혹시나 매칭 안 된 극초반 데이터 처리
        ldf = ldf.with_columns(pl.col("vix_zscore_20d").forward_fill().over("ticker"))

        return ldf
