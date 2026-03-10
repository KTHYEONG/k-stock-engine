import yfinance as yf
import polars as pl
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Optional

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("data.collectors.vix")

class VixCollector:
    """
    미국 CBOE VIX 지수 수집기 (yfinance 기반)
    과거 데이터 백필 및 실시간성 수집 지원
    """
    
    def __init__(self):
        self.ticker = "^VIX"
        self.output_dir = PROJECT_ROOT / "data" / "market_index"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_dir / "vix_daily.parquet"

    def collect_vix(self, start_date: str = "2016-01-01") -> pl.DataFrame:
        """
        VIX 데이터를 수집하고 로컬에 저장
        
        Args:
            start_date: 수집 시작 날짜 (YYYY-MM-DD)
            
        Returns:
            pl.DataFrame: 수집된 데이터
        """
        logger.info(f"🚀 Collecting VIX data from {start_date}...")
        
        try:
            # yfinance를 통해 데이터 다운로드
            df_raw = yf.download(self.ticker, start=start_date)
            
            if df_raw.empty:
                logger.warning("No VIX data found.")
                return pl.DataFrame()
            
            # 인덱스(Date)를 컬럼으로 변환하고 필요한 컬럼만 선택
            df_raw = df_raw.reset_index()
            # yfinance v0.2.x 이상에서는 컬럼이 MultiIndex일 수 있음
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            
            df_vix = df_raw[['Date', 'Close']].copy()
            df_vix.columns = ['vix_raw_date', 'vix_close']
            
            # Polars 변환 및 날짜 처리
            # vix_raw_date: 미국 시장 거래일
            # match_date: 한국 시장 매칭일 (vix_raw_date + 1일)
            vix_pl = pl.from_pandas(df_vix).with_columns([
                pl.col("vix_raw_date").dt.cast_time_unit("us").alias("vix_raw_date"),
                (pl.col("vix_raw_date") + pl.duration(days=1)).dt.date().alias("match_date") # Date 타입 강제
            ])
            
            # 기존 데이터가 있다면 병합 (중복 제거)
            if self.file_path.exists():
                existing_df = pl.read_parquet(self.file_path)
                vix_pl = pl.concat([existing_df, vix_pl]).unique(subset=["vix_raw_date"]).sort("vix_raw_date")
            
            vix_pl.write_parquet(self.file_path)
            logger.info(f"✅ VIX data saved to {self.file_path} (Total: {len(vix_pl)} rows)")
            
            return vix_pl
            
        except Exception as e:
            logger.error(f"Failed to collect VIX data: {e}")
            return pl.DataFrame()

    def get_vix_for_date(self, target_date: str) -> Optional[float]:
        """
        특정 날짜(한국 기준)에 사용할 수 있는 전일 VIX 종가를 반환
        
        Args:
            target_date: 한국 기준 날짜 (YYYY-MM-DD 또는 YYYYMMDD)
        """
        if not self.file_path.exists():
            self.collect_vix()
            
        try:
            # 날짜 형식 표준화
            if len(target_date) == 8:
                dt = datetime.strptime(target_date, "%Y%m%d")
            else:
                dt = datetime.strptime(target_date, "%Y-%m-%d")
            
            df = pl.read_parquet(self.file_path)
            
            # match_date가 target_date보다 작거나 같은 것 중 가장 최근 것
            # (주말/공휴일 등으로 match_date가 정확히 일치하지 않을 수 있음)
            result = df.filter(pl.col("match_date") <= dt).sort("match_date", descending=True).head(1)
            
            if not result.is_empty():
                vix_val = result["vix_close"][0]
                logger.info(f"VIX for {target_date}: {vix_val:.2f} (from US date {result['vix_raw_date'][0]})")
                return float(vix_val)
            else:
                logger.warning(f"No VIX data found for or before {target_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving VIX for date {target_date}: {e}")
            return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VIX Data Collector (yfinance)")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date for collection (YYYY-MM-DD)")
    parser.add_argument("--test_date", type=str, default="2024-03-04", help="Sample date to test retrieval (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    collector = VixCollector()
    
    # 1. 지정된 기간 데이터 수집 (기본 2016-01-01부터)
    collector.collect_vix(start_date=args.start)
    
    # 2. 특정 날짜 조회 테스트
    print(f"\n--- Testing Retrieval for {args.test_date} ---")
    val = collector.get_vix_for_date(args.test_date)
    if val:
        print(f"Retrieved VIX Value: {val:.2f}")
