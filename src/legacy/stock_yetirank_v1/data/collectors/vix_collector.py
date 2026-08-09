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

from src.legacy.stock_yetirank_v1.utils.logger import setup_logger

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

    def collect_vix(self, start_date: Optional[str] = None) -> pl.DataFrame:
        """
        VIX 데이터를 수집하고 로컬에 저장 (기존 데이터 이후부터 자동 수집)
        
        Args:
            start_date: 수집 시작 날짜 (YYYY-MM-DD). 생략 시 기존 파일의 마지막 날짜 익일부터 수집.
            
        Returns:
            pl.DataFrame: 전체 VIX 데이터 (기존 + 신규)
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 1. 기존 데이터 로드 및 시작일 결정
        existing_df = pl.DataFrame()
        if self.file_path.exists():
            existing_df = pl.read_parquet(self.file_path)
            if start_date is None and not existing_df.is_empty():
                last_db_date = existing_df["vix_raw_date"].max()
                if isinstance(last_db_date, (datetime, pd.Timestamp)):
                    # 다음 날부터 수집
                    start_date = (last_db_date + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    start_date = "2016-01-01"
        
        # fallback
        if start_date is None:
            start_date = "2016-01-01"

        if start_date >= end_date:
            logger.info(f"VIX data is already up to date. (Last: {start_date})")
            return existing_df

        logger.info(f"🚀 Collecting VIX data: {start_date} ~ {end_date}...")
        
        try:
            # yfinance를 통해 데이터 다운로드
            df_raw = yf.download(self.ticker, start=start_date, end=end_date)
            
            if df_raw.empty:
                logger.warning(f"No new VIX data found since {start_date}.")
                return existing_df
            
            df_raw = df_raw.reset_index()
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            
            df_vix = df_raw[['Date', 'Close']].copy()
            df_vix.columns = ['vix_raw_date', 'vix_close']
            
            # Polars 변환 및 날짜 처리
            new_vix_pl = pl.from_pandas(df_vix).with_columns([
                pl.col("vix_raw_date").dt.cast_time_unit("us").alias("vix_raw_date"),
                (pl.col("vix_raw_date") + pl.duration(days=1)).dt.date().alias("match_date")
            ])
            
            # 기존 데이터와 병합 (중복 제거 및 정렬)
            if not existing_df.is_empty():
                # match_date 타입 유동성 대응
                existing_df = existing_df.with_columns(pl.col("match_date").dt.date())
                combined_df = pl.concat([existing_df, new_vix_pl]).unique(subset=["vix_raw_date"]).sort("vix_raw_date")
            else:
                combined_df = new_vix_pl
            
            combined_df.write_parquet(self.file_path)
            logger.info(f"✅ VIX data saved to {self.file_path} (Total: {len(combined_df)} rows)")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to collect VIX data: {e}")
            return existing_df

    def get_vix_for_date(self, target_date: str) -> Optional[float]:
        """
        특정 날짜(한국 기준)에 사용할 수 있는 전일 VIX 종가를 반환
        
        Args:
            target_date: 한국 기준 날짜 (YYYY-MM-DD 또는 YYYYMMDD)
        """
        if not self.file_path.exists():
            self.collect_vix()
            
        try:
            if len(target_date) == 8:
                dt = datetime.strptime(target_date, "%Y%m%d").date()
            else:
                dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            
            df = pl.read_parquet(self.file_path)
            
            # match_date 타입 Date 강제
            df = df.with_columns(pl.col("match_date").dt.date())
            
            # match_date가 target_date보다 작거나 같은 것 중 가장 최근 것
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
    parser.add_argument("--start", type=str, default=None, help="Start date for collection (YYYY-MM-DD). Default is auto-incremental.")
    parser.add_argument("--test_date", type=str, default="2024-03-04", help="Sample date to test retrieval (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    collector = VixCollector()
    
    # 1. 데이터 수집 (start 인자가 None이면 DB 마지막 날짜 이어서 수집)
    collector.collect_vix(start_date=args.start)
    
    # 2. 특정 날짜 조회 테스트
    print(f"\n--- Testing Retrieval for {args.test_date} ---")
    val = collector.get_vix_for_date(args.test_date)
    if val:
        print(f"Retrieved VIX Value: {val:.2f}")
    else:
        print("Failed to retrieve VIX value.")

