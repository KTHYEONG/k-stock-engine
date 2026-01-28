
import polars as pl
import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가 (필요 시)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# KRX Collector 임포트
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

try:
    from src.data.collectors.krx_openapi_collector import KRXOpenAPICollector
except ImportError:
    # 패키지 구조에 따라 경로 조정
    from .collectors.krx_openapi_collector import KRXOpenAPICollector

logger = logging.getLogger("data.etf_manager")

class ETFManager:
    """
    ETF 및 지수 데이터 수집/관리 매니저
    KOSPI 200, KOSDAQ 150 관련 ETF 및 지수 데이터를 수집합니다.
    """
    
    # Target Universe Definitions
    TARGET_ETFS = {
        "KOSPI": {
            "069500": "KODEX 200",               # 1X
            "252670": "KODEX 200선물레버리지",      # 2X
            "252710": "KODEX 200선물인버스2X"     # -2X
        },
        "KOSDAQ": {
            "229200": "KODEX 코스닥150",          # 1X
            "233740": "KODEX 코스닥150레버리지",    # 2X
            "250780": "TIGER 코스닥150선물인버스"   # -1X
        }
    }
    
    # 지수 티커 (KRX 표준 코드 확인 필요, 여기서는 이름으로 매핑 로직 처리 가능)
    # KOSPI 200, KOSDAQ 150 지수 코드가 API에서 어떻게 나오는지 확인 후 필터링
    # 통상 KOSPI 200은 "102800" 등의 코드를 가질 수 있음.
    
    def __init__(self):
        self.collector = KRXOpenAPICollector()
        
    async def collect_daily_data(self, date_str: str) -> Dict[str, pl.DataFrame]:
        """
        특정 일자의 Target ETF 및 지수 데이터를 수집합니다.
        
        Args:
            date_str: "YYYYMMDD"
            
        Returns:
            Dict containing:
            - 'etf': Target ETF Dataframe
            - 'index': Market Index Dataframe
        """
        logger.info(f"Starting ETF & Index collection for {date_str}...")
        
        # 1. ETF 데이터 수집 (전체 수집 후 필터링)
        full_etf_df = await self.collector.collect_etf_daily_trade(date_str)
        
        target_etf_df = pl.DataFrame()
        if not full_etf_df.is_empty():
            # 타겟 티커 리스트 추출
            target_tickers = []
            for mkt in self.TARGET_ETFS.values():
                target_tickers.extend(mkt.keys())
            
            # 필터링
            target_etf_df = full_etf_df.filter(pl.col("ticker").is_in(target_tickers))
            logger.info(f"Filtered {len(target_etf_df)} target ETFs from {len(full_etf_df)} total ETFs.")
            
        # 2. 지수 데이터 수집 (KOSPI/KOSDAQ)
        # collect_market_indices는 KOSPI, KOSDAQ 전체 지수 목록을 가져옴.
        index_df = await self.collector.collect_market_indices(date_str)
        
        # [TODO] KOSPI 200, KOSDAQ 150 지수만 남기고 싶다면 여기서 필터링 로직 추가
        # 현재는 전체 지수 데이터를 반환하여 저장하도록 함.
        
        return {
            "etf": target_etf_df,
            "index": index_df
        }

    async def fetch_history(self, start_date: str, end_date: str) -> Dict[str, pl.DataFrame]:
        """
        기간별 데이터 수집 (유틸리티)
        """
        # 날짜 생성
        dates = pd.date_range(start_date, end_date, freq='B') # Business days
        date_strs = [d.strftime("%Y%m%d") for d in dates]
        
        all_etfs = []
        all_indices = []
        
        total = len(date_strs)
        for i, date_str in enumerate(date_strs):
            logger.info(f"[{i+1}/{total}] Fetching {date_str}...")
            data = await self.collect_daily_data(date_str)
            
            if not data["etf"].is_empty():
                all_etfs.append(data["etf"])
            
            if not data["index"].is_empty():
                all_indices.append(data["index"])
                
            # API Rate Limit 준수를 위한 짧은 대기 (Collector 내부적으로 처리하지만 안전장치)
            await asyncio.sleep(0.1)
            
        return {
            "etf": pl.concat(all_etfs) if all_etfs else pl.DataFrame(),
            "index": pl.concat(all_indices) if all_indices else pl.DataFrame()
        }

if __name__ == "__main__":
    import argparse

    async def test_run():
        parser = argparse.ArgumentParser(description="ETF Manager Test Run")
        parser.add_argument("--date", type=str, help="Target date (YYYYMMDD)", default=datetime.now().strftime("%Y%m%d"))
        parser.add_argument("--start", type=str, help="Start date for history fetch (YYYYMMDD)")
        parser.add_argument("--end", type=str, help="End date for history fetch (YYYYMMDD)")
        
        args = parser.parse_args()
        
        logging.basicConfig(level=logging.INFO)
        manager = ETFManager()
        
        if args.start and args.end:
            logger.info(f"Fetching history: {args.start} ~ {args.end}")
            result = await manager.fetch_history(args.start, args.end)
        else:
            logger.info(f"Fetching single date: {args.date}")
            result = await manager.collect_daily_data(args.date)
        
        print("\n=== ETF Data ===")
        print(result["etf"])
        
        print("\n=== Index Data ===")
        print(result["index"])

    asyncio.run(test_run())
