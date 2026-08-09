
import polars as pl
import pandas as pd
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

from src.legacy.stock_yetirank_v1.data.collectors.krx_openapi_collector import KRXOpenAPICollector

logger = logging.getLogger("data.etf_manager")

class ETFManager:
    """
    ETF 및 지수 데이터 수집/관리 매니저
    KOSPI 200, KOSDAQ 150 관련 ETF 및 지수 데이터를 수집합니다.
    """
    
    # Target Universe Definitions (User defined mapping - Step 219)
    TARGET_ETFS = {
        "KOSPI": {
            "069500": "KODEX 200",
            "122630": "KODEX 레버리지",
            "114800": "KODEX 인버스",
            "252670": "KODEX 200선물인버스2X"
        },
        "KOSDAQ": {
            "229200": "KODEX 코스닥 150",
            "233740": "KODEX 코스닥 150 레버리지",
            "251340": "KODEX 코스닥 150선물인버스",
            "252710": "TIGER 코스닥 150선물인버스2X"
        }
    }
    
    # 지수 티커 (KRX 표준 코드 확인 필요, 여기서는 이름으로 매핑 로직 처리 가능)
    # KOSPI 200, KOSDAQ 150 지수 코드가 API에서 어떻게 나오는지 확인 후 필터링
    # 통상 KOSPI 200은 "102800" 등의 코드를 가질 수 있음.
    
    def __init__(self):
        self._collector = None
        # 데이터 저장소 초기화 (data/etf, data/index)
        from src.legacy.stock_yetirank_v1.data.feature_store import FeatureStore
        self.etf_store = FeatureStore(base_path=Path("./data/etf_daily"))
        self.index_store = FeatureStore(base_path=Path("./data/market_index"))

    @property
    def collector(self):
        if self._collector is None:
            from src.legacy.stock_yetirank_v1.data.collectors.krx_openapi_collector import KRXOpenAPICollector
            self._collector = KRXOpenAPICollector()
        return self._collector
        
    async def collect_daily_data(self, date_str: str) -> Dict[str, pl.DataFrame]:
        """
        특정 일자의 전체 ETF 및 지수 데이터를 수집합니다.
        """
        # 1. ETF 데이터 수집 (전체 수집)
        full_etf_df = await self.collector.collect_etf_daily_trade(date_str)
        
        # 2. 지수 데이터 수집 (KOSPI/KOSDAQ)
        index_df = await self.collector.collect_market_indices(date_str)
        
        return {
            "etf": full_etf_df,
            "index": index_df
        }

    async def fetch_history(self, start_date: str, end_date: str):
        """
        기간별 데이터 수집 및 저장 (Parquet)
        """
        from tqdm import tqdm
        
        # 날짜 생성
        dates = pd.date_range(start_date, end_date, freq='B') # Business days
        date_strs = [d.strftime("%Y%m%d") for d in dates]
        
        # 이미 수집된 날짜 스캔 (중복 방지)
        collected_dates = self.etf_store.get_existing_dates()
        target_dates = [d for d in date_strs if d not in collected_dates]
        
        logger.info(f"Total: {len(date_strs)} days | Collected: {len(collected_dates)} days | Target: {len(target_dates)} days")
        
        if not target_dates:
            logger.info("All dates in range are already collected.")
            return

    async def fetch_history(self, start_date: str, end_date: str):
        """
        기간별 데이터 수집 및 저장 (Parquet) - 비동기 병렬 처리 적용 (속도 향상)
        """
        from tqdm import tqdm
        
        # 날짜 생성
        dates = pd.date_range(start_date, end_date, freq='B') # Business days
        date_strs = [d.strftime("%Y%m%d") for d in dates]
        
        # 이미 수집된 날짜 스캔 (중복 방지)
        collected_dates = self.etf_store.get_existing_dates()
        target_dates = [d for d in date_strs if d not in collected_dates]
        
        logger.info(f"Target: {len(target_dates)} days (Total: {len(date_strs)} / Collected: {len(collected_dates)})")
        
        if not target_dates:
            logger.info("All dates already collected.")
            return

        # 동시 실행 제한 (API 안정성 고려: 3)
        sem = asyncio.Semaphore(3)
        pbar = tqdm(total=len(target_dates), desc="Fetching ETF History")
        
        async def process_date(date_str):
            async with sem:
                try:
                    data = await self.collect_daily_data(date_str)
                    
                    # [개장 여부 확인] 
                    if data["etf"].is_empty() or data["etf"]["close"].sum() == 0:
                        pbar.update(1)
                        return
                    
                    # ETF 저장
                    self.etf_store.save_features(data["etf"])
                        
                    # Index 저장
                    if not data["index"].is_empty():
                        self.index_store.save_features(data["index"])
                    
                    pbar.update(1)
                    
                    # 429 에러 방지를 위한 미세 딜레이
                    await asyncio.sleep(0.1) 
                    
                except Exception as e:
                    # 에러 발생 시 로그만 남기고 진행 (전체 중단 방지)
                    pbar.write(f"[Error] {date_str}: {e}")
                    pbar.update(1)

        # 병렬 태스크 실행
        tasks = [process_date(d) for d in target_dates]
        await asyncio.gather(*tasks)
        
        pbar.close()
        logger.info("History fetch completed.")

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
            await manager.fetch_history(args.start, args.end)
        else:
            logger.info(f"Fetching single date: {args.date}")
            result = await manager.collect_daily_data(args.date)
        
            print("\n=== ETF Data ===")
            print(result["etf"])
            
            print("\n=== Index Data ===")
            print(result["index"])

    asyncio.run(test_run())
