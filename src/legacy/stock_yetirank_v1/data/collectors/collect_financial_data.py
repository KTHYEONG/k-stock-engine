"""전 종목 재무 데이터 과거 기간(2016~2025) 증분 수집 및 로컬 DB 구축 스크립트"""

import asyncio
import logging
import pandas as pd
import polars as pl
from tqdm import tqdm
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가 및 환경 변수 로드
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from src.legacy.stock_yetirank_v1.data.collectors.dart_collector import OpenDartCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("collect_financial")

async def collect_all_financials(start_year: int = 2014, end_year: int = 2025):
    """
    지정된 범위의 전 종목 재무 데이터를 증분 수집하여 Parquet으로 저장.
    """
    collector = OpenDartCollector()
    
    # 1. 고유번호 로드 (DART Ticker 매핑용)
    collector.load_corp_codes()
    tickers = list(collector.corp_codes.keys())
    
    # 2. 저장 경로 및 기존 데이터 로드 (중복 수집 방지용)
    data_dir = os.path.join(project_root, "data")
    save_path = os.path.join(data_dir, "financials.parquet")
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(save_path):
        try:
            existing_df = pl.read_parquet(save_path)
            # 수집 완료된 (ticker, year, reprt_code) 키 집합 생성
            existing_keys = set(zip(
                existing_df["ticker"].to_list(),
                existing_df["year"].to_list(),
                existing_df["reprt_code"].to_list()
            ))
            logger.info(f"Loaded {len(existing_keys)} existing records from {save_path}. Skipping these in this run.")
        except Exception as e:
            logger.error(f"Error reading existing parquet: {e}")
            existing_df = pl.DataFrame()
            existing_keys = set()
    else:
        existing_df = pl.DataFrame()
        existing_keys = set()
    
    # 3. 수집 대상 기간 설정
    years = [str(y) for y in range(start_year, end_year + 1)]
    report_codes = ["11013", "11012", "11014", "11011"]
    
    new_data = []
    daily_call_limit = 19800 # 안전을 위해 20,000회보다 조금 적게 설정
    call_count = 0
    
    logger.info(f"Starting incremental collection for {len(tickers)} stocks across years {start_year}~{end_year}...")

    # [Optimization] 동시 요청 제한 (OpenDART 서버 부하 고려)
    sem = asyncio.Semaphore(3)  # 3개의 배치를 동시에 처리 (안전)
    
    async def process_batch(chunk, year, code):
        """단일 배치 비동기 처리를 위한 래퍼"""
        nonlocal call_count
        async with sem:
            # 일일 한도 체크
            if call_count >= daily_call_limit:
                return None
            
            # API 호출
            loop = asyncio.get_event_loop()
            try:
                # 동기 함수를 비동기로 실행
                df = await loop.run_in_executor(
                    None, 
                    lambda: collector.collect_financial_stat_batch(chunk, year, code)
                )
                call_count += 1
                return df
            except Exception as e:
                logger.debug(f"Batch failed: {e}")
                return None

    try:
        # 연도 -> 보고서 -> 종목 순으로 순회
        for year in years:
            for rpt_code in report_codes:
                logger.info(f"Processing Period: Year {year}, Report Code {rpt_code}")
                
                # 해당 기간에 대해 수집 안 된 종목 필터링
                target_tickers = [t for t in tickers if (t, year, rpt_code) not in existing_keys]
                
                if not target_tickers:
                    continue
                
                # 배치 처리
                batch_size = 80
                ticker_chunks = [target_tickers[i:i + batch_size] for i in range(0, len(target_tickers), batch_size)]
                
                # 비동기 태스크 생성
                tasks = [process_batch(chunk, year, rpt_code) for chunk in ticker_chunks]
                
                # 실행 및 결과 수집 (tqdm 연동)
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Async Batch {year}-{rpt_code}"):
                    result_df = await f
                    if result_df is not None and not result_df.empty:
                        new_data.append(result_df)
                    
                    if call_count >= daily_call_limit:
                        raise StopIteration("Daily OpenDART API Limit reached.")

    except StopIteration as si:
        logger.warning(str(si))
    except Exception as e:
        logger.error(f"Unexpected error during collection: {e}")

    # 4. 데이터 통합 및 증분 저장
    if new_data:
        # Pandas 병합 후 Polars 전환
        new_combined_df = pd.concat(new_data, ignore_index=True)
        pl_new = pl.from_pandas(new_combined_df)
        
        # 기존 데이터와 결합
        if not existing_df.is_empty():
            final_df = pl.concat([existing_df, pl_new]).unique(subset=["ticker", "year", "reprt_code"])
        else:
            final_df = pl_new
            
        final_df.write_parquet(save_path)
        logger.info(f"Update Successful! Total records: {len(final_df)} (Added {len(pl_new)}).")
    else:
        logger.info("No new records were added in this run.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Financial Data from OpenDART")
    parser.add_argument("--start", type=int, default=2014, help="Start year (YYYY)")
    parser.add_argument("--end", type=int, default=2025, help="End year (YYYY)")
    
    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(collect_all_financials(start_year=args.start, end_year=args.end))
