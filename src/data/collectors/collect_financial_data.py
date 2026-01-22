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

from src.data.collectors.dart_collector import OpenDartCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("collect_financial")

async def collect_all_financials():
    """
    2015~2025 전 종목 재무 데이터를 증분 수집하여 Parquet으로 저장.
    OpenDART API의 일일 한도(20,000회)를 고려하여 이미 수집된 항목은 건너뛰며 진행합니다.
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
    
    # 3. 수집 대상 기간 설정 (2015 ~ 2025)
    # 2016년 초 백테스트 시 2015년 실적이 필요하므로 2015년부터 수집합니다.
    years = [str(y) for y in range(2014, 2026)]
    report_codes = ["11013", "11012", "11014", "11011"]
    
    new_data = []
    daily_call_limit = 19800 # 안전을 위해 20,000회보다 조금 적게 설정
    call_count = 0
    
    logger.info(f"Starting incremental collection for {len(tickers)} stocks across {len(years)} years...")

    try:
        # 연도 -> 보고서 -> 종목 순으로 순회 (가장 과거부터 채움)
        for year in years:
            for rpt_code in report_codes:
                logger.info(f"Processing Period: Year {year}, Report Code {rpt_code}")
                
                # 해당 기간에 대해 수집 안 된 종목 필터링
                target_tickers = [t for t in tickers if (t, year, rpt_code) not in existing_keys]
                
                if not target_tickers:
                    logger.info(f"All data for {year}-{rpt_code} already exists. Skipping.")
                    continue
                
                # 배치 처리 (50개씩 묶음, 최대 100개 가능하나 안정성 고려 50)
                batch_size = 50
                total_target = len(target_tickers)
                
                # chunking
                ticker_chunks = [target_tickers[i:i + batch_size] for i in range(0, total_target, batch_size)]
                
                for chunk in tqdm(ticker_chunks, desc=f"Batch Crawl {year}-{rpt_code}"):
                    try:
                        # 배치 호출
                        df = collector.collect_financial_stat_batch(chunk, year, rpt_code)
                        
                        if not df.empty:
                            new_data.append(df)
                        
                        call_count += 1
                        
                        # 초당 약 5회 요청 수준으로 지연
                        time.sleep(0.2)
                        
                        # 일일 한도 체크
                        if call_count >= daily_call_limit:
                            raise StopIteration("Daily OpenDART API Limit reached.")
                            
                    except Exception as e:
                        if isinstance(e, StopIteration): raise e
                        logger.debug(f"Failed for batch ({year}-{rpt_code}): {e}")
                        continue
                        
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
            # 컬럼 타입 일치 확인 (필요 시 캐스팅)
            final_df = pl.concat([existing_df, pl_new]).unique(subset=["ticker", "year", "reprt_code"])
        else:
            final_df = pl_new
            
        # Parquet 파일로 덮어쓰기 (내부적으로는 증분 업데이트된 상태)
        final_df.write_parquet(save_path)
        logger.info(f"Update Successful! Total records in DB: {len(final_df)} (Added {len(pl_new)} new records).")
    else:
        logger.info("No new records were added in this run.")

if __name__ == "__main__":
    # Windows 상에서 asyncio 루프 정책 설정 (호환성)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(collect_all_financials())
