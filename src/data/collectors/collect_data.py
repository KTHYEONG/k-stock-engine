import argparse
import logging
import sys
import asyncio
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트에 위치)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.collectors.market_data import MarketDataCollector
from src.data.feature_store import FeatureStore
from src.utils.logger import setup_logger

async def main():
    parser = argparse.ArgumentParser(description="Collect Daily Market Data")
    parser.add_argument("--date", type=str, help="Target date (YYYYMMDD).", default=None)
    parser.add_argument("--start", type=str, help="Start date (YYYYMMDD) for batch collection.", default="20160104")
    parser.add_argument("--end", type=str, help="End date (YYYYMMDD) for batch collection.", default="20251231")
    
    args = parser.parse_args()
    
    # 상위 패키지 'data'에 대한 로거를 설정하여 하위 모듈의 로그가 출력되도록 함
    logger = setup_logger("data")
    
    # ❌ 개별 콜렉터들의 자질구레한 로그(INFO)는 숨기고 WARNING 이상만 출력하여 가독성 개선
    import logging
    logging.getLogger("data.collectors").setLevel(logging.WARNING)
    
    collector = MarketDataCollector()
    store = FeatureStore()
    
    dates = []
    
    if args.date:
        # Single date mode
        dates.append(args.date)
    elif args.start and args.end:
        # Range mode
        s = datetime.strptime(args.start, "%Y%m%d")
        e = datetime.strptime(args.end, "%Y%m%d")
        delta = e - s
        for i in range(delta.days + 1):
            dates.append((s + timedelta(days=i)).strftime("%Y%m%d"))
    else:
        # Fallback to today if somehow logic fails
        dates.append(datetime.now().strftime("%Y%m%d"))
        
    # 이미 저장된 날짜 제외 (중복 수집 방지)
    existing_dates = store.get_existing_dates()
    if existing_dates:
        initial_count = len(dates)
        dates = [d for d in dates if d not in existing_dates]
        skip_count = initial_count - len(dates)
        if skip_count > 0:
            logger.info(f"Skipping {skip_count} dates that already exist in the feature store.")

    if not dates:
        logger.info("No new dates to collect. All dates in the range already exist.")
        # 지수 데이터는 날짜 범위와 상관없이 한 번 최신화해주는 것이 좋음
        indices_df = await collector.sync_all_indices(count=3000)
        if not indices_df.is_empty():
            store.save_features(indices_df)
            logger.info("[OK] Indices synced successfully.")
        return
        
    logger.info("=" * 50)
    logger.info(f"[START] Data Collection Started: {dates[0]} ~ {dates[-1]}")
    logger.info(f"[INFO] Total Target Days: {len(dates)}")
    logger.info("=" * 50)

    # 시작 전 지수 데이터 일괄 싱크 (MA120 등 과거 데이터를 위해 충분히 수집)
    indices_df = await collector.sync_all_indices(count=3000)
    if not indices_df.is_empty():
        store.save_features(indices_df)
        logger.info("[OK] Pre-sync: Indices updated.")
        
    sem = asyncio.Semaphore(5)  # 동시 실행할 날짜 수 (API 한도 및 부하 고려)
    
    async def process_date(d):
        async with sem:
            start_time = time.time()
            try:
                # 1. 개별 종목 데이터 수집
                df = await collector.collect_daily_data(d)
                
                # 2. 시장 지수 데이터 수집 (Relative Trend용)
                idx_df = await collector.collect_market_indices(d)
                
                # 두 데이터 합치기
                combined_df = pl.DataFrame()
                if not df.is_empty() and not idx_df.is_empty():
                    combined_df = pl.concat([df, idx_df], how="diagonal")
                elif not df.is_empty():
                    combined_df = df
                elif not idx_df.is_empty():
                    combined_df = idx_df

                if not combined_df.is_empty():
                    # 파생 지표 계산
                    combined_df = collector.calculate_derived_metrics(combined_df)
                    
                    # 저장 (년도별/일별 파티셔닝)
                    store.save_features(combined_df, partition_cols=["year", "date"])
                    
                    elapsed = time.time() - start_time
                    tqdm.write(f"[OK] [{d}] Collected {len(combined_df)} records ({elapsed:.2f}s)")
                else:
                    tqdm.write(f"[WARN] [{d}] No data collected")

            except Exception as e:
                tqdm.write(f"[FAIL] [{d}] Failed: {e}")

    # 병렬 실행 (tqdm으로 진행도 표시)
    tasks = [process_date(d) for d in dates]
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Overall Progress", unit="day"):
        await f
            
    logger.info("=" * 50)
    logger.info("[END] All tasks completed!")
    logger.info("=" * 50)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
