import polars as pl
from pathlib import Path
import logging
from typing import List, Optional
import sys
from datetime import datetime

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.legacy.stock_yetirank_v1.data.feature_store import FeatureStore
from src.legacy.stock_yetirank_v1.utils.logger import setup_logger
from src.legacy.stock_yetirank_v1.data.preprocessors.tech_processor import TechProcessor
from src.legacy.stock_yetirank_v1.data.preprocessors.flow_processor import FlowProcessor
from src.legacy.stock_yetirank_v1.data.preprocessors.fund_processor import FundProcessor
from src.legacy.stock_yetirank_v1.data.preprocessors.universe_filter import UniverseFilter
from src.legacy.stock_yetirank_v1.data.preprocessors.target_processor import TargetProcessor
from src.legacy.stock_yetirank_v1.data.preprocessors.cross_processor import CrossSectionalProcessor
from src.legacy.stock_yetirank_v1.data.preprocessors.macro_processor import MacroProcessor

logger = setup_logger("feature_engineer")

class FeatureEngineer:
    """
    전체 피처 엔지니어링 파이프라인을 관장하는 클래스 (Orchestrator)
    Raw Data -> Processors -> Filter -> Feature Store
    """
    
    def __init__(self):
        self.store = FeatureStore()
        
        # 등록된 프로세서들 (순서 중요)
        self.processors = [
            MacroProcessor(), # MacroData(VIX) 먼저 수집 및 Join
            TechProcessor(),
            FundProcessor(),  # FundProcessor 먼저 실행 (market_cap 확보)
            FlowProcessor(),
            UniverseFilter(),
            CrossSectionalProcessor(),  # 횡단면 연산은 베이스 피처 생성 후 마지막에 수행
            TargetProcessor(horizon=5),
        ]
        
        # 필터 (STEP 4)
        # self.universe_filter = UniverseFilter()
        
    def _show_test_summary(self, df: pl.DataFrame):
        """테스트 모드: 데이터의 샘플과 요약을 터미널에 출력"""
        print("\n" + "="*50)
        print(f"🚀 [TEST MODE] Feature Preview (Shape: {df.shape})")
        print("="*50)
        
        # 주요 컬럼 및 최근 추가된 피처 위주로 샘플 출력
        cols_to_show = ["ticker", "date", "close", "vix_zscore_20d"] + [c for c in df.columns if any(x in c for x in ["log_return", "disparity", "np_", "target"])]
        # 존재하는 컬럼만 필터링
        cols_to_show = [c for c in cols_to_show if c in df.columns][:15] 
        
        print(df.select(cols_to_show).tail(10))
        print("\n[Schema Preview]")
        for col, dtype in df.schema.items():
            if col in cols_to_show:
                print(f" - {col:<20}: {dtype}")
        print("..." if len(df.columns) > 15 else "")
        print("="*50 + "\n")

    def run_pipeline(self, start_date: str = "20160101", end_date: str = "20251231", is_test: bool = False):
        logger.info(f"Starting Feature Engineering Pipeline ({start_date} ~ {end_date}, Test={is_test})...")
        
        try:
            # 연도별로 처리하되, 해당 연도 내에서 사용자가 요청한 범위만 한정해서 로드
            years = sorted(list(set([int(start_date[:4]), int(end_date[:4])])))
            years = range(years[0], years[-1] + 1)
            
            for year in years:
                year_str = str(year)
                logger.info(f"Processing Year: {year_str}")
                
                # 1. Load Data with Warm-up
                # 해당 연도의 시작일과 종료일을 계산하되, 전체 요청 범위를 벗어나지 않게 함
                year_start = max(f"{year}0101", start_date)
                year_end = min(f"{year}1231", end_date)
                
                # 지표 계산을 위해 작년 데이터부터 로드 (웜업)
                warmup_start = (datetime.strptime(year_start, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
                
                logger.info(f"Loading data for processing: {warmup_start} ~ {year_end}")
                ldf = self.store.load_features(start_date=warmup_start, end_date=year_end)
                
                # 2. Apply Processors
                for processor in self.processors:
                    logger.debug(f"Queueing {processor.__class__.__name__}...")
                    ldf = processor.process(ldf)
                
                # 3. Filter back to target range (Exactly follow user input for this year)
                dt_start = datetime.strptime(year_start, "%Y%m%d").date()
                dt_end = datetime.strptime(year_end, "%Y%m%d").date()
                ldf = ldf.filter((pl.col("date") >= dt_start) & (pl.col("date") <= dt_end))
                    
                # 4. Collect
                logger.info(f"Executing pipeline for range {year_start} ~ {year_end}...")
                df = ldf.collect()
                
                if df.is_empty():
                    logger.warning(f"No data resulting for range {year_start} ~ {year_end}. Skipping.")
                    continue

                if is_test:
                    self._show_test_summary(df)
                    break 
                else:
                    self.store.save_features(df, partition_cols=["year", "date"], prefix="feat")
                    logger.info(f"Range {year_start} ~ {year_end} completed. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

if __name__ == "__main__":
    import argparse
    from datetime import timedelta
    # --date 인자가 없으면 기본 종료일은 어제 날짜(YYYYMMDD)로 설정
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    
    parser = argparse.ArgumentParser(description="Feature Engineering Orchestrator")
    parser.add_argument("--date", type=str, help="특정 날짜 하루만 수행 (YYYYMMDD)")
    parser.add_argument("--start", type=str, default="20160101", help="시작 날짜 (YYYYMMDD)")
    parser.add_argument("--end", type=str, default=yesterday, help="종료 날짜 (YYYYMMDD, 기본값: 어제)")
    parser.add_argument("--test", action="store_true", help="저장하지 않고 결과 미리보기")
    
    args = parser.parse_args()

    # --date 인자가 있으면 start, end를 해당 날짜로 고정
    start = args.date if args.date else args.start
    end = args.date if args.date else args.end

    engineer = FeatureEngineer()
    engineer.run_pipeline(start_date=start, end_date=end, is_test=args.test)
