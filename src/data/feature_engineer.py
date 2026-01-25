import polars as pl
from pathlib import Path
import logging
from typing import List, Optional
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.feature_store import FeatureStore
from src.utils.logger import setup_logger
from src.data.preprocessors.tech_processor import TechProcessor
from src.data.preprocessors.flow_processor import FlowProcessor
from src.data.preprocessors.fund_processor import FundProcessor
from src.data.preprocessors.universe_filter import UniverseFilter
from src.data.preprocessors.target_processor import TargetProcessor

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
            TechProcessor(),
            FundProcessor(),  # FundProcessor 먼저 실행 (market_cap 확보)
            FlowProcessor(),
            UniverseFilter(),
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
        cols_to_show = ["ticker", "date", "close"] + [c for c in df.columns if any(x in c for x in ["log_return", "disparity", "np_", "target"])]
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
            years = range(int(start_date[:4]), int(end_date[:4]) + 1)
            
            for year in years:
                year_str = str(year)
                logger.info(f"Processing Year: {year_str}")
                
                # 1. Load Data with Warm-up
                warmup_start = f"{year-1}0601"
                ldf = self.store.load_features(start_date=warmup_start, end_date=f"{year}1231")
                
                # 2. Apply Processors
                for processor in self.processors:
                    logger.debug(f"Queueing {processor.__class__.__name__}...")
                    ldf = processor.process(ldf)
                
                # 3. Filter back to target range (Exactly follow user input)
                dt_start = datetime.strptime(start_date, "%Y%m%d").date()
                dt_end = datetime.strptime(end_date, "%Y%m%d").date()
                ldf = ldf.filter((pl.col("date") >= dt_start) & (pl.col("date") <= dt_end))
                    
                # 4. Collect
                logger.info(f"Executing pipeline for year {year}...")
                df = ldf.collect()
                
                if df.is_empty():
                    logger.warning(f"No data resulting for year {year}. Skipping.")
                    continue

                if is_test:
                    self._show_test_summary(df)
                    # 테스트 모드일 경우 첫 번째 연도만 보여주고 중단 가능 (선택 사항)
                    break 
                else:
                    self.store.save_features(df, partition_cols=["year", "date"], prefix="feat")
                    logger.info(f"Year {year} processing completed. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Engineering Orchestrator")
    parser.add_argument("--start", type=str, default="20160101", help="시작 날짜 (YYYYMMDD)")
    parser.add_argument("--end", type=str, default="20251231", help="종료 날짜 (YYYYMMDD)")
    parser.add_argument("--test", action="store_true", help="저장하지 않고 결과 미리보기")
    
    args = parser.parse_args()

    engineer = FeatureEngineer()
    engineer.run_pipeline(start_date=args.start, end_date=args.end, is_test=args.test)
