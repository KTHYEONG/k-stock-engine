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
            FlowProcessor(),
            FundProcessor(),
            UniverseFilter(),
            TargetProcessor(horizon=5),
        ]
        
        # 필터 (STEP 4)
        # self.universe_filter = UniverseFilter()
        
    def run_pipeline(self, start_date: str = "20160101", end_date: str = "20251231"):
        logger.info(f"Starting Feature Engineering Pipeline ({start_date} ~ {end_date})...")
        
        try:
            years = range(int(start_date[:4]), int(end_date[:4]) + 1)
            
            for year in years:
                year_str = str(year)
                logger.info(f"Processing Year: {year_str}")
                
                # 1. Load Data (LazyFrame)
                ldf = self.store.load_features(start_date=f"{year}0101", end_date=f"{year}1231")
                
                # 2. Apply Processors (Lazily)
                for processor in self.processors:
                    logger.debug(f"Queueing {processor.__class__.__name__}...")
                    ldf = processor.process(ldf)
                    
                # 3. Collect & Save (Actual execution point)
                logger.info(f"Executing pipeline for year {year}...")
                
                # save_features 내부에 collect()가 포함되어 있으나, 
                # 여기서 명시적으로 collect하여 메모리 모니터링 가능하도록 함
                df = ldf.collect()
                
                if df.is_empty():
                    logger.warning(f"No data resulting for year {year}. Skipping.")
                    continue

                self.store.save_features(df)
                logger.info(f"Year {year} processing completed. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

if __name__ == "__main__":
    engineer = FeatureEngineer()
    # 테스트를 위해 최근 데이터만 처리
    engineer.run_pipeline(start_date="20160101", end_date="20251231")
