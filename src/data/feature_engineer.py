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
        
        # 1. Load Raw Data (Price/Volume)
        # FeatureStore에 수집된 데이터는 이미 OHLCV + Some Metrics가 포함됨
        # 가장 기초가 되는 market data를 로드해야 함. 
        # 현재 구조상 collect_data.py가 'features' 폴더에 저장하고 있음.
        # 이를 로드해서 추가 가공하는 방식.
        
        try:
            # Lazy Loading for memory efficiency
            # FeatureStore.load_features는 collect()를 해버림. 대용량 처리를 위해 lazy 사용 검토 필요.
            # 일단 메모리가 허용된다고 가정하고 진행 (연도별 파티셔닝 되어 있으므로).
            # 전체 기간 로드는 메모리 이슈 가능성 -> 연도별로 처리하여 저장하는 방식 권장.
            
            years = range(int(start_date[:4]), int(end_date[:4]) + 1)
            
            for year in years:
                year_str = str(year)
                logger.info(f"Processing Year: {year_str}")
                
                # 해당 연도 데이터 로드
                # start_date, end_date 필터링을 위해 FeatureStore 수정 필요할 수도 있으나,
                # 여기서는 raw parquet 파일을 직접 scan 하는 방식이 더 유연할 수 있음.
                # 하지만 일관성을 위해 store 메서드 활용.
                
                # FeatureStore가 '누적된 피처'를 저장하는 곳인지, 'Raw Market Data'를 저장하는 곳인지 명확히 해야 함.
                # collect_data.py: store.save_features(...) -> PROCESSED_DATA_DIR / "features"
                # 즉, 이미 1차 가공된 데이터가 features에 있음.
                # Feature Engineer는 이를 읽어서 '더 많은 피처'를 붙여서 '다시 저장'하거나 'model_ready' 폴더에 저장해야 함.
                # 덮어쓰기(Overwrite) 전략을 사용하면 컬럼이 계속 늘어나는 구조.
                
                df = self.store.load_features(start_date=f"{year}0101", end_date=f"{year}1231")
                
                if df.is_empty():
                    logger.warning(f"No data found for year {year}. Skipping.")
                    continue
                
                # 2. Apply Processors
                for processor in self.processors:
                    logger.info(f"Applying {processor.__class__.__name__}...")
                    df = processor.process(df)
                    
                # 3. Apply Filters (STEP 4)
                # if self.universe_filter:
                #     df = self.universe_filter.apply_all(df)
                
                # 4. Save Updated Features
                # 같은 경로에 덮어쓰기 (새로운 컬럼 추가됨)
                # 파티셔닝 유지를 위해 다시 저장
                self.store.save_features(df)
                
                logger.info(f"Year {year} processing completed. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise e

if __name__ == "__main__":
    engineer = FeatureEngineer()
    # 테스트를 위해 최근 데이터만 처리
    engineer.run_pipeline(start_date="20240101", end_date="20240131")
