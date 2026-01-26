import polars as pl
from pathlib import Path
import sys
import numpy as np
import logging
from typing import Tuple, List, Optional
from catboost import Pool

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.feature_store import FeatureStore
from src.utils.logger import setup_logger

logger = setup_logger("training.data_loader")

class YetiRankDataLoader:
    """
    YetiRank 학습을 위한 데이터 로더
    1. FeatureStore로부터 데이터 로드 (feat_*.parquet)
    2. 학습에 필요한 포맷(CatBoost Pool)으로 변환
    3. Expanding Window Walk-Forward Split 제공
    """
    
    def __init__(self, start_date: str = "20160401"):
        self.store = FeatureStore()
        self.start_date = start_date
        
        # 모델 학습에 사용할 피처 정의 (타겟 및 메타데이터 제외)
        self.exclude_cols = [
            # ID & Time
            "ticker", "date", "year", 
            # Raw Price (스케일 이슈)
            "open", "high", "low", "close", "trading_volume", "trading_value", "market_cap",
            # Intermediate / Raw Financials
            "net_purchase_total", "operating_income", "net_income", "total_assets", "total_equity", "revenue",
            # Targets (Leakage 방지)
            "target_return_5d", "target_rank"
        ]
        
    def load_full_data(self, end_date: str = "20251231", sample_ratio: float = 1.0) -> pl.DataFrame:
        """전체 데이터를 로드하고 전처리 (메모리 로드)"""
        logger.info(f"Loading feature data from {self.start_date} to {end_date} (Sample: {sample_ratio:.1f})...")
        
        # 1. Load Data
        ldf = self.store.load_features(start_date=self.start_date, end_date=end_date, file_pattern="feat_*.parquet")
        
        # Downsampling for Testing
        if sample_ratio < 1.0:
            ldf = ldf.filter(pl.int_range(0, pl.len()) % int(1/sample_ratio) == 0)
            
        df = ldf.collect()
        
        if df.is_empty():
            raise ValueError(f"No data found for range {self.start_date} ~ {end_date}")
            
        # 2. Drop Null Targets (타겟이 없는 데이터는 학습 불가)
        initial_rows = len(df)
        df = df.drop_nulls(subset=["target_rank"])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with null targets.")
            
        # 3. Create Group ID (Query ID for Ranking)
        # 날짜(date)를 정수형 ID로 변환하여 Query ID로 사용
        # 예: 2016-01-04 -> 0, 2016-01-05 -> 1 ...
        dates = df["date"].unique().sort()
        date_map = {d: i for i, d in enumerate(dates)}
        
        # map_dict는 python dict이므로 replace 등을 사용하여 매핑하거나 join 사용
        # Polars 효율성을 위해 join 사용
        date_df = pl.DataFrame({"date": dates, "group_id": range(len(dates))}, schema={"date": pl.Date, "group_id": pl.UInt32})
        df = df.join(date_df, on="date", how="left")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}, Groups: {df['group_id'].n_unique()}")
        return df

    def get_feature_names(self, df: pl.DataFrame) -> List[str]:
        """학습에 사용할 피처 이름 목록 추출"""
        return [c for c in df.columns if c not in self.exclude_cols and c != "group_id"]

    def create_pool(self, df: pl.DataFrame, feature_names: List[str]) -> Pool:
        """Polars DataFrame을 CatBoost Pool로 변환"""
        # CatBoost는 Pandas/Numpy/Polars 지원하지만, 명시적 컬럼 지정 권장
        
        # 피처 데이터 추출 (Numpy 변환 없이 Polars 그대로 전달 가능)
        X = df.select(feature_names).to_pandas() 
        y = df["target_rank"].to_pandas()
        groups = df["group_id"].to_pandas()
        
        # Sector가 있다면 cat_features로 지정 필요 (현재는 일단 수치형 위주로 가정)
        # 만약 sector 컬럼이 피처에 포함된다면 index를 찾아야 함
        cat_features = []
        if "sector" in feature_names:
            cat_features = ["sector"]
            # Ensure sector is string/category for CatBoost
            X["sector"] = X["sector"].astype(str)

        return Pool(
            data=X,
            label=y,
            group_id=groups,
            cat_features=cat_features,
            feature_names=feature_names
        )

    def walk_forward_split(self, df: pl.DataFrame, test_year: int) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Expanding Window 방식의 Walk-Forward Split
        
        Args:
            df: 전체 데이터셋
            test_year: 테스트(예측) 대상 연도 (예: 2024)
            
        Returns:
            (train_df, valid_df, test_df)
            
        Logic:
            Test:  Target Year (e.g., 2024)
            Valid: Test Year - 1 (e.g., 2023)
            Train: Start ~ Valid Year - 1 (e.g., 2016 ~ 2022)
        """
        
        # 형변환 안전장치
        if "year" not in df.columns:
             df = df.with_columns(pl.col("date").dt.year().cast(pl.Utf8).alias("year"))
             
        test_y = str(test_year)
        valid_y = str(test_year - 1)
        
        # Filter Sets
        test_df = df.filter(pl.col("year") == test_y)
        valid_df = df.filter(pl.col("year") == valid_y)
        train_df = df.filter(pl.col("year") < valid_y)
        
        # Validation
        if train_df.is_empty():
            raise ValueError(f"Train set is empty for Test Year {test_year}. Check start_date.")
        if valid_df.is_empty():
            raise ValueError(f"Valid set is empty (Year {valid_y}). Cannot tune hyperparameters.")
        if test_df.is_empty():
            logger.warning(f"Test set is empty (Year {test_year}). Only Train/Valid will be returned.")

        logger.info(f"Split for Test Year {test_year}:")
        logger.info(f" - Train: ~ {int(valid_y)-1} ({len(train_df)} rows)")
        logger.info(f" - Valid: {valid_y} ({len(valid_df)} rows)")
        logger.info(f" - Test : {test_y} ({len(test_df)} rows)")
        
        return train_df, valid_df, test_df

if __name__ == "__main__":
    # Test Code
    loader = YetiRankDataLoader(start_date="20160401")
    try:
        df = loader.load_full_data()
        feature_names = loader.get_feature_names(df)
        print(f"Features ({len(feature_names)}): {feature_names[:5]} ...")
        
        # Try splitting for 2024
        train, valid, test = loader.walk_forward_split(df, test_year=2024)
        
        # Try creating a small pool
        pool = loader.create_pool(valid.head(1000), feature_names)
        print("Pool created successfully:", pool.shape)
        
    except Exception as e:
        logger.error(f"Loader test failed: {e}")
