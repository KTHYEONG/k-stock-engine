import polars as pl
from pathlib import Path
import sys
import numpy as np
import logging
try:
    import yaml
except ImportError:
    yaml = None 
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
            "ticker", "date", "year", "name", "disclosure_date",
            # Raw Price (스케일 이슈)
            "open", "high", "low", "close", "trading_volume", "trading_value", "market_cap",
            # Intermediate / Raw Financials
            "net_purchase_total", "operating_income", "net_income", "total_assets", "total_equity", "revenue",
            # Targets & Leakage (수익률 지표는 피처에서 완전히 제외)
            "target_return_5d", "target_rank"
        ]
        
        self.feature_config_path = PROJECT_ROOT / "data" / "model_features" / "features_v1.yaml"
        
    def _load_feature_config(self) -> Optional[List[str]]:
        """YAML 설정 파일에서 사용할 피처 리스트 로드 (Whitelist)"""
        if yaml is None:
            return None
            
        if not self.feature_config_path.exists():
            return None
            
        try:
            with open(self.feature_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("features", [])
        except Exception as e:
            logger.error(f"Error loading feature config: {e}")
            return None
        
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
            
        # [CRITICAL UPDATE] 학습 데이터 퀄리티 강화 및 YetiRank 최적화
        
        # 1. Trading Value 생성 (만약 없다면) 및 Top 1000 필터링
        # - 잡주(동전주)의 노이즈 제거
        # - YetiRank GPU 제한(1023) 자동 준수
        if "trading_value" not in df.columns:
            if "close" in df.columns and "trading_volume" in df.columns:
                df = df.with_columns((pl.col("close") * pl.col("trading_volume")).alias("trading_value"))
            else:
                df = df.with_columns(pl.lit(1.0).alias("trading_value")) # 거래대금 없으면 필터링 스킵

        logger.info("Filtering Top 1000 stocks daily by Trading Value (High Quality Data Focus)...")
        df = df.sort(["date", "trading_value"], descending=[False, True]) \
               .with_columns(pl.int_range(0, pl.len()).over("date").alias("_daily_rank")) \
               .filter(pl.col("_daily_rank") < 1000) \
               .drop("_daily_rank")

        # 2. Shift(1) 제거
        # - 사유: T일 지표(RSI, MACD)로 T+5일 수익(Future)을 예측하는 것이므로, 
        #   같은 날짜 행에 두는 것이 논리적으로 맞음. (X_t -> Y_t+5)
        #   Shift를 하면 어제 지표(X_t-1)로 미래를 예측하게 되어 정보 지연 발생.
        
        # 3. Target Rank Decile 변환 (0~9 정수)
        # - YetiRank는 명확한 등급 차이(Integer Relevance)가 있을 때 학습 안정성이 높음.
        # - 실수(0.0~1.0)보다 Decile(0~9)이 구분이 확실함.
        logger.info("Generatinng Decile Ranks (0~9) for YetiRank optimization...")
        
        df = df.with_columns([
            pl.col("date").rank("dense").alias("group_id"),
            
            # Decile Rank Calculation: (Ordinal Rank / Total Count) * 10 -> int
            (pl.col("target_return_5d")
             .rank("ordinal")
             .over("date") / pl.len().over("date") * 9.99).cast(pl.Int32).alias("target_rank")
        ])
        
        # [DEBUG] Target Rank 검증 로그
        debug_corr = df.select(pl.corr("target_rank", "target_return_5d")).item()
        logger.info(f"Target Rank Check - Correlation with Return: {debug_corr:.4f} (Must be positive and high)")
        logger.info(f"Sample Data:\n{df.select(['date', 'target_return_5d', 'target_rank', 'trading_value']).head(5)}")

        logger.info(f"Data loaded successfully. Shape: {df.shape}, Groups: {df['group_id'].n_unique()}")
        return df

    def get_feature_names(self, df: pl.DataFrame) -> List[str]:
        """학습에 사용할 피처 이름 목록 추출"""
        
        # 0. YAML Config 우선 적용 (Whitelist)
        selected_features = self._load_feature_config()
        if selected_features:
            logger.info(f"Using {len(selected_features)} features from config: {self.feature_config_path.name}")
            
            # 유효성 검사: 데이터프레임에 없는 피처는 제외
            valid_features = []
            missing_features = []
            
            for col in selected_features:
                if col in df.columns:
                    valid_features.append(col)
                else:
                    missing_features.append(col)
            
            if missing_features:
                logger.warning(f"⚠️ Missing features in data: {missing_features}")
                
            return valid_features

        # 1. 명시적 제외 목록 필터링 (Blacklist - Fallback)
        base_features = [c for c in df.columns if c not in self.exclude_cols and c != "group_id"]
        
        # 2. 데이터 타입 기반 방어적 필터링 (숫자형이거나 지정된 카테고리 컬럼만 포함)
        final_features = []
        for col in base_features:
            dtype = df.schema[col]
            if dtype.is_numeric() or col == "sector":
                final_features.append(col)
            else:
                logger.debug(f"Excluding non-numeric feature: {col} ({dtype})")
                
        return final_features

    def create_pool(self, df: pl.DataFrame, feature_names: List[str]) -> Pool:
        """Polars DataFrame을 CatBoost Pool로 변환"""
        # [CRITICAL] CatBoost Ranking은 동일 group_id가 연속해서 나타나야 함 (정렬 필수)
        # 또한 일관성을 위해 ticker 순으로 2차 정렬을 수행하여 순서 어긋남 방지
        df = df.sort(["group_id", "ticker"])
        
        # 피처 데이터 추출
        X = df.select(feature_names).to_pandas() 
        y = df["target_rank"].to_pandas()
        groups = df["group_id"].to_pandas()
        
        # 카테고리 피처 자동 감지 (실제 문자열인 것만)
        cat_features = []
        for col in feature_names:
            if df.schema[col] == pl.Utf8 or col == "sector":
                cat_features.append(col)
                # CatBoost 요구사항: 카테고리 피처는 반드시 문자열이어야 함
                X[col] = X[col].astype(str)

        # Check for weights
        weights = None
        if "sample_weight" in df.columns:
            weights = df["sample_weight"].to_pandas()

        return Pool(
            data=X,
            label=y,
            group_id=groups,
            weight=weights,
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
