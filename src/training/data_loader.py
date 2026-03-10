import polars as pl
from pathlib import Path
import sys
import numpy as np
import logging
import re
try:
    import yaml
except ImportError:
    yaml = None 
from typing import Tuple, List, Optional, Any
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

    @staticmethod
    def infer_target_horizon_days(df: pl.DataFrame, default: int = 5) -> int:
        """Infer target horizon from column name pattern: target_return_{N}d."""
        p = re.compile(r"^target_return_(\d+)d$")
        for col in df.columns:
            m = p.match(col)
            if m:
                return int(m.group(1))
        return int(max(1, default))

    def apply_time_decay_weights(
        self,
        df: pl.DataFrame,
        min_weight: float = 0.5,
        max_weight: float = 1.0,
        context: str = "train",
    ) -> pl.DataFrame:
        """
        Apply linear time-decay weights from min_weight to max_weight by date.
        Newer rows get larger weights. Always writes `sample_weight`.
        """
        if df.is_empty() or "date" not in df.columns:
            return df

        min_weight = float(min_weight)
        max_weight = float(max_weight)
        if max_weight < min_weight:
            min_weight, max_weight = max_weight, min_weight

        max_date = df["date"].max()
        min_date = df["date"].min()
        span_days = 0
        if max_date is not None and min_date is not None:
            try:
                span_days = int((max_date - min_date).days)
            except Exception:
                try:
                    span_days = int((max_date - min_date) / np.timedelta64(1, "D"))
                except Exception:
                    span_days = 0

        if span_days <= 0:
            out = df.with_columns(pl.lit(max_weight).cast(pl.Float32).alias("sample_weight"))
            logger.info(
                f"Applied Time-Decay Weights [{context}]: constant={max_weight:.3f} (rows={len(out)})"
            )
            return out

        out = df.with_columns(
            (
                min_weight
                + (max_weight - min_weight)
                * (pl.col("date") - pl.lit(min_date)).dt.total_days()
                / float(span_days)
            ).cast(pl.Float32).alias("sample_weight")
        )
        logger.info(
            f"Applied Time-Decay Weights [{context}]: {min_weight:.3f}~{max_weight:.3f} (rows={len(out)})"
        )
        return out
        
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
        
    def load_full_data(self, end_date: str = "20261231", sample_ratio: float = 1.0) -> pl.DataFrame:
        """전체 데이터를 로드하고 전처리 (메모리 로드)"""
        logger.info(f"Loading feature data from {self.start_date} to {end_date} (Sample: {sample_ratio:.1f})...")
        
        # 1. Load Data
        ldf = self.store.load_features(start_date=self.start_date, end_date=end_date, file_pattern="*_feat.parquet")
        
        # Downsampling for Testing
        if sample_ratio < 1.0:
            ldf = ldf.filter(pl.int_range(0, pl.len()) % int(1/sample_ratio) == 0)
            
        df = ldf.collect()
        
        if df.is_empty():
            raise ValueError(f"No data found for range {self.start_date} ~ {end_date}")
            
        # 2. Drop Null Targets (타겟이 없는 데이터는 학습 불가)
        # Use raw target first because target_rank is recomputed later in this loader.
        initial_rows = len(df)
        target_col = "target_return_5d" if "target_return_5d" in df.columns else "target_rank"
        df = df.drop_nulls(subset=[target_col])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with null targets ({target_col}).")
            
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
        
        # 3. [UPDATED] 아웃라이어 제어(Winsorization) 및 실수형 타겟(Continuous Label)
        # - 상하한가 등 극단적 아웃라이어가 Min-Max 스케일링을 망가뜨리는 것을 방지 (1% ~ 99% Clipping)
        # - 이후 Percentile 방식의 연속형 점수(0~10)로 변환하여 우량주 간의 미세한 순위 변별력 유지
        logger.info("Applying Winsorization and generating Continuous Relevance Scores...")
        
        df = df.with_columns([
            pl.col("date").rank("dense").alias("group_id"),
            
            # 1단계: 일별 1% 하위, 99% 상위 컷오프 계산 및 클리핑(Winsorization)
            pl.col("target_return_5d").clip(
                pl.col("target_return_5d").quantile(0.01).over("date"),
                pl.col("target_return_5d").quantile(0.99).over("date")
            ).alias("clipped_target")
        ]).with_columns([
            # 2단계: 클리핑된 수익률을 기반으로 0~10 사이의 연속형(Float) 점수로 Min-Max 스케일링
            ((pl.col("clipped_target") - pl.col("clipped_target").min().over("date")) / 
             (pl.col("clipped_target").max().over("date") - pl.col("clipped_target").min().over("date") + 1e-8) * 10.0)
            .alias("target_rank")
        ]).drop("clipped_target")
        
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

    def walk_forward_split(
        self,
        df: pl.DataFrame,
        test_year: int,
        embargo_days: int = 6,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
        embargo_days = int(max(0, embargo_days))
        
        # Base sets
        test_df_base = df.filter(pl.col("year") == test_y)
        valid_df_base = df.filter(pl.col("year") == valid_y)

        if valid_df_base.is_empty():
            raise ValueError(f"Valid set is empty (Year {valid_y}). Cannot tune hyperparameters.")

        # Build trading-date index map (embargo in trading days, not calendar days)
        unique_dates = (
            df.select("date")
            .unique()
            .sort("date")
            .get_column("date")
            .to_list()
        )
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        def _exclusive_end_date(next_start_date: Any, gap_days: int) -> Optional[Any]:
            idx = date_to_idx.get(next_start_date)
            if idx is None:
                return None
            end_idx = idx - gap_days - 1
            if end_idx < 0:
                return None
            return unique_dates[end_idx]

        # Apply embargo between Train-Valid boundary
        valid_start_date = valid_df_base["date"].min()
        train_end_date = _exclusive_end_date(valid_start_date, embargo_days)
        if train_end_date is None:
            raise ValueError(
                f"Train set becomes empty with embargo_days={embargo_days}. "
                f"Check start_date or reduce embargo."
            )

        # Apply embargo between Valid-Test boundary
        if test_df_base.is_empty():
            valid_end_date = valid_df_base["date"].max()
        else:
            test_start_date = test_df_base["date"].min()
            valid_end_date = _exclusive_end_date(test_start_date, embargo_days)
            if valid_end_date is None:
                raise ValueError(
                    f"Valid set becomes empty with embargo_days={embargo_days}. "
                    f"Reduce embargo or check data range."
                )

        # Filter Sets with embargo-aware date bounds
        train_df = df.filter(pl.col("date") <= train_end_date)
        valid_df = df.filter(
            (pl.col("year") == valid_y)
            & (pl.col("date") >= valid_start_date)
            & (pl.col("date") <= valid_end_date)
        )
        test_df = test_df_base
        
        # Validation
        if train_df.is_empty():
            raise ValueError(f"Train set is empty for Test Year {test_year}. Check start_date.")
        if valid_df.is_empty():
            raise ValueError(
                f"Valid set is empty for Test Year {test_year} after embargo_days={embargo_days}."
            )
        if test_df.is_empty():
            logger.warning(f"Test set is empty (Year {test_year}). Only Train/Valid will be returned.")

        logger.info(f"Split for Test Year {test_year}:")
        logger.info(f" - Embargo: {embargo_days} trading days")
        logger.info(f" - Train: ~ {int(valid_y)-1} ({len(train_df)} rows, end={train_end_date})")
        logger.info(f" - Valid: {valid_y} ({len(valid_df)} rows, end={valid_end_date})")
        logger.info(f" - Test : {test_y} ({len(test_df)} rows)")
        
        return train_df, valid_df, test_df

    def build_anchored_splits(
        self,
        df: pl.DataFrame,
        n_folds: int = 3,
        embargo_days: int = 6,
        min_valid_days: int = 40,
    ) -> List[Tuple[Any, Any, Any]]:
        """
        Build anchored walk-forward splits on date axis.

        Returns:
            List of (train_end_date, valid_start_date, valid_end_date)
        """
        if df.is_empty() or "date" not in df.columns:
            return []

        unique_dates = (
            df.select("date")
            .unique()
            .sort("date")
            .get_column("date")
            .to_list()
        )

        n_dates = len(unique_dates)
        if n_dates < (n_folds + 1) * 20:
            return []

        block = n_dates // (n_folds + 1)
        if block < 20:
            return []

        embargo_days = int(max(0, embargo_days))
        min_valid_days = int(max(5, min_valid_days))

        splits: List[Tuple[Any, Any, Any]] = []
        for i in range(1, n_folds + 1):
            train_end_idx = (block * i) - 1
            valid_start_idx = (block * i) + embargo_days
            valid_end_idx = ((block * (i + 1)) - 1) if i < n_folds else (n_dates - 1)

            if train_end_idx < 0 or valid_start_idx >= n_dates:
                continue
            if valid_start_idx > valid_end_idx:
                continue

            valid_len = (valid_end_idx - valid_start_idx + 1)
            if valid_len < min_valid_days:
                continue

            splits.append(
                (
                    unique_dates[train_end_idx],
                    unique_dates[valid_start_idx],
                    unique_dates[valid_end_idx],
                )
            )

        return splits

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
