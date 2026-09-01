import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import logging

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from legacy.stock_yetirank_v1.data.feature_store import FeatureStore
from legacy.stock_yetirank_v1.utils.logger import setup_logger

logger = setup_logger("validation.feature_selection")

class FeatureSelector:
    """
    YetiRank 학습 전, 스피어만 상관계수(Spearman Correlation) 기반 자동 피처 선택
    - Polars 기반 고속 연산 및 상세 진단 로직 포함
    """
    
    def __init__(self, target_col="target_return_5d", corr_threshold=0.7):
        self.store = FeatureStore()
        self.target_col = target_col
        self.corr_threshold = corr_threshold
        self.yaml_path = PROJECT_ROOT / "data" / "model_features" / "features_v1.yaml"
        
        # 최신 후보 피처 리스트
        self.candidate_features = [
            "overnight_ret", "intraday_ret", "ret_2_5d", "ret_6_20d", "ret_21_60d",
            "trend_120d_rank", "vol_20d_rank", "vol_regime", "flow_intensity_20d",
            "flow_consensus", "volume_shock", "mcap_rank", "sector_ret_5d",
            "vol_asymmetry_20d", "close_high_ratio_10d", "vix_zscore_20d"
        ]

    def select_features(self, start_date="20220101", end_date="20241231"):
        logger.info(f"🔍 Starting Advanced Feature Selection ({start_date} ~ {end_date})")
        
        # 1. 데이터 로드
        ldf = self.store.load_features(start_date=start_date, end_date=end_date, file_pattern="*_feat.parquet")
        df = ldf.collect()
        
        if df.is_empty():
            logger.error("❌ No data found.")
            return

        # 2. 타겟 유효성 검사
        if self.target_col not in df.columns:
            logger.error(f"❌ Target '{self.target_col}' missing.")
            return
            
        target_std = df[self.target_col].std()
        if target_std == 0 or target_std is None:
            logger.error(f"❌ Target '{self.target_col}' has ZERO variance. Correlation impossible.")
            return

        # 3. 피처별 개별 진단 및 IC 계산
        results = []
        valid_features_for_matrix = []
        
        logger.info(f"{'Feature':<25} | {'Non-Null':<8} | {'Unique':<8} | {'Spearman IC':<10}")
        logger.info("-" * 60)
        
        for feat in self.candidate_features:
            if feat not in df.columns:
                continue
                
            # 해당 피처와 타겟이 모두 있는 행만 추출
            pair_df = df.select([feat, self.target_col]).drop_nulls()
            
            if pair_df.is_empty():
                logger.warning(f"{feat:<25} | 0        | 0        | SKIP (Empty)")
                continue
                
            n_rows = len(pair_df)
            n_unique = pair_df[feat].n_unique()
            
            if n_unique <= 1:
                logger.warning(f"{feat:<25} | {n_rows:<8} | {n_unique:<8} | SKIP (Constant)")
                continue
                
            # 상관계수 계산 (Polars)
            ic = pair_df.select(pl.corr(feat, self.target_col, method="spearman")).item()
            
            logger.info(f"{feat:<25} | {n_rows:<8} | {n_unique:<8} | {ic:.4f}")
            
            results.append({"feature": feat, "ic": abs(ic)})
            valid_features_for_matrix.append(feat)

        if not results:
            logger.error("❌ No valid features found for selection.")
            return

        # 4. 다중공선성 제거 (Pandas matrix 사용 - 피처 수가 적으므로 안전)
        # 모든 유효 피처가 공존하는 데이터로 매트릭스 생성
        final_check_df = df.select(valid_features_for_matrix).drop_nulls()
        if final_check_df.is_empty():
            logger.warning("⚠️ No rows have all features. Skipping multicollinearity check, using all valid features.")
            final_features = valid_features_for_matrix
        else:
            corr_matrix = final_check_df.to_pandas().corr(method="spearman")
            dropped = set()
            target_ic_map = {r["feature"]: r["ic"] for r in results}
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    f1, f2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    if f1 in dropped or f2 in dropped: continue
                    
                    if abs(corr_matrix.iloc[i, j]) > self.corr_threshold:
                        # IC가 낮은 쪽 탈락
                        drop_f = f2 if target_ic_map[f1] >= target_ic_map[f2] else f1
                        dropped.add(drop_f)
                        logger.info(f"🚫 Dropping {drop_f} (High corr with {f1 if drop_f==f2 else f2})")
            
            final_features = [f for f in valid_features_for_matrix if f not in dropped]

        # 5. 결과 저장
        logger.info(f"\n✅ Selection Complete: {len(final_features)} features selected.")
        self._write_yaml(final_features)

    def _write_yaml(self, features: list):
        config_data = {
            "version": "1.1",
            "description": "Upgraded orthogonal features (Spearman + Diagnostics)",
            "target": self.target_col,
            "features": features
        }
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info(f"💾 Saved to {self.yaml_path}")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import dateutil.relativedelta
    
    # 동적 윈도우 할당: 시장 트렌드 변화를 반영하기 위해 롤링 '최근 3년' 사용
    now = datetime.now()
    default_start = (now - dateutil.relativedelta.relativedelta(years=3)).strftime("%Y%m%d")
    default_end = now.strftime("%Y%m%d")
    
    parser = argparse.ArgumentParser(description="Feature Selection Pipeline")
    parser.add_argument("--start", type=str, default=default_start, help=f"Start date (default: {default_start})")
    parser.add_argument("--end", type=str, default=default_end, help=f"End date (default: {default_end})")
    
    args = parser.parse_args()
    
    FeatureSelector().select_features(start_date=args.start, end_date=args.end)
