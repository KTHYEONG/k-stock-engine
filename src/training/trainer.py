from pathlib import Path
import sys
import json
import logging
from typing import Dict, Any, List
import polars as pl
from catboost import CatBoostRanker
import matplotlib.pyplot as plt
import pandas as pd

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.data_loader import YetiRankDataLoader
from src.training.tuner import YetiRankTuner
from src.utils.logger import setup_logger

logger = setup_logger("training.trainer")

class YetiRankTrainer:
    """
    Expanding Window Walk-Forward Training & Evaluation
    
    Process:
    1. Tune hyperparameters using Valid set of Phase 1 (e.g., 2023)
    2. Expand window and Retrain for each Test Year (2024, 2025)
    3. Evaluate and Save Results
    """
    
    def __init__(self, start_date: str = "20160401"):
        self.loader = YetiRankDataLoader(start_date=start_date)
        self.models = {}  # year -> model
        self.results = [] # Evaluation results
        self.output_dir = PROJECT_ROOT / "models" / "yetirank"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_and_evaluate(self, test_years: List[int] = [2024, 2025], n_trials: int = 30, sample_ratio: float = 1.0):
        # 1. Load Full Data
        full_df = self.loader.load_full_data(sample_ratio=sample_ratio)
        feature_names = self.loader.get_feature_names(full_df)
        
        # 2. Hyperparameter Tuning (Phase 1)
        # 2023년(가장 최근의 온전한 검증 데이터)을 기준으로 최적의 파라미터를 찾음
        logger.info(">>> Starting Phase 1: Hyperparameter Tuning (Target: Valid 2023)")
        # 튜닝 타겟 연도를 2024로 설정하면, 내부적으로 2016~2022 Train, 2023 Valid로 분할됨.
        tuner = YetiRankTuner(self.loader, target_year=2024, n_trials=n_trials)
        best_params = tuner.run_tuning()
        
        # Save Best Params
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
            
        # 3. Walk-Forward Training & Testing (Phase 2)
        logger.info(f">>> Starting Phase 2: Walk-Forward Training {test_years}")
        
        # 고정 파라미터 (손실 함수 등)
        static_params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20",
            "task_type": "CPU", # GPU 사용 시 변경
            "verbose": 100
        }
        final_params = {**static_params, **best_params}
        
        for year in test_years:
            logger.info(f"--- Processing Target Year: {year} ---")
            
            # Split Data (Expanding Window)
            train_df, valid_df, test_df = self.loader.walk_forward_split(full_df, test_year=year)
            
            # Create Pools
            # 학습 시에는 Train + Valid를 모두 합쳐서 학습 데이터로 쓰는 것이 일반적일 수 있으나,
            # Early Stopping을 위해 Valid를 분리 유지하거나, 
            # 충분히 튜닝되었다면 전체를 합치고 튜닝된 횟수(Iterations)만큼 학습하기도 함.
            # 여기서는 안전하게 튜닝 때와 동일한 구조(Train/Valid)로 학습하며 Early Stopping 적용.
            # (단, Train 데이터는 Expanding Window에 따라 매년 늘어남)
            
            train_pool = self.loader.create_pool(train_df, feature_names)
            valid_pool = self.loader.create_pool(valid_df, feature_names)
            test_pool = self.loader.create_pool(test_df, feature_names)
            
            # Train Model
            model = CatBoostRanker(**final_params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=50
            )
            
            # Save Model
            model_path = self.output_dir / f"yetirank_{year}.cbm"
            model.save_model(str(model_path))
            self.models[year] = model
            
            # Evaluate on Test Set
            # CatBoost의 score 메서드나 predict 메서드 활용
            # 랭킹 모델의 predict는 '점수'를 반환함. 이 점수로 NDCG 계산 필요.
            # 하지만 CatBoost는 eval_set에 대한 메트릭을 자동으로 계산해줌.
            # 여기서는 직접 eval_set=test_pool로 평가 점수를 얻거나, 예측값을 뽑아서 별도 분석.
            
            # 간단한 성능 지표 로깅 (Model internal metric)
            metrics = model.eval_metrics(test_pool, ["NDCG:top=20", "PFound", "AverageGain:top=20"])
            # 마지막 round의 점수
            final_ndcg = metrics["NDCG:top=20"][-1]
            logger.info(f"Year {year} Test NDCG@20: {final_ndcg:.4f}")
            
            # Feature Importance
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.get_feature_importance()
            }).sort_values(by="importance", ascending=False)
            
            fi_path = self.output_dir / f"feature_importance_{year}.csv"
            fi_df.to_csv(fi_path, index=False)
            
            self.results.append({
                "year": year,
                "ndcg_20": final_ndcg,
                "best_iteration": model.get_best_iteration()
            })
            
        # 4. Final Summary
        summary_df = pd.DataFrame(self.results)
        print("\n=== Walk-Forward Evaluation Summary ===")
        print(summary_df)
        summary_df.to_csv(self.output_dir / "evaluation_summary.csv", index=False)
        
        return summary_df

if __name__ == "__main__":
    trainer = YetiRankTrainer(start_date="20160401")
    # n_trials=1, sample_ratio=0.1 for quick test
    trainer.train_and_evaluate(test_years=[2024], n_trials=1, sample_ratio=0.1) 
