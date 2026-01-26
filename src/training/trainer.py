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
        self.task_type = self._get_task_type()
        
    def _get_task_type(self) -> str:
        """GPU 사용 가능 여부 확인"""
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                return "GPU"
        except:
            pass
        return "CPU"
        
    def train_and_evaluate(self, test_years: List[int] = [2024, 2025], n_trials: int = 100, sample_ratio: float = 1.0):
        # 1. Load Full Data
        full_df = self.loader.load_full_data(sample_ratio=sample_ratio)
        feature_names = self.loader.get_feature_names(full_df)
        
        # 2. Hyperparameter Tuning (Phase 1)
        # 입력받은 테스트 연도 중 첫 번째 연도를 기준으로 하이퍼파라미터 최적화 수행
        tuning_target_year = test_years[0]
        logger.info(f">>> Starting Phase 1: Hyperparameter Tuning (Target: Valid {tuning_target_year-1})")
        
        tuner = YetiRankTuner(self.loader, target_year=tuning_target_year, n_trials=n_trials, full_df=full_df)
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
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "bootstrap_type": "Bernoulli", # subsample 사용을 위해 필수
        }
        final_params = {**static_params, **best_params}
        
        for year in test_years:
            logger.info(f"⏳ Training Target Year: {year}...")
            
            # Split Data (Expanding Window)
            train_df, valid_df, test_df = self.loader.walk_forward_split(full_df, test_year=year)
            
            # Create Pools
            train_pool = self.loader.create_pool(train_df, feature_names)
            valid_pool = self.loader.create_pool(valid_df, feature_names)
            test_pool = self.loader.create_pool(test_df, feature_names)
            
            # Train Model
            model = CatBoostRanker(**final_params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Save Model
            model_path = self.output_dir / f"yetirank_{year}.cbm"
            model.save_model(str(model_path))
            self.models[year] = model
            
            # Evaluate on Test Set (Robust Key Detection)
            metrics = model.eval_metrics(test_pool, ["NDCG:top=20"])
            metric_key = next((m for m in metrics.keys() if "NDCG" in m), None)
            
            if metric_key:
                final_ndcg = metrics[metric_key][-1]
                logger.info(f"✅ Year {year} Completed. Test NDCG@20: {final_ndcg:.4f} (Best Iter: {model.get_best_iteration()})")
            else:
                final_ndcg = 0.0
                logger.warning(f"⚠️ Year {year} NDCG key not found in {list(metrics.keys())}")
            
            # Feature Importance
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.get_feature_importance(data=train_pool)
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
        print("\n" + "="*40)
        print("🏆 Walk-Forward Evaluation Summary")
        print("="*40)
        print(summary_df)
        print("="*40 + "\n")
        summary_df.to_csv(self.output_dir / "evaluation_summary.csv", index=False)
        
        return summary_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Model Trainer")
    parser.add_argument("--trials", type=int, default=30, help="Number of hyperparameter tuning trials")
    parser.add_argument("--sample", type=float, default=1.0, help="Data sampling ratio (0.1 ~ 1.0)")
    parser.add_argument("--years", type=str, default="2024,2025", help="Comma separated test years")
    parser.add_argument("--start", type=str, default="20160401", help="Start date (YYYYMMDD)")
    
    args = parser.parse_args()
    
    test_years = [int(y.strip()) for y in args.years.split(",")]
    
    trainer = YetiRankTrainer(start_date=args.start)
    trainer.train_and_evaluate(
        test_years=test_years, 
        n_trials=args.trials, 
        sample_ratio=args.sample
    ) 
