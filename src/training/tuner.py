import optuna
import logging
from typing import Dict, Any
from catboost import CatBoostRanker
from src.training.data_loader import YetiRankDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("training.tuner")

class YetiRankTuner:
    """
    Optuna를 활용한 CatBoost YetiRank 하이퍼파라미터 튜닝
    - Expanding Window Walk-Forward 튜닝은 비용이 크므로, 
      대표적인 구간(예: 가장 최근 Valid Year)에 대해 튜닝 수행.
    """
    
    def __init__(self, data_loader: YetiRankDataLoader, target_year: int = 2024, n_trials: int = 30):
        self.loader = data_loader
        self.target_year = target_year
        self.n_trials = n_trials
        self.task_type = self._get_task_type()
        
        logger.info(f"Using device: {self.task_type} for training/tuning.")
        full_df = self.loader.load_full_data()
        self.feature_names = self.loader.get_feature_names(full_df)
        
        # Split for Tuning (Phase 1 Strategy)
        # Train: ~ 2022, Valid: 2023 
        # (만약 target_year가 2024라면, valid_year는 2023)
        self.train_df, self.valid_df, _ = self.loader.walk_forward_split(full_df, test_year=target_year)
        
        logger.info(f"Preparing Pools for Tuning (Target Year {target_year})...")
        self.train_pool = self.loader.create_pool(self.train_df, self.feature_names)
        self.valid_pool = self.loader.create_pool(self.valid_df, self.feature_names)

    def _get_task_type(self) -> str:
        """GPU 사용 가능 여부 확인"""
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                return "GPU"
        except:
            pass
        return "CPU"

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Hyperparameter Search Space (Financial Data Optimized)
        params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20",
            "iterations": 2000,
            "od_type": "Iter",
            "od_wait": 50,  # Early Stopping Patience
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "verbose": False,
            "allow_writing_files": False,
            
            # Tuning Range
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 30),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0)
        }
        
        # 2. Train Model
        model = CatBoostRanker(**params)
        model.fit(
            self.train_pool,
            eval_set=self.valid_pool,
            early_stopping_rounds=params["od_wait"],
            verbose=False
        )
        
        # 3. Return Best Score (NDCG is maximized, but Optuna minimizes by default? 
        # No, Optuna maximizes if direction="maximize". We will set direction="maximize" later)
        best_score = model.get_best_score()["validation"]["NDCG:top=20"]
        return best_score

    def run_tuning(self) -> Dict[str, Any]:
        logger.info(f"Starting Hyperparameter Tuning ({self.n_trials} trials)...")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info("Tuning Completed.")
        logger.info(f"Best Score (NDCG@20): {study.best_value:.4f}")
        logger.info(f"Best Params: {study.best_params}")
        
        return study.best_params

if __name__ == "__main__":
    # Test Code
    loader = YetiRankDataLoader(start_date="20160401")
    tuner = YetiRankTuner(loader, target_year=2024, n_trials=5) # 5 trials for quick test
    best_params = tuner.run_tuning()
    print("Test Best Params:", best_params)
