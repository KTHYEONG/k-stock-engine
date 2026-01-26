import optuna
import logging
import polars as pl
from typing import Dict, Any, List, Optional
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
    
    def __init__(self, data_loader: YetiRankDataLoader, target_year: int = 2024, n_trials: int = 30, full_df: Optional[pl.DataFrame] = None):
        self.loader = data_loader
        self.target_year = target_year
        self.n_trials = n_trials
        self.task_type = self._get_task_type()
        
        logger.info(f"Using device: {self.task_type} for training/tuning.")
        
        if full_df is None:
            logger.info("Loading full data for tuning...")
            full_df = self.loader.load_full_data()
        
        self.feature_names = self.loader.get_feature_names(full_df)
        
        # Split for Tuning (Phase 1 Strategy)
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
        # 1. Hyperparameter Search Space
        params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20",
            "iterations": 1000, # 튜닝 시에는 속도를 위해 약간 줄임
            "od_type": "Iter",
            "od_wait": 50,
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "verbose": False,
            "allow_writing_files": False,
            "use_best_model": True, # 검증 점수 추적 활성화
            
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 30),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bootstrap_type": "Bernoulli",
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
        
        # 3. Return Best Score (안전한 점수 추출)
        # 우선순위: model.best_score_ -> model.get_best_score() -> model.get_evals_result()
        scores = {}
        if hasattr(model, "best_score_") and model.best_score_:
            scores = model.best_score_
        else:
            scores = model.get_best_score()
            
        if not scores:
            eval_result = model.get_evals_result()
            # evals_result에서 마지막 값이거나 가장 좋은 값을 추출 시도
            valid_key = next((k for k in eval_result.keys() if "validation" in k or "test" in k), None)
            if valid_key:
                metric_key = next((m for m in eval_result[valid_key].keys() if "NDCG" in m), None)
                if metric_key:
                    return float(max(eval_result[valid_key][metric_key]))
        
        # Validation Key 찾기
        valid_key = next((k for k in scores.keys() if "validation" in k or "test" in k), None)
        if valid_key is None:
            logger.error(f"Score extraction failed. Available keys in scores: {list(scores.keys())}")
            return 0.0
            
        metric_key = next((m for m in scores[valid_key].keys() if "NDCG" in m), "NDCG:top=20")
        if metric_key not in scores[valid_key]:
             return 0.0

        return float(scores[valid_key][metric_key])

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
