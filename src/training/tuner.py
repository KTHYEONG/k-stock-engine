import optuna
import logging
import polars as pl
from typing import Dict, Any, List, Optional
from catboost import CatBoostRanker
from src.training.data_loader import YetiRankDataLoader
from src.utils.logger import setup_logger
from optuna.integration import CatBoostPruningCallback

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
            "iterations": 600, 
            "od_type": "Iter",
            "od_wait": 50,
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "allow_writing_files": False,
            "use_best_model": True,
            "logging_level": "Silent",
            
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 30),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0)
        }
        
        # 2. Train Model
        model = CatBoostRanker(**params)
        
        try:
            model.fit(
                self.train_pool,
                eval_set=self.valid_pool,
                early_stopping_rounds=params["od_wait"],
                verbose=False,
                callbacks=[CatBoostPruningCallback(trial, "NDCG:top=20")]
            )
        except optuna.TrialPruned:
            raise optuna.TrialPruned()
        
        # 3. Return Best Score
        scores = {}
        if hasattr(model, "best_score_") and model.best_score_:
            scores = model.best_score_
        else:
            scores = model.get_best_score()
            
        score = 0.0
        if scores:
            valid_key = next((k for k in scores.keys() if "validation" in k or "test" in k), None)
            if valid_key:
                metric_key = next((m for m in scores[valid_key].keys() if "NDCG" in m), None)
                if metric_key:
                    score = float(scores[valid_key][metric_key])

        logger.info(f"Trial {trial.number:02d} | NDCG@20: {score:.4f} | LR: {params['learning_rate']:.4f}, Depth: {params['depth']}")
        return score

    def run_tuning(self) -> Dict[str, Any]:
        logger.info(f"Hyperparameter Tuning Start ({self.n_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"Tuning Completed. Best NDCG: {study.best_value:.4f}")
        return study.best_params

if __name__ == "__main__":
    loader = YetiRankDataLoader(start_date="20160401")
    tuner = YetiRankTuner(loader, target_year=2024, n_trials=5)
    best_params = tuner.run_tuning()
    print("Test Best Params:", best_params)
