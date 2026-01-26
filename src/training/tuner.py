import optuna
import logging
import polars as pl
import sys
import os
from typing import Dict, Any, List, Optional
from catboost import CatBoostRanker
from src.training.data_loader import YetiRankDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("training.tuner")

class YetiRankTuner:
    def __init__(self, data_loader: YetiRankDataLoader, target_year: int = 2024, n_trials: int = 30, full_df: Optional[pl.DataFrame] = None):
        self.loader = data_loader
        self.target_year = target_year
        self.n_trials = n_trials
        self.task_type = self._get_task_type()
        
        logger.info(f"🚀 Using device: {self.task_type} for training/tuning.")
        
        if full_df is None:
            full_df = self.loader.load_full_data()
        
        self.feature_names = self.loader.get_feature_names(full_df)
        self.train_df, self.valid_df, _ = self.loader.walk_forward_split(full_df, test_year=target_year)
        
        logger.info(f"📦 Preparing Pools (Target Year {target_year})...")
        self.train_pool = self.loader.create_pool(self.train_df, self.feature_names)
        self.valid_pool = self.loader.create_pool(self.valid_df, self.feature_names)

    def _get_task_type(self) -> str:
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                return "GPU"
        except:
            pass
        return "CPU"

    def objective(self, trial: optuna.Trial) -> float:
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
        
        model = CatBoostRanker(**params)
        
        # [Suppression] 불필요한 GPU 경고 로그 차단
        with open(os.devnull, 'w') as fnull:
            old_stderr = sys.stderr
            sys.stderr = fnull
            try:
                model.fit(self.train_pool, eval_set=self.valid_pool, early_stopping_rounds=50, verbose=False)
            finally:
                sys.stderr = old_stderr
        
        score = 0.0
        try:
            # 1. get_best_score 우선 확인
            best_scores = model.get_best_score()
            if best_scores:
                valid_key = next((k for k in best_scores.keys() if "learn" not in k.lower()), None)
                if valid_key:
                     metric_key = next((m for m in best_scores[valid_key].keys() if "NDCG" in m), None)
                     if metric_key:
                         score = float(best_scores[valid_key][metric_key])

            # 2. 실패시 get_evals_result 확인 (가장 확실한 방법)
            if score == 0.0:
                evals = model.get_evals_result()
                valid_key = next((k for k in evals.keys() if "learn" not in k.lower()), None)
                if valid_key:
                    metric_key = next((m for m in evals[valid_key].keys() if "NDCG" in m), None)
                    if metric_key:
                        history = evals[valid_key][metric_key]
                        if history:
                            score = float(max(history))
        except Exception as e:
            # 로깅은 최소화 (로그 파일에만 남게끔 debug 레벨 추천하나 여기선 pass)
            pass

        # 진행률 표시 강화: [01/30] 형식 추가
        progress = f"[{trial.number + 1:02d}/{self.n_trials}]"
        logger.info(f"📊 {progress} NDCG@20: {score:.4f} | LR: {params['learning_rate']:.4f}, Depth: {params['depth']}")
        return score

    def run_tuning(self) -> Dict[str, Any]:
        logger.info(f"⚡ Hyperparameter Tuning Start ({self.n_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"✅ Tuning Completed. Best NDCG: {study.best_value:.4f}")
        return study.best_params
