from optuna.integration import CatBoostPruningCallback

# ... (기존 import 유지)

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Hyperparameter Search Space
        params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20",
            "iterations": 600, # [SpeedUp] 튜닝 시 반복 횟수 축소 (1000 -> 600)
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
        
        # [SpeedUp] Pruning Callback 추가
        try:
            model.fit(
                self.train_pool,
                eval_set=self.valid_pool,
                early_stopping_rounds=params["od_wait"],
                verbose=False,
                callbacks=[CatBoostPruningCallback(trial, "NDCG:top=20")]
            )
        except optuna.TrialPruned:
            # Pruning 발생 시 Optuna에게 알림
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

        # Trial 결과 요약 출력
        logger.info(f"Trial {trial.number:02d} | NDCG@20: {score:.4f} | LR: {params['learning_rate']:.4f}, Depth: {params['depth']}")
        return score

    def run_tuning(self) -> Dict[str, Any]:
        logger.info(f"Hyperparameter Tuning Start ({self.n_trials} trials)...")
        
        # Optuna 기본 로그 끄기
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info(f"Tuning Completed. Best NDCG: {study.best_value:.4f}")
        return study.best_params

if __name__ == "__main__":
    # Test Code
    loader = YetiRankDataLoader(start_date="20160401")
    tuner = YetiRankTuner(loader, target_year=2024, n_trials=5) # 5 trials for quick test
    best_params = tuner.run_tuning()
    print("Test Best Params:", best_params)
