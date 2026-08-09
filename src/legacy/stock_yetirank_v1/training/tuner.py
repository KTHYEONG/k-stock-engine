import optuna
import logging
import polars as pl
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from catboost import CatBoostRanker
from src.legacy.stock_yetirank_v1.training.data_loader import YetiRankDataLoader
from src.legacy.stock_yetirank_v1.utils.logger import setup_logger

logger = setup_logger("training.tuner")

class YetiRankTuner:
    def __init__(
        self,
        data_loader: YetiRankDataLoader,
        target_year: str = "2024",
        n_trials: int = 30,
        full_df: Optional[pl.DataFrame] = None,
        use_awfo: bool = True,
        awfo_folds: int = 3,
        awfo_embargo_days: int = 6,
        awfo_min_valid_days: int = 40,
    ):
        self.loader = data_loader
        self.target_year = target_year
        self.n_trials = n_trials
        self.task_type = self._get_task_type()
        self.use_awfo = bool(use_awfo)
        self.awfo_folds = int(max(2, awfo_folds))
        self.awfo_embargo_days = int(max(0, awfo_embargo_days))
        self.awfo_min_valid_days = int(max(5, awfo_min_valid_days))
        self.awfo_pools: List[Tuple[Any, Any]] = []
        self.last_tuning_meta: Dict[str, Any] = {}
        
        logger.info(f"🚀 Using device: {self.task_type} for training/tuning.")
        
        if full_df is None:
            full_df = self.loader.load_full_data()

        target_horizon_days = self.loader.infer_target_horizon_days(full_df, default=5)
        min_required_embargo = int(target_horizon_days + 1)
        if self.awfo_embargo_days < min_required_embargo:
            logger.warning(
                f"AWFO embargo_days={self.awfo_embargo_days} is too small for target horizon={target_horizon_days}d. "
                f"Auto-adjusting to {min_required_embargo}."
            )
            self.awfo_embargo_days = min_required_embargo
        
        self.feature_names = self.loader.get_feature_names(full_df)

        is_quarter = isinstance(target_year, str) and "Q" in target_year
        if is_quarter:
            tune_df = full_df.filter(pl.col("period") < target_year)
        else:
            tune_df = full_df.filter(pl.col("year") < str(target_year))
            
        if self.use_awfo and not tune_df.is_empty():
            split_defs = self.loader.build_anchored_splits(
                tune_df,
                n_folds=self.awfo_folds,
                embargo_days=self.awfo_embargo_days,
                min_valid_days=self.awfo_min_valid_days,
            )

            logger.info(
                f"📦 Preparing AWFO Pools ({len(split_defs)} folds, embargo={self.awfo_embargo_days}d)..."
            )
            for train_end, valid_start, valid_end in split_defs:
                fold_train_df = tune_df.filter(pl.col("date") <= train_end)
                fold_valid_df = tune_df.filter(
                    (pl.col("date") >= valid_start) & (pl.col("date") <= valid_end)
                )
                if fold_train_df.is_empty() or fold_valid_df.is_empty():
                    continue
                fold_train_df = self.loader.apply_time_decay_weights(
                    fold_train_df,
                    min_weight=0.5,
                    max_weight=1.0,
                    context=f"awfo-fold-{len(self.awfo_pools)+1}",
                )
                train_pool = self.loader.create_pool(fold_train_df, self.feature_names)
                valid_pool = self.loader.create_pool(fold_valid_df, self.feature_names)
                self.awfo_pools.append((train_pool, valid_pool))

            if len(self.awfo_pools) >= 2:
                logger.info(f"✅ AWFO tuning enabled with {len(self.awfo_pools)} folds.")
            else:
                logger.warning("⚠️ AWFO folds insufficient. Falling back to single Train/Valid split.")
                self.use_awfo = False
        else:
            self.use_awfo = False

        if not self.use_awfo:
            self.train_df, self.valid_df, _ = self.loader.walk_forward_split(
                full_df,
                test_year=target_year,
                embargo_days=self.awfo_embargo_days,
            )
            self.train_df = self.loader.apply_time_decay_weights(
                self.train_df,
                min_weight=0.5,
                max_weight=1.0,
                context="single-split",
            )
            logger.info(f"📦 Preparing single split Pools (Target Year {target_year})...")
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

    def _extract_ndcg_score(self, model: CatBoostRanker) -> float:
        score = 0.0
        try:
            best_scores = model.get_best_score()
            if best_scores:
                valid_key = next((k for k in best_scores.keys() if "learn" not in k.lower()), None)
                if valid_key:
                    metric_key = next((m for m in best_scores[valid_key].keys() if "NDCG" in m), None)
                    if metric_key:
                        score = float(best_scores[valid_key][metric_key])

            if score == 0.0:
                evals = model.get_evals_result()
                valid_key = next((k for k in evals.keys() if "learn" not in k.lower()), None)
                if valid_key:
                    metric_key = next((m for m in evals[valid_key].keys() if "NDCG" in m), None)
                    if metric_key:
                        history = evals[valid_key][metric_key]
                        if history:
                            score = float(max(history))
        except Exception:
            pass
        return score

    @staticmethod
    def _allocate_seed_trials(
        total_trials: int,
        seeds: List[int],
        min_trials_per_seed: int,
    ) -> List[Tuple[int, int]]:
        total_trials = int(max(1, total_trials))
        if not seeds:
            return [(13, total_trials)]

        min_trials_per_seed = int(max(1, min_trials_per_seed))
        max_seed_count = max(1, total_trials // min_trials_per_seed)
        active = seeds[:max_seed_count]
        if not active:
            active = [seeds[0]]

        base = total_trials // len(active)
        rem = total_trials % len(active)
        alloc: List[Tuple[int, int]] = []
        for idx, seed in enumerate(active):
            n = base + (1 if idx < rem else 0)
            if n > 0:
                alloc.append((int(seed), int(n)))
        return alloc or [(int(active[0]), total_trials)]

    @staticmethod
    def _robust_value_from_study(study: optuna.Study) -> float:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not completed:
            return -float("inf")
        vals = np.array(sorted([float(t.value) for t in completed], reverse=True), dtype=np.float64)
        top_k = vals[: max(3, min(12, len(vals)))]
        top_mean = float(np.mean(top_k))
        top_p25 = float(np.percentile(top_k, 25))
        return (0.65 * top_mean) + (0.35 * top_p25)

    @staticmethod
    def _metric_summary(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "max": 0.0,
            }
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _hist_summary(values: List[float], bins: int = 6) -> Dict[str, Any]:
        if not values:
            return {"bin_edges": [], "counts": []}
        arr = np.asarray(values, dtype=np.float64)
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
        if abs(v_max - v_min) < 1e-12:
            return {
                "bin_edges": [v_min, v_max],
                "counts": [int(arr.size)],
            }
        hist, edges = np.histogram(arr, bins=int(max(3, bins)))
        return {
            "bin_edges": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in hist.tolist()],
        }

    def _summarize_awfo_distribution(self, study: optuna.Study) -> Dict[str, Any]:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        avg_vals: List[float] = []
        p25_vals: List[float] = []
        worst_vals: List[float] = []
        std_vals: List[float] = []
        gap_vals: List[float] = []
        for t in completed:
            ua = t.user_attrs or {}
            if "awfo_avg_ndcg" in ua:
                avg_vals.append(float(ua.get("awfo_avg_ndcg", 0.0)))
            if "awfo_p25_ndcg" in ua:
                p25_vals.append(float(ua.get("awfo_p25_ndcg", 0.0)))
            if "awfo_worst_ndcg" in ua:
                worst_vals.append(float(ua.get("awfo_worst_ndcg", 0.0)))
            if "awfo_std_ndcg" in ua:
                std_vals.append(float(ua.get("awfo_std_ndcg", 0.0)))
            if "awfo_downside_gap" in ua:
                gap_vals.append(float(ua.get("awfo_downside_gap", 0.0)))

        has_awfo = len(avg_vals) > 0
        summary = {
            "available": bool(has_awfo),
            "trial_count_with_awfo": int(len(avg_vals)),
            "avg_ndcg": self._metric_summary(avg_vals),
            "p25_ndcg": self._metric_summary(p25_vals),
            "worst_ndcg": self._metric_summary(worst_vals),
            "std_ndcg": self._metric_summary(std_vals),
            "downside_gap": self._metric_summary(gap_vals),
            "avg_ndcg_hist": self._hist_summary(avg_vals, bins=6),
        }
        return summary

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20",
            "iterations": 2000,
            "od_type": "Iter",
            "od_wait": 100,
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "allow_writing_files": False,
            "use_best_model": True,
            "logging_level": "Silent",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 30),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            # Deterministic training to reduce run-to-run noise.
            "random_seed": 42,
        }
        
        # [FIX] colsample_bylevel (rsm) is not supported on GPU for YetiRank
        if self.task_type == "CPU":
            params["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.5, 1.0)

        # 진행률 표시 강화: [01/30] 형식 추가
        progress = f"[{trial.number + 1:02d}/{self.n_trials}]"
        if self.use_awfo and self.awfo_pools:
            fold_scores = []
            try:
                with open(os.devnull, 'w') as fnull:
                    old_stderr = sys.stderr
                    sys.stderr = fnull
                    try:
                        for fold_idx, (train_pool, valid_pool) in enumerate(self.awfo_pools):
                            model = CatBoostRanker(**params)
                            model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50, verbose=False)
                            fold_score = self._extract_ndcg_score(model)
                            fold_scores.append(fold_score)
                            trial.report(fold_score, fold_idx + 1)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                    finally:
                        sys.stderr = old_stderr
            except optuna.TrialPruned:
                raise
            except Exception:
                return 0.0

            if not fold_scores:
                return 0.0

            avg_score = float(np.mean(fold_scores))
            p25_score = float(np.percentile(fold_scores, 25))
            worst_score = float(np.min(fold_scores))
            std_score = float(np.std(fold_scores))
            downside_gap = max(0.0, avg_score - worst_score)
            robust_score = (
                (0.40 * avg_score)
                + (0.40 * p25_score)
                + (0.20 * worst_score)
                - (0.05 * std_score)
                - (0.10 * downside_gap)
            )

            trial.set_user_attr("awfo_avg_ndcg", avg_score)
            trial.set_user_attr("awfo_p25_ndcg", p25_score)
            trial.set_user_attr("awfo_worst_ndcg", worst_score)
            trial.set_user_attr("awfo_std_ndcg", std_score)
            trial.set_user_attr("awfo_downside_gap", downside_gap)

            logger.info(
                f"📊 {progress} AWFO-Robust NDCG@50: {robust_score:.4f} "
                f"(avg={avg_score:.4f}, p25={p25_score:.4f}, worst={worst_score:.4f}) "
                f"| LR: {params['learning_rate']:.4f}, Depth: {params['depth']}"
            )
            return robust_score

        model = CatBoostRanker(**params)
        with open(os.devnull, 'w') as fnull:
            old_stderr = sys.stderr
            sys.stderr = fnull
            try:
                model.fit(self.train_pool, eval_set=self.valid_pool, early_stopping_rounds=50, verbose=False)
            finally:
                sys.stderr = old_stderr

        score = self._extract_ndcg_score(model)
        logger.info(f"📊 {progress} NDCG@50: {score:.4f} | LR: {params['learning_rate']:.4f}, Depth: {params['depth']}")
        return score

    def run_tuning(self, seeds: Optional[List[int]] = None, min_trials_per_seed: int = 40) -> Dict[str, Any]:
        logger.info(f"⚡ Hyperparameter Tuning Start ({self.n_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        seed_list = [int(s) for s in (seeds or [13])]
        alloc = self._allocate_seed_trials(self.n_trials, seed_list, min_trials_per_seed=min_trials_per_seed)

        best_study: Optional[optuna.Study] = None
        best_robust = -float("inf")
        best_seed: Optional[int] = None
        seed_summaries: List[Dict[str, Any]] = []

        for seed, seed_trials in alloc:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=max(5, seed_trials // 5),
                multivariate=True,
                seed=int(seed),
            )
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=max(5, seed_trials // 4),
                n_warmup_steps=1,
                interval_steps=1,
            )
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
            study.optimize(self.objective, n_trials=int(seed_trials))

            robust = self._robust_value_from_study(study)
            best_val = float(study.best_value) if len(study.trials) > 0 else -float("inf")
            complete_cnt = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned_cnt = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            awfo_dist = self._summarize_awfo_distribution(study) if self.use_awfo else {}
            seed_summaries.append(
                {
                    "seed": int(seed),
                    "trials": int(seed_trials),
                    "best_value": float(best_val),
                    "robust_value": float(robust),
                    "complete_trials": int(complete_cnt),
                    "pruned_trials": int(pruned_cnt),
                    "awfo_distribution": awfo_dist,
                }
            )
            logger.info(
                f"[SEED {seed}] trials={seed_trials} | best={best_val:.4f} | robust={robust:.4f}"
            )
            if np.isfinite(robust) and robust > best_robust:
                best_robust = float(robust)
                best_study = study
                best_seed = int(seed)

        if best_study is None:
            raise RuntimeError("Tuning failed: no completed trials across all seeds.")

        label = "AWFO-Robust NDCG" if self.use_awfo else "NDCG"
        self.last_tuning_meta = {
            "target_year": str(self.target_year),
            "use_awfo": bool(self.use_awfo),
            "awfo_folds": int(self.awfo_folds) if self.use_awfo else 0,
            "awfo_embargo_days": int(self.awfo_embargo_days) if self.use_awfo else 0,
            "awfo_min_valid_days": int(self.awfo_min_valid_days) if self.use_awfo else 0,
            "seed_allocations": [{"seed": int(s), "trials": int(t)} for s, t in alloc],
            "best_seed": int(best_seed) if best_seed is not None else None,
            "best_value": float(best_study.best_value),
            "best_robust": float(best_robust),
            "best_seed_awfo_distribution": self._summarize_awfo_distribution(best_study) if self.use_awfo else {},
            "seed_summaries": seed_summaries,
        }
        logger.info(
            f"✅ Tuning Completed. Best {label}: {best_study.best_value:.4f} "
            f"(robust={best_robust:.4f})"
        )
        return best_study.best_params
