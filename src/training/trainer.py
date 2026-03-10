from pathlib import Path
import sys
import json
import logging
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Union
import polars as pl
from catboost import CatBoostRanker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

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
    
    def __init__(self, start_date: str = "20180101"):
        self.start_date = start_date
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
        
    def train_and_evaluate(
        self,
        test_periods: List[str] = ["2024Q1", "2024Q2"],
        n_trials: int = 180,
        sample_ratio: float = 1.0,
        run_diagnostics: bool = True,
        phase2_embargo_days: int = 6,
        awfo_tuning: bool = True,
        awfo_folds: int = 3,
        awfo_embargo_days: int = 6,
        awfo_min_valid_days: int = 40,
        tuner_seeds: str = "13,37,73",
        tuner_min_trials_per_seed: int = 50,
    ):
        # Ensure deterministic chronological workflow and clear run state.
        if not test_periods:
            raise ValueError("test_periods must not be empty.")
        test_periods = sorted({str(y) for y in test_periods})
        self.models = {}
        self.results = []

        import time
        t0_total = time.perf_counter()
        # 1. Load Full Data
        t0 = time.perf_counter()
        full_df = self.loader.load_full_data(sample_ratio=sample_ratio)
        feature_names = self.loader.get_feature_names(full_df)
        logger.info(f"⏱️ Data load+feature selection took {time.perf_counter() - t0:.2f}s")
        target_horizon_days = self.loader.infer_target_horizon_days(full_df, default=5)
        min_required_embargo = int(target_horizon_days + 1)
        if awfo_embargo_days < min_required_embargo:
            logger.warning(
                f"AWFO embargo_days={awfo_embargo_days} is too small for target horizon={target_horizon_days}d. "
                f"Auto-adjusting to {min_required_embargo}."
            )
            awfo_embargo_days = min_required_embargo
        if phase2_embargo_days < min_required_embargo:
            logger.warning(
                f"Phase2 embargo_days={phase2_embargo_days} is too small for target horizon={target_horizon_days}d. "
                f"Auto-adjusting to {min_required_embargo}."
            )
            phase2_embargo_days = min_required_embargo
        
        # 2. Hyperparameter Tuning (Phase 1)
        # Always tune against the earliest test period to avoid temporal leakage.
        tuning_target_year = test_periods[0]
        logger.info(f">>> Starting Phase 1: Hyperparameter Tuning (Target: Valid before {tuning_target_year})")
        
        logger.info(
            f"   AWFO Tuning: {'ON' if awfo_tuning else 'OFF'} "
            f"(folds={awfo_folds}, embargo_days={awfo_embargo_days}, min_valid_days={awfo_min_valid_days})"
        )
        tuner = YetiRankTuner(
            self.loader,
            target_year=tuning_target_year,
            n_trials=n_trials,
            full_df=full_df,
            use_awfo=awfo_tuning,
            awfo_folds=awfo_folds,
            awfo_embargo_days=awfo_embargo_days,
            awfo_min_valid_days=awfo_min_valid_days,
        )
        parsed_seeds = []
        for raw in str(tuner_seeds).split(","):
            s = raw.strip()
            if not s:
                continue
            try:
                parsed_seeds.append(int(s))
            except ValueError:
                continue
        if not parsed_seeds:
            parsed_seeds = [13]
        logger.info(
            f"   Tuner seeds: {parsed_seeds} (min_trials_per_seed={tuner_min_trials_per_seed})"
        )
        t0 = time.perf_counter()
        best_params = tuner.run_tuning(
            seeds=parsed_seeds,
            min_trials_per_seed=tuner_min_trials_per_seed,
        )
        logger.info(f"⏱️ Hyperparameter tuning took {time.perf_counter() - t0:.2f}s")
        
        # Save Best Params
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
        if getattr(tuner, "last_tuning_meta", None):
            with open(self.output_dir / "tuning_meta.json", "w", encoding="utf-8") as f:
                json.dump(tuner.last_tuning_meta, f, indent=4, ensure_ascii=False)
            logger.info(f"🧾 Saved tuning metadata: {self.output_dir / 'tuning_meta.json'}")
            
        # 3. Walk-Forward Training & Testing (Phase 2)
        logger.info(f">>> Starting Phase 2: Walk-Forward Training {test_periods}")
        logger.info(f"   Phase 2 embargo_days: {phase2_embargo_days}")
        
        # 고정 파라미터 (13개 핵심 피처 최적화 세팅)
        static_params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=20", # 실전 타겟(Top-20)에 맞춘 정밀 타격
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "allow_writing_files": False,
            "bootstrap_type": "Bernoulli",
            "iterations": 2000,           # 피처 감소로 인한 조기 수렴 고려
            "early_stopping_rounds": 200, # 학습 인내심 상향 (안정적 수렴)
            "learning_rate": 0.03,        # 정교한 학습 속도 유지
            "depth": 7,                   # 13개 알짜 피처의 고차원 상호작용 탐색
            "l2_leaf_reg": 5.0,           # 과적합 방지 규제
            "subsample": 0.7,             # 데이터 무작위성 확보
            "min_data_in_leaf": 50,       # 리프 노드 안정성 확보
            "random_strength": 1.5        # 모델 일반화(Robustness) 강화
        }
        
        # [FIX] colsample_bylevel (rsm) is not supported on GPU for YetiRank
        if self.task_type == "CPU":
            static_params["colsample_bylevel"] = 0.8
        final_params = {**static_params, **best_params}
        
        latest_model_path = None
        
        for period in test_periods:
            logger.info(f"⏳ Training Target Period: {period}...")
            t0_year = time.perf_counter()
            
            # Split Data (Expanding Window)
            train_df, valid_df, test_df = self.loader.walk_forward_split(
                full_df,
                test_year=period,
                embargo_days=phase2_embargo_days,
            )
            
            # Apply the same recency weighting policy used in tuning.
            train_df = self.loader.apply_time_decay_weights(
                train_df,
                min_weight=0.5,
                max_weight=1.0,
                context=f"phase2-{year}",
            )
            
            # Create Pools
            train_pool = self.loader.create_pool(train_df, feature_names)
            valid_pool = self.loader.create_pool(valid_df, feature_names)
            test_pool = self.loader.create_pool(test_df, feature_names)
            
            # Train Model
            model = CatBoostRanker(**final_params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                verbose=False
            )
            
            # Save Model
            model_path = self.output_dir / f"yetirank_{period}.cbm"
            model.save_model(str(model_path))
            self.models[period] = model
            latest_model_path = model_path
            
            # [CRITICAL UPDATE] Save latest copy for live trading bot
            shutil.copy2(model_path, self.output_dir / "yetirank_latest.cbm")
            
            # [CRITICAL FIX] 2026Q1과 같이 정답(Test) 데이터가 없는 경우 평가 스킵
            if test_df.is_empty():
                logger.info(f"✅ Period {period} Model saved without evaluation (No test data yet).")
                self.results.append({
                    "period": period,
                    "ndcg_50": 0.0,
                    "rank_ic": 0.0,
                    "ic_ir": 0.0,
                    "best_iteration": model.get_best_iteration()
                })
                continue

            # Evaluate on Test Set (Robust Key Detection)
            metrics = model.eval_metrics(test_pool, ["NDCG:top=20"])
            metric_key = next((m for m in metrics.keys() if "NDCG" in m), None)
            
            if metric_key:
                final_ndcg = metrics[metric_key][-1]
                logger.info(f"✅ Period {period} Completed. Test NDCG@20: {final_ndcg:.4f} (Best Iter: {model.get_best_iteration()})")
            else:
                final_ndcg = 0.0
                logger.warning(f"⚠️ Period {period} NDCG key not found in {list(metrics.keys())}")
            
            quality_metrics = {"rank_ic": 0.0, "ic_ir": 0.0}
            if run_diagnostics:
                # Feature Importance
                fi_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": model.get_feature_importance(data=train_pool)
                }).sort_values(by="importance", ascending=False)
                
                fi_path = self.output_dir / f"feature_importance_{period}.csv"
                fi_df.to_csv(fi_path, index=False)
                
                # [New] Advanced Quality Evaluation
                preds = model.predict(test_pool)
                quality_metrics = self._evaluate_prediction_quality(period, test_df, preds, feature_names)
            else:
                logger.info("⏩ Diagnostics disabled: skipping feature importance and decile/IC analysis.")

            self.results.append({
                "period": period,
                "ndcg_20": final_ndcg,
                "rank_ic": quality_metrics["rank_ic"],
                "ic_ir": quality_metrics["ic_ir"],
                "best_iteration": model.get_best_iteration()
            })
            logger.info(f"⏱️ Period {period} total elapsed: {time.perf_counter() - t0_year:.2f}s")
            
        # 4. Final Summary
        summary_df = pd.DataFrame(self.results)
        print("\n" + "="*60)
        print("🏆 Walk-Forward Evaluation Summary")
        print("="*60)
        print(summary_df)
        print("="*60 + "\n")
        summary_df.to_csv(self.output_dir / "evaluation_summary.csv", index=False)
        awfo_profile = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "start_date": self.start_date,
            "test_periods": test_periods,
            "tuning_target_year": tuning_target_year,
            "awfo_tuning": bool(awfo_tuning),
            "awfo_folds": int(awfo_folds),
            "awfo_embargo_days": int(awfo_embargo_days),
            "awfo_min_valid_days": int(awfo_min_valid_days),
            "phase2_embargo_days": int(phase2_embargo_days),
            "n_trials": int(n_trials),
            "sample_ratio": float(sample_ratio),
            "model_files": [f"yetirank_{p}.cbm" for p in test_periods],
            "best_params_file": "best_params.json",
            "summary_file": "evaluation_summary.csv",
        }
        with open(self.output_dir / "awfo_profile.json", "w", encoding="utf-8") as f:
            json.dump(awfo_profile, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved AWFO profile metadata: {self.output_dir / 'awfo_profile.json'}")
        logger.info(f"⏱️ End-to-end training elapsed: {time.perf_counter() - t0_total:.2f}s")
        
        return summary_df

    def _evaluate_prediction_quality(self, period: str, df: pl.DataFrame, preds: np.ndarray, feature_names: List[str]):
        """모델의 예측 품질을 다각도로 분석 (Rank IC, Decile Analysis) - Polars 기반 고속 연산"""
        
        # 순서 보장을 위한 정렬 (create_pool과 동일 로직) 및 예측값 결합
        eval_df = df.sort(["group_id", "ticker"]).with_columns(
            pl.Series("pred_score", preds)
        )
        
        # 1. Rank IC (Information Coefficient) - Polars Vectorized
        # 날짜별 Spearman Correlation 계산
        ic_results = (
            eval_df
            .group_by("date")
            .agg(
                pl.corr("pred_score", "target_rank", method="spearman").alias("rank_ic")
            )
            .drop_nulls()
        )
        
        avg_rank_ic = ic_results["rank_ic"].mean() or 0.0
        ic_ir = avg_rank_ic / ic_results["rank_ic"].std() if ic_results["rank_ic"].std() > 0 else 0
        
        # 2. Decile Analysis (분위수 분석) - Polars 기반
        # 분위수 계산 (0~9) - 예측 점수가 높을수록 높은 분위수 할당
        decile_stats = (
            eval_df
            .with_columns(
                (pl.col("pred_score").rank("ordinal").over("date") / pl.len().over("date") * 9.99)
                .cast(pl.Int32).alias("decile")
            )
            .group_by("decile")
            .agg(pl.col("target_return_5d").mean().alias("avg_ret"))
            .sort("decile")
        )
        
        # 시각화 (Matplotlib는 데이터 양이 적은 집계 결과만 사용하므로 Pandas 변환)
        plot_df = decile_stats.to_pandas()
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(plot_df["decile"], plot_df["avg_ret"], color='skyblue')
            plt.title(f"Decile Analysis ({period}) - Mean 5D Return by Score Group")
            plt.xlabel("Decile (0=Lowest, 9=Highest Prediction)")
            plt.ylabel("Avg 5D Log Return")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(self.output_dir / f"decile_analysis_{period}.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save decile plot: {e}")
        
        logger.info(f"📊 [{period} Quality] Rank IC: {avg_rank_ic:.4f} | IC IR: {ic_ir:.4f}")
        
        return {
            "rank_ic": avg_rank_ic,
            "ic_ir": ic_ir
        }

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    # 현재 날짜 기준 동적 기본값 계산
    now = datetime.now()
    curr_period = f"{now.year}Q{(now.month-1)//3 + 1}"
    
    parser = argparse.ArgumentParser(description="YetiRank Model Trainer")
    parser.add_argument("--trials", type=int, default=180, help="Number of hyperparameter tuning trials")
    parser.add_argument("--sample", type=float, default=1.0, help="Data sampling ratio (0.1 ~ 1.0)")
    parser.add_argument("--periods", type=str, default=curr_period, help=f"Comma separated test periods (default: {curr_period})")
    parser.add_argument("--start", type=str, default="20180101", help="Start date (YYYYMMDD) - Practical 6~8y window default")
    parser.add_argument("--skip_diagnostics", action="store_true", help="Skip feature importance and decile/IC diagnostics for faster runtime")
    parser.add_argument("--phase2_embargo_days", type=int, default=6, help="Embargo days for Phase 2 walk-forward split")
    parser.add_argument("--disable_awfo_tuning", action="store_true", help="Disable AWFO tuning and use single train/valid split")
    parser.add_argument("--awfo_folds", type=int, default=3, help="AWFO folds for tuning")
    parser.add_argument("--awfo_embargo_days", type=int, default=6, help="Embargo days between train and validation folds")
    parser.add_argument("--awfo_min_valid_days", type=int, default=40, help="Minimum validation days per AWFO fold")
    parser.add_argument("--tuner_seeds", type=str, default="13,37,73", help="Comma-separated Optuna seeds for robust tuning")
    parser.add_argument("--tuner_min_trials_per_seed", type=int, default=50, help="Minimum allocated trials per seed")
    
    args = parser.parse_args()
    
    test_periods = [p.strip() for p in args.periods.split(",")]
    
    trainer = YetiRankTrainer(start_date=args.start)
    trainer.train_and_evaluate(
        test_periods=test_periods, 
        n_trials=args.trials, 
        sample_ratio=args.sample,
        run_diagnostics=not args.skip_diagnostics,
        phase2_embargo_days=args.phase2_embargo_days,
        awfo_tuning=not args.disable_awfo_tuning,
        awfo_folds=args.awfo_folds,
        awfo_embargo_days=args.awfo_embargo_days,
        awfo_min_valid_days=args.awfo_min_valid_days,
        tuner_seeds=args.tuner_seeds,
        tuner_min_trials_per_seed=args.tuner_min_trials_per_seed,
    ) 
