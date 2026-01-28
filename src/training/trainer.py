from pathlib import Path
import sys
import json
import logging
from typing import Dict, Any, List
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
        
        # 고정 파라미터 (한정된 데이터셋 최적화: 꼼꼼한 학습 전략)
        static_params = {
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=50",
            "task_type": self.task_type,
            "devices": "0" if self.task_type == "GPU" else None,
            "bootstrap_type": "Bernoulli",
            "iterations": 3000,           # 데이터가 적으므로 더 많이 반복해서 패턴 탐색
            "early_stopping_rounds": 150, # 인내심 상향
            "learning_rate": 0.03,        # 느리지만 정교하게 학습
            "l2_leaf_reg": 5.0,           # 과적합 방지를 위한 규제 강화
            "subsample": 0.7,             # 데이터의 무작위성을 부여하여 일반화 성능 향상
            "min_data_in_leaf": 50,       # 리프 노드 최소 데이터 수 (노이즈 방어)
        }
        
        # [FIX] colsample_bylevel (rsm) is not supported on GPU for YetiRank
        if self.task_type == "CPU":
            static_params["colsample_bylevel"] = 0.8
        final_params = {**static_params, **best_params}
        
        for year in test_years:
            logger.info(f"⏳ Training Target Year: {year}...")
            
            # Split Data (Expanding Window)
            train_df, valid_df, test_df = self.loader.walk_forward_split(full_df, test_year=year)
            
            # [NEW] Apply Time-Decay Sample Weighting (Recency Bias)
            # 최근 데이터일수록 가중치를 높여(0.5 ~ 1.0) 시장 변화(Regime Shift)에 적응
            max_date = train_df["date"].max()
            min_date = train_df["date"].min()
            
            if max_date > min_date:
                # Polars Date Difference -> Normalize -> Rescale
                train_df = train_df.with_columns(
                    (
                        0.5 + 0.5 * (pl.col("date") - pl.lit(min_date)).dt.total_days() / 
                        (pl.lit(max_date) - pl.lit(min_date)).dt.total_days()
                    ).cast(pl.Float32).alias("sample_weight")
                )
                logger.info(f"⚖️ Applied Time-Decay Weights: 0.5 ~ 1.0 (Train Size: {len(train_df)})")
            
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
            model_path = self.output_dir / f"yetirank_{year}.cbm"
            model.save_model(str(model_path))
            self.models[year] = model
            
            # Evaluate on Test Set (Robust Key Detection)
            metrics = model.eval_metrics(test_pool, ["NDCG:top=50"])
            metric_key = next((m for m in metrics.keys() if "NDCG" in m), None)
            
            if metric_key:
                final_ndcg = metrics[metric_key][-1]
                logger.info(f"✅ Year {year} Completed. Test NDCG@50: {final_ndcg:.4f} (Best Iter: {model.get_best_iteration()})")
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
            
            # [New] Advanced Quality Evaluation
            preds = model.predict(test_pool)
            quality_metrics = self._evaluate_prediction_quality(year, test_df, preds, feature_names)

            self.results.append({
                "year": year,
                "ndcg_50": final_ndcg,
                "rank_ic": quality_metrics["rank_ic"],
                "ic_ir": quality_metrics["ic_ir"],
                "best_iteration": model.get_best_iteration()
            })
            
        # 4. Final Summary
        summary_df = pd.DataFrame(self.results)
        print("\n" + "="*60)
        print("🏆 Walk-Forward Evaluation Summary")
        print("="*60)
        print(summary_df)
        print("="*60 + "\n")
        summary_df.to_csv(self.output_dir / "evaluation_summary.csv", index=False)
        
        return summary_df

    def _evaluate_prediction_quality(self, year: int, df: pl.DataFrame, preds: np.ndarray, feature_names: List[str]):
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
            plt.title(f"Decile Analysis ({year}) - Mean 5D Return by Score Group")
            plt.xlabel("Decile (0=Lowest, 9=Highest Prediction)")
            plt.ylabel("Avg 5D Log Return")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(self.output_dir / f"decile_analysis_{year}.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save decile plot: {e}")
        
        logger.info(f"📊 [{year} Quality] Rank IC: {avg_rank_ic:.4f} | IC IR: {ic_ir:.4f}")
        
        return {
            "rank_ic": avg_rank_ic,
            "ic_ir": ic_ir
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Model Trainer")
    parser.add_argument("--trials", type=int, default=100, help="Number of hyperparameter tuning trials")
    parser.add_argument("--sample", type=float, default=1.0, help="Data sampling ratio (0.1 ~ 1.0)")
    parser.add_argument("--years", type=str, default="2024,2025", help="Comma separated test years")
    parser.add_argument("--start", type=str, default="20200101", help="Start date (YYYYMMDD) - Recent Regime Focus")
    
    args = parser.parse_args()
    
    test_years = [int(y.strip()) for y in args.years.split(",")]
    
    trainer = YetiRankTrainer(start_date=args.start)
    trainer.train_and_evaluate(
        test_years=test_years, 
        n_trials=args.trials, 
        sample_ratio=args.sample
    ) 
