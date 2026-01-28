
import optuna
import logging
from pathlib import Path
import sys
import pandas as pd
import polars as pl
from typing import Optional, Dict, Any

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.backtester import YetiRankBacktester
from src.evaluation.walk_forward import StockWalkForwardAnalyzer
from src.evaluation.monte_carlo import StockMonteCarloSimulator
from src.utils.logger import setup_logger
import numpy as np

logger = setup_logger("evaluation.verifier")

class YetiRankVerifier:
    """
    YetiRank 전략 검증 도구 (Verification Suite)
    - OOS (Out-Sample) 백테스트
    - Walk-Forward Consistency Check
    - Monte Carlo Simulation
    """
    
    def __init__(self, mode: str = "UNIFIED", start_date: str = "20250101", end_date: str = "20251231", 
                 model_year: Optional[int] = None, sizing_mode: str = 'EQUAL'):
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.model_year = model_year
        self.sizing_mode = sizing_mode
        self.backtester = YetiRankBacktester(start_date=start_date, end_date=end_date, model_year=model_year)
        
    def load_best_params(self) -> Dict[str, Any]:
        """최적화된 DB에서 Best Parameter 로드"""
        db_path = PROJECT_ROOT / "results" / f"optimization_{self.mode.lower()}.db"
        
        if not db_path.exists():
            logger.error(f"❌ Optimization DB not found: {db_path}")
            return {}
            
        study_name = f"yetirank_{self.mode.lower()}_opt"
        storage_name = f"sqlite:///{db_path}"
        
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            logger.info(f"✅ Loaded Best Params from {db_path} (Trial #{study.best_trial.number})")
            logger.info(f"   Score: {study.best_value:.4f}")
            return study.best_params
        except Exception as e:
            logger.error(f"❌ Failed to load study: {e}")
            return {}

    def run_verification(self, n_wfa_splits: int = 5, n_mc_sims: int = 10000):
        print("\n" + "="*80)
        print(f"🚀 STRATEGY VERIFICATION: {self.mode}")
        print(f"   Period: {self.start_date} ~ {self.end_date}")
        print("="*80)
        
        # 1. Load Params
        best_params = self.load_best_params()
        if not best_params:
            print("⚠️ No optimized parameters found. Exiting.")
            return

        # Prepare Params (lowercase keys for backtester)
        run_params = {k.lower(): v for k, v in best_params.items()}
        print(f"\n🔎 Testing with Best Parameters (Sizing: {self.sizing_mode}):")
        print(run_params)
        
        # 2. Run Backtest
        # Enable return_details to get daily df and trade records
        metrics, daily_df, trade_records = self.backtester.run_backtest(
            save_plot=False, 
            return_details=True, 
            sizing_mode=self.sizing_mode,
            **run_params
        )
        
        print("\n" + "="*50)
        print("📊 BASE BACKTEST RESULT")
        print("="*50)
        for k, v in metrics.items():
            print(f"{k:<20}: {v}")
        print("="*50)

        if daily_df.is_empty():
            print("❌ Backtest returned no data.")
            return

        # 3. Walk-Forward Analysis (Consistency)
        print(f"\n🏃 Running Walk-Forward Consistency Check ({n_wfa_splits} Splits)...")
        wfa = StockWalkForwardAnalyzer(daily_df)
        wfa_results = wfa.run(n_splits=n_wfa_splits)
        
        if not wfa_results.is_empty():
            print(wfa_results)
            avg_ret = wfa_results["Return (%)"].mean()
            consistency = (wfa_results.filter(pl.col("Return (%)") > 0).height / wfa_results.height) * 100
            print(f"\n   Average Return per Split: {avg_ret:.2f}%")
            print(f"   Consistency (Positive %): {consistency:.0f}%")
        else:
            print("⚠️ Not enough data for WFA.")

        # 4. Monte Carlo Simulation (Unified Portfolio-Based)
        print(f"\n🎲 Running Monte Carlo Simulation ({n_mc_sims} runs)...")
        
        if not daily_df.is_empty():
            # [MODIFIED] 전체 타임라인 재조합 (0% 포함, 현실적/보수적 분석)
            daily_rets = daily_df["net_return"].to_numpy()
            
            if len(daily_rets) < 10:
                print("⚠️ Not enough daily data points for Monte Carlo.")
            else:
                final_cum_rets = []
                sim_mdds = []
                ruin_count = 0
                
                # Vectorized Simulation Loop (Efficient)
                for _ in range(n_mc_sims):
                    # 1. Resample Daily Returns (Bootstrap)
                    sim_daily = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
                    
                    # 2. Compound Returns
                    equity_curve = np.cumprod(1 + sim_daily)
                    
                    # 3. Final Return
                    total_ret_pct = (equity_curve[-1] - 1) * 100
                    final_cum_rets.append(total_ret_pct)
                    
                    # 4. MDD Calculation
                    peak = np.maximum.accumulate(equity_curve)
                    dd = (equity_curve - peak) / np.maximum(peak, 1e-6) * 100
                    min_dd = np.min(dd)
                    sim_mdds.append(min_dd)
                    
                    if min_dd <= -50.0: ruin_count += 1

                # Statistics
                final_cum_rets = np.array(final_cum_rets)
                sim_mdds = np.array(sim_mdds)
                
                prob_profit = np.mean(final_cum_rets > 0) * 100
                mean_ret = np.mean(final_cum_rets)
                median_ret = np.median(final_cum_rets)
                ruin_prob = (ruin_count / n_mc_sims) * 100
                worst_mdd_5pct = np.percentile(sim_mdds, 5) # 하위 5% MDD (보수적)
                
                lower_95 = np.percentile(final_cum_rets, 2.5)
                upper_95 = np.percentile(final_cum_rets, 97.5)

                print(f"{'='*50}")
                print(f"MONTE CARLO SIMULATION RESULT (95% Confidence)")
                print(f"Mode: Full Timeline (Includes 0% returns)")
                print(f"{'='*50}")
                print(f"Ruining Prob (>50% DD) : {ruin_prob:.2f}%")
                print(f"Probability of Profit : {prob_profit:.2f}%")
                print(f"Expected Return       : {mean_ret:.2f}% (Median: {median_ret:.2f}%)")
                print(f"Worst Case MDD (5%)   : {worst_mdd_5pct:.2f}%")
                print(f"Return Range (95%)    : {lower_95:.2f}% ~ {upper_95:.2f}%")
                print("="*50)
        else:
            print("⚠️ No data available for Monte Carlo.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Strategy Verifier")
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["UNIFIED", "ACTIVE", "SWING", "TREND"], help="Optimization Mode")
    parser.add_argument("--start", type=str, default="20250101", help="Start Date")
    parser.add_argument("--end", type=str, default="20251231", help="End Date")
    parser.add_argument("--splits", type=int, default=5, help="WFA Splits")
    parser.add_argument("--model_year", type=int, default=2025, help="Fixed model year to use")
    parser.add_argument("--sizing", type=str, default="CONFIDENCE", choices=["EQUAL", "CONFIDENCE", "RISK", "HYBRID"], help="Position Sizing Mode")
    
    args = parser.parse_args()
    
    verifier = YetiRankVerifier(mode=args.mode, start_date=args.start, end_date=args.end, model_year=args.model_year, sizing_mode=args.sizing)
    verifier.run_verification(n_wfa_splits=args.splits)
