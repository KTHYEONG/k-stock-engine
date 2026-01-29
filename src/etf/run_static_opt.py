
import asyncio
import logging
import argparse
from datetime import datetime
import polars as pl
from pathlib import Path
import sys
import json
import numpy as np
import optuna

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("run_static_opt")
logging.getLogger("optuna").setLevel(logging.ERROR)
logging.getLogger("etf").setLevel(logging.WARNING)

from src.data.etf_manager import ETFManager
from src.etf.optimizer import ETFOptimizer
from src.etf.backtester import ETFBacktester

async def load_all_data(start_date: datetime, end_date: datetime):
    manager = ETFManager()
    try:
        etf_df = pl.scan_parquet(str(manager.etf_store.base_path / "**/*.parquet")).collect()
        index_df = pl.scan_parquet(str(manager.index_store.base_path / "**/*.parquet")).collect()
        
        etf_df = etf_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        index_df = index_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        return etf_df, index_df
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        return None, None

def analyze_parameter_sensitivity(study, top_n=30):
    """
    Optuna Search Result Sensitivity Analysis
    - Check if Top N params are clustered (Stable) or scattered (Random Peak).
    """
    print("\n" + "="*70)
    print(f"🔬 PARAMETER SENSITIVITY ANALYSIS (Top {top_n} Trials)")
    print("="*70)
    
    # 1. Get Completed Trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # 2. Sort by Value (Score) Descending
    trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = trials[:top_n]
    
    if len(top_trials) < 5:
        print("⚠️ Not enough trials for sensitivity analysis.")
        return

    # 3. Collect Params
    param_keys = top_trials[0].params.keys()
    stats = {}
    
    for key in param_keys:
        values = [t.params[key] for t in top_trials]
        
        # Check type (Numerical vs Categorical)
        if isinstance(values[0], (int, float)):
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            best_val = top_trials[0].params[key]
            
            # Z-Score of Best Param (How far from the Top N mean?)
            z_score = abs(best_val - mean_val) / (std_val + 1e-9)
            
            stats[key] = {
                "type": "num",
                "mean": mean_val,
                "std": std_val,
                "range": f"[{min_val} ~ {max_val}]",
                "best": best_val,
                "z_score": z_score
            }
        else:
            # Categorical: Find most frequent
            from collections import Counter
            counts = Counter(values)
            most_common = counts.most_common(1)[0]
            best_val = top_trials[0].params[key]
            
            stats[key] = {
                "type": "cat",
                "top_choice": f"{most_common[0]} ({most_common[1]}/{top_n})",
                "best": best_val,
                "stability": "HIGH" if most_common[0] == best_val else "LOW"
            }

    # 4. Print Report
    print(f"{'Param':<20} | {'Best':<10} | {'Top-N Mean':<10} | {'StdDev':<8} | {'Range / Stability'}")
    print("-" * 80)
    
    for key, s in stats.items():
        if s['type'] == 'num':
            # Highlight if Best is far from Mean (Z-Score > 2.0)
            warning = "⚠️" if s['z_score'] > 2.0 else ""
            print(f"{key:<20} | {s['best']:<10.4g} | {s['mean']:<10.4g} | {s['std']:<8.4f} | {s['range']:<15} {warning}")
        else:
            warning = "⚠️" if s['stability'] == "LOW" else ""
            print(f"{key:<20} | {s['best']:<10} | {'-':<10} | {'-':<8} | {s['top_choice']:<15} {warning}")
            
    print("-" * 80)
    print("💡 Interpretation:")
    print("  - StdDev Low & Z-Score Low: Stable Parameter (Safe)")
    print("  - ⚠️ Marked: Best param is an outlier compared to other top performers (Possible Overfitting)")
    print("="*70)

def analyze_market_regime_performance(daily_returns: list, index_df: pl.DataFrame):
    """
    Market Regime (Bull/Bear) Performance Analysis
    """
    if not daily_returns or not isinstance(daily_returns, list):
        return

    # 1. Align Dates (Tail Alignment assumption)
    n_days = len(daily_returns)
    
    # Calculate SMA on full index_df first to ensure validity
    full_idx = index_df.clone()
    
    # Clean Close Price if needed
    if "close" not in full_idx.columns:
         if "CLSPRC_IDX" in full_idx.columns:
             full_idx = full_idx.with_columns(
                pl.col("CLSPRC_IDX").cast(pl.Utf8).str.replace(",", "").cast(pl.Float64, strict=False).alias("close")
            )
         else:
             print("⚠️ Cannot find close price column for Regime Analysis.")
             return
             
    # Calculate SMA 200
    full_idx = full_idx.sort("date").with_columns(
        pl.col("close").rolling_mean(200).alias("sma200")
    )
    
    # Slice to match daily_returns length (Tail)
    if full_idx.height < n_days:
        print("⚠️ Not enough index data for regime analysis.")
        return
        
    analysis_df = full_idx.tail(n_days)
    
    # Add strategy returns
    analysis_df = analysis_df.with_columns(
        pl.Series(name="strat_ret", values=daily_returns)
    )
    
    # 2. Define Regimes
    # Bull: Close > SMA200
    # Bear: Close <= SMA200
    analysis_df = analysis_df.with_columns(
        pl.when(pl.col("close") > pl.col("sma200")).then(pl.lit("Bull"))
          .otherwise(pl.lit("Bear")).alias("regime")
    )
    
    # 3. GroupBy Analysis
    print("\n" + "="*70)
    print("🐻🐮 MARKET REGIME STRESS TEST (Based on SMA 200)")
    print("="*70)
    print(f"{'Regime':<10} | {'Days':<6} | {'Return':<8} | {'WinRate':<8} | {'MDD':<8}")
    print("-" * 60)
    
    regimes = ["Bull", "Bear"]
    
    for r in regimes:
        sub_df = analysis_df.filter(pl.col("regime") == r)
        if sub_df.height == 0:
            print(f"{r:<10} | {'0':<6} | {'-':<8} | {'-':<8} | {'-':<8}")
            continue
            
        # Calc Stats
        rets = sub_df["strat_ret"].to_numpy()
        cum_ret = np.prod(1 + rets) - 1
        
        # Simple WinRate
        wins = np.sum(rets > 0)
        win_rate = (wins / len(rets)) * 100 if len(rets) > 0 else 0
        
        # MDD (In-Regime MDD)
        cum_equity = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum_equity)
        dd = (cum_equity - running_max) / running_max
        max_dd = np.min(dd)
        
        print(f"{r:<10} | {len(rets):<6} | {cum_ret*100:>7.2f}% | {win_rate:>7.1f}% | {max_dd*100:>7.2f}%")
        
    print("-" * 60)
    print("💡 Check if the strategy survives in Bear markets (MDD not too deep).")
    print("="*70)

def run_static_optimization(etf_df: pl.DataFrame, index_df: pl.DataFrame, args):
    """
    Static Optimization: Train (Start ~ End-1) -> Test (End)
    """
    s_date = datetime.strptime(args.start, "%Y-%m-%d")
    e_date = datetime.strptime(args.end, "%Y-%m-%d") # This is 2025-12-31 generally
    
    # Split Date: 2025-01-01
    # Train End = 2024-12-31
    test_year = e_date.year
    train_end_date = datetime(test_year - 1, 12, 31)
    test_start_date = datetime(test_year, 1, 1)
    
    # 1. Split Data
    train_index = index_df.filter(pl.col("date") <= train_end_date)
    train_etf = etf_df.filter(pl.col("date") <= train_end_date)
    
    test_index = index_df.filter(pl.col("date") >= test_start_date)
    test_etf = etf_df.filter(pl.col("date") >= test_start_date)
    
    market = args.market
    
    print("\n" + "="*70)
    print(f"🏋️ STATIC OPTIMIZATION START")
    print(f" Target Market: {market}")
    print(f" Train Period : {s_date.date()} ~ {train_end_date.date()} ({train_index.height} rows)")
    print(f" Test Period  : {test_start_date.date()} ~ {e_date.date()} ({test_index.height} rows)")
    print("="*70)
    
    # 2. Optimization (Train)
    print(f"\nrunning optimization ({args.trials} trials)...")
    optimizer = ETFOptimizer(train_index, train_etf, target_market=market, target_leverage="HYBRID")
    best_params = optimizer.run_optimization(n_trials=args.trials)
    
    # Save Params
    param_path = PROJECT_ROOT / "results" / "static_opt" / f"best_params_{market}_static.json"
    param_path.parent.mkdir(parents=True, exist_ok=True)
    with open(param_path, "w") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"✅ Best Params Saved: {param_path}")
    print(f"Best Params: {json.dumps(best_params, indent=2)}")

    # 2.1 Parameter Sensitivity Analysis
    # try:
    #     analyze_parameter_sensitivity(optimizer.study, top_n=30)
    # except Exception as e:
    #     print(f"⚠️ Sensitivity Analysis Failed: {e}")
    
    # 3. Backtest (In-Sample)
    print("\n[In-Sample Performance (2017-2024)]")
    bt_train = ETFBacktester(train_index, train_etf)
    train_res_list = bt_train.run(best_params, target_market=market)
    train_res = next((r for r in train_res_list if r['market'] == f"{market}_HYBRID"), None)
    
    if train_res:
         print(f" Return: {train_res['total_return']*100:.2f}%")
         print(f" CAGR  : {train_res['cagr']*100:.2f}%")
         print(f" MDD   : {train_res['mdd']*100:.2f}%")
         print(f" Trades: {train_res['trades']}")
         print(f" WinRt : {train_res['win_rate']:.1f}%")

         # --- Monte Carlo Simulation ---
         if train_res.get('trade_list') and len(train_res['trade_list']) > 5:
             from src.etf.monte_carlo import ETFMonteCarloSimulator
             print("\n🎲 Monte Carlo Analysis (10,000 Runs)")
             mc = ETFMonteCarloSimulator(train_res['trade_list'])
             mc_res = mc.run(n_simulations=10000)
             
             if mc_res['is_valid']:
                 print(f" Prob. Profit : {mc_res['prob_profit']:.1f}% (Chance of making money)")
                 print(f" Median Return: {mc_res['median_return_pct']:.1f}% (Most likely outcome)")
                 print(f" Worst MDD    : -{mc_res['worst_case_mdd']:.1f}% (95% Confidence Risk)")
                 print(f" Return Range : {mc_res['lower_bound_95']:.1f}% ~ {mc_res['upper_bound_95']:.1f}%")
             else:
                 print(" (Not enough trades for simulation)")
    
    # 4. Backtest (Out-of-Sample)
    print("\n[Out-of-Sample Performance (2025)]")
    test_res = None
    if test_index.height > 0:
        bt_test = ETFBacktester(test_index, test_etf)
        test_res_list = bt_test.run(best_params, target_market=market)
        test_res = next((r for r in test_res_list if r['market'] == f"{market}_HYBRID"), None)
        
        if test_res:
             print(f" Return: {test_res['total_return']*100:.2f}%")
             print(f" MDD   : {test_res['mdd']*100:.2f}%")
             print(f" Trades: {test_res['trades']}")
             print(f" WinRt : {test_res['win_rate']:.1f}%")
        else:
             print("Detailed results not available (maybe no trades)")
    else:
        print("No test data available.")

    # 5. Yearly Breakdown (Critical for Regime Check)
    print("\n" + "="*70)
    print("📅 YEARLY BREAKDOWN (Regime Check)")
    print("="*70)
    print(f"{'Year':<6} | {'Return':<10} | {'MDD':<10} | {'Trades':<8} | {'WinRate':<8} | {'Regime'}")
    print("-" * 75)
    
    # 2017 ~ 2025
    start_y = s_date.year
    end_y = e_date.year
    
    total_etf = pl.concat([train_etf, test_etf]) if test_etf.height > 0 else train_etf
    total_idx = pl.concat([train_index, test_index]) if test_index.height > 0 else train_index
    
    for y in range(start_y, end_y + 1):
        y_s = datetime(y, 1, 1)
        y_e = datetime(y, 12, 31)
        
        # Filter (Use strict date filter)
        sub_etf = total_etf.filter((pl.col("date") >= y_s) & (pl.col("date") <= y_e))
        sub_idx = total_idx.filter((pl.col("date") >= y_s) & (pl.col("date") <= y_e))
        
        if sub_etf.is_empty(): continue
        
        bt_year = ETFBacktester(sub_idx, sub_etf)
        y_res_list = bt_year.run(best_params, target_market=market)
        y_res = next((r for r in y_res_list if r['market'] == f"{market}_HYBRID"), None)
        
        if y_res:
            ret = y_res['total_return'] * 100
            mdd = y_res['mdd'] * 100
            trd = y_res['trades']
            win = y_res['win_rate']
            
            # Simple Regime Annotation
            regime = ""
            if y in [2018, 2022]: regime = "📉 Bear"
            elif y in [2021]: regime = "🦀 Chop"
            elif y in [2017, 2020, 2024, 2025]: regime = "🚀 Bull"
            else: regime = "Normal"
            
            print(f"{y:<6} | {ret:>8.2f}% | {mdd:>8.2f}% | {trd:<8} | {win:>6.1f}% | {regime}")

    print("="*70)

    # 6. Regime Stress Test
    # Collect all daily returns from Train + Test
    full_daily_returns = []
    if train_res and 'daily_returns' in train_res:
         full_daily_returns.extend(train_res['daily_returns'])
    if test_res and 'daily_returns' in test_res:
         full_daily_returns.extend(test_res['daily_returns'])
         
    # if full_daily_returns:
    #    analyze_market_regime_performance(full_daily_returns, total_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="KOSPI")
    parser.add_argument("--trials", type=int, default=1500)
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    args = parser.parse_args()
    
    s_date = datetime.strptime(args.start, "%Y-%m-%d")
    e_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    etf, idx = asyncio.run(load_all_data(s_date, e_date))
    if etf is not None:
        run_static_optimization(etf, idx, args)

if __name__ == "__main__":
    main()
