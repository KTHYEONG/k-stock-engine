import sys
import os
from pathlib import Path

# Add project root to sys.path before importing from src
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import argparse
import logging
import json
import polars as pl
import pandas as pd
from datetime import datetime

from src.legacy.stock_yetirank_v1.data.feature_store import FeatureStore
from src.legacy.etf_v1.optimizer import ETFOptimizer
from src.legacy.etf_v1.backtester import ETFBacktester
from src.legacy.etf_v1.etf_config import get_quarterly_window, ETFConfig

# Standard logger for diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("etf.run_static_opt")

# Clean logger for results (No prefix)
result_logger = logging.getLogger("etf.results")
result_logger.propagate = False
res_handler = logging.StreamHandler(sys.stdout)
res_handler.setFormatter(logging.Formatter('%(message)s'))
result_logger.addHandler(res_handler)

def main():
    parser = argparse.ArgumentParser(description="Run Static Optimization for ETF")
    parser.add_argument("--market", type=str, choices=["KOSPI", "KOSDAQ", "BOTH"], default="KOSPI")
    parser.add_argument("--trials", type=int, default=1500)
    parser.add_argument("--reference-date", type=str, default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    # Dates Configuration
    fetch_start, is_start, oos_start, oos_end = get_quarterly_window(args.reference_date)
    logger.info(f"Target IS Period: {is_start} ~ {oos_start}")
    logger.info(f"Target OOS Period: {oos_start} ~ {oos_end}")

    # Load Data
    etf_store = FeatureStore(base_path=Path("./data/etf_daily"))
    index_store = FeatureStore(base_path=Path("./data/market_index"))
    
    load_start = fetch_start.replace("-", "")
    load_end = oos_end.replace("-", "")

    try:
        etf_df = etf_store.load_features(start_date=load_start, end_date=load_end)
        index_df = index_store.load_features(start_date=load_start, end_date=load_end)
        
        if isinstance(etf_df, pl.LazyFrame): etf_df = etf_df.collect()
        if isinstance(index_df, pl.LazyFrame): index_df = index_df.collect()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    dt_fetch_start = datetime.strptime(fetch_start, "%Y-%m-%d").date()
    dt_oos_start = datetime.strptime(oos_start, "%Y-%m-%d").date()
    dt_oos_end = datetime.strptime(oos_end, "%Y-%m-%d").date()

    is_etf_df = etf_df.filter((pl.col("date") >= dt_fetch_start) & (pl.col("date") < dt_oos_start))
    is_index_df = index_df.filter((pl.col("date") >= dt_fetch_start) & (pl.col("date") < dt_oos_start))

    dt_fetch_start_oos = (datetime.strptime(oos_start, "%Y-%m-%d") - pd.Timedelta(days=400)).date()
    oos_etf_df = etf_df.filter((pl.col("date") >= dt_fetch_start_oos) & (pl.col("date") <= dt_oos_end))
    oos_index_df = index_df.filter((pl.col("date") >= dt_fetch_start_oos) & (pl.col("date") <= dt_oos_end))

    markets = ["KOSPI", "KOSDAQ"] if args.market == "BOTH" else [args.market]

    for market in markets:
        result_logger.info(f"\n{'='*60}\n>> Starting Optimization: {market}\n{'='*60}")
        
        optimizer = ETFOptimizer(is_index_df, is_etf_df, target_market=market, n_trials=args.trials)
        study = optimizer.optimize()

        trials = study.best_trials
        if not trials:
            logger.error(f"No trials completed for {market}")
            continue

        # Valid trials: positive Calmar, mdd under 15% (Wait, objective changed MDD to total_return_pct in values[1]?)
        # Let's check optimizer.py: return calmar, total_return_pct. So values[1] is Total Return!
        # So we want Calmar > 0.5 and we want to MAXIMIZE Return.
        valid_trials = [t for t in trials if t.values is not None and len(t.values) == 2 and t.values[0] > 0.5]
        
        if valid_trials:
            # Sort by Return (values[1]) primarily, since we want high yield.
            best_trial = max(valid_trials, key=lambda x: x.values[1]) 
        else:
            fallback_trials = [t for t in trials if t.values is not None and len(t.values) == 2]
            if fallback_trials:
                best_trial = max(fallback_trials, key=lambda x: x.values[1])
            else:
                logger.error(f"No valid trials found for {market}")
                continue

        best_params = best_trial.params.copy()
        
        # Display IS Performance
        is_results = optimizer.backtester.run(best_params, target_market=market)
        if is_results:
            res_is = is_results[0]
            trades_is = res_is["total_trades"]
            
            # Calculate IS CAGR
            equity_curve_is = res_is["equity_curve"]
            span_days_is = max(len(equity_curve_is), 1)
            total_ret_ratio_is = 1.0 + (res_is["total_return_pct"] / 100.0)
            cagr_is = ((total_ret_ratio_is ** (252.0 / span_days_is)) - 1.0) * 100.0 if total_ret_ratio_is > 0 else -100.0

            perf_is_output = [
                f"\n╔{'═'*58}╗",
                f"║ [IS Performance Summary: {market:<27}] ║",
                f"╠{'═'*58}╣",
                f"║  • Calmar Ratio   : {best_trial.values[0]:>10.2f}{' '*26}║",
                f"║  • Total Return   : {res_is['total_return_pct']:>10.2f}%{' '*24}║",
                f"║  • CAGR           : {cagr_is:>10.2f}%{' '*24}║",
                f"║  • Max Drawdown   : {res_is['mdd_pct']:>10.2f}%{' '*24}║",
                f"║  • Win Rate       : {res_is['win_rate']:>10.2f}%{' '*24}║",
                f"║  • Profit Factor  : {res_is['profit_factor']:>10.2f}{' '*25}║",
                f"║  • Total Trades   : {trades_is:>10}{' '*27}║",
                f"║  • Final Balance  : {res_is['final_balance']:>14,.0f} KRW{' '*17}║",
                f"╚{'═'*58}╝"
            ]
            result_logger.info("\n".join(perf_is_output))

        # Display Best Parameters
        param_output = [f"\n┌{'─'*58}┐", f"│ [Best IS Parameters Found: {market:<28}] │", f"├{'─'*58}┤"]
        for k, v in best_params.items():
            val_str = f"{v:>8.4f}" if isinstance(v, float) else f"{v:>8}"
            param_output.append(f"│  • {k:<25} : {val_str:<21} │")
        param_output.append(f"└{'─'*58}┘")
        result_logger.info("\n".join(param_output))

        # Evaluate OOS
        oos_backtester = ETFBacktester(oos_index_df, oos_etf_df)
        oos_results = oos_backtester.run(best_params, target_market=market)

        if not oos_results:
            logger.error(f"OOS Evaluation failed for {market}")
            continue

        res_oos = oos_results[0]
        
        # Calculate OOS CAGR
        equity_curve_oos = res_oos["equity_curve"]
        span_days_oos = max(len(equity_curve_oos), 1)
        total_ret_ratio_oos = 1.0 + (res_oos["total_return_pct"] / 100.0)
        cagr_oos = ((total_ret_ratio_oos ** (252.0 / span_days_oos)) - 1.0) * 100.0 if total_ret_ratio_oos > 0 else -100.0

        trades_df_oos = res_oos["trades_df"]
        if not trades_df_oos.empty:
            calmar_oos = res_oos['total_return_pct'] / abs(res_oos['mdd_pct']) if res_oos['mdd_pct'] > 0 else res_oos['total_return_pct']
        else:
            calmar_oos = 0.0

        perf_output = [
            f"\n╔{'═'*58}╗",
            f"║ [OOS Performance Summary: {market:<26}] ║",
            f"╠{'═'*58}╣",
            f"║  • Calmar Ratio   : {calmar_oos:>10.2f}{' '*26}║",
            f"║  • Total Return   : {res_oos['total_return_pct']:>10.2f}%{' '*24}║",
            f"║  • CAGR           : {cagr_oos:>10.2f}%{' '*24}║",
            f"║  • Max Drawdown   : {res_oos['mdd_pct']:>10.2f}%{' '*24}║",
            f"║  • Win Rate       : {res_oos['win_rate']:>10.2f}%{' '*24}║",
            f"║  • Profit Factor  : {res_oos['profit_factor']:>10.2f}{' '*25}║",
            f"║  • Total Trades   : {res_oos['total_trades']:>10}{' '*27}║",
            f"║  • Final Balance  : {res_oos['final_balance']:>14,.0f} KRW{' '*17}║",
            f"╚{'═'*58}╝"
        ]
        result_logger.info("\n".join(perf_output))

        # Save Params
        if res_oos['total_return_pct'] > 0 and res_oos['mdd_pct'] < 25.0 and res_oos['profit_factor'] >= 1.1 and res_oos['total_trades'] >= 5:
            results_dir = Path(PROJECT_ROOT) / "results" / "etf"
            results_dir.mkdir(parents=True, exist_ok=True)
            save_path = results_dir / f"best_params_{market}.json"
            with open(save_path, "w") as f:
                json.dump(best_params, f, indent=4)
            result_logger.info(f"✅ Saved Best Config to {save_path} (Passed Go-No-Go)")
        else:
            result_logger.info(f"🔴 NO-GO: Strategy failed to pass OOS checks.")

if __name__ == "__main__":
    main()
