import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_static_opt")
logging.getLogger("optuna").setLevel(logging.ERROR)
logging.getLogger("etf").setLevel(logging.WARNING)

from src.data.etf_manager import ETFManager
from src.etf.backtester import ETFBacktester
from src.etf.monte_carlo import ETFMonteCarloSimulator
from src.etf.optimizer import ETFOptimizer


def _parse_seed_list(seed_arg: str) -> List[int]:
    seeds = []
    for raw in str(seed_arg).split(","):
        s = raw.strip()
        if not s:
            continue
        try:
            seeds.append(int(s))
        except ValueError:
            continue
    return seeds or [13]


async def load_all_data(start_date: datetime, end_date: datetime) -> Tuple[pl.DataFrame, pl.DataFrame]:
    manager = ETFManager()
    etf_df = pl.scan_parquet(str(manager.etf_store.base_path / "**/*.parquet")).collect()
    index_df = pl.scan_parquet(str(manager.index_store.base_path / "**/*.parquet")).collect()
    etf_df = etf_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
    index_df = index_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
    return etf_df, index_df


def _equity_summary(daily_returns: List[float]) -> Dict[str, float]:
    arr = np.asarray(daily_returns, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return {
            "days": 0.0,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "mdd_pct": 0.0,
            "calmar": 0.0,
        }
    equity = np.cumprod(1.0 + arr)
    total = float(equity[-1] - 1.0)
    years = max(1.0 / 252.0, arr.size / 252.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    ann_mean = float(np.mean(arr) * 252.0)
    ann_vol = float(np.std(arr) * np.sqrt(252.0))
    sharpe = ann_mean / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    mdd = float(np.min(dd))
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0
    return {
        "days": float(arr.size),
        "total_return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": float(sharpe),
        "mdd_pct": mdd * 100.0,
        "calmar": float(calmar),
    }


def _run_awfo_verification(
    etf_df: pl.DataFrame,
    index_df: pl.DataFrame,
    market: str,
    best_params: Dict[str, Any],
    awfo_plan: List[Dict[str, Any]],
) -> None:
    rows = []
    all_daily = []
    all_trades = []
    for fold in awfo_plan:
        val_s = fold["val_start"]
        val_e = fold["val_end"]
        year = int(fold["eval_year"])
        val_etf = etf_df.filter((pl.col("date") >= val_s) & (pl.col("date") <= val_e))
        val_idx = index_df.filter((pl.col("date") >= val_s) & (pl.col("date") <= val_e))
        bt = ETFBacktester(val_idx, val_etf)
        res_list = bt.run(best_params, target_market=market)
        key = f"{market}_HYBRID"
        res = next((r for r in res_list if r.get("market") == key), None)
        if not res:
            continue
        daily = res.get("daily_returns", [])
        trades = res.get("trade_list", [])
        all_daily.extend(daily)
        all_trades.extend(trades)
        rows.append(
            {
                "Year": year,
                "Train Period": f"{fold['train_start'].date()} ~ {fold['train_end'].date()}",
                "Period": f"{val_s.date()} ~ {val_e.date()}",
                "Return (%)": round(float(res.get("total_return", 0.0)) * 100.0, 2),
                "MDD (%)": round(float(res.get("mdd", 0.0)) * 100.0, 2),
                "Trades": int(res.get("trades", 0)),
                "Win Rate (%)": round(float(res.get("win_rate", 0.0)), 2),
            }
        )

    if not rows:
        print("No AWFO OOS results were produced.")
        return

    df = pl.DataFrame(rows)
    print("\n" + "=" * 80)
    print("ETF AWFO OOS RESULT (Year-by-Year)")
    print("=" * 80)
    print(df.to_pandas().to_string(index=False))

    eq = _equity_summary(all_daily)
    ret_arr = np.asarray(df["Return (%)"].to_list(), dtype=float)
    print("\n" + "=" * 80)
    print("ETF AWFO OOS AGGREGATED SUMMARY")
    print("=" * 80)
    print(f"Years evaluated      : {df.height}")
    print(f"Mean yearly return   : {float(np.mean(ret_arr)):.2f}%")
    print(f"Median yearly return : {float(np.median(ret_arr)):.2f}%")
    print(f"Positive ratio       : {float(np.mean(ret_arr > 0) * 100.0):.0f}%")
    print(f"Combined return      : {eq['total_return_pct']:.2f}%")
    print(f"Combined CAGR        : {eq['cagr_pct']:.2f}%")
    print(f"Combined Sharpe      : {eq['sharpe']:.4f}")
    print(f"Combined MDD         : {eq['mdd_pct']:.2f}%")
    print(f"Combined Calmar      : {eq['calmar']:.4f}")

    print("\nMonte Carlo (10,000 runs)")
    mc = ETFMonteCarloSimulator(all_trades)
    mc_res = mc.run(n_simulations=10000)
    if mc_res.get("is_valid", False):
        print(f"Probability of Profit : {mc_res['prob_profit']:.2f}%")
        print(f"Expected Return       : {mc_res['mean_return_pct']:.2f}% (Median: {mc_res['median_return_pct']:.2f}%)")
        print(f"Worst Case MDD (5%)   : -{mc_res['worst_case_mdd']:.2f}%")
        print(f"Return Range (95%)    : {mc_res['lower_bound_95']:.2f}% ~ {mc_res['upper_bound_95']:.2f}%")
    else:
        print("Not enough trade samples for Monte Carlo.")


def run_static_optimization(etf_df: pl.DataFrame, index_df: pl.DataFrame, args) -> None:
    market = args.market.upper()

    print("\n" + "=" * 80)
    print(f"ETF OPTIMIZATION START | Market={market} | AWFO={bool(args.awfo)}")
    print("=" * 80)

    optimizer = ETFOptimizer(
        index_df=index_df,
        etf_df=etf_df,
        target_market=market,
        target_leverage="HYBRID",
        awfo=bool(args.awfo),
        awfo_start_year=args.awfo_start_year,
        awfo_end_year=args.awfo_end_year,
        awfo_train_years=args.awfo_train_years,
    )
    best_params = optimizer.run_optimization(
        n_trials=args.trials,
        seeds=_parse_seed_list(args.opt_seeds),
        min_trials_per_seed=args.opt_min_trials_per_seed,
        n_jobs=args.opt_jobs,
    )

    out_dir = PROJECT_ROOT / "results" / "etf"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "awfo" if args.awfo else "single"
    param_path = out_dir / f"best_params_{market.lower()}_{mode}.json"
    with open(param_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"Best params saved: {param_path}")

    if getattr(optimizer, "last_optimization_meta", None):
        meta_path = out_dir / f"opt_meta_{market.lower()}_{mode}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(optimizer.last_optimization_meta, f, indent=2, ensure_ascii=False)
        print(f"Optimization meta saved: {meta_path}")

    print("\nBest Params:")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))

    if args.awfo:
        _run_awfo_verification(
            etf_df=etf_df,
            index_df=index_df,
            market=market,
            best_params=best_params,
            awfo_plan=optimizer.awfo_plan,
        )
        return

    backtester = ETFBacktester(index_df, etf_df)
    res_list = backtester.run(best_params, target_market=market)
    res = next((r for r in res_list if r.get("market") == f"{market}_HYBRID"), None)
    if not res:
        print("No single-period backtest results.")
        return

    print("\n" + "=" * 80)
    print("SINGLE-PERIOD BACKTEST RESULT")
    print("=" * 80)
    print(f"Total Return : {float(res.get('total_return', 0.0)) * 100.0:.2f}%")
    print(f"CAGR         : {float(res.get('cagr', 0.0)) * 100.0:.2f}%")
    print(f"Sharpe       : {(_equity_summary(res.get('daily_returns', [])).get('sharpe', 0.0)):.4f}")
    print(f"MDD          : {float(res.get('mdd', 0.0)) * 100.0:.2f}%")
    print(f"Trades       : {int(res.get('trades', 0))}")
    print(f"Win Rate     : {float(res.get('win_rate', 0.0)):.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF static optimization / AWFO optimization")
    parser.add_argument("--market", type=str, choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--trials", type=int, default=1500)
    parser.add_argument("--opt_seeds", type=str, default="13,37,73", help="Comma-separated Optuna seeds")
    parser.add_argument("--opt_min_trials_per_seed", type=int, default=40)
    parser.add_argument("--opt_jobs", type=int, default=1)
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")

    parser.add_argument("--awfo", action="store_true", help="Enable AWFO-style multi-year robust optimization")
    parser.add_argument("--awfo_start_year", type=int, default=None, help="AWFO anchor start year. Default=data min year")
    parser.add_argument("--awfo_end_year", type=int, default=None, help="AWFO end year. Default=data max year")
    parser.add_argument("--awfo_train_years", type=int, default=3, help="Minimum train years before first OOS year")

    args = parser.parse_args()
    s_date = datetime.strptime(args.start, "%Y-%m-%d")
    e_date = datetime.strptime(args.end, "%Y-%m-%d")
    etf_df, index_df = asyncio.run(load_all_data(s_date, e_date))
    if etf_df is None or etf_df.is_empty() or index_df is None or index_df.is_empty():
        print("Failed to load ETF/index data.")
        return
    run_static_optimization(etf_df, index_df, args)


if __name__ == "__main__":
    main()
