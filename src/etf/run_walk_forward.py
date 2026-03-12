import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_walk_forward")
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


def _slice(df: pl.DataFrame, s: datetime, e: datetime) -> pl.DataFrame:
    return df.filter((pl.col("date") >= s) & (pl.col("date") <= e))


def _build_yearly_folds(start_year: int, end_year: int, train_years: int, anchored: bool) -> List[Dict[str, datetime]]:
    folds: List[Dict[str, datetime]] = []
    if end_year - start_year < train_years:
        return folds
    for eval_year in range(start_year + train_years, end_year + 1):
        train_start_year = start_year if anchored else (eval_year - train_years)
        fold = {
            "train_start": datetime(train_start_year, 1, 1),
            "train_end": datetime(eval_year - 1, 12, 31),
            "val_start": datetime(eval_year, 1, 1),
            "val_end": datetime(eval_year, 12, 31),
            "eval_year": datetime(eval_year, 1, 1),
        }
        folds.append(fold)
    return folds


def _equity_summary(daily_returns: List[float]) -> Dict[str, float]:
    arr = np.asarray(daily_returns, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0, "mdd_pct": 0.0}
    equity = np.cumprod(1.0 + arr)
    total = float(equity[-1] - 1.0)
    years = max(1.0 / 252.0, arr.size / 252.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    ann_mean = float(np.mean(arr) * 252.0)
    ann_vol = float(np.std(arr) * np.sqrt(252.0))
    sharpe = ann_mean / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = float(np.min((equity - peak) / np.maximum(peak, 1e-12)))
    return {
        "total_return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": float(sharpe),
        "mdd_pct": mdd * 100.0,
    }


def run_walk_forward(etf_df: pl.DataFrame, index_df: pl.DataFrame, args) -> None:
    market = args.market.upper()
    s_year = datetime.strptime(args.start, "%Y-%m-%d").year
    e_year = datetime.strptime(args.end, "%Y-%m-%d").year
    folds = _build_yearly_folds(
        start_year=s_year,
        end_year=e_year,
        train_years=int(args.train_years),
        anchored=bool(args.awfo_anchor),
    )
    if not folds:
        print("No valid folds generated. Check start/end/train_years.")
        return

    print("\n" + "=" * 80)
    print(f"ETF TRUE WFO START | Market={market} | Anchored={bool(args.awfo_anchor)}")
    print(f"Period: {args.start} ~ {args.end} | TrainYears={args.train_years}")
    print("=" * 80)

    all_daily: List[float] = []
    all_trades: List[float] = []
    rows: List[Dict[str, object]] = []

    for idx, fold in enumerate(folds, start=1):
        tr_s = fold["train_start"]
        tr_e = fold["train_end"]
        va_s = fold["val_start"]
        va_e = fold["val_end"]
        eval_year = fold["eval_year"].year

        print(f"\n[{idx}/{len(folds)}] Train {tr_s.date()}~{tr_e.date()} -> OOS {va_s.date()}~{va_e.date()}")

        tr_idx = _slice(index_df, tr_s, tr_e)
        tr_etf = _slice(etf_df, tr_s, tr_e)
        va_idx = _slice(index_df, va_s, va_e)
        va_etf = _slice(etf_df, va_s, va_e)
        if tr_idx.is_empty() or tr_etf.is_empty() or va_idx.is_empty() or va_etf.is_empty():
            print("Skipped: missing data in this fold.")
            continue

        optimizer = ETFOptimizer(
            index_df=tr_idx,
            etf_df=tr_etf,
            target_market=market,
            target_leverage="HYBRID",
            awfo=False,
        )
        best_params = optimizer.run_optimization(
            n_trials=args.trials,
            seeds=_parse_seed_list(args.opt_seeds),
            min_trials_per_seed=args.opt_min_trials_per_seed,
            n_jobs=args.opt_jobs,
        )

        fold_out = PROJECT_ROOT / "results" / "etf" / "walk_forward"
        fold_out.mkdir(parents=True, exist_ok=True)
        with open(fold_out / f"params_{market.lower()}_{eval_year}.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        if getattr(optimizer, "last_optimization_meta", None):
            with open(fold_out / f"opt_meta_{market.lower()}_{eval_year}.json", "w", encoding="utf-8") as f:
                json.dump(optimizer.last_optimization_meta, f, indent=2, ensure_ascii=False)

        val_bt = ETFBacktester(va_idx, va_etf)
        val_res_list = val_bt.run(best_params, target_market=market)
        target_key = f"{market}_HYBRID"
        res = next((r for r in val_res_list if r.get("market") == target_key), None)
        if not res:
            print("No OOS result for this fold.")
            continue

        all_daily.extend(res.get("daily_returns", []))
        all_trades.extend(res.get("trade_list", []))
        rows.append(
            {
                "Split": idx,
                "Train Period": f"{tr_s.date()} ~ {tr_e.date()}",
                "Period": f"{va_s.date()} ~ {va_e.date()}",
                "Return (%)": round(float(res.get("total_return", 0.0)) * 100.0, 2),
                "MDD (%)": round(float(res.get("mdd", 0.0)) * 100.0, 2),
                "Trades": int(res.get("trades", 0)),
                "Win Rate (%)": round(float(res.get("win_rate", 0.0)), 2),
            }
        )
        print(
            f"Result | Return {float(res.get('total_return', 0.0))*100.0:.2f}% | "
            f"MDD {float(res.get('mdd', 0.0))*100.0:.2f}% | Trades {int(res.get('trades', 0))}"
        )

    if not rows:
        print("No walk-forward results produced.")
        return

    df = pl.DataFrame(rows)
    print("\n" + "=" * 80)
    print("ETF TRUE WFO RESULT")
    print("=" * 80)
    print(df.to_pandas().to_string(index=False))

    ret_arr = np.asarray(df["Return (%)"].to_list(), dtype=float)
    eq = _equity_summary(all_daily)
    print("\n" + "=" * 80)
    print("AGGREGATED SUMMARY")
    print("=" * 80)
    print(f"Average Return per Split: {float(np.mean(ret_arr)):.2f}%")
    print(f"Std Return per Split    : {float(np.std(ret_arr)):.2f}%")
    print(f"Worst Split Return      : {float(np.min(ret_arr)):.2f}%")
    print(f"Consistency (Positive)  : {float(np.mean(ret_arr > 0) * 100.0):.0f}%")
    print(f"Combined OOS Return     : {eq['total_return_pct']:.2f}%")
    print(f"Combined OOS CAGR       : {eq['cagr_pct']:.2f}%")
    print(f"Combined OOS Sharpe     : {eq['sharpe']:.4f}")
    print(f"Combined OOS MDD        : {eq['mdd_pct']:.2f}%")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF true walk-forward verifier")
    parser.add_argument("--market", type=str, choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--trials", type=int, default=200, help="Optimization trials per fold")
    parser.add_argument("--opt_seeds", type=str, default="13,37,73")
    parser.add_argument("--opt_min_trials_per_seed", type=int, default=40)
    parser.add_argument("--opt_jobs", type=int, default=1)
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--train_years", type=int, default=3, help="Train years per fold")
    parser.add_argument(
        "--awfo_anchor",
        action="store_true",
        help="Use anchored windows (start fixed). If omitted, uses rolling train windows.",
    )
    args = parser.parse_args()

    s_date = datetime.strptime(args.start, "%Y-%m-%d")
    e_date = datetime.strptime(args.end, "%Y-%m-%d")
    etf_df, index_df = asyncio.run(load_all_data(s_date, e_date))
    if etf_df is None or etf_df.is_empty() or index_df is None or index_df.is_empty():
        print("Failed to load ETF/index data.")
        return
    run_walk_forward(etf_df, index_df, args)


if __name__ == "__main__":
    main()
