
import asyncio
import logging
import argparse
from datetime import datetime
import polars as pl
from pathlib import Path
import sys
import json

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_backtest")

from legacy.etf_v1.etf_manager import ETFManager
from legacy.etf_v1.backtester import ETFBacktester

async def load_data(start_date: datetime, end_date: datetime):
    """
    백테스트에 필요한 데이터를 로드합니다. (지정된 기간)
    """
    manager = ETFManager()
    
    # Feature Store 로드
    try:
        etf_df = pl.scan_parquet(str(manager.etf_store.base_path / "**/*.parquet")).collect()
        index_df = pl.scan_parquet(str(manager.index_store.base_path / "**/*.parquet")).collect()
        
        # 날짜 필터링
        etf_df = etf_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        index_df = index_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        
        logger.info(f"Loaded Data ({start_date.date()} ~ {end_date.date()}): ETF {len(etf_df)} rows, Index {len(index_df)} rows")
        return etf_df, index_df
        
    except Exception as e:
        logger.error(f"Data load failed: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Run ETF Strategy Backtest")
    parser.add_argument("--market", type=str, choices=["KOSPI", "KOSDAQ"], default="KOSPI", help="Target Market")
    parser.add_argument("--params", type=str, help="Path to JSON file containing parameters (optional)")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYY-MM-DD)")
    parser.add_argument("--leverage", type=str, choices=["1X", "2X", "ALL"], default="ALL", help="Leverage to test")
    
    args = parser.parse_args()
    
    # Parse Dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return

    # 1. Load Data
    etf_df, index_df = asyncio.run(load_data(start_date, end_date))
    
    if etf_df is None or etf_df.is_empty():
        logger.error("No data found. Please fetch data first.")
        return

    # 2. Determine leverages to test
    target_leverages = ["1X", "2X"] if args.leverage == "ALL" else [args.leverage]
    final_results = []

    for lev in target_leverages:
        # Load Params for this specific leverage
        if args.params:
            params_path = Path(args.params)
        else:
            results_dir = PROJECT_ROOT / "results"
            params_path = results_dir / f"best_params_{args.market}_{lev}.json"
            
            # Fallback to old naming convention if specific one not found
            if not params_path.exists():
                params_path = results_dir / f"best_params_{args.market}.json"

        if not params_path.exists():
            logger.warning(f"⚠️ Parameters file not found for {lev}: {params_path.name}. Skipping.")
            continue
            
        with open(params_path, "r") as f:
            params = json.load(f)
        
        logger.info(f"🚀 Running Backtest for {args.market}_{lev} using {params_path.name}...")

        # 3. Run Backtest
        backtester = ETFBacktester(index_df, etf_df)
        all_results = backtester.run(params)
        
        # Pick only the matching leverage result
        mkt_key = f"{args.market}_{lev}"
        match = next((r for r in all_results if r['market'] == mkt_key), None)
        if match:
            final_results.append(match)

    # 4. Display Results
    if final_results:
        print("\n" + "="*70)
        print(f" 📊 Backtest Report: {args.market} ({args.start} ~ {args.end})")
        print("="*70)
        print(f" {'Leverage':<12} | {'CAGR':<9} | {'MDD':<9} | {'Trades':<7} | {'WinRate':<8} | {'P.Factor':<8} | {'Equity':<8} ")
        print("-" * 82)
        
        for res in final_results:
            cagr = res['cagr'] * 100
            mdd = res['mdd'] * 100
            trades = res['trades']
            win_rate = res.get('win_rate', 0.0)
            equity = res['equity']
            pf = res.get('profit_factor', 0.0)
            mkt = res['market']
            
            print(f" {mkt:<12} | {cagr:>8.2f}% | {mdd:>8.2f}% | {trades:>7} | {win_rate:>7.2f}% | {pf:>8.2f} | {equity:>8.4f}")
            
        print("="*70 + "\n")
    else:
        logger.error(f"No results generated. Check parameter files and data.")

if __name__ == "__main__":
    main()
