
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
import polars as pl
from pathlib import Path
import sys
import json
import numpy as np

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("run_walk_forward")
logging.getLogger("optuna").setLevel(logging.ERROR)
logging.getLogger("etf").setLevel(logging.WARNING)
logging.getLogger("src.etf.strategy_engine").setLevel(logging.WARNING)

from src.data.etf_manager import ETFManager
from src.etf.optimizer import ETFOptimizer
from src.etf.backtester import ETFBacktester
from src.etf.monte_carlo import ETFMonteCarloSimulator

async def load_all_data(start_date: datetime, end_date: datetime):
    """
    전체 기간 데이터를 한 번에 로드
    """
    manager = ETFManager()
    try:
        etf_df = pl.scan_parquet(str(manager.etf_store.base_path / "**/*.parquet")).collect()
        index_df = pl.scan_parquet(str(manager.index_store.base_path / "**/*.parquet")).collect()
        
        etf_df = etf_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        index_df = index_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        
        logger.info(f"📚 전체 데이터 로드 완료 ({start_date.date()} ~ {end_date.date()})")
        return etf_df, index_df
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        return None, None

def run_walk_forward(etf_df: pl.DataFrame, index_df: pl.DataFrame, args):
    """
    Rolling Walk-Forward Analysis 수행
    """
    # 기간 설정 (Rolling 1 Year)
    # Train: 2 Years -> Test: 1 Year
    
    # 시작 연도부터 종료 연도까지 루프
    start_year = datetime.strptime(args.start, "%Y-%m-%d").year
    end_year = datetime.strptime(args.end, "%Y-%m-%d").year
    
    # Rolling Steps 생성
    steps = []
    current_year = start_year
    
    # 예: Start=2017, End=2024
    # Step 1: Train 2017-2019 (3년) -> Validate 2020 (1년)
    # Step 2: Train 2018-2020 (3년) -> Validate 2021 (1년)
    # ...
    
    while current_year + 3 <= end_year:
        train_start = datetime(current_year, 1, 1)
        train_end = datetime(current_year + 2, 12, 31)
        val_start = datetime(current_year + 3, 1, 1)
        val_end = datetime(current_year + 3, 12, 31)
        
        steps.append({
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "name": f"{current_year+3}" # 검증 연도
        })
        current_year += 1
        
    logger.info(f"🚀 Walk-Forward Steps: {len(steps)}개 구간")
    for s in steps:
        logger.info(f"   - {s['name']}: Train({s['train'][0].year}~{s['train'][1].year}) -> Val({s['val'][0].year})")
        
    all_trade_returns = []
    all_daily_returns = []
    
    # 결과 저장용
    history_records = []
    
    # 타겟 설정
    market = args.market
    # 레버리지는 이제 HYBRID 모드이므로 고정값 사용
    leverage = "HYBRID"
    
    for step in steps:
        train_s, train_e = step['train']
        val_s, val_e = step['val']
        step_name = step['name']
        
        print(f"\n{'='*60}")
        print(f"⌛ Step {step_name}: Training ({train_s.date()} ~ {train_e.date()})")
        print(f"{'='*60}")
        
        # 1. Slice Train Data
        train_etf = etf_df.filter((pl.col("date") >= train_s) & (pl.col("date") <= train_e))
        train_index = index_df.filter((pl.col("date") >= train_s) & (pl.col("date") <= train_e))
        
        # 2. Optimize
        # 최적화 시간을 아끼기 위해 trials 조절 가능
        optimizer = ETFOptimizer(train_index, train_etf, target_market=market, target_leverage="HYBRID")
        best_params = optimizer.run_optimization(n_trials=args.trials)
        
        # 파라미터 저장 (기록용)
        param_path = PROJECT_ROOT / "results" / "walk_forward" / f"params_{market}_HYBRID_{step_name}.json"
        param_path.parent.mkdir(parents=True, exist_ok=True)
        with open(param_path, "w") as f:
            json.dump(best_params, f, indent=4)
            
        print(f"   ✅ Optimized Params Saved: {param_path.name}")
        
        # 3. Validation (OOS)
        print(f"👉 Validating ({val_s.date()} ~ {val_e.date()})")
        val_etf = etf_df.filter((pl.col("date") >= val_s) & (pl.col("date") <= val_e))
        val_index = index_df.filter((pl.col("date") >= val_s) & (pl.col("date") <= val_e))
        
        backtester = ETFBacktester(val_index, val_etf)
        val_results = backtester.run(best_params, target_market=market)
        
        # 결과 처리
        # val_results는 list of dict. target_market 지정했으므로 하나만 나옴.
        target_key = f"{market}_HYBRID"
        res = next((r for r in val_results if r['market'] == target_key), None)
        
        if res:
            ret = res['total_return'] * 100
            mdd = res['mdd'] * 100
            trades = res['trades']
            win_rate = res['win_rate']
            
            print(f"   📊 {step_name} Result: Return {ret:.2f}% | MDD {mdd:.2f}% | Trades {trades}")
            
            # 누적용 데이터 수집
            all_trade_returns.extend(res['trade_list'])
            all_daily_returns.extend(res['daily_returns'])
            
            history_records.append({
                'Year': step_name,
                'Return': ret,
                'MDD': mdd,
                'Trades': trades,
                'WinRate': win_rate
            })
        else:
            logger.warning(f"   ⚠️ No validation results for {step_name}")
            
    # --- Final Analysis ---
    print("\n" + "="*70)
    print("🏆 WALK-FORWARD ANALYSIS FINAL REPORT")
    print("="*70)
    
    # 1. Period Performance
    df_res = pl.DataFrame(history_records)
    print(df_res)
    
    # 2. Cumulative Stats
    if all_daily_returns:
        # Calculate Final CAGR, MDD from concatenated daily returns
        cum_equity = np.cumprod(1 + np.array(all_daily_returns))
        final_equity = cum_equity[-1]
        
        # Total Years
        total_days = len(all_daily_returns)
        total_years = total_days / 252
        
        cagr = (final_equity ** (1/total_years)) - 1 if total_years > 0 else 0.0
        
        # MDD
        running_max = np.maximum.accumulate(cum_equity)
        dd = (cum_equity - running_max) / running_max
        max_dd = np.min(dd)
        
        print("-" * 70)
        print(f" TOTAL PERIOD: {steps[0]['val'][0].date()} ~ {steps[-1]['val'][1].date()}")
        print(f" Total Return : {(final_equity - 1)*100:.2f}%")
        print(f" CAGR         : {cagr*100:.2f}%")
        print(f" Max Drawdown : {max_dd*100:.2f}%")
        print("-" * 70)
        
    # 3. Monte Carlo
    print("\n🎲 MONTE CARLO SIMULATION (95% Confidence)")
    print("-" * 70)
    
    if len(all_trade_returns) > 10:
        mc = ETFMonteCarloSimulator(all_trade_returns)
        mc_res = mc.run(n_simulations=5000)
        
        if mc_res['is_valid']:
            print(f" Probability of Profit : {mc_res['prob_profit']:.1f}%")
            print(f" Expected Return       : {mc_res['mean_return_pct']:.2f}%")
            print(f" VaR 95% MDD           : {mc_res['worst_case_mdd']:.2f}% (Worst 5% Case)")
            print(f" Return Range (95%)    : {mc_res['lower_bound_95']:.2f}% ~ {mc_res['upper_bound_95']:.2f}%")
        else:
            print("⚠️ Simulation failed (not enough data)")
    else:
        print("⚠️ Not enough trades for Monte Carlo Simulation.")
        
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Run ETF Walk-Forward Analysis")
    parser.add_argument("--market", type=str, choices=["KOSPI", "KOSDAQ"], default="KOSPI")
    parser.add_argument("--trials", type=int, default=1500, help="Optimization trials per step")
    parser.add_argument("--start", type=str, default="2017-01-01", help="Data Load Start Date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="Data Load End Date")
    
    args = parser.parse_args()
    
    # Dates
    try:
        s_date = datetime.strptime(args.start, "%Y-%m-%d")
        e_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format")
        return

    # Load All Data
    etf_df, index_df = asyncio.run(load_all_data(s_date, e_date))
    
    if etf_df is None or etf_df.is_empty():
        return
        
    # Run WFA
    run_walk_forward(etf_df, index_df, args)

if __name__ == "__main__":
    main()
