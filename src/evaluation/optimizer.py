
import optuna
import logging
from pathlib import Path
import sys
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.backtester import YetiRankBacktester
from src.evaluation.optimization_config import GET_SEARCH_SPACE
from src.utils.logger import setup_logger

# 로그 설정
logger = setup_logger("evaluation.optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("training.data_loader").setLevel(logging.WARNING)

class YetiRankOptimizer:
    """
    Optuna를 이용한 전략 하이퍼파라미터 최적화 도구.
    Dynamic Search Space를 지원하여 다양한 전략 모드(ACTIVE, SWING, TREND)를 탐색.
    """
    
    def __init__(self, start_date: str = "20240101", end_date: str = "20241231", 
                 mode: str = 'UNIFIED', market_type: str = 'stock_spot',
                 model_year: Optional[int] = None, sizing_mode: str = 'EQUAL'):
        self.mode = mode
        self.market_type = market_type
        self.sizing_mode = sizing_mode
        self.search_space_config = GET_SEARCH_SPACE(mode=mode, market_type=market_type)
        
        # [MODIFIED] User 요청에 따라 지정된 모델 연도 사용
        target_model_year = model_year
        self.backtester = YetiRankBacktester(start_date=start_date, end_date=end_date, model_year=target_model_year)
        
        # 최적화 시작 전 모델 예측값 미리 캐싱
        self.backtester.generate_predictions()
        
    def check_indicators(self):
        """계산된 지표들의 건강 상태(결측치, 범위 등)를 점검"""
        logger.info("🔍 Checking Indicator Health...")
        df = self.backtester._cached_predictions
        
        if df.is_empty():
            logger.error("❌ No data found in cached predictions.")
            return
            
        indicator_cols = [
            "rsi_14", "mfi_14", "natr_14", "macd_hist", "cci", 
            "cmf", "obv", "stoch_rsi", "bb_position", "supertrend_direction"
        ]
        
        print("\n" + "="*60)
        print(f"{'Indicator':<25} | {'Nulls':<8} | {'Min':<10} | {'Max':<10}")
        print("-" * 60)
        
        for col in indicator_cols:
            if col not in df.columns:
                print(f"{col:<25} | {'MISSING':<8}")
                continue
                
            null_count = df[col].null_count()
            null_pct = (null_count / len(df)) * 100
            
            # Remove nulls for min/max calculation
            valid_data = df[col].drop_nulls()
            if len(valid_data) > 0:
                v_min = valid_data.min()
                v_max = valid_data.max()
                print(f"{col:<25} | {null_pct:>6.1f}% | {v_min:>10.2f} | {v_max:>10.2f}")
            else:
                print(f"{col:<25} | {null_pct:>6.1f}% | {'ALL NULL':<10} | {'ALL NULL':<10}")
                
        print("="*60 + "\n")
        
    def soft_sigmoid(self, x, L, k, x0):
        """Soft-Sigmoid mapping: L=Max, k=Steepness, x0=Midpoint"""
        z = -k * (x - x0)
        z_safe = np.clip(z, -500, 500)
        return L / (1 + np.exp(z_safe))

    def objective(self, trial):
        """Optuna 목적 함수: Config에서 정의된 동적 서치 스페이스 탐색"""
        
        params = {}
        # Dynamic parameter suggestion based on config type
        for param_name, config in self.search_space_config.items():
            # Skip non-parameter entries if any
            if not isinstance(config, dict):
                continue
            
            p_type = config.get('type')
            
            if p_type == 'int':
                # Handle log scale if present
                log = config.get('log', False)
                step = config.get('step', 1) if not log else None 
                params[param_name] = trial.suggest_int(param_name, config['low'], config['high'], step=step, log=log)
                
            elif p_type == 'float':
                log = config.get('log', False)
                step = config.get('step', None) 
                params[param_name] = trial.suggest_float(param_name, config['low'], config['high'], step=step, log=log)
                
            elif p_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, config['choices'])
                
            # Legacy format support (if any key missing 'type')
            elif 'low' in config and 'high' in config:
                 # Guess type based on value
                 if isinstance(config['low'], int):
                     params[param_name] = trial.suggest_int(param_name, config['low'], config['high'], step=config.get('step', 1))
                 else:
                     params[param_name] = trial.suggest_float(param_name, config['low'], config['high'], step=config.get('step'))

        # 매개변수 이름을 Backtester의 인자명으로 매핑 (필요시)
        final_params = {k.lower(): v for k, v in params.items()}
        
        # [DEBUG] Check what represents 'filter_candidates_ratio'
        if 'filter_candidates_ratio' in final_params:
             # Backtester might expect a different name or specific handling
             pass

        try:
            # 2. 백테스트 실행 (Realistic Fee: 0.35%)
            # [DEBUG] Log params for first trial to verify
            if trial.number == 0:
                logger.info(f"🔍 Trial 0 Params Check: {final_params}")
            
            # [FIX] Explicitly extract core params to avoid kwargs shadowing issues
            core_top_k = final_params.pop('top_k', 10)
            core_recal = final_params.pop('rebalance_period', 5)
                
            metrics, _, trade_records = self.backtester.run_backtest(
                top_k=core_top_k, 
                rebalance_period=core_recal,
                fee=0.0035, 
                return_details=True, 
                sizing_mode=self.sizing_mode,
                **final_params
            )
            
            if not metrics:
                return -100.0
            
            # 3. 상세 지표 추출 (Robustness Focused)
            # 기본 지표 파싱
            cagr = float(metrics["CAGR"].replace("%", ""))
            mdd = abs(float(metrics["MDD"].replace("%", "")))
            sharpe = float(metrics["Sharpe Ratio"])
            win_rate = float(metrics.get("Win Rate", "0").replace("%", ""))
            avg_trade_ret = float(metrics.get("Avg Trade Return", "0").replace("%", ""))
            total_trades = len(trade_records)
            turnover = float(metrics.get("Avg Turnover", "0").replace("%", "")) if "Avg Turnover" in metrics else 0.0

            # --- [NEW] Advanced Metrics Calculation ---
            returns = np.array(trade_records) # Trade returns in %
            N = len(returns)
            
            # 1. Profit Factor (PF)
            if N > 0:
                pos_sum = np.sum(returns[returns > 0])
                neg_sum = abs(np.sum(returns[returns < 0]))
                pf = pos_sum / neg_sum if neg_sum > 0 else 3.0
            else: 
                pf = 0.0
                
            # 2. System Quality Number (SQN)
            # SQN = sqrt(N) * (Mean / Std)
            if N > 1:
                r_avg = np.mean(returns)
                r_std = np.std(returns, ddof=1)
                sqn_raw = np.sqrt(N) * (r_avg / r_std) if r_std > 0 else 0
                sqn = np.clip(sqn_raw, 0, 10)
            else: 
                sqn = 0.0
                
            # 3. Consistency (R^2 of Equity Curve)
            # Measures linearity of growth (Steady > Volatile)
            if N > 5:
                equity_curve = np.cumsum(returns)
                x = np.arange(len(equity_curve))
                y = equity_curve
                correlation_matrix = np.corrcoef(x, y)
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2 if not np.isnan(correlation_xy) else 0
            else:
                r_squared = 0.0
            
            # --- Result Reporting ---
            trial.set_user_attr("cagr", cagr)
            trial.set_user_attr("mdd", mdd)
            trial.set_user_attr("sharpe", sharpe)
            trial.set_user_attr("pf", pf)             # Added
            trial.set_user_attr("sqn", sqn)           # Added
            trial.set_user_attr("consistency", r_squared) # Added
            trial.set_user_attr("win_rate", win_rate)
            trial.set_user_attr("avg_trade_ret", avg_trade_ret)
            trial.set_user_attr("trades", total_trades)

            # --- 4. Filtering (Survival Constraints - Relaxed for Phase 1) ---
            # [RELAXED] Avg Trade Return 0.15% 하한선 (수수료 전후라도 일단 생존)
            if avg_trade_ret < 0.15: return -100.0 + avg_trade_ret
            # 데이터 최소 샘플 확인 (1년 기준 15회)
            if total_trades < 15: return -50.0

            # --- 5. Scoring (Phase 2: Growth & Robustness) ---
            # Objective: Find high-alpha strategies without performance ceilings.
            
            # 1. Growth Score (Linear, no ceiling)
            # 10% CAGR = 5 points, 100% CAGR = 50 points
            s_growth = cagr * 0.5
            
            # 2. Adjusted Sharpe Score
            # Sharpe 1.0 = 5 points, 2.0 = 10 points
            s_sharpe = sharpe * 5.0
            
            # 3. Robustness Score (PF & SQN)
            # PF 1.5 = 7.5 points, SQN 3.0 = 6 points
            s_pf = pf * 5.0
            s_sqn = sqn * 2.0
            
            # 4. Consistency Bonus (R-Squared)
            # R^2 0.9 = 9 points
            s_consistency = r_squared * 10.0
            
            # MDD Penalty (Exponentially increases after 15%)
            mdd_penalty = 0
            if mdd > 15.0: 
                mdd_penalty = (mdd - 15.0) * 2.0
            if mdd > 25.0: 
                mdd_penalty += (mdd - 25.0) * 5.0 
            
            # Final Score Assembly
            score = s_growth + s_sharpe + s_pf + s_sqn + s_consistency
            
            # Penalties
            score -= mdd_penalty
            
            # Turnover Penalty (Stricter: 0.5)
            if turnover > 0.5: 
                score -= (turnover - 0.5) * 30.0 
            
            # Over-trading Penalty (Stricter: 300 trades/year)
            if total_trades > 300: 
                score -= (total_trades - 300) * 0.5
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return -100.0

    def run_optimization(self, n_trials: int = 50, study_name: Optional[str] = None, resume: bool = False, n_jobs: int = 1):
        if study_name is None:
            study_name = f"yetirank_{self.mode.lower()}_opt"
            
        logger.info(f"🚀 Starting Optimization [{self.mode}] | Trials: {n_trials} | Resume: {resume} | Jobs: {n_jobs}")
        
        db_path = PROJECT_ROOT / "results" / f"optimization_{self.mode.lower()}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not resume and db_path.exists():
            try:
                db_path.unlink()
                logger.warning(f"⚠️ Dleted existing DB: {db_path}")
            except: pass

        storage_name = f"sqlite:///{db_path}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize"
        )
        
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        
        logger.info("✅ Optimization Complete!")
        
        # Result Summary
        best_trial = study.best_trial
        print("\n" + "="*50)
        print(f"🏆 BEST STRATEGY PARAMETERS ({self.mode})")
        print("="*50)
        for key, value in best_trial.params.items():
            print(f"- {key:<25}: {value}")
        print("-" * 50)
        
        attrs = best_trial.user_attrs
        print(f"- Score              : {best_trial.value:.4f}")
        print(f"- CAGR               : {attrs.get('cagr', 0.0):.2f}%")
        print(f"- MDD                : -{attrs.get('mdd', 0.0):.2f}%")
        print(f"- Sharpe             : {attrs.get('sharpe', 0.0):.4f}")
        print(f"- Calmar             : {attrs.get('calmar', 0.0):.4f}")
        print(f"- Win Rate (Trade)   : {attrs.get('win_rate', 0.0):.2f}%")
        print(f"- Avg Trade Return   : {attrs.get('avg_trade_ret', 0.0):.2f}%")
        print("="*50 + "\n")
        
        # Final Backtest with Best Params
        print("📊 Running final backtest...")
        best_params = {k.lower(): v for k, v in best_trial.params.items()}
        
        self.backtester.run_backtest(save_plot=True, sizing_mode=self.sizing_mode, **best_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Strategy Optimizer")
    parser.add_argument("--trials", type=int, default=300, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["UNIFIED", "ACTIVE", "SWING", "TREND"], help="Optimization Mode")
    parser.add_argument("--check", action="store_true", help="Check indicator health and exit")
    parser.add_argument("--start", type=str, default="20240101", help="Start Date (YYYYMMDD) - Optimization Period Start")
    parser.add_argument("--end", type=str, default="20241231", help="End Date (YYYYMMDD) - Optimization Period End")
    parser.add_argument("--model_year", type=int, default="2024", help="Model Year (Default: None=Auto)")
    parser.add_argument("--sizing", type=str, default="CONFIDENCE", choices=["EQUAL", "CONFIDENCE", "RISK", "HYBRID"], help="Position Sizing Mode")
    
    args = parser.parse_args()
    
    optimizer = YetiRankOptimizer(mode=args.mode, start_date=args.start, end_date=args.end, 
                                 model_year=args.model_year, sizing_mode=args.sizing)
    
    if args.check:
        optimizer.check_indicators()
        sys.exit(0)
        
    optimizer.run_optimization(n_trials=args.trials, resume=args.resume, n_jobs=args.n_jobs)
