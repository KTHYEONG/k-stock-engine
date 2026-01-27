
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
                 mode: str = 'UNIFIED', market_type: str = 'stock_spot'):
        self.mode = mode
        self.market_type = market_type
        self.search_space_config = GET_SEARCH_SPACE(mode=mode, market_type=market_type)
        
        # [MODIFIED] User 요청에 따라 기본적으로 2023년 모델 사용
        target_model_year = 2023
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
        # 현재 optimization_config.py의 키와 backtester.py의 인자가 거의 일치하도록 설계됨.
        # 소문자로 변환하여 전달 (Config는 대문자 키 사용)
        final_params = {k.lower(): v for k, v in params.items()}

        try:
            # 2. 백테스트 실행
            metrics = self.backtester.run_backtest(fee=0.002, **final_params)
            
            if not metrics:
                return -100.0
            
            # 3. 스코어 계산 (Stability & Robustness Focused)
            cagr = float(metrics["CAGR"].replace("%", ""))
            mdd = abs(float(metrics["MDD"].replace("%", "")))
            sharpe = float(metrics["Sharpe Ratio"])
            win_rate = float(metrics["Win Rate"].replace("%", ""))
            total_ret = float(metrics["Total Return"].replace("%", ""))
            total_trades = float(metrics.get("Total Trades", "0").replace("회", ""))
            turnover = float(metrics["Avg Turnover"].replace("%", ""))
            
            # [Core Objective]: Stability-Adjusted Return
            # 단순히 많이 버는 것보다 '안정적으로' 버는 것에 집중
            
            # 1. Base Score: Sharpe Ratio를 기반으로 함
            # Sharpe가 높을수록 안정적 우상향을 의미함
            score = sharpe * 100.0
            
            # 2. Add Profitability context (Logged to prevent outlier dominance)
            if total_ret > 0:
                score += np.log1p(total_ret) * 10.0
            else:
                score += total_ret # 음수면 그대로 감점
            
            # 3. Hard Penalties for Fragility (부러지기 쉬운 전략 제거)
            # MDD 20% 초과는 실전에서 견디기 힘듦
            if mdd > 20.0:
                score -= (mdd - 20.0) * 5.0
            
            # 4. Win Rate Floor
            # 승률 25% 미만은 사실상 운에 맡기는 매매
            if win_rate < 25.0:
                score -= (25.0 - win_rate) * 10.0
                
            # 5. Trading Frequency Targeting (Sweet Spot: 150 ~ 250 trades/year)
            # 너무 적으면(100회 미만) 통계적 유의성 부족 및 기회 손실 -> 강한 감점
            if total_trades < 100:
                score -= 100.0 + (100 - total_trades) # 부족한 만큼 감점
            
            # 너무 많으면(300회 초과) 과도한 수수료 및 슬리피지 우려 -> 완만한 감점
            elif total_trades > 300:
                 score -= (total_trades - 300) * 0.2

            # Result Reporting
            trial.set_user_attr("cagr", cagr)
            trial.set_user_attr("mdd", mdd)
            trial.set_user_attr("sharpe", sharpe)
            trial.set_user_attr("turnover", turnover)
            trial.set_user_attr("win_rate", win_rate)
            trial.set_user_attr("trades", total_trades)
            
            return score
            
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
        print("="*50 + "\n")
        
        # Final Backtest with Best Params
        print("📊 Running final backtest...")
        best_params = {k.lower(): v for k, v in best_trial.params.items()}
        
        self.backtester.run_backtest(save_plot=True, **best_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Strategy Optimizer")
    parser.add_argument("--trials", type=int, default=300, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--mode", type=str, default="UNIFIED", choices=["UNIFIED", "ACTIVE", "SWING", "TREND"], help="Optimization Mode")
    parser.add_argument("--check", action="store_true", help="Check indicator health and exit")
    parser.add_argument("--start", type=str, default="20210101", help="Start Date (YYYYMMDD)")
    parser.add_argument("--end", type=str, default="20241231", help="End Date (YYYYMMDD)")
    
    args = parser.parse_args()
    
    optimizer = YetiRankOptimizer(mode=args.mode, start_date=args.start, end_date=args.end)
    
    if args.check:
        optimizer.check_indicators()
        sys.exit(0)
        
    optimizer.run_optimization(n_trials=args.trials, resume=args.resume, n_jobs=args.n_jobs)
