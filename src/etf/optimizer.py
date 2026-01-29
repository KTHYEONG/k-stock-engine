
import numpy as np
import optuna
from typing import Dict, Any
import polars as pl
from .backtester import ETFBacktester
from .etf_config import ETFConfig
import logging

logger = logging.getLogger("etf.optimizer")

class ETFOptimizer:
    """
    ETF Strategy Optimizer (Coin Trader Parity)
    """
    
    def __init__(self, index_df: pl.DataFrame, etf_df: pl.DataFrame, target_market: str = "KOSPI", target_leverage: str = "2X"):
        self.target_market = target_market
        self.target_leverage = target_leverage
        self.backtester = ETFBacktester(index_df, etf_df)
        self.config = ETFConfig.get_search_space(self.target_market)
        
    def objective(self, trial):
        # Generate Params from Config Search Space
        params = {}
        
        # Iterate over SEARCH_SPACE
        for key, conf in self.config.items():
            if isinstance(conf, list):
                # Categorical
                params[key] = trial.suggest_categorical(key, conf)
            elif isinstance(conf, dict) and 'type' in conf:
                # Numerical
                if conf['type'] == 'int':
                    step = conf.get('step', 1)
                    params[key] = trial.suggest_int(key, conf['low'], conf['high'], step=step)
                elif conf['type'] == 'float':
                    step = conf.get('step', None)
                    params[key] = trial.suggest_float(key, conf['low'], conf['high'], step=step)
                    
        # Apply Constraints (Logical consistency)
        # Prevent Entry Period > Trend Period (Generally trend should be longer)
        if 'MA_PERIOD' in params and 'ENTRY_PERIOD' in params:
            if params['MA_PERIOD'] <= params['ENTRY_PERIOD']:
                 # Force MA > Entry to detect macroscopic trend
                 # But Optuna selects random, so we can't force easily.
                 # Just penalty? Or simple swap?
                 pass
        
        # [Constraint] Trailing Stop Logic Validity
        # TS Trigger must be < Take Profit, otherwise TS never activates.
        if 'TS_TRIGGER_ATR' in params and 'TAKE_PROFIT_ATR' in params:
            if params['TS_TRIGGER_ATR'] >= params['TAKE_PROFIT_ATR']:
                return -150.0 # Invalid Logic Penalty

        # Run Backtest (Returns list of results for all markets)
        all_results = self.backtester.run(params, target_market=self.target_market)
        
        # Filter Result for Target Market (Optimizing for Specific Leverage)
        target_key = f"{self.target_market}_{self.target_leverage}"
        target_res = next((r for r in all_results if r['market'] == target_key), None)
        
        if not target_res:
             return -999.0 # Fail
            
        return self._calculate_score(target_res, trial)

    def _calculate_score(self, res: Dict[str, Any], trial: optuna.Trial) -> float:
        """
        Calculate Optimization Score based on Market Type
        """
        # Common Metrics
        cagr = res.get('cagr', 0.0) # Not used directly in score but good for ref
        mdd = abs(res.get('mdd', 0.0))
        trades = res.get('trades', 0)
        win_rate = res.get('win_rate', 0.0)
        pf = res.get('profit_factor', 0.0)
        tot_ret = res.get('total_return', 0.0)
        
        # Reporting
        trial.set_user_attr("CAGR", cagr)
        trial.set_user_attr("MDD", -mdd)
        trial.set_user_attr("Trades", trades)
        trial.set_user_attr("Win Rate", win_rate)
        trial.set_user_attr("Profit Factor", pf)
        
        # 1. Hard Constraints (Global)
        if trades == 0: return -100.0
        
        # 2. Market Specific Scoring
        if self.target_market == "KOSDAQ":
            return self._calculate_kosdaq_score(tot_ret, mdd, win_rate, pf, trades)
        else:
            return self._calculate_kospi_score(tot_ret, mdd, win_rate, pf, trades)

    def _calculate_kospi_score(self, tot_ret, mdd, win_rate, pf, trades) -> float:
        """
        KOSPI Scoring Logic (Trend Following Friendly)
        - Focus: Total Return
        - Constraint: MDD < 35%
        """
        # A. Constraints
        if mdd > 0.35: return -100.0 # MDD Limit
        if trades < 30: return -100.0 + trades # Statistical Significance
        if win_rate < 20.0: return -50.0 # Psychology Limit
        if pf < 0.8: return -50.0 
        
        # B. Score Calculation: Return * MDD Penalty
        # Score = Total_Return * (1 - MDD * 1.5)
        risk_penalty = 1.0 - (mdd * 1.5) 
        if risk_penalty < 0.1: risk_penalty = 0.1
        
        win_penalty = 1.0
        if win_rate < 40.0: win_penalty = 0.8 # Minor penalty
            
        score = tot_ret * risk_penalty * win_penalty
        return score

    def _calculate_kosdaq_score(self, tot_ret, mdd, win_rate, pf, trades) -> float:
        """
        KOSDAQ Scoring Logic (High Volatility / Mean Reversion Friendly)
        - Focus: Stability & Efficiency
        - Constraint: MDD < 25% (Stricter)
        - WinRate Importance: High (Chop market survival)
        """
        # A. Constraints (Stricter)
        if mdd > 0.25: return -100.0 # MDD Limit (25% is max pain for KOSDAQ)
        if trades < 40: return -100.0 + trades # Need more samples for KOSDAQ
        if pf < 1.0: return -50.0 # Must be profitable roughly
        
        # B. Score Calculation
        # 1. MDD Penalty (Exponential)
        # MDD 10% -> 0.9, MDD 20% -> 0.8, MDD 25% -> Penalty
        risk_score = 1.0 - (mdd * 2.5) # Stronger penalty than KOSPI
        if risk_score < 0: risk_score = 0
        
        # 2. WinRate Bonus/Penalty
        # KOSDAQ needs high win rate to survive chop
        # Target: > 50%
        wr_score = 1.0
        if win_rate < 45.0:
            wr_score = 0.5 # Severe Penalty
        elif win_rate > 55.0:
            wr_score = 1.1 # Bonus
            
        # 3. Profit Factor Stability
        pf_score = 1.0
        if pf > 1.5: pf_score = 1.1
        
        # Final Score
        # Return is still important, but risk kills it faster.
        score = tot_ret * risk_score * wr_score * pf_score
        
        return score

    def run_optimization(self, n_trials=100) -> Dict[str, Any]:
        """
        Run Optuna Optimization
        """
        # Optuna 기본 로그 완전 차단 (Trial별 파라미터 출력 방지)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        logger.info(f"🚀 ETF 최적화 시작 [{self.target_market} {self.target_leverage}] - 총 {n_trials}회 시도...")
        
        study = optuna.create_study(direction="maximize")
        self.study = study
        
        def logging_callback(study, frozen_trial):
            # 50회마다 또는 베스트 갱신 시 간단히 출력
            if frozen_trial.number % 50 == 0:
                logger.info(f"🔄 Progress: {frozen_trial.number}/{n_trials} | Best Score: {study.best_value:.4f}")

        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            show_progress_bar=True,
            callbacks=[logging_callback]
        )
        
        logger.info("\n🏆 최적 파라미터 검색 완료")
        logger.info(f"📈 최종 Best Score: {study.best_value:.4f}")
        
        return study.best_params
