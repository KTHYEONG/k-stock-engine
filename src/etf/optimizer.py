import optuna
import logging
from tqdm import tqdm
from typing import Dict, Any
from .backtester import ETFBacktester
from .etf_config import ETFConfig
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("etf.optimizer")

class ETFOptimizer:
    def __init__(self, index_df, etf_df, target_market: str = "KOSPI", n_trials: int = 150):
        self.backtester = ETFBacktester(index_df, etf_df)
        self.target_market = target_market
        self.n_trials = n_trials
        self.space = ETFConfig.get_search_space(target_market)
        
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for param_name, spec in self.space.items():
            if isinstance(spec, list):
                params[param_name] = trial.suggest_categorical(param_name, spec)
            else:
                t = spec["type"]
                if t == "int":
                    params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=spec.get("step", 1))
                elif t == "float":
                    params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], step=spec.get("step"))
        return params

    def objective(self, trial: optuna.Trial) -> tuple[float, float]:
        params = self._suggest_params(trial)
        
        try:
            results = self.backtester.run(params, self.target_market)
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()
            
        if not results:
            raise optuna.exceptions.TrialPruned()
            
        res = results[0]
        trades_df = res["trades_df"]
        num_trades = res["total_trades"]
        mdd = res["mdd_pct"]
        
        if num_trades < 5:
            raise optuna.exceptions.TrialPruned()
            
        # Limit excessive MDD early to keep Pareto front clean
        if mdd > 15.0:
            raise optuna.exceptions.TrialPruned()

        # Calculate Base Calmar
        total_return_pct = res["total_return_pct"]
        pf = res["profit_factor"]
        win_rate = res["win_rate"]
        
        calmar = total_return_pct / abs(mdd) if mdd > 0 else total_return_pct

        # Apply Strict Institutional Penalties to Calmar
        if win_rate < 55.0:
            calmar -= (55.0 - win_rate) * 0.1
            
        if pf < 1.3:
            calmar -= (1.3 - pf) * 2.0
            
        if num_trades < 15:
            penalty = (15 - num_trades) / 15.0  # 0 to 1 scale
            calmar = calmar * (1.0 - penalty)
            
        # Logging for user retrieval
        trial.set_user_attr("calmar", calmar)
        trial.set_user_attr("cagr", total_return_pct) # using total return pct as proxy
        trial.set_user_attr("mdd", mdd)
        trial.set_user_attr("pf", pf)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("trades", num_trades)

        # We MAXIMIZE Calmar, MAXIMIZE Total Return
        return calmar, total_return_pct

    def optimize(self, seed: int = 42) -> optuna.Study:
        pop_size = min(300, max(100, self.n_trials // 5)) 
        sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=pop_size)
        # Change direction to maximize both SQN and Return
        study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)        
        logger.info(f"Starting NSGA-II Optimization for {self.target_market} ({self.n_trials} trials)...")
        
        with tqdm(total=self.n_trials, desc=f"Optimizing {self.target_market}", unit="trial") as pbar:
            def callback(study, trial):
                pbar.update(1)
            
            study.optimize(self.objective, n_trials=self.n_trials, n_jobs=8, catch=(Exception,), callbacks=[callback])
        
        return study
