
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ETFMonteCarloSimulator:
    def __init__(self, trades: List[float]):
        """
        :param trades: List of percentage returns from trades (e.g., [0.05, -0.02, 0.10] for 5%, -2%, 10%)
        """
        self.trades = trades
    
    def run(self, n_simulations: int = 10000, initial_balance: float = 1.0) -> Dict[str, Any]:
        """
        Run Monte Carlo Simulation using Bootstrap Sampling.
        
        Args:
            n_simulations: Number of simulations to run
            initial_balance: Starting equity (default 1.0 for relative calculation)
            
        Returns:
            Dictionary containing statistics
        """
        if len(self.trades) < 5:
            return {
                'prob_profit': 0.0,
                'mean_return_pct': 0.0,
                'median_return_pct': 0.0,
                'worst_case_mdd': 0.0,
                'lower_bound_95': 0.0,
                'upper_bound_95': 0.0,
                'is_valid': False
            }
            
        simulation_final_balances = []
        simulation_mdds = []
        
        n_trades = len(self.trades)
        trades_arr = np.array(self.trades)
        
        for _ in range(n_simulations):
            # Bootstrap sampling: Randomly pick N trades with replacement
            shuffled_rets = np.random.choice(trades_arr, size=n_trades, replace=True)
            
            # Calculate Equity Curve
            # (1 + r1) * (1 + r2) * ...
            # Using cumprod for accurate compounding
            cumulative_growth = np.cumprod(1 + shuffled_rets)
            equity_curve = initial_balance * cumulative_growth
            equity_curve = np.insert(equity_curve, 0, initial_balance)
            
            # Final Balance
            final_bal = equity_curve[-1]
            simulation_final_balances.append(final_bal)
            
            # MDD Calculation
            running_max = np.maximum.accumulate(equity_curve)
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown = (equity_curve - running_max) / running_max * 100
                mdd = np.min(drawdown) # usually negative
                if np.isnan(mdd): mdd = 0.0
            simulation_mdds.append(abs(mdd)) # Store as positive value
            
        simulation_final_balances = np.array(simulation_final_balances)
        simulation_mdds = np.array(simulation_mdds)
        
        # Calculate Total Return % relative to initial balance
        sim_returns_pct = (simulation_final_balances - initial_balance) / initial_balance * 100
        
        # Statistics
        prob_profit = np.mean(sim_returns_pct > 0) * 100
        mean_return = np.mean(sim_returns_pct)
        median_return = np.median(sim_returns_pct)
        
        # 95% Confidence Interval for Returns (2.5th to 97.5th percentile)
        lower_bound = np.percentile(sim_returns_pct, 2.5)
        upper_bound = np.percentile(sim_returns_pct, 97.5)
        
        # Risk: Worst 5% Case MDD (We take 95th percentile of MDD since MDD is positive magnitude)
        # If we stored MDD as negative, we would take 5th percentile. 
        # Here we stored abs(mdd), so we want the "large" mdd values which are at the top.
        worst_case_mdd = np.percentile(simulation_mdds, 95)
        
        return {
            'prob_profit': prob_profit,
            'mean_return_pct': mean_return,
            'median_return_pct': median_return,
            'worst_case_mdd': worst_case_mdd,
            'lower_bound_95': lower_bound,
            'upper_bound_95': upper_bound,
            'is_valid': True
        }
