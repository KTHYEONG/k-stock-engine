"""Legacy ETF strategy implementation (quarantined).

This package is the former ``src.etf``. Modern code must never import it; the
only permitted consumer is the ETF integration parity test. ``ETFOptimizer``
(which depended on the removed Optuna search dependency) is preserved as data
only and is intentionally not re-exported.
"""
from src.legacy.etf_v1.backtester import ETFBacktester
from src.legacy.etf_v1.strategy_engine import ETFStrategyEngine

__all__ = ["ETFBacktester", "ETFStrategyEngine"]
