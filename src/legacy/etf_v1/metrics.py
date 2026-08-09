import numpy as np
import pandas as pd
from typing import Tuple

def calc_profit_factor_from_pnl(pnl_series: pd.Series) -> float:
    """Calculate Profit Factor from a pre-computed net PNL series (fee-deducted)."""
    if pnl_series.empty:
        return 1.0

    gross_profit: float = float(pnl_series[pnl_series > 0].sum())
    gross_loss: float = abs(float(pnl_series[pnl_series < 0].sum()))

    if gross_loss == 0.0:
        return 5.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss

def calc_mdd_from_equity(equity_curve: np.ndarray) -> float:
    """Calculate Maximum Drawdown from an aggregated equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    running_max[running_max == 0] = 1e-9
    drawdown = (equity_curve - running_max) / running_max * 100.0
    return float(abs(np.min(np.nan_to_num(drawdown, nan=0.0))))
