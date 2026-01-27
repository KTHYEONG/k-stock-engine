
import polars as pl
import numpy as np
from typing import Dict, Any

class StockWalkForwardAnalyzer:
    """
    Analyzes the consistency of a strategy by splitting the backtest results
    into multiple time segments (Walk-Forward Analysis substitute).
    """
    def __init__(self, result_df: pl.DataFrame):
        """
        result_df: DataFrame with 'date' and 'net_return' columns
        """
        self.df = result_df.sort("date")
        
    def run(self, n_splits: int = 5) -> pl.DataFrame:
        """
        Split the backtest period into N segments and calculate metrics for each.
        """
        n = len(self.df)
        if n < n_splits * 20: # At least 20 days per segment
            return pl.DataFrame()
            
        segment_size = n // n_splits
        results = []
        
        for i in range(n_splits):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < n_splits - 1 else n
            
            segment_df = self.df.slice(start_idx, end_idx - start_idx)
            if segment_df.is_empty(): continue
            
            # Simple Metrics
            rets = segment_df["net_return"].to_numpy()
            cum_ret = (1 + rets).cumprod()[-1] - 1
            
            # MDD
            equity = (1 + rets).cumprod()
            peak = np.maximum.accumulate(equity)
            mdd = np.min((equity - peak) / peak) * 100
            
            start_date = segment_df["date"].min()
            end_date = segment_df["date"].max()
            
            results.append({
                "Split": i + 1,
                "Period": f"{start_date} ~ {end_date}",
                "Return (%)": round(cum_ret * 100, 2),
                "MDD (%)": round(mdd, 2),
                "Days": len(segment_df)
            })
            
        return pl.DataFrame(results)

