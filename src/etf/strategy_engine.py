import polars as pl
from typing import Dict, Any

class ETFStrategyEngine:
    """
    ETF Strategy Engine (Advanced V4 - IBS & Price Action)
    Strategy tailored for low-volatility Index ETFs with strong mean-reverting properties.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def generate_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates signals fully vectorized using Polars.
        Returns DataFrame with ['signal_trigger', 'ibs', 'macro_ema', 'roc'].
        signal_trigger = 1 (Bull), -1 (Bear), 0 (None)
        """
        macro_ema_period = int(self.params.get("MACRO_EMA_PERIOD", 120))
        fast_ema_period = int(self.params.get("FAST_EMA_PERIOD", 20))
        roc_n = int(self.params.get("ROC_N", 2))
        roc_lower = float(self.params.get("ROC_LOWER", -0.02))
        ibs_entry = float(self.params.get("IBS_ENTRY", 0.15))

        df = df.with_columns([
            pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume"] if c in df.columns
        ])

        # 1. Calc Core Indicators
        # IBS: (Close - Low) / (High - Low + epsilon)
        # ROC: (Close - Close_Shift_N) / Close_Shift_N
        df = df.with_columns([
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8)).alias("ibs"),
            pl.col("close").ewm_mean(span=macro_ema_period, adjust=False).alias("macro_ema"),
            pl.col("close").ewm_mean(span=fast_ema_period, adjust=False).alias("fast_ema"),
            ((pl.col("close") - pl.col("close").shift(roc_n)) / pl.col("close").shift(roc_n)).alias("roc")
        ])

        # 2. Entry Logic (Trend-Aligned Mean Reversion with Dual EMA)
        # Bull: Fast EMA > Macro EMA & Short-term Drop & Intraday Panic (Low IBS)
        bull_alignment = (pl.col("fast_ema") > pl.col("macro_ema")) & (pl.col("close") > pl.col("macro_ema"))
        bull_signal = bull_alignment & (pl.col("roc") < roc_lower) & (pl.col("ibs") < ibs_entry)

        # Bear: Fast EMA < Macro EMA & Short-term Rally & Intraday Euphoria (High IBS) -> Buy Inverse
        bear_alignment = (pl.col("fast_ema") < pl.col("macro_ema")) & (pl.col("close") < pl.col("macro_ema"))
        bear_signal = bear_alignment & (pl.col("roc") > -roc_lower) & (pl.col("ibs") > (1.0 - ibs_entry))

        df = df.with_columns([
            pl.when(bull_signal).then(1)
              .when(bear_signal).then(-1)
              .otherwise(0).alias("signal_trigger")
        ])

        return df

    def get_required_warmup(self, freq: str = "daily") -> int:
        return int(self.params.get("MACRO_EMA_PERIOD", 120)) + 20
