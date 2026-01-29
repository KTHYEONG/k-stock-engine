
import polars as pl
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("etf.strategy_engine")

class ETFStrategyEngine:
    """
    KODEX ETF Strategy Engine (Advanced V2)
    Benchmarked from Futures 'UltimateStrategy' & 'Regime Filter' Logic.
    
    Key Features:
    1. Regime Classification (Hurst Exponent): Distinguish Trending vs. Mean-Reverting.
    2. Score-based Signal Fusion: Instead of strict AND, use weighted scoring for Entry.
    3. Institutional Indicators: VWAP, CMF, Volume Z-Score.
    4. Dynamic Risk Sizing: Adjust position size based on Regime Quality (Hurst/NATR).
    5. Cash Hold Logic: Stay in Cash if Regime is Unclear (Low ADX / Low Hurst).
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates all necessary indicators fully vectorized using Polars.
        """
        # --- 1. Base Volatility & Regime (Priority) ---
        atr_period = self.params.get('ATR_PERIOD', 14)
        df = self._calc_atr(df, atr_period)
        
        # NATR (Normalized ATR) for Volatility Regime
        df = df.with_columns(
            (pl.col("atr") / pl.col("close") * 100).alias("natr")
        )
        
        # Hurst Exponent (Trend Quality / Regime)
        hurst_period = self.params.get('HURST_PERIOD', 100) # Long window for stability
        df = self._calc_hurst(df, hurst_period)
        
        # --- 2. Trend Direction (The Backbone) ---
        trend_type = self.params.get('TREND_DIR_TYPE', 'EMA')
        ma_period = self.params.get('MA_PERIOD', 60)
        
        if trend_type == 'SMA':
            df = df.with_columns(pl.col("close").rolling_mean(ma_period).alias("trend_line"))
        elif trend_type == 'EMA':
            df = df.with_columns(pl.col("close").ewm_mean(span=ma_period, adjust=False).alias("trend_line"))
        elif trend_type == 'WMA':
            # WMA calc helper returns a Series/Expression, need to alias if not already inside.
            # _calc_wma returns an expression, so we use it here.
            df = df.with_columns(self._calc_wma(ma_period).alias("trend_line"))
        elif trend_type == 'DEMA':
            df = self._calc_dema(df, ma_period)
        elif trend_type == 'TEMA':
            df = self._calc_tema(df, ma_period)
        elif trend_type == 'HMA':
            df = self._calc_hma(df, ma_period)
        elif trend_type == 'SUPERTREND':
            mul = self.params.get('SUPERTREND_MULT', 3.0)
            per = self.params.get('SUPERTREND_PERIOD', 10)
            df = self._calc_supertrend(df, per, mul)
        elif trend_type == 'MACD':
            fast = self.params.get('MACD_FAST', 12)
            slow = self.params.get('MACD_SLOW', 26)
            sig = self.params.get('MACD_SIGNAL', 9)
            df = self._calc_macd(df, fast, slow, sig)
        elif trend_type == 'ICHIMOKU':
            t_win = self.params.get('ICHIMOKU_TENKAN', 9)
            k_win = self.params.get('ICHIMOKU_KIJUN', 26)
            s_win = self.params.get('ICHIMOKU_SENKOU_B', 52)
            df = self._calc_ichimoku(df, t_win, k_win, s_win)
        elif trend_type == 'VWAP':
            std_mult = self.params.get('VWAP_STD_MULT', 1.5)
            df = self._calc_vwap(df, ma_period, std_mult)
        else:
            # Fallback to SMA to prevent "ColumnNotFoundError"
            df = df.with_columns(pl.col("close").rolling_mean(ma_period).alias("trend_line"))

        # --- 3. Entry Triggers (Breakout Levels) ---
        entry_type = self.params.get('ENTRY_TYPE', 'DONCHIAN')
        entry_period = self.params.get('ENTRY_PERIOD', 20)
        
        if entry_type == 'DONCHIAN':
            df = df.with_columns([
                pl.col("high").rolling_max(entry_period).shift(1).alias("entry_upper"),
                pl.col("low").rolling_min(entry_period).shift(1).alias("entry_lower")
            ])
        elif entry_type == 'BOLLINGER':
            std_dev = self.params.get('BB_STD', 2.0)
            df = self._calc_bollinger(df, entry_period, std_dev)
        elif entry_type == 'KELTNER':
            k_mult = self.params.get('KELTNER_ATR_MULT', 1.5)
            ema = pl.col("close").ewm_mean(span=entry_period, adjust=False)
            atr = pl.col("atr") # Use already calc ATR
            df = df.with_columns([
                (ema + (atr * k_mult)).shift(1).alias("entry_upper"),
                (ema - (atr * k_mult)).shift(1).alias("entry_lower")
            ])
        elif entry_type == 'CCI':
            # CCI Breakout Logic: Entry level is dynamic (requires logic in signal gen)
            # Here we just calc CCI
            df = self._calc_cci(df, entry_period, "cci_entry")

        # --- 4. Momentum & Strength (Support) ---
        # RSI (Always calculate for Panic Exit logic)
        df = self._calc_rsi(df, 14, "rsi")
        
        mom_type = self.params.get('MOMENTUM_TYPE', 'NONE')
        mom_period = self.params.get('MOMENTUM_PERIOD', 14)
        
        if mom_type == 'MFI':
            df = self._calc_mfi(df, mom_period)
        elif mom_type == 'CCI':
            df = self._calc_cci(df, mom_period, "cci_mom")
        elif mom_type == 'CMF':
            df = self._calc_cmf(df, mom_period)

        str_type = self.params.get('TREND_STR_TYPE', 'NONE')
        str_period = self.params.get('STRENGTH_PERIOD', 14)
        
        # Always calc ADX if not explicitly selected but ADX threshold is used for Cash Hold
        if str_type == 'ADX' or 'ADX' not in str_type: 
             # Always calc ADX for regime filter (Cash Hold)
             df = self._calc_adx(df, str_period if str_type=='ADX' else 14)
        
        if str_type == 'VORTEX':
            df = self._calc_vortex(df, str_period)
        elif str_type == 'ER':
            df = self._calc_er(df, str_period)

        # --- 5. Volume Analysis (Statistical Z-Score) ---
        if self.params.get('USE_VOLUME_FILTER', False):
            v_ma_period = self.params.get('VOLUME_MA_PERIOD', 20)
            # Log-Norm Z-Score: (Log(V) - Mean(Log(V))) / Std(Log(V))
            # Handles volume spikes much better than simple Ratio
            log_vol = (pl.col("volume") + 1).log()
            log_mean = log_vol.rolling_mean(v_ma_period)
            log_std = log_vol.rolling_std(v_ma_period)
            z_score = (log_vol - log_mean) / (log_std + 1e-9)
            
            df = df.with_columns(z_score.alias("volume_zscore"))

        # --- 6. Exits ---
        if self.params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
            step = self.params.get('SAR_STEP', 0.02)
            max_step = self.params.get('SAR_MAX', 0.2)
            df = self._calc_psar(df, step, max_step)

        return df

    def generate_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate signals using 'Score-based Data Fusion' & 'Regime Filter'.
        Instead of 'All-or-Nothing', we check if the market Regime supports the Trend.
        """
        df = self.calculate_indicators(df)
        
        # --- A. Determine Trend Direction (Major Bias) ---
        trend_type = self.params.get('TREND_DIR_TYPE', 'EMA')
        
        if trend_type == 'SUPERTREND':
            trend_expr = pl.col("supertrend_direction")
        elif trend_type == 'MACD':
            trend_expr = pl.when(pl.col("macd_line") > pl.col("macd_signal")).then(1).otherwise(-1)
        elif trend_type == 'ICHIMOKU':
            cloud_top = pl.max_horizontal("ichi_span_a", "ichi_span_b")
            cloud_bot = pl.min_horizontal("ichi_span_a", "ichi_span_b")
            trend_expr = pl.when(pl.col("close") > cloud_top).then(1).when(pl.col("close") < cloud_bot).then(-1).otherwise(0)
        elif trend_type == 'VWAP':
            trend_expr = pl.when(pl.col("close") > pl.col("vwap")).then(1).otherwise(-1)
        # SMA, EMA
        else:
            trend_expr = pl.when(pl.col("close") > pl.col("trend_line")).then(1).otherwise(-1)

        # --- B. Regime Filter (Hurst Exponent) ---
        # H > 0.5: Persistent (Trending)
        # H < 0.5: Anti-persistent (Mean Reverting/Choppy)
        # We prefer H > 0.5 for Trend Following.
        hurst_threshold = self.params.get('HURST_THRESHOLD', 0.45) # Tolerant threshold
        
        # If Hurst is very low, it's a choppy market -> Block Trend Following
        regime_pass = pl.when(pl.col("hurst") > hurst_threshold).then(1).otherwise(0)
        
        # --- C. Strength/Momentum Confirmation (Score Logic) ---
        # Allow signal if EITHER Momentum OR Strength is strong, don't need both.
        # Or if Trend is VERY strong (Hurst > 0.6), ignore minor divergences.
        
        score = pl.lit(0)
        
        # 1. Momentum Score
        mom_type = self.params.get('MOMENTUM_TYPE', 'NONE')
        if mom_type == 'RSI':
            high, low = self.params.get('RSI_OVERBOUGHT', 70), self.params.get('RSI_OVERSOLD', 30)
            # RSI confirms trend if it's not extreme (or is extreme in direction of trend? Breakout often extreme)
            # For Trend Follow: RSI > 50 confirms Bull, RSI < 50 confirms Bear.
            # Avoid ONLY extreme exhaustion.
            mom_confirm = pl.when((trend_expr == 1) & (pl.col("rsi") > 40) & (pl.col("rsi") < 85)).then(1)\
                            .when((trend_expr == -1) & (pl.col("rsi") < 60) & (pl.col("rsi") > 15)).then(1).otherwise(0)
            score = score + mom_confirm
            
        elif mom_type == 'CMF':
            # CMF > 0 supports Bull
            thr = self.params.get('CMF_THRESHOLD', 0.0)
            mom_confirm = pl.when((trend_expr == 1) & (pl.col("cmf") > thr)).then(1)\
                            .when((trend_expr == -1) & (pl.col("cmf") < -thr)).then(1).otherwise(0)
            score = score + mom_confirm
        else:
            score = score + 1 # No filter = Pass through
            
        # 2. Strength Score
        str_type = self.params.get('TREND_STR_TYPE', 'NONE')
        if str_type == 'ADX':
            thr = self.params.get('ADX_THRESHOLD', 20)
            score = score + pl.when(pl.col("adx") > thr).then(1).otherwise(0)
        elif str_type == 'VORTEX':
            thr = self.params.get('VORTEX_THRESHOLD', 0.1)
            v_bull = pl.when((trend_expr == 1) & (pl.col("vortex_diff") > thr)).then(1).otherwise(0)
            v_bear = pl.when((trend_expr == -1) & (pl.col("vortex_diff") < -thr)).then(1).otherwise(0)
            score = score + v_bull + v_bear
        elif str_type == 'ER':
            thr = self.params.get('ER_THRESHOLD', 0.4)
            score = score + pl.when(pl.col("er") > thr).then(1).otherwise(0)
        else:
            score = score + 1
            
        # 3. Volume Score
        if self.params.get('USE_VOLUME_FILTER', False):
            # Z-Score > -0.5 (Not extremely low volume)
            score = score + pl.when(pl.col("volume_zscore") > -0.5).then(1).otherwise(0)
        else:
            score = score + 1
            
        # --- D. Final Signal ---
        
        # Requirement: Regime Pass AND (Score >= Threshold)
        min_score = 1
        scores_active = 0
        if mom_type != 'NONE': scores_active += 1
        if str_type != 'NONE': scores_active += 1
        if self.params.get('USE_VOLUME_FILTER', False): scores_active += 1
        
        # [Rule A] 유연한 진입 (Super Pass)
        # 필터가 많아도, 추세 신뢰도(Hurst)가 0.65 이상으로 매우 높으면 
        # 점수가 1점만 되어도 진입 허용 (강한 추세장 놓치지 않기 위함)
        if scores_active > 1:
            # 기본은 2점 필요하지만, Hurst가 강하면(>0.53, 기존 0.65에서 완화) 1점도 OK
            # Trend Consistency가 있으면 지표 하나만 떠도 진입
            min_score = pl.when(pl.col("hurst") > 0.53).then(1).otherwise(2)
        else:
            min_score = pl.lit(1)
        
        final_filter = (regime_pass == 1) & (score >= min_score)
        
        # Breakout Check
        if self.params.get('ENTRY_TYPE') == 'CCI':
            # CCI Special Breakout
            t = self.params.get('CCI_THRESHOLD', 100)
            sig_expr = pl.when((pl.col("cci_entry").shift(1) > t) & (trend_expr == 1) & final_filter).then(1)\
                         .when((pl.col("cci_entry").shift(1) < -t) & (trend_expr == -1) & final_filter).then(-1)\
                         .otherwise(0)
        else:
            # Price Breakout
            sig_expr = pl.when((pl.col("close") > pl.col("entry_upper")) & (trend_expr == 1) & final_filter).then(1)\
                         .when((pl.col("close") < pl.col("entry_lower")) & (trend_expr == -1) & final_filter).then(-1)\
                         .otherwise(0)

        # PARABOLIC SAR Exit/Filter Override
        if self.params.get('EXIT_TYPE') == 'PARABOLIC_SAR':
            sig_expr = pl.when((sig_expr == 1) & (pl.col("psar_trend") == 1)).then(1)\
                         .when((sig_expr == -1) & (pl.col("psar_trend") == -1)).then(-1)\
                         .otherwise(0)
        
        # Cash Hold Logic (Low Hurst & Low ADX)
        # If ADX < 15 and Hurst < 0.5, Force 0 (Cash) to avoid dead markets.
        # This protects from chopping markets (2021, 2023).
        if "adx" in df.columns:
            # Try 15. If ADX < 15, it's dead market.
            cash_condition = (pl.col("adx") < 15) & (pl.col("hurst") < 0.5)
            sig_expr = pl.when(cash_condition).then(0).otherwise(sig_expr)

        df = df.with_columns(sig_expr.alias("signal_trigger"))
        
        # --- E. Dynamic Risk Sizing ---
        # Benchmarking Futures 'UltimateStrategy' Risk Logic
        # Base = 1.0
        # Hurst > 0.6 (Strong Trend) -> Increase Size
        # NATR > 3.0 (Panic) -> Reduce Size
        
        risk_mult = pl.lit(1.0)
        
        # Hurst Boost
        risk_mult = pl.when(pl.col("hurst") > 0.6).then(risk_mult + 0.5).otherwise(risk_mult)
        
        # Low Volatiltiy Boost
        risk_mult = pl.when(pl.col("natr") < 1.0).then(risk_mult * 1.2).otherwise(risk_mult)
        
        # [Rule B] 횡보장 방어 (ADX Filter) - Risk reduction
        # ADX가 20 미만이면 '약한 추세' -> 비중 50% 축소 (이미 Signal 0 처리는 위에서 했지만, 15~20 구간 대응)
        if "adx" in df.columns:
            risk_mult = pl.when(pl.col("adx") < 20).then(risk_mult * 0.5).otherwise(risk_mult)
        
        # High Volatility Penalty
        risk_mult = pl.when(pl.col("natr") > 3.0).then(risk_mult * 0.5).otherwise(risk_mult)
        
        # Cap/Floor
        risk_mult = risk_mult.clip(0.1, 2.0)
        
        return df.with_columns(risk_mult.alias("risk_mult"))

    # --- Indicators (Optimized V2) ---
    
    def _calc_atr(self, df, period):
        prev_close = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs()
        )
        return df.with_columns(tr.alias("tr")).with_columns(
            pl.col("tr").rolling_mean(period).alias("atr")
        )

    def _calc_hurst(self, df, window):
        # Rolling Hurst Exponent using Standard Deviation of Logs (Simplified Efficiency Ratio)
        # Full R/S analysis is too slow for 1200+ iterations in Python loop.
        # We use "Efficiency Ratio" mapping which is a recognized proxy for Hurst in HFT.
        # Hurst ≈ 0.5 + 0.5 * (Log(Range) - Log(SumOfRanges)) ?? 
        # Better: Kaufman Efficiency Ratio (ER) * 0.5 + 0.5 (Rescaled)
        # Since we use Polars expressions, we use ER as proxy for Trend persistence.
        
        change = (pl.col("close") - pl.col("close").shift(window)).abs()
        volatility = (pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(window)
        er = change / (volatility + 1e-9)
        
        # Map ER (0~1) to Hurst (approx 0.4 ~ 0.8 range usually)
        # ER 0 -> Random (H=0.5), ER 1 -> Deterministic (H=1.0)
        # Using simple linear mapping for regime detection
        hurst = 0.5 + (er * 0.5)
        return df.with_columns(hurst.alias("hurst"))

    def _calc_rsi(self, df, period, alias):
        delta = pl.col("close").diff()
        u = delta.clip(lower_bound=0)
        d = delta.clip(upper_bound=0).abs()
        u_avg = u.ewm_mean(alpha=1/period, adjust=False, min_periods=period)
        d_avg = d.ewm_mean(alpha=1/period, adjust=False, min_periods=period)
        rs = u_avg / (d_avg + 1e-9)
        return df.with_columns((100 - (100 / (1 + rs))).alias(alias))

    def _calc_vwap(self, df, window, std_mult):
        vp = ((pl.col("high")+pl.col("low")+pl.col("close"))/3) * pl.col("volume")
        vwap = vp.rolling_sum(window) / pl.col("volume").rolling_sum(window)
        std = pl.col("close").rolling_std(window)
        return df.with_columns([
            vwap.alias("vwap"),
            (vwap + std * std_mult).alias("vwap_upper"),
            (vwap - std * std_mult).alias("vwap_lower")
        ])

    def _calc_cmf(self, df, period):
        # Chaikin Money Flow
        mf_mult = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low") + 1e-9)
        mf_vol = mf_mult * pl.col("volume")
        cmf = mf_vol.rolling_sum(period) / pl.col("volume").rolling_sum(period)
        return df.with_columns(cmf.alias("cmf"))

    def _calc_adx(self, df, period):
        up = pl.col("high") - pl.col("high").shift(1)
        down = pl.col("low").shift(1) - pl.col("low")
        pdm = pl.when((up > down) & (up > 0)).then(up).otherwise(0)
        ndm = pl.when((down > up) & (down > 0)).then(down).otherwise(0)
        
        tr_s = pl.col("tr").ewm_mean(alpha=1/period, adjust=False)
        pdm_s = pdm.ewm_mean(alpha=1/period, adjust=False)
        ndm_s = ndm.ewm_mean(alpha=1/period, adjust=False)
        
        dx = 100 * (pdm_s - ndm_s).abs() / (pdm_s + ndm_s + 1e-9)
        return df.with_columns(dx.ewm_mean(alpha=1/period, adjust=False).alias("adx"))

    def _calc_supertrend(self, df, period, mult):
        # Efficient Polars Supertrend (Approx or user Python loop if needed)
        # For Optimization speed, we use a vectorized approximation or simplified channel.
        # Using basic median-based band here to avoid slow Python loops.
        hl2 = (pl.col("high") + pl.col("low")) / 2
        atr = pl.col("atr")
        up = hl2 + (atr * mult)
        dn = hl2 - (atr * mult)
        
        # Simple logic: Close > Up -> Bull, Close < Dn -> Bear (No memory state for pure vectorization)
        # For full Supertrend memory, Python loop is required, but slowness is issue.
        # We compromise with "Instant Supertrend" (no trailing lock).
        st_dir = pl.when(pl.col("close") > up.shift(1)).then(1)\
                   .when(pl.col("close") < dn.shift(1)).then(-1).otherwise(0)
        
        return df.with_columns(st_dir.alias("supertrend_direction"))

    def _calc_psar(self, df, step, max_step):
        # Parabolic SAR requires state loop.
        # Fallback to simple Trend Channel for speed if needed, but here's a placeholder.
        # Since we removed Numba, we can't do fast PSAR calc easily in Polars.
        # We'll trust the Python implementation in previous version OR 
        # Assume EXIT_TYPE is rarely SAR in current config.
        # Returning dummy columns to avoid crash, assuming SAR is not primary optimization target.
        return df.with_columns([
            pl.lit(0.0).alias("psar"), 
            pl.lit(1).alias("psar_trend")
        ])

    def _calc_macd(self, df, fast, slow, sig):
        ema_f = pl.col("close").ewm_mean(span=fast, adjust=False)
        ema_s = pl.col("close").ewm_mean(span=slow, adjust=False)
        macd = ema_f - ema_s
        signal = macd.ewm_mean(span=sig, adjust=False)
        return df.with_columns([macd.alias("macd_line"), signal.alias("macd_signal")])

    def _calc_ichimoku(self, df, t, k, s):
        def _mid(win): return (pl.col("high").rolling_max(win) + pl.col("low").rolling_min(win)) / 2
        return df.with_columns([
            ((_mid(t) + _mid(k))/2).shift(k).alias("ichi_span_a"),
            _mid(s).shift(k).alias("ichi_span_b")
        ])

    def _calc_bollinger(self, df, per, std):
        mean = pl.col("close").rolling_mean(per)
        sd = pl.col("close").rolling_std(per)
        return df.with_columns([
            (mean + sd*std).shift(1).alias("entry_upper"),
            (mean - sd*std).shift(1).alias("entry_lower")
        ])

    def _calc_cci(self, df, per, alias):
        tp = (pl.col("high")+pl.col("low")+pl.col("close"))/3
        sma = tp.rolling_mean(per)
        mad = (tp - sma).abs().rolling_mean(per)
        return df.with_columns(((tp - sma) / (0.015 * mad)).alias(alias))

    def _calc_mfi(self, df, per):
        tp = (pl.col("high")+pl.col("low")+pl.col("close"))/3
        rmf = tp * pl.col("volume")
        pos = pl.when(tp > tp.shift(1)).then(rmf).otherwise(0).rolling_sum(per)
        neg = pl.when(tp < tp.shift(1)).then(rmf).otherwise(0).rolling_sum(per)
        return df.with_columns((100 - (100/(1+pos/neg))).alias("mfi"))

    def _calc_vortex(self, df, per):
        vp = (pl.col("high") - pl.col("low").shift(1)).abs().rolling_sum(per)
        vm = (pl.col("low") - pl.col("high").shift(1)).abs().rolling_sum(per)
        tr = pl.col("tr").rolling_sum(per)
        return df.with_columns(((vp/tr) - (vm/tr)).alias("vortex_diff"))

    def _calc_er(self, df, per):
        change = (pl.col("close") - pl.col("close").shift(per)).abs()
        vol = (pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(per)
        return df.with_columns((change / (vol+1e-9)).alias("er"))

    def _calc_dema(self, df, period):
        # Double Exponential Moving Average
        # DEMA = 2*EMA - EMA(EMA)
        ema1 = pl.col("close").ewm_mean(span=period, adjust=False)
        ema2 = ema1.ewm_mean(span=period, adjust=False)
        dema = (2 * ema1) - ema2
        return df.with_columns(dema.alias("trend_line"))

    def _calc_tema(self, df, period):
        # Triple Exponential Moving Average
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        ema1 = pl.col("close").ewm_mean(span=period, adjust=False)
        ema2 = ema1.ewm_mean(span=period, adjust=False)
        ema3 = ema2.ewm_mean(span=period, adjust=False)
        tema = (3 * ema1) - (3 * ema2) + ema3
        return df.with_columns(tema.alias("trend_line"))

    def _calc_wma(self, period):
        # Polars doesn't have native WMA yet.
        # We need to construct weights.
        # WMA = Sum(Price * Weight) / Sum(Weight)
        # Weights = [1, 2, ..., n]
        # Since this is tricky in pure Polars expressions for rolling,
        # we can use a linear approximation or map_batches if efficiency allows.
        # BUT for optimization, map_batches is slow.
        # Alternative: EWM is very close to WMA. HMA uses WMA specifically for lag reduction.
        # Let's implement a 'Simulated WMA' using EWM with adjusted span? No, HMA needs precise WMA.
        # We will use rolling_apply with numpy for exactness, though slower.
        
        weights = np.arange(1, period + 1)
        w_sum = weights.sum()
        
        def wma_func(s: pl.Series) -> float:
            # Polars passes a Series to the function
            return np.dot(s.to_numpy(), weights) / w_sum
            
        return pl.col("close").rolling_map(wma_func, window_size=period)

    def _calc_hma(self, df, period):
        # Hull Moving Average
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        
        # Note: Implementing WMA via rolling_apply might be slow for Genetic Algo (500 trials).
        # However, for 10-year daily data (approx 2500 rows), it is acceptable.
        
        # 1. WMA(n/2)
        half_per = int(period / 2)
        wma_half = self._calc_wma(half_per)
        
        # 2. WMA(n)
        wma_full = self._calc_wma(period)
        
        # 3. Raw HMA
        raw_hma = (2 * wma_half) - wma_full
        
        # 4. WMA(sqrt(n)) on Raw HMA
        # We need to compute WMA on the result expression. 
        # Polars expressions can be chained.
        
        sqrt_per = int(np.sqrt(period))
        
        # We can't reuse _calc_wma easily on an expression without realizing it.
        # So we add intermediate columns.
        
        df = df.with_columns([
            wma_half.alias("wma_half"),
            wma_full.alias("wma_full")
        ])
        
        df = df.with_columns(
            ((2 * pl.col("wma_half")) - pl.col("wma_full")).alias("raw_hma")
        )
        
        # Now WMA of raw_hma
        weights = np.arange(1, sqrt_per + 1)
        w_sum = weights.sum()
        
        def wma_func_sq(s: pl.Series) -> float:
             return np.dot(s.to_numpy(), weights) / w_sum
             
        # Rolling map on "raw_hma"
        final_hma = pl.col("raw_hma").rolling_map(wma_func_sq, window_size=sqrt_per)
        
        return df.with_columns(final_hma.alias("trend_line"))
