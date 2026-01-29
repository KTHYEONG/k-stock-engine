
from typing import Dict, Any

class ETFConfig:
    """
    ETF Trading System Configuration
    Aligned with Coin Trader 'UltimateStrategy' parameters.
    """
    
    # 1. Trading Universe (User defined mapping - Step 219)
    UNIVERSE = {
        "KOSPI": {
            "index_ticker": "코스피 200", 
            "bull_1x": "069500",    # KODEX 200
            "bull_2x": "122630",    # KODEX 레버리지
            "bear_1x": "114800",    # KODEX 인버스
            "bear_2x": "252670"     # KODEX 200선물인버스2X
        },
        "KOSDAQ": {
            "index_ticker": "코스닥 150",
            "bull_1x": "229200",    # KODEX 코스닥 150
            "bull_2x": "233740",    # KODEX 코스닥 150 레버리지
            "bear_1x": "251340",    # KODEX 코스닥 150선물인버스
            "bear_2x": "252710"     # TIGER 코스닥 150선물인버스2X
        }
    }
    
    # ==============================================================================
    # 🔵 KOSPI Search Space
    # ==============================================================================
    SEARCH_SPACE_KOSPI = {
        # --- 1. Entry Logic ---
        "ENTRY_TYPE": ["DONCHIAN", "BOLLINGER", "KELTNER", "CCI"],
        "ENTRY_PERIOD": {"type": "int", "low": 3, "high": 100, "step": 1},
        "BB_STD": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
        "KELTNER_ATR_MULT": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
        "CCI_THRESHOLD": {"type": "int", "low": 50, "high": 150, "step": 10},
        
        # --- 2. Trend Direction ---
        "TREND_DIR_TYPE": ["SMA", "EMA", "DEMA", "TEMA", "WMA", "HMA"],
        "MA_PERIOD": {"type": "int", "low": 5, "high": 200, "step": 5},
        "SUPERTREND_MULT": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
        "SUPERTREND_PERIOD": {"type": "int", "low": 5, "high": 30, "step": 1},
        "MACD_FAST": {"type": "int", "low": 8, "high": 20, "step": 1},
        "MACD_SLOW": {"type": "int", "low": 20, "high": 60, "step": 2},
        "MACD_SIGNAL": {"type": "int", "low": 5, "high": 15, "step": 1},
        "ICHIMOKU_TENKAN": {"type": "int", "low": 7, "high": 20},
        "ICHIMOKU_KIJUN": {"type": "int", "low": 20, "high": 60},
        "ICHIMOKU_SENKOU_B": {"type": "int", "low": 40, "high": 100},
        "VWAP_STD_MULT": {"type": "float", "low": 0.5, "high": 3.0},
        
        # --- 3. Momentum ---
        "MOMENTUM_TYPE": ["NONE", "RSI", "MFI", "CCI", "CMF"],
        "MOMENTUM_PERIOD": {"type": "int", "low": 5, "high": 40, "step": 1},
        "RSI_OVERBOUGHT": {"type": "int", "low": 60, "high": 85},
        "RSI_OVERSOLD": {"type": "int", "low": 15, "high": 40},
        "MFI_THRESHOLD": {"type": "int", "low": 20, "high": 80},
        "CMF_THRESHOLD": {"type": "float", "low": -0.1, "high": 0.3},
        
        # --- 4. Trend Strength ---
        "TREND_STR_TYPE": ["NONE", "ADX", "VORTEX", "ER"],
        "STRENGTH_PERIOD": {"type": "int", "low": 5, "high": 40, "step": 1},
        "ADX_THRESHOLD": {"type": "int", "low": 15, "high": 40},
        "VORTEX_THRESHOLD": {"type": "float", "low": 0.1, "high": 0.8},
        "ER_THRESHOLD": {"type": "float", "low": 0.3, "high": 0.8},
        
        # --- 5. Volatility / Quality ---
        "VOLATILITY_TYPE": ["NONE", "NATR", "CHOP"],
        "VOLATILITY_PERIOD": {"type": "int", "low": 5, "high": 40, "step": 1},
        "NATR_THRESHOLD": {"type": "float", "low": 1.0, "high": 5.0},
        "CHOP_THRESHOLD": {"type": "int", "low": 40, "high": 60},
        
        # --- 6. Volume & Exit ---
        "USE_VOLUME_FILTER": [True, False],
        "VOLUME_MA_PERIOD": {"type": "int", "low": 10, "high": 60, "step": 5},
        "EXIT_TYPE": ["PARABOLIC_SAR", "NONE"],
        "SAR_STEP": {"type": "float", "low": 0.01, "high": 0.04, "step": 0.01},
        
        # --- 7. Risk Management ---
        "STOP_LOSS_ATR": {"type": "float", "low": 1.5, "high": 8.0, "step": 0.5},
        "TAKE_PROFIT_ATR": {"type": "float", "low": 10.0, "high": 200.0, "step": 5.0},
        "TS_TRIGGER_ATR": {"type": "float", "low": 1.0, "high": 8.0, "step": 0.5},
        "TS_DIST_ATR": {"type": "float", "low": 1.0, "high": 8.0, "step": 0.5},
        
        # --- 8. Regime & Advanced ---
        "HURST_PERIOD": {"type": "int", "low": 10, "high": 200, "step": 10},
        "HURST_THRESHOLD": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.05},
        "SAR_MAX": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.05},
        
        # --- 9. Hybrid Leverage ---
        "LEV_HURST": {"type": "float", "low": 0.1, "high": 0.7, "step": 0.05},
        "LEV_NATR": {"type": "float", "low": 1.0, "high": 10.0, "step": 0.5},
    }

    # ==============================================================================
    # 🟠 KOSDAQ Search Space (High Volatility Optimized)
    # ==============================================================================
    SEARCH_SPACE_KOSDAQ = {
        # --- 1. Entry Logic ---
        "ENTRY_TYPE": ["DONCHIAN", "BOLLINGER", "KELTNER", "CCI"],
        "ENTRY_PERIOD": {"type": "int", "low": 10, "high": 60, "step": 1},
        "BB_STD": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
        "KELTNER_ATR_MULT": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
        "CCI_THRESHOLD": {"type": "int", "low": 50, "high": 150, "step": 10},
        
        # --- 2. Trend Direction ---
        "TREND_DIR_TYPE": ["SMA", "EMA", "DEMA", "TEMA", "WMA", "HMA"],
        "MA_PERIOD": {"type": "int", "low": 5, "high": 200, "step": 5},
        "SUPERTREND_MULT": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
        "SUPERTREND_PERIOD": {"type": "int", "low": 5, "high": 30, "step": 1},
        "MACD_FAST": {"type": "int", "low": 8, "high": 20, "step": 1},
        "MACD_SLOW": {"type": "int", "low": 20, "high": 60, "step": 2},
        "MACD_SIGNAL": {"type": "int", "low": 5, "high": 15, "step": 1},
        "ICHIMOKU_TENKAN": {"type": "int", "low": 7, "high": 20},
        "ICHIMOKU_KIJUN": {"type": "int", "low": 20, "high": 60},
        "ICHIMOKU_SENKOU_B": {"type": "int", "low": 40, "high": 100},
        "VWAP_STD_MULT": {"type": "float", "low": 0.5, "high": 3.0},
        
        # --- 3. Momentum ---
        "MOMENTUM_TYPE": ["NONE", "RSI", "MFI", "CCI", "CMF"],
        "MOMENTUM_PERIOD": {"type": "int", "low": 5, "high": 40, "step": 1},
        "RSI_OVERBOUGHT": {"type": "int", "low": 60, "high": 85},
        "RSI_OVERSOLD": {"type": "int", "low": 15, "high": 40},
        "MFI_THRESHOLD": {"type": "int", "low": 20, "high": 80},
        "CMF_THRESHOLD": {"type": "float", "low": -0.1, "high": 0.3},
        
        # --- 4. Trend Strength ---
        "TREND_STR_TYPE": ["NONE", "ADX", "VORTEX", "ER"],
        "STRENGTH_PERIOD": {"type": "int", "low": 5, "high": 40, "step": 1},
        "ADX_THRESHOLD": {"type": "int", "low": 15, "high": 40},
        "VORTEX_THRESHOLD": {"type": "float", "low": 0.1, "high": 0.8},
        "ER_THRESHOLD": {"type": "float", "low": 0.3, "high": 0.7},
        
        # --- 5. Volatility / Quality ---
        "VOLATILITY_TYPE": ["NONE", "NATR", "CHOP"],
        "VOLATILITY_PERIOD": {"type": "int", "low": 5, "high": 40, "step": 1},
        "NATR_THRESHOLD": {"type": "float", "low": 1.0, "high": 5.0},
        "CHOP_THRESHOLD": {"type": "int", "low": 40, "high": 60},
        
        # --- 6. Volume & Exit ---
        "USE_VOLUME_FILTER": [True, False],
        "VOLUME_MA_PERIOD": {"type": "int", "low": 10, "high": 60, "step": 5},
        "EXIT_TYPE": ["PARABOLIC_SAR", "NONE"],
        "SAR_STEP": {"type": "float", "low": 0.01, "high": 0.04, "step": 0.01},
        
        # --- 7. Risk Management ---
        "STOP_LOSS_ATR": {"type": "float", "low": 1.5, "high": 8.0, "step": 0.5},
        "TAKE_PROFIT_ATR": {"type": "float", "low": 10.0, "high": 200.0, "step": 5.0},
        "TS_TRIGGER_ATR": {"type": "float", "low": 1.0, "high": 6.0, "step": 0.5},
        "TS_DIST_ATR": {"type": "float", "low": 1.0, "high": 6.0, "step": 0.5},
        
        # --- 8. Regime & Advanced ---
        "HURST_PERIOD": {"type": "int", "low": 10, "high": 200, "step": 10},
        "HURST_THRESHOLD": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.05},
        "SAR_MAX": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.05},
        
        # --- 9. Hybrid Leverage ---
        "LEV_HURST": {"type": "float", "low": 0.1, "high": 0.7, "step": 0.05},
        "LEV_NATR": {"type": "float", "low": 1.0, "high": 10.0, "step": 0.5},
    }
    
    @classmethod
    def get_search_space(cls, market: str) -> Dict[str, Any]:
        """
        Returns the appropriate search space based on the market.
        """
        if market and market.upper() == "KOSDAQ":
            return cls.SEARCH_SPACE_KOSDAQ
        return cls.SEARCH_SPACE_KOSPI

    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        """Provides a safe default parameter set for testing"""
        return {
            "ENTRY_TYPE": "DONCHIAN",
            "ENTRY_PERIOD": 20,
            "TREND_DIR_TYPE": "EMA",
            "MA_PERIOD": 60,
            "MOMENTUM_TYPE": "NONE",
            "TREND_STR_TYPE": "ADX",
            "STRENGTH_PERIOD": 14,
            "ADX_THRESHOLD": 20,
            "VOLATILITY_TYPE": "NONE",
            "USE_VOLUME_FILTER": False,
            "EXIT_TYPE": "NONE",
            "STOP_LOSS_ATR": 3.0,
            "TAKE_PROFIT_ATR": 15.0,
            "TS_TRIGGER_ATR": 4.0,
            "TS_DIST_ATR": 2.0,
            "HURST_PERIOD": 100,
            "HURST_THRESHOLD": 0.45,
            "LEV_HURST": 0.55,
            "LEV_NATR": 2.5
        }
