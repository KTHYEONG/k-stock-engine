
"""
Optimization Configuration for Phase 1: Pure Trend Following Baseline
Focus: Establish a solid foundation using only Price Action (MA) and Risk Management (Trailing Stop).
"""

from copy import deepcopy
from typing import Dict, Any

# =========================================================
# 1. CORE STRATEGY CONFIG (v5 - The Best Balanced)
# =========================================================

UNIFIED_CONFIG = {
    # 1. Candidate Selection (Concentrated Quality)
    'TOP_K':        {'low': 5,   'high': 10,  'step': 1},     # v5: 8
    
    # 2. Portfolio Management (Stable Trading)
    'REBALANCE':    {'low': 40,  'high': 100, 'step': 20},    # v5: 80
    'MAX_HOLD':     {'low': 60,  'high': 150, 'step': 30},    
    
    # 3. Entry Logic (Stable Trend)
    'ENTRY_PERIOD': {'low': 40,  'high': 80,  'step': 10},    # v5: 70
    'MA_PERIOD':    {'low': 80,  'high': 140, 'step': 20},    # v5: 120
    
    # 4. Exit Logic (ATR-based for Volatility Awareness)
    'SL_ATR':       {'low': 4.0, 'high': 6.0, 'step': 0.5},   # v5: 5.5
    'TP_ATR':       {'low': 8.0, 'high': 15.0, 'step': 1.0},  # v5: 10.0
    'ADX_THRESH':   {'low': 15,  'high': 30,   'step': 5},    
}

# =========================================================
# 2. SEARCH SPACE (v5 Finalized)
# =========================================================

BASE_SEARCH_SPACE = {
    # [Portfolio Strategy]
    'FILTER_CANDIDATES_RATIO': {'low': 1.2, 'high': 1.5, 'step': 0.1}, 
    'MARKET_TIMING_THRESHOLD': {'low': -0.1, 'high': 0.0, 'step': 0.02}, 
    
    # [Filter Activation - v5 Settings]
    'OSCILLATOR_FILTER': {'type': 'categorical', 'choices': ['None']}, # v6에서 성능 하락 확인됨
    'TREND_FILTER':      {'type': 'categorical', 'choices': ['MA']},   
    'STRENGTH_FILTER':   {'type': 'categorical', 'choices': ['ER']},   # [v5 Choice: Efficiency Ratio]
    'RISK_FILTER':       {'type': 'categorical', 'choices': ['NATR']}, # [v5 Choice: Volatility Filter]
    'VOLUME_FILTER':     {'type': 'categorical', 'choices': ['None']}, 

    # [Risk Management - The Core Engine]
    'USE_DYNAMIC_RISK': {'type': 'categorical', 'choices': [True]},  
    'USE_TRAILING_STOP': {'type': 'categorical', 'choices': [True]},  
    'TRAILING_ACTIVATION_ATR': {'low': 2.0, 'high': 4.5, 'step': 0.5}, # v5: 3.5
    
    # [Metric Constants - Fixed]
    'ADX_MIN': {'low': 15, 'high': 30, 'step': 5},
    
    # [Regime Settings - v5 Optimized]
    'STRONG_REGIME_ER': {'low': 0.6, 'high': 0.8, 'step': 0.1},      # v5: 0.8
    'STRONG_REGIME_NATR': {'type': 'categorical', 'choices': [2.0]}, 
    'PANIC_REGIME_NATR':  {'low': 10.0, 'high': 15.0, 'step': 1.0},  # v5: 12.0
    'VWAP_BAND_MULT': {'type': 'categorical', 'choices': [1.0]},
}

def GET_SEARCH_SPACE(mode: str = 'UNIFIED', market_type: str = 'stock_spot') -> Dict[str, Any]:
    """
    Returns the search space locked in with v5's robust framework.
    """
    space = deepcopy(BASE_SEARCH_SPACE)
    cfg = UNIFIED_CONFIG 

    # Apply Configuration
    space['TOP_K']              = cfg['TOP_K']
    space['REBALANCE_PERIOD']   = cfg['REBALANCE']
    space['ENTRY_LOOKBACK']     = cfg['ENTRY_PERIOD']
    space['MA_PERIOD']          = cfg['MA_PERIOD']
    space['STOP_LOSS_K']        = cfg['SL_ATR'] 
    space['TAKE_PROFIT_K']      = cfg['TP_ATR']
    space['MAX_HOLD_DAYS']      = cfg['MAX_HOLD']
    
    return space
    
    return space
