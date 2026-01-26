"""
Optimization Configuration for YetiRank Trading
Unified search space for all portfolio strategies.
"""

from typing import Dict, Any

# =========================================================
# 1. UNIFIED Search Space Configuration
# =========================================================

UNIFIED_CONFIG = {
    # Selection Strategy
    'TOP_K': {'low': 5, 'high': 50, 'step': 5},             # [MAX EXTENSION] Concentration vs Diversification
    'REBALANCE_PERIOD': {'low': 1, 'high': 20, 'step': 1},  # [MAX EXTENSION] Daily scalping to Monthly hold
    'FILTER_CANDIDATES_RATIO': {'low': 1.0, 'high': 5.0, 'step': 0.5},

    # Exit Strategy (K * Volatility)
    'STOP_LOSS_K': {'low': 1.0, 'high': 5.0, 'step': 0.5},
    'TAKE_PROFIT_K': {'low': 3.0, 'high': 50.0, 'step': 2.0}, # [MAX EXTENSION] Capture extreme trends
    'MAX_HOLD_DAYS': {'low': 10, 'high': 100, 'step': 5},     # [MAX EXTENSION] Short-term to Quarterly
    'MARKET_TIMING_THRESHOLD': {'low': 0.0, 'high': 0.5, 'step': 0.05},
    
    # Technical Filters
    # 1. RSI: 과열 판단 기준을 매우 넓게 (엄격함 ~ 관대함 모두 테스트)
    'USE_RSI_FILTER': {'choices': [True, False]},
    'RSI_MAX': {'low': 50, 'high': 95, 'step': 5},
    
    # 2. MFI: 자금 흐름 기준 확장
    'USE_MFI_FILTER': {'choices': [True, False]},
    'MFI_MAX': {'low': 50, 'high': 95, 'step': 5},
    
    # 3. ADX: 추세 강도 기준 확장
    'USE_ADX_FILTER': {'choices': [True, False]},
    'ADX_MIN': {'low': 10, 'high': 50, 'step': 5},
    
    # 4. Ichimoku: 동일
    'USE_ICHIMOKU_FILTER': {'choices': [True, False]},
    
    # 5. Bollinger Band: 밴드 상단 돌파도 폭넓게 허용
    'USE_BOLLINGER_FILTER': {'choices': [True, False]},
    'BB_POSITION_MAX': {'low': 0.8, 'high': 1.5, 'step': 0.1},
    
    # 6. Volume: 거래량 조건도 최소~최대 폭넓게
    'USE_VOLUME_FILTER': {'choices': [True, False]},
    'MIN_VOLUME_RATIO': {'low': 0.1, 'high': 1.0, 'step': 0.1},
    
    'USE_MA_FILTER': {'choices': [True, False]},
}

# =========================================================
# 2. Search Space Provider
# =========================================================

def GET_SEARCH_SPACE() -> Dict[str, Any]:
    """
    Returns the unified search space for optimizer.
    """
    return UNIFIED_CONFIG
