"""
Optimization Configuration for YetiRank Trading
Unified search space for all portfolio strategies.
"""

from typing import Dict, Any

# =========================================================
# 1. UNIFIED Search Space Configuration
# =========================================================

UNIFIED_CONFIG = {
    # Portfolio Settings
    'TOP_K': {'low': 5, 'high': 30, 'step': 5},
    'REBALANCE_PERIOD': {'low': 1, 'high': 10},
    
    # [NEW] Hybrid Selection (2-Stage)
    # 랭킹 상위 N배수를 뽑은 뒤 기술적 필터로 걸러냄 (예: Top-K가 20이면 40~60개 후보 중 선별)
    'FILTER_CANDIDATES_RATIO': {'low': 1.5, 'high': 3.0, 'step': 0.5},
    
    # Risk Management
    'STOP_LOSS_K': {'low': 1.5, 'high': 4.0, 'step': 0.5},
    'MARKET_TIMING_THRESHOLD': {'low': 0.1, 'high': 0.4, 'step': 0.05},
    
    # Technical Filters (코인 전략 이식)
    # 1. RSI: 과열 종목 진입 금지
    'USE_RSI_FILTER': {'choices': [True, False]},
    'RSI_MAX': {'low': 70, 'high': 85, 'step': 5},
    
    # 2. Bollinger Band: 밴드 상단 너무 뚫은거(단기 과열) 제외
    'USE_BOLLINGER_FILTER': {'choices': [True, False]},
    'BB_POSITION_MAX': {'low': 0.8, 'high': 1.1, 'step': 0.1}, # 1.0 = 상단 밴드
    
    # 3. Volume: 전일 대비 거래량 너무 없는거 제외 (소외주 방지)
    'USE_VOLUME_FILTER': {'choices': [True, False]},
    'MIN_VOLUME_RATIO': {'low': 0.3, 'high': 0.8, 'step': 0.1},
    
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
