
"""
Optimization Configuration for Distinct Trading Modes (AI-First Approach)
Defines specialized search spaces designed to COMPLEMENT, not OBSTRUCT, the YetiRank AI model.
"""

from copy import deepcopy
from typing import Dict, Any

# =========================================================
# 1. PERIOD & THRESHOLD CONSTANTS (AI-Model Supportive)
# =========================================================

# ACTIVE: AI가 찍은 급등주/변동성 종목을 빠르게 매매
# 특징: 필터 최소화, 짧은 보유, 빠른 손절
ACTIVE_CONFIG = {
    'TOP_K':        {'low': 5,   'high': 20,  'step': 1},     # 분산 투자로 기회 확대
    'REBALANCE':    {'low': 1,   'high': 5,   'step': 1},     # 매우 빠른 회전
    'ENTRY_PERIOD': {'low': 3,   'high': 10,  'step': 1},     # 초단기 모멘텀 확인
    'MA_PERIOD':    {'low': 3,   'high': 20,  'step': 1},     # 장기 이평 무시, 단기 추세만 봄
    'SL_PCT':       {'low': 0.02,'high': 0.07, 'step': 0.005},# 짧은 손절
    'TP_ATR':       {'low': 1.0, 'high': 4.0,  'step': 0.5},  # 줄 때 먹고 나옴
    'ADX_THRESH':   {'low': 15,  'high': 30,   'step': 5},    # 추세가 약해도 AI 믿고 진입
    'MAX_HOLD':     {'low': 3,   'high': 10,   'step': 1},    # 2주 내 청산
}

# SWING: AI의 '상승 추세' 예측을 믿고 파동을 타는 전략
# 특징: 눌림목 허용, 적당한 여유
SWING_CONFIG = {
    'TOP_K':        {'low': 5,   'high': 15,  'step': 1},
    'REBALANCE':    {'low': 5,   'high': 20,  'step': 5},     # 주~월 단위
    'ENTRY_PERIOD': {'low': 10,  'high': 40,  'step': 5},
    'MA_PERIOD':    {'low': 20,  'high': 60,  'step': 10},    # 중기 이평 지지 확인
    'SL_PCT':       {'low': 0.05,'high': 0.15, 'step': 0.01}, # 일반적인 스윙 손절폭
    'TP_ATR':       {'low': 3.0, 'high': 8.0,  'step': 1.0},  # 추세 추종
    'ADX_THRESH':   {'low': 20,  'high': 40,   'step': 5},
    'MAX_HOLD':     {'low': 10,  'high': 40,   'step': 5},
}

# TREND: AI가 선정한 주도주를 길게 가져가는 전략
# 특징: 잦은 매매 지양, 큰 추세만 필터링
TREND_CONFIG = {
    'TOP_K':        {'low': 5,   'high': 10,  'step': 1},
    'REBALANCE':    {'low': 20,  'high': 60,  'step': 10},
    'ENTRY_PERIOD': {'low': 60,  'high': 120, 'step': 20},
    'MA_PERIOD':    {'low': 60,  'high': 200, 'step': 20},
    'SL_PCT':       {'low': 0.10,'high': 0.25, 'step': 0.01},
    'TP_ATR':       {'low': 5.0, 'high': 15.0, 'step': 2.0},  # 대세 상승 향유
    'ADX_THRESH':   {'low': 25,  'high': 50,   'step': 5},    # 확실한 추세만
    'MAX_HOLD':     {'low': 40,  'high': 120,  'step': 10},
}

# UNIFIED: AI-Centric Entry (진입은 AI, 청산은 기술적 지표)
UNIFIED_CONFIG = {
    'TOP_K':        {'low': 3,   'high': 8,  'step': 1},      # 소액 집중 투자
    'REBALANCE':    {'low': 2,   'high': 5,  'step': 1},      # 빠른 회전
    'ENTRY_PERIOD': {'low': 3,   'high': 60, 'log': True},    # 넓은 탐색
    'MA_PERIOD':    {'low': 5,   'high': 60, 'log': True},    # 최소한의 추세 확인
    'SL_PCT':       {'low': 0.03,'high': 0.10, 'step': 0.01}, # 손절은 확실하게
    'TP_ATR':       {'low': 1.5, 'high': 5.0,  'step': 0.5},  # 익절은 넉넉하게
    'ADX_THRESH':   {'low': 10,  'high': 50,   'step': 5},    # 추세 강도 무관 진입 허용
    'MAX_HOLD':     {'low': 3,   'high': 20,   'step': 1},    # 회전율 중시
}

# =========================================================
# 2. BASE SEARCH SPACE (Entry-Light, Exit-Heavy)
# =========================================================
BASE_SEARCH_SPACE = {
    # [Portfolio Strategy]
    'FILTER_CANDIDATES_RATIO': {'low': 1.0, 'high': 2.0, 'step': 0.1},
    'MARKET_TIMING_THRESHOLD': {'low': 0.0, 'high': 0.2, 'step': 0.05}, # 0.0 허용 (시장 무관)
    
    # [Indicator Selection Logic] - 진입 장벽 최소화 (None 우선)
    'MOMENTUM_FILTER': {'type': 'categorical', 'choices': ['None', 'RSI', 'CCI']}, # 최소한의 과열 방지
    'TREND_FILTER':    {'type': 'categorical', 'choices': ['None', 'MA', 'ADX', 'SUPERTREND']},
    'VOLATILITY_FILTER':{'type': 'categorical', 'choices': ['None']}, # 볼린저 밴드 등 진입 억제 필터 제거
    'VOLUME_FILTER':   {'type': 'categorical', 'choices': ['None', 'Volume']}, 

    # [Detailed Indicator Parameters] - 진입 허용 범위 대폭 확대
    # RSI: 95~99까지 허용 (사실상 모든 구간 진입)
    'RSI_MAX': {'low': 70, 'high': 99, 'step': 5},
    'STOCH_RSI_OVERBOUGHT': {'low': 80, 'high': 100, 'step': 5},
    'CCI_THRESHOLD': {'low': 100, 'high': 250, 'step': 10}, # 높은 CCI도 허용 (급등주)

    # MFI
    'MFI_MAX': {'low': 80, 'high': 100, 'step': 5},
    
    # Trend Strength
    'VHF_THRESHOLD': {'low': 0.2, 'high': 0.5, 'step': 0.05},

    # Ichimoku
    'ICHIMOKU_FILTER_TYPE': {'type': 'categorical', 'choices': ['TK_Cross']}, # 구름대 등 복잡한 조건 제외

    # SuperTrend 
    'SUPERTREND_MULT': {'low': 2.0, 'high': 4.0, 'step': 0.5},
    'SUPERTREND_PERIOD': {'low': 10, 'high': 20, 'step': 1},

    # Bollinger & Keltner - 진입 억제용이므로 완화
    'BB_POSITION_MAX': {'low': 1.0, 'high': 1.5, 'step': 0.1}, # 밴드 돌파 허용
    'BB_STD': {'low': 2.0, 'high': 3.0, 'step': 0.1},
    'KELTNER_ATR_MULT': {'low': 1.5, 'high': 3.0, 'step': 0.1},

    # Volume
    'MIN_VOLUME_RATIO': {'low': 0.5, 'high': 1.5, 'step': 0.1}, # 거래량 적어도 AI 믿음
    'CMF_THRESHOLD': {'low': -0.1, 'high': 0.1, 'step': 0.05},

    # [Risk Management] - 여기에 집중 탐색
    'USE_DYNAMIC_RISK': {'type': 'categorical', 'choices': [True, False]},
    
    # Dynamic Sizing
    'STRONG_REGIME_NATR': {'low': 1.0, 'high': 3.0, 'step': 0.2}, 
    'PANIC_REGIME_NATR':  {'low': 5.0, 'high': 15.0, 'step': 1.0}, # 급등주 변동성 용인
    
    # Exit Strategy Enhancements (핵심)
    'USE_TRAILING_STOP': {'type': 'categorical', 'choices': [True]}, # 무조건 사용
    'TRAILING_ACTIVATION_ATR': {'low': 2.0, 'high': 5.0, 'step': 0.5}, # 수익 발생 시 추적 손절 가동
}

def GET_SEARCH_SPACE(mode: str = 'UNIFIED', market_type: str = 'stock_spot') -> Dict[str, Any]:
    """
    Returns the search space for a specific mode.
    """
    space = deepcopy(BASE_SEARCH_SPACE)
    mode = mode.upper()
    
    if mode == 'SCALP': mode = 'ACTIVE'
    if mode == 'DAY':   mode = 'SWING'
    
    if mode == 'ACTIVE':
        cfg = ACTIVE_CONFIG
    elif mode == 'SWING':
        cfg = SWING_CONFIG
    elif mode == 'TREND':
        cfg = TREND_CONFIG
    else: 
        cfg = UNIFIED_CONFIG

    # Apply Mode-Specific Parameters
    space['TOP_K']              = cfg['TOP_K']
    space['REBALANCE_PERIOD']   = cfg['REBALANCE']
    space['ENTRY_LOOKBACK']     = cfg['ENTRY_PERIOD']
    space['MA_PERIOD']          = cfg['MA_PERIOD']
    space['STOP_LOSS_K']        = cfg['SL_PCT'] 
    space['TAKE_PROFIT_K']      = cfg['TP_ATR']
    space['ADX_MIN']            = cfg['ADX_THRESH']
    space['MAX_HOLD_DAYS']      = cfg['MAX_HOLD']
    
    # [Stock Spot Constraints - AI Friendly]
    if market_type == 'stock_spot':
        # 세금/수수료 고려하되, 너무 옥죄지 않음
        pass

    return space
