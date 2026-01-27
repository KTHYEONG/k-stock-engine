from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# KRX OpenAPI Settings
KRX_OPENAPI_KEY = os.getenv("KRX_OPENAPI_KEY", None)
USE_KRX_OPENAPI = os.getenv("USE_KRX_OPENAPI", "True").lower() in ("true", "1", "yes")

# Layer 1: Universe Filter Settings
MIN_MARKET_CAP = 50_000_000_000  # 500억 (50B KRW) - 너무 작은 초소형주 제외
MIN_TRADING_VALUE = 1_000_000_000  # 10억 (1B KRW) - 일 거래대금 최소값
MAX_TURNOVER_RATIO = 3.0         # 300% - 급등주/테마주 포용을 위해 대폭 완화
PB_HARD_CUT = 0.1
PB_WARNING_THRESHOLD = 0.3
CAPITAL_EROSION_WARNING = 30     # 30%
CAPITAL_EROSION_HARD_CUT = 50    # 50%
DEBT_RATIO_LIMIT = 200           # 200%

# Layer 2: Alpha Generation Settings
TARGET_HORIZON = 5               # 5일 (스윙)
RETRAIN_FREQUENCY = "weekly"
HALF_LIFE_DAYS = 365             # Time-decay 가중치 (1년)

# Layer 3: Risk Control Settings
TARGET_ANNUAL_VOL = 0.20         # 20%
MAX_SINGLE_STOCK_WEIGHT = 0.10   # 10%
MAX_SECTOR_WEIGHT = 0.30         # 30%
TIME_STOP_DAYS = 3
RR_RATIO_TARGET = 2.0            # ATR Trailing Stop Multiplier
