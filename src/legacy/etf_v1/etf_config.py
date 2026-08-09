from typing import Dict, Any, Tuple
import datetime
from dateutil.relativedelta import relativedelta

OPT_ETF_CONFIG: Dict[str, Any] = {
    "total_trials": 2000,
    "n_startup_trials": 100,
    "seeds": [42],
    "n_jobs": 8,
    "task_workers": 1,
}

class ETFConfig:
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

    # IBS & Price Action Strategy Space
    SEARCH_SPACE_KOSPI = {
        "MACRO_EMA_PERIOD": {"type": "int", "low": 50, "high": 120, "step": 10},
        "FAST_EMA_PERIOD": {"type": "int", "low": 10, "high": 40, "step": 5},
        "ROC_N": {"type": "int", "low": 1, "high": 5, "step": 1},
        "ROC_LOWER": {"type": "float", "low": -0.03, "high": -0.005, "step": 0.005},
        "IBS_ENTRY": {"type": "float", "low": 0.10, "high": 0.50, "step": 0.05},
        "IBS_EXIT": {"type": "float", "low": 0.40, "high": 0.95, "step": 0.05},
        "MAX_HOLD_DAYS": {"type": "int", "low": 3, "high": 20, "step": 1},
        "STOP_LOSS_PCT": {"type": "float", "low": 0.05, "high": 0.15, "step": 0.01}
    }
    
    SEARCH_SPACE_KOSDAQ = SEARCH_SPACE_KOSPI.copy()

    @classmethod
    def get_search_space(cls, market: str) -> Dict[str, Any]:
        if market and market.upper() == "KOSDAQ":
            return cls.SEARCH_SPACE_KOSDAQ
        return cls.SEARCH_SPACE_KOSPI

    @staticmethod
    def get_default_params(market: str = "KOSPI") -> Dict[str, Any]:
        return {
            "TREND_PERIOD": 200,
            "ROC_N": 2,
            "ROC_LOWER": -0.02,
            "IBS_ENTRY": 0.15,
            "IBS_EXIT": 0.80,
            "MAX_HOLD_DAYS": 3,
            "STOP_LOSS_PCT": 0.10
        }

def get_quarterly_window(reference_date=None) -> Tuple[str, str, str, str]:
    if reference_date is None: 
        reference_date = datetime.date.today()
    elif isinstance(reference_date, str): 
        reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    elif isinstance(reference_date, datetime.datetime): 
        reference_date = reference_date.date()
        
    current_quarter_start_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_quarter_start: datetime.date = datetime.date(reference_date.year, current_quarter_start_month, 1)
    
    oos_end: datetime.date = current_quarter_start - datetime.timedelta(days=1)
    oos_start: datetime.date = current_quarter_start - relativedelta(months=6)
    
    is_start: datetime.date = oos_start - relativedelta(months=24)
    fetch_start: datetime.date = is_start - relativedelta(days=500)
    
    return (
        fetch_start.strftime("%Y-%m-%d"), 
        is_start.strftime("%Y-%m-%d"), 
        oos_start.strftime("%Y-%m-%d"), 
        oos_end.strftime("%Y-%m-%d")
    )