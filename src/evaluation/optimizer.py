import optuna
import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.backtester import YetiRankBacktester
from src.evaluation.optimization_config import GET_SEARCH_SPACE
from src.utils.logger import setup_logger

# 로그 설정
logger = setup_logger("evaluation.optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("training.data_loader").setLevel(logging.WARNING)

class YetiRankOptimizer:
    """
    Optuna를 이용한 전략 하이퍼파라미터 최적화 도구
    UNIFIED_CONFIG를 기반으로 최적의 하이퍼파라미터 탐색
    """
    
    def __init__(self, start_date: str = "20240101", end_date: str = "20251231"):
        self.search_space = GET_SEARCH_SPACE()
        self.backtester = YetiRankBacktester(start_date=start_date, end_date=end_date)
        # 최적화 시작 전 모델 예측값 미리 캐싱 (속도 향상 핵심)
        self.backtester.generate_predictions()
        
    def objective(self, trial):
        """Optuna 목적 함수: UNIFIED 서치 스페이스 탐색"""
        
        # 1. 탐색 범위(Search Space) - config에서 가져옴
        s = self.search_space
        
        top_k = trial.suggest_int("top_k", **s['TOP_K'])
        rebalance_period = trial.suggest_int("rebalance_period", **s['REBALANCE_PERIOD'])
        filter_candidates_ratio = trial.suggest_float("filter_candidates_ratio", **s['FILTER_CANDIDATES_RATIO'])
        
        stop_loss_k = trial.suggest_float("stop_loss_k", **s['STOP_LOSS_K'])
        market_timing_threshold = trial.suggest_float("market_timing_threshold", **s['MARKET_TIMING_THRESHOLD'])
        
        use_rsi_filter = trial.suggest_categorical("use_rsi_filter", s['USE_RSI_FILTER']['choices'])
        rsi_max = 80
        if use_rsi_filter:
            rsi_max = trial.suggest_int("rsi_max", **s['RSI_MAX'])
            
        use_bollinger_filter = trial.suggest_categorical("use_bollinger_filter", s['USE_BOLLINGER_FILTER']['choices'])
        bb_position_max = 1.0
        if use_bollinger_filter:
            bb_position_max = trial.suggest_float("bb_position_max", **s['BB_POSITION_MAX'])
            
        use_volume_filter = trial.suggest_categorical("use_volume_filter", s['USE_VOLUME_FILTER']['choices'])
        min_volume_ratio = 0.5
        if use_volume_filter:
            min_volume_ratio = trial.suggest_float("min_volume_ratio", **s['MIN_VOLUME_RATIO'])
            
        use_ma_filter = trial.suggest_categorical("use_ma_filter", s['USE_MA_FILTER']['choices'])
        
        try:
            # 2. 백테스트 실행
            metrics = self.backtester.run_backtest(
                top_k=top_k, 
                rebalance_period=rebalance_period,
                filter_candidates_ratio=filter_candidates_ratio,
                stop_loss_k=stop_loss_k,
                market_timing_threshold=market_timing_threshold,
                use_rsi_filter=use_rsi_filter, rsi_max=rsi_max,
                use_bollinger_filter=use_bollinger_filter, bb_position_max=bb_position_max,
                use_volume_filter=use_volume_filter, min_volume_ratio=min_volume_ratio,
                use_ma_filter=use_ma_filter,
                fee=0.002,
                save_plot=False
            )
            
            if not metrics:
                return -100.0
            
            # 3. 스코어 계산
            sharpe = float(metrics["Sharpe Ratio"])
            mdd = abs(float(metrics["MDD"].replace("%", "")))
            cagr = float(metrics["CAGR"].replace("%", ""))
            
            # 점수 산식: 샤프 지수가 높고, MDD가 낮을수록 가중치
            if cagr < 0:
                score = cagr - mdd 
            else:
                score = (sharpe * 10) - (mdd * 0.5)
            
            # 보조 지표 기록 (Trial 결과에서 확인 가능)
            trial.set_user_attr("cagr", cagr)
            trial.set_user_attr("mdd", mdd)
            trial.set_user_attr("win_rate", float(metrics["Win Rate"].replace("%", "")))
            trial.set_user_attr("sortino", float(metrics["Sortino Ratio"]))
            trial.set_user_attr("pl_ratio", float(metrics["P/L Ratio"]))
            
            return score
            
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return -100.0

    def run_optimization(self, n_trials: int = 50, study_name: Optional[str] = None):
        if study_name is None:
            study_name = "yetirank_unified_opt"
            
        logger.info(f"🚀 Starting Optuna Optimization | Trials: {n_trials}")
        
        # DB 저장소 설정 (중단 후 재개 가능하도록)
        db_path = PROJECT_ROOT / "results" / "optimization.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{db_path}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize"
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info("✅ Optimization Complete!")
        
        # 결과 요약
        best_trial = study.best_trial
        print("\n" + "="*50)
        print("🏆 BEST STRATEGY PARAMETERS (UNIFIED)")
        print("="*50)
        for key, value in best_trial.params.items():
            print(f"- {key:<20}: {value}")
        print("-" * 50)
        print(f"- Best Score         : {best_trial.value:.4f}")
        print(f"- Result CAGR        : {best_trial.user_attrs['cagr']:.2f}%")
        print(f"- Result MDD         : -{best_trial.user_attrs['mdd']:.2f}%")
        print("="*50 + "\n")
        
        # 최적 파라미터로 최종 백테스트 수행 및 리포트 저장
        print("📊 Running final backtest with best parameters...")
        self.backtester.run_backtest(
            **best_trial.params,
            save_plot=True
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Strategy Optimizer")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    
    args = parser.parse_args()
    
    optimizer = YetiRankOptimizer()
    optimizer.run_optimization(n_trials=args.trials)
