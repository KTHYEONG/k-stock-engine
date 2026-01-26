import optuna
import logging
from pathlib import Path
import sys
from typing import Optional, List, Dict, Any
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
        take_profit_k = trial.suggest_float("take_profit_k", **s['TAKE_PROFIT_K'])
        max_hold_days = trial.suggest_int("max_hold_days", **s['MAX_HOLD_DAYS'])
        
        market_timing_threshold = trial.suggest_float("market_timing_threshold", **s['MARKET_TIMING_THRESHOLD'])
        
        # --- [EXCLUSIVE SELECTION LOGIC] ---
        # 지표 중복으로 인한 과최적화 방지 및 탐색 효율성 극대화
        
        # 1. Momentum Category (과열/침체)
        momentum_choice = trial.suggest_categorical("momentum_filter", ["None", "RSI", "MFI"])
        use_rsi_filter = (momentum_choice == "RSI")
        use_mfi_filter = (momentum_choice == "MFI")
        
        rsi_max = 80
        if use_rsi_filter:
            rsi_max = trial.suggest_int("rsi_max", **s['RSI_MAX'])
        mfi_max = 80
        if use_mfi_filter:
            mfi_max = trial.suggest_int("mfi_max", **s['MFI_MAX'])
            
        # 2. Trend Category (추세 강도 및 방향)
        trend_choice = trial.suggest_categorical("trend_filter", ["None", "MA", "ADX", "Ichimoku"])
        use_ma_filter = (trend_choice == "MA")
        use_adx_filter = (trend_choice == "ADX")
        use_ichimoku_filter = (trend_choice == "Ichimoku")
        
        adx_min = 20
        if use_adx_filter:
            adx_min = trial.suggest_int("adx_min", **s['ADX_MIN'])
            
        # 3. Volatility Category (가격 위치)
        volat_choice = trial.suggest_categorical("volatility_filter", ["None", "Bollinger"])
        use_bollinger_filter = (volat_choice == "Bollinger")
        
        bb_position_max = 1.0
        if use_bollinger_filter:
            bb_position_max = trial.suggest_float("bb_position_max", **s['BB_POSITION_MAX'])
            
        # 4. Volume Category (수급)
        volume_choice = trial.suggest_categorical("volume_filter", ["None", "Volume"])
        use_volume_filter = (volume_choice == "Volume")
        
        min_volume_ratio = 0.5
        if use_volume_filter:
            min_volume_ratio = trial.suggest_float("min_volume_ratio", **s['MIN_VOLUME_RATIO'])
        
        try:
            # 2. 백테스트 실행
            metrics = self.backtester.run_backtest(
                top_k=top_k, 
                rebalance_period=rebalance_period,
                filter_candidates_ratio=filter_candidates_ratio,
                stop_loss_k=stop_loss_k,
                take_profit_k=take_profit_k,
                max_hold_days=max_hold_days,
                market_timing_threshold=market_timing_threshold,
                use_rsi_filter=use_rsi_filter, rsi_max=rsi_max,
                use_mfi_filter=use_mfi_filter, mfi_max=mfi_max,
                use_adx_filter=use_adx_filter, adx_min=adx_min,
                use_ichimoku_filter=use_ichimoku_filter,
                use_bollinger_filter=use_bollinger_filter, bb_position_max=bb_position_max,
                use_volume_filter=use_volume_filter, min_volume_ratio=min_volume_ratio,
                use_ma_filter=use_ma_filter,
                fee=0.002,
                save_plot=False
            )
            
            if not metrics:
                return -100.0
            
            # 3. 스코어 계산 (Advanced Robust Scoring)
            cagr = float(metrics["CAGR"].replace("%", ""))
            mdd = abs(float(metrics["MDD"].replace("%", "")))
            sharpe = float(metrics["Sharpe Ratio"])
            win_rate = float(metrics["Win Rate"].replace("%", ""))
            turnover = float(metrics["Avg Turnover"].replace("%", ""))
            
            # [Advanced Robust Scoring V2]
            # 1. CAGR 가중치 상향: 실질 수익 중심
            # 2. MDD 페널티 현실화: 너무 높으면 수익 기회를 놓침
            
            score = (cagr * 2.0) + (sharpe * 10.0) - (mdd * 8.0) + (win_rate * 0.1)
            
            # [Trade-off Penalty] 잦은 매매 제재 (회전율 10% 초과 시)
            if turnover > 10.0:
                score -= (turnover - 10.0) * 5.0
                
            if cagr < 0:
                score = -200.0 + cagr - (mdd * 2) # 원금 손실 전략은 강력 감점
                
            # 최적화 과정 분석용 데이터 기록
            trial.set_user_attr("cagr", cagr)
            trial.set_user_attr("mdd", mdd)
            trial.set_user_attr("sharpe", sharpe)
            trial.set_user_attr("turnover", turnover)
            trial.set_user_attr("win_rate", float(metrics["Win Rate"].replace("%", "")))
            trial.set_user_attr("sortino", float(metrics["Sortino Ratio"]))
            trial.set_user_attr("pl_ratio", float(metrics["P/L Ratio"]))
            
            return score
            
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return -100.0

    def run_optimization(self, n_trials: int = 50, study_name: Optional[str] = None, resume: bool = False, n_jobs: int = 1):
        if study_name is None:
            study_name = "yetirank_unified_opt"
            
        logger.info(f"🚀 Starting Optuna Optimization | Trials: {n_trials} | Resume: {resume} | Jobs: {n_jobs}")
        
        # DB 저장소 설정
        db_path = PROJECT_ROOT / "results" / "optimization.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # [RESET] 새로 시작 옵션이면 기존 DB 삭제
        if not resume and db_path.exists():
            logger.warning(f"⚠️ Deleting existing study DB: {db_path}")
            try:
                db_path.unlink()
            except PermissionError:
                logger.error("❌ Cannot delete DB file. It might be in use.")

        # [PERFORMANCE] SQLite WAL 모드 활성화 (병목 해소)
        import sqlite3
        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;") 
        except Exception as e:
            logger.warning(f"⚠️ Failed to enable WAL mode: {e}")
                
        storage_name = f"sqlite:///{db_path}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize"
        )
        # Multiprocessing: n_jobs > 1
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        
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
        print(f"- Result CAGR        : {best_trial.user_attrs.get('cagr', 0.0):.2f}%")
        print(f"- Result MDD         : -{best_trial.user_attrs.get('mdd', 0.0):.2f}%")
        print("="*50 + "\n")
        
        # 최적 파라미터로 최종 백테스트 수행 및 리포트 저장
        print("📊 Running final backtest with best parameters...")
        
        # [MAPPING] Categorical parameters를 Backtester가 이해하는 Boolean flags로 변환
        p = best_trial.params
        final_params = {k: v for k, v in p.items() if not k.endswith('_filter')}
        
        # Momentum
        m_choice = p.get('momentum_filter', 'None')
        final_params['use_rsi_filter'] = (m_choice == 'RSI')
        final_params['use_mfi_filter'] = (m_choice == 'MFI')
        
        # Trend
        t_choice = p.get('trend_filter', 'None')
        final_params['use_ma_filter'] = (t_choice == 'MA')
        final_params['use_adx_filter'] = (t_choice == 'ADX')
        final_params['use_ichimoku_filter'] = (t_choice == 'Ichimoku')
        
        # Volatility
        v_choice = p.get('volatility_filter', 'None')
        final_params['use_bollinger_filter'] = (v_choice == 'Bollinger')
        
        # Volume
        vol_choice = p.get('volume_filter', 'None')
        final_params['use_volume_filter'] = (vol_choice == 'Volume')
        
        self.backtester.run_backtest(
            **final_params,
            save_plot=True
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YetiRank Strategy Optimizer")
    parser.add_argument("--trials", type=int, default=300, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study (don't delete DB)")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel jobs (default: 2)")
    
    args = parser.parse_args()
    
    optimizer = YetiRankOptimizer()
    optimizer.run_optimization(n_trials=args.trials, resume=args.resume, n_jobs=args.n_jobs)
