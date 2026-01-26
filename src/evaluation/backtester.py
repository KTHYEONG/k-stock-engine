import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any
from catboost import CatBoostRanker
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.data_loader import YetiRankDataLoader
from src.utils.logger import setup_logger

# Mute verbose info logs from data_loader during grid search
import logging
logging.getLogger("training.data_loader").setLevel(logging.WARNING)

logger = setup_logger("evaluation.backtester")

class YetiRankBacktester:
    """
    YetiRank 모델의 실전 성과 검증 (Backtesting)
    - 2024, 2025년 Out-of-Sample 데이터를 대상으로 수행
    - Top-K 종목 선정 및 수익률/리스크 지표 산출
    """
    
    def __init__(self, start_date: str = "20240101", end_date: str = "20251231"):
        self.loader = YetiRankDataLoader(start_date="20160401") # 로더는 전체 기간 기준
        self.model_dir = PROJECT_ROOT / "models" / "yetirank"
        self.output_dir = PROJECT_ROOT / "results" / "backtest"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_date = start_date
        self.end_date = end_date
        
    def load_model(self, year: int) -> CatBoostRanker:
        model_path = self.model_dir / f"yetirank_{year}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = CatBoostRanker()
        model.load_model(str(model_path))
        return model

    def generate_predictions(self):
        """
        모델 예측값 + 기술적 지표 미리 생성
        """
        logger.info("⚡ Generating Model Predictions & Indicators for caching...")
        
        # 1. 데이터 로드
        if not hasattr(self, "_cached_full_df"):
             # load_full_data 시 close, open 등 raw price가 포함되어야 지표 계산 가능
             # loader가 feature만 리턴한다면 raw 컬럼도 유지하도록 확인 필요.
             # 여기선 loader가 전체 컬럼을 로드한다고 가정.
             self._cached_full_df = self.loader.load_full_data(end_date=self.end_date, sample_ratio=1.0)
             self._cached_feature_names = self.loader.get_feature_names(self._cached_full_df)
        
        full_df = self._cached_full_df
        feature_names = self._cached_feature_names
        
        # 테스트 구간 필터링
        test_df = full_df.filter(
            (pl.col("date") >= pl.lit(self.start_date).str.to_date("%Y%m%d")) & 
            (pl.col("date") <= pl.lit(self.end_date).str.to_date("%Y%m%d"))
        ).sort("date")
        
        if test_df.is_empty():
            logger.warning("⚠️ No data found for prediction.")
            self._cached_predictions = pl.DataFrame()
            return

        # 2. 예측 및 지표 생성
        predictions = []
        years = test_df.select(pl.col("date").dt.year()).unique().to_series().to_list()
        
        for year in sorted(years):
            year_data = test_df.filter(pl.col("date").dt.year() == year)
            if year_data.is_empty(): continue
            try:
                model = self.load_model(year)
                # Feature만 선택하여 추론
                X = year_data.select(feature_names).to_pandas()
                scores = model.predict(X)
                
                # 예측 결과 DF 생성
                # 지표 계산을 위해 'close' 컬럼이 필수. (loader 데이터에 있다고 가정)
                # 만약 없다면 features 중 가격 대용 변수를 찾아야 함.
                cols_to_keep = ["date", "ticker", "log_return_1d", "volatility_20d"]
                if "close" in year_data.columns: cols_to_keep.append("close")
                
                pred_df = year_data.select(cols_to_keep).with_columns(pl.Series("pred_score", scores))
                
                # [INDICATORS] Polars 표현식으로 보조지표 계산
                # 1. RSI (14) - Wilder's Smoothing 대신 Simple RSI로 근사 (속도 위주)
                # 2. SMA (20, 60)
                
                # Close가 있다면 사용, 없다면 수익률로 가상 인덱스 생성
                price_col = pl.col("close") if "close" in pred_df.columns else (1 + pl.col("log_return_1d")).cumprod().over("ticker")
                
                pred_df = pred_df.with_columns([
                    pl.col("log_return_1d").shift(-1).over("ticker").alias("next_day_ret"),
                    
                    # SMA
                    price_col.rolling_mean(20).over("ticker").fill_null(price_col).alias("sma_20"),
                    price_col.rolling_mean(60).over("ticker").fill_null(price_col).alias("sma_60"),
                    
                    # RSI Calculation (Simplified)
                    # u = up moves, d = down moves
                ])
                
                # RSI 계산 (별도 단계로 분리하여 가독성 확보)
                # change = price_col.diff()
                # gain = change.clip(lower_bound=0)
                # loss = -change.clip(upper_bound=0)
                # avg_gain = gain.rolling_mean(14)
                # avg_loss = loss.rolling_mean(14)
                # rs = avg_gain / (avg_loss + 1e-9)
                # rsi = 100 - (100 / (1 + rs))
                
                # Polars 복합 표현식
                # 2. Bollinger Band (20, 2)
                # bb_mean = rolling_mean(20)
                # bb_std = rolling_std(20)
                # bb_upper = mean + 2*std
                # bb_pos = (price - mean) / (2*std) -> 정규화된 위치 (0: mean, 1: upper, -1: lower)
                
                # 3. Volume Ratio (vs 20MA)
                # Volume data가 있다고 가정. 없다면 1.0 처리
                vol_col = pl.col("volume") if "volume" in pred_df.columns else pl.lit(1.0)
                
                pred_df = pred_df.with_columns(
                    price_col.diff().over("ticker").alias("diff")
                ).with_columns([
                    # RSI components
                    pl.col("diff").clip(lower_bound=0).rolling_mean(14).over("ticker").alias("avg_gain"),
                    (-pl.col("diff").clip(upper_bound=0)).rolling_mean(14).over("ticker").alias("avg_loss"),
                    
                    # BB components
                    price_col.rolling_mean(20).over("ticker").alias("bb_mean"),
                    price_col.rolling_std(20).over("ticker").alias("bb_std"),
                    
                    # Volume MA
                    vol_col.rolling_mean(20).over("ticker").alias("vol_ma_20")
                ]).with_columns([
                    # RSI
                    (100 - (100 / (1 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).fill_null(50).alias("rsi_14"),
                    
                    # BB Position (Upper Band Cross check)
                    # (Price - Mean) / (2 * Std) -> >1.0 means upper band crossed
                    ((price_col - pl.col("bb_mean")) / (2 * pl.col("bb_std") + 1e-9)).alias("bb_position"),
                    
                    # Volume Ratio
                    (vol_col / (pl.col("vol_ma_20") + 1e-9)).alias("vol_ratio")
                    
                ]).drop(["diff", "avg_gain", "avg_loss", "bb_mean", "bb_std", "vol_ma_20"]) # 임시 컬럼 제거
                
                predictions.append(pred_df)
                
            except Exception as e:
                logger.error(f"Prediction failed for year {year}: {e}")

        if not predictions:
            self._cached_predictions = pl.DataFrame()
        else:
            self._cached_predictions = pl.concat(predictions).sort(["date", "pred_score"], descending=[False, True])
            
        logger.info(f"✅ Predictions & Indicators cached! Rows: {len(self._cached_predictions)}")

    def run_backtest(self, top_k: int = 20, fee: float = 0.002, rebalance_period: int = 5, 
                     stop_loss_k: float = 2.5, market_timing_threshold: float = 0.3,
                     filter_candidates_ratio: float = 2.0,
                     use_rsi_filter: bool = False, rsi_max: float = 80,
                     use_ma_filter: bool = False,
                     use_bollinger_filter: bool = False, bb_position_max: float = 1.0,
                     use_volume_filter: bool = False, min_volume_ratio: float = 0.5,
                     save_plot: bool = False):
        """
        [Enhanced] 백테스팅 엔진 (Hybrid Filtering)
        """
        # 캐싱된 예측값이 없으면 생성
        if not hasattr(self, "_cached_predictions"):
            self.generate_predictions()
            
        combined_df = self._cached_predictions
        if combined_df.is_empty(): return {}

        # 3. 시뮬레이션
        portfolio_results = []
        dates = combined_df["date"].unique().sort()
        current_holdings = [] 
        current_tickers = []
        
        for idx, date in enumerate(dates[:-1]):
            day_df = combined_df.filter(pl.col("date") == date)
            
            # --- Market Timing Check ---
            up_ratio = (day_df.filter(pl.col("log_return_1d") > 0).height / day_df.height) if day_df.height > 0 else 0.5
            market_condition = 1.0 if up_ratio > market_timing_threshold else 0.5 
            
            is_rebal_day = (idx % rebalance_period == 0)
            daily_turnover = 0.0
            
            if is_rebal_day:
                # [Hybrid 2-Stage Selection]
                # 1. Ranking Pool Expansion: 상위 N배수 후보 추출
                pool_size = int(top_k * filter_candidates_ratio)
                candidates_pool = day_df.head(pool_size)
                
                # 2. Techincail Filtering (코인 전략)
                # RSI Filter
                if use_rsi_filter:
                    candidates_pool = candidates_pool.filter(pl.col("rsi_14") < rsi_max)
                    
                # MA Filter
                if use_ma_filter and "sma_60" in candidates_pool.columns:
                    # 정배열 (지수 > 60일선)
                    p_col = "close" if "close" in candidates_pool.columns else "sma_20"
                    candidates_pool = candidates_pool.filter(pl.col(p_col) > pl.col("sma_60"))
                    
                # Bollinger Filter (과열 방지)
                if use_bollinger_filter:
                    # 상단 밴드 돌파 시 매수 보류 (단기 고점 위험)
                    candidates_pool = candidates_pool.filter(pl.col("bb_position") < bb_position_max)
                    
                # Volume Filter (소외주 방지)
                if use_volume_filter:
                    candidates_pool = candidates_pool.filter(pl.col("vol_ratio") > min_volume_ratio)

                # 3. Final Top-K Selection (Ranking 순)
                final_candidates = candidates_pool.head(top_k)["ticker"].to_list()
                
                # 교체 비용 등 계산
                if current_tickers:
                    old_set = set(current_tickers)
                    new_set = set(final_candidates)
                    stay = len(old_set & new_set)
                    # 분모 0 방지
                    denom = len(old_set) if len(old_set) > 0 else 1
                    daily_turnover = (len(old_set) - stay) / denom
                else:
                    daily_turnover = 1.0
                    
                current_tickers = final_candidates
            
            else:
                # ATR Stop Loss Logic
                survivors = []
                for t in current_tickers:
                    row = day_df.filter(pl.col("ticker")==t)
                    if row.is_empty(): 
                        survivors.append(t)
                        continue
                    
                    ret = row["log_return_1d"][0]
                    vol = row["volatility_20d"][0] if row["volatility_20d"][0] is not None else 0.02
                    
                    # [Exit] 차트가 망가졌는지 확인 (코인 스타일 즉시 청산)
                    # 여기서는 간단히 ATR 손절만 적용 (복잡도 관리)
                    if ret < (-stop_loss_k * vol):
                         continue
                    else:
                        survivors.append(t)
                
                # Replenish
                needed = len(current_tickers) - len(survivors)
                if needed > 0:
                    replacements = day_df.filter(~pl.col("ticker").is_in(survivors)).head(needed)["ticker"].to_list()
                    current_tickers = survivors + replacements
                    daily_turnover = needed / top_k
                else:
                    current_tickers = survivors
            
            # --- Calculate Return (Previous code) ---

            # --- Calculate Return ---
            holding_df = day_df.filter(pl.col("ticker").is_in(current_tickers))
            if holding_df.is_empty():
                raw_ret = 0.0
            else:
                raw_ret = holding_df["next_day_ret"].mean()
                if raw_ret is None: raw_ret = 0.0
                
            # [Market Timing Apply] 현금 비중 반영
            # 하락장이면 주식비중 50%, 현금 50% (현금 수익률 0 가정)
            final_daily_ret = (np.exp(raw_ret) - 1) * market_condition
            
            # 비용 차감
            net_ret = final_daily_ret - (daily_turnover * fee * market_condition) # 거래한 만큼만 비용
            
            portfolio_results.append({"date": date, "net_return": net_ret, "turnover": daily_turnover})

        perf_df = pl.DataFrame(portfolio_results)
        metrics = self.calculate_metrics(perf_df)
        
        if save_plot:
            self.save_results(perf_df, metrics, top_k)
            
        return metrics

    def run_grid_search(self):
        """
        Top-K와 Rebalance 주기 최적 조합 탐색
        """
        # 탐색 범위 설정 (현실적이고 효율적인 범위)
        k_options = [5, 10, 15, 20]     # 너무 적으면 분산X, 너무 많으면 수익 희석
        p_options = [1, 3, 5, 10]       # 1일은 비용 과다, 10일은 정보 감쇠
        
        results = []
        print(f"\n🔎 Starting Grid Search (Total {len(k_options)*len(p_options)} combinations)...")
        print(f"{'Top-K':<6} | {'Period':<6} | {'CAGR':<8} | {'MDD':<8} | {'Sharpe':<8} | {'Score'}")
        print("-" * 60)
        
        best_score = -999
        best_params = None
        
        for k in k_options:
            for p in p_options:
                metrics = self.run_backtest(top_k=k, rebalance_period=p, fee=0.002, save_plot=False)
                
                # 문자열 퍼센트 제거 및 실수 변환
                cagr = float(metrics['CAGR'].replace('%',''))
                mdd = float(metrics['MDD'].replace('%',''))
                sharpe = float(metrics['Sharpe Ratio'])
                
                # Custom Score: Sharpe * 10 - |MDD| (안정성 중시)
                score = (sharpe * 10) - (abs(mdd) * 0.5)
                
                print(f"{k:<6} | {p:<6} | {cagr:>7.2f}% | {mdd:>7.2f}% | {sharpe:>8.4f} | {score:>6.2f}")
                
                results.append({
                    "k": k, "p": p, "metrics": metrics, "score": score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = (k, p)

        print("-" * 60)
        print(f"🏆 Best Combination: Top-{best_params[0]}, Period-{best_params[1]} days (Score: {best_score:.2f})")
        
        # 최적 조합으로 상세 리포트 저장
        print("\nSaving best result details...")
        self.run_backtest(top_k=best_params[0], rebalance_period=best_params[1], save_plot=True)

    def calculate_metrics(self, df: pl.DataFrame) -> Dict[str, Any]:
        rets = df["net_return"].to_numpy()
        cum_rets = (1 + rets).cumprod() # numpy 배열이므로 여기는 유지 (rets는 to_numpy() 결과임)
        
        # CAGR (연평균 수익률)
        days = len(df)
        total_ret = cum_rets[-1] - 1
        cagr = (1 + total_ret) ** (252 / days) - 1
        
        # Volatility (연화)
        vol = np.std(rets) * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (cagr / vol) if vol > 0 else 0
        
        # Sortino Ratio (Downside deviation)
        downside_rets = rets[rets < 0]
        downside_vol = np.std(downside_rets) * np.sqrt(252) if len(downside_rets) > 0 else 1e-6
        sortino = (cagr / downside_vol)
        
        # MDD
        peak = np.maximum.accumulate(cum_rets)
        drawdown = (cum_rets - peak) / peak
        mdd = np.min(drawdown)
        
        # Win Rate
        win_rate = np.sum(rets > 0) / len(rets)
        
        # P/L Ratio
        avg_profit = np.mean(rets[rets > 0]) if any(rets > 0) else 0
        avg_loss = abs(np.mean(rets[rets < 0])) if any(rets < 0) else 1e-6
        pl_ratio = avg_profit / avg_loss

        return {
            "Total Return": f"{total_ret*100:.2f}%",
            "CAGR": f"{cagr*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.4f}",
            "Sortino Ratio": f"{sortino:.4f}",
            "MDD": f"{mdd*100:.2f}%",
            "Win Rate": f"{win_rate*100:.2f}%",
            "P/L Ratio": f"{pl_ratio:.4f}",
            "Avg Turnover": f"{df['turnover'].mean()*100:.2f}%"
        }

    def save_results(self, perf_df: pl.DataFrame, metrics: Dict[str, Any], top_k: int):
        # Save CSV
        perf_df.write_csv(self.output_dir / f"backtest_top{top_k}_daily.csv")
        
        # Save Metrics
        with open(self.output_dir / f"backtest_top{top_k}_summary.json", "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
        # Plot Cumulative Return
        plt.figure(figsize=(12, 6))
        dates = perf_df["date"].to_list()
        cum_rets = ((1 + perf_df["net_return"]).cum_prod() - 1) * 100
        
        plt.plot(dates, cum_rets, label=f"YetiRank Top-{top_k} (Net)", color="navy", lw=2)
        plt.title(f"Cumulative Return Simulation (Top-{top_k})", fontsize=14)
        plt.ylabel("Return (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(self.output_dir / f"cumulative_return_top{top_k}.png")
        logger.info(f"✅ Backtest results saved to {self.output_dir}")
        
        print("\n" + "="*40)
        print(f"📊 Backtest Summary (Top-{top_k})")
        print("="*40)
        for k, v in metrics.items():
            print(f"{k:<20}: {v}")
        print("="*40 + "\n")

if __name__ == "__main__":
    backtester = YetiRankBacktester(start_date="20240101", end_date="20251231")
    # Grid Search 모드 실행
    backtester.run_grid_search()
