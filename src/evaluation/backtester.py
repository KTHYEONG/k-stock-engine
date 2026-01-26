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

    def run_backtest(self, top_k: int = 20, fee: float = 0.002, rebalance_period: int = 5, exit_threshold_ratio: float = 3.0, save_plot: bool = False):
        """
        [Enhanced] 백테스팅 엔진 (ATR Stop-Loss + Market Timing + Grid Search Support)
        """
        # 1. 데이터 로드 (2024~2025)
        # Grid Search 시 데이터 로드 반복을 피하기 위해 캐싱하면 좋으나, 여기서는 안전하게 매번 로드 (속도 최적화 가능)
        if not hasattr(self, "_cached_full_df"):
             self._cached_full_df = self.loader.load_full_data(end_date=self.end_date, sample_ratio=1.0)
             self._cached_feature_names = self.loader.get_feature_names(self._cached_full_df)
        
        full_df = self._cached_full_df
        feature_names = self._cached_feature_names
        
        # 테스트 구간 필터링
        test_df = full_df.filter(
            (pl.col("date") >= pl.lit(self.start_date).str.to_date("%Y%m%d")) & 
            (pl.col("date") <= pl.lit(self.end_date).str.to_date("%Y%m%d"))
        ).sort("date")
        
        if test_df.is_empty(): return {}

        # 2. 모델 예측 (연도별)
        # 예측 결과도 캐싱 가능하지만 파라미터에 따라 달라지지 않으므로 재사용 가능
        # 여기서는 로직 단순화를 위해 매번 수행하되, 실전에서는 분리 권장
        predictions = []
        years = test_df.select(pl.col("date").dt.year()).unique().to_series().to_list()
        
        for year in sorted(years):
            year_data = test_df.filter(pl.col("date").dt.year() == year)
            if year_data.is_empty(): continue
            try:
                model = self.load_model(year)
                X = year_data.select(feature_names).to_pandas()
                scores = model.predict(X)
                predictions.append(year_data.with_columns(pl.Series("pred_score", scores)))
            except: pass

        if not predictions: return {}
        combined_df = pl.concat(predictions).sort(["date", "pred_score"], descending=[False, True])
        
        # Next Day Return 생성
        combined_df = combined_df.with_columns(pl.col("log_return_1d").shift(-1).over("ticker").alias("next_day_ret"))

        # [NEW] Market Timing Signal (KOSPI MA Filter)
        # 데이터 내에 KOSPI('ticker'=='KOSPI')가 있다고 가정하거나, 별도 로드
        # 여기서는 간편하게 '종목들의 상승 비율'이 30% 미만이면 하락장으로 간주하는 "Internal Breadth" 사용
        # (실제 KOSPI 지수 사용이 정확하나 데이터 구조상 복잡성 회피)
        
        # 3. 시뮬레이션
        portfolio_results = []
        dates = combined_df["date"].unique().sort()
        current_holdings = {} # {ticker: buy_price} for ATR calc (여기선 단순화하여 ticker list만 관리하고 ATR은 당일 기준 사용)
        current_tickers = []
        
        # [NEW] ATR 계산 (미리 변동성 준비)
        # volatility_20d는 이미 피처에 있음 (log return std) -> ATR 대용으로 사용 가능
        # ATR Trailing Stop: Price * (1 - Volatility * K)
        
        for idx, date in enumerate(dates[:-1]):
            day_df = combined_df.filter(pl.col("date") == date)
            
            # --- Market Timing Check ---
            # 오늘 상승 종목 비율 계산
            up_ratio = (day_df.filter(pl.col("log_return_1d") > 0).height / day_df.height) if day_df.height > 0 else 0.5
            market_condition = 1.0 if up_ratio > 0.3 else 0.5 # 하락장이면 비중 50%
            
            # --- Rebalancing or Maintenance ---
            is_rebal_day = (idx % rebalance_period == 0)
            daily_turnover = 0.0
            
            if is_rebal_day:
                # Top-K 교체
                candidates = day_df.head(top_k)["ticker"].to_list()
                
                # 교체 비용 계산
                if current_tickers:
                    old_set = set(current_tickers)
                    new_set = set(candidates)
                    stay = len(old_set & new_set)
                    daily_turnover = (len(old_set) - stay) / len(old_set)
                else:
                    daily_turnover = 1.0 # 첫 진입
                    
                current_tickers = candidates
            
            else:
                # [ATR Stop-Loss Logic]
                # 보유 종목 중 변동성이 너무 커서 하락한 놈(손절) 퇴출
                # 여기서는 간략히: 당일 수익률 < -2 * Volatility_20d 인 경우 손절 처리
                # (일별 데이터만 있으므로 장중 대응 불가 -> 종가 기준 퇴출 시뮬레이션)
                survivors = []
                for t in current_tickers:
                    row = day_df.filter(pl.col("ticker")==t)
                    if row.is_empty(): # 데이터 없으면 유지 (불가항력)
                        survivors.append(t)
                        continue
                        
                    ret = row["log_return_1d"][0]
                    vol = row["volatility_20d"][0] if row["volatility_20d"][0] is not None else 0.02
                    
                    # Stop Condition: -2.5 * Volatility (약 -3~5% 변동)
                    if ret < (-2.5 * vol):
                         # 손절 발생 -> 현금화 (Survivors에 포함 X)
                         continue
                    else:
                        survivors.append(t)
                
                # 빈자리 채우기 (Replenish)
                needed = len(current_tickers) - len(survivors)
                if needed > 0:
                    # Survivors 제외 상위 랭커로 충원
                    replacements = day_df.filter(~pl.col("ticker").is_in(survivors)).head(needed)["ticker"].to_list()
                    current_tickers = survivors + replacements
                    daily_turnover = needed / top_k
                else:
                    current_tickers = survivors

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
