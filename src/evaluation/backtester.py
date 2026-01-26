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

    def run_backtest(self, top_k: int = 20, fee: float = 0.002, rebalance_period: int = 5):
        """
        전체 테스트 구간에 대한 백테스팅 실행
        rebalance_period: 모델 예측 주기(5일)에 맞춰 리밸런싱 주기 설정 (기본값: 5일)
        """
        logger.info(f"🚀 Starting Backtest: {self.start_date} ~ {self.end_date} (Top-{top_k}, Period-{rebalance_period}d)")
        
        # 1. 데이터 로드 (2024~2025)
        full_df = self.loader.load_full_data(end_date=self.end_date, sample_ratio=1.0)
        feature_names = self.loader.get_feature_names(full_df)
        
        # 테스트 구간 필터링 (2024, 2025)
        test_df = full_df.filter(
            (pl.col("date") >= pl.lit(self.start_date).str.to_date("%Y%m%d")) & 
            (pl.col("date") <= pl.lit(self.end_date).str.to_date("%Y%m%d"))
        ).sort("date")
        
        if test_df.is_empty():
            logger.error("No data found for the test period.")
            return

        # 2. 연도별 모델 적용 및 예측 (Expanding Window 대응)
        predictions = []
        years = test_df.select(pl.col("date").dt.year()).unique().to_series().to_list()
        
        for year in sorted(years):
            year_data = test_df.filter(pl.col("date").dt.year() == year)
            if year_data.is_empty(): continue
            
            logger.info(f"Predicting for year {year}...")
            # 2024년은 2024 모델, 2025년은 2025 모델 사용
            try:
                model = self.load_model(year)
                X = year_data.select(feature_names).to_pandas()
                # Score 계산
                scores = model.predict(X)
                
                year_data = year_data.with_columns(
                    pl.Series("pred_score", scores)
                )
                predictions.append(year_data)
            except Exception as e:
                logger.warning(f"Could not load/predict for year {year}: {e}")

        if not predictions:
            return
            
        # [MODIFIED] 데이터 정렬 안정성 강화
        combined_df = pl.concat(predictions).sort(["ticker", "date"])
        
        if "log_return_1d" not in combined_df.columns:
            logger.error("CRITICAL: 'log_return_1d' column missing in data used for backtest.")
            return

        combined_df = combined_df.with_columns(
            pl.col("log_return_1d").shift(-1).over("ticker").alias("next_day_ret")
        )
        
        # 날짜/점수 순 정렬 (랭킹용)
        combined_df = combined_df.sort(["date", "pred_score"], descending=[False, True])
        
        # 3. 포트폴리오 수익률 시뮬레이션 (주기적 리밸런싱 적용)
        portfolio_results = []
        dates = combined_df["date"].unique().sort()
        
        current_holdings = [] # 현재 보유 종목
        
        # 마지막 날은 수익률 데이터가 없으므로 제외
        for idx, date in enumerate(dates[:-1]):
            day_df = combined_df.filter(pl.col("date") == date)
            
            # 리밸런싱 주기 체크
            is_rebalancing_day = (idx % rebalance_period == 0)
            
            if is_rebalancing_day:
                # Top-K 종목 선정 및 포트폴리오 교체
                top_k_stocks = day_df.head(top_k)
                new_holdings = top_k_stocks["ticker"].to_list()
                
                # Turnover 계산
                if current_holdings:
                    old_set = set(current_holdings)
                    new_set = set(new_holdings)
                    stay_count = len(old_set & new_set)
                    turnover = (len(old_set) - stay_count) / len(old_set) if len(old_set) > 0 else 1.0
                else:
                    turnover = 1.0
                
                current_holdings = new_holdings
            else:
                # 포트폴리오 유지
                turnover = 0.0
            
            # 보유 종목의 익일 수익률 계산 (리밸런싱 여부와 무관하게 보유 종목은 가격 변동함)
            holding_df = day_df.filter(pl.col("ticker").is_in(current_holdings))
            
            if holding_df.is_empty():
                avg_daily_ret_val = 0.0
            else:
                avg_daily_ret_val = holding_df["next_day_ret"].mean()
                
            avg_daily_ret = np.exp(avg_daily_ret_val) - 1 if avg_daily_ret_val is not None else 0.0
            
            # 거래비용 반영 (Turnover 발생 시에만)
            net_ret = avg_daily_ret - (turnover * fee)
            
            portfolio_results.append({
                "date": date,
                "raw_return": avg_daily_ret,
                "net_return": net_ret,
                "turnover": turnover
            })

        perf_df = pl.DataFrame(portfolio_results)
        
        # 4. 종합 지표 산출
        metrics = self.calculate_metrics(perf_df)
        self.save_results(perf_df, metrics, top_k)
        
        return perf_df, metrics

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
    backtester.run_backtest(top_k=20)
