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

    def run_backtest(self, top_k: int = 20, fee: float = 0.002):
        """전체 테스트 구간에 대한 백테스팅 실행"""
        logger.info(f"🚀 Starting Backtest: {self.start_date} ~ {self.end_date} (Top-{top_k})")
        
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
            
        combined_df = pl.concat(predictions).sort(["date", "pred_score"], descending=[False, True])
        
        # [FIX] 타임라인 정정: T일 예측 후 T+1일 수익률을 얻어야 함.
        # 기존 log_return_1d는 '오늘'의 수익률이므로, 이를 위로 1칸 밀어(shift -1) '내일'의 수익률로 사용.
        combined_df = combined_df.with_columns(
            pl.col("log_return_1d").shift(-1).over("ticker").alias("next_day_ret")
        )
        
        # 3. 일별 포트폴리오 수익률 시뮬레이션
        # target_return_5d는 ln(Close_t+5 / Close_t) 임을 유의
        # 백테스팅의 단순화를 위해 매일 상위 20개를 사고 '5일 뒤 매도'하는 것이 아니라,
        # '매일 상위 20개를 리밸런싱하며 1일치 수익률을 추적'하는 방식으로 구현 (현실적)
        # 1일 수익률: exp(log_return_1d) - 1
        
        portfolio_results = []
        dates = combined_df["date"].unique().sort()
        
        prev_top_tickers = set()
        
        # 마지막 날은 다음 날 수익률을 모르므로 제외
        for date in dates[:-1]:
            day_df = combined_df.filter(pl.col("date") == date)
            # 예측 점수 상위 K개 선정
            top_k_stocks = day_df.head(top_k)
            
            # [DEBUG] 첫 번째 날짜에 대해 상위 종목 점수 확인
            if date == dates[0]:
                avg_score = top_k_stocks["pred_score"].mean()
                logger.info(f"Check Top-K sorting (First Day): Avg Score = {avg_score:.4f} (Should be close to max score)")
                logger.info(f"Top 3 Scores: {top_k_stocks['pred_score'].head(3).to_list()}")
            
            # T일에 선정된 종목의 T+1일 수익률(next_day_ret) 평균 계산
            avg_daily_ret_val = top_k_stocks["next_day_ret"].drop_nulls().mean()
            
            if avg_daily_ret_val is None:
                logger.warning(f"⚠️ No valid return data for {date}. Skipping...")
                continue
                
            avg_daily_ret = np.exp(avg_daily_ret_val) - 1
            
            # Turnover 계산 (종목 교체 비율)
            curr_top_tickers = set(top_k_stocks["ticker"].to_list())
            if prev_top_tickers:
                moved_out = len(prev_top_tickers - curr_top_tickers)
                turnover = moved_out / top_k
            else:
                turnover = 1.0 # 첫날
            
            # 거래비용 반영 (교체 발생 시에만 부과)
            net_ret = avg_daily_ret - (turnover * fee)
            
            portfolio_results.append({
                "date": date,
                "raw_return": avg_daily_ret,
                "net_return": net_ret,
                "turnover": turnover
            })
            prev_top_tickers = curr_top_tickers

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
