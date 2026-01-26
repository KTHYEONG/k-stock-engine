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

    def run_backtest(self, top_k: int = 20, fee: float = 0.002, rebalance_period: int = 5, exit_threshold_ratio: float = 5.0):
        """
        전체 테스트 구간에 대한 백테스팅 실행
        rebalance_period: 모델 예측 주기(5일)에 맞춰 리밸런싱 주기 설정 (기본값: 5일)
        exit_threshold_ratio: 조기 청산 임계값 비율 (Top_K * ratio 순위 밖으로 밀리면 즉시 교체)
        """
        logger.info(f"🚀 Starting Backtest: {self.start_date} ~ {self.end_date} (Top-{top_k}, Period-{rebalance_period}d, ExitThreshold-{exit_threshold_ratio}x)")
        
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
        
        # 3. 포트폴리오 수익률 시뮬레이션 (주기적 리밸런싱 + 조기 청산)
        portfolio_results = []
        dates = combined_df["date"].unique().sort()
        
        current_holdings = [] # 현재 보유 종목
        exit_rank_threshold = int(top_k * exit_threshold_ratio) # 예: 20 * 5 = 100위
        
        # 마지막 날은 수익률 데이터가 없으므로 제외
        for idx, date in enumerate(dates[:-1]):
            day_df = combined_df.filter(pl.col("date") == date)
            
            # 리밸런싱 주기 체크
            is_rebalancing_day = (idx % rebalance_period == 0)
            daily_turnover = 0.0
            
            if is_rebalancing_day:
                # [Regular Rebalancing] Top-K 종목 선정 및 포트폴리오 전면 교체
                top_k_stocks = day_df.head(top_k)
                new_holdings = top_k_stocks["ticker"].to_list()
                
                # Turnover 계산
                if current_holdings:
                    old_set = set(current_holdings)
                    new_set = set(new_holdings)
                    stay_count = len(old_set & new_set)
                    daily_turnover = (len(old_set) - stay_count) / len(old_set) if len(old_set) > 0 else 1.0
                else:
                    daily_turnover = 1.0
                
                current_holdings = new_holdings
                
            else:
                # [Early Exit Logic] 리밸런싱 날이 아닐 때, 랭킹 급락 종목 방어
                if current_holdings:
                    # 1. 현재 보유 종목들의 오늘자 랭킹 확인
                    # day_df는 이미 pred_score 내림차순 정렬 상태 -> row index가 곧 랭킹(0-based)
                    # Ticker 별 랭킹 매핑
                    # 최적화를 위해 상위 (Threshold + α) 까지만 검색하거나 전체를 map으로 변환
                    
                    # 전체 종목에 랭킹 부여
                    day_df_w_rank = day_df.with_columns(
                        pl.int_range(0, pl.len()).alias("daily_rank")
                    )
                    
                    # 현재 보유 종목의 상태 조회
                    holdings_status = day_df_w_rank.filter(pl.col("ticker").is_in(current_holdings))
                    
                    # 2. 퇴출 대상 산출 (랭킹 > Threshold)
                    # 주의: 데이터 누락 등으로 holdings_status에 없을 수도 있음 (보수적 유지)
                    survivors = holdings_status.filter(pl.col("daily_rank") <= exit_rank_threshold)["ticker"].to_list()
                    
                    # 데이터 누락된 종목은 일단 유지 (survivors에 포함되지 않았으므로 아래 로직에서 탈락 처리될 수 있음 -> 누락된건 매도 불가하므로 유지해야함)
                    # holdings_status에 없는 종목(거래정지 등)은 current_holdings에 있었으나 오늘 데이터에 없는 경우임.
                    # 안전을 위해 '오늘 데이터에 있고 + 랭킹 안에 든' 놈들만 survivors로 취급하면, 데이터 없는 놈은 강제 매도됨(가상).
                    # 현실적으로 데이터 없으면 매도 못하므로, missing_tickers는 current_holdings에서 유지시켜야 함.
                    
                    current_set = set(current_holdings)
                    found_set = set(holdings_status["ticker"].to_list())
                    missing_tickers = list(current_set - found_set) # 데이터 없는 종목들
                    
                    final_survivors = survivors + missing_tickers
                    
                    # 3. 빈 자리 채우기 (Replenish)
                    needed_count = len(current_holdings) - len(final_survivors)
                    
                    if needed_count > 0:
                        # 탈락한 종목 수만큼 교체 발생
                        # 당일 Top 종목 중, 이미 보유(생존)한 것 제외하고 상위 N개 선택
                        candidates = day_df.filter(~pl.col("ticker").is_in(final_survivors)).head(needed_count)
                        new_recruits = candidates["ticker"].to_list()
                        
                        # 포트폴리오 갱신
                        current_holdings = final_survivors + new_recruits
                        
                        # Turnover 발생 (교체된 비율)
                        # 여기서는 전체 포트폴리오 크기(top_k) 대비 교체된 종목 수
                        daily_turnover = needed_count / top_k
            
            # 보유 종목의 익일 수익률 계산
            holding_df = day_df.filter(pl.col("ticker").is_in(current_holdings))
            
            if holding_df.is_empty():
                avg_daily_ret_val = 0.0
            else:
                avg_daily_ret_val = holding_df["next_day_ret"].mean()
                
            avg_daily_ret = np.exp(avg_daily_ret_val) - 1 if avg_daily_ret_val is not None else 0.0
            
            # 거래비용 반영
            net_ret = avg_daily_ret - (daily_turnover * fee)
            
            portfolio_results.append({
                "date": date,
                "raw_return": avg_daily_ret,
                "net_return": net_ret,
                "turnover": daily_turnover
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
