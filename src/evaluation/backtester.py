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
        모델 예측값 + 기술적 지표 생성 (데이터 연속성 보장 버전)
        """
        logger.info("⚡ Generating Model Predictions & Indicators (Full Continuity)...")
        
        if not hasattr(self, "_cached_full_df"):
             self._cached_full_df = self.loader.load_full_data(end_date=self.end_date, sample_ratio=1.0)
             self._cached_feature_names = self.loader.get_feature_names(self._cached_full_df)
        
        full_df = self._cached_full_df
        feature_names = self._cached_feature_names

        # --- [1단계] 전체 데이터에 대해 기술 지표 선계산 (단절 방지) ---
        price_col = pl.col("close") if "close" in full_df.columns else (1 + pl.col("log_return_1d")).cumprod().over("ticker")
        vol_col = pl.col("volume") if "volume" in full_df.columns else pl.col("trading_volume") if "trading_volume" in full_df.columns else pl.lit(1.0)
        high_col = pl.col("high") if "high" in full_df.columns else price_col
        low_col = pl.col("low") if "low" in full_df.columns else price_col

        indicator_df = full_df.with_columns(
            price_col.diff().over("ticker").alias("diff"),
            price_col.shift(1).over("ticker").alias("prev_close"),
            high_col.shift(1).over("ticker").alias("prev_high"),
            low_col.shift(1).over("ticker").alias("prev_low"),
            price_col.alias("_price"), # 원본 Close 보존
            vol_col.alias("_vol")
        ).with_columns([
            pl.col("diff").clip(lower_bound=0).alias("gain"),
            (-pl.col("diff").clip(upper_bound=0)).alias("loss"),
            ((high_col + low_col + price_col) / 3).alias("tp")
        ]).with_columns([
            (pl.col("tp") * pl.col("_vol")).alias("mf")
        ]).with_columns([
            pl.when(pl.col("tp") > pl.col("tp").shift(1).over("ticker")).then(pl.col("mf")).otherwise(0).alias("pos_mf"),
            pl.when(pl.col("tp") < pl.col("tp").shift(1).over("ticker")).then(pl.col("mf")).otherwise(0).alias("neg_mf"),
            pl.max_horizontal([high_col - low_col, (high_col - pl.col("prev_close")).abs(), (low_col - pl.col("prev_close")).abs()]).alias("tr"),
            pl.when((high_col - pl.col("prev_high")) > (pl.col("prev_low") - low_col)).then((high_col - pl.col("prev_high")).clip(lower_bound=0)).otherwise(0).alias("p_dm"),
            pl.when((pl.col("prev_low") - low_col) > (high_col - pl.col("prev_high"))).then((pl.col("prev_low") - low_col).clip(lower_bound=0)).otherwise(0).alias("n_dm")
        ]).with_columns([
            pl.col("gain").rolling_mean(14).over("ticker").alias("avg_gain"),
            pl.col("loss").rolling_mean(14).over("ticker").alias("avg_loss"),
            pl.col("pos_mf").rolling_sum(14).over("ticker").alias("pmf_14"),
            pl.col("neg_mf").rolling_sum(14).over("ticker").alias("nmf_14"),
            pl.col("tr").rolling_mean(14).over("ticker").fill_null(0).alias("atr_14"),
            pl.col("p_dm").rolling_mean(14).over("ticker").fill_null(0).alias("pdm_14"),
            pl.col("n_dm").rolling_mean(14).over("ticker").fill_null(0).alias("ndm_14"),
            ((high_col.rolling_max(9).over("ticker") + low_col.rolling_min(9).over("ticker")) / 2).alias("tenkan"),
            ((high_col.rolling_max(26).over("ticker") + low_col.rolling_min(26).over("ticker")) / 2).alias("kijun"),
            ((high_col.rolling_max(52).over("ticker") + low_col.rolling_min(52).over("ticker")) / 2).alias("senkou_b_raw"),
            pl.col("_price").rolling_mean(20).over("ticker").alias("sma_20_raw"),
            pl.col("_price").rolling_mean(60).over("ticker").alias("sma_60_raw"),
            pl.col("_price").rolling_std(20).over("ticker").alias("bb_std"),
            pl.col("_vol").rolling_mean(20).over("ticker").alias("vol_ma_20")
        ]).with_columns([
            (100 - (100 / (1 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).alias("rsi_raw"),
            (100 - (100 / (1 + (pl.col("pmf_14") / (pl.col("nmf_14") + 1e-9))))).alias("mfi_raw"),
            (100 * pl.col("pdm_14") / (pl.col("atr_14") + 1e-9)).alias("p_di"),
            (100 * pl.col("ndm_14") / (pl.col("atr_14") + 1e-9)).alias("n_di"),
            ((pl.col("sma_20_raw") + pl.col("kijun")) / 2).alias("senkou_a_raw"),
            ((pl.col("_price") - pl.col("sma_20_raw")) / (2 * pl.col("bb_std") + 1e-9)).alias("bb_pos_raw")
        ]).with_columns([
            (100 * (pl.col("p_di") - pl.col("n_di")).abs() / (pl.col("p_di") + pl.col("n_di") + 1e-9)).alias("dx")
        ]).with_columns([
            pl.col("dx").rolling_mean(14).over("ticker").fill_null(0).alias("adx_raw"),
            pl.max_horizontal([pl.col("senkou_a_raw"), pl.col("senkou_b_raw")]).alias("cloud_top_raw")
        ]).with_columns([
            # [SHIFT APPLY] 내일의 의사결정에 쓰일 오늘의 지표
            pl.col("rsi_raw").shift(1).over("ticker").fill_null(50).alias("rsi_14"),
            pl.col("mfi_raw").shift(1).over("ticker").fill_null(50).alias("mfi_14"),
            pl.col("adx_raw").shift(1).over("ticker").fill_null(0).alias("adx_14"),
            (pl.col("_price") > pl.col("cloud_top_raw").shift(1).over("ticker")).alias("ichimoku_ok"),
            pl.col("bb_pos_raw").shift(1).over("ticker").fill_null(0).alias("bb_position"),
            (pl.col("_vol") / (pl.col("vol_ma_20").shift(1).over("ticker") + 1e-9)).alias("vol_ratio"),
            pl.col("sma_20_raw").shift(1).over("ticker").alias("sma_20"),
            pl.col("sma_60_raw").shift(1).over("ticker").alias("sma_60"),
        ]).select(["date", "ticker", "rsi_14", "mfi_14", "adx_14", "ichimoku_ok", "bb_position", "vol_ratio", "sma_20", "sma_60"])

        # --- [2단계] 모델 예측 및 지표 병합 ---
        test_df = full_df.filter(
            (pl.col("date") >= pl.lit(self.start_date).str.to_date("%Y%m%d")) & 
            (pl.col("date") <= pl.lit(self.end_date).str.to_date("%Y%m%d"))
        ).sort("date")
        
        predictions = []
        years = test_df.select(pl.col("date").dt.year()).unique().to_series().to_list()
        
        for year in sorted(years):
            year_data = test_df.filter(pl.col("date").dt.year() == year)
            if year_data.is_empty(): continue
            try:
                model = self.load_model(year)
                X = year_data.select(feature_names).to_pandas()
                scores = model.predict(X)
                
                cols_to_keep = ["date", "ticker", "log_return_1d", "volatility_20d"]
                for col in ["close", "open", "high", "low", "volume", "trading_volume"]:
                    if col in year_data.columns: cols_to_keep.append(col)
                
                pred_df = year_data.select(list(set(cols_to_keep))).with_columns(pl.Series("pred_score", scores))
                
                # 지표 병합
                pred_df = pred_df.join(indicator_df, on=["date", "ticker"], how="left")
                pred_df = pred_df.with_columns(
                    pl.col("log_return_1d").shift(-1).over("ticker").alias("next_day_ret")
                )
                predictions.append(pred_df)
                
            except Exception as e:
                logger.error(f"Prediction failed for year {year}: {e}")

        if not predictions:
            self._cached_predictions = pl.DataFrame()
        else:
            self._cached_predictions = pl.concat(predictions).sort(["date", "pred_score"], descending=[False, True])
            
        logger.info(f"✅ Predictions & Indicators cached! Rows: {len(self._cached_predictions)}")
    def run_backtest(self, top_k: int = 20, fee: float = 0.002, rebalance_period: int = 5, 
                     filter_candidates_ratio: float = 2.0,
                     stop_loss_k: float = 2.5, take_profit_k: float = 8.0, max_hold_days: int = 20,
                     market_timing_threshold: float = 0.3,
                     use_rsi_filter: bool = False, rsi_max: float = 80,
                     use_mfi_filter: bool = False, mfi_max: float = 80,
                     use_adx_filter: bool = False, adx_min: float = 20,
                     use_ichimoku_filter: bool = False,
                     use_ma_filter: bool = False,
                     use_bollinger_filter: bool = False, bb_position_max: float = 1.0,
                     use_volume_filter: bool = False, min_volume_ratio: float = 0.5,
                     save_plot: bool = False):
        """
        [Enhanced] 백테스팅 엔진 (Hybrid Filtering + Profit Taking + Time-based Exit)
        """
        # 캐싱된 예측값이 없으면 생성
        if not hasattr(self, "_cached_predictions"):
            self.generate_predictions()
            
        combined_df = self._cached_predictions
        if combined_df.is_empty(): return {}

        # [OPTIMIZATION] 루프 진입 전 필터 조건 미리 계산 (Vectorization)
        # 1. RSI Pass
        if use_rsi_filter:
            combined_df = combined_df.with_columns((pl.col("rsi_14") < rsi_max).alias("is_rsi_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_rsi_ok"))
            
        # 2. MFI Pass
        if use_mfi_filter:
            combined_df = combined_df.with_columns((pl.col("mfi_14") < mfi_max).alias("is_mfi_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_mfi_ok"))
            
        # 3. ADX Pass
        if use_adx_filter:
            combined_df = combined_df.with_columns((pl.col("adx_14") > adx_min).alias("is_adx_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_adx_ok"))
            
        # 4. Ichimoku Pass
        if use_ichimoku_filter:
            combined_df = combined_df.with_columns(pl.col("ichimoku_ok").alias("is_ichimoku_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_ichimoku_ok"))
            
        # 5. MA Pass
        if use_ma_filter and "sma_60" in combined_df.columns:
            p_col = "close" if "close" in combined_df.columns else "sma_20"
            combined_df = combined_df.with_columns((pl.col(p_col) > pl.col("sma_60")).alias("is_ma_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_ma_ok"))
            
        # 6. Bollinger Pass
        if use_bollinger_filter:
            combined_df = combined_df.with_columns((pl.col("bb_position") < bb_position_max).alias("is_bb_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_bb_ok"))
            
        # 7. Volume Pass
        if use_volume_filter:
            combined_df = combined_df.with_columns((pl.col("vol_ratio") > min_volume_ratio).alias("is_vol_ok"))
        else:
            combined_df = combined_df.with_columns(pl.lit(True).alias("is_vol_ok"))
            
        # 최종 합격 여부 (AND 조건)
        combined_df = combined_df.with_columns(
            (pl.col("is_rsi_ok") & pl.col("is_mfi_ok") & pl.col("is_adx_ok") & 
             pl.col("is_ichimoku_ok") & pl.col("is_ma_ok") & pl.col("is_bb_ok") & pl.col("is_vol_ok")).alias("is_buyable")
        )

        # 3. 시뮬레이션
        portfolio_results = []
        dates = combined_df["date"].unique().sort()
        
        # [REFACTORED] current_holdings: {ticker: {"entry_price": float, "hold_days": int}}
        current_holdings = {} 
        
        for idx, date in enumerate(dates[:-1]):
            day_df = combined_df.filter(pl.col("date") == date)
            
            # --- Market Timing Check ---
            up_ratio = (day_df.filter(pl.col("log_return_1d") > 0).height / day_df.height) if day_df.height > 0 else 0.5
            market_condition = 1.0 if up_ratio > market_timing_threshold else 0.5 
            
            is_rebal_day = (idx % rebalance_period == 0)
            daily_turnover = 0.0
            
            if is_rebal_day:
                # [Hybrid 2-Stage Selection]
                pool_size = int(top_k * filter_candidates_ratio)
                candidates_pool = day_df.head(pool_size)
                
                # --- Filtering (Pre-calculated Boolean Mask 사용) ---
                filtered_pool = candidates_pool.filter(pl.col("is_buyable"))

                # 최종 선정
                final_candidates = filtered_pool.head(top_k)["ticker"].to_list()
                
                # --- Turnover & Holdings Update ---
                old_tickers = list(current_holdings.keys())
                new_tickers = final_candidates
                
                # 교체 비용용 turnover
                stay = len(set(old_tickers) & set(new_tickers))
                daily_turnover = (len(old_tickers) - stay) / (len(old_tickers) if len(old_tickers) > 0 else 1)
                
                # [Performance] Price Map 생성 (루프 내 필터링 제거)
                # close가 없으면 sma_20 등을 대용으로 사용
                price_col_name = "close" if "close" in day_df.columns else "sma_20"
                price_map = dict(zip(day_df["ticker"].to_list(), day_df[price_col_name].to_list()))
                
                # Update Holdings: 새로 들어온 놈들은 entry_price 기록
                new_holdings = {}
                for t in new_tickers:
                    if t in current_holdings:
                        # 유지 종목: 기존 정보 승계
                        new_holdings[t] = current_holdings[t]
                        new_holdings[t]["hold_days"] += 1
                    else:
                        # 신규 진입: Map에서 조회 (O(1))
                        entry_p = price_map.get(t, 1.0)
                        new_holdings[t] = {"entry_price": entry_p, "hold_days": 1}
                
                current_holdings = new_holdings
            
            else:
                # [Performance] Price Map 생성
                price_col_name = "close" if "close" in day_df.columns else "sma_20"
                price_map = dict(zip(day_df["ticker"].to_list(), day_df[price_col_name].to_list()))
                
                # --- [Exit Strategy Loop] ---
                survivors = {}
                for t, info in current_holdings.items():
                    # 데이터 존재 여부 확인 (Map에 있는지)
                    if t not in price_map:
                        # 데이터 없으면 일단 유지 (상장폐지 등 특수 상황 제외)
                        survivors[t] = info
                        continue
                        
                    # 지표 추출 - 여기는 어쩔 수 없이 row 접근 필요하지만, 최소화
                    # 수익률/변동성은 별도 맵으로 만들거나 row filter 사용
                    # (정확성을 위해 여기는 filter 유지하되, 리스크 관리 발동 시에만)
                    
                    # [Optimization] Return/Vol Map도 생성
                    # (매일 생성하는 비용 vs 루프 비용 비교 -> 루프가 훨씬 비쌈)
                    # 위에서 한꺼번에 만드는게 나음
                    pass 
                
                # Re-do optimzed loop
                # 지표 맵 생성
                ret_map = dict(zip(day_df["ticker"].to_list(), day_df["log_return_1d"].to_list()))
                vol_map = dict(zip(day_df["ticker"].to_list(), day_df["volatility_20d"].to_list()))
                
                for t, info in current_holdings.items():
                    if t not in price_map:
                        survivors[t] = info
                        continue
                        
                    curr_p = price_map[t]
                    ret = ret_map.get(t, 0.0)
                    vol = vol_map.get(t, 0.02); vol = vol if vol is not None else 0.02
                    
                    # 1. Stop Loss
                    if ret < (-stop_loss_k * vol):
                         continue
                         
                    # 2. Take Profit
                    if (curr_p - info["entry_price"]) / (info["entry_price"] + 1e-9) > (take_profit_k * vol):
                        continue
                        
                    # 3. Time-based Exit
                    if info["hold_days"] > max_hold_days:
                        continue
                        
                    # All pass: Keep
                    info["hold_days"] += 1
                    survivors[t] = info
                
                # 빈자리 충원
                needed = top_k - len(survivors)
                if needed > 0:
                    replacements = day_df.filter(~pl.col("ticker").is_in(list(survivors.keys()))).head(needed)
                    # 여기도 Map 사용
                    rep_tickers = replacements["ticker"].to_list()
                    for t in rep_tickers:
                        entry_p = price_map.get(t, 1.0)
                        survivors[t] = {"entry_price": entry_p, "hold_days": 1}
                        
                    daily_turnover = needed / top_k
                
                current_holdings = survivors
            
            # --- Calculate Return (Previous code) ---

            # --- Calculate Return ---
            holding_df = day_df.filter(pl.col("ticker").is_in(list(current_holdings.keys())))
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

        if not portfolio_results:
            logger.warning("⚠️ No portfolio results generated.")
            return {}

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
                
                if not metrics or 'CAGR' not in metrics:
                    print(f"{k:<6} | {p:<6} | {'N/A':>7} | {'N/A':>7} | {'N/A':>8} | {'N/A'}")
                    continue

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
        if df.is_empty():
            return {
                "Total Return": "0.00%", "CAGR": "0.00%", "Sharpe Ratio": "0.0000",
                "Sortino Ratio": "0.0000", "MDD": "0.00%", "Win Rate": "0.00%",
                "P/L Ratio": "0.0000", "Avg Turnover": "0.00%"
            }

        rets = df["net_return"].to_numpy()
        cum_rets = (1 + rets).cumprod() 
        
        # CAGR (연평균 수익률)
        days = len(df)
        total_ret = cum_rets[-1] - 1 if len(cum_rets) > 0 else 0
        cagr = (1 + total_ret) ** (252 / days) - 1 if days > 0 else 0
        
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
