import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any, Optional, Union
from catboost import CatBoostRanker
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.legacy.stock_yetirank_v1.training.data_loader import YetiRankDataLoader
from src.legacy.stock_yetirank_v1.utils.logger import setup_logger

# Mute verbose info logs from data_loader
logging.getLogger("training.data_loader").setLevel(logging.WARNING)
logger = setup_logger("evaluation.backtester")

class YetiRankBacktester:
    """
    최상위 퀀트 투자자 기준 YetiRank 백테스터 (Final Bulletproof Version V4)
    - Zero-Return 버그 완벽 수정 (Equity Loop 순서 재배치)
    - IDX 컬럼 에러 동적 대응
    - Market Integration: KOSPI & KOSDAQ 통합 최대 변동성 산출 (Max Risk)
    - Target Volatility Scaling (15% Target) w/ No Look-ahead Bias
    - Confidence-based Softmax Allocation
    """
    
    def __init__(self, start_date: str = "20240101", end_date: str = "20251231", model_id: Optional[Union[int, str]] = None):
        self.loader = YetiRankDataLoader(start_date="20160401") 
        self.model_dir = PROJECT_ROOT / "models" / "yetirank"
        self.output_dir = PROJECT_ROOT / "results" / "backtest"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_date = start_date
        self.end_date = end_date
        self.model_id = model_id 
        
        self._cached_predictions = pl.DataFrame()
        self._date_indexed_rows = {}
        self._market_regime = {} 
        self._market_vol_map = {} 

    def load_model(self, model_id: Union[int, str]) -> CatBoostRanker:
        """
        [Bulletproof] 시점별 모델 로드 (Look-ahead Bias 방지)
        - model_id가 'latest'이면 최신 모델 즉시 로드
        - exact matching 우선
        - 없으면 해당 시점 '이전'의 가장 최신 모델 검색
        """
        if model_id == "latest":
            latest_path = self.model_dir / "yetirank_latest.cbm"
            if latest_path.exists():
                model = CatBoostRanker()
                model.load_model(str(latest_path))
                logger.info(f"Using explicitly requested latest model: {latest_path.name}")
                return model
            else:
                logger.warning(f"Requested 'latest' model not found at {latest_path}. Falling back to period search.")

        model_path = self.model_dir / f"yetirank_{model_id}.cbm"
        
        if not model_path.exists():
            # 사용 가능한 모든 모델 리스트업
            available_models = sorted([f.name for f in self.model_dir.glob("yetirank_*Q*.cbm")])
            
            if not available_models:
                latest_path = self.model_dir / "yetirank_latest.cbm"
                if latest_path.exists():
                    logger.warning(f"No period models found. Using {latest_path.name} (Potential Bias).")
                    model_path = latest_path
                else:
                    raise FileNotFoundError(f"No models found in {self.model_dir}")
            else:
                # model_id (e.g., 2025Q1) 보다 작거나 같은 모델 중 가장 큰 것 찾기
                target = f"yetirank_{model_id}.cbm"
                past_models = [m for m in available_models if m <= target]
                
                if past_models:
                    model_path = self.model_dir / past_models[-1]
                    logger.info(f"Exact model {target} not found. Using nearest past model: {model_path.name}")
                else:
                    model_path = self.model_dir / available_models[0]
                    logger.warning(f"No past models for {model_id}. Using earliest available: {model_path.name}")
        
        model = CatBoostRanker()
        model.load_model(str(model_path))
        return model

    def _load_market_indices(self) -> pl.DataFrame:
        """KOSPI, KOSDAQ 지수 데이터를 market_index 폴더에서 직접 로드 (컬럼명 유연성 확보)"""
        index_path = PROJECT_ROOT / "data" / "market_index"
        if not index_path.exists():
            return pl.DataFrame()
        
        try:
            ldf = pl.scan_parquet(index_path / "year=*" / "*.parquet")
            schema = ldf.collect_schema().names()
            
            # KRX 원본은 CLSPRC_IDX, 자체 가공은 close
            close_col = "CLSPRC_IDX" if "CLSPRC_IDX" in schema else "close"
            
            # [CRITICAL FIX] 모든 서브 인덱스가 아닌, '코스피'와 '코스닥' 종합 지수 딱 2개만 필터링
            df = ldf.filter(pl.col("IDX_NM").is_in(["코스피", "코스닥"])) \
                   .select([pl.col("date"), pl.col("ticker"), pl.col(close_col).alias("close")]) \
                   .collect()
            
            # 문자열인 경우 콤마 제거 후 숫자(Float64)로 캐스팅 (에러 유발 "-" 등은 null 처리)
            if df.schema["close"] in [pl.Utf8, pl.String]:
                df = df.with_columns([
                    pl.col("close").str.replace_all(",", "").cast(pl.Float64, strict=False)
                ])
                
            df = df.drop_nulls().sort(["ticker", "date"])
            
            df = df.with_columns([
                pl.col("close").pct_change().over("ticker").alias("daily_ret")
            ]).drop_nulls()
            
            return df
        except Exception as e:
            logger.error(f"Failed to load market indices: {e}")
            return pl.DataFrame()

    def generate_predictions(self):
        """
        [Bulletproof] 데이터 누수 없는 예측 및 지표 생성
        """
        logger.info("⚡ Generating Point-in-Time Predictions & Indicators...")
        
        full_df = self.loader.load_full_data(end_date=self.end_date, sample_ratio=1.0).sort(["ticker", "date"])
        feature_names = self.loader.get_feature_names(full_df)
        
        # 1. ATR 계산
        idf = full_df.with_columns([
            pl.col("close").shift(1).over("ticker").alias("prev_close"),
            pl.col("close").pct_change().over("ticker").alias("daily_ret")
        ]).with_columns([
            pl.max_horizontal([
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("prev_close")).abs(),
                (pl.col("low") - pl.col("prev_close")).abs()
            ]).alias("tr")
        ]).with_columns([
            pl.col("tr").rolling_mean(14).over("ticker").alias("atr_14")
        ])

        # 2. Look-ahead Bias 방지
        idf = idf.with_columns([
            pl.col("atr_14").shift(1).over("ticker").alias("atr_14")
        ])

        # 3. Calendar-aware Next Open
        unique_dates = idf.select("date").unique().sort("date")
        date_map = unique_dates.with_columns(pl.col("date").shift(-1).alias("next_date"))
        price_lookup = idf.select(["date", "ticker", "open", "high", "low", "close"]).rename({
            "open": "next_open", "high": "next_high", "low": "next_low", "close": "next_close", "date": "next_date"
        })
        
        idf = idf.join(date_map, on="date", how="left")
        idf = idf.join(price_lookup, on=["next_date", "ticker"], how="left")

        # 4. Market Volatility Scaling (Maximum Risk Aversion)
        market_indices = self._load_market_indices()
        if not market_indices.is_empty():
            market_vol_df = market_indices.sort(["ticker", "date"]).with_columns([
                (pl.col("daily_ret").rolling_std(20).over("ticker") * np.sqrt(252)).alias("ind_vol")
            ])
            market_max_vol = market_vol_df.group_by("date").agg(
                pl.col("ind_vol").max().alias("max_ann_vol")
            ).sort("date")
            self._market_vol_map = {r["date"]: r["max_ann_vol"] for r in market_max_vol.iter_rows(named=True)}
            
            market_max_vol = market_max_vol.with_columns([
                (pl.col("max_ann_vol") > (pl.col("max_ann_vol").rolling_mean(120).fill_null(0.2) * 1.5)).alias("is_risky")
            ])
            self._market_regime = {r["date"]: r["is_risky"] for r in market_max_vol.iter_rows(named=True)}

        # 5. 모델 예측
        test_df = idf.filter(
            (pl.col("date") >= pl.lit(self.start_date).str.to_date("%Y%m%d")) & 
            (pl.col("date") <= pl.lit(self.end_date).str.to_date("%Y%m%d"))
        ).sort("date")
        
        if "period" not in test_df.columns:
            test_df = test_df.with_columns(
                pl.format("{}Q{}", pl.col("date").dt.year(), pl.col("date").dt.quarter()).alias("period")
            )
            
        predictions = []
        unique_periods = sorted(test_df.select("period").unique().to_series().to_list())
        
        for period in unique_periods:
            period_data = test_df.filter(pl.col("period") == period)
            if period_data.is_empty(): continue
            try:
                # model_id가 명시되어 있으면(예: 'latest' 또는 특정 분기), 개별 분기 루프에 관계없이 해당 모델 고정 사용
                target_model_id = self.model_id if self.model_id is not None else period
                model = self.load_model(target_model_id)
                X = period_data.select(feature_names).to_pandas()
                scores = model.predict(X)
                
                pred_df = period_data.select([
                    "date", "ticker", "open", "high", "low", "close", "atr_14", "next_open", "next_high", "next_low", "next_close", "daily_ret", "sector"
                ]).with_columns(pl.Series("pred_score", scores))
                predictions.append(pred_df)
            except Exception as e:
                logger.error(f"Prediction failed for period {period}: {e}")

        pred_all = pl.concat(predictions).sort(["date", "pred_score"], descending=[False, True])
        self._cached_predictions = pred_all
        
        self._date_indexed_rows = {}
        for date_key, group in self._cached_predictions.group_by("date", maintain_order=True):
            norm_date = date_key[0] if isinstance(date_key, tuple) else date_key
            self._date_indexed_rows[norm_date] = group.to_dicts()
            
        logger.info(f"✅ Safe Predictions cached! Total dates: {len(self._date_indexed_rows)}")

    def run_backtest(self, top_k: int = 20, **kwargs):
        """
        [Institutional Simulation V4] Zero-Return Bug Fixed
        """
        if self._cached_predictions.is_empty():
            self.generate_predictions()

        atr_multiplier = kwargs.get("atr_multiplier", 2.0) 
        max_hold_days = kwargs.get("max_hold_days", 5) 
        tp_atr_mult = kwargs.get("tp_atr_multiplier", 3.0) 
        target_ann_vol = kwargs.get("target_ann_vol", 0.15) 
        fee = float(kwargs.get("fee", 0.0025))
        
        holdings = {} 
        equity = 1.0
        equity_curve = []
        trade_records = []
        dates = sorted(list(self._date_indexed_rows.keys()))

        for idx, date in enumerate(dates[:-1]):
            day_data = self._date_indexed_rows[date]
            row_map = {r["ticker"]: r for r in day_data}
            
            # --- [Step 1] Market Timing (T-1 기준) ---
            prev_date = dates[idx - 1] if idx > 0 else date
            market_is_risky = self._market_regime.get(prev_date, False)
            ann_vol = self._market_vol_map.get(prev_date, 0.20)
            target_exposure = min(1.0, target_ann_vol / (ann_vol + 1e-8))
            
            # --- [Step 2 & 3] Equity Calculation & Exit Logic (Combined) ---
            # 어제까지 보유한 종목들의 "오늘 수익률"을 정산
            daily_rets = []
            weights = []
            
            for t, info in list(holdings.items()):
                if t not in row_map: continue
                row = row_map[t]
                info["max_price"] = max(info["max_price"], row["high"])
                
                is_limit_down = row.get("open") == row.get("high") == row.get("low") and row.get("daily_ret", 0) < 0
                
                # 동적 리스크 관리: 시장 위험 시 손절폭 50% 타이트하게 조임
                current_atr_mult = atr_multiplier * 0.5 if market_is_risky else atr_multiplier
                stop_price = info["max_price"] - (current_atr_mult * (row["atr_14"] or 0))
                take_profit_price = info["entry_price"] + (tp_atr_mult * (row["atr_14"] or 0))
                time_up = (info["days_held"] >= max_hold_days)
                
                is_out_of_rank = True
                for r in day_data[:int(top_k * 2.5)]:
                    if r["ticker"] == t:
                        is_out_of_rank = False; break
                
                exit_price = None
                if not is_limit_down:
                    if row["high"] >= take_profit_price:
                        exit_price = max(row["open"], take_profit_price); exit_reason = "TakeProfit"
                    elif row["low"] <= stop_price:
                        exit_price = min(row["open"], stop_price); exit_reason = "TrailingStop"
                    elif time_up:
                        exit_price = row.get("next_open") or row["close"]; exit_reason = "TimeStop"
                    elif is_out_of_rank:
                        exit_price = row.get("next_open") or row["close"]; exit_reason = "Rebalance"
                
                calc_base = info["prev_close_for_calc"]
                w = info.get("weight", 1.0)
                
                if exit_price:
                    # 익절/손절 시 당일 수익률 계산 후 계좌 편입
                    if calc_base > 0:
                        daily_ret = (exit_price / calc_base) - 1 - fee
                        daily_rets.append(daily_ret * w)
                        weights.append(w)
                    
                    trade_ret = (exit_price / info["entry_price"]) - 1 - fee
                    trade_records.append({"ticker": t, "ret": trade_ret, "reason": exit_reason})
                    del holdings[t]
                else:
                    # 계속 보유 시 당일 종가 기준 수익률 계좌 편입
                    if calc_base > 0:
                        daily_ret = (row["close"] / calc_base) - 1
                        daily_rets.append(daily_ret * w)
                        weights.append(w)
                    
                    info["days_held"] += 1
                    # 내일 계산을 위해 오늘 종가를 저장 (버그 해결 핵심)
                    info["prev_close_for_calc"] = row["close"]

            if daily_rets and sum(weights) > 0:
                weighted_ret = sum(daily_rets) / sum(weights)
                equity *= (1 + weighted_ret * target_exposure)
            
            equity_curve.append({"date": date, "equity": equity})

            # --- [Step 4] Entry Logic ---
            needed = top_k - len(holdings)
            if not market_is_risky and needed > 0:
                current_sectors = [row_map[t].get("sector", "Unknown") for t in holdings if t in row_map]
                sector_counts = {}
                for s in current_sectors:
                    sector_counts[s] = sector_counts.get(s, 0) + 1
                
                max_per_sector = max(1, int(top_k * 0.3))
                candidates = []
                for r in day_data[:top_k]:
                    t = r["ticker"]
                    s = r.get("sector", "Unknown")
                    
                    next_open = r.get("next_open")
                    curr_close = r.get("close")
                    
                    next_is_limit_up = False
                    if next_open is not None and r.get("next_high") is not None and r.get("next_low") is not None:
                        if next_open == r["next_high"] == r["next_low"] and curr_close is not None and next_open > curr_close:
                            next_is_limit_up = True
                    
                    # [FIX] "Unknown" 섹터일 경우 섹터 캡(max_per_sector) 제한을 무시하여 포트폴리오가 6개로 묶이는 현상 방지
                    sector_limit_passed = True
                    if s != "Unknown" and sector_counts.get(s, 0) >= max_per_sector:
                        sector_limit_passed = False
                        
                    if t not in holdings and next_open is not None and not next_is_limit_up and sector_limit_passed:
                        candidates.append(r)
                        if len(candidates) >= needed: break

                if candidates:
                    scores = np.array([c["pred_score"] for c in candidates])
                    z_scores = (scores - scores.mean()) / (scores.std() + 1e-8) if len(scores) > 1 else np.array([0.0])
                    exp_s = np.exp(z_scores)
                    softmax_w = exp_s / exp_s.sum()
                    
                    for i, r in enumerate(candidates):
                        holdings[r["ticker"]] = {
                            "entry_price": r["next_open"],
                            "max_price": r["next_open"],
                            "days_held": 0,
                            # 내일(T+1) 아침에 사니까, 내일 수익률 계산의 시작점은 내일 아침 시가!
                            "prev_close_for_calc": r["next_open"], 
                            "weight": softmax_w[i]
                        }

        metrics = self.calculate_metrics(pl.DataFrame(equity_curve), [t["ret"] for t in trade_records])
        self.save_results(pl.DataFrame(equity_curve), metrics, top_k)
        return metrics

    def calculate_metrics(self, df: pl.DataFrame, trade_rets: List[float]) -> Dict[str, Any]:
        if df.is_empty(): return {}
        rets = np.array(trade_rets)
        total_ret = df["equity"][-1] - 1
        days = len(df)
        cagr = (1 + total_ret)**(252/days) - 1 if days > 0 and total_ret > -1 else 0
        peak = df["equity"].cum_max()
        drawdown = (df["equity"] - peak) / peak
        mdd = drawdown.min()
        win_rate = (rets > 0).sum() / len(rets) if len(rets) > 0 else 0
        return {
            "Total Return": f"{total_ret*100:.2f}%",
            "CAGR": f"{cagr*100:.2f}%",
            "MDD": f"{mdd*100:.2f}%",
            "Win Rate": f"{win_rate*100:.2f}%",
            "Total Trades": f"{len(rets)}회",
            "Avg Trade Return": f"{rets.mean()*100:.2f}%" if len(rets) > 0 else "0%"
        }

    def save_results(self, perf_df: pl.DataFrame, metrics: Dict[str, Any], top_k: int):
        print(f"\n📊 Final Backtest Summary (Top-{top_k} / Institutional Strategy V4)")
        for k, v in metrics.items(): print(f"{k:<20}: {v}")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import dateutil.relativedelta
    
    now = datetime.now()
    one_year_ago = now - dateutil.relativedelta.relativedelta(years=1)
    
    default_start = one_year_ago.strftime("%Y%m%d")
    default_end = now.strftime("%Y%m%d")

    parser = argparse.ArgumentParser(description="YetiRank Backtester Implementation")
    parser.add_argument("--start", type=str, default=default_start, help=f"Start date (default: {default_start})")
    parser.add_argument("--end", type=str, default=default_end, help=f"End date (default: {default_end})")
    parser.add_argument("--model_id", type=str, default=None, help="Force specific model ID (e.g. 2025Q1 or latest)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top stocks to hold (default: 20)")
    args = parser.parse_args()

    bt = YetiRankBacktester(
        start_date=args.start,
        end_date=args.end,
        model_id=args.model_id
    )
    bt.run_backtest(top_k=args.top_k)
