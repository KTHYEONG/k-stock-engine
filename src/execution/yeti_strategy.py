from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from src.evaluation.backtester import YetiRankBacktester
from src.execution.yeti_state import PositionState


@dataclass
class LiveTradePlan:
    signal_date: date
    target_weights: Dict[str, float]
    entry_symbols: List[str]
    exit_reasons: Dict[str, str]
    latest_rows: Dict[str, Dict[str, Any]]
    params: Dict[str, Any]
    market_is_risky: bool
    target_exposure: float


class YetiLiveStrategy:
    """Create live trade plans from the same signal logic used by backtester."""

    def __init__(
        self,
        start_date: str = "20240101",
        end_date: str = "20251231",
        model_id: Optional[Union[int, str]] = "latest",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.model_id = model_id
        self.backtester = YetiRankBacktester(
            start_date=start_date,
            end_date=end_date,
            model_id=model_id if model_id != "latest" else None,
        )
        # 백테스트와 정확히 일치하는 파라미터 구성
        self._best_params: Dict[str, Any] = {
            "top_k": 20,
            "atr_multiplier": 2.0,
            "tp_atr_mult": 3.0,
            "max_hold_days": 5,
            "target_ann_vol": 0.15,
        }

    def _ensure_ready(self) -> None:
        if self.backtester._cached_predictions.is_empty():
            self.backtester.generate_predictions()

        if self.backtester._cached_predictions.is_empty():
            raise RuntimeError("No prediction cache. Check feature data/model.")

    @property
    def best_params(self) -> Dict[str, Any]:
        return self._best_params

    def resolve_signal_date(self, signal_date: Optional[str] = None) -> date:
        self._ensure_ready()
        dates = self.backtester._cached_predictions["date"].unique().sort().to_list()
        if not dates:
            raise RuntimeError("No prediction dates available.")

        if signal_date is None:
            return dates[-1]

        target = datetime.strptime(signal_date, "%Y%m%d").date()
        valid = [d for d in dates if d <= target]
        if not valid:
            raise ValueError(f"No available signal data <= {signal_date}")
        return valid[-1]

    def _check_exit_reason(
        self,
        pos: PositionState,
        row: Dict[str, Any],
        params: Dict[str, Any],
        market_is_risky: bool,
        is_out_of_rank: bool,
    ) -> Optional[str]:
        # 하한가 진입 시 청산 보류 (MOO)
        c_open = float(row.get("open") or 0.0)
        c_high = float(row.get("high") or 0.0)
        c_low = float(row.get("low") or 0.0)
        daily_ret = float(row.get("daily_ret") or 0.0)
        
        is_limit_down = (c_open == c_high == c_low) and (daily_ret < 0)
        if is_limit_down:
            return None

        stop_price, take_profit_price = self.get_stop_prices(pos, row, params, market_is_risky)
        max_h = int(params.get("max_hold_days", 5))

        if c_high >= take_profit_price:
            return "TakeProfit"
        elif c_low <= stop_price:
            return "TrailingStop"
        elif int(pos.hold_days) >= max_h:
            return "TimeStop"
        elif is_out_of_rank:
            return "Rebalance"

        return None

    def get_stop_prices(
        self,
        pos: PositionState,
        row: Dict[str, Any],
        params: Dict[str, Any],
        market_is_risky: bool,
    ) -> Tuple[float, float]:
        curr_p = float(row.get("close") or 0.0)
        entry_price = float(pos.entry_price) if float(pos.entry_price) > 0 else curr_p
        
        atr = float(row.get("atr_14") or 0.0)
        atr = max(atr, 1e-9)

        atr_mult = float(params.get("atr_multiplier", 2.0))
        if market_is_risky:
            atr_mult *= 0.5
            
        tp_atr_mult = float(params.get("tp_atr_mult", 3.0))

        stop_price = float(pos.max_price) - (atr_mult * atr)
        take_profit_price = entry_price + (tp_atr_mult * atr)

        return stop_price, take_profit_price

    def _compute_weights(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """백테스트 코드와 완전히 일치하는 Softmax 포지션 사이징"""
        if not candidates:
            return {}

        scores = np.array([c["pred_score"] for c in candidates])
        z_scores = (scores - scores.mean()) / (scores.std() + 1e-8) if len(scores) > 1 else np.array([0.0])
        exp_s = np.exp(z_scores)
        softmax_w = exp_s / exp_s.sum()
        
        return {c["ticker"]: float(w) for c, w in zip(candidates, softmax_w)}

    def generate_trade_plan(
        self,
        current_positions: Dict[str, PositionState],
        signal_date: Optional[str] = None,
    ) -> LiveTradePlan:
        self._ensure_ready()
        params = self.best_params
        df = self.backtester._cached_predictions
        trade_date = self.resolve_signal_date(signal_date)

        # Market Regime 확인 (전일 기준 - 08:50 실행 시점에도 전일 종가 데이터가 최심 데이터임)
        market_is_risky = self.backtester._market_regime.get(trade_date, False)
        ann_vol = self.backtester._market_vol_map.get(trade_date, 0.20)
        target_ann_vol = float(params.get("target_ann_vol", 0.15))
        # 백테스트 일치: Target Exposure
        target_exposure = min(1.0, target_ann_vol / (ann_vol + 1e-8))

        top_k = int(params.get("top_k", 20))
        
        day_df = df.filter(pl.col("date") == trade_date).sort("pred_score", descending=True)
        row_map = {row["ticker"]: row for row in day_df.iter_rows(named=True)}
        
        # Rank Exit 조건 (top_k * 2.5)
        ranks = day_df.select("ticker").to_series().to_list()
        rank_limit = int(top_k * 2.5)
        valid_rank_set = set(ranks[:rank_limit])

        exit_reasons: Dict[str, str] = {}
        survivors: Dict[str, PositionState] = {}

        for ticker, pos in current_positions.items():
            row = row_map.get(ticker)
            if not row:
                exit_reasons[ticker] = "UniverseExit"
                continue
                
            is_out_of_rank = ticker not in valid_rank_set
            reason = self._check_exit_reason(pos, row, params, market_is_risky, is_out_of_rank)
            if reason:
                exit_reasons[ticker] = reason
            else:
                survivors[ticker] = pos

        needed = max(top_k - len(survivors), 0)
        entry_candidates: List[Dict[str, Any]] = []

        # 시장 위험시 신규 진입 차단 (백테스트 일치)
        if not market_is_risky and needed > 0:
            current_sectors = [row_map[t].get("sector", "Unknown") for t in survivors if t in row_map]
            sector_counts = {}
            for s in current_sectors:
                sector_counts[s] = sector_counts.get(s, 0) + 1
                
            max_per_sector = max(1, int(top_k * 0.3))
            
            for r in day_df.iter_rows(named=True):
                t = r["ticker"]
                s = r.get("sector", "Unknown")
                
                # 섹터 분산
                sector_limit_passed = True
                if s != "Unknown" and sector_counts.get(s, 0) >= max_per_sector:
                    sector_limit_passed = False
                    
                if t not in survivors and sector_limit_passed:
                    entry_candidates.append(r)
                    sector_counts[s] = sector_counts.get(s, 0) + 1
                    if len(entry_candidates) >= needed:
                        break

        survivor_candidates = [row_map[t] for t in survivors if t in row_map]
        all_candidates = survivor_candidates + entry_candidates
        
        target_weights = self._compute_weights(all_candidates)
        entry_symbols = [c["ticker"] for c in entry_candidates]

        interested = set(current_positions.keys()) | set(target_weights.keys())
        latest_rows = {ticker: row_map.get(ticker, {}) for ticker in interested}

        return LiveTradePlan(
            signal_date=trade_date,
            target_weights=target_weights,
            entry_symbols=entry_symbols,
            exit_reasons=exit_reasons,
            latest_rows=latest_rows,
            params=params,
            market_is_risky=market_is_risky,
            target_exposure=target_exposure,
        )
