import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.execution.kis_client import KisClient
from src.execution.yeti_strategy import LiveTradePlan, YetiLiveStrategy
from src.execution.yeti_state import StateManager
from src.utils.logger import setup_logger


@dataclass
class ExecutionConfig:
    dry_run: bool = True
    order_type: str = "market"
    exchange_code: str = "KRX"
    min_order_krw: float = 50000.0
    max_orders_per_run: int = 30
    sleep_between_orders: float = 0.3  # Rate Limit 안전망 확충

    @classmethod
    def from_env(cls, dry_run_override: Optional[bool] = None) -> "ExecutionConfig":
        dry_env = os.getenv("TRADING_DRY_RUN", "true").strip().lower() in {"1", "true", "yes"}
        return cls(
            dry_run=dry_env if dry_run_override is None else dry_run_override,
            order_type=os.getenv("TRADING_ORDER_TYPE", "market").strip().lower(),
            exchange_code=os.getenv("TRADING_EXCHANGE_CODE", "KRX").strip().upper(),
            min_order_krw=float(os.getenv("TRADING_MIN_ORDER_KRW", "50000")),
            max_orders_per_run=int(os.getenv("TRADING_MAX_ORDERS_PER_RUN", "30")),
            sleep_between_orders=float(os.getenv("TRADING_ORDER_SLEEP", "0.3")),
        )


class YetiLiveTrader:
    def __init__(
        self,
        broker: KisClient,
        strategy: YetiLiveStrategy,
        state: Optional[StateManager] = None,
        config: Optional[ExecutionConfig] = None,
    ):
        self.broker = broker
        self.strategy = strategy
        self.state = state or StateManager()
        self.config = config or ExecutionConfig()
        self.logger = setup_logger("execution.yetirank_live_trader")

    def _account_snapshot(
        self,
    ) -> Tuple[Dict[str, int], float, float, List[Dict[str, Any]]]:
        """Returns (positions, cash, total_equity, raw_output1)."""
        output1, output2 = self.broker.inquire_balance()
        positions = self.broker.parse_positions(output1)
        cash = self.broker.extract_cash(output2)
        total_equity = self.broker.extract_total_equity(output2)
        return positions, cash, total_equity, output1

    def _safe_price(self, ticker: str, plan: LiveTradePlan) -> Tuple[float, Dict[str, Any]]:
        try:
            quote = self.broker.inquire_price(ticker)
            px = self.broker.extract_current_price(quote)
            if px > 0:
                return float(px), quote
        except Exception:
            pass

        row = plan.latest_rows.get(ticker, {})
        from_signal = float(row.get("close") or 0.0)
        return from_signal, {}

    def _build_price_map(self, symbols: List[str], plan: LiveTradePlan) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        px_map: Dict[str, float] = {}
        quote_map: Dict[str, Dict[str, Any]] = {}
        for ticker in symbols:
            px, quote = self._safe_price(ticker, plan)
            if px > 0:
                px_map[ticker] = px
                if quote:
                    quote_map[ticker] = quote
            time.sleep(0.25)  # KIS OpenAPI Rate Limit (최대 초당 5건 반영하여 0.25초 휴식)
        return px_map, quote_map

    def _target_quantities(
        self,
        target_weights: Dict[str, float],
        total_equity: float,
        price_map: Dict[str, float],
        target_exposure: float,
    ) -> Dict[str, int]:
        target_qty: Dict[str, int] = {}
        for ticker, weight in target_weights.items():
            px = float(price_map.get(ticker, 0.0))
            if px <= 0:
                continue
            
            # Target Volatility Scaling: 포트폴리오 노출 비중 즉각 반영
            desired_value = float(total_equity) * float(weight) * target_exposure
            qty = int(desired_value // px)
            
            if qty <= 0 and desired_value >= self.config.min_order_krw:
                qty = 1
            if qty > 0:
                target_qty[ticker] = qty
        return target_qty

    def _place_order(self, side: str, ticker: str, qty: int, price: float, reason: str) -> bool:
        if qty <= 0:
            return False

        if self.config.dry_run:
            self.logger.info(
                "[DRY] %s %s x%d @~%.0f (%s)",
                side.upper(),
                ticker,
                qty,
                price,
                reason,
            )
            self.state.log_order(
                ticker=ticker,
                side=side,
                quantity=qty,
                price=price,
                status="DRY_RUN",
                order_no=None,
                reason=reason,
            )
            return True

        try:
            response = self.broker.place_order(
                symbol=ticker,
                side=side,
                qty=qty,
                price=price if self.config.order_type == "limit" else None,
                order_type=self.config.order_type,
                exchange_code=self.config.exchange_code,
            )
            order_no = self.broker.extract_order_number(response)
            self.logger.info(
                "[LIVE] %s %s x%d -> order_no=%s",
                side.upper(),
                ticker,
                qty,
                order_no or "N/A",
            )
            self.state.log_order(
                ticker=ticker,
                side=side,
                quantity=qty,
                price=price,
                status="SENT",
                order_no=order_no,
                reason=reason,
            )
            return True
        except Exception as exc:
            self.logger.error("Order failed: %s %s x%d (%s)", side, ticker, qty, exc)
            self.state.log_order(
                ticker=ticker,
                side=side,
                quantity=qty,
                price=price,
                status="FAILED",
                order_no=None,
                reason=f"{reason} | {exc}",
            )
            return False

    def run_once(self, signal_date: Optional[str] = None) -> LiveTradePlan:
        resolved_date = self.strategy.resolve_signal_date(signal_date)
        cycle_key = resolved_date.strftime("%Y%m%d")
        self.state.mark_cycle(cycle_key)

        broker_positions, cash, total_equity, raw_output1 = self._account_snapshot()
        sellable_qty: Dict[str, int] = self.broker.parse_sellable_quantities(raw_output1)
        initial_symbols = list(broker_positions.keys())

        bootstrap_price_map, _ = self._build_price_map(initial_symbols, LiveTradePlan(
            signal_date=resolved_date, target_weights={}, entry_symbols=[], 
            exit_reasons={}, latest_rows={}, params={}, market_is_risky=False, target_exposure=1.0
        ))

        self.state.sync_with_broker_positions(
            broker_positions=broker_positions,
            price_map=bootstrap_price_map,
            as_of=cycle_key,
        )
        current_state = self.state.get_open_positions()

        plan = self.strategy.generate_trade_plan(current_state, signal_date=cycle_key)

        symbols_needed = sorted(set(broker_positions.keys()) | set(plan.target_weights.keys()) | set(plan.latest_rows.keys()))
        # 08:50 실행 시, inquire_price는 전일 종가 또는 예상 체결가를 반환함.
        price_map, quote_map = self._build_price_map(symbols_needed, plan)

        if total_equity <= 0:
            total_equity = cash + sum(price_map.get(t, 0.0) * q for t, q in broker_positions.items())

        target_qty = self._target_quantities(plan.target_weights, total_equity, price_map, plan.target_exposure)

        sell_orders = []
        buy_orders = []
        
        # 1. 완전 매도 (Exit logic)
        for ticker in broker_positions:
            if ticker in plan.exit_reasons:
                cur = int(broker_positions.get(ticker, 0))
                px = price_map.get(ticker, 0.0)
                if cur > 0 and px > 0:
                    sell_orders.append((ticker, cur, px))
                    
        # 2. 신규 매수 (Entry logic)
        for ticker in plan.entry_symbols:
            tgt = int(target_qty.get(ticker, 0))
            cur = int(broker_positions.get(ticker, 0))
            diff = tgt - cur
            if diff > 0:
                px = price_map.get(ticker, 0.0)
                if px > 0:
                    buy_orders.append((ticker, diff, px))

        sell_orders.sort(key=lambda x: x[1] * x[2], reverse=True)
        buy_orders.sort(key=lambda x: x[1] * x[2], reverse=True)

        max_orders = self.config.max_orders_per_run
        orders_sent = 0

        for ticker, qty, px in sell_orders:
            if orders_sent >= max_orders:
                break

            # ord_psbl_qty 기반 실제 매도 가능 수량으로 clamp
            avail_sell = sellable_qty.get(ticker, 0)
            if avail_sell <= 0:
                self.logger.warning(
                    "Skip selling %s - ord_psbl_qty=0 (D+2 pending or already sold)", ticker
                )
                continue
            qty = min(qty, avail_sell)

            quote = quote_map.get(ticker, {})
            _, limit_down = KisClient.extract_limit_prices(quote)
            if limit_down > 0 and px <= limit_down:
                self.logger.warning("Skip selling %s - Current price is Limit Down (%.0f)", ticker, px)
                continue

            reason = plan.exit_reasons.get(ticker, "rebalance_sell")
            if self._place_order("sell", ticker, qty, px, reason):
                orders_sent += 1
            time.sleep(self.config.sleep_between_orders)

        # 매도 주문 후 잔고 재조회 (현금 갱신)
        broker_positions, cash, total_equity, raw_output1 = self._account_snapshot()
        sellable_qty = self.broker.parse_sellable_quantities(raw_output1)

        for ticker, qty, px in buy_orders:
            if orders_sent >= max_orders:
                break

            # 종목별 매수가능조회: nrcvb_buy_amt (미수없는 매수금액) 기준
            # 시장가(ORD_DVSN=01) 조회로 증거금율 반영
            try:
                psbl = self.broker.inquire_psbl_order(symbol=ticker, price=0.0, ord_dvsn="01")
                avail_cash = float(psbl.get("nrcvb_buy_amt") or psbl.get("ord_psbl_cash") or 0)
                if avail_cash <= 0:
                    self.logger.warning("Skip buying %s - nrcvb_buy_amt=0", ticker)
                    continue
                # API 반환 수량이 신뢰도 높으면 그대로 사용, 아니면 금액으로 역산
                api_qty_raw = psbl.get("nrcvb_buy_qty") or "0"
                api_qty = int(float(api_qty_raw)) if api_qty_raw else 0
                if api_qty > 0:
                    qty = min(qty, api_qty)
                else:
                    # 시장가 단가 미제공 시 현재가로 역산
                    max_afford = int(avail_cash // max(px, 1.0))
                    qty = min(qty, max_afford)
                time.sleep(0.2)  # rate limit
            except Exception as psbl_exc:
                self.logger.warning("inquire_psbl_order failed for %s: %s. Falling back to local cash.", ticker, psbl_exc)
                max_afford = int(cash // max(px, 1.0))
                qty = min(qty, max_afford)

            if qty <= 0:
                continue

            quote = quote_map.get(ticker, {})
            limit_up, _ = KisClient.extract_limit_prices(quote)
            if limit_up > 0 and px >= limit_up:
                self.logger.warning("Skip buying %s - Current price is Limit Up (%.0f)", ticker, px)
                continue

            est_cost = qty * px
            if est_cost < self.config.min_order_krw:
                continue

            reason = "entry" if ticker in plan.entry_symbols else "rebalance_buy"
            if self._place_order("buy", ticker, qty, px, reason):
                orders_sent += 1
                cash -= est_cost
            time.sleep(self.config.sleep_between_orders)

        broker_positions, _, _, _ = self._account_snapshot()
        final_symbols = sorted(set(broker_positions.keys()) | set(plan.latest_rows.keys()))
        final_price_map, _ = self._build_price_map(final_symbols, plan)

        self.state.sync_with_broker_positions(
            broker_positions=broker_positions,
            price_map=final_price_map,
            as_of=cycle_key,
        )
        for ticker, row in plan.latest_rows.items():
            if ticker in broker_positions and broker_positions[ticker] > 0:
                close_px = float(row.get("close") or 0.0)
                if close_px > 0:
                    self.state.set_max_price(ticker, close_px)

        self.logger.info(
            "Run finished: date=%s dry_run=%s exposure=%.2f orders=%d holdings=%d",
            plan.signal_date,
            self.config.dry_run,
            plan.target_exposure,
            orders_sent,
            len(broker_positions),
        )
        return plan
