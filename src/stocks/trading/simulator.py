"""Multi-session stock fills and event/ledger portfolio simulator.

The simulator iterates ordered KRX sessions and maintains cash, unsettled cash,
positions, orders, fills, costs, and equity as a reconciled ledger. It never
invents a price: missing open, halt, zero volume, or a limit-locked state
produces an unfilled order with a reason code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, default_base_schedule
from src.core.instruments import AssetKind
from src.stocks.research.metrics import max_drawdown
from src.stocks.trading.allocation_policy import AllocationPolicy, StockTargetWeight

REQUIRED_PANEL_COLUMNS = (
    "instrument_id",
    "session",
    "open",
    "close",
    "volume",
    "trading_value",
    "pred_score",
)


def _num(value: object) -> float:
    """Coerce a ledger/trade value to float, treating None as zero."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"non-numeric ledger value: {value!r}")


@dataclass(frozen=True, slots=True)
class SimResult:
    """Outcome of a multi-session simulation: ledger plus derived metrics."""

    ledger: list[dict[str, object]]
    trades: list[dict[str, object]]
    final_value: float
    total_return: float
    metrics: dict[str, float]
    stress_metrics: dict[str, float] | None = None
    stress_final_value: float | None = None

    @property
    def equity_curve(self) -> list[float]:
        return [_num(row["equity"]) for row in self.ledger]

    @property
    def total_trades(self) -> int:
        return len(self.trades)


class StockSimulator:
    """Event-driven simulator with explicit cost, settlement, and capacity inputs."""

    def __init__(
        self,
        cost_schedule: CostSchedule | None = None,
        initial_cash: float = 100_000_000.0,
        settlement_lag_sessions: int | None = None,
        adtv_participation_limit: float = 0.01,
        adtv_window: int = 20,
        stress_schedule: CostSchedule | None = None,
        seed: int = 42,
    ):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if adtv_participation_limit < 0:
            raise ValueError("adtv_participation_limit must be non-negative")
        if adtv_window < 1:
            raise ValueError("adtv_window must be positive")
        self.cost_schedule = cost_schedule or default_base_schedule()
        self.initial_cash = initial_cash
        self.settlement_lag_sessions = settlement_lag_sessions
        self.adtv_participation_limit = adtv_participation_limit
        self.adtv_window = adtv_window
        self.stress_schedule = stress_schedule
        self.seed = seed

    def simulate(
        self,
        panel: pl.DataFrame,
        policy: AllocationPolicy,
        asset_kind: AssetKind,
        rebalance_sessions: tuple[int, ...] | None = None,
    ) -> SimResult:
        """Simulate the panel over ordered sessions and reconcile every session."""
        missing = [c for c in REQUIRED_PANEL_COLUMNS if c not in panel.columns]
        if missing:
            raise ValueError(f"panel must carry {', '.join(missing)}")

        decision_set = (
            None if rebalance_sessions is None else {int(s) for s in rebalance_sessions}
        )
        ledger, trades = self._run_ledger(
            panel, policy, self.cost_schedule, decision_set
        )
        if not ledger:
            raise ValueError("simulation produced no ledger rows")
        final_value = _num(ledger[-1]["equity"])
        total_return = (final_value - self.initial_cash) / self.initial_cash
        metrics = self._metrics(ledger, trades)
        stress_metrics: dict[str, float] | None = None
        stress_final_value: float | None = None
        if self.stress_schedule is not None:
            stress_ledger, stress_trades = self._run_ledger(
                panel, policy, self.stress_schedule, decision_set
            )
            stress_metrics = self._metrics(stress_ledger, stress_trades)
            stress_final_value = _num(stress_ledger[-1]["equity"])
        return SimResult(
            ledger=ledger,
            trades=trades,
            final_value=final_value,
            total_return=total_return,
            metrics=metrics,
            stress_metrics=stress_metrics,
            stress_final_value=stress_final_value,
        )

    def _run_ledger(
        self,
        panel: pl.DataFrame,
        policy: AllocationPolicy,
        schedule: CostSchedule,
        decision_set: set[int] | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        df = panel.sort(["session", "instrument_id"]).with_columns(
            pl.col("trading_value")
            .rolling_mean(self.adtv_window, min_samples=1)
            .over("instrument_id")
            .alias("adtv"),
            pl.col("close")
            .rolling_std(20, min_samples=2)
            .over("instrument_id")
            .alias("volatility"),
        )
        sessions = sorted(df["session"].unique().to_list())
        by_session = dict(
            zip(sessions, df.sort("session").partition_by(["session"]), strict=True)
        )
        first_session_time = (
            datetime.combine(sessions[0], time.min, tzinfo=UTC)
            if isinstance(sessions[0], date) and not isinstance(sessions[0], datetime)
            else sessions[0]
        )
        settlement_lag = (
            self.settlement_lag_sessions
            if self.settlement_lag_sessions is not None
            else schedule.cost_for(first_session_time).settlement_days
        )

        settled_cash = self.initial_cash
        unsettled_cash = 0.0
        accrued_costs = 0.0
        positions: dict[str, int] = {}
        pending: list[StockTargetWeight] = []
        pending_equity = 0.0
        settlements: dict[int, float] = {}
        trades: list[dict[str, object]] = []
        ledger: list[dict[str, object]] = []
        last_close: dict[str, float] = {}

        for i, session in enumerate(sessions):
            rows = by_session[session]
            if settlement_lag > 0:
                due = settlements.pop(i - settlement_lag, 0.0)
                settled_cash += due
                unsettled_cash -= due

            if pending:
                settled_cash, unsettled_cash, accrued_costs, settlements = self._execute_orders(
                    rows,
                    pending,
                    positions,
                    settled_cash,
                    unsettled_cash,
                    accrued_costs,
                    settlements,
                    trades,
                    schedule,
                    settlement_lag,
                    i,
                    pending_equity,
                )

            positions_value = self._mark_positions(rows, positions, last_close)
            equity = (
                settled_cash
                + unsettled_cash
                + positions_value
                - accrued_costs
            )
            ledger.append(
                {
                    "session": session,
                    "settled_cash": settled_cash,
                    "unsettled_cash": unsettled_cash,
                    "positions_value": positions_value,
                    "accrued_costs": accrued_costs,
                    "equity": equity,
                }
            )

            if (decision_set is None or i in decision_set) and i + 1 < len(sessions):
                selected = self._compute_targets(rows, policy)
                selected_ids = {target.instrument_id for target in selected}
                pending = selected + [
                    StockTargetWeight(
                        instrument_id=instrument_id,
                        target_weight=0.0,
                        reason="rebalance-exit",
                    )
                    for instrument_id in sorted(positions)
                    if instrument_id not in selected_ids
                ]
                pending_equity = equity

        return ledger, trades

    @staticmethod
    def _compute_targets(
        rows: pl.DataFrame,
        policy: AllocationPolicy,
    ) -> list[StockTargetWeight]:
        decision = rows.filter(
            pl.col("pred_score").is_not_null()
            & pl.col("close").is_not_null()
            & pl.col("adtv").is_not_null()
            & pl.col("volatility").is_not_null()
        )
        if decision.is_empty():
            return []
        return policy.targets(
            decision,
            AssetKind.STOCK,
            instrument_column="instrument_id",
            score_column="pred_score",
        )

    def _execute_orders(
        self,
        rows: pl.DataFrame,
        pending: list[StockTargetWeight],
        positions: dict[str, int],
        settled_cash: float,
        unsettled_cash: float,
        accrued_costs: float,
        settlements: dict[int, float],
        trades: list[dict[str, object]],
        schedule: CostSchedule,
        settlement_lag: int,
        session_index: int,
        decision_equity: float,
    ) -> tuple[float, float, float, dict[int, float]]:
        row_by_instrument = {
            str(r["instrument_id"]): r for r in rows.to_dicts()
        }
        for target in pending:
            instrument_id = target.instrument_id
            weight = target.target_weight
            row = row_by_instrument.get(instrument_id)
            if row is None:
                trades.append(self._unfilled(instrument_id, rows["session"][0], "no-session-row"))
                continue
            open_price = row.get("open")
            volume = row.get("volume")
            if open_price is None or _num(open_price) <= 0:
                trades.append(self._unfilled(instrument_id, rows["session"][0], "missing-open"))
                continue
            if volume is None or _num(volume) <= 0:
                trades.append(self._unfilled(instrument_id, rows["session"][0], "zero-volume"))
                continue
            if row.get("limit_locked") is True:
                trades.append(self._unfilled(instrument_id, rows["session"][0], "limit-locked"))
                continue

            cost_point = schedule.cost_for(rows["session"][0])
            price = _num(open_price)
            notional = weight * decision_equity
            if weight > 0 and self.adtv_participation_limit > 0:
                adtv = _num(row.get("adtv"))
                if adtv <= 0:
                    trades.append(
                        self._unfilled(instrument_id, rows["session"][0], "no-capacity-data")
                    )
                    continue
                max_notional = self.adtv_participation_limit * adtv
                notional = min(notional, max_notional)
            current_qty = positions.get(instrument_id, 0)
            target_qty = 0 if weight <= 0 else int(notional // price)
            delta = target_qty - current_qty
            if delta == 0:
                continue
            if delta > 0:
                buy_cost_rate = cost_point.commission_rate + cost_point.slippage_bps / 10_000.0
                max_affordable = int(
                    settled_cash
                    / (price * (1.0 + buy_cost_rate))
                    if settled_cash > 0
                    else 0
                )
                delta = min(delta, max_affordable)
                if delta <= 0:
                    trades.append(
                        self._unfilled(instrument_id, rows["session"][0], "insufficient-cash")
                    )
                    continue
                gross = delta * price
                cost = gross * buy_cost_rate
                settled_cash -= gross
                accrued_costs += cost
                positions[instrument_id] = current_qty + delta
                trades.append(self._fill(instrument_id, rows["session"][0], delta, price, "buy", gross, cost))
            else:
                sell_qty = -delta
                sell_cost_rate = (
                    cost_point.commission_rate
                    + cost_point.tax_rate
                    + cost_point.slippage_bps / 10_000.0
                )
                gross = sell_qty * price
                cost = gross * sell_cost_rate
                positions[instrument_id] = current_qty - sell_qty
                accrued_costs += cost
                unsettled_cash += gross
                settlements[session_index + settlement_lag] = (
                    settlements.get(session_index + settlement_lag, 0.0) + gross
                )
                trades.append(self._fill(instrument_id, rows["session"][0], sell_qty, price, "sell", gross, cost))
        return settled_cash, unsettled_cash, accrued_costs, settlements

    @staticmethod
    def _fill(
        instrument_id: str,
        session: object,
        quantity: int,
        price: float,
        side: str,
        gross: float,
        cost: float,
    ) -> dict[str, object]:
        return {
            "instrument_id": instrument_id,
            "session": session,
            "quantity": quantity,
            "price": price,
            "side": side,
            "gross": gross,
            "cost": cost,
        }

    @staticmethod
    def _unfilled(instrument_id: str, session: object, reason: str) -> dict[str, object]:
        return {
            "instrument_id": instrument_id,
            "session": session,
            "quantity": 0,
            "price": None,
            "side": None,
            "reason": reason,
        }

    @staticmethod
    def _mark_positions(
        rows: pl.DataFrame,
        positions: dict[str, int],
        last_close: dict[str, float],
    ) -> float:
        for r in rows.to_dicts():
            close = r.get("close")
            if close is not None:
                last_close[str(r["instrument_id"])] = float(close)
        total = 0.0
        for instrument_id, qty in positions.items():
            if qty == 0:
                continue
            if instrument_id not in last_close:
                raise RuntimeError(
                    f"no observed price for held instrument {instrument_id!r}"
                )
            total += qty * last_close[instrument_id]
        return total

    @staticmethod
    def _metrics(
        ledger: list[dict[str, object]],
        trades: list[dict[str, object]],
    ) -> dict[str, float]:
        equity = np.asarray([_num(row["equity"]) for row in ledger], dtype=float)
        if equity.size < 2 or equity[0] <= 0:
            return {
                "cagr": 0.0,
                "annualized_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "cost_drag": 0.0,
                "exposure": 0.0,
            }
        returns = np.diff(equity) / equity[:-1]
        mean_equity = float(np.mean(equity))
        cagr = (equity[-1] / equity[0]) ** (252.0 / equity.size) - 1.0
        volatility = float(np.std(returns, ddof=0) * np.sqrt(252.0))
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=0)) * np.sqrt(252.0)
            if np.std(returns, ddof=0) > 0
            else 0.0
        )
        drawdown = max_drawdown(equity.tolist())
        gross_notional = sum(_num(t.get("gross")) for t in trades)
        total_cost = _num(ledger[-1].get("accrued_costs"))
        return {
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "turnover": gross_notional / mean_equity,
            "cost_drag": total_cost / mean_equity,
            "exposure": float(np.mean([_num(row["positions_value"]) for row in ledger])) / mean_equity,
        }
