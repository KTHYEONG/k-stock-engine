"""Stock fills and event/ledger portfolio simulator."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.costs import CostModel
from src.core.instruments import AssetKind
from src.core.portfolio import Position
from src.stocks.trading.allocation_policy import AllocationPolicy


@dataclass(frozen=True, slots=True)
class SimResult:
    equity_curve: list[float]
    trades: list[dict[str, object]]
    final_value: float
    total_return: float


class StockSimulator:
    """Event-driven simulator with explicit cost and settlement inputs."""

    def __init__(self, cost_model: CostModel, initial_cash: float = 100_000_000.0):
        self.cost_model = cost_model
        self.initial_cash = initial_cash

    def simulate(
        self,
        scores: pl.DataFrame,
        policy: AllocationPolicy,
        asset_kind: AssetKind,
        price_frame: pl.DataFrame | None = None,
    ) -> SimResult:
        allocations = policy.targets(scores, asset_kind)
        cash = self.initial_cash
        positions: dict[str, Position] = {}
        equity_curve: list[float] = [cash]
        trades: list[dict[str, object]] = []

        for allocation in allocations:
            instrument = allocation.instrument
            price = self._price_of(price_frame, instrument.instrument_id, allocation.target_value)
            if price <= 0:
                continue
            spend = allocation.target_value * self.initial_cash
            spend = min(spend, cash)
            cost = self.cost_model.round_trip_cost(spend)
            quantity = spend / price
            cash -= spend + cost
            positions[instrument.instrument_id] = Position(
                instrument=instrument, quantity=quantity, average_cost=price
            )
            trades.append(
                {
                    "instrument_id": instrument.instrument_id,
                    "quantity": quantity,
                    "price": price,
                    "value": spend,
                    "cost": cost,
                }
            )
            equity_curve.append(cash + spend)

        final_value = cash + sum(p.quantity * p.average_cost for p in positions.values())
        return SimResult(
            equity_curve=equity_curve,
            trades=trades,
            final_value=final_value,
            total_return=(final_value - self.initial_cash) / self.initial_cash,
        )

    def _price_of(
        self,
        price_frame: pl.DataFrame | None,
        instrument_id: str,
        fallback_value: float,
    ) -> float:
        if price_frame is None or price_frame.is_empty():
            return max(fallback_value, 1.0)
        matched = price_frame.filter(pl.col("instrument_id") == instrument_id)
        if matched.is_empty():
            return max(fallback_value, 1.0)
        return float(matched["close"][0])
