"""Canonical chronological scenario execution state for backtests.

``PreparedExecutionState`` owns one base or stress replay's mutable
portfolio/settlement/order/ledger state. Base and stress scenarios each advance
their own independent state over the shared immutable prepared market, so no
positions, pending orders, settlement, or ledger rows ever cross a scenario or
segment boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.portfolio import PortfolioSnapshot, Position
from src.stocks.backtesting.contracts import BacktestLedgerRow, BacktestTrade


@dataclass(slots=True)
class PreparedExecutionState:
    """Mutable base or stress replay state advanced in one paired session loop.

    ``settlements`` is keyed by the *due session index* and released exactly
    once at that session; ``pending_orders`` carries deferred intents whose
    remaining entry-delay budget survives across sessions; ``trades`` and
    ``ledger`` accumulate the scenario's own immutable audit trail.
    """

    account_snapshot_id: str
    settled_cash: float
    unsettled_cash: float
    accrued_costs: float
    positions: dict[str, int]
    settlements: dict[int, float]
    pending_orders: list[dict[str, object]]
    trades: list[BacktestTrade]
    ledger: list[BacktestLedgerRow]
    last_close: dict[str, float]
    attempted_orders: int
    base_positions: tuple[Position, ...]

    @classmethod
    def from_initial(cls, initial_portfolio: PortfolioSnapshot) -> PreparedExecutionState:
        """Fresh state from an immutable portfolio snapshot boundary."""
        return cls(
            account_snapshot_id=initial_portfolio.account_snapshot_id,
            settled_cash=initial_portfolio.settled_cash,
            unsettled_cash=initial_portfolio.unsettled_cash,
            accrued_costs=0.0,
            positions={
                p.instrument.instrument_id: int(p.quantity)
                for p in initial_portfolio.positions
                if p.quantity > 0
            },
            settlements={},
            pending_orders=[],
            trades=[],
            ledger=[],
            last_close={},
            attempted_orders=0,
            base_positions=tuple(initial_portfolio.positions),
        )

    def reset_for_segment(self) -> None:
        """Drop all mutable replay state so a new segment starts clean."""
        self.settlements.clear()
        self.pending_orders.clear()
        self.trades.clear()
        self.ledger.clear()
        self.last_close.clear()


class ScenarioExecutor:
    """Compatibility wrapper driving one backtester's run entry point."""

    def __init__(self, backtester: object) -> None:
        self._backtester = backtester

    def run(self, *args: object, **kwargs: object) -> object:
        """Delegate to the underlying backtester's run method."""
        return self._backtester.run(*args, **kwargs)  # type: ignore[attr-defined]
