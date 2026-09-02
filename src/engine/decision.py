"""Strategy decision port for unified backtester."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.core.portfolio import PortfolioSnapshot
from src.execution.domain.intents import TradeIntent


@dataclass(frozen=True, slots=True)
class DecisionContext:
    decision_time: datetime
    portfolio: PortfolioSnapshot
    market_snapshot: object

    def __post_init__(self) -> None:
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be aware")


class StrategyDecisionPort(Protocol):
    def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]: ...
