"""ETF bounded-context settings (primitive fields only)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EtfSettings:
    """Configuration artifact for the ETF switching backtest pipeline."""

    strategy_name: str = "IndexSwitchV1"
    initial_balance: float = 10_000_000.0
    fee_rate: float = 0.00015
    capital_use: float = 0.99
    version: str = "v1"


DEFAULT_ETF = EtfSettings()
