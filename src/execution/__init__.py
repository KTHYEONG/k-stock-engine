"""Execution package: public domain and application boundary.

The legacy live-trading Yeti/KIS code is quarantined under ``legacy``; the
package root exports the modern paper-gated boundary only.
"""
from src.execution.domain.intents import TradeIntent
from src.execution.domain.orders import OrderRequest, OrderSide, OrderState, OrderStateRecord
from src.execution.settings import DEFAULT_EXECUTION, ExecutionSettings

__all__ = [
    "DEFAULT_EXECUTION",
    "ExecutionSettings",
    "OrderRequest",
    "OrderSide",
    "OrderState",
    "OrderStateRecord",
    "TradeIntent",
]
