"""Execution package exports with lazy loading."""

from importlib import import_module
from typing import Any

__all__ = [
    "KisClient",
    "KisCredentials",
    "LiveTradePlan",
    "YetiLiveStrategy",
    "ExecutionConfig",
    "YetiLiveTrader",
    "PositionState",
    "StateManager",
]

_EXPORT_MAP = {
    "KisClient": ("src.execution.kis_client", "KisClient"),
    "KisCredentials": ("src.execution.kis_client", "KisCredentials"),
    "LiveTradePlan": ("src.execution.yeti_strategy", "LiveTradePlan"),
    "YetiLiveStrategy": ("src.execution.yeti_strategy", "YetiLiveStrategy"),
    "ExecutionConfig": ("src.execution.yeti_trader", "ExecutionConfig"),
    "YetiLiveTrader": ("src.execution.yeti_trader", "YetiLiveTrader"),
    "PositionState": ("src.execution.yeti_state", "PositionState"),
    "StateManager": ("src.execution.yeti_state", "StateManager"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
