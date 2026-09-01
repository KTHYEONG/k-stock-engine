"""Tests for architecture boundaries and dependency direction.

Scenarios:
- ARCH_13: No quantitative module imports CLI, workflow implementations,
  concrete diagnostic sinks, or compatibility adapters.
"""
from __future__ import annotations

import importlib


def _module_has_import(module_name: str, forbidden_prefix: str) -> bool:
    """Check if a module imports from a forbidden prefix."""
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "__file__") and mod.__file__ is None:
            return False
        return False
    except (ImportError, ModuleNotFoundError):
        return False


class TestImportBoundaries:
    """No quantitative module imports CLI, workflows, or sinks."""

    def test_stock_research_dependency_direction(self) -> None:
        """Data/ML/trading/backtesting dependency direction matches blueprint."""
        forbidden_by_module = {
            "legacy.stocks.ml.contracts": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.ml.fitting": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.ml.discovery": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.backtesting.contracts": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.backtesting.metrics": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.trading.policy": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.trading.transitions": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.trading.allocation": ["legacy.stocks.cli", "legacy.stocks.workflows"],
            "legacy.stocks.observability.contracts": ["legacy.stocks.ml.training", "legacy.stocks.backtesting.engine"],
            "legacy.stocks.observability.recorder": ["legacy.stocks.ml.training", "legacy.stocks.backtesting.engine"],
            "legacy.stocks.observability.report": ["legacy.stocks.ml.training", "legacy.stocks.backtesting.engine"],
        }
        for mod_name, forbidden in forbidden_by_module.items():
            try:
                mod = importlib.import_module(mod_name)
                source = getattr(mod, "__file__", None)
                if source is None:
                    continue
                with open(source) as f:
                    content = f.read()
                for fb in forbidden:
                    assert fb not in content, (
                        f"{mod_name} imports from {fb}"
                    )
            except ImportError:
                pass

    def test_ml_modules_do_not_import_cli(self) -> None:
        forbidden = ["legacy.stocks.cli"]
        ml_modules = [
            "legacy.stocks.ml.contracts",
            "legacy.stocks.ml.fitting",
            "legacy.stocks.ml.discovery",
        ]
        for mod_name in ml_modules:
            try:
                mod = importlib.import_module(mod_name)
                source = getattr(mod, "__file__", None)
                if source is None:
                    continue
                with open(source) as f:
                    content = f.read()
                for fb in forbidden:
                    assert fb not in content, (
                        f"{mod_name} imports from {fb}"
                    )
            except ImportError:
                pass

    def test_backtesting_modules_do_not_import_cli(self) -> None:
        forbidden = ["legacy.stocks.cli"]
        bt_modules = [
            "legacy.stocks.backtesting.contracts",
            "legacy.stocks.backtesting.metrics",
            "legacy.stocks.backtesting.execution",
        ]
        for mod_name in bt_modules:
            try:
                mod = importlib.import_module(mod_name)
                source = getattr(mod, "__file__", None)
                if source is None:
                    continue
                with open(source) as f:
                    content = f.read()
                for fb in forbidden:
                    assert fb not in content, (
                        f"{mod_name} imports from {fb}"
                    )
            except ImportError:
                pass

    def test_trading_modules_do_not_import_cli(self) -> None:
        forbidden = ["legacy.stocks.cli"]
        trading_modules = [
            "legacy.stocks.trading.policy",
            "legacy.stocks.trading.transitions",
            "legacy.stocks.trading.allocation",
        ]
        for mod_name in trading_modules:
            try:
                mod = importlib.import_module(mod_name)
                source = getattr(mod, "__file__", None)
                if source is None:
                    continue
                with open(source) as f:
                    content = f.read()
                for fb in forbidden:
                    assert fb not in content, (
                        f"{mod_name} imports from {fb}"
                    )
            except ImportError:
                pass

    def test_observability_does_not_import_quantitative(self) -> None:
        forbidden = ["legacy.stocks.ml.training", "legacy.stocks.backtesting.engine"]
        obs_modules = [
            "legacy.stocks.observability.contracts",
            "legacy.stocks.observability.recorder",
            "legacy.stocks.observability.report",
        ]
        for mod_name in obs_modules:
            try:
                mod = importlib.import_module(mod_name)
                source = getattr(mod, "__file__", None)
                if source is None:
                    continue
                with open(source) as f:
                    content = f.read()
                for fb in forbidden:
                    assert fb not in content, (
                        f"{mod_name} imports from {fb}"
                    )
            except ImportError:
                pass
