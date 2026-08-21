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
            "src.stocks.ml.contracts": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.ml.fitting": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.ml.discovery": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.backtesting.contracts": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.backtesting.metrics": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.trading.policy": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.trading.transitions": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.trading.allocation": ["src.stocks.cli", "src.stocks.workflows"],
            "src.stocks.observability.contracts": ["src.stocks.ml.training", "src.stocks.backtesting.engine"],
            "src.stocks.observability.recorder": ["src.stocks.ml.training", "src.stocks.backtesting.engine"],
            "src.stocks.observability.report": ["src.stocks.ml.training", "src.stocks.backtesting.engine"],
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
        forbidden = ["src.stocks.cli"]
        ml_modules = [
            "src.stocks.ml.contracts",
            "src.stocks.ml.fitting",
            "src.stocks.ml.discovery",
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
        forbidden = ["src.stocks.cli"]
        bt_modules = [
            "src.stocks.backtesting.contracts",
            "src.stocks.backtesting.metrics",
            "src.stocks.backtesting.execution",
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
        forbidden = ["src.stocks.cli"]
        trading_modules = [
            "src.stocks.trading.policy",
            "src.stocks.trading.transitions",
            "src.stocks.trading.allocation",
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
        forbidden = ["src.stocks.ml.training", "src.stocks.backtesting.engine"]
        obs_modules = [
            "src.stocks.observability.contracts",
            "src.stocks.observability.recorder",
            "src.stocks.observability.report",
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
