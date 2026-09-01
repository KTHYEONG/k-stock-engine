"""Facade for portfolio construction: delegates to allocation."""
# mypy: ignore-errors
# ruff: noqa: I001, F401, F403, F811
from __future__ import annotations

import sys

import src.stocks.trading.portfolio_allocation as _alloc
from src.stocks.trading.portfolio_allocation import *  # noqa: F403

construct_target_allocations = _alloc.construct_target_allocations  # noqa: F811
sys.modules[__name__] = _alloc
__all__ = ["construct_target_allocations"]
