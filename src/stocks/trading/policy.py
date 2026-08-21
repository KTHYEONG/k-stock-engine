"""Stock trading policy types extracted from portfolio_constructor.py.

``StockRiskPolicy``, ``ExecutionUtility``, and ``SizingMethod`` define the
immutable risk profile and semantic enum types for trading decisions.
"""
from __future__ import annotations

from enum import StrEnum

from src.stocks.trading.portfolio_constructor import (
    CompoundingPolicyConfig,
    stock_risk_policy_fingerprint,
)
from src.stocks.trading.portfolio_constructor import (
    StockRiskPolicy as _StockRiskPolicy,
)


class ExecutionUtility(StrEnum):
    """Semantic execution utility modes."""

    LEGACY_TARGET_INTERPOLATION = "legacy_target_interpolation_v1"
    DELTA_COST_AWARE = "delta_cost_aware_v1"
    SPARSE_HOLD_REPLACE = "sparse_hold_replace_v2"


class SizingMethod(StrEnum):
    """Semantic sizing methods."""

    ALPHA_VOL_SQUARED = "alpha_vol_squared_v1"
    RISK_BALANCED_WATERFILL = "risk_balanced_waterfill_v2"
    CONFIDENCE_MEAN_VARIANCE = "confidence_mean_variance_v1"


class StockRiskPolicy(_StockRiskPolicy):
    """Stable policy type name during decomposition."""


__all__ = [
    "CompoundingPolicyConfig",
    "ExecutionUtility",
    "SizingMethod",
    "StockRiskPolicy",
    "stock_risk_policy_fingerprint",
]
