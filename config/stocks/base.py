"""Versioned, validated stock configuration. No asset policy lives in
``config/base.py`` globals."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StockAlphaConfig:
    """Configuration artifact for the stock alpha baseline pipeline."""

    feature_set: str = "stock_alpha_v1"
    label_definition: str = "fwd_ret_5d"
    label_horizon_sessions: int = 5
    n_folds: int = 3
    embargo_sessions: int = 5
    top_k: int = 5
    max_single_weight: float = 0.2
    max_exposure: float = 1.0
    version: str = "v1"


DEFAULT_STOCK_ALPHA = StockAlphaConfig()
