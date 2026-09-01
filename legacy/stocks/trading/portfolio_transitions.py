# mypy: ignore-errors
"""Turnover-aware transition selection, sparse de-risk state."""
from __future__ import annotations

import numpy as np


def _select_delta_cost_aware_transition(scores, turnover_costs) -> int:
    utility = np.asarray(scores, dtype=np.float64)
    costs = np.asarray(turnover_costs, dtype=np.float64)
    if utility.ndim != 1 or costs.shape != utility.shape or utility.size == 0:
        raise ValueError("scores and turnover_costs must be aligned non-empty vectors")
    if not np.all(np.isfinite(utility)) or not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("transition inputs must be finite and costs non-negative")
    return int(np.argmax(utility - costs))
