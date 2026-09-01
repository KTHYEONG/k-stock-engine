# mypy: ignore-errors
"""Economic gates, covariance/volatility constraints, compounding scaling."""
from __future__ import annotations

import numpy as np


def _apply_constraints(weights, *, max_single_weight: float, max_exposure: float) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a finite vector")
    if not 0.0 < max_single_weight <= 1.0 or not 0.0 < max_exposure <= 1.0:
        raise ValueError("weight limits must be in (0, 1]")
    bounded = np.clip(values, -max_single_weight, max_single_weight)
    exposure = float(np.sum(np.abs(bounded)))
    if exposure > max_exposure and exposure > 0.0:
        bounded = bounded * (max_exposure / exposure)
    return bounded
