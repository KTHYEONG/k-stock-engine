# mypy: ignore-errors
"""Reference weighting, risk-balanced waterfill and deterministic target weights."""
from __future__ import annotations

import numpy as np


def _risk_balanced_waterfill(weights, *, cap: float, total: float = 1.0) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a finite non-empty vector")
    if cap <= 0.0 or total < 0.0 or not np.isfinite(cap + total):
        raise ValueError("cap must be positive and total non-negative")
    positive = np.maximum(values, 0.0)
    if not np.any(positive) or total == 0.0:
        return np.zeros_like(positive)
    result = positive / float(np.sum(positive)) * float(total)
    for _ in range(values.size):
        over = result > cap
        if not np.any(over):
            break
        excess = float(np.sum(result[over] - cap))
        result[over] = cap
        free = ~over
        if not np.any(free):
            break
        result[free] += excess * positive[free] / float(np.sum(positive[free]))
    return result
