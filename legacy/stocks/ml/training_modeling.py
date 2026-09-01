# mypy: ignore-errors
"""Family fitting, challenger selection, refit, holdout scoring."""
from __future__ import annotations

import numpy as np


def _refit_selected(features, targets, *, ridge: float = 1e-8) -> dict[str, object]:
    """Refit a linear selected model deterministically on the supplied panel."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or x.shape[0] == 0:
        raise ValueError("features and targets have incompatible shapes")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("refit inputs must be finite")
    gram = x.T @ x + float(ridge) * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, x.T @ y)
    return {"coef": coef, "train_rows": int(x.shape[0]), "train_mse": float(np.mean((x @ coef - y) ** 2))}
