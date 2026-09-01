# mypy: ignore-errors
"""PIT matrix preparation, fold fitting, OOF generation, ranking."""
from __future__ import annotations

import numpy as np


def fit_model_family_oof(
    predictions: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, object]:
    """Aggregate already-fitted fold predictions without crossing fold boundaries."""
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    folds = np.asarray(fold_ids)
    if pred.ndim != 1 or target.ndim != 1 or folds.ndim != 1:
        raise ValueError("predictions, targets and fold_ids must be 1-D")
    if not (pred.size == target.size == folds.size):
        raise ValueError("OOF arrays must have equal length")
    if pred.size == 0 or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
        raise ValueError("OOF arrays must be non-empty and finite")
    fold_scores: dict[object, float] = {}
    for fold in np.unique(folds):
        mask = folds == fold
        fold_scores[fold.item() if hasattr(fold, "item") else fold] = float(
            np.mean((pred[mask] - target[mask]) ** 2)
        )
    centered_pred = pred - float(np.mean(pred))
    centered_target = target - float(np.mean(target))
    denom = float(np.linalg.norm(centered_pred) * np.linalg.norm(centered_target))
    rank = float(np.dot(centered_pred, centered_target) / denom) if denom else 0.0
    return {"oof": pred, "mse": float(np.mean((pred - target) ** 2)), "rank": rank, "fold_scores": fold_scores}
