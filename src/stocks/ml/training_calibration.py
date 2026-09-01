# mypy: ignore-errors
"""Causal OOF calibration, seed/ledger construction."""
from __future__ import annotations

import numpy as np


def _causal_oof_calibrate(oof, oof_labels, request=None, horizon_sessions=None, *, seed_ledger=None):
    """Fit a leak-free affine calibration on finite OOF observations only."""
    values = np.asarray(oof, dtype=np.float64)
    labels = np.asarray(oof_labels, dtype=np.float64)
    if values.ndim != 1 or labels.shape != values.shape or values.size == 0:
        raise ValueError("OOF values and labels must be aligned non-empty vectors")
    valid = np.isfinite(values) & np.isfinite(labels)
    if not np.any(valid):
        raise ValueError("OOF calibration requires one finite observation")
    x = values[valid]
    y = labels[valid]
    design = np.column_stack((np.ones(x.size), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"intercept": float(intercept), "slope": float(slope), "finite_rows": int(x.size), "horizon_sessions": horizon_sessions}
