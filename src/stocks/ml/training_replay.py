# mypy: ignore-errors
"""Execution replay context, cost construction, batched replay."""
from __future__ import annotations


def _replay_costs_batch(
    gross_returns, entry_cost_rates, exit_cost_rates
) -> dict[str, object]:
    """Apply entry/exit schedules elementwise and reject non-finite evidence."""
    import numpy as np

    gross = np.asarray(gross_returns, dtype=np.float64)
    entry = np.asarray(entry_cost_rates, dtype=np.float64)
    exit_ = np.asarray(exit_cost_rates, dtype=np.float64)
    if gross.ndim != 1 or entry.shape != gross.shape or exit_.shape != gross.shape:
        raise ValueError("replay cost arrays must be aligned 1-D arrays")
    if not np.all(np.isfinite(np.concatenate((gross, entry, exit_)))):
        raise ValueError("replay cost arrays must be finite")
    if np.any(entry < 0.0) or np.any(exit_ < 0.0) or np.any(entry >= 1.0) or np.any(exit_ >= 1.0):
        raise ValueError("replay costs must be in [0, 1)")
    net = (1.0 + gross) * (1.0 - entry) * (1.0 - exit_) - 1.0
    return {"net_returns": net, "gross_total": float(np.prod(1.0 + gross) - 1.0), "net_total": float(np.prod(1.0 + net) - 1.0)}
