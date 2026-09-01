# mypy: ignore-errors
"""Growth-route projection, certification and capital-route evidence."""
from __future__ import annotations

import math
from collections.abc import Sequence


def _growth_route_projection(log_growth: Sequence[float], initial_cash: float) -> dict[str, float]:
    if not math.isfinite(float(initial_cash)) or float(initial_cash) <= 0.0:
        raise ValueError("initial_cash must be finite and positive")
    values = tuple(float(value) for value in log_growth)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("growth observations must be finite")
    terminal = float(initial_cash) * math.exp(math.fsum(values))
    if not math.isfinite(terminal):
        raise ValueError("growth projection overflow")
    return {"initial_cash": float(initial_cash), "terminal_wealth": terminal, "observed_return": terminal / float(initial_cash) - 1.0}
