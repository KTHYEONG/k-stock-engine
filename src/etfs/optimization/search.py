"""ETF search-space and walk-forward orchestration (configuration artifacts)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Parameter search space for the IndexSwitchV1 strategy."""

    macro_ema_period: tuple[int, int] = (50, 120)
    fast_ema_period: tuple[int, int] = (10, 40)
    roc_n: tuple[int, int] = (1, 5)
    roc_lower: tuple[float, float] = (-0.03, -0.005)
    ibs_entry: tuple[float, float] = (0.10, 0.50)
    ibs_exit: tuple[float, float] = (0.40, 0.95)
    max_hold_days: tuple[int, int] = (3, 20)
    stop_loss_pct: tuple[float, float] = (0.05, 0.15)

    def keys(self) -> list[str]:
        return list(self.__dataclass_fields__.keys())


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One walk-forward split: train on ``train_end``-bounded data, validate after."""

    train_end: object
    validation_start: object
    validation_end: object
    eval_year: object
