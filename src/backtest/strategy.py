"""Strategy protocol and baseline strategies for the backtest engine."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from src.backtest.view import PITView


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    session_idx: int
    cash: int
    nav: int
    positions: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class Targets:
    """Target portfolio weights of decision-session NAV; the remainder is cash."""

    weights: Mapping[int, float]

    def __post_init__(self) -> None:
        total = 0.0
        for weight in self.weights.values():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0.0
            ):
                raise ValueError(f"target weight must be a finite non-negative number, got {weight!r}")
            total += float(weight)
        if total > 1.0 + 1e-9:
            raise ValueError(f"target weights sum to {total}, above 1")


class Strategy(Protocol):
    name: str
    def params(self) -> Mapping[str, str | int | float | bool]: ...
    def is_rebalance(self, view: PITView) -> bool: ...
    def decide(self, view: PITView, portfolio: PortfolioSnapshot) -> Targets: ...


class EqualWeightLiquid(Strategy):
    """Baseline: equal weights over eligible instruments whose decision-day adtv20 ≥ ``min_adtv20_krw``, rebalanced on the first session of each month."""

    def __init__(self, *, min_adtv20_krw: int, max_names: int) -> None:
        if isinstance(min_adtv20_krw, bool) or not isinstance(min_adtv20_krw, int) or min_adtv20_krw < 0:
            raise ValueError(f"min_adtv20_krw must be a non-negative int, got {min_adtv20_krw!r}")
        if isinstance(max_names, bool) or not isinstance(max_names, int) or max_names < 1:
            raise ValueError(f"max_names must be a positive int, got {max_names!r}")
        self._min_adtv20_krw = min_adtv20_krw
        self._max_names = max_names
        self._last_month: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return "equal_weight_liquid"

    def params(self) -> Mapping[str, str | int | float | bool]:
        return {"min_adtv20_krw": self._min_adtv20_krw, "max_names": self._max_names}

    def is_rebalance(self, view: PITView) -> bool:
        month = (view.session_date.year, view.session_date.month)
        if month == self._last_month:
            return False
        self._last_month = month
        return True

    def decide(self, view: PITView, portfolio: PortfolioSnapshot) -> Targets:
        t = view.t
        eligible = view.field("eligible")[t]
        adtv = view.field("adtv20")[t]
        # Instrument ids are ascending, so an ascending index tiebreak is an id tiebreak.
        ranked = sorted(
            (
                n
                for n in range(len(eligible))
                if eligible[n] and math.isfinite(adtv[n]) and adtv[n] >= self._min_adtv20_krw
            ),
            key=lambda n: (-float(adtv[n]), n),
        )
        selected = ranked[: self._max_names]
        if not selected:
            return Targets(weights={})
        weight = 1.0 / len(selected)
        return Targets(weights={n: weight for n in selected})


STRATEGIES: Mapping[str, Callable[..., Strategy]] = {"equal_weight_liquid": EqualWeightLiquid}
