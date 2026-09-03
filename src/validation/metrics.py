"""Ledger-derived metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from src.core.ledger import LedgerNav


@dataclass(frozen=True, slots=True)
class LedgerMetrics:
    annualized_log_growth: float
    cagr: float
    annualized_volatility: float
    max_drawdown: float
    calmar: float | None
    sortino: float | None


def calculate_ledger_metrics(
    daily_nav: tuple[LedgerNav, ...],
    *,
    sessions_per_year: int = 252,
) -> LedgerMetrics:

    if not isinstance(daily_nav, tuple):
        raise ValueError("daily_nav must be tuple")
    if len(daily_nav) < 2:
        raise ValueError("daily_nav must have at least two marks")
    if isinstance(sessions_per_year, bool) or not isinstance(sessions_per_year, int):
        raise ValueError("sessions_per_year must be integer")
    if sessions_per_year <= 0:
        raise ValueError("sessions_per_year must be positive")

    # Validate marks
    for nav in daily_nav:
        if not isinstance(nav, LedgerNav):
            raise ValueError("daily_nav must contain LedgerNav")
        if not isinstance(nav.as_of, datetime) or nav.as_of.tzinfo is None:
            raise ValueError("LedgerNav as_of must be aware")
        if isinstance(nav.nav, bool) or not isinstance(nav.nav, (int, float)):
            raise ValueError("LedgerNav nav must be finite")
        if not math.isfinite(float(nav.nav)) or float(nav.nav) <= 0:
            raise ValueError("LedgerNav nav must be positive finite")

    # Strictly time-increasing
    for i in range(1, len(daily_nav)):
        if daily_nav[i].as_of <= daily_nav[i - 1].as_of:
            raise ValueError("daily_nav must be strictly time-increasing")

    # Log returns
    log_returns: list[float] = []
    for i in range(1, len(daily_nav)):
        prev = float(daily_nav[i - 1].nav)
        cur = float(daily_nav[i].nav)
        if prev <= 0 or cur <= 0:
            raise ValueError("nav must be positive")
        r = math.log(cur / prev)
        if not math.isfinite(r):
            raise ValueError("log return must be finite")
        log_returns.append(r)

    n = len(log_returns)
    sum_log = sum(log_returns)
    annualized_log_growth = float(sessions_per_year) / float(n) * float(sum_log)
    if not math.isfinite(annualized_log_growth):
        raise ValueError("annualized_log_growth must be finite")
    cagr = math.expm1(annualized_log_growth)
    if not math.isfinite(cagr):
        raise ValueError("cagr must be finite")

    # Annualized volatility: sample std * sqrt(sessions_per_year)
    if n == 1:
        annualized_volatility = 0.0
    else:
        mean = sum_log / n
        var = sum((x - mean) ** 2 for x in log_returns) / (n - 1)
        if var < 0:
            var = 0.0
        std = math.sqrt(var)
        annualized_volatility = std * math.sqrt(float(sessions_per_year))
    if not math.isfinite(annualized_volatility) or annualized_volatility < 0:
        raise ValueError("annualized_volatility must be finite nonnegative")

    # Max drawdown
    peak = float(daily_nav[0].nav)
    max_dd = 0.0
    for nav in daily_nav:
        cur = float(nav.nav)
        if cur > peak:
            peak = cur
        dd = (peak - cur) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    if not math.isfinite(max_dd) or max_dd < 0:
        raise ValueError("max_drawdown must be finite nonnegative")

    # Calmar: cagr / max_drawdown, None if denominator zero
    calmar: float | None
    if max_dd == 0.0:
        calmar = None
    else:
        calmar_val = cagr / max_dd
        if not math.isfinite(calmar_val):  # noqa: SIM108
            calmar = None
        else:
            calmar = float(calmar_val)

    # Sortino: annualized_log_growth / downside_vol (annualized)
    downside = [r for r in log_returns if r < 0]
    sortino: float | None
    if len(downside) == 0:
        sortino = None
    else:
        if len(downside) == 1:
            downside_vol = 0.0
        else:
            mean_down = sum(downside) / len(downside)
            var_down = sum((x - mean_down) ** 2 for x in downside) / (len(downside) - 1)
            if var_down < 0:
                var_down = 0.0
            std_down = math.sqrt(var_down)
            downside_vol = std_down * math.sqrt(float(sessions_per_year))
        if downside_vol == 0.0 or not math.isfinite(downside_vol):
            sortino = None
        else:
            sortino_val = annualized_log_growth / downside_vol
            if not math.isfinite(sortino_val):  # noqa: SIM108
                sortino = None
            else:
                sortino = float(sortino_val)

    return LedgerMetrics(
        annualized_log_growth=float(annualized_log_growth),
        cagr=float(cagr),
        annualized_volatility=float(annualized_volatility),
        max_drawdown=float(max_dd),
        calmar=calmar,
        sortino=sortino,
    )
