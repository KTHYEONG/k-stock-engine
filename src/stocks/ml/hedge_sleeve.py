"""Research-only futures-hedged sleeve projection over certified excess streams.

Converts one certified exposure-matched excess interval log-growth series into
absolute sleeve scenarios per leverage rung. The projection is bounded and
read-only: it never publishes artifacts, never sets promotion flags, and never
enters manifests. A rung is admissible only when its projected drawdown stays
under the cap and its model margin buffer clears the minimum.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# Initial margin fraction per unit of index notional (KOSPI200-style futures).
_MARGIN_FRACTION_PER_LEVERAGE = 0.15


def project_hedge_sleeve(
    excess_log_growth: Sequence[float],
    *,
    leverage_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
    annualization_sessions: int = 250,
    max_projected_mdd: float = 0.25,
    min_margin_buffer: float = 0.30,
) -> dict[str, object]:
    """Project absolute sleeve CAGR/MDD per leverage over a certified series.

    Args:
        excess_log_growth: per-interval excess log growth of the certified
            matched-excess stream (base minus benchmark), strictly interval-
            parallel to the route series.
        leverage_grid: notional leverage multipliers to evaluate.
        annualization_sessions: sessions per year for annualization.
        max_projected_mdd: projected drawdown cap for rung admissibility.
        min_margin_buffer: minimum free-margin buffer fraction for rung
            admissibility under the fixed initial-margin constant.

    Returns:
        Bounded projection payload with the raw excess point CAGR and one
        ladder rung per leverage carrying point/stress CAGR, projected vol,
        projected MDD, margin buffer, and admissibility flags. The stress leg
        deducts the continuous-time variance drag ``0.5 * (L * sigma)^2`` so a
        levered rung never claims gross compounding without volatility cost.

    Raises:
        ValueError: when the series is empty or carries any non-finite value.
    """
    if len(excess_log_growth) == 0:
        raise ValueError("excess-series-incomplete")
    values = [float(value) for value in excess_log_growth]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("excess-series-incomplete")

    total_log = math.fsum(values)
    intervals = len(values)
    years = intervals / annualization_sessions if annualization_sessions > 0 else 0.0
    if years <= 0:
        raise ValueError("excess-series-incomplete")
    mean_log = sum(values) / intervals
    variance = sum((value - mean_log) ** 2 for value in values) / intervals

    def _cagr(log_total: float) -> float:
        base = log_total / years
        return math.expm1(max(min(base, 50.0), -50.0))

    def _mdd(leverage: float) -> float:
        equity = 1.0
        peak = 1.0
        mdd = 0.0
        for value in values:
            equity *= math.exp(leverage * value)
            if equity <= 0.0:
                return 1.0
            peak = max(peak, equity)
            mdd = max(mdd, 1.0 - equity / peak)
        return mdd

    excess_point_cagr = _cagr(total_log)
    session_vol = math.sqrt(variance)

    ladder: list[dict[str, object]] = []
    admissible: list[float] = []
    for leverage in leverage_grid:
        leverage_f = float(leverage)
        vol_annualized = session_vol * math.sqrt(annualization_sessions) * leverage_f
        stress_cagr = _cagr(
            leverage_f * total_log - 0.5 * (vol_annualized ** 2) * years
        )
        margin_buffer = max(0.0, min(1.0, 1.0 - leverage_f * _MARGIN_FRACTION_PER_LEVERAGE))
        projected_mdd = _mdd(leverage_f)
        within_mdd_cap = projected_mdd <= max_projected_mdd
        margin_ok = margin_buffer >= min_margin_buffer
        rung_admissible = within_mdd_cap and margin_ok
        if rung_admissible:
            admissible.append(leverage_f)
        ladder.append(
            {
                "leverage": leverage_f,
                "point_cagr": _cagr(leverage_f * total_log),
                "stress_cagr": stress_cagr,
                "projected_vol": vol_annualized,
                "projected_mdd": projected_mdd,
                "margin_buffer": margin_buffer,
                "within_mdd_cap": within_mdd_cap,
                "margin_ok": margin_ok,
                "admissible": rung_admissible,
            }
        )

    return {
        "series_intervals": intervals,
        "years": years,
        "excess_point_cagr": excess_point_cagr,
        "leverage_ladder": ladder,
        "admissible_leverages": admissible,
    }
