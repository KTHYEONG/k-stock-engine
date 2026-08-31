"""Research-only futures-hedged sleeve projection over certified excess streams.

Converts one certified exposure-matched excess interval log-growth series into
absolute sleeve scenarios per leverage rung. Two variants are evaluated per
leverage: a static ladder and a causal vol-managed overlay that targets the
trailing annualized volatility of the excess stream itself. The projection is
bounded and read-only: it never publishes artifacts, never sets promotion
flags, and never enters manifests. A rung is admissible only when its
projected drawdown stays under the cap and its model margin buffer clears the
minimum.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from src.stocks.ml.contracts import HedgeDeploymentEvidence, SmallCapitalPlanSettings

# Initial margin fraction per unit of index notional (KOSPI200-style futures).
_MARGIN_FRACTION_PER_LEVERAGE = 0.15

# Vol-managed overlay bounds: warm-up scale and scale clamp.
_VOL_MANAGED_WARMUP_SCALE = 0.5
_VOL_MANAGED_SCALE_CLAMP = (0.25, 2.0)


def _vol_managed_scales(
    values: Sequence[float],
    *,
    lookback: int = 26,
    target_annualized_vol: float = 0.10,
    annualization_sessions: int = 250,
) -> list[float]:
    """Causal trailing-vol scales using strictly past windows.

    Index ``t`` sees only ``values[t - lookback : t]``; warm-up indices carry
    the fixed warm-up scale. The scale is clamped to
    ``_VOL_MANAGED_SCALE_CLAMP`` before multiplication.
    """
    if lookback < 1:
        raise ValueError("vol-managed lookback must be positive")
    ann_const = math.sqrt(annualization_sessions)
    scales: list[float] = []
    for t in range(len(values)):
        if t < lookback:
            scales.append(_VOL_MANAGED_WARMUP_SCALE)
            continue
        window = values[t - lookback : t]
        mean = sum(window) / lookback
        variance = sum((v - mean) ** 2 for v in window) / lookback
        trailing_ann_vol = math.sqrt(variance) * ann_const
        raw = target_annualized_vol / max(trailing_ann_vol, 1e-12)
        clamp_low, clamp_high = _VOL_MANAGED_SCALE_CLAMP
        scales.append(min(max(raw, clamp_low), clamp_high))
    return scales


def project_hedge_sleeve(
    excess_log_growth: Sequence[float],
    *,
    leverage_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
    annualization_sessions: int = 250,
    max_projected_mdd: float = 0.25,
    min_margin_buffer: float = 0.30,
    vol_managed_lookback: int = 26,
    vol_managed_target_annualized_vol: float = 0.10,
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
        vol_managed_lookback: trailing window length for the vol-managed
            overlay variant.
        vol_managed_target_annualized_vol: target annualized vol for the
            vol-managed overlay variant.

    Returns:
        Bounded projection payload with the raw excess point CAGR and two
        ladder entries per leverage (``variant='static'`` first, then
        ``variant='vol_managed'``), each carrying point/stress CAGR, projected
        vol, projected MDD, margin buffer, and admissibility flags. The stress
        leg deducts the continuous-time variance drag ``0.5 * (L * sigma)^2``
        so a levered rung never claims gross compounding without volatility
        cost. ``admissible_leverages`` maps each variant to its admissible
        multipliers.

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

    def _cagr(log_total: float) -> float:
        base = log_total / years
        return math.expm1(max(min(base, 50.0), -50.0))

    def _mdd(series: Sequence[float], leverage: float) -> float:
        equity = 1.0
        peak = 1.0
        mdd = 0.0
        for value in series:
            equity *= math.exp(leverage * value)
            if equity <= 0.0:
                return 1.0
            peak = max(peak, equity)
            mdd = max(mdd, 1.0 - equity / peak)
        return mdd

    excess_point_cagr = _cagr(total_log)

    vol_scales = _vol_managed_scales(
        values,
        lookback=vol_managed_lookback,
        target_annualized_vol=vol_managed_target_annualized_vol,
        annualization_sessions=annualization_sessions,
    )
    vol_values = [
        value * scale for value, scale in zip(values, vol_scales, strict=True)
    ]
    vol_total_log = math.fsum(vol_values)
    vol_mean = sum(vol_values) / len(vol_values)
    vol_variance = sum((v - vol_mean) ** 2 for v in vol_values) / len(vol_values)

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)

    ladder: list[dict[str, object]] = []
    admissible: dict[str, list[float]] = {"static": [], "vol_managed": []}
    variants = (
        ("static", values, total_log, variance),
        ("vol_managed", vol_values, vol_total_log, vol_variance),
    )
    for variant, series, variant_total_log, variant_variance in variants:
        session_vol = math.sqrt(variant_variance)
        for leverage in leverage_grid:
            leverage_f = float(leverage)
            vol_annualized = session_vol * math.sqrt(annualization_sessions) * leverage_f
            stress_cagr = _cagr(
                leverage_f * variant_total_log - 0.5 * (vol_annualized ** 2) * years
            )
            margin_buffer = max(
                0.0, min(1.0, 1.0 - leverage_f * _MARGIN_FRACTION_PER_LEVERAGE)
            )
            projected_mdd = _mdd(series, leverage_f)
            within_mdd_cap = projected_mdd <= max_projected_mdd
            margin_ok = margin_buffer >= min_margin_buffer
            rung_admissible = within_mdd_cap and margin_ok
            if rung_admissible:
                admissible[variant].append(leverage_f)
            ladder.append(
                {
                    "variant": variant,
                    "leverage": leverage_f,
                    "point_cagr": _cagr(leverage_f * variant_total_log),
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
        "vol_managed_lookback": int(vol_managed_lookback),
        "vol_managed_target_annualized_vol": float(
            vol_managed_target_annualized_vol
        ),
        "leverage_ladder": ladder,
        "admissible_leverages": admissible,
        "research_projection_note": "base-minus-benchmark residual is not an executable futures P&L",
    }


def project_executable_hedged_route(
    stock_base_log_growth: Sequence[float],
    stock_stress_log_growth: Sequence[float],
    hedge: HedgeDeploymentEvidence,
    settings: SmallCapitalPlanSettings,
) -> dict[str, object]:
    """Executable hedged route using tradable proxy and discrete lots.

    Builds stock minus beta*hedge residual, deducts costs exactly once,
    applies discrete-lot scaling, and fails closed on non-parallel or
    negative cash.
    """
    import math as _math  # noqa: I001

    base_vals = [float(v) for v in stock_base_log_growth]
    stress_vals = [float(v) for v in stock_stress_log_growth]
    hedge_base = [float(v) for v in hedge.hedge_base_log_growth]
    hedge_stress = [float(v) for v in hedge.hedge_stress_log_growth]
    # non-finite check
    for arr in (base_vals, stress_vals, hedge_base, hedge_stress):
        if not arr:
            raise ValueError("parallel: empty series")
        if not all(_math.isfinite(v) for v in arr):
            raise ValueError("parallel: non-finite value")
    n = len(base_vals)
    if not (len(stress_vals) == n and len(hedge_base) == n and len(hedge_stress) == n):
        raise ValueError("parallel: series lengths must be identical")
    if not hedge.tradable_proxy_id:
        raise ValueError("parallel: missing tradable_proxy_id")
    # Cash reserve is ring-fenced and variation margin is settled against it.
    seed = float(settings.seed_capital_krw)
    reserve = seed * float(settings.cash_reserve_fraction)
    lot_notional = float(settings.mini_futures_lot_notional_krw)
    margin_frac = float(hedge.initial_margin_fraction)
    # Build the net series with the actually purchased discrete hedge ratio.
    base_cost_per = float(hedge.base_cost_drag) / n if n else 0.0
    stress_cost_per = float(hedge.stress_cost_drag) / n if n else 0.0
    max_lots = _math.floor((seed - reserve) / (lot_notional * margin_frac))
    best = None
    for lots in range(1, max_lots + 1):
        margin = lots * lot_notional * margin_frac
        stock_notional = seed - reserve - margin
        if stock_notional < float(settings.min_position_notional_krw):
            continue
        if stock_notional + margin + reserve > seed + 1e-9:
            continue
        target = float(hedge.beta) * stock_notional
        hedge_notional = lots * lot_notional
        coverage = abs(hedge_notional - target) / target if target > 1e-12 else (0.0 if hedge_notional == 0 else 1.0)
        if coverage > float(settings.max_futures_coverage_error):
            continue
        actual_ratio = hedge_notional / stock_notional
        candidate_base = [s - actual_ratio * h - base_cost_per for s, h in zip(base_vals, hedge_base, strict=True)]
        candidate_stress = [s - actual_ratio * h - stress_cost_per for s, h in zip(stress_vals, hedge_stress, strict=True)]
        if not all(_math.isfinite(v) for v in candidate_base + candidate_stress):
            raise ValueError("parallel: non-finite hedged value")
        # Variation margin is the only cash movement in this route. Stock cash
        # and initial margin are reserved at inception; no negative reserve is allowed.
        cash = reserve
        negative = False
        for h in hedge_base:
            cash += hedge_notional * (_math.expm1(h))
            if cash < -1e-9:
                negative = True
                break
            if not _math.isfinite(cash):
                negative = True
                break
        if negative:
            continue
        cand = (coverage, margin, lots, stock_notional, margin_frac * lots * lot_notional / seed if seed else 0)
        if best is None or cand < best[0]:
            best = (cand, lots, stock_notional, margin, coverage)
    if best is None:
        raise ValueError("parallel: no affordable lot candidate preserves cash")
    _, lots_sel, stock_notional_sel, margin_sel, coverage_sel = best
    # compute bounded aggregates
    ann = int(settings.annualization_sessions)

    def _cagr(vals: Sequence[float]) -> float:
        total = _math.fsum(vals)
        years = len(vals) / ann if ann else 0
        if years <= 0:
            return 0.0
        return _math.expm1(max(min(total / years, 50.0), -50.0))

    def _mdd(vals: Sequence[float]) -> float:
        eq = 1.0
        pk = 1.0
        m = 0.0
        for vv in vals:
            eq *= _math.exp(vv)
            pk = max(pk, eq)
            m = max(m, 1.0 - eq / pk if pk else 1.0)
        return m

    hedged_base = [s - (lots_sel * lot_notional / stock_notional_sel) * h - base_cost_per for s, h in zip(base_vals, hedge_base, strict=True)]
    hedged_stress = [s - (lots_sel * lot_notional / stock_notional_sel) * h - stress_cost_per for s, h in zip(stress_vals, hedge_stress, strict=True)]
    # Scale the net return to the funded stock sleeve.
    scale = stock_notional_sel / seed if seed else 0.0
    base_scaled = [v * scale for v in hedged_base]
    stress_scaled = [v * scale for v in hedged_stress]
    base_lower = round(float(_cagr(base_scaled)), 12)
    stress_lower = round(float(_cagr(stress_scaled)), 12)
    mdd_val = round(float(_mdd(base_scaled)), 12)
    max_margin_use = round(float(margin_sel / seed if seed else 0), 12)
    return {
        "selected_lots": int(lots_sel),
        "lots": int(lots_sel),
        "stock_notional_krw": round(float(stock_notional_sel), 12),
        "funded_stock_notional_krw": round(float(stock_notional_sel), 12),
        "cash_reserve_krw": round(float(reserve), 12),
        "reserve_krw": round(float(reserve), 12),
        "initial_margin_krw": round(float(margin_sel), 12),
        "max_margin_fraction": max_margin_use,
        "max_margin_use": max_margin_use,
        "coverage_error": round(float(coverage_sel), 12),
        "base_lower_cagr": base_lower,
        "stress_lower_cagr": stress_lower,
        "mdd": mdd_val,
        "reasons": [],
    }
