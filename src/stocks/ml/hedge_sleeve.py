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
    }
