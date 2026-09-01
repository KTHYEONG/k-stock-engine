"""ETF satellite overlay projection over certified gated routes.

Folds the net-exposure gate's freed capital into leveraged/inverse index ETF
satellites and projects the combined book on top of a certified route series.
Pure and deterministic: O(n) per series, no RNG, no I/O. Reset drag is modeled
analytically from the benchmark interval variance; taxation applies the full
statutory gain rate to positive satellite PnL only (losses carry zero tax, and
the statutory 과표기준가 Min rule can only reduce the real burden below this
conservative assumption). Missing or mismatched inputs fail open to the raw
ungated series with verdict ``INPUTS_INSUFFICIENT``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from legacy.stocks.ml.contracts import SatelliteOverlaySettings

_ANNUALIZATION_SESSIONS = 252
_INSUFFICIENT_REASON = "satellite-inputs-insufficient"

__all__ = ["project_satellite_overlay"]


def _round12(value: float) -> float:
    return round(float(value), 12)


def _cagr(series: list[float]) -> float:
    if not series:
        return 0.0
    return math.expm1(
        math.fsum(series) * _ANNUALIZATION_SESSIONS / len(series)
    )


def _mdd(series: list[float]) -> float:
    equity = peak = 1.0
    mdd = 0.0
    for value in series:
        equity *= math.exp(value)
        if equity <= 0.0:
            return 1.0
        peak = max(peak, equity)
        mdd = max(mdd, 1.0 - equity / peak)
    return mdd


def _as_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nem_component_series(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, float]]:
    """Extract causal NEM component records into bounded float dicts."""
    out: list[dict[str, float]] = []
    keys = (
        "gross_pre_nem",
        "gross_post_nem",
        "nem_s_trend",
        "nem_s_vol",
    )
    for record in records:
        values = {key: _as_float(record.get(key)) for key in keys}
        if any(value is None for value in values.values()):
            continue
        out.append(
            {key: float(value) for key, value in values.items() if value is not None}
        )
    return out


def project_satellite_overlay(
    base_log_growth: Sequence[float],
    stress_log_growth: Sequence[float],
    benchmark_log_growth: Sequence[float],
    nem_components: Sequence[Mapping[str, float]],
    settings: SatelliteOverlaySettings,
    *,
    horizon_sessions: int = 10,
) -> dict[str, object]:
    """Project the combined stock+satellite book for a certified gated route.

    Satellite weights are decided causally from the recorded gate components:
    trend-triggered de-risking funds the inverse satellite; benign uptrend
    headroom may fund the leveraged satellite. Reset drag uses the analytic
    form ``k*B_t - 0.5*k*(k-1)*Var_B``; costs accrue annual fees pro-rata plus
    spread on weight changes; taxation hits positive satellite log-PnL only.
    """
    base = [float(v) for v in base_log_growth]
    stress = [float(v) for v in stress_log_growth]
    bench = [float(v) for v in benchmark_log_growth]
    ungated_point = _cagr(base)
    ungated_stress = _cagr(stress)
    ungated_mdd = _mdd(base)

    def _fail_open() -> dict[str, object]:
        return {
            "verdict": "INPUTS_INSUFFICIENT",
            "reasons": [_INSUFFICIENT_REASON],
            "combined_point_cagr": _round12(ungated_point),
            "combined_stress_cagr": _round12(ungated_stress),
            "combined_mdd": _round12(ungated_mdd),
            "ungated_point_cagr": _round12(ungated_point),
            "ungated_stress_cagr": _round12(ungated_stress),
            "ungated_mdd": _round12(ungated_mdd),
            "inverse_weight_mean": None,
            "leveraged_weight_mean": None,
            "combined_log_growth": list(base),
        }

    if (
        len(base) < 2
        or len(base) != len(bench)
        or len(base) != len(stress)
        or not settings.enabled
    ):
        return _fail_open()

    comps = [
        comp
        for comp in (
            {
                key: float(value)
                for key, value in item.items()
                if key
                in (
                    "gross_pre_nem",
                    "gross_post_nem",
                    "nem_s_trend",
                    "nem_s_vol",
                )
            }
            if isinstance(item, Mapping)
            else None
            for item in nem_components
        )
        if comp is not None and len(comp) == 4
    ]

    var_b = (
        sum((v - sum(bench) / len(bench)) ** 2 for v in bench) / len(bench)
        if bench
        else 0.0
    )
    fee_per_interval = settings.fee_bps_annual / 1e4 / max(1, horizon_sessions)
    spread_rate = settings.spread_bps / 1e4

    combined_base: list[float] = []
    combined_stress: list[float] = []
    inv_weights: list[float] = []
    lev_weights: list[float] = []
    prev_inv = prev_lev = 0.0
    k_inv = float(settings.inverse_multiplier)
    k_lev = float(settings.leveraged_multiplier)

    for t in range(len(base)):
        comp = comps[min(t, len(comps) - 1)] if comps else None
        inv_w = lev_w = 0.0
        if comp is not None:
            freed = max(0.0, comp["gross_pre_nem"] - comp["gross_post_nem"])
            if comp["nem_s_trend"] < 1.0:
                inv_w = min(settings.inverse_weight_cap, freed)
            elif comp["nem_s_vol"] == 1.0:
                headroom = max(
                    0.0, settings.cash_reserve_cap - comp["gross_post_nem"]
                )
                lev_w = min(settings.leveraged_weight_cap, headroom)

        b_t = bench[t]
        log_inv = k_inv * b_t - 0.5 * k_inv * (k_inv - 1.0) * var_b
        log_lev = k_lev * b_t - 0.5 * k_lev * (k_lev - 1.0) * var_b
        tax_inv = settings.gain_tax_rate * max(0.0, log_inv)
        tax_lev = settings.gain_tax_rate * max(0.0, log_lev)
        fee_total = fee_per_interval * (inv_w + lev_w)
        spread_cost = spread_rate * (
            abs(inv_w - prev_inv) + abs(lev_w - prev_lev)
        )

        sat_base = (
            inv_w * (log_inv - tax_inv)
            + lev_w * (log_lev - tax_lev)
            - fee_total
            - spread_cost
        )
        combined_base.append(base[t] + sat_base)
        combined_stress.append(stress[t] + sat_base)
        inv_weights.append(inv_w)
        lev_weights.append(lev_w)
        prev_inv, prev_lev = inv_w, lev_w

    combined_mdd = _mdd(combined_base)
    verdict = (
        "WITHIN_BUDGET"
        if combined_mdd <= settings.mdd_budget
        else "MDD_EXCEEDED"
    )
    reasons = [] if verdict == "WITHIN_BUDGET" else ["combined-mdd-exceeded"]
    return {
        "verdict": verdict,
        "reasons": reasons,
        "combined_point_cagr": _round12(_cagr(combined_base)),
        "combined_stress_cagr": _round12(_cagr(combined_stress)),
        "combined_mdd": _round12(combined_mdd),
        "ungated_point_cagr": _round12(ungated_point),
        "ungated_stress_cagr": _round12(ungated_stress),
        "ungated_mdd": _round12(ungated_mdd),
        "inverse_weight_mean": _round12(sum(inv_weights) / len(inv_weights)),
        "leveraged_weight_mean": _round12(sum(lev_weights) / len(lev_weights)),
        "combined_log_growth": [_round12(v) for v in combined_base],
    }
