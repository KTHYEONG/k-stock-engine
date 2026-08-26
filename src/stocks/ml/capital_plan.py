"""Absolute-capital implementation planning over certified growth routes.

Converts one certified growth route plus an explicit seed-capital plan into a
bounded, deterministic implementation verdict: per-instrument-class
admissibility (index futures lots, inverse-ETF overlay, unhedged) under
quantitative predicates, and the admissible sub-unit-leverage projection for
the small-capital fallback route. Pure and read-only: no I/O, no artifact
writes, no promotion side effects. A route is ``IMPLEMENTABLE`` only when at
least one instrument class clears every declared threshold; otherwise the
verdict fails closed to ``NO_IMPLEMENTATION_ROUTE`` with normalized reasons.
"""
from __future__ import annotations

import math
from typing import cast

from src.stocks.ml.contracts import SmallCapitalPlanSettings
from src.stocks.ml.hedge_sleeve import project_hedge_sleeve
from src.stocks.ml.horizons import GrowthRouteEvidence

__all__ = ["build_small_capital_route_plan"]

_IMPLEMENTABLE_VERDICT = "IMPLEMENTABLE"
_NO_IMPLEMENTATION_VERDICT = "NO_IMPLEMENTATION_ROUTE"


def _round12(value: float) -> float:
    return round(float(value), 12)


def _fail_closed(
    settings: SmallCapitalPlanSettings, reasons: list[str]
) -> dict[str, object]:
    return {
        "seed_capital_krw": _round12(settings.seed_capital_krw),
        "equity_notional_krw": None,
        "position_count": None,
        "per_position_notional_krw": None,
        "verdict": _NO_IMPLEMENTATION_VERDICT,
        "reasons": sorted(set(reasons)),
        "instrument_routes": [],
        "unhedged_projection": {},
    }


def _futures_route(
    *,
    instrument_class: str,
    lot_notional_krw: float,
    target_hedge_notional: float,
    settings: SmallCapitalPlanSettings,
    floor_ok: bool,
) -> dict[str, object]:
    reasons: list[str] = []
    lots = round(target_hedge_notional / lot_notional_krw)
    coverage_error = (
        abs(lots * lot_notional_krw - target_hedge_notional)
        / target_hedge_notional
    )
    margin_locked_fraction = (
        lots
        * lot_notional_krw
        * settings.initial_margin_fraction
        / settings.seed_capital_krw
    )
    if lots < 1:
        reasons.append("futures-lot-unavailable")
    else:
        if coverage_error > settings.max_futures_coverage_error:
            reasons.append("futures-coverage-error")
        if margin_locked_fraction > settings.max_margin_locked_fraction:
            reasons.append("margin-lockup-exceeded")
    if not floor_ok:
        reasons.append("position-notional-floor")
    return {
        "instrument_class": instrument_class,
        "lots": lots,
        "lot_notional_krw": _round12(lot_notional_krw),
        "coverage_error": _round12(coverage_error),
        "margin_locked_fraction": _round12(margin_locked_fraction),
        "admissible": not reasons,
        "reasons": sorted(reasons),
    }


def _overlay_route(
    *,
    target_hedge_notional: float,
    position_count: int,
    settings: SmallCapitalPlanSettings,
    floor_ok: bool,
) -> dict[str, object]:
    overlay_capital_krw = (
        settings.seed_capital_krw * settings.max_overlay_capital_fraction
    )
    achieved_hedge_ratio = min(1.0, overlay_capital_krw / target_hedge_notional)
    residual_per_position = (
        settings.seed_capital_krw - overlay_capital_krw
    ) / position_count
    reasons: list[str] = []
    if not floor_ok or residual_per_position < settings.min_position_notional_krw:
        reasons.append("position-notional-floor")
    if achieved_hedge_ratio < settings.min_overlay_hedge_ratio:
        reasons.append("overlay-hedge-ratio-insufficient")
    return {
        "instrument_class": "inverse_etf_overlay",
        "overlay_capital_krw": _round12(overlay_capital_krw),
        "achieved_hedge_ratio": _round12(achieved_hedge_ratio),
        "admissible": not reasons,
        "reasons": sorted(reasons),
    }


def _unhedged_projection(
    series: list[float], settings: SmallCapitalPlanSettings
) -> dict[str, object]:
    try:
        payload = project_hedge_sleeve(
            series,
            leverage_grid=settings.unhedged_leverage_grid,
            annualization_sessions=settings.annualization_sessions,
            max_projected_mdd=settings.max_projected_mdd,
        )
    except ValueError:
        return {
            "leverage_grid": [
                _round12(value) for value in settings.unhedged_leverage_grid
            ],
            "admissible_leverages": [],
            "best_rung": None,
        }
    ladder_rows = [
        row
        for row in cast(
            "list[dict[str, object]]", payload.get("leverage_ladder", [])
        )
        if bool(row.get("admissible"))
    ]
    admissible_leverages = sorted(
        {float(cast("float", row["leverage"])) for row in ladder_rows}
    )
    best_rung: dict[str, object] | None = None
    if ladder_rows:
        variant_rank = {"vol_managed": 0, "static": 1}
        best_row = min(
            ladder_rows,
            key=lambda row: (
                -float(cast("float", row["stress_cagr"])),
                variant_rank.get(str(row["variant"]), 2),
                float(cast("float", row["leverage"])),
            ),
        )
        best_rung = {
            "variant": str(best_row["variant"]),
            "leverage": _round12(cast("float", best_row["leverage"])),
            "point_cagr": _round12(cast("float", best_row["point_cagr"])),
            "stress_cagr": _round12(cast("float", best_row["stress_cagr"])),
            "projected_mdd": _round12(cast("float", best_row["projected_mdd"])),
        }
    return {
        "leverage_grid": [
            _round12(value) for value in settings.unhedged_leverage_grid
        ],
        "leverage_ladder": [
            {
                **row,
                "leverage": _round12(cast("float", row["leverage"])),
                "point_cagr": _round12(cast("float", row["point_cagr"])),
                "stress_cagr": _round12(cast("float", row["stress_cagr"])),
                "projected_vol": _round12(cast("float", row["projected_vol"])),
                "projected_mdd": _round12(cast("float", row["projected_mdd"])),
                "margin_buffer": _round12(cast("float", row["margin_buffer"])),
            }
            for row in cast(
                "list[dict[str, object]]", payload.get("leverage_ladder", [])
            )
        ],
        "admissible_leverages": [_round12(value) for value in admissible_leverages],
        "best_rung": best_rung,
    }


def build_small_capital_route_plan(
    route: GrowthRouteEvidence,
    settings: SmallCapitalPlanSettings,
) -> dict[str, object]:
    """Judge route implementability at an explicit seed capital, fail-closed.

    Evaluates every declared instrument class against quantitative
    thresholds (futures lot coverage error and margin lockup, inverse-ETF
    achievable hedge ratio and residual position floor, per-position notional
    floor) and projects the sub-unit-leverage unhedged fallback via the
    shared hedge-sleeve kernel on the base log-growth series.

    Raises:
        ValueError: never at runtime; incomplete routes fail closed to
            ``NO_IMPLEMENTATION_ROUTE`` with normalized reasons instead.
    """
    series = [float(value) for value in route.base_log_growth]
    if not series or not all(math.isfinite(value) for value in series):
        return _fail_closed(settings, ["period-series-incomplete"])
    policy = next(
        (key for key in reversed(route.selected_policies) if key is not None),
        None,
    )
    if policy is None:
        return _fail_closed(settings, ["no-invested-policy"])

    position_count = int(policy[2])
    equity_notional = settings.seed_capital_krw * settings.equity_utilization
    per_position_notional = equity_notional / position_count
    floor_ok = per_position_notional >= settings.min_position_notional_krw
    target_hedge_notional = equity_notional * settings.target_beta

    routes: list[dict[str, object]] = []
    for instrument_class, lot_notional in (
        ("index_futures_full", settings.full_futures_lot_notional_krw),
        ("index_futures_mini", settings.mini_futures_lot_notional_krw),
    ):
        routes.append(
            _futures_route(
                instrument_class=instrument_class,
                lot_notional_krw=lot_notional,
                target_hedge_notional=target_hedge_notional,
                settings=settings,
                floor_ok=floor_ok,
            )
        )
    routes.append(
        _overlay_route(
            target_hedge_notional=target_hedge_notional,
            position_count=position_count,
            settings=settings,
            floor_ok=floor_ok,
        )
    )
    routes.append(
        {
            "instrument_class": "unhedged",
            "admissible": floor_ok,
            "reasons": [] if floor_ok else ["position-notional-floor"],
        }
    )

    verdict = (
        _IMPLEMENTABLE_VERDICT
        if any(bool(row["admissible"]) for row in routes)
        else _NO_IMPLEMENTATION_VERDICT
    )
    all_reasons = sorted(
        {reason for row in routes for reason in cast("list[str]", row["reasons"])}
    )

    return {
        "seed_capital_krw": _round12(settings.seed_capital_krw),
        "equity_notional_krw": _round12(equity_notional),
        "position_count": position_count,
        "per_position_notional_krw": _round12(per_position_notional),
        "verdict": verdict,
        "reasons": all_reasons,
        "instrument_routes": routes,
        "unhedged_projection": _unhedged_projection(series, settings),
    }
