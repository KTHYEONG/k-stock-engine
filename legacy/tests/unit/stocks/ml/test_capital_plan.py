"""Small-capital route plan scenarios: SCENARIO_CAP_PLAN_01..05."""
from __future__ import annotations

import math

import numpy as np

from legacy.stocks.ml.capital_plan import build_small_capital_route_plan
from legacy.stocks.ml.contracts import SmallCapitalPlanSettings
from legacy.stocks.ml.horizons import GrowthRouteEvidence
from legacy.stocks.ml.training import _growth_route_projection

_POLICY = (10, 5, 8, "growth_full_utilization")


def _route(series: list[float]) -> GrowthRouteEvidence:
    count = len(series)
    return GrowthRouteEvidence(
        base_log_growth=tuple(series),
        stress_log_growth=tuple(series),
        segment_ids=(0,) * count,
        selected_policies=(_POLICY,),
        interval_policies=(_POLICY,) * count,
        observed_interval_count=count,
        invested_interval_count=count,
        filled_orders=count,
    )


def _positive_drift_series() -> list[float]:
    rng = np.random.default_rng(11)
    draws = 0.0012 + 0.002 * rng.standard_normal(600)
    centered = draws - draws.mean() + 0.0012
    return centered.tolist()


def _hostile_decline_series() -> list[float]:
    return [-0.006] * 600


def _route_by_class(
    plan: dict[str, object], label: str
) -> dict[str, object]:
    rows = [
        row
        for row in plan["instrument_routes"]  # type: ignore[union-attr]
        if str(row["instrument_class"]) == label
    ]
    assert len(rows) == 1, f"expected exactly one {label} route"
    return rows[0]


def test_futures_infeasible_at_5m_unhedged_only() -> None:
    """SCENARIO_CAP_PLAN_01_FUTURES_INFEASIBLE_AT_5M."""
    plan = build_small_capital_route_plan(
        _route(_positive_drift_series()),
        SmallCapitalPlanSettings(seed_capital_krw=5_000_000.0),
    )
    assert plan["verdict"] == "IMPLEMENTABLE"
    assert plan["position_count"] == 8
    assert plan["per_position_notional_krw"] == 593750.0

    full = _route_by_class(plan, "index_futures_full")
    assert full["lots"] == 0
    assert full["admissible"] is False
    assert "futures-lot-unavailable" in full["reasons"]

    mini = _route_by_class(plan, "index_futures_mini")
    assert mini["lots"] == 1
    assert mini["admissible"] is False
    assert abs(float(mini["coverage_error"]) - 0.842105263158) < 1e-9
    assert "futures-coverage-error" in mini["reasons"]

    overlay = _route_by_class(plan, "inverse_etf_overlay")
    assert abs(float(overlay["achieved_hedge_ratio"]) - 0.368421052632) < 1e-9
    assert overlay["admissible"] is False
    assert "overlay-hedge-ratio-insufficient" in overlay["reasons"]

    unhedged = _route_by_class(plan, "unhedged")
    assert unhedged["admissible"] is True
    assert unhedged["reasons"] == []


def test_tiny_seed_fail_closed() -> None:
    """SCENARIO_CAP_PLAN_02_TINY_SEED_FAIL_CLOSED."""
    plan = build_small_capital_route_plan(
        _route(_hostile_decline_series()),
        SmallCapitalPlanSettings(seed_capital_krw=400_000.0),
    )
    assert plan["verdict"] == "NO_IMPLEMENTATION_ROUTE"
    assert plan["per_position_notional_krw"] == 47500.0
    for row in plan["instrument_routes"]:
        assert row["admissible"] is False
        assert "position-notional-floor" in row["reasons"]
    assert "position-notional-floor" in plan["reasons"]
    projection = plan["unhedged_projection"]
    assert projection["admissible_leverages"] == []
    assert projection["best_rung"] is None


def test_mini_admissible_at_30m() -> None:
    """SCENARIO_CAP_PLAN_03_MINI_ADMISSIBLE_AT_30M."""
    plan = build_small_capital_route_plan(
        _route(_positive_drift_series()),
        SmallCapitalPlanSettings(seed_capital_krw=30_000_000.0),
    )
    assert plan["verdict"] == "IMPLEMENTABLE"

    mini = _route_by_class(plan, "index_futures_mini")
    assert mini["lots"] == 3
    assert mini["coverage_error"] <= 0.20
    assert abs(float(mini["coverage_error"]) - 0.078947368421) < 1e-9
    assert abs(float(mini["margin_locked_fraction"]) - 0.13125) < 1e-12
    assert mini["admissible"] is True


def test_projection_wiring_identity() -> None:
    """SCENARIO_CAP_PLAN_04_PROJECTION_WIRING_IDENTITY."""
    route = _route(_positive_drift_series())
    certificate: dict[str, object] = {"reasons": []}

    legacy = _growth_route_projection(route, certificate)
    assert "small_capital_route_plan" not in legacy

    planned = _growth_route_projection(
        route,
        certificate,
        capital_plan_settings=SmallCapitalPlanSettings(seed_capital_krw=5_000_000.0),
    )
    section = planned["small_capital_route_plan"]
    assert section["verdict"] == "IMPLEMENTABLE"

    shared_planned = {
        key: value
        for key, value in planned.items()
        if key != "small_capital_route_plan"
    }
    assert shared_planned == legacy


def test_unhedged_projection_bounds() -> None:
    """SCENARIO_CAP_PLAN_05_UNHEDGED_PROJECTION_BOUNDS."""
    grid = (0.25, 0.5, 0.75, 1.0)
    plan = build_small_capital_route_plan(
        _route(_positive_drift_series()),
        SmallCapitalPlanSettings(
            seed_capital_krw=5_000_000.0,
            unhedged_leverage_grid=grid,
            max_projected_mdd=0.35,
        ),
    )
    projection = plan["unhedged_projection"]
    admissible = [float(v) for v in projection["admissible_leverages"]]
    assert admissible
    assert set(admissible) <= set(grid)

    best_rung = projection["best_rung"]
    assert best_rung is not None
    assert float(best_rung["stress_cagr"]) > 0.0
    assert float(best_rung["projected_mdd"]) <= 0.35

    ladder = projection["leverage_ladder"]
    assert all(bool(row["margin_ok"]) for row in ladder)


def test_settings_reject_invalid_domains() -> None:
    """Guard rails: SCENARIO_CAP_PLAN settings fail closed outside domains."""
    for kwargs in (
        {"seed_capital_krw": 0.0},
        {"seed_capital_krw": -1.0},
        {"seed_capital_krw": 1000.0, "equity_utilization": 0.0},
        {"seed_capital_krw": 1000.0, "max_projected_mdd": 1.0},
        {"seed_capital_krw": 1000.0, "unhedged_leverage_grid": (1.5,)},
        {"seed_capital_krw": 1000.0, "unhedged_leverage_grid": (0.5, 0.5)},
        {"seed_capital_krw": 1000.0, "target_beta": 0.0},
    ):
        try:
            SmallCapitalPlanSettings(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_fail_closed_without_invested_policy() -> None:
    """Runtime fail-closed path without exceptions on cash-only routes."""
    route = GrowthRouteEvidence(
        base_log_growth=(0.01, 0.02, -0.005),
        stress_log_growth=(0.01, 0.02, -0.005),
        segment_ids=(0, 0, 0),
        selected_policies=(None,),
        observed_interval_count=3,
        invested_interval_count=0,
        filled_orders=0,
    )
    plan = build_small_capital_route_plan(
        route, SmallCapitalPlanSettings(seed_capital_krw=5_000_000.0)
    )
    assert plan["verdict"] == "NO_IMPLEMENTATION_ROUTE"
    assert plan["reasons"] == ["no-invested-policy"]


def test_fail_closed_on_empty_series() -> None:
    """Empty route series fails closed without raising (requirement parity)."""
    route = GrowthRouteEvidence(
        base_log_growth=(),
        stress_log_growth=(),
        segment_ids=(),
        selected_policies=(_POLICY,),
        observed_interval_count=0,
        invested_interval_count=0,
        filled_orders=0,
    )
    plan = build_small_capital_route_plan(
        route, SmallCapitalPlanSettings(seed_capital_krw=5_000_000.0)
    )
    assert plan["verdict"] == "NO_IMPLEMENTATION_ROUTE"
    assert plan["reasons"] == ["period-series-incomplete"]


def test_all_scalars_rounded_to_twelve_places() -> None:
    """Bounded-scalar hygiene: every float payload entry is 12-dp rounded."""
    plan = build_small_capital_route_plan(
        _route(_positive_drift_series()),
        SmallCapitalPlanSettings(seed_capital_krw=5_000_000.0),
    )

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, float):
            assert math.isfinite(value)
            assert value == round(value, 12)

    _walk(plan)


def test_small_capital_mini_hedge_requires_stock_margin_and_reserve_funding() -> None:
    from legacy.stocks.ml.capital_plan import build_small_capital_route_plan
    from legacy.stocks.ml.contracts import SmallCapitalPlanSettings
    from tests.unit.stocks.ml.test_capital_plan import _route, _positive_drift_series, _route_by_class

    settings = SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0, cash_reserve_fraction=0.05)
    plan = build_small_capital_route_plan(_route(_positive_drift_series()), settings)
    mini = _route_by_class(plan, "index_futures_mini")
    assert float(mini["stock_notional_krw"]) + float(mini["initial_margin_krw"]) + float(plan["cash_reserve_krw"]) <= 10_000_000.0
    assert float(mini["stock_notional_krw"]) < 9_500_000.0


def test_hedge_deployment_rejects_research_residual_without_tradable_evidence() -> None:
    from legacy.stocks.ml.capital_plan import build_small_capital_route_plan
    from legacy.stocks.ml.contracts import SmallCapitalPlanSettings
    from tests.unit.stocks.ml.test_capital_plan import _route, _positive_drift_series

    plan = build_small_capital_route_plan(
        _route(_positive_drift_series()),
        SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0),
        absolute_certificate={"passed": True},
        hedge_certificate={"passed": True},
    )
    assert plan["executable_hedge_verdict"] == "RESEARCH_ONLY_HEDGE"
    assert "tradable-hedge-evidence-missing" in plan["reasons"]
    assert plan["deployment_verdict"] != "DEPLOYABLE"
