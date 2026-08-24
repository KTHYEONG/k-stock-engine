"""PROMOTABLE_EXCESS verdict scenarios: SCENARIO_PROMOTABLE_EXCESS_*."""
from __future__ import annotations

from src.stocks.ml.horizons import GrowthRouteEvidence
from src.stocks.ml.training import _growth_route_projection


def _route() -> GrowthRouteEvidence:
    return GrowthRouteEvidence(
        base_log_growth=tuple(0.001 for _ in range(1000)),
        stress_log_growth=tuple(0.001 for _ in range(1000)),
        segment_ids=tuple(0 for _ in range(1000)),
        selected_policies=((10, 10, 8, "lower_bound_only"),),
        observed_interval_count=1000,
        invested_interval_count=1000,
        filled_orders=500,
    )


def _certificate(
    reasons: list[str],
    matched: float | None,
    *,
    mdd: float = 0.24,
) -> dict[str, object]:
    return {
        "passed": False,
        "reasons": list(reasons),
        "cagr_base": 0.054,
        "cagr_stress": 0.053,
        "base_lower_cagr": -0.02,
        "stress_lower_cagr": -0.03,
        "matched_lower_excess_cagr": matched,
        "mdd": mdd,
        "observed_intervals": 1000,
        "invested_intervals": 1000,
        "filled_orders": 500,
    }


def test_promotable_excess_when_absolute_fails() -> None:
    """SCENARIO_PROMOTABLE_EXCESS_VERDICT."""
    certificate = _certificate(
        ["non-positive-base-lower-cagr", "non-positive-stress-lower-cagr"],
        0.128,
    )
    projection = _growth_route_projection(_route(), certificate)
    assert projection["promotion_status"] == "PROMOTABLE_EXCESS"
    assert projection.get("promoted") is not True
    assert "non-positive-base-lower-cagr" in projection["rejection_reason_counts"]


def test_no_excess_verdict_when_matched_lower_nonpositive() -> None:
    """SCENARIO_PROMOTABLE_EXCESS_FAIL_CLOSED."""
    for matched in (None, 0.0, -0.01):
        certificate = _certificate(["non-positive-base-lower-cagr"], matched)
        projection = _growth_route_projection(_route(), certificate)
        assert projection["promotion_status"] == "NO_TRADE"

    # coverage or drawdown gate failure also blocks the excess verdict
    blocked_route = GrowthRouteEvidence(
        base_log_growth=tuple(0.001 for _ in range(1000)),
        stress_log_growth=tuple(0.001 for _ in range(1000)),
        segment_ids=tuple(0 for _ in range(1000)),
        selected_policies=((10, 10, 8, "lower_bound_only"),),
        observed_interval_count=1000,
        invested_interval_count=100,
        filled_orders=500,
    )
    certificate = _certificate(
        [
            "non-positive-base-lower-cagr",
            "invested-coverage-insufficient",
        ],
        0.128,
    )
    projection = _growth_route_projection(blocked_route, certificate)
    assert projection["promotion_status"] == "NO_TRADE"

    certificate_dd = _certificate(
        ["non-positive-base-lower-cagr", "max-drawdown-exceeded"],
        0.128,
        mdd=0.31,
    )
    projection_dd = _growth_route_projection(_route(), certificate_dd)
    assert projection_dd["promotion_status"] == "NO_TRADE"


def test_promoted_when_certificate_passes() -> None:
    certificate = _certificate([], None)
    certificate["passed"] = True
    certificate["base_lower_cagr"] = 0.02
    certificate["stress_lower_cagr"] = 0.01
    projection = _growth_route_projection(_route(), certificate)
    assert projection["promotion_status"] == "PROMOTED"
