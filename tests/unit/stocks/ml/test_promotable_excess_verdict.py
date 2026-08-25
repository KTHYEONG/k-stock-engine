"""PROMOTABLE_EXCESS verdict scenarios: SCENARIO_PROMOTABLE_EXCESS_*, W3 blend."""
from __future__ import annotations

from types import SimpleNamespace

from src.stocks.ml.horizons import GrowthRouteEvidence
from src.stocks.ml.training import (
    _blend_champion_no_trade,
    _growth_route_projection,
    _policy_frontier_projection,
)


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


def test_blend_champion_publishes_excess_verdict() -> None:
    """SCENARIO_BLEND_EXCESS_VERDICT_CARVEOUT."""
    excess_route = dict(_growth_route_projection(_route(), _certificate(
        ["non-positive-base-lower-cagr", "non-positive-stress-lower-cagr"], 0.128,
    )))
    assert excess_route["promotion_status"] == "PROMOTABLE_EXCESS"
    reason, payload = _blend_champion_no_trade(excess_route, _certificate(
        ["non-positive-base-lower-cagr", "non-positive-stress-lower-cagr"], 0.128,
    ))
    assert reason == "blend-champion-excess-verdict"
    assert payload["growth_route"]["promotion_status"] == "PROMOTABLE_EXCESS"

    plain_route = dict(_growth_route_projection(_route(), _certificate(
        ["non-positive-base-lower-cagr"], None,
    )))
    reason_plain, _ = _blend_champion_no_trade(plain_route, _certificate(
        ["non-positive-base-lower-cagr"], None,
    ))
    assert reason_plain == "blend-champion-holdout-unsupported"


def test_frontier_publishes_blend_lower_growth() -> None:
    """SCENARIO_BLEND_LOWER_GROWTH_PUBLISHED."""
    from src.stocks.ml.horizons import HorizonOOFEvidence

    def _candidate(profile_id: str) -> HorizonOOFEvidence:
        n = 40
        return HorizonOOFEvidence(
            horizon_sessions=10,
            profile_id=profile_id,
            model_family="economic_rawnet_lgbm",
            base_log_growth=tuple(0.01 for _ in range(n)),
            stress_log_growth=tuple(0.008 for _ in range(n)),
            cohort_segment_ids=tuple(0 for _ in range(n)),
            complete_cohort_count=n,
            active_cohort_count=n,
            partial_cohort_count=0,
            missing_cohort_count=0,
            segment_count=1,
            fold_rank_ics=(0.2,),
            rebalance_frequency_sessions=5,
            top_k=12,
        )

    request = SimpleNamespace(
        policy_profiles=(
            SimpleNamespace(profile_id="lower_bound_only"),
            SimpleNamespace(profile_id="lower_bound_half_kelly"),
        ),
        bootstrap_alpha=0.05,
        seed=42,
        bootstrap_resamples=200,
    )
    discovery = SimpleNamespace(
        evidence=(_candidate("lower_bound_only"), _candidate("lower_bound_only:blend")),
        dropout_reasons={},
        execution_evidence_by_candidate={},
        sizing_diagnostics_by_candidate={},
    )
    frontier = _policy_frontier_projection(request, discovery, None)
    blend_map = frontier["blend_lower_growth"]
    assert set(blend_map.keys()) == {"10:5:12:lower_bound_only:blend"}
    entry = next(iter(blend_map.values()))
    assert set(entry.keys()) == {"base_lower_growth", "stress_lower_growth"}
    assert all(isinstance(v, float) for v in entry.values())

    plain_discovery = SimpleNamespace(
        evidence=(_candidate("lower_bound_only"),),
        dropout_reasons={},
        execution_evidence_by_candidate={},
        sizing_diagnostics_by_candidate={},
    )
    empty_frontier = _policy_frontier_projection(request, plain_discovery, None)
    assert empty_frontier["blend_lower_growth"] == {}
