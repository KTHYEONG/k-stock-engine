"""Frozen-policy compound track: fail-closed stitch, point CAGR, projection."""
from __future__ import annotations

import math

import pytest

from legacy.stocks.ml.compound_track import (
    frozen_compound_track_projection,
    point_cagr_from_log_growth,
    resolve_frozen_policy_key,
    stitch_frozen_policy_growth_route,
)
from legacy.stocks.ml.contracts import (
    LOWER_BOUND_ONLY_PROFILE_ID,
    ExecutionFrontierSettings,
    NetAlphaTrainingRequest,
)
from legacy.stocks.ml.horizons import (
    GROWTH_ROUTE_VERSION,
    HorizonOOFEvidence,
    stitch_prequential_growth_route,
)

_KEY = (10, 5, 12, LOWER_BOUND_ONLY_PROFILE_ID)


def _evidence(
    horizon: int = 10,
    base: tuple[float, ...] = (-0.01, -0.02, -0.005),
    *,
    profile_id: str = LOWER_BOUND_ONLY_PROFILE_ID,
    rebalance_frequency_sessions: int = 5,
    top_k: int = 12,
) -> HorizonOOFEvidence:
    segment_ids = tuple(index % 3 for index in range(len(base)))
    return HorizonOOFEvidence(
        horizon_sessions=horizon,
        profile_id=profile_id,
        model_family="net_alpha_elastic_net",
        base_log_growth=base,
        stress_log_growth=base,
        cohort_segment_ids=segment_ids,
        complete_cohort_count=len(base),
        active_cohort_count=len(base),
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=max(set(segment_ids), default=0) + 1,
        fold_rank_ics=(0.1,),
        rebalance_frequency_sessions=rebalance_frequency_sessions,
        top_k=top_k,
    )


def test_frozen_stitch_invests_every_interval_when_prequential_is_cash() -> None:
    """SCENARIO_FROZEN_STITCH_ALWAYS_INVESTED."""
    candidate = _evidence()
    route = stitch_prequential_growth_route(
        (candidate,), 0.05, seed=42, n_bootstrap=200
    )
    assert route.observed_interval_count == 3
    assert route.invested_interval_count == 0

    frozen = stitch_frozen_policy_growth_route((candidate,), _KEY)
    assert frozen.observed_interval_count == 3
    assert frozen.invested_interval_count == frozen.observed_interval_count
    assert frozen.filled_orders == 0
    assert frozen.base_log_growth == candidate.base_log_growth
    assert frozen.stress_log_growth == candidate.stress_log_growth
    assert frozen.segment_ids == candidate.cohort_segment_ids
    assert frozen.selected_policies == (_KEY, _KEY, _KEY)
    assert all(policy is not None for policy in frozen.interval_policies)


def test_frozen_stitch_missing_policy_raises() -> None:
    """SCENARIO_FROZEN_KEY_MISSING_FAILS_CLOSED."""
    with pytest.raises(ValueError, match="frozen policy"):
        stitch_frozen_policy_growth_route((), _KEY)
    other_profile = _evidence(profile_id="legacy_overlay_5bps")
    with pytest.raises(ValueError, match="frozen policy"):
        stitch_frozen_policy_growth_route((other_profile,), _KEY)
    other_cell = _evidence(top_k=16)
    with pytest.raises(ValueError, match="frozen policy"):
        stitch_frozen_policy_growth_route((other_cell,), _KEY)


def test_point_cagr_matches_expm1_sum() -> None:
    """SCENARIO_POINT_CAGR_COMPOUND."""
    values = tuple(math.log1p(0.001) for _ in range(252))
    result = point_cagr_from_log_growth(values, annualization_sessions=252)
    expected = math.expm1(math.fsum(values) * 252 / len(values))
    assert result == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError, match="non-empty"):
        point_cagr_from_log_growth((), annualization_sessions=252)
    with pytest.raises(ValueError, match="finite"):
        point_cagr_from_log_growth((0.01, float("nan")), annualization_sessions=252)
    with pytest.raises(ValueError, match="finite"):
        point_cagr_from_log_growth((0.01, float("inf")), annualization_sessions=252)
    with pytest.raises(ValueError, match="annualization_sessions"):
        point_cagr_from_log_growth((0.01,), annualization_sessions=0)


def test_resolve_frozen_policy_key_first_feasible_cell_lower_bound_only() -> None:
    request = NetAlphaTrainingRequest(artifact_id="v1")
    key = resolve_frozen_policy_key(request)
    assert key == (10, 5, 12, LOWER_BOUND_ONLY_PROFILE_ID)


def test_resolve_frozen_policy_key_fail_closed_without_feasible_cell() -> None:
    request = NetAlphaTrainingRequest(
        artifact_id="v1",
        candidate_horizon_sessions=(10,),
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=(10,),
            candidate_rebalance_frequency_sessions=(20,),
            candidate_top_k=(12,),
        ),
    )
    with pytest.raises(ValueError, match="frozen policy"):
        resolve_frozen_policy_key(request)


def test_seed_ladder_growth_optin_prefers_declared_top_rung() -> None:
    """SCENARIO_SEED_LADDER_GROWTH_OPTIN_04."""
    from legacy.stocks.config.research import policy_profiles_with_growth_rungs
    from legacy.stocks.ml.contracts import GROWTH_FULL_UTILIZATION_PROFILE_ID

    growth_request = NetAlphaTrainingRequest(
        artifact_id="growth",
        policy_profiles=policy_profiles_with_growth_rungs(),
    )
    key = resolve_frozen_policy_key(growth_request)
    assert key[3] == GROWTH_FULL_UTILIZATION_PROFILE_ID
    assert key[0] == int(growth_request.candidate_horizon_sessions[0])
    profile = next(
        profile
        for profile in growth_request.policy_profiles
        if profile.profile_id == key[3]
    )
    feasible = growth_request.execution_frontier.feasible_cells_for_profile(
        growth_request.portfolio.max_exposure,
        growth_request.portfolio.max_single_weight,
        single_name_cap_override=profile.single_name_cap_override,
        gross_utilization_target=profile.gross_utilization_target,
    )
    assert (key[0], key[1], key[2]) in feasible

    # Flag-off ladder keeps the legacy lower_bound_only seed byte-identical.
    legacy_request = NetAlphaTrainingRequest(artifact_id="legacy")
    legacy_key = resolve_frozen_policy_key(legacy_request)
    assert legacy_key[3] == LOWER_BOUND_ONLY_PROFILE_ID
    assert legacy_key[:3] == (10, 5, 12)

    # The frozen stitch still fails closed when the declared seed has no
    # matching discovery candidate.
    with pytest.raises(ValueError, match="frozen policy"):
        stitch_frozen_policy_growth_route((), key)


def test_projection_emits_bounded_scalars_only() -> None:
    candidate = _evidence(
        base=tuple(math.log1p(0.002) for _ in range(4)),
    )
    frozen = stitch_frozen_policy_growth_route((candidate,), _KEY)
    projection = frozen_compound_track_projection(frozen, annualization_sessions=252)
    assert set(projection) == {
        "policy",
        "point_cagr",
        "observed_interval_count",
        "invested_interval_count",
        "filled_orders",
        "route_version",
    }
    assert projection["policy"] == [10, 5, 12, LOWER_BOUND_ONLY_PROFILE_ID]
    assert projection["point_cagr"] == pytest.approx(
        math.expm1(252 * math.log1p(0.002)), abs=1e-9
    )
    assert projection["observed_interval_count"] == 4
    assert projection["invested_interval_count"] == 4
    assert projection["filled_orders"] == 0
    assert projection["route_version"] == GROWTH_ROUTE_VERSION


def test_projection_emits_invested_fraction_and_seed() -> None:
    """SCENARIO_GROWTH_ROUTE_SEED_05_PROJECTION_SCALARS."""
    from legacy.stocks.ml.training import _growth_route_projection

    candidate = _evidence(
        base=tuple(math.log1p(0.002) for _ in range(4)),
    )
    route = stitch_prequential_growth_route(
        (candidate,), 0.05, seed=42, n_bootstrap=200, seed_policy=_KEY
    )
    projection = _growth_route_projection(route, {"reasons": []})
    observed = projection["observed_intervals"]
    invested = projection["invested_intervals"]
    assert isinstance(observed, int)
    assert isinstance(invested, int)
    fraction = projection["invested_interval_fraction"]
    assert isinstance(fraction, float)
    assert 0.0 <= fraction <= 1.0
    assert fraction == round(invested / observed, 12)
    assert projection["seed_policy"] == "10:5:12:lower_bound_only"

    cash_projection = _growth_route_projection(
        stitch_prequential_growth_route((candidate,), 0.05, seed=42, n_bootstrap=200),
        {"reasons": ["no-filled-orders"]},
    )
    assert cash_projection["seed_policy"] is None
    assert cash_projection["invested_interval_fraction"] == 0.0
    # Bounded scalars only: no per-interval series ever leaks into the payload.
    for key, value in projection.items():
        assert not isinstance(value, list), key


def test_champion_frozen_track_parity_anchor() -> None:
    """SCENARIO_GROWTH_ROUTE_SEED_06_CHAMPION_PARITY_ANCHOR."""
    import json
    from pathlib import Path

    artifact = Path(
        "data/artifacts/stocks/ml_rawnet_h20_20260824/metrics.json"
    )
    if not artifact.exists():
        pytest.skip("champion run artifact is not present")
    metrics = json.loads(artifact.read_text())
    frozen = metrics["growth_route"]["frozen_compound_track"]
    assert frozen["policy"] == [20, 10, 8, LOWER_BOUND_ONLY_PROFILE_ID]
    assert frozen["point_cagr"] == pytest.approx(0.045781471671, abs=1e-9)
