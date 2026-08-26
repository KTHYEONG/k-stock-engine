"""Frozen-policy research compound track (always-invested second series).

The live promotion certificate stays ``stitch_prequential_growth_route``;
this module adds one named, pre-registered policy series replayed on every
OOF interval so research fitness is a measurable costed point CAGR instead
of a cash-default ``no-filled-orders`` verdict.
"""
from __future__ import annotations

import math

from src.stocks.ml.contracts import (
    EXCESS_FULL_KELLY_PROFILE_ID,
    GROWTH_FULL_UTILIZATION_PROFILE_ID,
    LOWER_BOUND_ONLY_PROFILE_ID,
    NetAlphaTrainingRequest,
)
from src.stocks.ml.horizons import (
    GROWTH_ROUTE_VERSION,
    GrowthRouteEvidence,
    HorizonOOFEvidence,
    PolicyKey,
)

__all__ = [
    "frozen_compound_track_projection",
    "point_cagr_from_log_growth",
    "resolve_frozen_policy_key",
    "stitch_frozen_policy_growth_route",
]


# Ex-ante seed preference: highest declared utilization rung first. The
# ordering is contract-only (declared profile membership) and never reads
# outcomes, so the seed stays causal by construction. The net-exposure-gated
# rung outranks its ungated sibling only when explicitly declared on the
# request, so flag-off runs resolve exactly as before.
_SEED_PROFILE_PREFERENCE = (
    "unhedged_nem_v1",
    GROWTH_FULL_UTILIZATION_PROFILE_ID,
    EXCESS_FULL_KELLY_PROFILE_ID,
    LOWER_BOUND_ONLY_PROFILE_ID,
)


def resolve_frozen_policy_key(request: NetAlphaTrainingRequest) -> PolicyKey:
    """First feasible ``(H, C, K)`` cell at ``candidate_horizon_sessions[0]``
    under the highest declared seed-profile preference; absent feasibility
    fails closed.

    Preference walks the fixed rung ladder and picks the first profile
    declared on ``request.policy_profiles``, resolving feasibility with that
    profile's own cap overrides. Flag-off requests declare no extra rungs and
    resolve to the legacy ``lower_bound_only`` key unchanged.
    """
    declared = {
        profile.profile_id: profile for profile in request.policy_profiles
    }
    horizon = int(request.candidate_horizon_sessions[0])
    for profile_id in _SEED_PROFILE_PREFERENCE:
        profile = declared.get(profile_id)
        if profile is None:
            continue
        cell = next(
            (
                (h, cadence, top_k)
                for h, cadence, top_k in request.execution_frontier.feasible_cells_for_profile(
                    request.portfolio.max_exposure,
                    request.portfolio.max_single_weight,
                    single_name_cap_override=profile.single_name_cap_override,
                    gross_utilization_target=profile.gross_utilization_target,
                )
                if h == horizon
            ),
            None,
        )
        if cell is not None:
            return (horizon, cell[1], cell[2], profile_id)
    raise ValueError(
        f"frozen policy requires a feasible (H,C,K) cell at horizon "
        f"H={horizon}; refusing to default to cash"
    )


def stitch_frozen_policy_growth_route(
    candidates: tuple[HorizonOOFEvidence, ...],
    key: PolicyKey,
) -> GrowthRouteEvidence:
    """Copy the matching candidate's full log-growth series in cohort order.

    Every interval invests under the frozen policy (O(n) over one series);
    a missing candidate raises ``ValueError`` instead of emitting cash zeros.
    """
    horizon, cadence, top_k, profile_id = key
    candidate = next(
        (
            item
            for item in candidates
            if item.horizon_sessions == horizon
            and item.rebalance_frequency_sessions == cadence
            and item.top_k == top_k
            and item.profile_id == profile_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError(
            f"frozen policy {key!r} matches no discovery candidate; "
            "refusing to emit cash zeros"
        )
    count = len(candidate.base_log_growth)
    return GrowthRouteEvidence(
        base_log_growth=tuple(float(value) for value in candidate.base_log_growth),
        stress_log_growth=tuple(float(value) for value in candidate.stress_log_growth),
        segment_ids=tuple(int(segment) for segment in candidate.cohort_segment_ids),
        selected_policies=tuple(key for _ in range(count)),
        interval_policies=tuple(key for _ in range(count)),
        candidate_count=1,
        observed_interval_count=count,
        invested_interval_count=count,
        filled_orders=int(getattr(candidate, "filled_orders", 0)),
        route_version=GROWTH_ROUTE_VERSION,
    )


def point_cagr_from_log_growth(
    log_growth: tuple[float, ...], *, annualization_sessions: int
) -> float:
    """Annualized point CAGR: ``expm1(sum(logs) * sessions / n)``."""
    if not log_growth:
        raise ValueError("point CAGR requires a non-empty log growth series")
    if annualization_sessions < 1:
        raise ValueError("annualization_sessions must be positive")
    if not all(math.isfinite(float(value)) for value in log_growth):
        raise ValueError("log growth series must be finite")
    total = float(sum(float(value) for value in log_growth))
    return float(math.expm1(total * int(annualization_sessions) / len(log_growth)))


def frozen_compound_track_projection(
    route: GrowthRouteEvidence, *, annualization_sessions: int
) -> dict[str, object]:
    """Bounded scalar-only frozen-track projection; no per-name series."""
    policy = route.selected_policies[-1] if route.selected_policies else None
    if policy is None:
        raise ValueError("frozen compound track requires an invested policy route")
    return {
        "policy": [int(policy[0]), int(policy[1]), int(policy[2]), str(policy[3])],
        "point_cagr": round(
            point_cagr_from_log_growth(
                route.base_log_growth, annualization_sessions=annualization_sessions
            ),
            12,
        ),
        "observed_interval_count": int(route.observed_interval_count),
        "invested_interval_count": int(route.invested_interval_count),
        "filled_orders": int(route.filled_orders),
        "route_version": route.route_version,
    }
