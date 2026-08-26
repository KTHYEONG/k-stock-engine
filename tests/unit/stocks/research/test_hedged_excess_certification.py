"""Hedged-excess sleeve certification: certified lower-bound gates, fail-closed."""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from src.stocks.ml.contracts import (
    CompoundingCertificationSettings,
    ExecutionFrontierSettings,
)
from src.stocks.ml.horizons import GrowthRouteEvidence
from src.stocks.research.metrics import certify_hedged_excess_route

_SETTINGS = CompoundingCertificationSettings(
    annualization_sessions=252,
    min_observed_sessions=252,
    min_active_cohort_fraction=0.2,
    max_drawdown=0.5,
    bootstrap_alpha=0.05,
    bootstrap_resamples=2000,
    seed=42,
)


def _route(
    excess: tuple[float, ...],
    *,
    benchmark: tuple[float, ...] | None = None,
    filled_orders: int = 500,
    invested: int | None = None,
    horizon: int = 10,
) -> GrowthRouteEvidence:
    benchmark_series = (
        tuple(0.0 for _ in excess) if benchmark is None else benchmark
    )
    return GrowthRouteEvidence(
        base_log_growth=excess,
        stress_log_growth=excess,
        segment_ids=tuple(0 for _ in excess),
        selected_policies=((horizon, 5, 12, "lower_bound_only"),),
        benchmark_log_growth=benchmark_series,
        observed_interval_count=len(excess),
        invested_interval_count=(
            len(excess) if invested is None else invested
        ),
        filled_orders=filled_orders,
    )


def test_positive_drift_excess_certifies_with_best_rung() -> None:
    """SCENARIO_HEDGED_EXCESS_CERT_PASS_01."""
    rng = np.random.default_rng(7)
    excess = tuple(0.002 + float(v) for v in rng.normal(0.0, 0.012, size=2500))
    payload = certify_hedged_excess_route(_route(excess), 10, _SETTINGS)
    assert payload["passed"] is True
    assert payload["reasons"] == []
    assert isinstance(payload["excess_lower_cagr"], float)
    assert payload["excess_lower_cagr"] > 0.0
    assert payload["sleeve_lower_stress_cagr"] > 0.0
    assert payload["hedge_leverage"] == 2.0
    assert payload["hedge_variant"] in ("static", "vol_managed")
    assert set(payload) == {
        "passed",
        "reasons",
        "excess_lower_cagr",
        "sleeve_lower_stress_cagr",
        "hedge_variant",
        "hedge_leverage",
        "hedge_point_cagr",
        "hedge_stress_cagr",
        "hedge_projected_mdd",
        "hedge_margin_buffer",
        "observed_intervals",
        "invested_intervals",
        "filled_orders",
    }
    assert payload["hedge_projected_mdd"] <= _SETTINGS.max_drawdown
    for value in payload.values():
        if isinstance(value, float):
            assert round(value, 12) == value


def test_fail_closed_gates() -> None:
    """SCENARIO_HEDGED_EXCESS_CERT_FAIL_CLOSED_02."""
    rng = np.random.default_rng(3)
    noise = tuple(float(v) for v in rng.normal(0.0, 0.02, size=1500))
    noisy = certify_hedged_excess_route(_route(noise), 10, _SETTINGS)
    assert noisy["passed"] is False
    assert "non-positive-excess-lower-cagr" in noisy["reasons"]

    bare = GrowthRouteEvidence(
        base_log_growth=(0.001,) * 400,
        stress_log_growth=(0.001,) * 400,
        segment_ids=tuple(0 for _ in range(400)),
        selected_policies=((10, 5, 12, "lower_bound_only"),),
        observed_interval_count=400,
        invested_interval_count=400,
        filled_orders=100,
    )
    missing = certify_hedged_excess_route(bare, 10, _SETTINGS)
    assert missing["passed"] is False
    assert "matched-benchmark-missing" in missing["reasons"]

    unfilled = certify_hedged_excess_route(
        _route((0.001,) * 400, filled_orders=0), 10, _SETTINGS
    )
    assert unfilled["passed"] is False
    assert "no-filled-orders" in unfilled["reasons"]

    thin = certify_hedged_excess_route(
        _route((0.001,) * 400, invested=40), 10, _SETTINGS
    )
    assert thin["passed"] is False
    assert "invested-coverage-insufficient" in thin["reasons"]

    short = certify_hedged_excess_route(
        _route((0.001,) * 100), 10, _SETTINGS
    )
    assert short["passed"] is False
    assert "insufficient-observed-sessions" in short["reasons"]

    crash = tuple(([0.002] * 60 + [-1.2]) * 5)
    crashed = certify_hedged_excess_route(
        _route(crash),
        10,
        CompoundingCertificationSettings(
            bootstrap_resamples=200, seed=42, max_drawdown=0.25
        ),
    )
    assert crashed["passed"] is False
    assert "no-admissible-hedge-rung" in crashed["reasons"]

    with pytest.raises(ValueError, match="resolvable minimum"):
        certify_hedged_excess_route(
            _route((0.001,) * 400),
            10,
            CompoundingCertificationSettings(bootstrap_resamples=19),
        )


def test_vol_clustered_series_selects_vol_managed_under_cap() -> None:
    """SCENARIO_VOL_CLUSTERED_VOLMANAGED_03."""
    rng = np.random.default_rng(11)
    regimes = np.where(np.arange(3000) % 40 < 30, 0.008, 0.028)
    values = tuple(
        float(0.002 + regime * draw)
        for regime, draw in zip(regimes, rng.standard_normal(3000), strict=True)
    )
    capped = CompoundingCertificationSettings(
        bootstrap_resamples=2000, seed=42, max_drawdown=0.25
    )
    payload = certify_hedged_excess_route(_route(values), 10, capped)
    assert payload["passed"] is True
    assert payload["hedge_variant"] == "vol_managed"
    assert payload["hedge_leverage"] == 2.0
    assert payload["hedge_projected_mdd"] <= 0.25


def test_lower_stress_gate_uses_lower_bound_not_point_estimate() -> None:
    """Sleeve stress gate must reject when only the point estimate is positive."""
    # Near-zero-mean, low-vol stream: point stress at L=1.0 can stay positive
    # while the certified lower bound is negative.
    rng = np.random.default_rng(5)
    marginal = tuple(float(v) for v in rng.normal(0.00005, 0.004, size=1200))
    payload = certify_hedged_excess_route(_route(marginal), 10, _SETTINGS)
    assert payload["passed"] is False
    assert "non-positive-excess-lower-cagr" in payload["reasons"]
    assert not math.isclose(
        float(payload["sleeve_lower_stress_cagr"]), 0.0, abs_tol=1e-12
    )


def test_excess_route_sleeve_upgrade_gatekeeping() -> None:
    """excess_route_sleeve_upgrade_gatekeeping.

    A passing excess-route certificate upgrades a research-only outcome to
    PROMOTED_EXCESS_SLEEVE with provenance when the primary sleeve failed and
    no blocking gate reason survives; any blocking gate reason (or a passing
    primary sleeve) keeps the primary verdict authoritative. Flag-off callers
    receive the projection byte-identical, and outcome.promoted stays False.
    """
    from dataclasses import replace as dc_replace
    from datetime import UTC, datetime

    import polars as pl

    from src.stocks.ml.contracts import NetAlphaTrainingRequest
    from src.stocks.ml.discovery import HorizonDiscovery
    from src.stocks.ml.execution_replay import ExecutionReplayEvidence
    from src.stocks.ml.horizons import HorizonOOFEvidence
    from src.stocks.ml.training import _attach_excess_route_certificate

    sessions = [datetime(2024, 3, 1, tzinfo=UTC) + timedelta(days=i) for i in range(40)]
    rows = []
    for session in sessions:
        for t in range(3):
            price = 40.0 + t
            rows.append(
                {
                    "instrument_id": f"KRX:{t + 1:05d}",
                    "session": session,
                    "observation_time": session.replace(hour=15, minute=30),
                    "available_time": session.replace(hour=15, minute=31),
                    "open": price,
                    "close": price * 1.01,
                    "volume": 1_000_000.0,
                    "trading_value": 100_000_000.0,
                    "sector": "S0",
                    "adtv": 100_000_000.0,
                }
            )
    panel = pl.DataFrame(rows)
    growth_length = 36
    key = (10, 5, 12, "lower_bound_only")
    bounds = (tuple(sessions[: growth_length + 1]),)
    evidence = ExecutionReplayEvidence(
        base_log_growth=tuple(0.002 for _ in range(growth_length)),
        stress_log_growth=tuple(0.0015 for _ in range(growth_length)),
        segment_ids=tuple(0 for _ in range(growth_length)),
        planned_cycles=4,
        filled_orders=24,
        cash_session_fraction=0.0,
        turnover=0.5,
        observed_interval_count=growth_length,
        invested_interval_count=growth_length,
        invested_interval_fraction=1.0,
        base_interval_exposure=tuple(0.9 for _ in range(growth_length)),
        stress_interval_exposure=tuple(0.9 for _ in range(growth_length)),
        base_interval_session_bounds=bounds,
    )
    oof = HorizonOOFEvidence(
        horizon_sessions=key[0],
        profile_id=key[3],
        model_family="net_alpha_elastic_net",
        base_log_growth=tuple(0.002 for _ in range(growth_length)),
        stress_log_growth=tuple(0.0015 for _ in range(growth_length)),
        cohort_segment_ids=tuple(0 for _ in range(growth_length)),
        complete_cohort_count=growth_length,
        active_cohort_count=growth_length,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=1,
        fold_rank_ics=(0.2,),
        rebalance_frequency_sessions=key[1],
        top_k=key[2],
    )
    discovery = HorizonDiscovery(
        evidence=(oof,),
        diagnostics=(),
        oof_by_horizon={},
        execution_evidence_by_candidate={key: evidence},
    )
    small_settings = CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=growth_length,
        min_active_cohort_fraction=0.2,
        max_drawdown=0.5,
        bootstrap_alpha=0.05,
        bootstrap_resamples=2000,
        seed=42,
    )
    request = NetAlphaTrainingRequest(
        artifact_id="na_excess_gate",
        candidate_horizon_sessions=(10,),
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=(10,),
            candidate_rebalance_frequency_sessions=(5,),
            candidate_top_k=(12,),
        ),
        enable_excess_route=True,
        compounding=small_settings,
    )

    def _projection(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": "v2",
            "promotion_status": "NO_TRADE",
            "candidate_count": 1,
            "segment_count": 1,
            "cash_segment_count": 0,
            "selected_policy": "10:5:12:lower_bound_only",
            "observed_intervals": growth_length,
            "invested_intervals": growth_length,
            "filled_orders": 24,
            "rejection_reason_counts": {"non-positive-base-lower-cagr": 1},
            "benchmark_reconcile_failure": "",
            "hedge_sleeve_projection": {},
        }
        payload.update(overrides)
        return payload

    # Upgrade path: primary sleeve absent/failing, no blocking reasons.
    growth_route = _attach_excess_route_certificate(
        _projection(), discovery, request, panel, key[0]
    )
    block = growth_route["excess_route"]
    assert block["passed"] is True
    assert block["provenance"] == "excess-route-v1"
    assert block["filled_orders"] == 24
    assert growth_route["promotion_status"] == "PROMOTED_EXCESS_SLEEVE"
    sourced = growth_route["hedged_excess_certificate"]
    assert sourced["provenance"] == "excess-route-v1"
    assert sourced["passed"] is True

    # Blocking gates veto the upgrade even with a passing excess certificate.
    blocked = _attach_excess_route_certificate(
        _projection(rejection_reason_counts={"no-filled-orders": 1}),
        discovery,
        request,
        panel,
        key[0],
    )
    assert blocked["excess_route"]["passed"] is True
    assert blocked["promotion_status"] == "NO_TRADE"
    assert "hedged_excess_certificate" not in blocked

    # A passing primary sleeve stays authoritative and untouched.
    primary_pass = _projection(
        hedged_excess_certificate={"passed": True, "reasons": []}
    )
    kept = _attach_excess_route_certificate(primary_pass, discovery, request, panel, key[0])
    assert kept["hedged_excess_certificate"] == {"passed": True, "reasons": []}
    assert "excess_route" not in kept or kept["excess_route"]["passed"] is True
    assert kept["promotion_status"] == "NO_TRADE"

    # Flag-off callers are untouched.
    off = dc_replace(request, enable_excess_route=False)
    untouched = _attach_excess_route_certificate(
        _projection(), discovery, off, panel, key[0]
    )
    assert untouched == _projection()

    # The research verdict never promotes the artifact itself.
    assert growth_route.get("promoted", False) is False
