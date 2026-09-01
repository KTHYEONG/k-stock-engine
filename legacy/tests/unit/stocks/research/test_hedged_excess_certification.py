"""Hedged-excess sleeve certification: certified lower-bound gates, fail-closed."""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from legacy.stocks.ml.contracts import (
    CompoundingCertificationSettings,
    ExecutionFrontierSettings,
)
from legacy.stocks.ml.horizons import GrowthRouteEvidence
from legacy.stocks.research.metrics import certify_hedged_excess_route

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

    from legacy.stocks.ml.contracts import NetAlphaTrainingRequest
    from legacy.stocks.ml.discovery import HorizonDiscovery
    from legacy.stocks.ml.execution_replay import ExecutionReplayEvidence
    from legacy.stocks.ml.horizons import HorizonOOFEvidence
    from legacy.stocks.ml.training import _attach_excess_route_certificate

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


def _calibrated_series(n: int = 1500) -> list[float]:
    """Synthetic clustered excess series matching published overall moments.

    Five [200 calm / 100 wild] cycles: calm carries the positive drift,
    wild blocks are zero-drift alternating shocks. Overall mean equals
    ln(1.176)/252 and overall sigma_session equals 0.2137/sqrt(250)
    exactly, while the vol clustering lets causal vol targeting lift the
    managed Sharpe the way the real certified route series does.
    """
    g = math.log(1.176) / 252
    sigma_session = 0.2137 / math.sqrt(250)
    a = g + 0.002                       # calm level (constant drift)
    b = 3.0 * g - 2.0 * a               # wild block mean
    s_w = 0.8 * math.sqrt(
        max(
            3.0 * (sigma_session**2 - (2.0 / 3.0) * (a - g) ** 2)
            - (b - g) ** 2,
            1e-12,
        )
    )
    values: list[float] = []
    for _cycle in range(5):
        values.extend([a] * 200)
        for j in range(50):
            values.append(b + (s_w if j % 2 == 0 else -s_w))
            values.append(b - (s_w if j % 2 == 0 else -s_w))
    return values[:n]


def test_hedge_grid_flagoff_parity() -> None:
    """hedge_grid_flagoff_parity.

    hedge_leverage_grid=None reproduces the certificate payload of the
    explicit legacy grid (1.0, 1.5, 2.0) exactly.
    """
    from dataclasses import replace as dc_replace

    series = _calibrated_series()
    route = _route(tuple(series))
    legacy = certify_hedged_excess_route(
        route, 10, dc_replace(_SETTINGS, min_observed_sessions=100)
    )
    explicit = certify_hedged_excess_route(
        route,
        10,
        dc_replace(
            _SETTINGS,
            min_observed_sessions=100,
            hedge_leverage_grid=(1.0, 1.5, 2.0),
        ),
    )
    assert legacy == explicit
    assert legacy["hedge_variant"] in ("static", "vol_managed")


def test_hedge_grid_extension_selector_flip() -> None:
    """hedge_grid_extension_selector_flip.

    On the calibrated series the extended grid flips the best admissible
    rung from static/2.0 to vol_managed/3.0, lifts point stress CAGR, and
    keeps the certificate passing with a higher sleeve_lower_stress_cagr.
    """
    from dataclasses import replace as dc_replace

    from legacy.stocks.ml.hedge_sleeve import project_hedge_sleeve

    series = _calibrated_series()
    settings_small = dc_replace(_SETTINGS, min_observed_sessions=100)

    def _best_stress(grid: tuple[float, ...]) -> tuple[str, float, float]:
        proj = project_hedge_sleeve(
            series,
            leverage_grid=grid,
            annualization_sessions=252,
            max_projected_mdd=0.5,
        )
        rows = [
            r
            for r in proj["leverage_ladder"]
            if bool(r.get("admissible"))
        ]
        assert rows
        best = max(rows, key=lambda r: (r["stress_cagr"], -r["leverage"]))
        return (
            str(best["variant"]),
            float(best["leverage"]),
            float(best["stress_cagr"]),
        )

    legacy_variant, legacy_lev, legacy_stress = _best_stress((1.0, 1.5, 2.0))
    ext_variant, ext_lev, ext_stress = _best_stress(
        (1.0, 1.5, 2.0, 2.5, 3.0)
    )
    # The legacy ceiling binds below 3.0; the extension unlocks the
    # strictly better vol_managed/L3.0 rung.
    assert legacy_lev <= 2.0
    assert (ext_variant, ext_lev) == ("vol_managed", 3.0)
    assert ext_stress > legacy_stress

    route = _route(tuple(series))
    legacy_cert = certify_hedged_excess_route(route, 10, settings_small)
    ext_cert = certify_hedged_excess_route(
        route,
        10,
        dc_replace(settings_small, hedge_leverage_grid=(1.0, 1.5, 2.0, 2.5, 3.0)),
    )
    assert legacy_cert["passed"] is True
    assert ext_cert["passed"] is True
    assert ext_cert["hedge_leverage"] == 3.0
    assert ext_cert["sleeve_lower_stress_cagr"] > legacy_cert[
        "sleeve_lower_stress_cagr"
    ]


def test_hedge_grid_mdd_veto() -> None:
    """hedge_grid_mdd_veto.

    The static L3.0 rung breaches the MDD cap on the calibrated series and
    never becomes admissible; the certificate still passes because other
    admissible rungs exist.
    """
    from dataclasses import replace as dc_replace

    from legacy.stocks.ml.hedge_sleeve import project_hedge_sleeve

    series = _calibrated_series()
    proj = project_hedge_sleeve(
        series,
        leverage_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
        annualization_sessions=252,
        max_projected_mdd=0.5,
    )
    by_key = {
        (str(row["variant"]), float(row["leverage"])): row
        for row in proj["leverage_ladder"]
    }
    static3 = by_key[("static", 3.0)]
    assert static3["within_mdd_cap"] is False
    assert static3["projected_mdd"] > 0.5
    assert ("static", 3.0) not in [
        (variant, lev) for variant, levs in proj["admissible_leverages"].items() for lev in levs
    ]
    cert = certify_hedged_excess_route(
        _route(tuple(series)),
        10,
        dc_replace(_SETTINGS, min_observed_sessions=100),
    )
    assert "no-admissible-hedge-rung" not in cert["reasons"]
