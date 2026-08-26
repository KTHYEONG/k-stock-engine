"""Hedged-excess sleeve certification: certified lower-bound gates, fail-closed."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.stocks.ml.contracts import CompoundingCertificationSettings
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
