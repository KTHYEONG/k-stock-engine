"""Satellite overlay scenarios: SCENARIO_SAT_01..06."""
from __future__ import annotations

import math

from legacy.stocks.config.research import (
    policy_profiles_with_unhedged_nem,
    policy_profiles_with_unhedged_stack,
)
from legacy.stocks.ml.contracts import (
    ALLOWED_EXTRA_PROFILE_IDS,
    ExecutionFrontierSettings,
    NetAlphaTrainingRequest,
    SatelliteOverlaySettings,
)
from legacy.stocks.ml.compound_track import resolve_frozen_policy_key
from legacy.stocks.ml.horizons import GrowthRouteEvidence
from legacy.stocks.ml.satellite_overlay import project_satellite_overlay
from legacy.stocks.ml.training import _growth_route_projection


def _mdd(series: list[float]) -> float:
    if not series:
        return 0.0
    eq = peak = 1.0
    mdd = 0.0
    for value in series:
        eq *= math.exp(value)
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return mdd


def _cagr(series: list[float]) -> float:
    if not series:
        return 0.0
    return math.expm1(math.fsum(series) * 252 / len(series))


def _benign() -> dict[str, float]:
    return {
        "gross_pre_nem": 0.90,
        "gross_post_nem": 0.90,
        "nem_s_trend": 1.0,
        "nem_s_vol": 1.0,
    }


def _risk_off(freed_pre: float = 0.25, freed_post: float = 0.0625) -> dict[str, float]:
    return {
        "gross_pre_nem": freed_pre,
        "gross_post_nem": freed_post,
        "nem_s_trend": 0.25,
        "nem_s_vol": 1.0,
    }


def _bear_inputs() -> tuple[list[float], list[float], list[dict[str, float]]]:
    base = [0.001] * 100 + [-0.004] * 100
    bench = [0.0005] * 100 + [-0.002] * 100
    comps = [_benign()] * 100 + [_risk_off()] * 100
    return base, bench, comps


def test_downtrend_hedge_improves_mdd() -> None:
    """SCENARIO_SAT_01_DOWNTREND_HEDGE_IMPROVES_MDD."""
    base, bench, comps = _bear_inputs()
    out = project_satellite_overlay(
        base,
        base,
        bench,
        comps,
        SatelliteOverlaySettings(enabled=True),
    )
    assert out["verdict"] == "WITHIN_BUDGET"
    combined = math.fsum(out["combined_log_growth"])
    raw = math.fsum(base)
    assert abs(combined - raw) > 0.0
    assert out["combined_mdd"] < _mdd(base)
    assert out["combined_mdd"] <= 0.35
    assert out["combined_point_cagr"] > _cagr(base)


def test_benign_uptrend_noop() -> None:
    """SCENARIO_SAT_02_BENIGN_UPTREND_NOOP."""
    base = [0.0015] * 150
    bench = [0.001] * 150
    comps = [_benign()] * 150
    settings = SatelliteOverlaySettings(enabled=True, cash_reserve_cap=0.90)
    out = project_satellite_overlay(base, base, bench, comps, settings)
    assert [round(v, 12) for v in out["combined_log_growth"]] == [
        round(v, 12) for v in base
    ]
    assert abs(out["combined_point_cagr"] - _cagr(base)) < 1e-11
    assert abs(out["combined_mdd"] - _mdd(base)) < 1e-12


def test_costs_bite_negative_control() -> None:
    """SCENARIO_SAT_03_COSTS_BITE_NEGATIVE_CONTROL."""
    base, bench, comps = _bear_inputs()
    taxed = project_satellite_overlay(
        base,
        base,
        bench,
        comps,
        SatelliteOverlaySettings(enabled=True),
    )
    cheap = project_satellite_overlay(
        base,
        base,
        bench,
        comps,
        SatelliteOverlaySettings(
            enabled=True,
            gain_tax_rate=0.000001,
            fee_bps_annual=0.0,
            spread_bps=0.0,
        ),
    )
    assert taxed["combined_point_cagr"] < cheap["combined_point_cagr"]


def test_inputs_insufficient_fail_open() -> None:
    """SCENARIO_SAT_04_INPUTS_INSUFFICIENT_FAIL_OPEN."""
    for base, bench, comps in (
        ([], [], []),
        ([0.01, 0.02], [0.01], []),
    ):
        out = project_satellite_overlay(
            base,
            base,
            bench,
            comps,
            SatelliteOverlaySettings(enabled=True),
        )
        assert out["verdict"] == "INPUTS_INSUFFICIENT"
        assert out["reasons"] == ["satellite-inputs-insufficient"]
        assert abs(out["combined_point_cagr"] - _cagr(list(base))) < 1e-11
        assert abs(out["combined_mdd"] - _mdd(list(base))) < 1e-12


def test_wiring_flag_off_parity() -> None:
    """SCENARIO_SAT_05_WIRING_FLAG_OFF_PARITY."""
    base, bench, comps = _bear_inputs()
    route = GrowthRouteEvidence(
        base_log_growth=tuple(base),
        stress_log_growth=tuple(base),
        benchmark_log_growth=tuple(bench),
        segment_ids=(0,) * len(base),
        selected_policies=((10, 5, 8, "unhedged_nem_v1"),),
        observed_interval_count=len(base),
        invested_interval_count=len(base),
        filled_orders=len(base),
    )
    certificate: dict[str, object] = {"reasons": []}
    legacy = _growth_route_projection(route, certificate)
    assert "satellite_overlay_projection" not in legacy
    planned = _growth_route_projection(
        route,
        certificate,
        satellite_settings=SatelliteOverlaySettings(enabled=True),
        nem_component_records=comps,
    )
    section = planned["satellite_overlay_projection"]
    assert section["verdict"] in ("WITHIN_BUDGET", "MDD_EXCEEDED")
    shared_planned = {
        key: value
        for key, value in planned.items()
        if key != "satellite_overlay_projection"
    }
    assert shared_planned == legacy


def test_stack_ladder_and_seed_preference() -> None:
    """SCENARIO_SAT_06_STACK_LADDER_AND_SEED_PREFERENCE."""
    nem_ladder = policy_profiles_with_unhedged_nem()
    stack = policy_profiles_with_unhedged_stack()
    assert stack[: len(nem_ladder)] == nem_ladder
    rung = stack[-1]
    assert rung.profile_id == "unhedged_stack_v1"
    assert rung.single_name_cap_override == 0.25
    assert rung.net_exposure_gate_mode == "trend_vol_v1"
    assert "unhedged_stack_v1" in ALLOWED_EXTRA_PROFILE_IDS

    request = NetAlphaTrainingRequest(
        artifact_id="sat-test",
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=(10,),
            candidate_rebalance_frequency_sessions=(5,),
            candidate_top_k=(4,),
        ),
        policy_profiles=policy_profiles_with_unhedged_stack(),
    )
    key = resolve_frozen_policy_key(request)
    assert key[3] == "unhedged_stack_v1"
