"""Gross-utilization override scenarios: SCENARIO_PROFILE_CAP_*, FLAG_OFF parity."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.stocks.config.research import CanonicalResearchProfile
from src.stocks.ml.contracts import PolicyProfile
from src.stocks.ml.hedge_sleeve import project_hedge_sleeve
from src.stocks.ml.training import _risk_policy_for_profile


def _request(max_exposure: float = 0.90):
    return SimpleNamespace(
        enable_sparse_retained_rewaterfill=False,
        portfolio=SimpleNamespace(
            max_exposure=max_exposure,
            max_single_weight=0.08,
            participation_limit=0.005,
        )
    )


def test_override_fields_shape_risk_policy() -> None:
    """SCENARIO_PROFILE_CAP_OVERRIDE_APPLIED."""
    overridden = PolicyProfile(
        profile_id="excess_full_kelly",
        no_trade_band_bps=0.0,
        growth_risk_aversion=1.0,
        single_name_cap_override=0.125,
        gross_utilization_target=0.92,
    )
    policy = _risk_policy_for_profile(
        _request(), overridden, 10, rebalance_frequency_sessions=5, top_k=8
    )
    # K=8 equal-weight basis: min(override, 1/K) == 0.125
    assert policy.single_name_cap == 0.125
    # gross utilization target clamped by request exposure cap
    assert policy.gross_cap == pytest.approx(min(0.92, 0.90))

    plain = PolicyProfile(
        profile_id="lower_bound_only",
        no_trade_band_bps=0.0,
        growth_risk_aversion=1.0,
    )
    baseline = _risk_policy_for_profile(
        _request(), plain, 10, rebalance_frequency_sessions=5, top_k=8
    )
    assert baseline.single_name_cap == 0.08
    assert baseline.gross_cap == 0.90

    # K=12 resolution: min(0.16 ceiling, 1/12)
    ceiling = PolicyProfile(
        profile_id="excess_full_kelly",
        single_name_cap_override=0.16,
        gross_utilization_target=0.92,
    )
    policy_k12 = _risk_policy_for_profile(
        _request(), ceiling, 10, rebalance_frequency_sessions=5, top_k=12
    )
    assert policy_k12.single_name_cap == pytest.approx(1.0 / 12)


def test_invalid_override_rejected() -> None:
    """SCENARIO_OVERRIDE_VALIDATION_FAIL_CLOSED."""
    for bad in (0.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="override"):
            PolicyProfile(profile_id="p", single_name_cap_override=bad)
        with pytest.raises(ValueError, match="override"):
            PolicyProfile(profile_id="p", gross_utilization_target=bad)
    PolicyProfile(profile_id="p", single_name_cap_override=None)
    PolicyProfile(profile_id="p", gross_utilization_target=None)


def test_flag_off_defaults_unchanged() -> None:
    """SCENARIO_FLAG_OFF_PARITY_LEAP + SCENARIO_CLOSEOUT_FLAG_OFF_PARITY."""
    profile = CanonicalResearchProfile()
    assert tuple(p.profile_id for p in profile.policy_profiles) == (
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
    )
    assert (
        profile.fingerprint()
        == "ba5c69811d74d34e76450af1e43b0fc7c27f818222add379a39fad8a8aa576ed"
    )

    rng = np.random.default_rng(11)
    series = (0.0008 + 0.007 * rng.standard_normal(1250)).tolist()
    total_log = math.fsum(series)
    years = 1250 / 250
    projection = project_hedge_sleeve(series)
    ladder = projection["leverage_ladder"]
    static_first = [
        rung for rung in ladder if rung["variant"] == "static"
    ]
    assert [rung["leverage"] for rung in static_first] == [1.0, 1.5, 2.0]
    for rung in static_first:
        expected = float(np.expm1(rung["leverage"] * total_log / years))
        assert abs(rung["point_cagr"] - expected) < 1e-12
