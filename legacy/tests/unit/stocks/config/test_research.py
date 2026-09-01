from __future__ import annotations

from legacy.stocks.config.research import CanonicalResearchProfile, ParameterSource


def test_canonical_profile_has_stable_fingerprint() -> None:
    profile = CanonicalResearchProfile()

    assert len(profile.fingerprint()) == 64
    assert ParameterSource.PROFILE.value == "profile"


def test_profile_mode_parity_with_default_policy_profiles() -> None:
    """PROFILE_MODE_PARITY: canonical ladder mirrors contracts-owned v5 modes."""
    from legacy.stocks.ml.contracts import DEFAULT_POLICY_PROFILES
    from legacy.stocks.config.research import policy_profiles_with_excess_full_kelly

    canonical = {p.profile_id: p for p in CanonicalResearchProfile().policy_profiles}
    for default_profile in DEFAULT_POLICY_PROFILES:
        entry = canonical[default_profile.profile_id]
        assert entry.execution_utility_mode == default_profile.execution_utility_mode
        assert entry.sizing_mode == default_profile.sizing_mode
        assert entry.execution_utility_mode == "sparse_hold_replace_v2"

    ladder = policy_profiles_with_excess_full_kelly()
    assert [p.profile_id for p in ladder] == [
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
        "excess_full_kelly",
    ]
    kelly = ladder[-1]
    assert kelly.execution_utility_mode == "sparse_hold_replace_v2"
    assert kelly.sizing_mode == "risk_balanced_waterfill_v2"
    assert kelly.single_name_cap_override == 0.16
    assert kelly.gross_utilization_target == 0.92


def test_SCENARIO_GROWTH_RUNG_BUILDER_ORDER_02() -> None:
    """SCENARIO_GROWTH_RUNG_BUILDER_ORDER_02."""
    from legacy.stocks.ml.contracts import validate_policy_profiles
    from legacy.stocks.config.research import policy_profiles_with_growth_rungs

    ladder = policy_profiles_with_growth_rungs()
    assert [p.profile_id for p in ladder] == [
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
        "excess_full_kelly",
        "growth_full_utilization",
    ]
    rung = ladder[-1]
    assert rung.no_trade_band_bps == 0.0
    assert rung.growth_risk_aversion == 1.0
    assert rung.execution_utility_mode == "sparse_hold_replace_v2"
    assert rung.sizing_mode == "risk_balanced_waterfill_v2"
    assert rung.single_name_cap_override == 0.16
    assert rung.gross_utilization_target == 0.95
    assert rung.vol_target_override == 0.20

    validated = validate_policy_profiles(ladder)
    assert [p.profile_id for p in validated] == [p.profile_id for p in ladder]


def test_SCENARIO_GROWTH_RUNG_LIMITS_DECLARED_02() -> None:
    """SCENARIO_GROWTH_RUNG_LIMITS_DECLARED_02."""
    from legacy.stocks.ml.contracts import validate_policy_profiles
    from legacy.stocks.config.research import policy_profiles_with_growth_rungs

    ladder = policy_profiles_with_growth_rungs()
    assert [p.profile_id for p in ladder] == [
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
        "excess_full_kelly",
        "growth_full_utilization",
    ]
    rung = ladder[-1]
    assert rung.participation_limit_override == 0.02
    assert rung.turnover_budget_override == 0.40
    assert rung.vol_target_override == 0.20
    assert rung.gross_utilization_target == 0.95
    kelly = ladder[3]
    assert kelly.participation_limit_override is None
    assert kelly.turnover_budget_override is None
    assert validate_policy_profiles(ladder)[-1] is rung
