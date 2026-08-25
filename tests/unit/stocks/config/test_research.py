from __future__ import annotations

from src.stocks.config.research import CanonicalResearchProfile, ParameterSource


def test_canonical_profile_has_stable_fingerprint() -> None:
    profile = CanonicalResearchProfile()

    assert len(profile.fingerprint()) == 64
    assert ParameterSource.PROFILE.value == "profile"


def test_profile_mode_parity_with_default_policy_profiles() -> None:
    """PROFILE_MODE_PARITY: canonical ladder mirrors contracts-owned v5 modes."""
    from src.stocks.ml.contracts import DEFAULT_POLICY_PROFILES
    from src.stocks.config.research import policy_profiles_with_excess_full_kelly

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
