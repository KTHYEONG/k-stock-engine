from __future__ import annotations

from src.stocks.config.research import CanonicalResearchProfile, ParameterSource


def test_canonical_profile_has_stable_fingerprint() -> None:
    profile = CanonicalResearchProfile()

    assert len(profile.fingerprint()) == 64
    assert ParameterSource.PROFILE.value == "profile"
