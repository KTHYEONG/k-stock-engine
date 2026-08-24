"""Hedge sleeve projection scenarios: SCENARIO_HEDGE_SLEEVE_*, FLAG_OFF parity."""
from __future__ import annotations

import numpy as np
import pytest

from src.stocks.config.research import (
    EXCESS_FULL_KELLY_PROFILE_ID,
    CanonicalResearchProfile,
)
from src.stocks.ml.contracts import DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS
from src.stocks.ml.hedge_sleeve import project_hedge_sleeve


def _certified_excess_series() -> list[float]:
    rng = np.random.default_rng(7)
    sessions = 1250
    drift = 0.0008  # ~20% annualized log growth at 250 sessions/yr
    vol = 0.007  # ~11% annualized vol
    draws = drift + vol * rng.standard_normal(sessions)
    # Center exactly on the target drift so the certified stream's annualized
    # log growth is deterministic regardless of the sampled noise.
    centered = draws - draws.mean() + drift
    return centered.tolist()


def test_leverage_ladder_meets_thirty_percent_at_l2() -> None:
    """SCENARIO_HEDGE_SLEEVE_LADDER."""
    series = _certified_excess_series()
    projection = project_hedge_sleeve(series)
    ladder = {rung["leverage"]: rung for rung in projection["leverage_ladder"]}
    raw_cagr = projection["excess_point_cagr"]
    assert abs(ladder[1.0]["point_cagr"] - raw_cagr) < 1e-9
    assert ladder[2.0]["point_cagr"] >= 0.30
    assert ladder[2.0]["stress_cagr"] >= 0.25


def test_projection_raises_on_empty_series() -> None:
    """SCENARIO_HEDGE_SLEEVE_REJECTS_UNCERTIFIED."""
    with pytest.raises(ValueError, match="excess-series-incomplete"):
        project_hedge_sleeve([])
    with pytest.raises(ValueError, match="excess-series-incomplete"):
        project_hedge_sleeve([0.01, float("nan"), 0.02])


def test_flag_off_profile_grid_unchanged() -> None:
    """SCENARIO_FLAG_OFF_BYTE_PARITY.

    Default profile ids, canonical fingerprint, and frontier cadence grid are
    pinned to their pre-change values; the excess-full-Kelly profile exists as
    an opt-in constant only.
    """
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
    assert DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS == (5, 10, 20)
    assert EXCESS_FULL_KELLY_PROFILE_ID == "excess_full_kelly"
