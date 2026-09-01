"""Hedge sleeve projection scenarios: SCENARIO_HEDGE_SLEEVE_*, FLAG_OFF parity."""
from __future__ import annotations

import numpy as np
import pytest

from src.stocks.config.research import (
    EXCESS_FULL_KELLY_PROFILE_ID,
    CanonicalResearchProfile,
)
from src.stocks.ml.contracts import DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS
from src.stocks.ml.hedge_sleeve import (
    _vol_managed_scales,
    project_hedge_sleeve,
)


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
    ladder = {
        rung["leverage"]: rung
        for rung in projection["leverage_ladder"]
        if rung["variant"] == "static"
    }
    raw_cagr = projection["excess_point_cagr"]
    assert abs(ladder[1.0]["point_cagr"] - raw_cagr) < 1e-9
    assert ladder[2.0]["point_cagr"] >= 0.30
    assert ladder[2.0]["stress_cagr"] >= 0.25


def _clustered_vol_series() -> list[float]:
    rng = np.random.default_rng(23)
    calm = 0.0008 + 0.005 * rng.standard_normal(650)
    volatile = 0.0008 + 0.020 * rng.standard_normal(600)
    draws = np.concatenate([calm, volatile])
    return (draws - draws.mean() + 0.0008).tolist()


def test_vol_managed_rung_admits_higher_leverage() -> None:
    """SCENARIO_VOL_MANAGED_RUNG_REDUCES_MDD."""
    series = _clustered_vol_series()
    projection = project_hedge_sleeve(
        series, vol_managed_target_annualized_vol=0.08
    )
    by_key = {
        (rung["variant"], rung["leverage"]): rung
        for rung in projection["leverage_ladder"]
    }
    static_15 = by_key[("static", 1.5)]
    volman_15 = by_key[("vol_managed", 1.5)]
    assert static_15["projected_mdd"] > 0.25
    assert volman_15["projected_mdd"] <= 0.25


def test_no_lookahead_in_scaling() -> None:
    """SCENARIO_VOL_MANAGED_NO_LOOKAHEAD."""
    base = _certified_excess_series()
    lookback = 26
    scales_a = _vol_managed_scales(base, lookback=lookback)
    perturbed = list(base)
    k = 400
    perturbed[k + 3] = perturbed[k + 3] + 10.0
    scales_b = _vol_managed_scales(perturbed, lookback=lookback)
    for t in range(k + 1):
        assert scales_a[t] == scales_b[t]
    for t in range(lookback):
        assert scales_a[t] == 0.5


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


def test_executable_hedge_requires_parallel_cash_safe_proxy_series() -> None:
    import pytest

    from src.stocks.ml.contracts import HedgeDeploymentEvidence, SmallCapitalPlanSettings
    from src.stocks.ml.hedge_sleeve import project_executable_hedged_route

    hedge = HedgeDeploymentEvidence(
        tradable_proxy_id="KOSPI200_PROXY", beta=1.0, hedge_base_log_growth=(0.001,),
        hedge_stress_log_growth=(0.001,), base_cost_drag=0.0, stress_cost_drag=0.0,
        initial_margin_fraction=0.15,
    )
    with pytest.raises(ValueError, match="parallel"):
        project_executable_hedged_route((0.001, 0.002), (0.001, 0.002), hedge, SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0))


def test_hedge_execution_evidence_rejects_static_lot_and_missing_tax_model() -> None:
    from datetime import UTC, datetime

    from src.stocks.ml.contracts import HedgeExecutionEvidence, SmallCapitalPlanSettings
    from src.stocks.ml.hedge_sleeve import certify_small_capital_hedge_execution

    evidence = HedgeExecutionEvidence(
        tradable_proxy_id='MINI_KOSPI200', asset_class='index_futures',
        observed_at=datetime(2026, 9, 1, tzinfo=UTC), evidence_hash='a' * 64,
        contract_multiplier=None, decision_price=0.0, initial_margin_fraction=0.15,
        per_side_cost_rate=0.0, tax_model={}, base_log_growth=(0.0,), stress_log_growth=(0.0,),
    )
    verdict = certify_small_capital_hedge_execution(
        (0.0,), (0.0,), evidence, SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0)
    )
    assert verdict['passed'] is False
    assert set(verdict['reasons']) >= {'hedge-price-invalid', 'hedge-multiplier-missing', 'hedge-tax-model-missing'}


def test_hedge_execution_evidence_rejects_stress_variation_margin_cash_breach() -> None:
    from datetime import UTC, datetime

    from src.stocks.ml.contracts import HedgeExecutionEvidence, SmallCapitalPlanSettings
    from src.stocks.ml.hedge_sleeve import certify_small_capital_hedge_execution

    evidence = HedgeExecutionEvidence(
        tradable_proxy_id='MINI_KOSPI200', asset_class='index_futures',
        observed_at=datetime(2026, 9, 1, tzinfo=UTC), evidence_hash='b' * 64,
        contract_multiplier=50_000.0, decision_price=200.0, initial_margin_fraction=0.10,
        per_side_cost_rate=0.0001, tax_model={'kind': 'futures', 'timing': 'per_fill'},
        base_log_growth=(0.0, 0.0), stress_log_growth=(1.0, 1.0),
    )
    verdict = certify_small_capital_hedge_execution(
        (0.0, 0.0), (0.0, 0.0), evidence, SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0, cash_reserve_fraction=0.05)
    )
    assert verdict['passed'] is False
    assert 'stress-variation-margin-cash-breach' in verdict['reasons']
