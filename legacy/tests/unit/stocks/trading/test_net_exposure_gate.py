"""NEM gate scenarios: SCENARIO_NEM_01..06."""
from __future__ import annotations

import json
import math

from legacy.stocks.config.research import (
    policy_profiles_with_growth_rungs,
    policy_profiles_with_unhedged_nem,
)
from legacy.stocks.ml.contracts import NetAlphaTrainingRequest
from legacy.stocks.ml.training import (
    _policy_profile_params,
    _risk_policy_for_profile,
)
from legacy.stocks.trading.portfolio_constructor import (
    StockRiskPolicy,
    apply_net_exposure_gate,
    net_exposure_gate_scale,
    stock_risk_policy_fingerprint,
)


def _gated_policy(**overrides: object) -> StockRiskPolicy:
    return StockRiskPolicy(
        net_exposure_gate_mode="trend_vol_v1",  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def test_off_v1_parity() -> None:
    """SCENARIO_NEM_01_OFF_V1_PARITY."""
    policy = StockRiskPolicy()
    assert policy.net_exposure_gate_mode == "off_v1"
    scale, components = net_exposure_gate_scale([-0.05] * 200, policy)
    assert scale == 1.0
    assert components == {}
    weights = {"a": 0.5, "b": 0.3}
    out, diag = apply_net_exposure_gate(weights, [-0.05] * 200, policy)
    assert out == weights
    assert diag == {}


def test_downtrend_floor() -> None:
    """SCENARIO_NEM_02_DOWNTREND_FLOOR."""
    proxy = [-0.002] * 120
    scale, components = net_exposure_gate_scale(proxy, _gated_policy())
    assert float(components["nem_s_trend"]) == 0.25
    assert scale == 0.25
    out, diag = apply_net_exposure_gate(
        {"a": 0.5, "b": 0.5}, proxy, _gated_policy()
    )
    gross = sum(out.values())
    assert gross <= 0.25 * 1.0 + 1e-12
    assert diag["nem_scale"] == 0.25


def test_uptrend_benign_noop() -> None:
    """SCENARIO_NEM_03_UPTREND_BENIGN_NOOP."""
    proxy = [0.0015, -0.0005] * 60
    weights = {"a": 0.4, "b": 0.4}
    scale, components = net_exposure_gate_scale(proxy, _gated_policy())
    assert scale == 1.0
    assert float(components["nem_s_trend"]) == 1.0
    assert float(components["nem_s_vol"]) == 1.0
    out_gated, diag = apply_net_exposure_gate(weights, proxy, _gated_policy())
    out_off, diag_off = apply_net_exposure_gate(weights, proxy, StockRiskPolicy())
    assert out_gated == out_off == weights
    assert diag
    assert not diag_off


def test_highvol_trim() -> None:
    """SCENARIO_NEM_04_HIGHVOL_TRIM."""
    proxy = [0.02, -0.02] * 60
    policy = _gated_policy()
    scale, components = net_exposure_gate_scale(proxy, policy)
    s_vol = min(
        1.0,
        policy.target_annual_volatility / max(0.02 * math.sqrt(252), 1e-12),
    )
    assert float(components["nem_s_vol"]) < 1.0
    assert 0.25 <= scale < 1.0
    assert abs(scale - max(0.25, 1.0 * s_vol)) < 1e-12


def test_thin_history_fail_open() -> None:
    """SCENARIO_NEM_05_THIN_HISTORY_FAIL_OPEN."""
    scale, components = net_exposure_gate_scale([-0.05] * 30, _gated_policy())
    assert scale == 1.0
    assert components == {"reason": "gate-history-insufficient"}


def test_profile_plumbing_fingerprint() -> None:
    """SCENARIO_NEM_06_PROFILE_PLUMBING_FINGERPRINT."""
    ladder = policy_profiles_with_growth_rungs()
    nem_ladder = policy_profiles_with_unhedged_nem()
    assert nem_ladder[: len(ladder)] == ladder
    rung = nem_ladder[-1]
    assert rung.profile_id == "unhedged_nem_v1"
    assert rung.net_exposure_gate_mode == "trend_vol_v1"

    request = NetAlphaTrainingRequest(artifact_id="nem-test")
    gated = _risk_policy_for_profile(
        request, rung, 10, rebalance_frequency_sessions=5, top_k=8
    )
    assert gated.net_exposure_gate_mode == "trend_vol_v1"
    assert gated.gate_floor == 0.25
    assert gated.gate_trend_lookback_sessions == 60

    growth = ladder[-1]
    ungated = _risk_policy_for_profile(
        request, growth, 10, rebalance_frequency_sessions=5, top_k=8
    )
    assert stock_risk_policy_fingerprint(gated) != stock_risk_policy_fingerprint(
        ungated
    )

    params_off = json.loads(
        _policy_profile_params(
            request, growth, 10, rebalance_frequency_sessions=5, top_k=8
        )
    )
    assert "net_exposure_gate_mode" not in params_off
    params_on = json.loads(
        _policy_profile_params(
            request, rung, 10, rebalance_frequency_sessions=5, top_k=8
        )
    )
    assert params_on["net_exposure_gate_mode"] == "trend_vol_v1"


def test_stock_only_gate_moves_to_cash_when_proxy_history_is_insufficient() -> None:
    from legacy.stocks.trading.portfolio_constructor import StockRiskPolicy, apply_net_exposure_gate, net_exposure_gate_scale

    policy = StockRiskPolicy(
        net_exposure_gate_mode='trend_vol_v1', gate_floor=0.0,
        gate_history_mode='cash_on_insufficient_v1',
    )
    scale, detail = net_exposure_gate_scale([0.001] * 20, policy)
    weights, applied = apply_net_exposure_gate({'KRX:000001': 0.5}, [0.001] * 20, policy)
    assert scale == 0.0
    assert detail == {'reason': 'gate-history-insufficient-cash'}
    assert weights == {'KRX:000001': 0.0}
    assert applied['nem_scale'] == 0.0

    from legacy.stocks.trading.portfolio_allocation import net_exposure_gate_scale as prepared_gate_scale

    prepared_scale, prepared_detail = prepared_gate_scale([0.001] * 20, policy)
    assert prepared_scale == 0.0
    assert prepared_detail == {'reason': 'gate-history-insufficient-cash'}
