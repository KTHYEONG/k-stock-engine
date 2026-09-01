from __future__ import annotations

from legacy.stocks.trading.policy import ExecutionUtility, SizingMethod


def test_policy_modes_are_explicit() -> None:
    assert ExecutionUtility.SPARSE_HOLD_REPLACE.value.endswith("v2")
    assert SizingMethod.RISK_BALANCED_WATERFILL.value.endswith("v2")


def test_risk_policy_fingerprint_is_compatible_across_import_paths() -> None:
    from legacy.stocks.trading.policy import StockRiskPolicy as PolicyPolicy
    from legacy.stocks.trading.portfolio_constructor import StockRiskPolicy as PortfolioPolicy
    from legacy.stocks.trading.policy import stock_risk_policy_fingerprint as policy_fp
    from legacy.stocks.trading.portfolio_constructor import stock_risk_policy_fingerprint as portfolio_fp

    assert PolicyPolicy is PortfolioPolicy
    # dataclass field values equality
    p1 = PolicyPolicy(top_k=20, gross_cap=0.9, single_name_cap=0.08, participation_limit=0.005)
    p2 = PortfolioPolicy(top_k=20, gross_cap=0.9, single_name_cap=0.08, participation_limit=0.005)
    assert p1 == p2
    # validation errors equality
    import pytest

    with pytest.raises(ValueError, match="top_k") as e1:
        PolicyPolicy(top_k=0)
    with pytest.raises(ValueError, match="top_k") as e2:
        PortfolioPolicy(top_k=0)
    assert type(e1.value) is type(e2.value)
    assert str(e1.value) == str(e2.value)
    # fingerprint equality
    assert policy_fp(p1) == portfolio_fp(p2)
    # fingerprint is SHA-256 hex
    import hashlib
    import json

    payload = json.dumps(
        {
            "top_k": 20,
            "target_count": None,
            "enter_rank": 15,
            "keep_rank": 30,
            "gross_cap": 0.9,
            "single_name_cap": 0.08,
            "sector_cap": 0.25,
            "participation_limit": 0.005,
            "no_trade_band_bps": 0.0,
            "target_annual_volatility": 0.12,
            "turnover_budget": 0.2,
            "volatility_lookback_sessions": 20,
            "covariance_lookback_sessions": 60,
            "rebalance_frequency_sessions": 5,
            "annualization_sessions": 252,
            "economic_hysteresis": True,
            "compounding": {"enabled": True, "growth_risk_aversion": 1.0, "forecast_horizon_sessions": None},
            "economic_ranking_mode": "raw_score_v1",
            "execution_utility_mode": "legacy_target_interpolation_v1",
            "sizing_mode": "alpha_vol_squared_v1",
            "economic_gate_mode": "lower_bound_v1",
            "retained_sizing_mode": "freeze_v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()
    assert policy_fp(p1) == expected
    assert portfolio_fp(p2) == expected
