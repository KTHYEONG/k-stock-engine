"""Executable hedged growth tests (contract skeletons)."""
from __future__ import annotations

def test_build_executable_hedge_evidence_aligns_exact_intervals() -> None:
    from datetime import UTC, datetime
    import math
    import polars as pl
    import pytest
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.core.instruments import AssetKind, Instrument
    from legacy.stocks.ml.contracts import ExecutableOverlayData
    from legacy.stocks.ml.hedge_sleeve import build_executable_hedge_evidence
    from legacy.stocks.ml.horizons import GrowthRouteEvidence

    d0, d1, d2 = (datetime(2024, 1, day, tzinfo=UTC) for day in (2, 3, 4))
    route = GrowthRouteEvidence(base_log_growth=(0.01, 0.02), stress_log_growth=(0.009, 0.018), segment_ids=(0, 0), selected_policies=((10, 10, 12, "lower_bound_only"),), interval_session_pairs=((d0, d1), (d1, d2)))
    frame = pl.DataFrame({"instrument_id": ["KRX:252670"] * 3, "session": [d0, d1, d2], "open": [100.0, 90.0, 99.0], "high": [101.0, 91.0, 100.0], "low": [99.0, 89.0, 98.0], "close": [100.0, 90.0, 99.0], "volume": [1_000_000.0] * 3, "available_time": [d0, d1, d2]})
    overlay = ExecutableOverlayData(instrument=Instrument("KRX:252670", AssetKind.ETF, "KRX", "252670", "KRW"), frame=frame, evidence_hash="a" * 64, base_cost_schedule=default_base_schedule(), stress_cost_schedule=default_stress_schedule(), beta=-2.0)

    evidence = build_executable_hedge_evidence(route, overlay)

    assert evidence.interval_session_pairs == ((d0, d1), (d1, d2))
    assert evidence.base_log_growth == pytest.approx((math.log(0.9), math.log(1.1)))
    assert evidence.stress_log_growth == pytest.approx(evidence.base_log_growth)
    assert evidence.evidence_hash == "a" * 64
    assert evidence.stress_per_side_cost_rate >= evidence.per_side_cost_rate

def test_build_executable_hedge_evidence_rejects_missing_endpoint() -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.core.instruments import AssetKind, Instrument
    from legacy.stocks.ml.contracts import ExecutableOverlayData
    from legacy.stocks.ml.hedge_sleeve import build_executable_hedge_evidence
    from legacy.stocks.ml.horizons import GrowthRouteEvidence

    d0, d1 = datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    route = GrowthRouteEvidence(base_log_growth=(0.01,), stress_log_growth=(0.009,), segment_ids=(0,), selected_policies=((10, 10, 12, "lower_bound_only"),), interval_session_pairs=((d0, d1),))
    frame = pl.DataFrame({"instrument_id": ["KRX:252670"], "session": [d0], "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1_000_000.0], "available_time": [d0]})
    overlay = ExecutableOverlayData(instrument=Instrument("KRX:252670", AssetKind.ETF, "KRX", "252670", "KRW"), frame=frame, evidence_hash="b" * 64, base_cost_schedule=default_base_schedule(), stress_cost_schedule=default_stress_schedule(), beta=-2.0)

    with pytest.raises(ValueError, match="hedge-interval"):
        build_executable_hedge_evidence(route, overlay)

def test_executable_hedged_certificate_uses_combined_account_path() -> None:
    from datetime import UTC, datetime, timedelta
    from legacy.stocks.ml.contracts import AccountCertificationSettings, CompoundingCertificationSettings, HedgeExecutionEvidence, SmallCapitalPlanSettings
    from legacy.stocks.ml.hedge_sleeve import certify_executable_hedged_growth_route
    from legacy.stocks.ml.horizons import GrowthRouteEvidence

    start = datetime(2020, 1, 1, tzinfo=UTC)
    pairs = tuple((start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(504))
    route = GrowthRouteEvidence(base_log_growth=tuple(0.0002 for _ in pairs), stress_log_growth=tuple(0.0001 for _ in pairs), segment_ids=tuple(i // 168 for i in range(len(pairs))), selected_policies=((10, 10, 12, "lower_bound_only"),) * 3, interval_session_pairs=pairs, observed_interval_count=len(pairs), invested_interval_count=len(pairs), filled_orders=100)
    hedge = HedgeExecutionEvidence(tradable_proxy_id="KRX:252670", asset_class="inverse_etf", observed_at=pairs[-1][1], evidence_hash="c" * 64, contract_multiplier=None, decision_price=10_000.0, initial_margin_fraction=1.0, per_side_cost_rate=0.0, stress_per_side_cost_rate=0.0, tax_model={"kind": "etf", "timing": "at_exit", "rate": 0.0}, base_log_growth=tuple(0.003 for _ in pairs), stress_log_growth=tuple(0.0028 for _ in pairs), interval_session_pairs=pairs)

    certificate = certify_executable_hedged_growth_route(route, hedge, SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0), AccountCertificationSettings(10_000_000.0, minimum_lower_cagr=0.30, max_drawdown=0.25), CompoundingCertificationSettings(bootstrap_resamples=200, seed=42))

    assert certificate["passed"] is True
    assert certificate["base_lower_cagr"] >= 0.30
    assert certificate["stress_lower_cagr"] >= 0.30
    assert certificate["mdd"] <= 0.25
    assert certificate["evidence_hash"] == "c" * 64

def test_synthetic_hedge_projection_never_becomes_deployable() -> None:
    from legacy.stocks.ml.contracts import AccountCertificationSettings, CompoundingCertificationSettings, SmallCapitalPlanSettings
    from legacy.stocks.ml.horizons import GrowthRouteEvidence
    from legacy.stocks.ml.training import _growth_route_projection

    route = GrowthRouteEvidence(base_log_growth=tuple(0.001 for _ in range(300)), stress_log_growth=tuple(0.0009 for _ in range(300)), benchmark_log_growth=tuple(-0.001 for _ in range(300)), segment_ids=tuple(i // 100 for i in range(300)), selected_policies=((10, 10, 12, "lower_bound_only"),) * 3, observed_interval_count=300, invested_interval_count=300, filled_orders=100)
    absolute = {"passed": False, "reasons": ["non-positive-base-lower-cagr"], "base_lower_cagr": -0.01, "stress_lower_cagr": -0.02, "matched_lower_excess_cagr": 0.08, "mdd": 0.20, "observed_intervals": 300, "invested_intervals": 300, "filled_orders": 100}

    projection = _growth_route_projection(route, absolute, compounding=CompoundingCertificationSettings(bootstrap_resamples=200), horizon_sessions=10, capital_plan_settings=SmallCapitalPlanSettings(seed_capital_krw=10_000_000.0), account_certification=AccountCertificationSettings(10_000_000.0), executable_overlay_data=None)

    assert projection["promotion_status"] != "PROMOTED_EXECUTABLE_HEDGE"
    assert projection["small_capital_route_plan"]["deployment_verdict"] != "DEPLOYABLE"
    assert "tradable-hedge-evidence-missing" in projection["small_capital_route_plan"]["reasons"]
