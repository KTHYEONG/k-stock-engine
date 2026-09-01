"""Ranking-quality and economic-attribution metrics."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from legacy.stocks.research.metrics import economic_transfer_attribution


def _scored_frame(
    n_sessions: int = 6,
    scores: list[float] | None = None,
) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "session": start + timedelta(days=i),
            "instrument_id": name,
        }
        for i in range(n_sessions)
        for name in ("A", "B", "C", "D", "E")
    ]
    frame = pl.DataFrame(rows)
    default_scores = [0.9, 0.8, 0.7, 0.6, 0.5] * n_sessions
    return frame.with_columns(
        pl.Series("pred_score", scores or default_scores),
        pl.Series("residual_o2o_5d", [0.08, 0.07, 0.06, 0.05, 0.04] * n_sessions),
    )


def test_economic_transfer_attribution_contract_assertion() -> None:
    attribution = economic_transfer_attribution(
        _scored_frame(n_sessions=2), "residual_o2o_5d", 2
    )
    assert attribution["decision_count"] == 2
    assert attribution["retained_session_count"] == 2
    assert attribution["session_coverage"] == 1.0
    assert attribution["positive_rank_ic_session_count"] == 2
    assert attribution["mean_rank_ic"] > 0.0
    assert attribution["mean_top_k_active_label"] > 0.0
    assert set(attribution) == {
        "decision_count",
        "retained_session_count",
        "session_coverage",
        "positive_rank_ic_session_count",
        "mean_rank_ic",
        "mean_top_k_label",
        "mean_universe_label",
        "mean_top_k_active_label",
        "mean_membership_turnover",
    }


def test_economic_transfer_attribution_penalizes_rotating_membership() -> None:
    stable = economic_transfer_attribution(
        _scored_frame(n_sessions=30),
        "residual_o2o_5d",
        2,
    )
    rotating_scores = ([0.9, 0.5, 0.1, 0.3, 0.2] * 15) + (
        [0.1, 0.9, 0.5, 0.3, 0.2] * 15
    )
    rotating = economic_transfer_attribution(
        _scored_frame(n_sessions=30, scores=rotating_scores),
        "residual_o2o_5d",
        2,
    )
    assert stable["mean_membership_turnover"] < rotating["mean_membership_turnover"]
    assert stable["decision_count"] == rotating["decision_count"] == 30
    assert stable["mean_top_k_active_label"] > 0.0


def test_economic_transfer_attribution_excludes_missing_labels_never_zero_fills() -> None:
    frame = _scored_frame(n_sessions=3).with_columns(
        pl.when(pl.col("session") == pl.col("session").max())
        .then(None)
        .otherwise(pl.col("residual_o2o_5d"))
        .alias("residual_o2o_5d")
    )
    attribution = economic_transfer_attribution(frame, "residual_o2o_5d", 2)
    assert attribution["decision_count"] == 2
    assert attribution["session_coverage"] == pytest.approx(2 / 3)
    retained = frame.filter(pl.col("residual_o2o_5d").is_not_null())
    assert retained["residual_o2o_5d"].min() > 0.0


def test_economic_transfer_attribution_empty_valid_frame_returns_zeroes() -> None:
    frame = _scored_frame(n_sessions=2).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("residual_o2o_5d")
    )
    attribution = economic_transfer_attribution(frame, "residual_o2o_5d", 2)
    assert attribution["decision_count"] == 0
    assert attribution["mean_rank_ic"] == 0.0
    assert attribution["mean_membership_turnover"] == 0.0
    assert attribution["session_coverage"] == 0.0


def test_economic_transfer_attribution_rejects_invalid_inputs() -> None:
    frame = _scored_frame(n_sessions=2)
    with pytest.raises(ValueError, match="top_k"):
        economic_transfer_attribution(frame, "residual_o2o_5d", 0)
    with pytest.raises(ValueError, match="requires"):
        economic_transfer_attribution(
            frame.drop("pred_score"), "residual_o2o_5d", 2
        )
    non_finite = frame.with_columns(
        pl.when(pl.col("instrument_id") == "A")
        .then(float("inf"))
        .otherwise(pl.col("pred_score"))
        .alias("pred_score")
    )
    with pytest.raises(ValueError, match="non-finite"):
        economic_transfer_attribution(non_finite, "residual_o2o_5d", 2)

def test_compounded_growth_metrics_is_exact_and_evidence_complete() -> None:
    from legacy.stocks.research.metrics import compounded_growth_metrics

    daily = [0.01] * 252
    metrics = compounded_growth_metrics(daily, 252)
    assert metrics["evidence_complete"] == 1.0
    assert metrics["cagr"] == pytest.approx((1.01) ** 252 - 1, rel=1e-12)
    assert metrics["mdd"] == 0.0
    assert math.isinf(metrics["calmar"])

    short = compounded_growth_metrics([0.01, 0.02], 252)
    expected = math.expm1((math.log1p(0.01) + math.log1p(0.02)) * 252 / 2)
    assert short["evidence_complete"] == 1.0
    assert short["cagr"] == pytest.approx(expected, rel=1e-12)

    drawdown = compounded_growth_metrics(
        [0.01, 0.02, -0.03, 0.01, 0.02], 252
    )
    assert drawdown["evidence_complete"] == 1.0
    assert drawdown["mdd"] > 0.0
    assert drawdown["calmar"] == pytest.approx(
        drawdown["cagr"] / drawdown["mdd"], rel=1e-12
    )

    flat = compounded_growth_metrics([0.0] * 252, 252)
    assert flat["evidence_complete"] == 1.0
    assert flat["cagr"] == 0.0
    assert flat["calmar"] == 0.0


def test_compounded_growth_metrics_fails_closed_on_incomplete_evidence() -> None:
    from legacy.stocks.research.metrics import compounded_growth_metrics

    for invalid in ([], [-1.0], [-1.01], [float("nan")], [float("inf")], [None], [1.0, float("nan")]):
        metrics = compounded_growth_metrics(list(invalid), 252)
        assert metrics["evidence_complete"] == 0.0, invalid
        assert metrics["cagr"] == 0.0
        assert metrics["mdd"] == 0.0
        assert metrics["calmar"] == 0.0
    with pytest.raises(ValueError, match="annualization"):
        compounded_growth_metrics([0.01], 0)


def test_compounded_growth_metrics_contract_assertion() -> None:
    from legacy.stocks.research.metrics import compounded_growth_metrics

    assert compounded_growth_metrics([0.01] * 252, 252)["cagr"] > 0.0

def test_certify_compounded_holdout_passes_positive_growth() -> None:
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    settings = CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=252,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.5,
    )
    returns = [0.01] * 84
    certificate = certify_compounded_holdout(
        returns, returns, horizon_sessions=3, observed_sessions=252,
        active_cohort_count=84, settings=settings,
    )
    assert certificate.passed is True
    assert certificate.reasons == ()
    base = certificate.base
    assert base["passed"] is True
    assert base["period_count"] == 84
    assert base["observed_sessions"] == 252
    assert base["active_cohort_count"] == 84
    assert base["cagr"] == pytest.approx((1.01) ** 84 - 1, rel=1e-9)
    assert base["mdd"] == 0.0
    assert base["calmar"] is None
    assert certificate.stress["passed"] is True
    assert certificate.to_json()["passed"] is True
    assert certificate.to_json()["reasons"] == []


def test_certify_compounded_holdout_bootstrap_is_deterministic() -> None:
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    settings = CompoundingCertificationSettings(
        min_observed_sessions=60,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.9,
        bootstrap_resamples=200,
        seed=7,
    )
    returns = [0.005 + 0.002 * (index % 5) for index in range(60)]
    first = certify_compounded_holdout(
        returns, returns, 1, 60, 60, settings,
    )
    second = certify_compounded_holdout(
        returns, returns, 1, 60, 60, settings,
    )
    assert first.to_json() == second.to_json()
    assert first.base["lower_cagr"] > 0.0


def test_certify_compounded_holdout_fails_closed_on_each_gate() -> None:
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    settings = CompoundingCertificationSettings(
        min_observed_sessions=60,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.1,
    )
    base_settings = CompoundingCertificationSettings(
        min_observed_sessions=60,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.9,
    )

    incomplete = certify_compounded_holdout(
        [], [], 1, 0, 0, settings,
    )
    assert incomplete.passed is False
    assert "period-series-incomplete" in incomplete.reasons

    non_finite = certify_compounded_holdout(
        [0.01, float("nan")], [0.01, float("nan")], 1, 2, 2, settings,
    )
    assert "period-series-incomplete" in non_finite.reasons

    too_short = certify_compounded_holdout(
        [0.01] * 10, [0.01] * 10, 1, 10, 10, settings,
    )
    assert too_short.passed is False
    assert "insufficient-observed-sessions" in too_short.reasons

    no_coverage = certify_compounded_holdout(
        [0.01] * 60, [0.01] * 60, 1, 60, 0, base_settings,
    )
    assert "active-coverage-insufficient" in no_coverage.reasons

    negative = certify_compounded_holdout(
        [-0.01] * 60, [-0.01] * 60, 1, 60, 60, base_settings,
    )
    assert "non-positive-cagr" in negative.reasons
    assert "non-positive-lower-cagr" in negative.reasons

    drawdown = certify_compounded_holdout(
        [-0.30, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        * 5,
        [-0.30, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        * 5,
        1, 60, 60, settings,
    )
    assert "max-drawdown-exceeded" in drawdown.reasons


def test_certify_compounded_holdout_stress_path_must_pass() -> None:
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    settings = CompoundingCertificationSettings(
        min_observed_sessions=60,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.9,
    )
    certificate = certify_compounded_holdout(
        [0.01] * 60, [-0.02] * 60, 1, 60, 60, settings,
    )
    assert certificate.passed is False
    assert certificate.base["passed"] is True
    assert certificate.stress["passed"] is False
    assert "non-positive-cagr" in certificate.reasons


def test_certify_compounded_holdout_rejects_invalid_arguments() -> None:
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    settings = CompoundingCertificationSettings(
        min_observed_sessions=1, min_active_cohort_fraction=0.5, max_drawdown=0.9
    )
    with pytest.raises(ValueError, match="horizon_sessions"):
        certify_compounded_holdout([0.01], [0.01], 0, 1, 1, settings)
    with pytest.raises(ValueError, match="observed_sessions"):
        certify_compounded_holdout([0.01], [0.01], 1, -1, 1, settings)
    with pytest.raises(ValueError, match="active_cohort_count"):
        certify_compounded_holdout([0.01], [0.01], 1, 1, -1, settings)


def test_annualize_bootstrap_lower_cagr_daily_cadence() -> None:
    """CGRA-01-daily-cadence-lower-cagr"""
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    daily_returns = [0.01] * 252
    settings = CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=252,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.5,
        bootstrap_resamples=200,
        seed=42,
    )
    expected = (1.01 ** 252) - 1
    daily_horizon = certify_compounded_holdout(
        daily_returns,
        daily_returns,
        horizon_sessions=1,
        observed_sessions=252,
        active_cohort_count=252,
        settings=settings,
    )
    model_horizon = certify_compounded_holdout(
        daily_returns,
        daily_returns,
        horizon_sessions=10,
        observed_sessions=252,
        active_cohort_count=252,
        settings=settings,
    )
    assert daily_horizon.base["lower_cagr"] == pytest.approx(expected, rel=1e-9)
    assert model_horizon.base["lower_cagr"] == pytest.approx(expected, rel=1e-9)


def test_annualize_bootstrap_lower_cagr_sparse_cadence_preserved() -> None:
    """CGRA-02-sparse-cadence-preserved"""
    from legacy.stocks.research.metrics import (
        annualize_bootstrap_lower_cagr,
        _bootstrap_lower_mean_log_growth,
    )

    returns = [0.01] * 84
    log_growth = __import__("numpy").log1p(__import__("numpy").array(returns))
    lower_mean = _bootstrap_lower_mean_log_growth(log_growth, 10, 200, 42, 0.05)
    result = annualize_bootstrap_lower_cagr(
        lower_mean,
        annualization_sessions=252,
        period_count=84,
        observed_sessions=252,
    )
    expected = (1.01 ** 84) - 1
    assert result == pytest.approx(expected, rel=1e-9)


def test_invalid_observed_evidence_fails_closed() -> None:
    """CGRA-03-invalid-observed-evidence-fails-closed"""
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_compounded_holdout

    settings = CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=252,
        min_active_cohort_fraction=0.5,
        max_drawdown=0.5,
    )
    returns = [0.01] * 84
    certificate = certify_compounded_holdout(
        returns, returns, horizon_sessions=10, observed_sessions=0,
        active_cohort_count=84, settings=settings,
    )
    assert certificate.passed is False
    assert "invalid-observed-sessions" in certificate.reasons
    assert certificate.base["lower_cagr"] == 0.0


def test_compounding_certification_settings_validate_fail_closed() -> None:
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings

    with pytest.raises(ValueError, match="annualization_sessions"):
        CompoundingCertificationSettings(annualization_sessions=0)
    with pytest.raises(ValueError, match="min_observed_sessions"):
        CompoundingCertificationSettings(min_observed_sessions=0)
    with pytest.raises(ValueError, match="min_active_cohort_fraction"):
        CompoundingCertificationSettings(min_active_cohort_fraction=0.0)
    with pytest.raises(ValueError, match="min_active_cohort_fraction"):
        CompoundingCertificationSettings(min_active_cohort_fraction=1.5)
    with pytest.raises(ValueError, match="max_drawdown"):
        CompoundingCertificationSettings(max_drawdown=1.0)
    with pytest.raises(ValueError, match="bootstrap_alpha"):
        CompoundingCertificationSettings(bootstrap_alpha=0.0)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        CompoundingCertificationSettings(bootstrap_resamples=1)


def test_targeted_regression_suite() -> None:
    """CGRA-06-targeted-regression-suite

    Ensures the three-profile frontier is admissible only when both base and
    stress Holm-adjusted lower-growth bounds are strictly positive, and
    simulation still rejects any policy mismatch.
    """
    from legacy.stocks.ml.contracts import (
        DEFAULT_POLICY_PROFILES,
        validate_policy_profiles,
    )

    validated = validate_policy_profiles(DEFAULT_POLICY_PROFILES)
    assert len(validated) == 3
    for profile in validated:
        assert profile.growth_risk_aversion > 0.0


GROWTH_ROUTE_03_CERTIFICATE = "GROWTH_ROUTE_03_CERTIFICATE"


def _growth_route(
    base: tuple[float, ...],
    stress: tuple[float, ...] | None = None,
    benchmark: tuple[float, ...] | None = None,
    *,
    filled_orders: int = 6,
    invested_intervals: int | None = None,
    observed_intervals: int | None = None,
):
    from legacy.stocks.ml.horizons import GrowthRouteEvidence

    count = len(base)
    return GrowthRouteEvidence(
        base_log_growth=base,
        stress_log_growth=stress if stress is not None else base,
        segment_ids=(0,) * count,
        selected_policies=((10, 5, 20, "lower_bound_only"),),
        interval_policies=((10, 5, 20, "lower_bound_only"),) * count,
        benchmark_log_growth=benchmark if benchmark is not None else (),
        candidate_count=3,
        observed_interval_count=observed_intervals if observed_intervals else count,
        invested_interval_count=(
            count if invested_intervals is None else invested_intervals
        ),
        filled_orders=filled_orders,
        filled_cycle_count=filled_orders,
    )


def _route_settings():
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings

    return CompoundingCertificationSettings(
        annualization_sessions=252,
        min_observed_sessions=252,
        min_active_cohort_fraction=0.2,
        max_drawdown=0.5,
        bootstrap_alpha=0.05,
        bootstrap_resamples=200,
        seed=42,
    )


def test_growth_route_03_certificate_passes_positive_paths() -> None:
    """GROWTH_ROUTE_03_CERTIFICATE (positive branch).

    A 252-interval finite route with positive base/stress and matched-excess
    paths, >=20% invested intervals, an order count > 0, and MDD within the
    configured cap passes with all three lower-CAGR predicates strictly
    positive; the certificate is deterministic for fixed inputs.
    """
    from legacy.stocks.research.metrics import certify_growth_route

    route = _growth_route(
        (0.001,) * 252,
        (0.0009,) * 252,
        (0.0005,) * 252,
        invested_intervals=60,
    )
    certificate = certify_growth_route(route, 10, _route_settings())
    assert certificate["passed"] is True
    assert certificate["reasons"] == []
    assert certificate["base_lower_cagr"] > 0.0
    assert certificate["stress_lower_cagr"] > 0.0
    assert certificate["matched_lower_excess_cagr"] > 0.0
    again = certify_growth_route(route, 10, _route_settings())
    assert again == certificate


def test_growth_route_03_certificate_fails_closed_per_predicate() -> None:
    """GROWTH_ROUTE_03_CERTIFICATE (fail-closed branches).

    Independently making the stress lower CAGR <= 0, the matched lower excess
    <= 0, the filled order count zero, or one interval non-finite fails closed
    with that exact normalized predicate.
    """
    from legacy.stocks.research.metrics import certify_growth_route

    settings = _route_settings()
    negative_stress = certify_growth_route(
        _growth_route((0.001,) * 252, (-0.001,) * 252, (0.0005,) * 252), 10, settings
    )
    assert negative_stress["passed"] is False
    assert "non-positive-stress-lower-cagr" in negative_stress["reasons"]

    dominated = certify_growth_route(
        _growth_route((0.001,) * 252, (0.0009,) * 252, (0.004,) * 252), 10, settings
    )
    assert dominated["passed"] is False
    assert "non-positive-matched-lower-excess" in dominated["reasons"]

    no_orders = certify_growth_route(
        _growth_route(
            (0.001,) * 252, (0.0009,) * 252, (0.0005,) * 252, filled_orders=0
        ),
        10,
        settings,
    )
    assert no_orders["passed"] is False
    assert "no-filled-orders" in no_orders["reasons"]

    non_finite_series = (0.001,) * 125 + (float("nan"),) + (0.001,) * 126
    corrupted = _growth_route(
        (0.001,) * 252, (0.0009,) * 252, (0.0005,) * 252
    )
    # The immutable contract rejects non-finite series at construction; the
    # certificate must still fail closed on a corrupted legacy payload.
    object.__setattr__(corrupted, "base_log_growth", non_finite_series)
    broken = certify_growth_route(corrupted, 10, settings)
    assert broken["passed"] is False
    assert "period-series-incomplete" in broken["reasons"]
    assert broken["base_lower_cagr"] is None

    missing_benchmark = certify_growth_route(
        _growth_route((0.001,) * 252, (0.0009,) * 252), 10, settings
    )
    assert missing_benchmark["passed"] is False
    assert "matched-benchmark-missing" in missing_benchmark["reasons"]


def test_growth_route_03_certificate_requires_resolvable_bootstrap() -> None:
    """The route-level bootstrap must resolve its alpha: resamples < ceil(1/alpha)."""
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_growth_route

    weak = CompoundingCertificationSettings(bootstrap_resamples=19)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        certify_growth_route(_growth_route((0.001,) * 252), 10, weak)

def test_SCENARIO_SMALL_ACCOUNT_CAGR_03_TARGET_GATE():
    """SCENARIO_SMALL_ACCOUNT_CAGR_03_TARGET_GATE"""
    from legacy.stocks.ml.horizons import GrowthRouteEvidence
    from legacy.stocks.ml.contracts import CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_growth_route
    settings = CompoundingCertificationSettings(annualization_sessions=252, min_observed_sessions=252, min_active_cohort_fraction=0.2, max_drawdown=0.5, bootstrap_alpha=0.05, bootstrap_resamples=200, seed=42)
    good = GrowthRouteEvidence(base_log_growth=(0.002,)*252, stress_log_growth=(0.002,)*252, segment_ids=(0,)*252, selected_policies=((10,5,20,"lower_bound_only"),), interval_policies=((10,5,20,"lower_bound_only"),)*252, benchmark_log_growth=(0.0005,)*252, candidate_count=1, observed_interval_count=252, invested_interval_count=60, filled_orders=10)
    cert = certify_growth_route(good, 10, settings, minimum_lower_cagr=0.30, max_drawdown=0.25)
    assert cert["passed"] is True
    assert cert["base_lower_cagr"] >= 0.30
    assert cert["stress_lower_cagr"] >= 0.30
    assert cert["mdd"] <= 0.25
    # base below target
    low_base = GrowthRouteEvidence(base_log_growth=(0.0001,)*252, stress_log_growth=(0.002,)*252, segment_ids=(0,)*252, selected_policies=((10,5,20,"lower_bound_only"),), interval_policies=((10,5,20,"lower_bound_only"),)*252, benchmark_log_growth=(0.0005,)*252, candidate_count=1, observed_interval_count=252, invested_interval_count=60, filled_orders=10)
    cert2 = certify_growth_route(low_base, 10, settings, minimum_lower_cagr=0.30, max_drawdown=0.25)
    assert cert2["passed"] is False
    assert "base-lower-cagr-below-target" in cert2["reasons"]
    # stress below target
    low_stress = GrowthRouteEvidence(base_log_growth=(0.002,)*252, stress_log_growth=(0.0001,)*252, segment_ids=(0,)*252, selected_policies=((10,5,20,"lower_bound_only"),), interval_policies=((10,5,20,"lower_bound_only"),)*252, benchmark_log_growth=(0.0005,)*252, candidate_count=1, observed_interval_count=252, invested_interval_count=60, filled_orders=10)
    cert3 = certify_growth_route(low_stress, 10, settings, minimum_lower_cagr=0.30, max_drawdown=0.25)
    assert cert3["passed"] is False
    assert "stress-lower-cagr-below-target" in cert3["reasons"]
    # mdd exceed
    big_dd = GrowthRouteEvidence(base_log_growth=tuple(-0.3 if i==10 else 0.002 for i in range(252)), stress_log_growth=(0.002,)*252, segment_ids=(0,)*252, selected_policies=((10,5,20,"lower_bound_only"),), interval_policies=((10,5,20,"lower_bound_only"),)*252, benchmark_log_growth=(0.0005,)*252, candidate_count=1, observed_interval_count=252, invested_interval_count=60, filled_orders=10)
    cert4 = certify_growth_route(big_dd, 10, settings, minimum_lower_cagr=0.30, max_drawdown=0.25)
    assert cert4["passed"] is False
    assert "max-drawdown-exceeded" in cert4["reasons"]
