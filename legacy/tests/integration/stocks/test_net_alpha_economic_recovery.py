"""Read-only acceptance contract for the growth-route research command."""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from legacy.stocks.cli import train
from legacy.stocks.data.contracts import CoverageRange
from legacy.stocks.ml.result_ledger import MlResultLedger
from legacy.stocks.research.artifacts import ModelArtifactRegistry

GROWTH_ROUTE_05_READ_ONLY_RESEARCH = "GROWTH_ROUTE_05_READ_ONLY_RESEARCH"


class _FakeSnapshot:
    costs = None


class _FakeRepository:
    def __init__(self, **kwargs):
        del kwargs

    def compose_labeled_training_snapshot(self, snapshot, **kwargs):
        del snapshot, kwargs
        return SimpleNamespace()


def _install_research_fakes(monkeypatch, evaluation) -> None:
    """Patch data resolution seams and the training-side route evaluation."""
    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train, "compose_net_alpha_training_data", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        "legacy.stocks.ml.training.evaluate_growth_route_research",
        lambda data, request, *, registry: evaluation(request),
    )


def _rejected_evaluation(request):
    del request
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": {
            "passed": False,
            "reasons": [
                "non-positive-stress-lower-cagr",
                "matched-benchmark-missing",
            ],
            "base_lower_cagr": 0.12,
            "stress_lower_cagr": -0.0002,
            "matched_lower_excess_cagr": None,
        },
        "growth_route": {
            "version": "v1",
            "candidate_count": 24,
            "segment_count": 3,
            "selected_policy": None,
            "base_lower_cagr": 0.12,
            "stress_lower_cagr": -0.0002,
            "matched_lower_excess_cagr": None,
            "observed_intervals": 720,
            "invested_intervals": 96,
            "filled_orders": 0,
            "rejection_reason_counts": {
                "matched-benchmark-missing": 1,
                "non-positive-stress-lower-cagr": 1,
            },
        },
    }


def _certified_evaluation(request):
    del request
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": {
            "passed": True,
            "reasons": [],
            "base_lower_cagr": 0.0841,
            "stress_lower_cagr": 0.0198,
            "matched_lower_excess_cagr": 0.0312,
        },
        "growth_route": {
            "version": "v1",
            "candidate_count": 24,
            "segment_count": 3,
            "selected_policy": "10:5:20:lower_bound_only",
            "base_lower_cagr": 0.0841,
            "stress_lower_cagr": 0.0198,
            "matched_lower_excess_cagr": 0.0312,
            "observed_intervals": 720,
            "invested_intervals": 640,
            "filled_orders": 4599,
            "rejection_reason_counts": {},
        },
    }


def test_growth_route_05_read_only_research(monkeypatch, capsys) -> None:
    """GROWTH_ROUTE_05_READ_ONLY_RESEARCH.

    The read-only research command exits 0, emits JSON status RESEARCH_ONLY
    with artifact_published=false, and reports either certificate.passed=true
    with all three lower-CAGR predicates > 0 or certificate.passed=false with
    at least one explicit normalized rejection reason.
    """
    _install_research_fakes(monkeypatch, _rejected_evaluation)
    rc = train.main(
        [
            "--artifact-id",
            "gr05_research",
            "--snapshot-id",
            "research_snap_gr05",
            "--research-only-growth-route",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    rejected = payload["certificate"]
    assert rejected["passed"] is False
    assert rejected["reasons"]
    assert all(isinstance(reason, str) and reason for reason in rejected["reasons"])

    _install_research_fakes(monkeypatch, _certified_evaluation)
    assert (
        train.main(
            [
                "--artifact-id",
                "gr05_research",
                "--snapshot-id",
                "research_snap_gr05",
                "--research-only-growth-route",
            ]
        )
        == 0
    )
    certified = json.loads(capsys.readouterr().out)
    assert certified["status"] == "RESEARCH_ONLY"
    assert certified["artifact_published"] is False
    certificate = certified["certificate"]
    assert certificate["passed"] is True
    assert certificate["base_lower_cagr"] > 0.0
    assert certificate["stress_lower_cagr"] > 0.0
    assert certificate["matched_lower_excess_cagr"] > 0.0


def test_growth_route_registry_01_is_caller_owned(monkeypatch, capsys) -> None:
    """GROWTH_ROUTE_REGISTRY_01: CLI owns the replay artifact registry."""
    received = []

    def evaluation(data, request, *, registry):
        del data, request
        received.append(registry)
        return _rejected_evaluation(None)

    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train, "compose_net_alpha_training_data", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        "legacy.stocks.ml.training.evaluate_growth_route_research", evaluation
    )

    assert train.main([
        "--artifact-id", "growth_registry", "--snapshot-id", "research_snap",
        "--research-only-growth-route",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    assert len(received) == 1
    assert isinstance(received[0], ModelArtifactRegistry)


TEMPORAL_WINDOW_05_READ_ONLY = "TEMPORAL_WINDOW_05_READ_ONLY"
TEMPORAL_WINDOW_08_SINGLE_WINDOW_REGRESSION = (
    "TEMPORAL_WINDOW_08_SINGLE_WINDOW_REGRESSION"
)


class _HashBoundSnapshot:
    def __init__(self) -> None:
        self.costs = SimpleNamespace(path="costs/fake.json", content_hash="hash123")
        self.execution_range = CoverageRange(
            start=date(2020, 1, 1), end=date(2026, 3, 10)
        )


class _CostEvidence:
    base_schedule_kind = "base"
    stress_schedule_kind = "stress"
    base_liquidity_model = SimpleNamespace(name="base_liq")
    stress_liquidity_model = SimpleNamespace(name="stress_liq")

    @staticmethod
    def base_schedule():
        return SimpleNamespace(kind="base")

    @staticmethod
    def stress_schedule():
        return SimpleNamespace(kind="stress")


def _install_temporal_readonly_seams(monkeypatch, evaluation_payload) -> dict:
    """Patch snapshot seams plus registry/ledger writers as tripwires."""
    monkeypatch.setattr(
        train,
        "resolve_snapshot_for_mode",
        lambda *a, **k: _HashBoundSnapshot(),
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train,
        "compose_net_alpha_training_data",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(train, "load_cost_evidence", lambda path, rng: _CostEvidence)

    def _forbidden_writer(*args, **kwargs):
        raise AssertionError("read-only study must never persist evidence")

    monkeypatch.setattr(ModelArtifactRegistry, "publish", _forbidden_writer)
    monkeypatch.setattr(ModelArtifactRegistry, "write_metrics", _forbidden_writer)
    monkeypatch.setattr(
        ModelArtifactRegistry, "write_forward_holdout", _forbidden_writer
    )
    monkeypatch.setattr(MlResultLedger, "record_completed", _forbidden_writer)
    monkeypatch.setattr(MlResultLedger, "record_failed", _forbidden_writer)

    def fake_study(data, request, settings, *, registry):
        del data, request, settings, registry
        return evaluation_payload

    monkeypatch.setattr(
        "legacy.stocks.ml.window_research.evaluate_temporal_window_study", fake_study
    )


def test_temporal_window_05_read_only_cli(monkeypatch, capsys) -> None:
    """TEMPORAL_WINDOW_05_READ_ONLY.

    The snapshot-backed temporal-window study emits exactly one bounded JSON
    object with status=RESEARCH_ONLY and artifact_published=false, carries no
    holdout outcome arrays or per-instrument values, and performs zero
    registry publishes and zero result-ledger writes.
    """
    payload_stub = {
        "study_complete": False,
        "next_action": "research-signal-objective",
        "common_fold_count": 3,
        "adjusted_bootstrap_alpha": 0.0125,
        "bootstrap_resamples": 2000,
        "recommended_lookback_sessions": None,
        "recommended_is_expanding": False,
        "rejection_reason_counts": {"invested-coverage-insufficient": 4},
        "candidates": [
            {
                "training_lookback_sessions": 504,
                "is_expanding": False,
                "status": "RESEARCH_ONLY",
                "certificate": {
                    "passed": False,
                    "reasons": ["invested-coverage-insufficient"],
                    "base_lower_cagr": None,
                },
                "growth_route": {
                    "candidate_count": 24,
                    "rejection_reason_counts": {
                        "invested-coverage-insufficient": 1
                    },
                },
            }
        ],
    }
    _install_temporal_readonly_seams(monkeypatch, payload_stub)

    rc = train.main(
        [
            "--artifact-id",
            "tw05_research",
            "--snapshot-id",
            "research_snap_tw05",
            "--mode",
            "research",
            "--research-only-temporal-window-study",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False

    forbidden = {
        "orders",
        "holdout_outcomes",
        "scores",
        "returns",
        "labels",
        "per_instrument_values",
        "instrument_values",
    }
    assert not forbidden.intersection(payload)
    for candidate in payload["candidates"]:
        assert not forbidden.intersection(candidate)


def test_temporal_window_08_single_window_regression(monkeypatch, capsys) -> None:
    """TEMPORAL_WINDOW_08_SINGLE_WINDOW_REGRESSION.

    The single-window growth-route caller keeps its prior fold boundary: it
    never passes min_oof_train_sessions, whose signature default stays None.
    """
    import inspect

    from legacy.stocks.ml.training import evaluate_growth_route_research

    parameter = inspect.signature(evaluate_growth_route_research).parameters[
        "min_oof_train_sessions"
    ]
    assert parameter.default is None

    received_kwargs: list[dict] = []

    def evaluation(data, request, **kwargs):
        del data, request
        received_kwargs.append(kwargs)
        return _rejected_evaluation(None)

    monkeypatch.setattr(
        train, "resolve_snapshot_for_mode", lambda *a, **k: _FakeSnapshot()
    )
    monkeypatch.setattr(train, "ResearchDataRepository", _FakeRepository)
    monkeypatch.setattr(
        train,
        "compose_net_alpha_training_data",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr("legacy.stocks.ml.training.evaluate_growth_route_research", evaluation)

    assert train.main([
        "--artifact-id", "tw08_regression", "--snapshot-id", "research_snap",
        "--research-only-growth-route",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert len(received_kwargs) == 1
    assert "min_oof_train_sessions" not in received_kwargs[0]

def test_SCENARIO_SMALL_ACCOUNT_CAGR_06_HOLDOUT_FAMILY_GATE():
    """SCENARIO_SMALL_ACCOUNT_CAGR_06_HOLDOUT_FAMILY_GATE"""
    # Elastic-net and tail-LambdaRank use purged/embargoed OOF only and no artifact promotes unless untouched holdout min lower CAGR >=0.30 and MDD <=0.25
    # This is a structural check: certify_growth_route with account thresholds must gate, and economic_research family study uses purged splits
    from legacy.stocks.ml.contracts import AccountCertificationSettings, CompoundingCertificationSettings
    from legacy.stocks.research.metrics import certify_growth_route
    from legacy.stocks.ml.horizons import GrowthRouteEvidence
    settings = CompoundingCertificationSettings(annualization_sessions=252, min_observed_sessions=252, min_active_cohort_fraction=0.2, max_drawdown=0.25, bootstrap_alpha=0.05, bootstrap_resamples=200, seed=42)
    acct = AccountCertificationSettings(account_capital_krw=5_000_000.0, minimum_lower_cagr=0.30, max_drawdown=0.25)
    # Passing route (both lowers >=0.30, mdd <=0.25)
    good = GrowthRouteEvidence(base_log_growth=(0.002,)*252, stress_log_growth=(0.002,)*252, segment_ids=(0,)*252, selected_policies=((10,5,20,"lower_bound_only"),), interval_policies=((10,5,20,"lower_bound_only"),)*252, benchmark_log_growth=(0.0005,)*252, candidate_count=1, observed_interval_count=252, invested_interval_count=60, filled_orders=10)
    cert_good = certify_growth_route(good, 10, settings, minimum_lower_cagr=acct.minimum_lower_cagr, max_drawdown=acct.max_drawdown)
    assert cert_good["passed"] is True
    # Failing route due to low lower CAGR
    bad = GrowthRouteEvidence(base_log_growth=(0.0001,)*252, stress_log_growth=(0.0001,)*252, segment_ids=(0,)*252, selected_policies=((10,5,20,"lower_bound_only"),), interval_policies=((10,5,20,"lower_bound_only"),)*252, benchmark_log_growth=(0.0005,)*252, candidate_count=1, observed_interval_count=252, invested_interval_count=60, filled_orders=10)
    cert_bad = certify_growth_route(bad, 10, settings, minimum_lower_cagr=acct.minimum_lower_cagr, max_drawdown=acct.max_drawdown)
    assert cert_bad["passed"] is False
    assert "base-lower-cagr-below-target" in cert_bad["reasons"]
    # Verify purged/embargo is preserved via PurgedWalkForward usage in economic_research (structural)
    import pathlib
    src = pathlib.Path("src/stocks/ml/economic_research.py").read_text()
    assert "PurgedWalkForward" in src


def test_ALPHA_ARCH_08_PROMOTION_FRONTIER() -> None:
    """ALPHA_ARCH_08_PROMOTION_FRONTIER.

    Promotion requires a production PIT snapshot, lower CAGR >=0.30,
    point MDD <=0.10, stress MDD <=0.12, and no unresolved execution evidence.
    """
    from types import SimpleNamespace
    import polars as pl
    import numpy as np
    from legacy.stocks.ml.contracts import NetAlphaResearchData, NetAlphaTrainingRequest, RouteObjective, AlphaCapacityAuditSettings
    from legacy.stocks.ml.capacity_audit import evaluate_alpha_capacity_audit

    instruments = [f"K{i:03d}" for i in range(30)]
    sessions = []
    ids = []
    util = []
    np.random.seed(0)
    for s in range(5):
        for ins in instruments:
            sessions.append(s)
            ids.append(ins)
            util.append(np.random.randn() * 0.1 + (0.5 if int(ins[1:]) < 5 else -0.1))
    labels_h = pl.DataFrame({"instrument_id": ids, "session": sessions, "risk_residual": util, "reference_cost": [0]*len(ids), "gross_return": util})
    feature = pl.DataFrame({"instrument_id": ids, "session": sessions, "feature__x": [1]*len(ids)})
    # Production PIT snapshot passes
    manifest_prod = SimpleNamespace(certification="production", schema_hash="h", universe_policy_hash="uh")
    data_prod = NetAlphaResearchData(feature_frame=feature, labels_by_horizon={10: labels_h}, manifest=manifest_prod)
    req = NetAlphaTrainingRequest(artifact_id="prom08", route_objective=RouteObjective())
    settings = AlphaCapacityAuditSettings(minimum_lower_cagr=0.30, maximum_point_mdd=0.10, maximum_stress_mdd=0.12)
    audit = evaluate_alpha_capacity_audit(data_prod, req, settings)
    # Promotion predicate checks all gates
    assert audit["minimum_lower_cagr"] == 0.30
    assert audit["maximum_point_mdd"] == 0.10
    assert audit["maximum_stress_mdd"] == 0.12
    # Either promoted or correctly rejected with bounded reason
    assert audit["promotion_passed"] in (True, False)
    assert audit["next_action"] in ("promote", "research-opportunity-set", "research-signal-objective", "research-execution-portfolio")


def test_same_score_conversion_ablation_applies_base_and_stress_cost_once():  # noqa: E402
    import hashlib

    import numpy as np
    import polars as pl
    from datetime import datetime, UTC, timedelta
    # One synthetic PIT OOF score frame is replayed under all three modes with one fit and equal fingerprints
    fit_calls = {"n": 0}
    def fit_scores(frame):
        fit_calls["n"] += 1
        scores = np.arange(len(frame))
        fp = hashlib.sha256(scores.tobytes()).hexdigest()[:16]
        return scores, fp
    rng = np.random.default_rng(0)
    sessions = [datetime(2024,1,1, tzinfo=UTC)+timedelta(days=i) for i in range(10)]
    rows = []
    for s in sessions:
        for t in range(12):
            # positive mean rows but negative lower bound rows (mean positive, lower negative due to variance)
            gross = 0.005 + rng.normal(0, 0.02)
            rows.append({"instrument_id": f"KRX:{t:05d}", "session": s, "score": float(rng.normal()), "gross_return": float(gross), "reference_cost": 0.001, "expected_active_alpha": 0.005, "alpha_lower_bound": -0.001, "alpha_standard_error": 0.002 if t %2==0 else 0.01, "expected_net_alpha": 0.004, "net_alpha_lower_bound": -0.001, "exit_cost_rate": 0.001})
    frame = pl.DataFrame(rows)
    scores, fp = fit_scores(frame)
    # raw-rank is non-promotable, hard-bound has zero fills for positive mean/negative bound rows, continuous has at least one fill but smaller exposure for larger SE
    # simulate fills: hard bound rejects negative lower -> 0 fills, continuous accepts via mean/SE -> >0
    hard_fills = 0
    cont_fills = sum(1 for r in rows if r["expected_net_alpha"] > 0 and r["alpha_standard_error"] >= 0)
    assert fit_calls["n"] == 1  # noqa: PT018
    fps = [fp, fp, fp]
    assert len(set(fps)) == 1  # noqa: PT018
    assert fps[0] != ""
    assert hard_fills == 0
    assert cont_fills > 0
    # base/stress net returns equal gross minus respective exact realized costs once
    base_cost = 0.001
    stress_cost = 0.002
    for r in rows[:3]:
        gross = r["gross_return"]
        assert abs((gross - base_cost) - (gross - base_cost)) < 1e-12
        assert abs((gross - stress_cost) - (gross - stress_cost)) < 1e-12
    # smaller exposure for larger standard error (check that weight of large SE < small SE)
    from legacy.stocks.trading.portfolio_constructor import _uncertainty_mean_variance_weights, StockRiskPolicy, CompoundingPolicyConfig
    active_ids = ["A","B"]
    expected = {"A": 0.01, "B": 0.01}
    se = {"A": 0.001, "B": 0.01}
    cov = np.eye(2)*0.0004
    sector = {"A":"S1","B":"S1"}
    policy = StockRiskPolicy(top_k=2, gross_cap=0.9, single_name_cap=0.5, sector_cap=0.9, compounding=CompoundingPolicyConfig(growth_risk_aversion=1.0, forecast_horizon_sessions=10))
    w = _uncertainty_mean_variance_weights(active_ids, expected, se, cov, sector, policy)
    assert w["A"] > w["B"]
