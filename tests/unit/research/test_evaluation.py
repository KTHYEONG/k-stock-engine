"""Report-card invariants: robust-growth objective, survival guards, diagnostics."""
from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta

import numpy as np
import pytest

from src.research.evaluation import EvaluationPolicy, build_report_card
from src.research.ledger_bridge import LedgerOutcome


def _policy(**overrides):
    base = {
        "sessions_per_year": 252,
        "block_sessions": 21,
        "draws": 100,
        "seed": 11,
        "horizon_sessions": 126,
        "objective_quantile": 0.10,
        "ruin_mdd_limit": -0.5,
        "max_p_ruin": 0.05,
        "max_p_growth_le_zero": 0.05,
        "report_mdd_limits": (-0.3, -0.5),
        "recent_sessions": 252,
    }
    base.update(overrides)
    return EvaluationPolicy(**base)


def _outcome(logs, *, sessions, stock_scale=0.9, journal=None, capital=100_000_000):
    logs = np.asarray(logs, dtype=np.float64)
    return LedgerOutcome(
        capital_krw=capital,
        halted_exit_policy="zero",
        sessions=tuple(sessions),
        log_returns=np.ascontiguousarray(logs),
        stock_book_returns=np.ascontiguousarray(logs * stock_scale),
        journal_totals_krw=dict(journal or {"commission": -1000}),
        avg_stock_exposure=0.8,
        avg_margin_share=0.1,
        avg_inverse_share=0.05,
        turnover_per_year=2.0,
        ledger_hash="h",
    )


def _evidence(n=400, seed=9, mismatches=0, base_shift=0.0008):
    rng = np.random.default_rng(seed)
    sessions = tuple(date(2018, 1, 1) + timedelta(days=i) for i in range(n))
    base = rng.normal(base_shift, 0.01, size=n)
    slip = base - 0.0002
    delay = base - 0.0005
    from src.research.evaluation import EvaluationEvidence

    return EvaluationEvidence(
        sessions=sessions,
        base=_outcome(base, sessions=sessions),
        stress_slippage=_outcome(slip, sessions=sessions),
        stress_delay=_outcome(delay, sessions=sessions),
        cost_grid={
            0.0: _outcome(base, sessions=sessions),
            0.5: _outcome(slip, sessions=sessions),
            1.0: _outcome(delay, sessions=sessions),
        },
        unhedged=_outcome(base, sessions=sessions),
        placebo=_outcome(rng.normal(0, 0.01, size=n), sessions=sessions),
        index_log_returns=np.ascontiguousarray(rng.normal(0.0003, 0.01, size=n)),
        universe_ew_log_returns=np.ascontiguousarray(rng.normal(0.0003, 0.01, size=n)),
        fast_sim_growth=0.15,
        perturbation_mismatches=mismatches,
    )


def test_j_picks_worse_stress_stream():
    from src.research.stats import growth_profile

    evidence = _evidence()
    policy = _policy()
    report = build_report_card(evidence, policy, spec_hash="s", run_id="r", protocol_id="abc")
    slip = growth_profile(
        np.asarray(evidence.stress_slippage.log_returns), block=21, draws=100, seed=11,
        horizon=126, sessions_per_year=252, quantile=0.10, mdd_limits=(-0.3, -0.5, -0.5),
    )
    delay = growth_profile(
        np.asarray(evidence.stress_delay.log_returns), block=21, draws=100, seed=11,
        horizon=126, sessions_per_year=252, quantile=0.10, mdd_limits=(-0.3, -0.5, -0.5),
    )
    worse = "stress_delay" if delay.g_quantile < slip.g_quantile else "stress_slippage"
    assert report.objective_stream == worse
    assert report.objective_j == pytest.approx(min(slip.g_quantile, delay.g_quantile))


def test_report_carries_the_protocol_id():
    import json

    report = build_report_card(_evidence(), _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert report.protocol_id == "abc"
    payload = json.loads(report.canonical_json())
    assert payload["protocol_id"] == "abc"
    assert "protocol_version" not in payload


def test_ruin_guard_blocks_but_diagnostics_populated():
    n = 400
    sessions = tuple(date(2018, 1, 1) + timedelta(days=i) for i in range(n))
    crash = np.tile(np.concatenate([np.full(50, 0.001), [math.log(0.4)]]), 20)[:n]
    from src.research.evaluation import EvaluationEvidence

    evidence = EvaluationEvidence(
        sessions=sessions,
        base=_outcome(crash, sessions=sessions),
        stress_slippage=_outcome(crash, sessions=sessions),
        stress_delay=_outcome(crash - 0.0001, sessions=sessions),
        cost_grid={0.0: _outcome(crash, sessions=sessions), 1.0: _outcome(crash, sessions=sessions)},
        unhedged=_outcome(crash, sessions=sessions),
        placebo=_outcome(crash, sessions=sessions),
        index_log_returns=np.zeros(n),
        universe_ew_log_returns=np.zeros(n),
        fast_sim_growth=0.0,
        perturbation_mismatches=0,
    )
    report = build_report_card(evidence, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert report.passed is False
    assert any(not guard.passed for guard in report.guards)
    assert math.isfinite(report.metrics["g"])
    assert math.isfinite(report.objective_j)
    assert report.yearly_growth
    assert report.controls


def test_integrity_blocks_on_perturbation():
    evidence = _evidence(mismatches=1)
    report = build_report_card(evidence, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert report.passed is False
    assert any(c.name == "perturbation_mismatches" and not c.passed for c in report.integrity)


def test_length_mismatch_is_integrity_failure():
    evidence = _evidence()
    short = np.asarray(evidence.base.log_returns)[:-1]
    sessions = evidence.sessions
    from src.research.evaluation import EvaluationEvidence

    broken = EvaluationEvidence(
        sessions=sessions, base=_outcome(short, sessions=sessions[:-1]),
        stress_slippage=evidence.stress_slippage, stress_delay=evidence.stress_delay,
        cost_grid=evidence.cost_grid, unhedged=evidence.unhedged, placebo=evidence.placebo,
        index_log_returns=evidence.index_log_returns,
        universe_ew_log_returns=evidence.universe_ew_log_returns,
        fast_sim_growth=evidence.fast_sim_growth, perturbation_mismatches=0,
    )
    report = build_report_card(broken, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert report.passed is False
    assert any(not c.passed for c in report.integrity)


def test_regime_split_and_canonical_json():
    n = 500
    sessions = tuple(date(2018, 1, 1) + timedelta(days=i) for i in range(n))
    rng = np.random.default_rng(4)
    base = rng.normal(0.001, 0.005, size=n)
    index = np.concatenate([np.full(n // 2, -0.001), np.full(n - n // 2, 0.002)])
    from src.research.evaluation import EvaluationEvidence

    evidence = EvaluationEvidence(
        sessions=sessions, base=_outcome(base, sessions=sessions),
        stress_slippage=_outcome(base, sessions=sessions),
        stress_delay=_outcome(base, sessions=sessions),
        cost_grid={0.0: _outcome(base, sessions=sessions), 1.0: _outcome(base - 0.001, sessions=sessions)},
        unhedged=_outcome(base, sessions=sessions), placebo=_outcome(base, sessions=sessions),
        index_log_returns=np.ascontiguousarray(index),
        universe_ew_log_returns=np.ascontiguousarray(index),
        fast_sim_growth=0.1, perturbation_mismatches=0,
    )
    policy = _policy()
    first = build_report_card(evidence, policy, spec_hash="s", run_id="r", protocol_id="abc")
    second = build_report_card(evidence, policy, spec_hash="s", run_id="r", protocol_id="abc")
    assert set(first.regime_growth) == {"index_up_years", "index_down_years"}
    assert first.digest == second.digest
    assert first.digest == hashlib.sha256(first.canonical_json().encode()).hexdigest()
    assert first.metrics["g_median_5y"] is not None
    assert "p_mdd_below_-0.3" in first.metrics
    assert "p_mdd_below_-0.5" in first.metrics
    assert first.recent["sessions"] == 252.0


def test_policy_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _policy(objective_quantile=0.0)
    with pytest.raises(ValidationError):
        _policy(ruin_mdd_limit=0.0)
    with pytest.raises(ValidationError):
        _policy(report_mdd_limits=())
    with pytest.raises(ValidationError):
        _policy(sessions_per_year=0)
    with pytest.raises(ValidationError):
        _policy(seed=-1)
    with pytest.raises(ValidationError):
        _policy(max_p_ruin=2.0)


def test_empty_sessions_raises():
    from src.research.evaluation import EvaluationEvidence

    evidence = _evidence(n=10)
    broken = EvaluationEvidence(
        sessions=(), base=evidence.base, stress_slippage=evidence.stress_slippage,
        stress_delay=evidence.stress_delay, cost_grid=evidence.cost_grid,
        unhedged=evidence.unhedged, placebo=evidence.placebo,
        index_log_returns=np.zeros(0), universe_ew_log_returns=np.zeros(0),
        fast_sim_growth=0.0, perturbation_mismatches=0,
    )
    with pytest.raises(ValueError, match="non-empty"):
        build_report_card(broken, _policy(), spec_hash="s", run_id="r", protocol_id="abc")


def test_nonfinite_and_single_point_grid_edges():
    from src.research.evaluation import EvaluationEvidence

    evidence = _evidence(n=60)
    bad = np.asarray(evidence.base.log_returns)
    bad = bad.copy()
    bad[0] = math.inf
    broken = EvaluationEvidence(
        sessions=evidence.sessions, base=_outcome(bad, sessions=evidence.sessions),
        stress_slippage=_outcome(bad, sessions=evidence.sessions),
        stress_delay=_outcome(bad, sessions=evidence.sessions),
        cost_grid={0.0: _outcome(bad, sessions=evidence.sessions)},
        unhedged=evidence.unhedged, placebo=evidence.placebo,
        index_log_returns=np.full(60, math.nan), universe_ew_log_returns=np.full(60, math.nan),
        fast_sim_growth=math.nan, perturbation_mismatches=0,
    )
    report = build_report_card(broken, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert report.passed is False
    assert math.isnan(report.breakeven_ticks)
    assert "cost_commission_per_year" in report.metrics


def test_edge_branches_cover_helpers():
    from src.research.evaluation import EvaluationEvidence

    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(12))
    empty = np.zeros(0, dtype=np.float64)
    all_nan = np.full(12, math.nan)
    neg_inf_grid = np.full(12, math.inf)
    neg_only = np.full(12, -2.0)
    base = EvaluationEvidence(
        sessions=sessions, base=_outcome(empty, sessions=()),
        stress_slippage=_outcome(empty, sessions=()),
        stress_delay=_outcome(empty, sessions=()),
        cost_grid={0.0: _outcome(neg_inf_grid, sessions=sessions)},
        unhedged=_outcome(neg_only, sessions=sessions),
        placebo=_outcome(all_nan, sessions=sessions),
        index_log_returns=np.zeros(0, dtype=np.float64),
        universe_ew_log_returns=np.zeros(0, dtype=np.float64),
        fast_sim_growth=math.inf, perturbation_mismatches=0,
    )
    report = build_report_card(base, _policy(block_sessions=21), spec_hash="s", run_id="r", protocol_id="abc")
    assert report.passed is False
    assert math.isnan(report.metrics["g"])
    assert "-inf" in report.canonical_json() or "nan" in report.canonical_json()

    # constant index -> beta zero-variance branch; single-finite-pair branch
    idx_const = np.ones(12)
    idx_single = np.full(12, math.nan)
    idx_single[0] = 0.01
    idx_single[1] = 0.02
    ev2 = _evidence(n=12, seed=5)
    from src.research import evaluation as _evaluation

    for idx in (idx_const, idx_single):
        ev = _evaluation.EvaluationEvidence(
            sessions=ev2.sessions, base=ev2.base, stress_slippage=ev2.stress_slippage,
            stress_delay=ev2.stress_delay, cost_grid=ev2.cost_grid, unhedged=ev2.unhedged,
            placebo=ev2.placebo, index_log_returns=np.ascontiguousarray(idx, dtype=np.float64),
            universe_ew_log_returns=ev2.universe_ew_log_returns,
            fast_sim_growth=ev2.fast_sim_growth, perturbation_mismatches=0,
        )
        rep = build_report_card(ev, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
        assert math.isfinite(rep.metrics["beta_to_index"])

    # bad report limit rejected
    from pydantic import ValidationError as _ValidationError

    with pytest.raises(_ValidationError):
        _policy(report_mdd_limits=(-0.3, 0.0))

    # huge logs -> mean_nav non-finite branch
    huge = np.full(12, 1000.0)
    ev3 = _evaluation.EvaluationEvidence(
        sessions=ev2.sessions, base=_outcome(huge, sessions=ev2.sessions),
        stress_slippage=ev2.stress_slippage, stress_delay=ev2.stress_delay,
        cost_grid={0.0: _outcome(huge, sessions=ev2.sessions), 1.0: _outcome(huge, sessions=ev2.sessions)},
        unhedged=ev2.unhedged, placebo=ev2.placebo, index_log_returns=ev2.index_log_returns,
        universe_ew_log_returns=ev2.universe_ew_log_returns,
        fast_sim_growth=0.0, perturbation_mismatches=0,
    )
    rep3 = build_report_card(ev3, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert math.isnan(rep3.metrics["cost_commission_per_year"])

    # zero-drawdown -> calmar inf branch; empty yearly branch via short base
    flat = np.full(30, 0.001)
    long_sessions = tuple(date(2021, 1, 1) + timedelta(days=i) for i in range(400))
    short_base = _outcome(flat, sessions=long_sessions[:30])
    ev4 = _evaluation.EvaluationEvidence(
        sessions=long_sessions, base=short_base,
        stress_slippage=_outcome(np.full(400, 0.001), sessions=long_sessions),
        stress_delay=_outcome(np.full(400, 0.0009), sessions=long_sessions),
        cost_grid={0.0: _outcome(np.full(400, 0.001), sessions=long_sessions)},
        unhedged=short_base, placebo=short_base,
        index_log_returns=np.full(400, 0.001),
        universe_ew_log_returns=np.full(400, 0.001),
        fast_sim_growth=0.0, perturbation_mismatches=0,
    )
    rep4 = build_report_card(ev4, _policy(), spec_hash="s", run_id="r", protocol_id="abc")
    assert rep4.passed is False

    # empty/raw and all-nan point branches
    tiny_sessions = tuple(date(2022, 1, 1) + timedelta(days=i) for i in range(8))
    all_nan_base = _outcome(np.full(8, math.nan), sessions=tiny_sessions)
    empty_stock = _outcome(np.full(8, 0.001), sessions=tiny_sessions)
    empty_stock = LedgerOutcome(
        capital_krw=empty_stock.capital_krw, halted_exit_policy=empty_stock.halted_exit_policy,
        sessions=empty_stock.sessions, log_returns=empty_stock.log_returns,
        stock_book_returns=np.zeros(0, dtype=np.float64),
        journal_totals_krw={}, turnover_per_year=0.0, ledger_hash="h",
    )
    ev5 = _evaluation.EvaluationEvidence(
        sessions=tiny_sessions, base=all_nan_base,
        stress_slippage=all_nan_base, stress_delay=all_nan_base,
        cost_grid={
            0.0: _outcome(np.zeros(0, dtype=np.float64), sessions=()),
            1.0: _outcome(np.full(8, 0.001), sessions=tiny_sessions),
        },
        unhedged=empty_stock, placebo=all_nan_base,
        index_log_returns=np.full(8, math.nan),
        universe_ew_log_returns=np.zeros(0, dtype=np.float64),
        fast_sim_growth=0.0, perturbation_mismatches=0,
    )
    rep5 = build_report_card(ev5, _policy(block_sessions=4, draws=10, horizon_sessions=8), spec_hash="s", run_id="r", protocol_id="abc")
    assert rep5.passed is False
    assert math.isnan(rep5.metrics["g"])
