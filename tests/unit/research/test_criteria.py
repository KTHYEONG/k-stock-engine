"""Termination criteria C1-C4 invariants."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.data.research_protocol import (
    CriteriaC1,
    CriteriaC2,
    CriteriaC3,
    CriteriaC4,
    CriteriaBootstrap,
    CriteriaPolicy,
    ResearchProtocol,
    Segment,
    StatisticsPolicy,
)
from src.research.criteria import DiscoveryEvidence, _lower_median, evaluate_discovery, evaluate_holdout
from src.research.stats import point_metrics

SPY = 252
CAP = 10_000_000


def _protocol(**overrides: object) -> ResearchProtocol:
    criteria_kwargs: dict[str, object] = {
        "bootstrap": CriteriaBootstrap(block_sessions=10, draws=100, seed=7, horizon_sessions=60, mdd_limit=-0.5),
        "c1": CriteriaC1(
            stress_extra_slippage=0.001,
            stress_execution_delay=1,
            max_p_cagr_le_zero=0.05,
            perturbation_cuts=3,
            perturbation_seed=7,
        ),
        "c2": CriteriaC2(
            min_point_calmar=1.2,
            min_p_calmar=0.75,
            max_p_mdd_below_limit=0.05,
            max_underwater_median_sessions=252,
            max_underwater_p95_sessions=630,
            min_worst_phase_calmar=1.0,
        ),
        "c3": CriteriaC3(
            stress_min_point_calmar=1.0,
            ledger_capitals=(CAP,),
            ledger_min_calmar=1.0,
            parity_max_growth_gap=0.01,
        ),
        "c4": CriteriaC4(holdout_max_p_mean_le_zero=0.05, holdout_min_point_calmar=0.7),
        "prior_trials": 1533,
        "prior_effective_trials": 100.0,
        "prior_trial_sharpe_std_annual": 0.42,
    }
    criteria_kwargs.update(overrides.pop("criteria", {}))  # type: ignore[arg-type]
    return ResearchProtocol(
        version="test-protocol-v1",
        discovery_start=date(2017, 4, 1),
        discovery_end=date(2023, 12, 31),
        holdout_start=date(2024, 1, 1),
        holdout_end=date(2025, 12, 31),
        forward_start=date(2026, 1, 1),
        max_finalists=1,
        prior_trials=1533,
        sessions_per_year=SPY,
        primary_capital_krw=CAP,
        statistics=StatisticsPolicy(
            bootstrap_block_sessions=63, bootstrap_draws=10, bootstrap_seed=1, cscv_blocks=8
        ),
        criteria=CriteriaPolicy(**criteria_kwargs),  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def _good_phases(seed: int = 100, n: int = 3, length: int = 120) -> tuple[np.ndarray, ...]:
    return tuple(
        np.random.default_rng(seed + i).normal(loc=0.003, scale=0.01, size=length) for i in range(n)
    )


def _evidence(**overrides: object) -> DiscoveryEvidence:
    base = _good_phases(100)
    ledger = np.full(120, 0.001)
    fields: dict[str, object] = {
        "sessions": (),
        "base": base,
        "stress_slippage": _good_phases(seed=110),
        "stress_delay": _good_phases(seed=120),
        "fast_growth": {(CAP, "zero"): float(np.mean(ledger) * SPY), (CAP, "last_close"): float(np.mean(ledger) * SPY)},
        "ledger_returns": {(CAP, "zero"): ledger, (CAP, "last_close"): ledger},
        "perturbation_mismatches": 0,
        "price_cap_growth_delta": 0.001,
        "effective_registry_trials": 5.0,
    }
    fields.update(overrides)
    return DiscoveryEvidence(**fields)  # type: ignore[arg-type]


def _names(report: object) -> dict[str, object]:
    return {check.name: check for check in report.checks}  # type: ignore[union-attr]


def test_discovery_passes_on_strong_evidence() -> None:
    report = evaluate_discovery(_evidence(), _protocol(), spec_hash="abc", trial_id="t1")
    assert report.passed
    assert report.segment is Segment.DISCOVERY
    assert all(check.passed for check in report.checks)
    names = _names(report)
    assert "C1.perturbation" in names
    assert "C1.dsr_reported" in names
    assert "C2.worst_phase_calmar" in names
    assert f"C3.ledger_calmar.{CAP}.zero" in names
    assert f"C3.parity.{CAP}" in names


def test_perturbation_flip_fails_only_its_check() -> None:
    report = evaluate_discovery(
        _evidence(perturbation_mismatches=1), _protocol(), spec_hash="abc", trial_id="t1"
    )
    assert not report.passed
    failed = [check.name for check in report.checks if not check.passed]
    assert failed == ["C1.perturbation"]


def test_missing_ledger_key_fails_targeted_check() -> None:
    evidence = _evidence()
    ledgers = dict(evidence.ledger_returns)
    del ledgers[(CAP, "last_close")]
    report = evaluate_discovery(
        DiscoveryEvidence(
            sessions=evidence.sessions,
            base=evidence.base,
            stress_slippage=evidence.stress_slippage,
            stress_delay=evidence.stress_delay,
            fast_growth=evidence.fast_growth,
            ledger_returns=ledgers,
            perturbation_mismatches=evidence.perturbation_mismatches,
            price_cap_growth_delta=evidence.price_cap_growth_delta,
            effective_registry_trials=evidence.effective_registry_trials,
        ),
        _protocol(),
        spec_hash="abc",
        trial_id="t1",
    )
    assert not report.passed
    failed = [check.name for check in report.checks if not check.passed]
    assert failed == [f"C3.ledger_calmar.{CAP}.last_close"]


def test_non_blocking_checks_never_fail_the_report() -> None:
    flat = tuple(np.full(120, 0.001) for _ in range(3))
    ledger = np.full(120, 0.001)
    evidence = _evidence(
        base=flat,
        stress_slippage=flat,
        stress_delay=flat,
        ledger_returns={(CAP, "zero"): ledger, (CAP, "last_close"): ledger},
        price_cap_growth_delta=float("nan"),
    )
    report = evaluate_discovery(evidence, _protocol(), spec_hash="abc", trial_id="t1")
    names = _names(report)
    assert names["C1.dsr_reported"].passed is False
    assert names["C3.price_cap_effect"].passed is False
    assert names["C1.dsr_reported"].blocking is False
    assert names["C3.price_cap_effect"].blocking is False
    assert report.passed


def test_nan_stress_stream_fails_closed() -> None:
    good = _good_phases()
    delay = (good[0], np.array([]), good[2])
    report = evaluate_discovery(
        _evidence(stress_delay=delay), _protocol(), spec_hash="abc", trial_id="t1"
    )
    assert not report.passed
    names = _names(report)
    assert names["C1.growth_positive.delay"].passed is False
    assert names["C1.bootstrap_loss.delay"].passed is False
    assert names["C3.stress_calmar.delay"].passed is False
    assert names["C1.growth_positive.base"].passed is True
    assert names["C3.stress_calmar.slippage"].passed is True


def test_lower_median_aggregation() -> None:
    assert _lower_median([0.5, 1.5, 1.4, 1.3]) == 1.3
    assert _lower_median([2.0, 1.0, 3.0]) == 2.0
    assert np.isnan(_lower_median([]))
    assert np.isnan(_lower_median([1.0, float("nan")]))
    base = _good_phases()
    report = evaluate_discovery(_evidence(base=base), _protocol(), spec_hash="abc", trial_id="t1")
    calmars = sorted(point_metrics(p, sessions_per_year=SPY).calmar for p in base)
    expected = calmars[(len(calmars) - 1) // 2]
    assert _names(report)["C2.point_calmar"].value == pytest.approx(expected)


def test_report_digest_is_stable() -> None:
    evidence = _evidence()
    protocol = _protocol()
    first = evaluate_discovery(evidence, protocol, spec_hash="abc", trial_id="t1")
    second = evaluate_discovery(evidence, protocol, spec_hash="abc", trial_id="t1")
    assert first.canonical_json() == second.canonical_json()
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_holdout_is_one_trial() -> None:
    rng = np.random.default_rng(51)
    holdout = rng.normal(loc=0.002, scale=0.01, size=120)
    report = evaluate_holdout(holdout, _protocol(), spec_hash="abc", trial_id="t9")
    assert report.segment is Segment.HOLDOUT
    assert [check.name for check in report.checks] == [
        "C4.growth_positive",
        "C4.p_mean_le_zero",
        "C4.point_calmar",
    ]
    assert report.passed
    weak = rng.normal(loc=-0.002, scale=0.01, size=120)
    failed = evaluate_holdout(weak, _protocol(), spec_hash="abc", trial_id="t9")
    assert not failed.passed
    assert _names(failed)["C4.growth_positive"].passed is False
